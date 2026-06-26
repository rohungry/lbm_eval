"""
Benchmark the "Rolling an Elastoplastic Dough" task (roll.cc) from
arXiv:2503.05046 (IROS 2025) on Modal GPUs, and reproduce the paper's
runtime-per-step and real-time-rate numbers.

Paper reference (Table I + Sec. VI-B, RTX 4090):
    Dough Rolling: dt=10 ms, N=10 substeps, ~21.7 ms/step,
                   ~46% real-time rate at eps_r=5e-2 (59% at eps_r=1e-1).

What this measures, and how:
    The deformable driver prints `frame=<N> time=<ms>ms N(substeps)=<k>` per
    step, where <ms> is the GPU-synced wall-clock COMPUTE time for that step.
    We run unthrottled (--realtime_rate=0), with visualization and file dumps
    OFF (they add overhead that would corrupt the timing), parse those lines,
    and compute real-time rate = dt / runtime-per-step. See bench_realtime.py.

Usage:
    modal run drake_dough_modal.py                 # default: paper settings
    modal run drake_dough_modal.py --record        # also dump html + obj (slow)
    modal run drake_dough_modal.py::bench --run-id dough-...   # re-analyze a log
    modal volume get drake-mpm-outputs <run_id> ./results

Image note:
    The image is byte-identical to drake_mpm_modal.py through every expensive
    layer (clone, prereqs, Drake build), so Modal reuses those cached layers.
    Only one cheap final layer is new: patch roll.html's hardcoded path and
    build //examples/multibody/deformable:roll (Drake is already compiled in
    the cached bazel tree, so this is minutes, not hours).
"""

import subprocess
import time
from pathlib import Path

import modal

app = modal.App("drake-mpm-dough-rolling")

# Same Volume as the t-shirt harness, so results land together.
outputs = modal.Volume.from_name("drake-mpm-outputs", create_if_missing=True)

REPO = "https://github.com/g1n0st/drake.git"
BRANCH = "cuda-mpm-weak-coupling-clean"
COMMIT = "c0d004f5d5f9a279b81e50c8a8071c1ac493663c"
DRAKE_DIR = "/root/drake"
EXAMPLE_PKG = "//examples/multibody/deformable"
ROLL_TARGET = "roll"  # cc_binary name for roll.cc; adjust if BUILD names it otherwise

# --- These five strings are byte-identical to drake_mpm_modal.py so the ---
# --- expensive image layers are reused from cache, not rebuilt.          ---
PATCH_MESH_CMD = (
    "cd /root/drake && "
    "grep -rl '/home/changyu/Desktop/tshirt' examples/multibody/deformable "
    "| xargs -r sed -i 's|/home/changyu/Desktop/\\(tshirt[a-z0-9_]*\\.obj\\)|/root/drake/\\1|g'"
)
PATCH_OUT_CMD = (
    "cd /root/drake && "
    "grep -rl '/home/changyu/Desktop' examples/multibody/deformable "
    "| xargs -r sed -i 's|/home/changyu/Desktop/|/tmp/sim_out/|g'"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "wget", "curl", "ca-certificates", "lsb-release", "sudo")
    .run_commands(
        f"git clone --branch {BRANCH} {REPO} {DRAKE_DIR}",
        f"cd {DRAKE_DIR} && git checkout {COMMIT}",
        f"cd {DRAKE_DIR} && DEBIAN_FRONTEND=noninteractive "
        "./setup/ubuntu/install_prereqs.sh -y || "
        f"(cd {DRAKE_DIR} && DEBIAN_FRONTEND=noninteractive yes | ./setup/ubuntu/install_prereqs.sh)",
        PATCH_MESH_CMD,
        PATCH_OUT_CMD,
        "mkdir -p /tmp/sim_out",
    )
    .run_commands(
        "cd /root/drake && bazel build --config omp --jobs=8 //multibody/plant",
    )
    .run_commands(
        "cd /root/drake && bazel build --config omp --jobs=8 "
        + " ".join(f"{EXAMPLE_PKG}:{t}" for t in ("mpm_tshirt_folding", "dual_arm_flipping")),
    )
    .env({"LCM_DEFAULT_URL": "memq://"})
    # --- New, cheap layer: patch roll.html output path, then build roll. ---
    # Drake is already compiled in the cached bazel tree above, so this only
    # compiles roll.cc + links.
    .run_commands(
        "cd /root/drake && "
        "sed -i 's|/home/changyu/drake/roll.html|/tmp/sim_out/roll.html|g' "
        "examples/multibody/deformable/roll.cc && "
        f"bazel build --config omp --jobs=8 {EXAMPLE_PKG}:{ROLL_TARGET}",
    )
    .pip_install("numpy")
    .add_local_python_source("bench_realtime")
)


@app.function(
    image=image,
    gpu="L40S",  # sm_89, same Ada arch as the paper's RTX 4090
    timeout=2 * 60 * 60,
    volumes={"/outputs": outputs},
)
def run_dough(
    time_step: float = 1e-2,     # dt = 10 ms  (paper dough)
    substep: float = 1e-3,       # -> N = dt/substep = 10 substeps (paper dough)
    friction: float = 1.0,       # paper dough
    stiffness: float = 1e3,
    simulation_time: float = 10.0,
    ppc: float = 3,
    record: bool = False,        # if True: dump html+obj (overhead; not for timing)
    run_id: str | None = None,
) -> dict:
    """Run roll.cc unthrottled, parse per-step timing, compute real-time rate."""
    import os
    import shutil

    from bench_realtime import analyze_full, parse_step_times

    run_id = run_id or f"dough-dt{int(time_step * 1000)}ms-mu{friction}-{int(time.time())}"
    out_dir = Path("/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["nvidia-smi"], check=False)
    scratch = Path("/tmp/sim_out")
    scratch.mkdir(exist_ok=True)
    for f in scratch.glob("*"):
        if f.is_file():
            f.unlink()

    # Marker so artifact collection picks up only THIS run's dumps (Dump()
    # writes relative paths -> they land in the bazel runfiles cwd, not here).
    marker = Path("/tmp/.dough_marker")
    marker.touch()
    time.sleep(1.1)

    n_substeps = round(time_step / substep)

    cmd = [
        "bazel", "run", "--config", "omp", f"{EXAMPLE_PKG}:{ROLL_TARGET}", "--",
        f"--simulation_time={simulation_time}",
        f"--time_step={time_step}",
        f"--substep={substep}",
        f"--friction={friction}",
        f"--stiffness={stiffness}",
        f"--ppc={ppc}",
        # Critical for benchmarking: 0 = run as fast as possible (no throttle).
        # The default 1.0 would cap wall-clock at real-time and hide true speed.
        "--realtime_rate=0",
        # Off for clean timing; --record flips both on.
        f"--write_files={'true' if record else 'false'}",
        f"--visualize={'true' if record else 'false'}",
    ]

    # Host-side wall clock around the whole bazel-run (includes a few seconds of
    # bazel launch overhead; the per-step parse below is the clean measurement).
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=DRAKE_DIR, capture_output=True, text=True,
        env={**os.environ, "LCM_DEFAULT_URL": "memq://"},
    )
    host_wall = time.time() - t0

    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

    step_ms, substeps = parse_step_times(proc.stdout)
    bench = analyze_full(proc.stdout, dt_s=time_step,
                         simulation_time_s=simulation_time, reference="Dough Rolling")
    bench["config"] = {
        "time_step_s": time_step, "substep_s": substep, "N_substeps": n_substeps,
        "friction": friction, "stiffness": stiffness, "ppc": ppc,
        "simulation_time_s": simulation_time, "record": record,
    }
    bench["returncode"] = proc.returncode
    bench["host_wall_s_incl_bazel"] = round(host_wall, 1)
    bench["substeps_observed"] = sorted(set(s for s in substeps if s > 0))

    if record:  # collect html + obj dumps if we made them
        import re as _re
        find = subprocess.run(
            ["find", "/tmp", str(Path.home()), "/root/drake",
             "(", "-name", "test*.obj", "-o", "-name", "*.html", ")",
             "-newer", str(marker)],
            capture_output=True, text=True,
        )
        found = [Path(p) for p in find.stdout.splitlines()
                 if p.strip() and "/outputs/" not in p]
        arts = []
        for f in found:
            if f.is_file():
                dest = out_dir / f.name
                if dest.exists():
                    dest = out_dir / f"{f.parent.name}__{f.name}"
                shutil.copy2(f, dest)
                arts.append(dest.name)
        bench["artifacts"] = {"count": len(arts),
                              "examples": [a for a in arts if not a.startswith("test")][:4]}
        # Ground-truth particle count: the dough is a volumetric MPM body, so
        # each 'v' line in a frame dump is one particle (n_faces == 0, no mesh).
        obj_dumps = sorted(
            (out_dir / a for a in arts if "test" in a and a.endswith(".obj")),
            key=lambda p: int(_re.findall(r"\d+", p.stem)[-1]) if _re.findall(r"\d+", p.stem) else 0,
        )
        if obj_dumps:
            n_particles = sum(1 for ln in obj_dumps[0].read_text().splitlines()
                              if ln.startswith("v "))
            bench["particle_count"] = n_particles
            bench["particle_count_vs_paper"] = {
                "ours": n_particles, "paper": 5920,
                "ratio": round(n_particles / 5920, 3),
            }

    (out_dir / "bench.json").write_text(_json_dumps(bench))
    outputs.commit()

    print(_json_dumps({k: v for k, v in bench.items() if k != "config"}))
    if proc.returncode != 0:
        print("---- stderr tail ----")
        print(proc.stderr[-3000:])
    elif not step_ms:
        print("WARNING: no 'frame=... time=...ms' lines parsed. Check that the "
              "deformable driver logging is present and the run actually stepped.")
    return bench


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


@app.function(image=image, gpu="L40S", timeout=30 * 60, volumes={"/outputs": outputs})
def probe_count(ppc: float, time_step: float = 1e-2) -> dict:
    """Cheaply instantiate the dough at a given ppc and count particles.

    Runs only ~3 steps with file dumps on, then counts 'v' lines in the first
    frame dump (= particle count for the volumetric dough). This is the ground
    truth the analytic geometry estimate can't pin down, since it depends on
    the MPM sampler's exact cell-rounding convention.
    """
    import os
    from pathlib import Path

    scratch = Path("/tmp/sim_out")
    scratch.mkdir(exist_ok=True)
    for f in scratch.glob("test*.obj"):
        try:
            f.unlink()
        except OSError:
            pass

    # Marker so we only pick up dumps from THIS run.
    marker = Path("/tmp/.probe_marker")
    marker.touch()
    time.sleep(1.1)

    cmd = [
        "bazel", "run", "--config", "omp", f"{EXAMPLE_PKG}:{ROLL_TARGET}", "--",
        "--simulation_time=0.1", f"--time_step={time_step}", f"--ppc={ppc}",
        "--write_files=true", "--visualize=false", "--realtime_rate=0",
    ]
    proc = subprocess.run(cmd, cwd=DRAKE_DIR, capture_output=True, text=True,
                          env={**os.environ, "LCM_DEFAULT_URL": "memq://"})

    # Dump() writes "test<frame>.obj" to a RELATIVE path, so under `bazel run`
    # the files land in the runfiles cwd (~/.cache/bazel/...), not /tmp/sim_out.
    # Find them by mtime across the likely roots, like the t-shirt harness did.
    import re as _re
    find = subprocess.run(
        ["find", "/tmp", str(Path.home()), "/root/drake", "-name", "test*.obj",
         "-newer", str(marker)],
        capture_output=True, text=True,
    )
    candidates = [Path(p) for p in find.stdout.splitlines() if p.strip()]

    def frame_idx(p: Path) -> int:
        nums = _re.findall(r"\d+", p.stem)
        return int(nums[-1]) if nums else 0

    dumps = sorted(candidates, key=frame_idx)
    count = None
    counted_file = None
    if dumps:
        counted_file = str(dumps[0])
        count = sum(1 for ln in dumps[0].read_text().splitlines() if ln.startswith("v "))
    hints = [l.strip() for l in proc.stdout.splitlines()
             if any(k in l.lower() for k in ("particle", "n_vert", "dof", " nv"))][:5]
    result = {"ppc": ppc, "particle_count": count, "counted_file": counted_file,
              "n_dumps_found": len(dumps),
              "returncode": proc.returncode, "log_hints": hints}
    if count is None:
        result["stderr_tail"] = proc.stderr[-1500:]
        result["stdout_tail"] = proc.stdout[-1500:]
    print(_json_dumps(result))
    return result


@app.local_entrypoint()
def findppc(candidates: str = "1,2,3,4", target: int = 5920):
    """Probe ppc values in parallel, map the ppc->particle-count curve, and
    recommend the ppc nearest the paper's count. Defaults span low ppc because
    the count scales steeply (ppc=6 already gives ~13.6k). Pass fractional
    candidates too, e.g. --candidates 2,2.5,3,3.5
    e.g. modal run drake_dough_modal.py::findppc
    """
    cs = sorted(float(x) for x in candidates.split(","))
    handles = [(c, probe_count.spawn(c)) for c in cs]
    print(f"\n{'ppc':>6} {'particles':>10}  {'x paper':>8}  {'delta':>8}")
    pairs = []
    best = None
    for c, h in handles:
        r = h.get()
        n = r.get("particle_count")
        if n:
            pairs.append((c, n))
            d = abs(n - target)
            print(f"{c:>6} {n:>10}  {n / target:>7.2f}x  {d:>8}")
            if best is None or d < best[2]:
                best = (c, n, d)
        else:
            print(f"{c:>6} {'FAILED':>10}  (rc={r.get('returncode')}, "
                  f"dumps={r.get('n_dumps_found')})")

    if not pairs:
        print("\nNo counts returned -- check probe_count logs.")
        return

    # Linear interpolation across the bracketing pair to suggest a ppc that
    # would hit the target exactly (the sampler may quantize, so treat as a
    # starting guess to verify with one more probe).
    pairs.sort()
    suggestion = None
    for (c0, n0), (c1, n1) in zip(pairs, pairs[1:]):
        if n0 <= target <= n1 and n1 != n0:
            suggestion = c0 + (c1 - c0) * (target - n0) / (n1 - n0)
            break

    print(f"\nClosest probed: ppc={best[0]} -> {best[1]} ({best[1] / target:.2f}x paper)")
    if suggestion:
        print(f"Interpolated ppc for ~{target}: {suggestion:.2f}  "
              f"(verify: modal run drake_dough_modal.py::probe_count --ppc {suggestion:.2f})")
    print(f"Then matched run: modal run drake_dough_modal.py::main "
          f"--ppc <chosen> --record")


@app.function(image=image, volumes={"/outputs": outputs}, timeout=600)
def rescore(run_id: str, dt_s: float = 1e-2, simulation_time: float = 10.0) -> dict:
    """Re-analyze a saved stdout.log on the Volume without re-simulating."""
    from bench_realtime import analyze_log_file

    outputs.reload()
    out = analyze_log_file(Path("/outputs") / run_id / "stdout.log",
                           dt_s=dt_s, simulation_time_s=simulation_time)
    print(_json_dumps(out))
    return out


@app.local_entrypoint()
def main(simulation_time: float = 10.0, record: bool = False, ppc: float = 3):
    """Single dough run at paper settings (dt=10 ms, N=10, mu=1.0).
    Pass --ppc to match the paper's particle count (use ::findppc to calibrate);
    --record dumps frames so the particle count is reported."""
    b = run_dough.remote(simulation_time=simulation_time, record=record, ppc=ppc)
    rt = b.get("real_time_rate", {})
    pc = b.get("paper_comparison", {})
    print("\n=== Dough rolling benchmark ===")
    print(f"  steps logged       : {b.get('n_steps_logged')}")
    if b.get("particle_count") is not None:
        pcvp = b.get("particle_count_vs_paper", {})
        print(f"  particles          : {b['particle_count']}"
              f"   [paper: 5920, ratio {pcvp.get('ratio')}]")
    print(f"  runtime/step (mean): {b.get('runtime_ms_per_step', {}).get('all_steps', {}).get('mean')} ms"
          f"   [paper: {pc.get('paper_runtime_ms')} ms]")
    print(f"  real-time rate     : {rt.get('from_mean_all')}"
          f"   [paper: {pc.get('paper_rt_rate')} at eps_r=5e-2]")
    print(f"  note               : {pc.get('note')}")
    ca = b.get("contact_analysis")
    if ca:
        print("\n  --- contact-regime split (measured n_contacts) ---")
        print(f"  {ca['interpretation']}")
    print(f"\nFetch: modal volume get drake-mpm-outputs "
          f"{b.get('config', {}).get('run_id', '<run_id>')} ./results/")

#
@app.local_entrypoint()
def bench(run_id: str, dt: float = 1e-2, sim_time: float = 10.0):
    """Re-analyze a saved run: modal run drake_dough_modal.py::bench --run-id ..."""
    rescore.remote(run_id, dt_s=dt, simulation_time=sim_time)