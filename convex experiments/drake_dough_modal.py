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

# --- YOUR FORK (set these two to test your HvP branch) ----------------------
# The image keeps cloning g1n0st (so the expensive Drake build stays cached from
# your earlier runs), then adds your fork as a remote and builds YOUR branch in
# one cheap extra layer. At runtime, functions can `git fetch fork` + rebuild to
# pick up new commits without rebuilding the image -- minutes, not hours.
# Leave FORK_REPO = "" to run stock g1n0st (no fork layer added).
FORK_REPO = "https://github.com/rohungry/convex_mpm_gpu.git"           # e.g. "https://github.com/rohungry/drake.git"
FORK_BRANCH = "hvp-newton-cg"

ROLL_HTML_SED = (
    "sed -i 's|/home/changyu/drake/roll.html|/tmp/sim_out/roll.html|g' "
    "examples/multibody/deformable/roll.cc"
)

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
)

# One extra, cheap layer that builds YOUR fork branch on top of the cached Drake
# build. Added only if FORK_REPO is set. Because every layer above is byte-
# identical to your prior runs, they're reused from cache; this layer just adds
# the fork remote, checks out your branch, re-applies the roll.html patch, and
# incrementally rebuilds roll (warm bazel cache => minutes).
if FORK_REPO:
    image = image.run_commands(
        f"cd {DRAKE_DIR} && "
        f"(git remote add fork {FORK_REPO} || git remote set-url fork {FORK_REPO}) && "
        f"git fetch fork {FORK_BRANCH} && "
        f"git reset --hard fork/{FORK_BRANCH} && "
        f"{ROLL_HTML_SED} && "
        f"bazel build --config omp --jobs=8 {EXAMPLE_PKG}:{ROLL_TARGET}",
    )

# add_local_* MUST be the final image op (Modal forbids build steps after it),
# so this comes AFTER the conditional fork layer above.
image = image.add_local_python_source("bench_realtime")


def _sync_and_build(commit: str | None) -> tuple[bool, str]:
    """Inside the container: fetch the fork, hard-checkout `commit` (or the fork
    branch head if commit in {None,'latest'}), re-apply the roll.html patch, and
    incrementally rebuild roll on the warm bazel cache (minutes). Returns
    (ok, tail-log). No-op-with-error if the fork remote isn't configured."""
    import os

    if not FORK_REPO:
        return False, ("FORK_REPO is empty. Set FORK_REPO/FORK_BRANCH at the top "
                       "of drake_dough_modal.py and re-deploy the image once.")
    target = f"fork/{FORK_BRANCH}" if commit in (None, "latest") else commit
    log = []

    def run(cmd, shell=False):
        p = subprocess.run(cmd, cwd=DRAKE_DIR, capture_output=True, text=True, shell=shell)
        log.append(f"$ {cmd if shell else ' '.join(cmd)}\n{p.stdout[-400:]}{p.stderr[-400:]}")
        return p

    if run(["git", "fetch", "fork", FORK_BRANCH]).returncode != 0:
        return False, "git fetch failed:\n" + "\n".join(log)
    if run(["git", "reset", "--hard", target]).returncode != 0:
        return False, "git reset failed (bad commit/branch?):\n" + "\n".join(log)
    run(ROLL_HTML_SED, shell=True)  # idempotent
    b = run(["bazel", "build", "--config", "omp", "--jobs=8",
             f"{EXAMPLE_PKG}:{ROLL_TARGET}"])
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=DRAKE_DIR, capture_output=True, text=True).stdout.strip()
    if b.returncode != 0:
        return False, f"bazel build FAILED at {head}:\n" + b.stderr[-3000:]
    return True, f"built {head}"


def _execute_dough(*, time_step, substep, friction, stiffness, simulation_time,
                   ppc, record=False):
    """Run the roll binary once and parse it. No volume I/O -- returns
    (bench dict, completed-process, parsed per-step list). Callable in a loop
    within one container for variance/determinism measurement."""
    import os
    from bench_realtime import analyze_full, parse_step_times, parse_steps_with_contacts

    cmd = [
        "bazel", "run", "--config", "omp", f"{EXAMPLE_PKG}:{ROLL_TARGET}", "--",
        f"--simulation_time={simulation_time}",
        f"--time_step={time_step}",
        f"--substep={substep}",
        f"--friction={friction}",
        f"--stiffness={stiffness}",
        f"--ppc={ppc}",
        # 0 = run unthrottled (the default 1.0 caps wall-clock at real time).
        "--realtime_rate=0",
        f"--write_files={'true' if record else 'false'}",
        f"--visualize={'true' if record else 'false'}",
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=DRAKE_DIR, capture_output=True, text=True,
        env={**os.environ, "LCM_DEFAULT_URL": "memq://"},
    )
    host_wall = time.time() - t0

    _, substeps = parse_step_times(proc.stdout)
    bench = analyze_full(proc.stdout, dt_s=time_step,
                         simulation_time_s=simulation_time, reference="Dough Rolling")
    bench["config"] = {
        "time_step_s": time_step, "substep_s": substep,
        "N_substeps": round(time_step / substep),
        "friction": friction, "stiffness": stiffness, "ppc": ppc,
        "simulation_time_s": simulation_time, "record": record,
    }
    bench["returncode"] = proc.returncode
    bench["host_wall_s_incl_bazel"] = round(host_wall, 1)
    bench["substeps_observed"] = sorted(set(s for s in substeps if s > 0))
    steps = parse_steps_with_contacts(proc.stdout)
    return bench, proc, steps


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
    commit: str | None = None,   # if set: git fetch fork + checkout + rebuild first
) -> dict:
    """Run roll.cc unthrottled, parse per-step timing, compute real-time rate."""
    import os
    import shutil

    if commit is not None:
        ok, tail = _sync_and_build(commit)
        if not ok:
            print(tail)
            return {"built": False, "build_error": tail}

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

    bench, proc, _ = _execute_dough(
        time_step=time_step, substep=substep, friction=friction,
        stiffness=stiffness, simulation_time=simulation_time, ppc=ppc, record=record,
    )

    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

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
    elif not bench.get("n_steps_logged"):
        print("WARNING: no 'frame=... time=...ms' lines parsed. Check that the "
              "deformable driver logging is present and the run actually stepped.")
    return bench


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


def _mean_std(xs: list[float]) -> dict:
    import statistics
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    mean = statistics.mean(xs)
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "n": len(xs),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "cv_pct": round(100 * std / mean, 2) if mean else None,  # noise floor as %
        "min": round(min(xs), 3),
        "max": round(max(xs), 3),
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 60 * 60,
    volumes={"/outputs": outputs},
)
def repeat_runs(
    n: int = 5,
    time_step: float = 1e-2,
    substep: float = 1e-3,
    friction: float = 1.0,
    stiffness: float = 1e3,
    simulation_time: float = 10.0,
    ppc: float = 1.45,           # paper-matched (5,999 particles)
    run_id: str | None = None,
) -> dict:
    """Run the SAME config n times in ONE container (one warm GPU) to measure
    run-to-run variance and determinism. Same-instance is deliberate: it's the
    condition you'd A/B two algorithms under (back-to-back on one GPU), so its
    spread is the noise floor a real % lift must clear. Inter-instance variance
    would be larger."""
    run_id = run_id or f"dough-repeat-n{n}-ppc{ppc}-{int(time.time())}"
    out_dir = Path("/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["nvidia-smi"], check=False)

    per_run = []
    step_seqs = []  # per-run [(peak_contacts, total_iters), ...] for determinism
    for i in range(n):
        bench, proc, steps = _execute_dough(
            time_step=time_step, substep=substep, friction=friction,
            stiffness=stiffness, simulation_time=simulation_time, ppc=ppc, record=False,
        )
        if bench["returncode"] != 0:
            per_run.append({"run": i, "returncode": bench["returncode"], "failed": True})
            continue
        rt = bench["real_time_rate"]
        st = bench["runtime_ms_per_step"]["all_steps"]
        ca = bench.get("contact_analysis", {})
        ic = ca.get("regimes", {}).get("in_contact", {})
        per_run.append({
            "run": i,
            "runtime_ms_mean": st["mean"],
            "runtime_ms_median": st["median"],
            "real_time_rate": rt["from_mean_all"],
            "in_contact_mean_ms": ic.get("mean_ms"),
            "contact_steps_fraction": ca.get("contact_steps_fraction"),
            "host_wall_s": bench["host_wall_s_incl_bazel"],
        })
        step_seqs.append([(s["peak_contacts"], s["total_iters"]) for s in steps])

    ok = [r for r in per_run if not r.get("failed")]

    # Determinism: compare each run's per-frame (contacts, iters) to run 0.
    determinism = {"checked": len(step_seqs) >= 2}
    if len(step_seqs) >= 2:
        ref = step_seqs[0]
        diffs = []
        for i, seq in enumerate(step_seqs[1:], 1):
            m = min(len(ref), len(seq))
            c_mis = sum(1 for a, b in zip(ref[:m], seq[:m]) if a[0] != b[0])
            it_mis = sum(1 for a, b in zip(ref[:m], seq[:m]) if a[1] != b[1])
            diffs.append({"vs_run0": i, "frames": m,
                          "contact_count_mismatches": c_mis,
                          "iter_count_mismatches": it_mis})
        determinism["bit_identical_contacts"] = all(d["contact_count_mismatches"] == 0 for d in diffs)
        determinism["bit_identical_iters"] = all(d["iter_count_mismatches"] == 0 for d in diffs)
        determinism["detail"] = diffs

    summary = {
        "run_id": run_id, "n_requested": n, "n_succeeded": len(ok),
        "config": {"time_step_s": time_step, "substep_s": substep, "ppc": ppc,
                   "friction": friction, "simulation_time_s": simulation_time},
        "variance": {
            "runtime_ms_per_step": _mean_std([r["runtime_ms_mean"] for r in ok]),
            "real_time_rate": _mean_std([r["real_time_rate"] for r in ok]),
            "in_contact_mean_ms": _mean_std([r["in_contact_mean_ms"] for r in ok]),
            "contact_steps_fraction": _mean_std([r["contact_steps_fraction"] for r in ok]),
        },
        "determinism": determinism,
        "per_run": per_run,
    }
    # Actionable: the smallest lift distinguishable from this baseline's own noise.
    rt_cv = summary["variance"]["real_time_rate"].get("cv_pct")
    if rt_cv is not None:
        summary["min_distinguishable_lift_pct"] = round(2 * rt_cv, 2)
        summary["min_lift_note"] = (
            "A real-time-rate lift should exceed ~2x the baseline CV "
            f"(~{round(2 * rt_cv, 1)}%) to be distinguishable from run-to-run noise "
            "on the same GPU. Inter-instance comparison needs a larger margin."
        )

    (out_dir / "repeat_summary.json").write_text(_json_dumps(summary))
    outputs.commit()
    print(_json_dumps(summary))
    return summary


@app.function(image=image, gpu="L40S", timeout=60 * 60, volumes={"/outputs": outputs})
def run_selftest(commit: str | None = None, ppc: float = 1.45,
                 simulation_time: float = 5.0) -> dict:
    """Build (optionally syncing to `commit`) and run the HvP self-test.
    Sets MPM_HVP_SELFTEST=1, runs roll until the first contact-bearing substep,
    and echoes the three [hvp] PASS/FAIL lines. The C++ hook std::exit(0)s after
    printing, so the run stops as soon as contacts appear."""
    import os
    import re

    if commit is not None:
        ok, tail = _sync_and_build(commit)
        if not ok:
            print(tail)
            return {"built": False, "build_error": tail}

    subprocess.run(["nvidia-smi"], check=False)
    env = {**os.environ, "LCM_DEFAULT_URL": "memq://", "MPM_HVP_SELFTEST": "1"}
    cmd = [
        "bazel", "run", "--config", "omp", f"{EXAMPLE_PKG}:{ROLL_TARGET}", "--",
        f"--simulation_time={simulation_time}", f"--ppc={ppc}",
        "--realtime_rate=0", "--visualize=false", "--write_files=false",
    ]
    p = subprocess.run(cmd, cwd=DRAKE_DIR, capture_output=True, text=True, env=env)

    hvp = [ln for ln in p.stdout.splitlines() if "[hvp" in ln]
    # strip ANSI color if the driver added any
    hvp = [re.sub(r"\x1b\[[0-9;]*m", "", ln).strip() for ln in hvp]
    passed = any("OVERALL: PASS" in ln for ln in hvp)
    saw_overall = any("OVERALL" in ln for ln in hvp)

    print("\n=== HvP self-test ===")
    if hvp:
        for ln in hvp:
            print("  " + ln)
    else:
        print("  (no [hvp] lines found)")
        print("  --- stdout tail ---\n" + p.stdout[-1800:])
        print("  --- stderr tail ---\n" + p.stderr[-1800:])
        # common cause: never reached a contact-bearing substep in the window
        if "no contacts" not in p.stdout:
            print("  hint: contacts may not have formed within "
                  f"simulation_time={simulation_time}s; try a larger value.")

    return {"built": True, "returncode": p.returncode, "passed": passed,
            "saw_overall": saw_overall, "hvp_lines": hvp}


@app.function(image=image, gpu="L40S", timeout=30 * 60, volumes={"/outputs": outputs})
def probe_count(ppc: float, time_step: float = 1e-2,
                commit: str | None = None) -> dict:
    """Cheaply instantiate the dough at a given ppc and count particles.

    Runs only ~3 steps with file dumps on, then counts 'v' lines in the first
    frame dump (= particle count for the volumetric dough). This is the ground
    truth the analytic geometry estimate can't pin down, since it depends on
    the MPM sampler's exact cell-rounding convention.
    """
    import os
    from pathlib import Path

    if commit is not None:
        ok, tail = _sync_and_build(commit)
        if not ok:
            print(tail)
            return {"built": False, "build_error": tail, "particle_count": None}

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
def selftest(commit: str = "latest", ppc: float = 1.45, simulation_time: float = 5.0):
    """Push your fork, then: modal run drake_dough_modal.py::selftest
    Syncs the container to your fork branch head (or --commit <sha>), rebuilds
    roll incrementally, runs the HvP self-test, prints the three PASS/FAIL lines.
    Tight loop: edit -> git push -> this (minutes), no image rebuild."""
    r = run_selftest.remote(commit=commit, ppc=ppc, simulation_time=simulation_time)
    if not r.get("built", True):
        print("\nBUILD FAILED — see tail above.")
        return
    if r.get("passed"):
        print("\n  => operator validated. Next: Newton-CG.")
    elif r.get("saw_overall"):
        print("\n  => a check FAILED. FD-large: v0/current_velocities binding; "
              "symmetry: scatter transpose; PD: precompute sign. Paste the lines.")
    else:
        print("\n  => self-test didn't run to completion (no OVERALL line). "
              "Check the tail above.")


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


@app.local_entrypoint()
def bench(run_id: str, dt: float = 1e-2, sim_time: float = 10.0):
    """Re-analyze a saved run: modal run drake_dough_modal.py::bench --run-id ..."""
    rescore.remote(run_id, dt_s=dt, simulation_time=sim_time)


@app.local_entrypoint()
def repeat(n: int = 5, ppc: float = 1.45, simulation_time: float = 10.0):
    """Characterize run-to-run variance + determinism for a fixed config.
    e.g. modal run drake_dough_modal.py::repeat --n 8
    Run this for your baseline AND your new algorithm; a % lift is real only
    if it exceeds the reported min-distinguishable-lift."""
    s = repeat_runs.remote(n=n, ppc=ppc, simulation_time=simulation_time)
    v = s["variance"]
    d = s["determinism"]
    print(f"\n=== Repeatability over {s['n_succeeded']}/{s['n_requested']} runs "
          f"(ppc={ppc}, same GPU) ===")
    for key, label in [("runtime_ms_per_step", "runtime/step (ms)"),
                       ("real_time_rate", "real-time rate"),
                       ("in_contact_mean_ms", "in-contact step (ms)")]:
        m = v[key]
        if m.get("n"):
            print(f"  {label:<22}: {m['mean']} +/- {m['std']}  "
                  f"(CV {m['cv_pct']}%, range {m['min']}-{m['max']})")
    if d.get("checked"):
        det = d.get("bit_identical_contacts") and d.get("bit_identical_iters")
        print(f"  determinism            : "
              f"{'bit-identical across runs' if det else 'NON-deterministic (GPU atomics)'}")
    if s.get("min_distinguishable_lift_pct") is not None:
        print(f"\n  -> {s['min_lift_note']}")