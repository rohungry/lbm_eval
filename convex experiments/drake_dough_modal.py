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
    for f in scratch.iterdir():
        f.unlink()

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
        arts = []
        for f in scratch.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)
                arts.append(f.name)
        bench["artifacts"] = {"count": len(arts),
                              "examples": [a for a in arts if not a.startswith("test")][:4]}

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
def main(simulation_time: float = 10.0, record: bool = False):
    """Single dough run at paper settings (dt=10 ms, N=10, mu=1.0)."""
    b = run_dough.remote(simulation_time=simulation_time, record=record)
    rt = b.get("real_time_rate", {})
    pc = b.get("paper_comparison", {})
    print("\n=== Dough rolling benchmark ===")
    print(f"  steps logged       : {b.get('n_steps_logged')}")
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