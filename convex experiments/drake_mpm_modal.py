"""
Run the t-shirt folding eval from "A Convex Formulation of Material Points and
Rigid Bodies with GPU-Accelerated Async-Coupling" (arXiv:2503.05046, IROS 2025)
on Modal GPUs.

The paper's implementation lives in the g1n0st/drake fork on the
`cuda-mpm-weak-coupling-clean` branch. Official support matrix per the README:
Ubuntu 22.04 + CUDA 12.1, built with Bazel (`bazel run <target> --config omp`).

Usage:
    # one-off run (default: mpm_tshirt_folding, 25 s sim)
    modal run drake_mpm_modal.py

    # the README's "official" target for the paper's Sec. VI-E figure
    modal run drake_mpm_modal.py --target dual_arm_flipping

    # parallel parameter sweep across GPUs
    modal run drake_mpm_modal.py::sweep

    # pull results down afterwards
    modal volume get drake-mpm-outputs / ./results

Notes:
  * The Drake-from-source build is BIG (~1-2 h, needs >= 16 GB RAM). It runs
    once at image-build time and is cached by Modal thereafter.
  * L40S is sm_89 -- the same Ada architecture as the RTX 4090 used in the
    paper -- which avoids any surprises if the fork's nvcc flags assume Ada.
"""

import subprocess
import time
from pathlib import Path

import modal

app = modal.App("drake-mpm-tshirt-folding")

# Persisted storage for simulation outputs (Meshcat HTML recordings, MPM dumps).
outputs = modal.Volume.from_name("drake-mpm-outputs", create_if_missing=True)

REPO = "https://github.com/g1n0st/drake.git"
BRANCH = "cuda-mpm-weak-coupling-clean"
# The commit you linked. Pin it for reproducibility.
COMMIT = "c0d004f5d5f9a279b81e50c8a8071c1ac493663c"
DRAKE_DIR = "/root/drake"
EXAMPLE_PKG = "//examples/multibody/deformable"

# Demo targets we pre-build into the image. Add more here if you want them
# baked in (anything not pre-built will compile lazily on first run, which is
# slow but works since the bazel cache lives in the image).
PREBUILT_TARGETS = [
    "mpm_tshirt_folding",   # the file you linked: 4 floating prismatic grippers
    "dual_arm_flipping",    # README's target for the paper's Sec. VI-E figure
]

# ---------------------------------------------------------------------------
# Image: CUDA 12.1 devel (nvcc available at build time -- no GPU needed to
# *compile* CUDA code), Drake prereqs, clone + patch + bazel build.
# ---------------------------------------------------------------------------

# The example hardcodes the author's desktop paths; patch them to stable
# container paths. The t-shirt OBJ meshes ship at the repo root.
# NOTE: each run_commands() entry becomes one Dockerfile RUN line, so these
# must be single-line strings (no embedded newlines / backslash continuations).
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
        # Clone the fork at the pinned commit.
        f"git clone --branch {BRANCH} {REPO} {DRAKE_DIR}",
        f"cd {DRAKE_DIR} && git checkout {COMMIT}",
        # Drake's official prereq installer (installs bazelisk, compilers,
        # and all third-party system deps).
        f"cd {DRAKE_DIR} && DEBIAN_FRONTEND=noninteractive "
        "./setup/ubuntu/install_prereqs.sh -y || "
        f"(cd {DRAKE_DIR} && DEBIAN_FRONTEND=noninteractive yes | ./setup/ubuntu/install_prereqs.sh)",
        # Patch hardcoded author paths -> container paths.
        PATCH_MESH_CMD,
        PATCH_OUT_CMD,
        "mkdir -p /tmp/sim_out",
    )
    .run_commands(
        # The expensive step: compile Drake + the CUDA MPM extension.
        # nvcc compiles fine without a GPU attached; only *running* needs one.
        # Split into two layers so a checkpoint lands partway through: bazel's
        # cache (~/.cache/bazel) persists between layers, so the second layer
        # resumes instead of starting over. --jobs caps peak RAM on the builder.
        "cd /root/drake && bazel build --config omp --jobs=8 //multibody/plant",
    )
    .run_commands(
        "cd /root/drake && bazel build --config omp --jobs=8 "
        + " ".join(f"{EXAMPLE_PKG}:{t}" for t in PREBUILT_TARGETS),
    )
    .env(
        {
            # LCM multicast doesn't work inside containers; use an in-memory
            # queue so the DrakeVisualizer publisher doesn't error out.
            "LCM_DEFAULT_URL": "memq://",
        }
    )
    # Scoring deps, appended AFTER the bazel layers so the expensive build
    # cache stays valid. add_local_python_source ships at runtime (no rebuild).
    .pip_install("numpy")
    .add_local_python_source("score_fold")
)


# ---------------------------------------------------------------------------
# The eval runner
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L40S",  # sm_89, same arch as the paper's RTX 4090
    timeout=4 * 60 * 60,
    volumes={"/outputs": outputs},
)
def run_eval(
    target: str = "mpm_tshirt_folding",
    simulation_time: float = 25.0,
    time_step: float = 1e-2,
    substep: float = 5e-4,
    stiffness: float = 100.0,
    friction: float = 1.0,
    contact_approximation: str = "sap",
    write_files: bool = False,
    run_id: str | None = None,
) -> dict:
    """Run one simulation; copy artifacts to the shared Volume; return stats."""
    import os
    import shutil

    run_id = run_id or f"{target}-{int(time.time())}"
    out_dir = Path("/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check the GPU is visible.
    subprocess.run(["nvidia-smi"], check=False)

    scratch = Path("/tmp/sim_out")
    scratch.mkdir(exist_ok=True)
    for f in scratch.iterdir():
        f.unlink()

    # mtime marker: anything created after this is a candidate artifact.
    marker = Path("/tmp/.run_marker")
    marker.touch()
    time.sleep(1.1)  # ensure mtimes strictly after marker on coarse filesystems

    cmd = [
        "bazel",
        "run",
        "--config",
        "omp",
        f"{EXAMPLE_PKG}:{target}",
        "--",
        f"--simulation_time={simulation_time}",
        f"--time_step={time_step}",
        f"--substep={substep}",
        f"--stiffness={stiffness}",
        f"--friction={friction}",
        f"--contact_approximation={contact_approximation}",
        f"--write_files={'true' if write_files else 'false'}",
        # Headless eval: don't throttle to wall-clock.
        "--realtime_rate=0",
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=DRAKE_DIR,
        capture_output=True,
        text=True,
        env={**os.environ, "LCM_DEFAULT_URL": "memq://"},
    )
    wall = time.time() - t0

    # The deformable driver logs per-step solver timing:
    #   "frame=<N> time=<ms>ms N(substeps)=<k>"  (ANSI-colored)
    # which is directly comparable to the paper's Table I runtime column
    # (21.5 ms/step for t-shirt folding on an RTX 4090).
    import re

    step_ms = [int(m) for m in re.findall(r"frame=\d+ time=(\d+)ms", proc.stdout)]
    timing = (
        {
            "steps_logged": len(step_ms),
            "ms_per_step_mean": round(sum(step_ms) / len(step_ms), 2),
            "ms_per_step_median": float(sorted(step_ms)[len(step_ms) // 2]),
            "ms_per_step_max": max(step_ms),
        }
        if step_ms
        else None
    )

    # Persist logs.
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

    # Collect artifacts: the patched scratch dir, PLUS an mtime sweep of the
    # places the MPM dump code might write to with paths we didn't patch --
    # notably the bazel runfiles tree, since `bazel run` sets cwd inside
    # ~/.cache/bazel/.../execroot, so relative-path dumps land there.
    dump_exts = (
        "*.obj *.ply *.npy *.vtk *.vtu *.bgeo *.xyz *.csv *.dat *.bin *.html"
    ).split()
    find_cmd = ["find", "/tmp", str(Path.home()), "/root/drake", "-type", "f",
                "-newer", str(marker), "("]
    for i, pat in enumerate(dump_exts):
        if i:
            find_cmd.append("-o")
        find_cmd += ["-name", pat]
    find_cmd.append(")")
    found = subprocess.run(find_cmd, capture_output=True, text=True)
    candidates = {Path(p) for p in found.stdout.splitlines() if p.strip()}
    candidates |= set(scratch.iterdir())

    artifacts = []
    manifest = []
    for f in sorted(candidates):
        if not f.is_file() or "/outputs/" in str(f):
            continue
        manifest.append(str(f))  # record original location for debugging
        # Frame series can collide on bare names across dirs; disambiguate
        # with the parent dir when needed.
        dest = out_dir / f.name
        if dest.exists():
            dest = out_dir / f"{f.parent.name}__{f.name}"
        shutil.copy2(f, dest)
        artifacts.append(dest.name)
    (out_dir / "artifact_manifest.txt").write_text("\n".join(manifest))

    # Score in-container while the frames are on local disk -- much faster
    # than re-reading thousands of small files back through the Volume mount.
    fold_score = None
    if write_files:
        try:
            from score_fold import score_run_dir

            fold_score = score_run_dir(out_dir, sim_time=simulation_time)
        except Exception as e:  # never let scoring sink a finished sim
            fold_score = {"success": None, "error": repr(e)}

    outputs.commit()

    result = {
        "run_id": run_id,
        "target": target,
        "returncode": proc.returncode,
        "wall_clock_s": round(wall, 1),
        "sim_time_s": simulation_time,
        "realtime_rate": round(simulation_time / wall, 3) if wall > 0 else None,
        "solver_timing": timing,
        "score": fold_score,
        "artifacts": {
            "count": len(artifacts),
            "examples": sorted(a for a in artifacts if not a.startswith("test"))[:5]
            + artifacts[:3],
        },
        "params": {
            "time_step": time_step,
            "substep": substep,
            "stiffness": stiffness,
            "friction": friction,
            "contact_approximation": contact_approximation,
        },
    }
    print(result)
    if proc.returncode != 0:
        # Surface the tail of stderr for quick debugging in Modal logs.
        print("---- stderr tail ----")
        print(proc.stderr[-4000:])
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=600)
def inspect_dump_paths() -> str:
    """Grep the fork's source to find where write_files actually dumps data.

    Run with:  modal run drake_mpm_modal.py::inspect
    """
    chunks = []
    greps = [
        # Where the write_files flag is consumed.
        ["grep", "-rn", "--include=*.cc", "--include=*.h", "--include=*.cu",
         "--include=*.cuh", "-B2", "-A8", "write_files", "/root/drake/multibody"],
        # Any file-writing calls in the GPU MPM code.
        ["grep", "-rnE", "--include=*.cc", "--include=*.h", "--include=*.cu",
         "--include=*.cuh", "-B2", "-A4",
         r"ofstream|fopen|WriteObj|ExportTo|DumpParticles|\.obj\"|\.ply\"|\.npy\"|\.bgeo\"",
         "/root/drake/multibody/gpu_mpm"],
        # Any remaining hardcoded absolute paths anywhere in multibody.
        ["grep", "-rn", "--include=*.cc", "--include=*.h", "--include=*.cu",
         "--include=*.cuh", "/home/changyu", "/root/drake/multibody"],
    ]
    for cmd in greps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        chunks.append(f"$ {' '.join(cmd[:4])} ... {cmd[-2]}\n{r.stdout or '(no matches)'}")
    report = "\n\n".join(chunks)
    print(report)
    return report


@app.local_entrypoint()
def inspect():
    inspect_dump_paths.remote()


# ---------------------------------------------------------------------------
# Scoring (lbm_eval-style: binary success + scalar diagnostics per rollout)
# ---------------------------------------------------------------------------
score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .add_local_python_source("score_fold")
)


@app.function(image=score_image, volumes={"/outputs": outputs}, timeout=1800)
def score_run(run_id: str, sim_time: float = 25.0, eval_time: float | None = None) -> dict:
    """Re-score one rollout from the Volume (CPU only). Normal runs are scored
    in-container by run_eval; this exists for re-scoring with new thresholds."""
    from score_fold import score_run_dir

    outputs.reload()
    result = score_run_dir(
        Path("/outputs") / run_id, sim_time=sim_time, eval_time=eval_time
    )
    outputs.commit()  # persist score.json next to the dumps
    print(result)
    return result


@app.local_entrypoint()
def score(run_id: str, sim_time: float = 25.0, eval_time: float = None):
    """e.g. modal run drake_mpm_modal.py::score --run-id sweep-mu1.0-sub0.0005"""
    score_run.remote(run_id, sim_time=sim_time, eval_time=eval_time)


@app.function(image=score_image, volumes={"/outputs": outputs}, timeout=300)
def trajectory(run_id: str, rows: int = 30) -> None:
    """Print the fold trajectory (area & thickness vs time) from score.json.

    Reading the curve against the controller's timeline:
      t<0.5    free-fall/settle          ~9.75-10.75  regrasp for fold 2
      4->9.75  fold 1 (grasp/lift/fwd)   ~10.75-16.5  fold 2
      >16.5    release, at rest
    A good fold: area steps down ~0.5 then ~0.25-0.45 and PLATEAUS.
    A slip: area drops then creeps back up after release.
    """
    import json as _json

    outputs.reload()
    data = _json.loads((Path("/outputs") / run_id / "score.json").read_text())
    pf = [m for m in data.get("per_frame", []) if "hull_area_xy" in m]
    if not pf:
        print("no per-frame metrics in score.json; re-score this run first")
        return
    ref = next((m for m in pf if m["frame"] == data["ref_frame"]), pf[0])
    stride = max(1, len(pf) // rows)
    sampled = pf[::stride]
    if sampled[-1] is not pf[-1]:
        sampled.append(pf[-1])
    print(f"{'t[s]':>6} {'area_ratio':>10} {'thick_ratio':>11}  area")
    for m in sampled:
        ar = m["hull_area_xy"] / ref["hull_area_xy"]
        tr = m["thickness_z"] / ref["thickness_z"]
        print(f"{m['t']:>6.2f} {ar:>10.3f} {tr:>11.2f}  {'#' * int(round(ar * 40))}")


@app.local_entrypoint()
def traj(run_id: str, rows: int = 30):
    """e.g. modal run drake_mpm_modal.py::traj --run-id sweep-mu1.0-sub0.00025"""
    trajectory.remote(run_id, rows=rows)


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    target: str = "mpm_tshirt_folding",
    simulation_time: float = 25.0,
    write_files: bool = False,
):
    """Single run."""
    result = run_eval.remote(
        target=target,
        simulation_time=simulation_time,
        write_files=write_files,
    )
    print("\nDone. Fetch artifacts with:")
    print(f"  modal volume get drake-mpm-outputs {result['run_id']} ./results/")


@app.local_entrypoint()
def sweep():
    """Parallel eval sweep -- each config gets its own GPU container.

    This is where Modal earns its keep: fan out N simulations concurrently
    instead of serializing them on one workstation GPU.
    """
    configs = []
    for friction in (0.6, 1.0):
        for substep in (5e-4, 2.5e-4):
            configs.append(
                dict(
                    target="mpm_tshirt_folding",
                    friction=friction,
                    substep=substep,
                    write_files=True,  # needed for fold scoring
                    run_id=f"sweep-mu{friction}-sub{substep}",
                )
            )

    # spawn() fans out immediately; each call lands on its own GPU container.
    handles = [run_eval.spawn(**cfg) for cfg in configs]

    print("\n=== Sweep summary ===")
    n_success = 0
    for h in handles:
        try:
            r = h.get()
            s = r.get("score") or score_run.remote(r["run_id"], sim_time=r["sim_time_s"])
            n_success += bool(s.get("success"))
            ms = (r.get("solver_timing") or {}).get("ms_per_step_median")
            print(
                f"{r['run_id']}: rc={r['returncode']} wall={r['wall_clock_s']}s "
                f"ms/step~{ms} | success={s.get('success')} "
                f"area_ratio={s.get('area_ratio')} layers~{s.get('est_layers')}"
            )
        except Exception as e:
            print(f"FAILED: {e}")
    print(f"\nSuccess rate: {n_success}/{len(configs)}")
    print("\nFetch all artifacts with:")
    print("  modal volume get drake-mpm-outputs / ./results/")