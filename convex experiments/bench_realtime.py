"""
Real-time-rate analysis for the GPU MPM benchmark (arXiv:2503.05046).

The deformable driver prints one line per time step:

    frame=<N> time=<ms>ms N(substeps)=<k>

where <ms> is the wall-clock COMPUTE time for that step (the N-substep loop,
GPU-synced by the blocking cudaMemcpy at the end of the contact solve). This
module parses those lines and reproduces the paper's two reported quantities:

  * Runtime [ms] per time step   -> Table I  (dough: 21.7 ms)
  * Real-time rate = sim / wall  -> Sec. VI-B (dough: ~46% at eps_r=5e-2,
                                    59% at the looser eps_r=1e-1)

Definitions, made explicit because the paper overloads "simulation time":
  - simulation (physical) time advanced per step = dt           [numerator]
  - wall-clock compute time per step             = runtime ms   [denominator]
  - real-time rate                               = dt / runtime

Cross-check available from the paper itself: the Laundry row (dt=5 ms,
runtime=18.9 ms) gives 5/18.9 = 26.5%, exactly the rate stated in Sec. VI-D.
So if you ever run laundry, that 26.5% is a free unit test of this analyzer.

CLI:
    python bench_realtime.py stdout.log --dt 0.01 --sim-time 10
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

STEP_RE = re.compile(r"frame=(\d+)\s+time=(\d+)ms(?:\s+N\(substeps\)=(\d+))?")

# Paper reference points (RTX 4090). dt in ms, runtime in ms per step.
PAPER = {
    "Dough Rolling": {"dt": 10.0, "runtime_ms": 21.7, "rt_rate": 0.46,
                      "note": "Table I at eps_r=5e-2; 59% at eps_r=1e-1"},
    "Box Bagging":   {"dt": 5.0,  "runtime_ms": 45.3, "rt_rate": 5.0 / 45.3},
    "Laundry":       {"dt": 5.0,  "runtime_ms": 18.9, "rt_rate": 0.265},
    "T-shirt Folding": {"dt": 2.0, "runtime_ms": 21.5, "rt_rate": 2.0 / 21.5},
    "Shake":         {"dt": 0.1,  "runtime_ms": 231.1, "rt_rate": 0.1 / 231.1},
}


def parse_step_times(stdout: str) -> tuple[list[int], list[int]]:
    """Return (per-step ms, per-step substep counts) in frame order."""
    frames: dict[int, tuple[int, int]] = {}
    for m in STEP_RE.finditer(stdout):
        frame = int(m.group(1))
        ms = int(m.group(2))
        n = int(m.group(3)) if m.group(3) else -1
        frames[frame] = (ms, n)  # last write wins if a frame repeats
    ordered = [frames[k] for k in sorted(frames)]
    return [t for t, _ in ordered], [n for _, n in ordered]


def analyze(
    step_ms: list[int],
    dt_s: float,
    simulation_time_s: float | None = None,
    drop_warmup: int = 3,
    reference: str | None = "Dough Rolling",
) -> dict:
    """Compute per-step runtime stats and real-time rate, paper-style."""
    if len(step_ms) < 2:
        return {"error": f"only {len(step_ms)} step-timing lines found; "
                         "was the run unthrottled (--realtime_rate=0) and "
                         "did it actually start stepping?"}

    dt_ms = dt_s * 1000.0
    n = len(step_ms)
    steady = step_ms[drop_warmup:] if n > drop_warmup + 1 else step_ms

    def stats(xs: list[int]) -> dict:
        return {
            "mean": round(statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 2),
            "p10": round(statistics.quantiles(xs, n=10)[0], 2) if len(xs) >= 10 else None,
            "p90": round(statistics.quantiles(xs, n=10)[8], 2) if len(xs) >= 10 else None,
            "min": min(xs),
            "max": max(xs),
        }

    all_stats = stats(step_ms)
    steady_stats = stats(steady)

    # Real-time rate, computed several internally-consistent ways.
    rt_from_mean_all = dt_ms / all_stats["mean"] if all_stats["mean"] else None
    rt_from_median = dt_ms / all_stats["median"] if all_stats["median"] else None
    rt_from_mean_steady = dt_ms / steady_stats["mean"] if steady_stats["mean"] else None
    total_wall_ms = sum(step_ms)
    rt_from_total = (n * dt_ms) / total_wall_ms if total_wall_ms else None

    result = {
        "n_steps_logged": n,
        "dt_ms": dt_ms,
        "runtime_ms_per_step": {
            "all_steps": all_stats,
            "steady_state": {**steady_stats, "dropped_warmup": drop_warmup},
        },
        "wall_clock": {
            "total_compute_ms": total_wall_ms,
            "total_compute_s": round(total_wall_ms / 1000.0, 3),
            "simulated_time_s": round(n * dt_s, 3),
        },
        "real_time_rate": {
            "from_mean_all": _r(rt_from_mean_all),       # closest to Table I method
            "from_median": _r(rt_from_median),           # robust to warmup
            "from_mean_steady": _r(rt_from_mean_steady),  # excludes alloc/first-sort
            "from_total": _r(rt_from_total),
        },
    }

    if simulation_time_s is not None:
        expected = round(simulation_time_s / dt_s)
        result["coverage"] = {
            "expected_steps": expected,
            "logged_steps": n,
            "complete": abs(expected - n) <= max(2, 0.02 * expected),
        }

    if reference and reference in PAPER:
        ref = PAPER[reference]
        result["paper_comparison"] = {
            "reference": reference,
            "paper_runtime_ms": ref["runtime_ms"],
            "ours_runtime_ms_mean": all_stats["mean"],
            "runtime_ratio_ours_over_paper": round(all_stats["mean"] / ref["runtime_ms"], 2),
            "paper_rt_rate": ref["rt_rate"],
            "ours_rt_rate_mean_all": _r(rt_from_mean_all),
            "note": ref.get("note", ""),
        }
    return result


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def analyze_log_file(path: Path, dt_s: float, simulation_time_s: float | None,
                     reference: str | None = "Dough Rolling") -> dict:
    step_ms, _ = parse_step_times(path.read_text())
    return analyze(step_ms, dt_s, simulation_time_s, reference=reference)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="stdout log containing frame=... time=...ms lines")
    ap.add_argument("--dt", type=float, default=0.01, help="time step dt in SECONDS (dough: 0.01)")
    ap.add_argument("--sim-time", type=float, default=None, help="total simulated time [s]")
    ap.add_argument("--reference", default="Dough Rolling",
                    help="paper row to compare against, or 'none'")
    args = ap.parse_args()
    ref = None if args.reference.lower() == "none" else args.reference
    out = analyze_log_file(args.log, args.dt, args.sim_time, reference=ref)
    print(json.dumps(out, indent=2))