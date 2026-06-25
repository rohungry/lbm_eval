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

# UpdateContact() in cuda_mpm_solver.cu prints this per substep that HAS
# contacts (it returns early when n_contacts==0, so its absence means a
# contact-free substep). Lets us split steps by *measured* contact, not by
# guessing from runtime.
CONTACT_RE = re.compile(
    r"Iteration count :(\d+),.*?n_contacts (\d+), time ([\d.]+)ms"
)

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


def parse_steps_with_contacts(stdout: str) -> list[dict]:
    """Walk the log, grouping each step's per-substep contact-solve lines under
    the `frame=...` line that follows them (the driver prints the frame line
    after its substep loop). Returns one dict per step."""
    steps: list[dict] = []
    pending: list[tuple[int, int, float]] = []  # (iters, n_contacts, solve_ms)
    for line in stdout.splitlines():
        cm = CONTACT_RE.search(line)
        if cm:
            pending.append((int(cm.group(1)), int(cm.group(2)), float(cm.group(3))))
            continue
        sm = STEP_RE.search(line)
        if sm:
            steps.append({
                "frame": int(sm.group(1)),
                "step_ms": int(sm.group(2)),
                "n_substeps": int(sm.group(3)) if sm.group(3) else -1,
                "peak_contacts": max((c for _, c, _ in pending), default=0),
                "n_contact_substeps": len(pending),
                "total_iters": sum(i for i, _, _ in pending),
                "contact_solve_ms": round(sum(t for _, _, t in pending), 3),
            })
            pending = []
    return steps


def text_histogram(values: list[float], bins: int = 12, width: int = 44) -> str:
    """Linear-bucket ASCII histogram — chosen to make bimodality visible."""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    counts = [0] * bins
    for v in values:
        k = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[k] += 1
    mx = max(counts) or 1
    out = []
    for i in range(bins):
        e0 = lo + (hi - lo) * i / bins
        e1 = lo + (hi - lo) * (i + 1) / bins
        bar = "#" * round(counts[i] / mx * width)
        out.append(f"{e0:6.1f}-{e1:6.1f} ms | {bar} {counts[i]}")
    return "\n".join(out)


def _regime_stats(step_ms: list[int], dt_ms: float, total_wall_ms: int,
                  label: str) -> dict:
    if not step_ms:
        return {"label": label, "n_steps": 0}
    s = sum(step_ms)
    mean = statistics.mean(step_ms)
    return {
        "label": label,
        "n_steps": len(step_ms),
        "mean_ms": round(mean, 2),
        "median_ms": round(statistics.median(step_ms), 2),
        "wall_ms": s,
        "frac_of_wall": round(s / total_wall_ms, 3) if total_wall_ms else None,
        # Illustrative: the rate you'd see if the WHOLE run looked like this
        # regime. Not the run's actual rate -- that's the blended mean.
        "rate_if_uniform": _r(dt_ms / mean) if mean else None,
    }


def contact_analysis(stdout: str, dt_s: float) -> dict | None:
    """Split steps into contact / contact-free regimes using the measured
    n_contacts from UpdateContact, and summarize each. Returns None if the
    log has no contact-solve lines (e.g. logging disabled)."""
    steps = parse_steps_with_contacts(stdout)
    if not steps or not any(s["peak_contacts"] > 0 for s in steps):
        return None

    dt_ms = dt_s * 1000.0
    total_wall_ms = sum(s["step_ms"] for s in steps)
    contact_steps = [s for s in steps if s["peak_contacts"] > 0]
    free_steps = [s for s in steps if s["peak_contacts"] == 0]

    heavy = _regime_stats([s["step_ms"] for s in contact_steps], dt_ms,
                          total_wall_ms, "in-contact")
    light = _regime_stats([s["step_ms"] for s in free_steps], dt_ms,
                          total_wall_ms, "contact-free")

    iters = [s["total_iters"] for s in contact_steps if s["total_iters"] > 0]
    peaks = [s["peak_contacts"] for s in contact_steps]
    return {
        "regimes": {"in_contact": heavy, "contact_free": light},
        "contact_steps_fraction": round(len(contact_steps) / len(steps), 3),
        "peak_contacts": {
            "max": max(peaks), "mean_when_in_contact": round(statistics.mean(peaks), 1),
        },
        "newton_iters_per_step_when_in_contact": {
            "mean": round(statistics.mean(iters), 1) if iters else None,
            "max": max(iters) if iters else None,
        },
        "interpretation": (
            f"{light['n_steps']} contact-free steps avg {light['mean_ms']} ms; "
            f"{heavy['n_steps']} in-contact steps avg {heavy['mean_ms']} ms and "
            f"consume {heavy['frac_of_wall']:.0%} of wall time. The blended mean "
            "is the run's real-time rate; neither regime's rate is."
        ),
        "histogram_ms_per_step": text_histogram([s["step_ms"] for s in steps]),
    }


def analyze_full(stdout: str, dt_s: float, simulation_time_s: float | None = None,
                 reference: str | None = "Dough Rolling") -> dict:
    """Timing analysis + (when present) contact-regime breakdown."""
    step_ms, _ = parse_step_times(stdout)
    result = analyze(step_ms, dt_s, simulation_time_s, reference=reference)
    ca = contact_analysis(stdout, dt_s)
    if ca is not None:
        result["contact_analysis"] = ca
    return result


def analyze_log_file(path: Path, dt_s: float, simulation_time_s: float | None,
                     reference: str | None = "Dough Rolling") -> dict:
    return analyze_full(path.read_text(), dt_s, simulation_time_s, reference=reference)


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
    ca = out.pop("contact_analysis", None)
    print(json.dumps(out, indent=2))
    if ca:
        print("\n=== Contact-regime breakdown ===")
        print(ca["interpretation"])
        print(f"\nper-step runtime histogram (n={out['n_steps_logged']}):")
        print(ca["histogram_ms_per_step"])