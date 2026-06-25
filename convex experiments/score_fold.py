"""
Fold-success scoring for the MPM t-shirt folding eval (arXiv:2503.05046).

Consumes per-frame cloth particle dumps produced by the simulation's
--write_files flag and produces lbm_eval-style outputs: a binary success
flag plus scalar diagnostics, so rollouts can be aggregated into success
rates across seeds / parameter sweeps.

Scoring logic
-------------
A successful two-stage fold should, relative to the settled flat shirt:
  * shrink the XY footprint (convex-hull area) to roughly a quarter
    (two halvings; threshold defaults to 0.35 to allow slop), and
  * increase cloth stack thickness (up to ~8 layers per the paper),
    measured robustly as the z p95-p5 spread.

We take the reference frame after the shirt settles on the table (default
t = 1.0 s; free-fall settles by t = 0.5 s per the paper) and evaluate on the
mean of the last `eval_frac` of frames. Note: this assumes the rollout ends
folded (true for `mpm_tshirt_folding`). For `dual_arm_flipping`, which
unfolds at the end, score with --eval-time set inside the folded window
instead (folding completes ~t = 4.5 s, unfold begins ~t = 8.3 s).

Frame formats auto-detected: .obj (vertex lines), ascii .ply, .npy (Nx3),
and whitespace xyz text (.xyz/.txt/.csv). If the fork dumps something else
(e.g. .bgeo), convert or extend `load_points`.

CLI:
    python score_fold.py ./results/<run_id> --sim-time 25.0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

POINT_EXTS = (".obj", ".ply", ".npy", ".xyz", ".txt", ".csv")
_NUM_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_points(path: Path) -> np.ndarray | None:
    """Load one frame of particle positions as an (N, 3) float array."""
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
            return arr.reshape(-1, 3).astype(float) if arr.size else None
        if ext == ".obj":
            pts = [
                tuple(map(float, line.split()[1:4]))
                for line in path.read_text().splitlines()
                if line.startswith("v ")
            ]
            return np.asarray(pts) if pts else None
        if ext == ".ply":
            lines = path.read_text().splitlines()
            try:
                n = next(
                    int(l.split()[-1]) for l in lines if l.startswith("element vertex")
                )
                start = next(i for i, l in enumerate(lines) if l.strip() == "end_header") + 1
            except StopIteration:
                return None  # binary ply or malformed; extend if needed
            pts = [tuple(map(float, l.split()[:3])) for l in lines[start : start + n]]
            return np.asarray(pts)
        # whitespace/comma xyz text
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        return arr.reshape(-1, 3).astype(float) if arr.size else None
    except Exception:
        return None


def discover_frames(run_dir: Path) -> list[Path]:
    """Find the per-frame dump series: the largest group of same-extension,
    numbered files, sorted by their trailing frame index."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in POINT_EXTS and _NUM_RE.search(f.stem):
            groups.setdefault(f.suffix.lower(), []).append(f)
    if not groups:
        return []
    series = max(groups.values(), key=len)

    def frame_idx(p: Path) -> int:
        return int(_NUM_RE.findall(p.stem)[-1])

    return sorted(series, key=frame_idx)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def hull_area_xy(points: np.ndarray) -> float:
    """Convex-hull area of the XY projection (monotone chain; no scipy dep)."""
    uniq = np.unique(points[:, :2], axis=0)
    if len(uniq) < 3:
        return 0.0
    pts = sorted(map(tuple, uniq.tolist()))  # plain floats: fast scalar math

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out: list[tuple] = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(hull, hull[1:] + hull[:1]):
        area += x0 * y1 - x1 * y0
    return 0.5 * abs(area)


def frame_metrics(points: np.ndarray) -> dict:
    z = points[:, 2]
    return {
        "n_particles": int(len(points)),
        "hull_area_xy": float(hull_area_xy(points)),
        "bbox_area_xy": float(np.ptp(points[:, 0]) * np.ptp(points[:, 1])),
        "thickness_z": float(np.percentile(z, 95) - np.percentile(z, 5)),
        "z_mean": float(z.mean()),
        "centroid_xy": [float(points[:, 0].mean()), float(points[:, 1].mean())],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class FoldScore:
    success: bool
    area_ratio: float          # final hull area / reference hull area
    thickness_ratio: float     # final z-spread / reference z-spread
    est_layers: float          # alias of thickness_ratio (flat shirt ~= 1 layer)
    area_drift: float | None   # relative area change over the tail window
    ref_frame: int
    eval_frames: list[int]
    n_frames: int
    notes: str = ""


def score_rollout(
    frames: list[Path],
    sim_time: float,
    ref_time: float = 1.0,
    eval_time: float | None = None,
    eval_frac: float = 0.1,
    area_thresh: float = 0.42,
    thickness_thresh: float = 2.0,
    drift_thresh: float = 0.15,
    max_frames: int = 200,
) -> tuple[FoldScore, list[dict]]:
    """Score a rollout, loading at most ~max_frames frames (lazy + strided).

    Thresholds calibrated against real fork rollouts (see repo discussion):
    a settled double fold of the tshirt.obj mesh plateaus at area ~0.35,
    a single fold at ~0.46-0.48, so 0.42 splits them with margin on both
    sides. drift_thresh rejects rollouts still unfurling at the end: settled
    cloth relaxes by a few percent with decaying increments; a slipping fold
    keeps growing.
    """
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} frames found; need >= 3 to score")

    n = len(frames)
    dt = sim_time / max(n - 1, 1)  # assumes uniform dump cadence

    # Strided sample for the diagnostic curve, plus the exact indices the
    # score depends on (reference frame and evaluation tail).
    stride = max(1, n // max_frames)
    sample = set(range(0, n, stride)) | {n - 1}
    ref_target = min(n - 1, max(0, round(ref_time / dt)))
    sample.add(ref_target)
    if eval_time is not None:
        eval_targets = [min(n - 1, max(0, round(eval_time / dt)))]
    else:
        k = max(1, int(n * eval_frac) // stride or 1)
        eval_targets = sorted(sample)[-k:]
    sample |= set(eval_targets)
    sample_idx = sorted(sample)

    per_frame: list[dict] = []
    pts: dict[int, np.ndarray] = {}
    for i in sample_idx:
        p = load_points(frames[i])
        if p is not None and len(p) >= 3:
            pts[i] = p
            per_frame.append({"frame": i, "t": round(i * dt, 4),
                              "file": frames[i].name, **frame_metrics(p)})
        else:
            per_frame.append({"frame": i, "t": round(i * dt, 4),
                              "file": frames[i].name, "parse_error": True})

    if not pts:
        raise ValueError("no parsable frames; check the dump format")

    valid = sorted(pts)
    ref_i = min(valid, key=lambda i: abs(i - ref_target))
    eval_is = [min(valid, key=lambda i: abs(i - t)) for t in eval_targets]
    eval_is = sorted(set(eval_is))

    ref = frame_metrics(pts[ref_i])
    finals = [frame_metrics(pts[i]) for i in eval_is]
    f_area = float(np.mean([m["hull_area_xy"] for m in finals]))
    f_thick = float(np.mean([m["thickness_z"] for m in finals]))

    area_ratio = f_area / ref["hull_area_xy"] if ref["hull_area_xy"] > 0 else np.inf
    thick_ratio = f_thick / ref["thickness_z"] if ref["thickness_z"] > 0 else np.inf

    # Plateau stability: relative area change across the last ~20% of the
    # rollout. Settled cloth shows small, decaying relaxation; a slipping or
    # unfurling fold keeps growing. Undefined for single-frame eval_time.
    tail = [i for i in valid if i >= 0.8 * (n - 1)]
    if eval_time is None and len(tail) >= 2:
        a0 = hull_area_xy(pts[tail[0]])
        a1 = hull_area_xy(pts[tail[-1]])
        area_drift = round((a1 - a0) / a0, 4) if a0 > 0 else None
    else:
        area_drift = None

    score = FoldScore(
        success=bool(
            area_ratio < area_thresh
            and thick_ratio > thickness_thresh
            and (area_drift is None or area_drift < drift_thresh)
        ),
        area_ratio=round(area_ratio, 4),
        thickness_ratio=round(thick_ratio, 4),
        est_layers=round(thick_ratio, 2),
        area_drift=area_drift,
        ref_frame=ref_i,
        eval_frames=eval_is,
        n_frames=n,
        notes=(
            f"dt~{dt:.4f}s/frame (assumed uniform); ref t~{ref_i * dt:.2f}s; "
            f"loaded {len(sample_idx)}/{n} frames (stride {stride}); "
            f"thresholds: area<{area_thresh}, thickness>{thickness_thresh}, "
            f"drift<{drift_thresh}"
        ),
    )
    return score, per_frame


def score_run_dir(run_dir: Path, sim_time: float, **kw) -> dict:
    frames = discover_frames(run_dir)
    if not frames:
        return {
            "success": None,
            "error": (
                f"no per-frame particle dumps found in {run_dir} "
                f"(looked for numbered {', '.join(POINT_EXTS)} files). "
                "Was the run launched with --write-files?"
            ),
        }
    score, per_frame = score_rollout(frames, sim_time=sim_time, **kw)
    result = {
        "run_dir": str(run_dir),
        "format": frames[0].suffix,
        **asdict(score),
    }
    (run_dir / "score.json").write_text(
        json.dumps({**result, "per_frame": per_frame}, indent=2)
    )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--sim-time", type=float, default=25.0)
    ap.add_argument("--ref-time", type=float, default=1.0)
    ap.add_argument("--eval-time", type=float, default=None,
                    help="score at this sim time instead of the last frames "
                         "(use for rollouts that unfold at the end)")
    ap.add_argument("--area-thresh", type=float, default=0.35)
    ap.add_argument("--thickness-thresh", type=float, default=2.0)
    args = ap.parse_args()
    out = score_run_dir(
        args.run_dir,
        sim_time=args.sim_time,
        ref_time=args.ref_time,
        eval_time=args.eval_time,
        area_thresh=args.area_thresh,
        thickness_thresh=args.thickness_thresh,
    )
    print(json.dumps(out, indent=2))