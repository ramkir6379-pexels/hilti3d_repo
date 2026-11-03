"""Stage 7 - accuracy check against the Hilti ground-truth control points.

The ground truth (exp21) is a handful of surveyed points touched by the rig's measuring
tip, in the surveyors' frame - not our map frame. So we use frame-independent checks:
  * distances between every pair of points (GT vs estimate)
  * vertical offset of each point relative to the first one
  * a rigid best-fit residual (rotation + translation, no scale)
If poses_raw.npy exists (tilt stage ran) both the raw and the corrected trajectory are scored.

Writes: accuracy_report.txt, accuracy.json
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log


def _tip_positions(P: np.ndarray, t_scan: np.ndarray, t_gt: np.ndarray, tip: np.ndarray):
    from scipy.spatial.transform import Rotation as R, Slerp

    ok = (t_gt >= t_scan[0]) & (t_gt <= t_scan[-1])
    slerp = Slerp(t_scan, R.from_matrix(P[:, :3, :3]))
    pos = np.full((len(t_gt), 3), np.nan)
    for i in np.where(ok)[0]:
        t = t_gt[i]
        p = np.array([np.interp(t, t_scan, P[:, :3, 3][:, c]) for c in range(3)])
        pos[i] = p + slerp([t]).as_matrix()[0] @ tip
    return pos, ok


def _umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, my - R @ mx


def score(label: str, E: np.ndarray, G: np.ndarray, lines: list[str]) -> dict:
    n = len(E)
    lines.append(f"--- {label} ---")
    errs = []
    for i, j in itertools.combinations(range(n), 2):
        dg, de = np.linalg.norm(G[i] - G[j]), np.linalg.norm(E[i] - E[j])
        errs.append(de - dg)
        lines.append(f"  P{i+1}-P{j+1}: GT {dg:7.2f} m  est {de:7.2f} m  error {de-dg:+6.2f} m ({100*(de-dg)/max(dg,1e-9):+5.1f}%)")
    out = {"mean_abs_distance_error_m": float(np.mean(np.abs(errs))) if errs else None,
           "max_abs_distance_error_m": float(np.max(np.abs(errs))) if errs else None}
    dz = (E[:, 2] - E[0, 2]) - (G[:, 2] - G[0, 2])
    lines.append("  vertical error relative to P1 (est - GT), m: " + ", ".join(f"{v:+.2f}" for v in dz))
    out["vertical_error_vs_P1_m"] = dz.round(3).tolist()
    if n >= 3:
        R, t = _umeyama(E, G)
        res = np.linalg.norm(E @ R.T + t - G, axis=1)
        out["rigid_fit_rmse_m"] = float(np.sqrt((res ** 2).mean()))
        out["rigid_fit_residuals_m"] = res.round(3).tolist()
        lines.append(f"  rigid-fit residuals (m): {np.round(res, 2).tolist()}  RMSE {out['rigid_fit_rmse_m']:.2f}")
    if out["mean_abs_distance_error_m"] is not None:
        lines.append(f"  mean |distance error| {out['mean_abs_distance_error_m']:.2f} m, max {out['max_abs_distance_error_m']:.2f} m")
    return out


def run(cfg: Config, work: Path) -> dict:
    if not cfg.gt_file:
        log.info("No gt_file configured -> evaluation skipped (set --set gt_file=PATH).")
        return {"skipped": True}
    gt_path = Path(cfg.gt_file)
    if not gt_path.exists():
        raise FileNotFoundError(f"gt_file not found: {gt_path}")
    gt = np.atleast_2d(np.loadtxt(gt_path, comments="#"))
    t_gt, G = gt[:, 0], gt[:, 1:4]
    with open(work / "scans_index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t_scan = np.array([0.5 * (float(r["t_min"]) + float(r["t_max"])) for r in rows])
    tip = np.asarray(cfg.tip_in_lidar, dtype=float)

    variants = []
    if (work / "poses_raw.npy").exists():
        variants.append(("RAW LiDAR odometry (before tilt correction)", np.load(work / "poses_raw.npy")))
    variants.append(("FINAL poses (poses.npy)", np.load(work / "poses.npy")))

    lines: list[str] = [f"Ground truth: {gt_path.name} ({len(gt)} points); tip offset in LiDAR frame {tip.tolist()}"]
    result = {}
    for label, P in variants:
        E, ok = _tip_positions(P, t_scan, t_gt, tip)
        if ok.sum() < 2:
            lines.append(f"{label}: fewer than 2 ground-truth points fall inside the processed time window.")
            continue
        if (~ok).any():
            lines.append(f"(points outside processed window skipped: {[int(i)+1 for i in np.where(~ok)[0]]})")
        result[label] = score(label, E[ok], G[ok], lines)
    text = "\n".join(lines)
    (work / "accuracy_report.txt").write_text(text)
    (work / "accuracy.json").write_text(json.dumps(result, indent=2))
    for ln in lines:
        log.info(ln)
    return result
