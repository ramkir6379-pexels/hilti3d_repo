"""Stage 3 - gravity (tilt) correction of the LiDAR-only trajectory using the IMU.

Problem: LiDAR odometry slowly accumulates roll/pitch error, so a flat courtyard
ends up as a ramp (we measured ~6 m of height drift on exp21).
Idea: an accelerometer measures gravity, so its time-averaged direction is the true
"up". We compare it with the "up" implied by the SLAM orientation and rotate the
poses to remove the difference (around horizontal axes only, so heading is kept).

Steps
 1. Average the accelerometer, expressed in the WORLD frame via each scan's rotation,
    over a sliding window (default 6 s; walking accelerations average out).
 2. Level the world with the first seconds (rig assumed ~still) -> world z = up.
 3. For every scan: minimal rotation Q_k taking the measured up to +z; R'_k = Q_k R_gw R_k.
 4. Re-integrate translation from the ORIGINAL sensor-frame steps with corrected rotations.

Writes: poses_raw.npy (backup of the uncorrected poses), poses.npy (corrected, gravity-aligned),
        tilt_stats.json.  Safe to re-run (always starts from poses_raw.npy).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log

EZ = np.array([0.0, 0.0, 1.0])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # opposite vectors: rotate 180 deg about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = _skew(v)
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def correct_poses(P: np.ndarray, t_scan: np.ndarray, imu: np.ndarray, R_li: np.ndarray,
                  window_s: float, init_s: float):
    """Pure function (unit-testable): returns (corrected poses, per-scan correction angles in deg,
    angle between initial measured up and world z in deg)."""
    n = len(P)
    t_imu = imu[:, 0]
    a_lid = (R_li @ imu[:, 4:7].T).T                      # accel in LiDAR frame
    # rotation of the scan nearest to each IMU sample
    idx = np.clip(np.searchsorted(t_scan, t_imu), 0, n - 1)
    left = np.clip(idx - 1, 0, n - 1)
    idx = np.where(np.abs(t_scan[left] - t_imu) < np.abs(t_scan[idx] - t_imu), left, idx)
    a_world = np.einsum("nij,nj->ni", P[idx, :3, :3], a_lid)  # accel in (uncorrected) world frame
    csum = np.vstack([np.zeros(3), np.cumsum(a_world, axis=0)])

    def mean_up(t0, t1):
        i0, i1 = np.searchsorted(t_imu, t0), np.searchsorted(t_imu, t1)
        if i1 - i0 < 5:
            return None
        v = (csum[i1] - csum[i0]) / (i1 - i0)
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-6 else None

    u0 = mean_up(t_scan[0], t_scan[0] + init_s)
    if u0 is None:
        raise RuntimeError("Not enough IMU samples at the start to level the world.")
    R_gw = rot_between(u0, EZ)
    init_tilt = angle_deg(R_gw)

    Rn = np.empty((n, 3, 3))
    corr = np.zeros(n)
    last_Q = np.eye(3)
    for k in range(n):
        u = mean_up(t_scan[k] - window_s / 2, t_scan[k] + window_s / 2)
        if u is None:                       # no IMU coverage: keep the previous correction
            Q = last_Q
        else:
            Q = rot_between(R_gw @ u, EZ)
            last_Q = Q
        corr[k] = angle_deg(Q)
        Rn[k] = Q @ R_gw @ P[k, :3, :3]

    # re-integrate translation using the original sensor-frame steps
    pn = np.zeros((n, 3))
    pn[0] = R_gw @ P[0, :3, 3]
    for k in range(n - 1):
        d_sensor = P[k, :3, :3].T @ (P[k + 1, :3, 3] - P[k, :3, 3])
        pn[k + 1] = pn[k] + Rn[k] @ d_sensor
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Rn
    out[:, :3, 3] = pn
    return out, corr, init_tilt


def run(cfg: Config, work: Path) -> dict:
    if not cfg.tilt_correction:
        log.info("tilt_correction=false -> skipping.")
        return {"skipped": True}
    imu_path = work / "imu.npy"
    if not imu_path.exists():
        log.warning("No imu.npy found - cannot correct tilt. Poses left unchanged.")
        return {"skipped": True, "reason": "no imu"}

    raw = work / "poses_raw.npy"
    if not raw.exists():
        shutil.copy2(work / "poses.npy", raw)
    P = np.load(raw)
    imu = np.load(imu_path)
    with open(work / "scans_index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t_scan = np.array([0.5 * (float(r["t_min"]) + float(r["t_max"])) for r in rows])
    if len(t_scan) != len(P):
        raise RuntimeError(f"{len(t_scan)} scans vs {len(P)} poses - rerun 'odometry'.")
    if imu[-1, 0] < t_scan[0] or imu[0, 0] > t_scan[-1]:
        raise RuntimeError("IMU and LiDAR time ranges do not overlap (clock mismatch?).")

    R_li = np.asarray(cfg.imu_to_lidar_R, dtype=float).reshape(3, 3)
    new, corr, init_tilt = correct_poses(P, t_scan, imu, R_li, cfg.tilt_window_s, cfg.tilt_init_s)
    if init_tilt > 45:
        log.warning("Initial 'up' is %.0f deg from the LiDAR z-axis - check imu_to_lidar_R!", init_tilt)

    np.save(work / "poses.npy", new)
    z0, z1 = P[:, 2, 3], new[:, 2, 3]
    stats = {"initial_world_tilt_deg": round(init_tilt, 2),
             "mean_correction_deg": round(float(corr.mean()), 2),
             "max_correction_deg": round(float(corr.max()), 2),
             "final_correction_deg": round(float(corr[-1]), 2),
             "z_range_before_m": [round(float(z0.min()), 2), round(float(z0.max()), 2)],
             "z_range_after_m": [round(float(z1.min()), 2), round(float(z1.max()), 2)],
             "final_z_before_m": round(float(z0[-1]), 2), "final_z_after_m": round(float(z1[-1]), 2)}
    (work / "tilt_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Tilt correction done: %s", stats)
    return stats
