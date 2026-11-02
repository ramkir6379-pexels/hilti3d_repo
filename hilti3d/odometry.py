"""Stage 2 - LiDAR odometry (the SLAM front-end) with KISS-ICP.

For every scan: deskew it (undo motion distortion using per-point time), register it
against a running local map (point-to-point ICP), and store the resulting pose.

Writes (under <work>/):
  poses.npy            (N,4,4) sensor pose of each scan in the first-scan frame
  trajectory_tum.txt   t tx ty tz qx qy qz qw   (mid-scan absolute time)
  deskewed/scan_*.npz  xyz(float32) intensity(uint8) in the sensor frame, motion-corrected
"""
from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log, mem


def _read_index(work: Path):
    with open(work / "scans_index.csv", newline="") as f:
        return list(csv.DictReader(f))


def run(cfg: Config, work: Path) -> dict:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    from scipy.spatial.transform import Rotation as R

    rows = _read_index(work)
    if not rows:
        raise RuntimeError("scans_index.csv is empty - run the 'extract' stage first.")

    kcfg = KISSConfig()
    kcfg.data.max_range = float(cfg.max_range)
    kcfg.data.min_range = float(cfg.min_range)
    kcfg.data.deskew = bool(cfg.deskew)
    kcfg.mapping.voxel_size = float(cfg.icp_voxel)
    odom = KissICP(kcfg)
    log.info("KISS-ICP ready | max_range=%.0f min_range=%.2f voxel=%.2f deskew=%s",
             cfg.max_range, cfg.min_range, cfg.icp_voxel, cfg.deskew)

    out_dir = work / "deskewed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("scan_*.npz"):
        old.unlink()

    poses: list[np.ndarray] = []
    times: list[float] = []
    n_fail = 0
    n_len_mismatch = 0
    last_pose = np.eye(4)

    for k, row in enumerate(rows):
        i = int(row["idx"])
        try:
            d = np.load(work / "scans" / f"scan_{i:06d}.npz")
            xyz = d["xyz"].astype(np.float64)
            inten = d["intensity"]
            t = d["t"].astype(np.float64)
            span = float(t.max() - t.min())
            tn = (t - t.min()) / span if span > 1e-9 else np.zeros_like(t)

            frame, _source = odom.register_frame(xyz, tn)
            frame = np.asarray(frame)
            pose = np.array(odom.last_pose, dtype=np.float64)
            if len(frame) != len(inten):
                # KISS-ICP dropped a few points; recover intensities by nearest raw point
                # (deskew moves points by only a few cm, so the match is unambiguous).
                n_len_mismatch += 1
                from scipy.spatial import cKDTree
                _, nn = cKDTree(xyz).query(frame, k=1)
                inten = inten[nn]
            np.savez(out_dir / f"scan_{i:06d}.npz",
                     xyz=frame.astype(np.float32), intensity=inten)
            last_pose = pose
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            pose = last_pose.copy()
            log.error("Odometry failed on scan %d (%s) - keeping previous pose.", i, e)
            log.debug(traceback.format_exc())
        poses.append(pose)
        times.append(0.5 * (float(row["t_min"]) + float(row["t_max"])))
        if (k + 1) % 100 == 0 or k == len(rows) - 1:
            step = np.linalg.norm(poses[-1][:3, 3] - poses[max(0, k - 99)][:3, 3])
            log.info("  scan %d/%d | travelled (last 100) %.2f m | %s",
                     k + 1, len(rows), step, mem())

    P = np.asarray(poses)
    np.save(work / "poses.npy", P)
    q = R.from_matrix(P[:, :3, :3]).as_quat()  # x y z w
    tum = np.column_stack([np.asarray(times), P[:, :3, 3], q])
    np.savetxt(work / "trajectory_tum.txt", tum, fmt="%.9f")

    path_len = float(np.sum(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)))
    stats = {"scans": len(P), "failed": n_fail, "path_length_m": round(path_len, 2),
             "intensity_len_mismatch": n_len_mismatch}
    if n_len_mismatch:
        log.info("KISS-ICP dropped a few points in %d scans; intensities re-matched by nearest point.",
                 n_len_mismatch)
    if n_fail > 0.05 * len(P):
        log.warning("More than 5%% of scans failed (%d/%d) - the map will be unreliable.",
                    n_fail, len(P))
    log.info("Odometry done: %s", stats)
    return stats
