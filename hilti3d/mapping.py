"""Stage 3 - merge all deskewed scans into ONE map using the trajectory.

Memory-safe: scans are transformed chunk by chunk, voxel-thinned, and merged
incrementally, so the ~89M raw points never sit in RAM together.

Writes: <work>/map.ply (binary, gray colour = LiDAR intensity), <work>/map_stats.json
"""
from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log, mem


def _to_pcd(xyz: np.ndarray, inten: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    g = (inten.astype(np.float32) / 255.0)[:, None].repeat(3, axis=1)
    pcd.colors = o3d.utility.Vector3dVector(g.astype(np.float64))
    return pcd


def run(cfg: Config, work: Path) -> dict:
    import open3d as o3d

    poses = np.load(work / "poses.npy")
    with open(work / "scans_index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(poses):
        raise RuntimeError(f"{len(rows)} scans in index but {len(poses)} poses - rerun 'odometry'.")

    acc = None            # running merged cloud
    chunk_pts, chunk_int = [], []
    n_raw = 0
    n_skipped = 0
    since_merge = 0

    def flush_chunk():
        nonlocal acc, chunk_pts, chunk_int, since_merge
        if not chunk_pts:
            return
        pcd = _to_pcd(np.concatenate(chunk_pts), np.concatenate(chunk_int)).voxel_down_sample(cfg.map_voxel)
        chunk_pts, chunk_int = [], []
        acc = pcd if acc is None else acc + pcd
        since_merge += 1
        if since_merge >= 8:   # re-thin overlap between chunks periodically
            acc = acc.voxel_down_sample(cfg.map_voxel)
            since_merge = 0

    for k, row in enumerate(rows):
        i = int(row["idx"])
        try:
            d = np.load(work / "deskewed" / f"scan_{i:06d}.npz")
            xyz = d["xyz"].astype(np.float64)
            inten = d["intensity"]
            r = np.linalg.norm(xyz, axis=1)
            keep = r <= cfg.map_max_range
            xyz, inten = xyz[keep], inten[keep]
            if len(xyz) == 0:
                continue
            T = poses[k]
            world = xyz @ T[:3, :3].T + T[:3, 3]
            chunk_pts.append(world.astype(np.float32))
            chunk_int.append(inten)
            n_raw += len(world)
        except Exception as e:  # noqa: BLE001
            n_skipped += 1
            log.error("Skipping scan %d in map merge: %s", i, e)
            log.debug(traceback.format_exc())
        if (k + 1) % cfg.map_chunk_scans == 0:
            flush_chunk()
        if (k + 1) % 200 == 0:
            n_acc = 0 if acc is None else len(acc.points)
            log.info("  merged %d/%d scans | map %d pts | %s", k + 1, len(rows), n_acc, mem())
    flush_chunk()

    if acc is None or len(acc.points) == 0:
        raise RuntimeError("Map is empty - check earlier stages.")
    acc = acc.voxel_down_sample(cfg.map_voxel)
    n_before = len(acc.points)

    if cfg.map_outlier_nb > 0:
        acc, _ = acc.remove_radius_outlier(nb_points=cfg.map_outlier_nb,
                                           radius=cfg.map_outlier_radius_mult * cfg.map_voxel)
    n_after = len(acc.points)

    # Hesai intensities are mostly low -> map looked near-black. Stretch to the 99th percentile
    # (display only; per-point raw intensity is not needed downstream in Phase 1).
    g = np.asarray(acc.colors)
    p99 = float(np.percentile(g[:, 0], 99)) or 1.0
    g = np.clip(g / p99, 0.0, 1.0) ** 0.7
    acc.colors = o3d.utility.Vector3dVector(g)
    o3d.io.write_point_cloud(str(work / "map.ply"), acc, write_ascii=False)

    bb = acc.get_axis_aligned_bounding_box()
    stats = {"raw_points_used": int(n_raw), "map_points_before_filter": int(n_before),
             "map_points": int(n_after), "voxel_m": cfg.map_voxel,
             "bbox_min": bb.get_min_bound().round(2).tolist(),
             "bbox_max": bb.get_max_bound().round(2).tolist(),
             "skipped_scans": n_skipped}
    (work / "map_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Map done: %s | %s", stats, mem())
    return stats
