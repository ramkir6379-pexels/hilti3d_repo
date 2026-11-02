"""Stage 4 - LiDAR-only surface mesh (Poisson reconstruction).

Key detail: Poisson needs consistently oriented normals. We estimate normals, then
flip each so it points toward the nearest sensor position on the trajectory (a point
was seen from the sensor side of its surface). Heuristic, but good for scans taken
by a moving rig; documented limitation: thin double-sided structures may be imperfect.

Writes: <work>/mesh_full.ply, <work>/mesh_web.ply, <work>/mesh_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log, mem


def run(cfg: Config, work: Path) -> dict:
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.io.read_point_cloud(str(work / "map.ply"))
    if len(pcd.points) == 0:
        raise RuntimeError("map.ply is empty or missing - run the 'map' stage first.")
    log.info("Loaded map: %d points | %s", len(pcd.points), mem())

    pcd = pcd.voxel_down_sample(cfg.mesh_voxel)
    log.info("Thinned to %d points (mesh_voxel=%.2f m)", len(pcd.points), cfg.mesh_voxel)

    radius = cfg.normal_radius_mult * cfg.mesh_voxel
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=cfg.normal_max_nn))

    poses = np.load(work / "poses.npy")
    sensor = poses[:, :3, 3]
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals).copy()
    _, nn = cKDTree(sensor).query(pts, k=1)
    to_sensor = sensor[nn] - pts
    flip = np.einsum("ij,ij->i", nrm, to_sensor) < 0
    nrm[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    log.info("Oriented normals toward sensor (%.1f%% flipped) | %s", 100.0 * flip.mean(), mem())

    log.info("Poisson reconstruction depth=%d ... (this is the slow, RAM-heavy step)", cfg.poisson_depth)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    n_v0, n_t0 = len(mesh.vertices), len(mesh.triangles)
    log.info("Raw mesh: %d vertices, %d triangles | %s", n_v0, n_t0, mem())

    # Poisson invents surface in empty space; cut away vertices with little data support.
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, cfg.density_trim_quantile))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(work / "mesh_full.ply"), mesh)

    web = mesh
    if len(mesh.triangles) > cfg.mesh_web_triangles:
        web = mesh.simplify_quadric_decimation(target_number_of_triangles=int(cfg.mesh_web_triangles))
        web.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(work / "mesh_web.ply"), web)

    stats = {"vertices_full": len(mesh.vertices), "triangles_full": len(mesh.triangles),
             "triangles_web": len(web.triangles), "watertight": bool(mesh.is_watertight()),
             "has_colors": bool(mesh.has_vertex_colors())}
    (work / "mesh_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Mesh done: %s | %s", stats, mem())
    return stats
