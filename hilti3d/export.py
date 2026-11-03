"""Stage 5 - build a small static web viewer bundle (Three.js) in <work>/viewer/.

  viewer/index.html        three.js viewer (orbit controls, toggle points / mesh / path)
  viewer/map_preview.ply   thinned point cloud (<= preview_max_points)
  viewer/mesh_web.ply      decimated mesh
  viewer/trajectory.json   sensor path (every 5th pose)
  viewer/meta.json         bounding box etc.

View it:   cd <work>/viewer && python -m http.server 8000   ->  http://localhost:8000
(Browsers block loading local files directly, hence the tiny local server.)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log

INDEX_HTML = (Path(__file__).parent / "viewer_index.html").read_text(encoding="utf-8")


def run(cfg: Config, work: Path) -> dict:
    import open3d as o3d

    out = work / "viewer"
    out.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(work / "map.ply"))
    n = len(pcd.points)
    if n == 0:
        raise RuntimeError("map.ply is empty or missing.")
    if n > cfg.preview_max_points:
        step = int(np.ceil(n / cfg.preview_max_points))
        pcd = pcd.uniform_down_sample(step)
    o3d.io.write_point_cloud(str(out / "map_preview.ply"), pcd, write_ascii=False)

    mesh_src = work / "mesh_web.ply"
    if mesh_src.exists():
        shutil.copy2(mesh_src, out / "mesh_web.ply")
    else:
        log.warning("No mesh_web.ply found - viewer will show points only.")

    poses = np.load(work / "poses.npy")
    path = poses[::5, :3, 3].round(3).tolist()
    (out / "trajectory.json").write_text(json.dumps(path))

    bb = pcd.get_axis_aligned_bounding_box()
    meta = {"bbox_min": bb.get_min_bound().tolist(), "bbox_max": bb.get_max_bound().tolist(),
            "preview_points": len(pcd.points)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    log.info("Viewer bundle ready: %s  (serve with: python -m http.server 8000)", out)
    return {"viewer_dir": str(out), **meta}
