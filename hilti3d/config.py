from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # ---- input -------------------------------------------------------------
    bag: str = ""
    lidar_topic: str = "/hesai/pandar"
    imu_topic: str = "/alphasense/imu"
    t_start: float = 0.0           # seconds after bag start
    t_end: Optional[float] = None  # seconds after bag start (None = whole bag)

    # ---- odometry (KISS-ICP, LiDAR only) ----------------------------------
    min_range: float = 0.5         # drop points closer than this (rig/operator)
    max_range: float = 60.0        # drop points farther than this (far returns are noisy)
    icp_voxel: float = 0.3         # voxel size of KISS-ICP's internal map (m); tune per scene
    deskew: bool = True            # correct motion distortion using per-point time

    # ---- map merge ---------------------------------------------------------
    map_voxel: float = 0.05        # voxel size of the merged map (m)
    map_max_range: float = 30.0    # points farther than this are not added to the map
    map_chunk_scans: int = 25      # scans merged per chunk (memory control)
    map_outlier_nb: int = 4        # radius-outlier filter: min neighbours
    map_outlier_radius_mult: float = 3.0  # radius = mult * map_voxel

    # ---- mesh (Poisson) ----------------------------------------------------
    mesh_voxel: float = 0.10       # cloud is thinned to this before Poisson
    normal_radius_mult: float = 3.0
    normal_max_nn: int = 30
    poisson_depth: int = 9
    density_trim_quantile: float = 0.08   # drop least-supported mesh vertices
    mesh_web_triangles: int = 500_000

    # ---- export ------------------------------------------------------------
    preview_max_points: int = 1_500_000

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
            cfg.update(data)
        return cfg

    def update(self, data: dict) -> None:
        valid = {f.name for f in fields(self)}
        for k, v in data.items():
            if k not in valid:
                raise KeyError(f"Unknown config key '{k}'. Valid keys: {sorted(valid)}")
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return asdict(self)
