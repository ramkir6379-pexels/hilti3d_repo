"""Stage 1 - read the ROS1 bag ONCE and stream LiDAR scans + IMU to small files.

Writes (under <work>/):
  scans/scan_000000.npz   xyz(float32,N,3) intensity(uint8,N) t(float32,N: seconds from scan start)
  scans_index.csv         idx, bag_time_ns, t_min, t_max (absolute per-point times), n_points
  imu.npy                 [t_sec, wx, wy, wz, ax, ay, az]  (kept for Phase 2 / IMU-aided SLAM)
"""
from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np

from .config import Config
from .logutil import log, mem

PF = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
TIME_NAMES = ("timestamp", "timestamps", "time", "t", "stamps")


def decode_pointcloud2(msg):
    """PointCloud2 -> (xyz float32 Nx3, intensity float32 N, t float64 N | None)."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.count == 1 and f.datatype in PF and f.name:
            names.append(f.name)
            formats.append((">" if msg.is_bigendian else "<") + PF[f.datatype])
            offsets.append(f.offset)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                   "itemsize": msg.point_step})
    n = msg.width * msg.height
    arr = np.frombuffer(np.asarray(msg.data).tobytes(), dtype=dt)[:n]
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError(f"PointCloud2 has no '{need}' field (fields: {names})")
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    if "intensity" in names:
        inten = arr["intensity"].astype(np.float32)
    else:
        inten = np.zeros(len(arr), np.float32)
    tname = next((nm for nm in TIME_NAMES if nm in names), None)
    t = arr[tname].astype(np.float64) if tname else None
    return xyz, inten, t


def run(cfg: Config, work: Path) -> dict:
    from rosbags.highlevel import AnyReader

    bag = Path(cfg.bag)
    if not bag.exists():
        raise FileNotFoundError(f"Bag not found: {bag}")
    scans_dir = work / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)
    for old in scans_dir.glob("scan_*.npz"):  # never mix old and new extractions
        old.unlink()

    index_rows: list[list] = []
    imu_rows: list[list[float]] = []
    n_bad = 0
    n_no_time = 0

    with AnyReader([bag]) as reader:
        t0 = reader.start_time + int(cfg.t_start * 1e9)
        if cfg.t_end is not None:
            t1 = reader.start_time + int(cfg.t_end * 1e9)
        else:
            t1 = reader.end_time + 1
        log.info("Bag duration %.1f s | window %.1f s .. %s",
                 (reader.end_time - reader.start_time) / 1e9, cfg.t_start,
                 "end" if cfg.t_end is None else f"{cfg.t_end:.1f} s")

        topics = {c.topic for c in reader.connections}
        lidar = [c for c in reader.connections if c.topic == cfg.lidar_topic]
        imu = [c for c in reader.connections if c.topic == cfg.imu_topic]
        if not lidar:
            raise ValueError(f"LiDAR topic '{cfg.lidar_topic}' not in bag. Available: {sorted(topics)}")
        if not imu:
            log.warning("IMU topic '%s' not in bag - continuing without IMU.", cfg.imu_topic)
        lidar_topics = {c.topic for c in lidar}

        idx = 0
        for conn, ts, raw in reader.messages(connections=lidar + imu, start=t0, stop=t1):
            if conn.topic in lidar_topics:
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    xyz, inten, t = decode_pointcloud2(msg)
                    ok = np.isfinite(xyz).all(axis=1)
                    r = np.linalg.norm(xyz, axis=1)
                    ok &= (r >= cfg.min_range) & (r <= cfg.max_range)
                    xyz, inten = xyz[ok], inten[ok]
                    if t is None:
                        n_no_time += 1
                        t_abs = np.full(len(xyz), ts * 1e-9)
                    else:
                        t_abs = t[ok]
                    if len(xyz) < 100:
                        raise ValueError(f"only {len(xyz)} valid points")
                    tmin, tmax = float(t_abs.min()), float(t_abs.max())
                    np.savez(scans_dir / f"scan_{idx:06d}.npz",
                             xyz=xyz,
                             intensity=np.clip(inten, 0, 255).astype(np.uint8),
                             t=(t_abs - tmin).astype(np.float32))
                    index_rows.append([idx, ts, f"{tmin:.9f}", f"{tmax:.9f}", len(xyz)])
                    idx += 1
                    if idx % 100 == 0:
                        log.info("  extracted %d scans | %s", idx, mem())
                except Exception as e:  # noqa: BLE001
                    n_bad += 1
                    log.error("Bad LiDAR message at bag time %d: %s", ts, e)
                    log.debug(traceback.format_exc())
            else:
                try:
                    m = reader.deserialize(raw, conn.msgtype)
                    w, a = m.angular_velocity, m.linear_acceleration
                    imu_rows.append([ts * 1e-9, w.x, w.y, w.z, a.x, a.y, a.z])
                except Exception as e:  # noqa: BLE001
                    n_bad += 1
                    log.error("Bad IMU message: %s", e)

    if not index_rows:
        raise RuntimeError("No usable LiDAR scans were extracted (see log above).")

    with open(work / "scans_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "bag_time_ns", "t_min", "t_max", "n_points"])
        w.writerows(index_rows)
    if imu_rows:
        np.save(work / "imu.npy", np.asarray(imu_rows, dtype=np.float64))
    if n_no_time:
        log.warning("%d scans had no per-point time field -> deskewing will be ineffective.", n_no_time)
    stats = {"scans": len(index_rows), "imu_samples": len(imu_rows),
             "bad_messages": n_bad, "scans_without_time": n_no_time}
    log.info("Extract done: %s | %s", stats, mem())
    return stats
