"""Simulate a spinning 32-beam LiDAR (Hesai-like fields) moving through a room with
box obstacles, and write it as a ROS1 bag + ground-truth poses (.npy).

Includes motion distortion: every point is measured at its own time while the sensor
moves, exactly like a real rotating LiDAR, so deskewing is genuinely exercised.

    python tests/make_synthetic_bag.py out.bag gt_poses.npy [duration_s]
"""
import sys

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

ROOM = np.array([[0.0, 0.0, 0.0], [30.0, 20.0, 6.0]])
rng = np.random.default_rng(7)
BOXES = []
for cx, cy, sx, sy, h in [(8, 6, 2, 3, 2.5), (22, 5, 3, 2, 3), (15, 14, 4, 1.5, 2), (6, 15, 2, 2, 4),
                          (25, 15, 2.5, 3, 1.5), (12, 3, 1, 1, 5), (18, 9, 1.5, 1.5, 2.2), (3, 8, 1, 4, 1.8)]:
    BOXES.append(np.array([[cx - sx / 2, cy - sy / 2, 0.0], [cx + sx / 2, cy + sy / 2, h]]))

T0 = 1650529602.0
RATE = 10.0


def pose_at(t):
    """Analytic sensor pose (x, y, z, yaw) at time t seconds."""
    w = 2 * np.pi / 60.0
    x = 15 + 9 * np.cos(w * t + 0.3)
    y = 10 + 6 * np.sin(w * t + 0.3)
    vx = -9 * w * np.sin(w * t + 0.3)
    vy = 6 * w * np.cos(w * t + 0.3)
    yaw = np.arctan2(vy, vx) + 0.4 * np.sin(0.3 * t)
    return x, y, 1.5 + 0.0 * t, yaw


def raycast(o, d):
    """o: (N,3) origins, d: (N,3) unit directions -> (range N, surface id N)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # inside the room: exit distance
        tmax = np.where(d > 0, (ROOM[1] - o) / d, (ROOM[0] - o) / d)
        axis = np.argmin(tmax, axis=1)
        best = tmax[np.arange(len(o)), axis]
        sid = axis.copy() + 1
        for k, b in enumerate(BOXES):
            t1 = (b[0] - o) / d
            t2 = (b[1] - o) / d
            tn = np.nanmax(np.minimum(t1, t2), axis=1)
            tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tn < tf) & (tf > 0) & (tn > 0.05) & (tn < best)
            best = np.where(hit, tn, best)
            sid = np.where(hit, 10 + k, sid)
    return best, sid


def main(bag_path, gt_path, duration=40.0):
    ts = get_typestore(Stores.ROS1_NOETIC)
    PC = ts.types["sensor_msgs/msg/PointCloud2"]
    PFd = ts.types["sensor_msgs/msg/PointField"]
    Hd = ts.types["std_msgs/msg/Header"]
    Tm = ts.types["builtin_interfaces/msg/Time"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    V3 = ts.types["geometry_msgs/msg/Vector3"]
    Q = ts.types["geometry_msgs/msg/Quaternion"]

    n_az, rings = 720, 32
    el = np.deg2rad(np.linspace(-25, 25, rings))
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    AZ, EL = np.meshgrid(az, el)
    AZ, EL = AZ.ravel(), EL.ravel()
    frac = AZ / (2 * np.pi)
    dloc = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], axis=1)
    ring = np.repeat(np.arange(rings), n_az).astype(np.uint16)

    dt = np.dtype({"names": ["x", "y", "z", "intensity", "timestamp", "ring"],
                   "formats": ["<f4", "<f4", "<f4", "<f4", "<f8", "<u2"],
                   "offsets": [0, 4, 8, 16, 24, 32], "itemsize": 48})
    fields = [PFd(name="x", offset=0, datatype=7, count=1), PFd(name="y", offset=4, datatype=7, count=1),
              PFd(name="z", offset=8, datatype=7, count=1), PFd(name="intensity", offset=16, datatype=7, count=1),
              PFd(name="timestamp", offset=24, datatype=8, count=1), PFd(name="ring", offset=32, datatype=4, count=1)]

    n_scans = int(duration * RATE)
    gt = []
    with Writer(bag_path) as w:
        cl = w.add_connection("/hesai/pandar", "sensor_msgs/msg/PointCloud2", typestore=ts)
        ci = w.add_connection("/alphasense/imu", "sensor_msgs/msg/Imu", typestore=ts)
        for s in range(n_scans):
            t_scan = s / RATE
            tp = t_scan + frac / RATE                       # each point has its own time
            x, y, z, yaw = pose_at(tp)
            c, sn = np.cos(yaw), np.sin(yaw)
            dw = np.stack([c * dloc[:, 0] - sn * dloc[:, 1], sn * dloc[:, 0] + c * dloc[:, 1], dloc[:, 2]], axis=1)
            o = np.stack([x, y, z], axis=1)
            r, sid = raycast(o, dw)
            r = r + rng.normal(0, 0.01, len(r))
            xyz = dloc * r[:, None]                          # point in sensor frame AT ITS OWN TIME
            arr = np.zeros(len(r), dtype=dt)
            arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            arr["intensity"] = np.clip(30 + sid * 12, 0, 255)
            arr["timestamp"] = T0 + tp
            arr["ring"] = ring
            sec = int(T0 + t_scan)
            msg = PC(header=Hd(seq=s, stamp=Tm(sec=sec, nanosec=int(((T0 + t_scan) % 1) * 1e9)), frame_id="PandarXT-32"),
                     height=1, width=len(arr), fields=fields, is_bigendian=False, point_step=48,
                     row_step=48 * len(arr), data=np.frombuffer(arr.tobytes(), dtype=np.uint8), is_dense=True)
            w.write(cl, int((T0 + t_scan) * 1e9), ts.serialize_ros1(msg, "sensor_msgs/msg/PointCloud2"))
            # ground truth at mid-scan (KISS-ICP convention)
            gx, gy, gz, gyaw = pose_at(t_scan + 0.5 / RATE)
            gt.append([gx, gy, gz, gyaw])
        for k in range(int(duration * 200)):
            t = k / 200.0
            cov = np.zeros(9)
            m = Imu(header=Hd(seq=k, stamp=Tm(sec=int(T0 + t), nanosec=int(((T0 + t) % 1) * 1e9)), frame_id="imu"),
                    orientation=Q(x=0.0, y=0.0, z=0.0, w=0.0), orientation_covariance=cov,
                    angular_velocity=V3(x=0.0, y=0.0, z=0.05), angular_velocity_covariance=cov,
                    linear_acceleration=V3(x=0.0, y=0.0, z=9.81), linear_acceleration_covariance=cov)
            w.write(ci, int((T0 + t) * 1e9), ts.serialize_ros1(m, "sensor_msgs/msg/Imu"))
    np.save(gt_path, np.asarray(gt))
    print(f"wrote {bag_path} ({n_scans} scans) and {gt_path}")


if __name__ == "__main__":
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    main(sys.argv[1], sys.argv[2], dur)
