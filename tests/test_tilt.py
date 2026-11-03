"""Unit test for the gravity tilt correction.

Truth: a level sensor walking a curved path (yaw only). We inject a slow roll+pitch drift
(up to ~4 deg) into the poses, the way LiDAR odometry drifts, and give the IMU the TRUE
gravity (accelerometer sees (0,0,g) in the sensor frame plus walking-like noise).
The correction must remove most of the drift in orientation and height.
    python tests/test_tilt.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hilti3d.tilt import correct_poses, angle_deg

rng = np.random.default_rng(1)
N, DT = 1500, 0.1
t = np.arange(N) * DT
# true path: walk 1 m/s along a gentle S-curve, sensor level
heading = 0.6 * np.sin(0.05 * t)
pos = np.cumsum(np.stack([np.cos(heading), np.sin(heading), 0 * t], axis=1) * DT, axis=0)
Rt = R.from_rotvec(np.stack([0 * heading, 0 * heading, heading], axis=1)).as_matrix()
P_true = np.tile(np.eye(4), (N, 1, 1)); P_true[:, :3, :3] = Rt; P_true[:, :3, 3] = pos

# injected drift: world-frame tilt that grows to ~4 deg (about an axis in the horizontal plane)
theta = np.radians(4.0) * t / t[-1]
E = R.from_rotvec(np.stack([theta * 0.8, theta * 0.6, 0 * theta], axis=1)).as_matrix()
P_bad = P_true.copy()
for k in range(N):
    P_bad[k, :3, :3] = E[k] @ Rt[k]
for k in range(N - 1):                       # odometry re-integrates steps with the tilted rotations
    d = Rt[k].T @ (pos[k + 1] - pos[k])
    P_bad[k + 1, :3, 3] = P_bad[k, :3, 3] + P_bad[k, :3, :3] @ d

# IMU: 200 Hz, true gravity in the sensor frame + walking-like oscillation (zero mean) + noise
ti = np.arange(0, t[-1], 0.005)
a = np.tile([0.0, 0.0, 9.81], (len(ti), 1))
a += np.stack([1.5 * np.sin(2 * np.pi * 1.8 * ti), 1.0 * np.sin(2 * np.pi * 0.9 * ti + 1),
               2.0 * np.sin(2 * np.pi * 1.8 * ti + 2)], axis=1) + rng.normal(0, 0.3, (len(ti), 3))
imu = np.column_stack([ti, np.zeros((len(ti), 3)), a])

new, corr, init_tilt = correct_poses(P_bad, t, imu, np.eye(3), window_s=6.0, init_s=3.0)

def tilt(P):   # angle between the sensor's z-axis (true: level) and world vertical, from roll/pitch only
    return np.degrees(np.arccos(np.clip(P[:, 2, 2], -1, 1)))
up_bad, up_new = tilt(P_bad), tilt(new)
print(f"tilt of sensor-up vs vertical: BEFORE max {up_bad.max():.2f} deg, final {up_bad[-1]:.2f} | AFTER max {up_new.max():.2f}, final {up_new[-1]:.2f}")
print(f"final height error: BEFORE {abs(P_bad[-1,2,3]):.2f} m  AFTER {abs(new[-1,2,3]):.2f} m  (true 0.00)")
assert up_bad[-1] > 3.0, "test setup: injected drift too small"
assert up_new.max() < 1.0, "tilt correction should bring tilt below 1 degree"
assert abs(new[-1, 2, 3]) < 0.3 * abs(P_bad[-1, 2, 3]), "height drift should shrink by more than 70%"
print("TILT TEST PASSED")
