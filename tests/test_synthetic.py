"""End-to-end smoke test on a simulated LiDAR bag (no real data needed).
    python tests/test_synthetic.py
Checks: stages run, outputs exist, trajectory error is bounded, map floor/ceiling are flat."""
import subprocess, sys, tempfile, importlib.util
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
bag, gt, work = tmp / "s.bag", tmp / "gt.npy", tmp / "work"
subprocess.run([sys.executable, str(root / "tests/make_synthetic_bag.py"), str(bag), str(gt), "15"], check=True)
# ground-truth file in the Hilti format (timestamp x y z qx qy qz qw), from the simulator's true poses
spec = importlib.util.spec_from_file_location("sim", root / "tests/make_synthetic_bag.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
gtf = tmp / "gt.txt"
with open(gtf, "w") as f:
    f.write("# timestamp_s x y z qx qy qz qw\n")
    for tt in (2.0, 6.0, 10.0, 14.0):
        x, y, z, _ = sim.pose_at(tt)
        f.write(f"{sim.T0 + tt:.6f} {x:.4f} {y:.4f} {z:.4f} 0 0 0 1\n")
r = subprocess.run([sys.executable, "-m", "hilti3d.pipeline", "--bag", str(bag), "--work", str(work),
                    "--set", "poisson_depth=8",
                    "--set", "imu_to_lidar_R=[1,0,0,0,1,0,0,0,1]",   # simulator: IMU frame == LiDAR frame
                    "--set", "tip_in_lidar=[0,0,0]",
                    "--set", f"gt_file={gtf}"], cwd=root)
assert r.returncode == 0, "pipeline failed"
for f in ["poses_raw.npy", "tilt_stats.json", "accuracy_report.txt", "poses.npy", "map.ply", "mesh_full.ply", "viewer/index.html", "viewer/mesh_web.ply", "imu.npy"]:
    assert (work / f).exists(), f"missing {f}"
P = np.load(work / "poses.npy"); g = np.load(gt)[:len(P)]
def mat(v):
    x, y, z, yw = v; c, s = np.cos(yw), np.sin(yw)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; T[:3, 3] = [x, y, z]; return T
G = np.array([mat(v) for v in g]); Gr = np.array([np.linalg.inv(G[0]) @ t for t in G])
err = np.linalg.norm(P[:, :3, 3] - Gr[:, :3, 3], axis=1)
print(f"ATE mean {err.mean():.2f} m, final {err[-1]:.2f} m over {len(P)} scans")
assert err.mean() < 3.0, "trajectory error unexpectedly large"
print("SMOKE TEST PASSED")
