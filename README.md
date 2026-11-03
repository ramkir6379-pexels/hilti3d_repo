# hilti3d - Phase 1: LiDAR bag -> merged map -> mesh -> web viewer

Takes a ROS1 bag (built for Hilti SLAM Challenge 2022, Hesai PandarXT-32) and produces:
a motion-corrected **trajectory**, ONE merged **point-cloud map**, a LiDAR-only **mesh**, and a
static **Three.js viewer**. Heavy work is meant to run on **free Colab**; your laptop only edits code.

## Pipeline
| Stage | Output (in the work dir) | How |
|---|---|---|
| extract  | scans/, scans_index.csv, imu.npy | stream the bag once with `rosbags` |
| odometry | poses.npy, trajectory_tum.txt, deskewed/ | KISS-ICP (LiDAR odometry, per-point-time deskew) |
| tilt     | poses.npy (corrected), poses_raw.npy, tilt_stats.json | IMU gravity correction: removes accumulated roll/pitch drift, levels the world (z = up) |
| map      | map.ply | transform scans by poses, chunked voxel merge, outlier filter |
| mesh     | mesh_full.ply, mesh_web.ply | Poisson reconstruction, normals oriented toward the sensor path |
| export   | viewer/ | Three.js viewer (points + mesh + path) |
| evaluate | accuracy_report.txt, accuracy.json | pairwise distances / vertical error vs Hilti ground-truth control points (needs `gt_file`) |

Odometry is LiDAR-only; the IMU is used afterwards, only to correct tilt (gravity direction).

## Use on Colab
1. Push this folder to GitHub (`git init; git add .; git commit -m "phase1"; git remote add origin <url>; git push -u origin main`).
2. Open `notebooks/phase1_colab.ipynb` in Colab, set `REPO_URL`, run cells top to bottom.
   Start with `T_END = 60`, then `0` for the whole 153 s bag.

## Use locally (any machine with enough RAM)
    pip install -r requirements.txt
    python -m hilti3d.pipeline --config configs/exp21.yaml --bag path/to/exp21.bag --work work/exp21 --set t_end=60
Overrides: `--set key=value` (all keys in `hilti3d/config.py`). Redo a stage: `--stages mesh --force`.

## View the result
    cd <work>/viewer && python -m http.server 8000     # open http://localhost:8000

## Test without real data
    python tests/test_synthetic.py      # simulated spinning LiDAR with known truth (full pipeline)
    python tests/test_tilt.py           # injected tilt drift must be removed

## Known limits (honest list)
- Measured on exp21 (full run, before tilt correction): most point-pair distances within 0-3% but about 6 m of vertical drift. The tilt stage targets that; check `accuracy_report.txt` for the before/after numbers.
- Tilt correction assumes the IMU/LiDAR clocks agree and `imu_to_lidar_R` is right (it is logged: a large `initial_world_tilt_deg` means it is wrong). Accelerometer bias (~0.3 deg) is not removed.
- LiDAR-only odometry has no loop closure: long loops drift. On a deliberately hard simulated
  room (flat walls) the best setting gave about 1.2 m mean error over a 13 m path; real scenes with
  more structure should do better but this MUST be checked against the Hilti ground truth.
- `icp_voxel` / `max_range` change accuracy a lot (see tests). Tune per scene.
- Poisson invents surface in empty regions; low-density trimming reduces but does not remove this.
- Map frame = first-scan LiDAR frame (not gravity-aligned). Gravity alignment comes with the IMU in Phase 2.
- Viewer was not browser-tested in the build environment; if it shows nothing, open the browser console
  and use the http server (file:// is blocked by browsers).
