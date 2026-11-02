# hilti3d - Phase 1: LiDAR bag -> merged map -> mesh -> web viewer

Takes a ROS1 bag (built for Hilti SLAM Challenge 2022, Hesai PandarXT-32) and produces:
a motion-corrected **trajectory**, ONE merged **point-cloud map**, a LiDAR-only **mesh**, and a
static **Three.js viewer**. Heavy work is meant to run on **free Colab**; your laptop only edits code.

## Pipeline
| Stage | Output (in the work dir) | How |
|---|---|---|
| extract  | scans/, scans_index.csv, imu.npy | stream the bag once with `rosbags` |
| odometry | poses.npy, trajectory_tum.txt, deskewed/ | KISS-ICP (LiDAR odometry, per-point-time deskew) |
| map      | map.ply | transform scans by poses, chunked voxel merge, outlier filter |
| mesh     | mesh_full.ply, mesh_web.ply | Poisson reconstruction, normals oriented toward the sensor path |
| export   | viewer/ | Three.js viewer (points + mesh + path) |

The IMU is extracted and stored for Phase 2 / IMU-aided SLAM, but Phase 1 odometry is LiDAR-only.

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
    python tests/test_synthetic.py      # simulated spinning LiDAR with known truth

## Known limits (honest list)
- LiDAR-only odometry has no loop closure: long loops drift. On a deliberately hard simulated
  room (flat walls) the best setting gave about 1.2 m mean error over a 13 m path; real scenes with
  more structure should do better but this MUST be checked against the Hilti ground truth.
- `icp_voxel` / `max_range` change accuracy a lot (see tests). Tune per scene.
- Poisson invents surface in empty regions; low-density trimming reduces but does not remove this.
- Map frame = first-scan LiDAR frame (not gravity-aligned). Gravity alignment comes with the IMU in Phase 2.
- Viewer was not browser-tested in the build environment; if it shows nothing, open the browser console
  and use the http server (file:// is blocked by browsers).
