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

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LiDAR scene viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#101216;color:#dfe3ea;font:13px system-ui,sans-serif;overflow:hidden}
  #ui{position:absolute;top:10px;left:10px;background:rgba(20,24,30,.85);padding:10px 12px;border-radius:8px;line-height:1.9}
  #msg{position:absolute;bottom:10px;left:10px;opacity:.8}
  label{display:block;cursor:pointer}
</style></head><body>
<div id="ui"><b>LiDAR scene</b>
  <label><input type="checkbox" id="cPts" checked> Point cloud</label>
  <label><input type="checkbox" id="cMesh" checked> Mesh</label>
  <label><input type="checkbox" id="cPath" checked> Sensor path</label>
  <label>Point size <input type="range" id="sz" min="1" max="6" step="0.5" value="2"></label>
</div><div id="msg">Loading...</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';

const msg = document.getElementById('msg');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101216);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.05, 5000);
camera.up.set(0,0,1);                     // data is Z-up
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x303040, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(30,-20,60); scene.add(sun);

const loader = new PLYLoader();
const load = (url) => new Promise((res, rej) => loader.load(url, res, undefined, rej));
let pts, mesh, path;

async function main(){
  const errors = [];
  try {
    const g = await load('map_preview.ply');
    const hasCol = g.hasAttribute('color');
    pts = new THREE.Points(g, new THREE.PointsMaterial({size:2, sizeAttenuation:false, vertexColors:hasCol, color:0xcccccc}));
    scene.add(pts);
    g.computeBoundingBox();
    const c = new THREE.Vector3(); g.boundingBox.getCenter(c);
    const sz = new THREE.Vector3(); g.boundingBox.getSize(sz);
    const d = Math.max(sz.x, sz.y, sz.z);
    controls.target.copy(c);
    camera.position.set(c.x + d*0.6, c.y - d*0.8, c.z + d*0.6);
    controls.update();
  } catch(e){ errors.push('map_preview.ply'); }
  try {
    const g = await load('mesh_web.ply');
    if(!g.hasAttribute('normal')) g.computeVertexNormals();
    mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors:g.hasAttribute('color'), color:0xdddddd, roughness:.9, metalness:0, side:THREE.DoubleSide}));
    scene.add(mesh);
  } catch(e){ errors.push('mesh_web.ply'); }
  try {
    const t = await (await fetch('trajectory.json')).json();
    const v = t.flat();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
    path = new THREE.Line(g, new THREE.LineBasicMaterial({color:0xffb000}));
    scene.add(path);
  } catch(e){ errors.push('trajectory.json'); }
  msg.textContent = errors.length ? ('Could not load: ' + errors.join(', ') + '  (open via http server, not file://)') : 'Drag = rotate, wheel = zoom, right-drag = pan';
}
main();

const $ = (id)=>document.getElementById(id);
$('cPts').onchange  = e => { if(pts)  pts.visible  = e.target.checked; };
$('cMesh').onchange = e => { if(mesh) mesh.visible = e.target.checked; };
$('cPath').onchange = e => { if(path) path.visible = e.target.checked; };
$('sz').oninput     = e => { if(pts)  pts.material.size = parseFloat(e.target.value); };
addEventListener('resize', ()=>{ camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>
"""


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
