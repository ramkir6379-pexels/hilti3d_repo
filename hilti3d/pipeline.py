"""Command-line runner.

  python -m hilti3d.pipeline --bag /content/data/exp21.bag --work /content/drive/MyDrive/hilti_work \
         --stages all --set t_end=60

Stages: extract -> odometry -> tilt -> map -> mesh -> export -> evaluate.  Each stage writes a marker
(state/<stage>.done) so a re-run skips finished stages (use --force to redo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

from .config import Config
from .logutil import log, mem, setup_logging

STAGES = ["extract", "odometry", "tilt", "map", "mesh", "export", "evaluate"]


def _stage_fn(name: str):
    if name == "extract":
        from . import extract as m
    elif name == "odometry":
        from . import odometry as m
    elif name == "tilt":
        from . import tilt as m
    elif name == "evaluate":
        from . import evaluate as m
    elif name == "map":
        from . import mapping as m
    elif name == "mesh":
        from . import mesh as m
    elif name == "export":
        from . import export as m
    else:
        raise ValueError(name)
    return m.run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LiDAR bag -> map -> mesh -> viewer")
    ap.add_argument("--config", default=None, help="YAML config (see configs/)")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--work", required=True, help="output/working directory (use Google Drive on Colab)")
    ap.add_argument("--stages", default="all", help=f"'all' or comma list from {STAGES}")
    ap.add_argument("--force", action="store_true", help="re-run even if stage marker exists")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. --set t_end=60 --set map_voxel=0.08")
    a = ap.parse_args(argv)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "state").mkdir(exist_ok=True)
    log_path = setup_logging(work)
    log.info("Log file: %s", log_path)

    try:
        cfg = Config.from_yaml(a.config)
        if a.bag:
            cfg.bag = a.bag
        overrides = {}
        for kv in a.set:
            if "=" not in kv:
                raise ValueError(f"--set expects KEY=VALUE, got '{kv}'")
            k, v = kv.split("=", 1)
            overrides[k.strip()] = yaml.safe_load(v)
        cfg.update(overrides)
    except Exception as e:  # noqa: BLE001
        log.critical("Config error: %s", e)
        log.debug(traceback.format_exc())
        return 2

    stages = STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        log.critical("Unknown stage(s) %s. Valid: %s", bad, STAGES)
        return 2

    log.info("Config: %s", json.dumps(cfg.to_dict()))
    (work / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log.info("Start | %s", mem())

    t_all = time.time()
    for s in stages:
        marker = work / "state" / f"{s}.done"
        if marker.exists() and not a.force:
            log.info("[%s] already done - skipping (use --force to redo)", s)
            continue
        log.info("=" * 70)
        log.info("[%s] starting", s)
        t0 = time.time()
        try:
            stats = _stage_fn(s)(cfg, work)
            marker.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "stats": stats}, default=str))
            log.info("[%s] finished in %.1f s", s, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.critical("[%s] FAILED: %s", s, e)
            log.critical(traceback.format_exc())
            log.critical("Fix the problem and re-run the same command; finished stages are skipped.")
            return 1
        # a changed upstream result invalidates downstream markers
        for later in STAGES[STAGES.index(s) + 1:]:
            m = work / "state" / f"{later}.done"
            if m.exists() and s in stages:
                m.unlink()

    log.info("ALL DONE in %.1f s. Results in %s", time.time() - t_all, work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
