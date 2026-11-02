from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger("hilti3d")


def setup_logging(work: Path, name: str = "pipeline") -> Path:
    logdir = Path(work) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    return path


def mem() -> str:
    """Current process RAM + system free RAM, for log lines."""
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        return f"RAM used {rss:.2f} GB, system free {avail:.2f} GB"
    except Exception:  # noqa: BLE001
        return "RAM n/a"
