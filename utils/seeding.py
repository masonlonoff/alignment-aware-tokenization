# utils/seeding.py
"""
Unified seeding & run metadata logging.
"""
from __future__ import annotations
import os
import time
import json
import hashlib
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:8]


def log_run_meta(out_dir: str, cfg: dict, extras: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg.get("seed", None),
        "config_hash": config_hash(cfg),
        "config": cfg,
    }
    if extras:
        meta.update(extras)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
