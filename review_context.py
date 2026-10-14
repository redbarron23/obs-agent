"""Fingerprints binding an approval to the inventory and inference implementation."""

import hashlib
import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from inventory import INVENTORY_DF
from label_config import inventory_config
from labeling import AUTO_APPLY_THRESHOLD, BEST_STRATEGY


def inventory_id(inventory=None) -> str:
    df = INVENTORY_DF if inventory is None else inventory
    df = df.reindex(sorted(df.columns), axis=1).copy()
    for name in ("avg_daily_cost", "weekend_ratio", "cost_cv"):
        df[name] = df[name].astype(float)
    content = df.to_json(orient="split", index=False, double_precision=15, date_format="iso")
    return hashlib.sha256(content.encode()).hexdigest()


@lru_cache(maxsize=1)
def _implementation_id() -> str:
    digest = hashlib.sha256()
    for name in ("labeling.py", "label_config.py", "inventory.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for package in ("scikit-learn", "numpy", "scipy"):
        digest.update(f"{package}={version(package)}".encode())
    return digest.hexdigest()


def review_context(inventory=None) -> dict:
    settings = json.dumps(
        {
            "config": inventory_config(inventory),
            "strategy": BEST_STRATEGY,
            "threshold": AUTO_APPLY_THRESHOLD,
            "implementation": _implementation_id(),
        },
        sort_keys=True,
    )
    return {
        "inventory_id": inventory_id(inventory),
        "model_version": "sha256:" + hashlib.sha256(settings.encode()).hexdigest(),
    }
