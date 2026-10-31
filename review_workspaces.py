"""Local saved review snapshots, each with its own decision database."""

import hashlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from drift import compare_snapshots
from inventory_import import load_inventory_csv
from label_config import inventory_config
from label_review import DecisionStore
from review_context import inventory_id


def review_directory() -> Path:
    return Path(os.environ.get("OBS_AGENT_REVIEW_DIR", "review-workspaces"))


def _path(review_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", review_id):
        raise ValueError("Invalid saved review ID.")
    return review_directory() / f"{review_id}.db"


def save_review(csv: bytes, name: str, config: dict | None = None) -> str:
    inventory = load_inventory_csv(csv, config)
    vocabulary = json.dumps(inventory_config(inventory), sort_keys=True)
    review_id = hashlib.sha256((inventory_id(inventory) + vocabulary).encode()).hexdigest()
    review_directory().mkdir(parents=True, exist_ok=True, mode=0o700)
    store = DecisionStore(_path(review_id))
    store.path.chmod(0o600)
    with store._connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS workspace "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL, "
            "csv BLOB NOT NULL, config TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO workspace VALUES (1, ?, ?, ?, ?)",
            (name[:200], csv, vocabulary, datetime.now(UTC).isoformat(timespec="seconds")),
        )
    return review_id


def load_review(review_id: str):
    path = _path(review_id)
    # A missing selection must not silently create an empty database.
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT csv, config FROM workspace WHERE id = 1").fetchone()
    if row is None:
        raise ValueError("Saved review has no inventory.")
    return load_inventory_csv(row[0], json.loads(row[1])), DecisionStore(path)


def saved_reviews() -> list[dict]:
    reviews = []
    for path in sorted(review_directory().glob("*.db")):
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            if not conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'workspace'"
            ).fetchone():
                continue
            row = conn.execute("SELECT name, created_at FROM workspace WHERE id = 1").fetchone()
        if row:
            reviews.append({"id": path.stem, "name": row[0], "created_at": row[1]})
    return sorted(reviews, key=lambda r: r["created_at"], reverse=True)


def compare_reviews(previous_id: str, current_id: str) -> dict:
    """Compare two saved reviews; read-only for both decision databases."""
    if previous_id == current_id:
        raise ValueError("Choose two different saved reviews to compare.")
    previous, previous_store = load_review(previous_id)
    current, _ = load_review(current_id)
    return compare_snapshots(previous, current, previous_decisions=previous_store.all())
