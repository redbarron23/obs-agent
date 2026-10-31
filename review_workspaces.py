"""Local saved review snapshots, each with its own decision database."""

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import workflow_metrics
from drift import compare_snapshots
from inventory_import import InventoryImportError, apply_column_mapping, load_inventory_csv
from label_config import LabelConfigError, inventory_config, load_label_config
from label_review import DecisionStore, demo_enabled
from review_context import inventory_id, review_context


class UnknownReviewError(ValueError):
    """No saved review has this ID. Callers must never fall back to synthetic data."""


class PrivateReviewsDisabled(PermissionError):
    """Saved private reviews are switched off (the public demo, OBS_AGENT_DEMO=1)."""


def require_private_reviews() -> None:
    """Refuse private imports and saved reviews in demo mode, as the Import page does."""
    if demo_enabled():
        raise PrivateReviewsDisabled(
            "Private inventory imports and saved reviews are disabled in the public demo."
        )


def review_directory() -> Path:
    return Path(os.environ.get("OBS_AGENT_REVIEW_DIR", "review-workspaces"))


def is_review_id(review_id) -> bool:
    """A saved review ID is exactly 64 lowercase hex digits (so never a path)."""
    return isinstance(review_id, str) and re.fullmatch(r"[0-9a-f]{64}", review_id) is not None


def _path(review_id: str) -> Path:
    if not is_review_id(review_id):
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


def _workspace(review_id: str) -> tuple:
    """(name, created_at, csv, config) of a saved review, opened read-only."""
    path = _path(review_id)
    # A missing selection must not silently create an empty database.
    if not path.is_file():
        raise UnknownReviewError("Unknown saved review ID.")
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        if not conn.execute("SELECT name FROM sqlite_master WHERE name = 'workspace'").fetchone():
            raise UnknownReviewError("Unknown saved review ID.")
        row = conn.execute(
            "SELECT name, created_at, csv, config FROM workspace WHERE id = 1"
        ).fetchone()
    if row is None:
        raise UnknownReviewError("Saved review has no inventory.")
    return row


def load_review(review_id: str):
    _, _, csv, config = _workspace(review_id)
    return load_inventory_csv(csv, json.loads(config)), DecisionStore(_path(review_id))


def review_summary(review_id: str, inventory=None) -> dict:
    """ID, name, creation time, size and approval-binding context of one saved review."""
    name, created_at, _, _ = _workspace(review_id)
    if inventory is None:
        inventory, _ = load_review(review_id)
    return {
        "id": review_id,
        "name": name,
        "created_at": created_at,
        "resources": len(inventory),
        **review_context(inventory),
    }


def import_review(
    csv: bytes,
    name: str,
    vocabulary: bytes | None = None,
    column_mapping: dict[str, str | None] | None = None,
) -> dict:
    """Validate and save an inventory the way the Import page does; return its summary.

    Re-importing the same snapshot with the same vocabulary reopens the same review.
    Raises InventoryImportError / LabelConfigError for invalid input.
    """
    require_private_reviews()
    with workflow_metrics.operation("import", invalid=(InventoryImportError, LabelConfigError)):
        config = load_label_config(vocabulary) if vocabulary is not None else None
        if column_mapping is not None:
            # Store the mapped CSV so the saved review reopens without the mapping.
            csv = apply_column_mapping(csv, column_mapping)
        review_id = save_review(csv, name, config)
    return review_summary(review_id)


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
