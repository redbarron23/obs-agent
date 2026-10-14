"""Read-only validation and normalisation for sanitized Azure/GCP inventories."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from label_config import default_config, load_label_config
from labeling import LABELS, normalise

REQUIRED_COLUMNS = {
    "cloud",
    "resource_id",
    "parent",
    "owners",
    "created_by",
    "top_services",
    "avg_daily_cost",
    "weekend_ratio",
    "cost_cv",
    "team",
    "env",
    "cost_center",
}
OPTIONAL_COLUMNS = {"billing_account", "created_at", "region"}


class InventoryImportError(ValueError):
    """A CSV cannot safely be used by the local review workflow."""


def _list(value, column: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        if text.startswith("["):
            raise InventoryImportError(f"{column} contains an invalid JSON list.") from exc
        parsed = [part.strip() for part in text.split("|") if part.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise InventoryImportError(f"{column} must be a JSON string list or pipe-delimited text.")
    return parsed


def load_inventory_csv(source: str | Path | bytes, config: dict | None = None) -> pd.DataFrame:
    """Return a normalized, local-only inventory suitable for label inference."""
    config = (
        load_label_config(json.dumps(config).encode()) if config is not None else default_config()
    )
    if isinstance(source, bytes):
        source = BytesIO(source)
    try:
        df = pd.read_csv(source, keep_default_na=True, dtype={"resource_id": str})
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InventoryImportError(f"Could not read CSV: {exc}") from exc
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise InventoryImportError(f"Missing required columns: {', '.join(missing)}.")
    if df.empty:
        raise InventoryImportError("Inventory CSV contains no resources.")
    if df["cloud"].isna().any() or not set(df["cloud"]).issubset({"azure", "gcp"}):
        raise InventoryImportError("cloud must be 'azure' or 'gcp'.")
    if df["resource_id"].isna().any() or df["resource_id"].duplicated().any():
        raise InventoryImportError("resource_id must be present and unique.")
    df = df.copy()
    df["resource_id"] = df["resource_id"].astype(str).str.strip()
    if df["resource_id"].eq("").any() or df["resource_id"].duplicated().any():
        raise InventoryImportError("resource_id must be present and unique.")
    df.attrs["label_config"] = config
    for column in ("owners", "top_services"):
        df[column] = df[column].map(lambda value, name=column: _list(value, name))
    for column in ("avg_daily_cost", "weekend_ratio", "cost_cv"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(df[column]).all() or (df[column] < 0).any():
            raise InventoryImportError(f"{column} must contain finite non-negative numeric values.")
    for column in OPTIONAL_COLUMNS:
        if column not in df:
            df[column] = None
    for label in LABELS:
        cleaned = df[label].map(lambda value, name=label: normalise(name, value, config)).dropna()
        counts = cleaned.value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise InventoryImportError(
                f"{label} needs at least two recognized values with two labelled resources each "
                "for local review."
            )
    return df
