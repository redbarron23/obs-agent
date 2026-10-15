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
CLOUDS = ("azure", "gcp")
# Template order, so a mapped CSV reads like the downloadable example.
COLUMN_ORDER = (
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
    "billing_account",
    "created_at",
    "region",
)


class InventoryImportError(ValueError):
    """A CSV cannot safely be used by the local review workflow."""


def _rows(mask: pd.Series) -> str:
    """Spreadsheet row numbers (the header is row 1) for the rows a reviewer must correct."""
    lines = [str(i + 2) for i in mask[mask].index]
    more = f" and {len(lines) - 10} more" if len(lines) > 10 else ""
    return f"row{'s' if len(lines) > 1 else ''} {', '.join(lines[:10])}{more}"


def _list(value, column: str, line: int) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        if text.startswith("["):
            raise InventoryImportError(
                f"{column} contains an invalid JSON list (row {line})."
            ) from exc
        parsed = [part.strip() for part in text.split("|") if part.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise InventoryImportError(
            f"{column} must be a JSON string list or pipe-delimited text (row {line})."
        )
    return parsed


# Common export headers, compared after lowercasing and dropping non-alphanumerics. Suggestions
# only pre-fill the mapping form: the reviewer confirms every field before anything is saved.
SYNONYMS = {
    "cloud": {"provider", "cloudprovider", "platform"},
    "resource_id": {"id", "subscriptionid", "projectid", "subscription", "project", "resource"},
    "parent": {"managementgroup", "folder", "parentid", "hierarchy", "path"},
    "owners": {"owner", "ownerlist", "contacts"},
    "created_by": {"creator", "createdby", "author"},
    "top_services": {"services", "service", "topservice"},
    "avg_daily_cost": {"dailycost", "averagedailycost", "avgcost", "costperday"},
    "weekend_ratio": {"weekendratio", "weekendcostratio"},
    "cost_cv": {"costcv", "costvolatility", "coefficientofvariation"},
    "team": {"teamtag", "owningteam"},
    "env": {"environment", "envtag", "stage"},
    "cost_center": {"costcentre", "costcenter", "cc"},
}


def _key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def csv_columns(source: bytes) -> list[str]:
    """Header of an uploaded CSV, for the column-mapping step."""
    try:
        return list(pd.read_csv(BytesIO(source), nrows=0).columns)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InventoryImportError(f"Could not read CSV: {exc}") from exc


def suggest_mapping(columns: list[str]) -> dict[str, str]:
    """Best-guess field -> source column, by exact name first and then common synonyms."""
    fields = sorted(REQUIRED_COLUMNS | OPTIONAL_COLUMNS)
    by_key: dict[str, list[str]] = {}
    for column in columns:
        by_key.setdefault(_key(column), []).append(column)
    mapping, used = {}, set()
    for exact in (True, False):
        for field in fields:
            if field in mapping:
                continue
            keys = {_key(field)} if exact else SYNONYMS.get(field, set())
            # Skip ambiguous headers rather than guess between them.
            candidates = [c for k in keys for c in by_key.get(k, []) if c not in used]
            if len(candidates) == 1:
                mapping[field] = candidates[0]
                used.add(candidates[0])
    return mapping


def apply_column_mapping(source: bytes, mapping: dict[str, str | None]) -> bytes:
    """Rename mapped source columns to the import schema and drop the rest.

    Values pass through unchanged for load_inventory_csv to validate. Every required field
    must come from a real column: nothing is defaulted or invented.
    """
    mapping = {field: column for field, column in mapping.items() if column}
    unknown = sorted(set(mapping) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
    if unknown:
        raise InventoryImportError(f"Unknown target fields: {', '.join(unknown)}.")
    missing = sorted(REQUIRED_COLUMNS - set(mapping))
    if missing:
        raise InventoryImportError(
            f"Map a source column to each required field; unmapped: {', '.join(missing)}."
        )
    fields_by_column: dict[str, list[str]] = {}
    for field, column in mapping.items():
        fields_by_column.setdefault(column, []).append(field)
    reused = {c: sorted(f) for c, f in fields_by_column.items() if len(f) > 1}
    if reused:
        raise InventoryImportError(
            "Each source column can fill only one field; "
            + "; ".join(f"{c} is mapped to {', '.join(f)}" for c, f in sorted(reused.items()))
            + "."
        )
    try:
        df = pd.read_csv(BytesIO(source), dtype=str, keep_default_na=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InventoryImportError(f"Could not read CSV: {exc}") from exc
    absent = sorted(set(mapping.values()) - set(df.columns))
    if absent:
        raise InventoryImportError(f"Source columns not in the CSV: {', '.join(absent)}.")
    fields = [field for field in COLUMN_ORDER if field in mapping]
    mapped = df[[mapping[field] for field in fields]].set_axis(fields, axis=1)
    return mapped.to_csv(index=False).encode()


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
    unsupported = ~df["cloud"].isin(CLOUDS)
    if unsupported.any():
        values = sorted(set(df.loc[unsupported, "cloud"].fillna("<blank>").astype(str)))
        raise InventoryImportError(
            f"cloud must be 'azure' or 'gcp'; found {', '.join(values)} at {_rows(unsupported)}."
        )
    df = df.copy()
    df["resource_id"] = df["resource_id"].fillna("").astype(str).str.strip()
    blank = df["resource_id"].eq("")
    if blank.any():
        raise InventoryImportError(f"resource_id must be present: blank at {_rows(blank)}.")
    duplicated = df["resource_id"].duplicated(keep=False)
    if duplicated.any():
        raise InventoryImportError(f"resource_id must be unique: repeated at {_rows(duplicated)}.")
    df.attrs["label_config"] = config
    for column in ("owners", "top_services"):
        df[column] = [_list(value, column, i + 2) for i, value in df[column].items()]
    for column in ("avg_daily_cost", "weekend_ratio", "cost_cv"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(df[column]) | (df[column] < 0)
        if invalid.any():
            raise InventoryImportError(
                f"{column} must contain finite non-negative numeric values; "
                f"missing or invalid at {_rows(invalid)}."
            )
    for column in OPTIONAL_COLUMNS:
        if column not in df:
            df[column] = None
    for label in LABELS:
        cleaned = df[label].map(lambda value, name=label: normalise(name, value, config)).dropna()
        counts = cleaned.value_counts()
        if len(counts) < 2 or counts.min() < 2:
            found = ", ".join(f"{value}={count}" for value, count in counts.items()) or "none"
            raise InventoryImportError(
                f"{label} needs at least two recognized values with two labelled resources each "
                f"for local review; found {found}. Check the {label} column mapping, add labelled "
                "resources, or add aliases to the vocabulary."
            )
    return df
