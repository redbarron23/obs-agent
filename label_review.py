"""
Human-in-the-loop label review on top of labeling.py.

  resource_report()  one resource: current tags, suggestions, confidence,
                     reasons, suspected-wrong flags (shared by the agent
                     tool, the REST API and the Streamlit review page)
  review_queue()     every resource with an untagged or suspect label
  DecisionStore      reviewer decisions (accept / reject / override), with an
                     append-only feedback trail shared by the API and UI
  export_*()         resolved label changes as CSV, a shell script of
                     az / gcloud commands, or a Terraform tfvars map

Resolution order for a label's final value: reviewer override → reviewer
accept → (optionally) high-confidence auto-apply → the existing tag.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from labeling import (
    AUTO_APPLY_THRESHOLD,
    BEST_STRATEGY,
    LABELS,
    LabelInference,
    get_inference,
    get_suspects,
    normalise,
)

STATUSES = ("ok", "untagged", "suspect")
DECISIONS = ("accept", "reject", "override")
EXPORT_FORMATS = ("csv", "sh", "tfvars")

DEFAULT_DECISIONS_PATH = "label_decisions.db"


def demo_enabled() -> bool:
    """True for the public demo: OBS_AGENT_DEMO=1 in the environment or in
    Streamlit secrets (Community Cloud's settings dialog sets the latter)."""
    if os.environ.get("OBS_AGENT_DEMO") == "1":
        return True
    try:
        import streamlit as st

        return str(st.secrets.get("OBS_AGENT_DEMO", "")) == "1"
    except Exception:  # no secrets file, or not running under Streamlit
        return False


# ── Reports ────────────────────────────────────────────────────────────


def _raw(value) -> str | None:
    return None if pd.isna(value) else str(value)


def _label_report(inf: LabelInference, row, label: str, suspects: pd.DataFrame) -> dict:
    current = row[f"{label}_clean"]
    out = {
        "current": current,
        "raw": _raw(row[label]),
        "status": "ok",
        "suggested": None,
        "confidence": None,
        "action": None,
        "source": None,
        "reason": None,
        "alternatives": [],
    }
    if current is not None:
        hit = suspects[suspects["resource_id"] == row["resource_id"]]
        if hit.empty:
            return out
        # Tagged: the fitted model has seen this tag, so judge it with the
        # out-of-fold prediction instead.
        h = hit.iloc[0]
        return out | {
            "status": "suspect",
            "suggested": h["predicted"],
            "confidence": round(float(h["confidence"]), 3),
            "action": "needs review",
            "source": "out-of-fold model",
            "reason": f"tagged {label}={current}, but a model that never saw this "
            f"resource predicts {label}={h['predicted']}",
        }
    s = inf.suggest(row, label, BEST_STRATEGY[label])
    return out | {
        "status": "untagged",
        "suggested": s.label,
        "confidence": s.confidence,
        "action": "auto-apply" if s.confidence >= AUTO_APPLY_THRESHOLD else "needs review",
        "source": s.source,
        "reason": s.reason,
        "alternatives": [{"value": c, "confidence": p} for c, p in s.alternatives[1:]],
    }


def _report(inf: LabelInference, row) -> dict:
    labels = {label: _label_report(inf, row, label, get_suspects(label)) for label in LABELS}
    team = labels["team"]["suggested"] or labels["team"]["current"]
    return {
        "resource_id": row["resource_id"],
        "cloud": row["cloud"],
        "parent": row["parent"],
        "owners": list(row["owners"]),
        "labels": labels,
        "cost_center": {
            "current": _raw(row["cost_center"]),
            "implied": inf.cost_center_of.get(team),
        },
        "needs_attention": any(lbl["status"] != "ok" for lbl in labels.values()),
    }


@lru_cache(maxsize=1)
def all_reports() -> tuple[dict, ...]:
    """Reports for the whole synthetic inventory (computed once)."""
    inf = get_inference()
    return tuple(_report(inf, row) for _, row in inf.df.iterrows())


def resource_report(resource_id: str) -> dict | None:
    for r in all_reports():
        if r["resource_id"] == resource_id:
            return r
    return None


def review_queue(
    *,
    label: str | None = None,
    status: str | None = None,
    cloud: str | None = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    search: str | None = None,
) -> list[dict]:
    """Resources needing review, most confident suggestions first.

    *status* is 'untagged' or 'suspect' (default: either); *label* limits
    the status check to one label.
    """
    labels = [label] if label else list(LABELS)
    wanted = {status} if status else {"untagged", "suspect"}
    out = []
    for r in all_reports():
        if cloud and r["cloud"] != cloud:
            continue
        if search and search.lower() not in r["resource_id"].lower():
            continue
        hits = [r["labels"][lbl] for lbl in labels if r["labels"][lbl]["status"] in wanted]
        if not hits:
            continue
        if not any(min_confidence <= h["confidence"] <= max_confidence for h in hits):
            continue
        out.append(r)
    return sorted(out, key=lambda r: -max(r["labels"][lbl]["confidence"] or 0 for lbl in labels))


def coverage() -> dict:
    reports = all_reports()
    per_label = {}
    for label in LABELS:
        st = [r["labels"][label] for r in reports]
        per_label[label] = {
            "untagged": sum(s["status"] == "untagged" for s in st),
            "suspect": sum(s["status"] == "suspect" for s in st),
            "auto_apply_ready": sum(s["action"] == "auto-apply" for s in st),
        }
    return {
        "total": len(reports),
        "needs_attention": sum(r["needs_attention"] for r in reports),
        "auto_apply_threshold": AUTO_APPLY_THRESHOLD,
        "labels": per_label,
    }


# ── Decisions ──────────────────────────────────────────────────────────


class DecisionError(ValueError):
    """A decision that cannot be recorded (bad label, value, or no suggestion)."""


class DecisionStore:
    """Reviewer decisions keyed by (resource_id, label), stored in SQLite."""

    def __init__(self, path: str | os.PathLike | None = None):
        source = Path(path or os.environ.get("OBS_AGENT_DECISIONS", DEFAULT_DECISIONS_PATH))
        self.path = source.with_suffix(".db") if source.suffix == ".json" else source
        self._legacy_path = source if source.suffix == ".json" else self.path.with_suffix(".json")
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialise(self) -> None:
        is_new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    resource_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    value TEXT,
                    suggested TEXT,
                    previous TEXT,
                    decided_at TEXT NOT NULL,
                    PRIMARY KEY (resource_id, label)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    value TEXT,
                    suggested TEXT,
                    previous TEXT,
                    confidence REAL,
                    source TEXT,
                    reason TEXT,
                    decided_at TEXT NOT NULL
                )
                """
            )
            if is_new and self._legacy_path.exists():
                for entry in json.loads(self._legacy_path.read_text()):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO decisions
                        (resource_id, label, decision, value, suggested, previous, decided_at)
                        VALUES (
                            :resource_id, :label, :decision, :value,
                            :suggested, :previous, :decided_at
                        )
                        """,
                        entry,
                    )

    def all(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM decisions ORDER BY resource_id, label")
            return [dict(row) for row in rows]

    def get(self, resource_id: str, label: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE resource_id = ? AND label = ?", (resource_id, label)
            ).fetchone()
            return dict(row) if row else None

    def feedback(self) -> list[dict]:
        """All reviewer actions with the model snapshot they were based on."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM review_feedback ORDER BY id")
            return [dict(row) for row in rows]

    def set(self, resource_id: str, label: str, decision: str, value: str | None = None) -> dict:
        report = resource_report(resource_id)
        if report is None:
            raise KeyError(resource_id)
        if label not in LABELS:
            raise DecisionError(f"Unknown label '{label}'. Choose from {list(LABELS)}.")
        if decision not in DECISIONS:
            raise DecisionError(f"Unknown decision '{decision}'. Choose from {list(DECISIONS)}.")
        lr = report["labels"][label]
        if decision == "override":
            clean = normalise(label, value)
            if clean is None:
                raise DecisionError(f"'{value}' is not a valid {label}.")
            value = clean
        elif lr["suggested"] is None:
            raise DecisionError(f"{resource_id} has no {label} suggestion to {decision}.")
        else:
            value = lr["suggested"] if decision == "accept" else None
        entry = {
            "resource_id": resource_id,
            "label": label,
            "decision": decision,
            "value": value,
            "suggested": lr["suggested"],
            "previous": lr["current"],
            "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        feedback = entry | {
            "confidence": lr["confidence"],
            "source": lr["source"],
            "reason": lr["reason"],
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                (resource_id, label, decision, value, suggested, previous, decided_at)
                VALUES (:resource_id, :label, :decision, :value, :suggested, :previous, :decided_at)
                ON CONFLICT(resource_id, label) DO UPDATE SET
                    decision = excluded.decision,
                    value = excluded.value,
                    suggested = excluded.suggested,
                    previous = excluded.previous,
                    decided_at = excluded.decided_at
                """,
                entry,
            )
            conn.execute(
                """
                INSERT INTO review_feedback
                (resource_id, label, decision, value, suggested, previous,
                 confidence, source, reason, decided_at)
                VALUES (:resource_id, :label, :decision, :value, :suggested, :previous,
                        :confidence, :source, :reason, :decided_at)
                """,
                feedback,
            )
        return entry

    def delete(self, resource_id: str, label: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM decisions WHERE resource_id = ? AND label = ?", (resource_id, label)
            )
            return cursor.rowcount == 1


def _blank(value):
    """Table cells come back as None, NaN or "" when empty."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def editor_changes(
    before: pd.DataFrame, after: pd.DataFrame
) -> list[tuple[str, str | None, str | None]]:
    """(resource_id, decision, override) for rows a reviewer changed in the table.

    decision None means "clear the decision". Picking an override value
    without a decision counts as an override.
    """
    out = []
    for b, a in zip(before.itertuples(), after.itertuples(), strict=False):
        override = _blank(a.override)
        decision = _blank(a.decision) or ("override" if override else None)
        if (decision, override) != (_blank(b.decision), _blank(b.override)):
            out.append((a.resource_id, decision, override if decision == "override" else None))
    return out


# ── Resolution and export ──────────────────────────────────────────────


def resolve(report: dict, decisions: dict[str, dict], include_auto: bool) -> dict:
    """Final value and how it was reached, per label."""
    out = {}
    for label in LABELS:
        lr = report["labels"][label]
        d = decisions.get(f"{report['resource_id']}|{label}")
        if d and d["decision"] in ("accept", "override"):
            out[label] = (d["value"], d["decision"])
        elif d and d["decision"] == "reject":
            out[label] = (lr["current"], "rejected")
        elif include_auto and lr["action"] == "auto-apply":
            out[label] = (lr["suggested"], "auto-apply")
        else:
            out[label] = (lr["current"], "pending" if lr["status"] != "ok" else "existing")
    return out


def pending_changes(store: DecisionStore, include_auto: bool = True) -> list[dict]:
    """Resources whose resolved tags differ from what they carry today."""
    inf = get_inference()
    decisions = {f"{d['resource_id']}|{d['label']}": d for d in store.all()}
    changes = []
    for r in all_reports():
        final = resolve(r, decisions, include_auto)
        tags = {}
        for label in LABELS:
            value, how = final[label]
            if value is not None and value != r["labels"][label]["current"]:
                tags[label] = (value, how)
        if "team" in tags:
            # cost_center travels with team.
            tags["cost_center"] = (inf.cost_center_of[tags["team"][0]], tags["team"][1])
        if tags:
            changes.append(
                {
                    "resource_id": r["resource_id"],
                    "cloud": r["cloud"],
                    "tags": tags,
                    "previous": {lbl: r["labels"][lbl]["current"] for lbl in LABELS}
                    | {"cost_center": r["cost_center"]["current"]},
                }
            )
    return changes


def _gcp_value(value: str) -> str:
    # GCP label values: lowercase letters, digits, '-' and '_'.
    return value.lower()


def export(store: DecisionStore, fmt: str, include_auto: bool = True) -> str:
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {list(EXPORT_FORMATS)}.")
    changes = pending_changes(store, include_auto)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["resource_id", "cloud", "label", "previous", "new", "how"])
        for c in changes:
            for label, (value, how) in c["tags"].items():
                w.writerow(
                    [c["resource_id"], c["cloud"], label, c["previous"][label] or "", value, how]
                )
        return buf.getvalue()
    if fmt == "tfvars":
        doc = {"subscription_tags": {}, "project_labels": {}}
        for c in changes:
            if c["cloud"] == "azure":
                doc["subscription_tags"][c["resource_id"]] = {
                    k: v for k, (v, _) in c["tags"].items()
                }
            else:
                doc["project_labels"][c["resource_id"]] = {
                    k: _gcp_value(v) for k, (v, _) in c["tags"].items()
                }
        return json.dumps(doc, indent=2) + "\n"
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by obs-agent label review, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.",
        f"# {len(changes)} resource(s). Review before running; merges tags, never removes them.",
        "# Azure: resource_id is used as the subscription ID. GCP: as the project ID.",
        "set -euo pipefail",
        "",
    ]
    for c in changes:
        rid = c["resource_id"]
        if c["cloud"] == "azure":
            tags = " ".join(shlex.quote(f"{k}={v}") for k, (v, _) in c["tags"].items())
            lines.append(
                f"az tag update --resource-id {shlex.quote('/subscriptions/' + rid)} "
                f"--operation Merge --tags {tags}"
            )
        else:
            labels = ",".join(f"{k}={_gcp_value(v)}" for k, (v, _) in c["tags"].items())
            lines.append(
                f"gcloud projects update {shlex.quote(rid)} --update-labels={shlex.quote(labels)}"
            )
    return "\n".join(lines) + "\n"
