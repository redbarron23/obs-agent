"""
Human-in-the-loop label review on top of labeling.py.

  resource_report()  one resource: current tags, suggestions, confidence,
                     reasons, suspected-wrong flags (shared by the agent
                     tool, the REST API and the Streamlit review page)
  review_queue()     every resource with an untagged or suspect label
                     (including ones withheld for insufficient evidence)
  DecisionStore      reviewer decisions (accept / reject / override), with an
                     append-only feedback trail shared by the API and UI;
                     revoking a decision appends an attributable 'revoke' event;
                     assignments, notes and unresolved questions go to a separate
                     append-only collaboration history that never approves anything
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
import re
import shlex
import sqlite3
import threading
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

import workflow_metrics
from labeling import (
    AUTO_APPLY_THRESHOLD,
    BEST_STRATEGY,
    EVIDENCE_STATUSES,
    LABELS,
    Evidence,
    LabelInference,
    get_inference,
    get_suspects,
    normalise,
    suspect_labels,
)
from review_context import review_context

STATUSES = ("ok", "untagged", "suspect")
# Action for an untagged label whose evidence is insufficient: no suggestion,
# so it can only be resolved by an explicit override.
INSUFFICIENT_EVIDENCE = "insufficient evidence"
DECISIONS = ("accept", "reject", "override")
# review_feedback.decision for a revocation event; never a decision in its own right.
REVOKE = "revoke"
# Columns that describe a revocation; null on decision rows and on rows written before revocations
# were recorded. Not part of plan approval evidence.
REVOCATION_COLUMNS = {
    "revoked_decision": "TEXT",
    "revoked_reviewer": "TEXT",
    "revoked_decided_at": "TEXT",
    "revoked_feedback_id": "INTEGER",
    "revocation_reason": "TEXT",
}
MAX_REVOCATION_REASON = 500
# Collaboration metadata (#46): who should look at a label, notes, and unresolved questions.
# Stored in its own append-only table, so it never reaches decisions, feedback, plans or exports.
COLLABORATION_EVENTS = ("assign", "unassign", "note", "rationale", "question", "answer", "resolve")
QUESTION_STATUSES = ("open", "answered", "resolved")
MAX_NOTE = 1000
EXPORT_FORMATS = ("csv", "sh", "tfvars", "plan")

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


def _evidence_report(evidence: Evidence) -> dict:
    return {
        "status": evidence.status,
        "reason": evidence.reason,
        "supporting": list(evidence.supporting),
        "conflicting": [
            {"feature": f, "points_to": v, "count": c, "support": n}
            for f, v, c, n in evidence.conflicting
        ],
        "withheld": None
        if evidence.withheld is None
        else {"value": evidence.withheld[0], "model_score": evidence.withheld[1]},
    }


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
        "evidence": None,
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
    s, evidence = inf.decide(row, label, BEST_STRATEGY[label])
    if s.label is None:
        # Abstained (or nothing fired): no value to accept, never auto-apply.
        return out | {
            "status": "untagged",
            "action": INSUFFICIENT_EVIDENCE,
            "source": s.source,
            "reason": s.reason,
            "evidence": _evidence_report(evidence),
        }
    return out | {
        "status": "untagged",
        "suggested": s.label,
        "confidence": s.confidence,
        "action": "auto-apply" if s.confidence >= AUTO_APPLY_THRESHOLD else "needs review",
        "source": s.source,
        "reason": s.reason,
        "alternatives": [{"value": c, "confidence": p} for c, p in s.alternatives[1:]],
        "evidence": _evidence_report(evidence),
    }


def _report(inf: LabelInference, row, suspects: dict[str, pd.DataFrame]) -> dict:
    labels = {label: _label_report(inf, row, label, suspects[label]) for label in LABELS}
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
            "conflicting": inf.cost_center_conflicts.get(team, []),
        },
        "needs_attention": any(lbl["status"] != "ok" for lbl in labels.values()),
    }


@lru_cache(maxsize=1)
def all_reports() -> tuple[dict, ...]:
    """Reports for the whole synthetic inventory (computed once)."""
    with workflow_metrics.operation("inference"):
        inf = get_inference()
        suspects = {label: get_suspects(label) for label in LABELS}
        return tuple(_report(inf, row, suspects) for _, row in inf.df.iterrows())


_review_cache: OrderedDict[tuple[str, str], tuple[LabelInference, tuple[dict, ...]]] = OrderedDict()
_review_cache_lock = threading.Lock()


def _review_data(inventory: pd.DataFrame) -> tuple[LabelInference, tuple[dict, ...]]:
    context = review_context(inventory)
    key = (context["inventory_id"], context["model_version"])
    # Fit under the lock so concurrent requests for one snapshot do not duplicate training.
    with _review_cache_lock:
        if key not in _review_cache:
            with workflow_metrics.operation("inference"):
                inf = LabelInference(inventory)
                suspects = {label: suspect_labels(inventory, label) for label in LABELS}
                rows = tuple(_report(inf, row, suspects) for _, row in inf.df.iterrows())
            _review_cache[key] = (inf, rows)
            if len(_review_cache) > 4:
                _review_cache.popitem(last=False)
        _review_cache.move_to_end(key)
        return _review_cache[key]


def reports(inventory: pd.DataFrame | None = None) -> tuple[dict, ...]:
    """Reports for the synthetic demo or a caller-provided inventory snapshot."""
    if inventory is None:
        return all_reports()
    return deepcopy(_review_data(inventory)[1])


def resource_report(resource_id: str, inventory: pd.DataFrame | None = None) -> dict | None:
    for r in reports(inventory):
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
    inventory: pd.DataFrame | None = None,
    evidence: str | None = None,
    store: DecisionStore | None = None,
    assignee: str | None = None,
    unresolved: bool | None = None,
) -> list[dict]:
    """Resources needing review, highest model score first.

    *status* is 'untagged' or 'suspect' (default: either); *label* limits
    the status check to one label. *evidence* keeps only labels with that
    evidence status (e.g. 'conflicting'); 'insufficient' matches every
    abstained status. Abstained labels have no score; the confidence range
    treats them as 0, so they are listed last and kept by the default range.
    *assignee* and *unresolved* (has / has no unresolved question) filter on the
    collaboration state in *store*; they never change the suggestions shown.
    """
    labels = [label] if label else list(LABELS)
    wanted = {status} if status else {"untagged", "suspect"}
    state = {}
    if assignee is not None or unresolved is not None:
        if store is None:
            raise ValueError("Filtering by assignee or unresolved questions needs a store.")
        state = collaboration_state(store.collaboration_history())
    out = []
    for r in reports(inventory):
        if cloud and r["cloud"] != cloud:
            continue
        if search and search.lower() not in r["resource_id"].lower():
            continue
        names = [lbl for lbl in labels if r["labels"][lbl]["status"] in wanted]
        if assignee is not None or unresolved is not None:
            items = {lbl: state.get((r["resource_id"], lbl)) or {} for lbl in names}
            if assignee is not None:
                names = [lbl for lbl in names if items[lbl].get("assignee") == assignee]
            if unresolved is not None:
                names = [
                    lbl
                    for lbl in names
                    if bool(items[lbl].get("unresolved_questions")) == unresolved
                ]
        hits = [r["labels"][lbl] for lbl in names]
        if evidence:
            hits = [h for h in hits if _evidence_matches(h, evidence)]
        if not hits:
            continue
        if not any(min_confidence <= (h["confidence"] or 0.0) <= max_confidence for h in hits):
            continue
        out.append(r)
    return sorted(out, key=lambda r: -max(r["labels"][lbl]["confidence"] or 0 for lbl in labels))


def _evidence_matches(label_report: dict, wanted: str) -> bool:
    status = (label_report["evidence"] or {}).get("status")
    if wanted == "insufficient":
        return label_report["action"] == INSUFFICIENT_EVIDENCE
    return status == wanted


def coverage(inventory: pd.DataFrame | None = None) -> dict:
    report_rows = reports(inventory)
    per_label = {}
    for label in LABELS:
        st = [r["labels"][label] for r in report_rows]
        abstained = [s for s in st if s["action"] == INSUFFICIENT_EVIDENCE]
        per_label[label] = {
            "untagged": sum(s["status"] == "untagged" for s in st),
            "suspect": sum(s["status"] == "suspect" for s in st),
            "auto_apply_ready": sum(s["action"] == "auto-apply" for s in st),
            "insufficient_evidence": len(abstained),
            "insufficient_evidence_reasons": {
                status: sum((s["evidence"] or {}).get("status") == status for s in abstained)
                for status in EVIDENCE_STATUSES[1:]
            },
        }
    return {
        "total": len(report_rows),
        "needs_attention": sum(r["needs_attention"] for r in report_rows),
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
            conn.execute("BEGIN IMMEDIATE")
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
            # Additive migration for databases from earlier versions: existing rows get NULLs,
            # so no reviewer, version or revocation is invented for historical actions.
            for table in ("decisions", "review_feedback"):
                columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                wanted = dict.fromkeys(("reviewer", "inventory_id", "model_version"), "TEXT")
                if table == "review_feedback":
                    wanted |= REVOCATION_COLUMNS
                for name, kind in wanted.items():
                    if name not in columns:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            # Collaboration history (#46). A new table, so databases from earlier versions open
            # unchanged and start with no assignments, notes or questions.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_collaboration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    event TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    assignee TEXT,
                    question_id INTEGER,
                    feedback_id INTEGER,
                    text TEXT
                )
                """
            )
            for action in ("UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS review_collaboration_no_{action.lower()} "
                    f"BEFORE {action} ON review_collaboration "
                    "BEGIN SELECT RAISE(ABORT, 'review_collaboration is append-only'); END"
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

    def set(
        self,
        resource_id: str,
        label: str,
        decision: str,
        value: str | None = None,
        inventory: pd.DataFrame | None = None,
        *,
        reviewer: str | None = None,
        rationale: str | None = None,
    ) -> dict:
        """Record a decision. An optional *rationale* (e.g. why a suggestion was overridden) is
        kept in the collaboration history, linked to this decision's feedback row; it is never
        part of the decision, the feedback row, plans or exports."""
        reviewer = _reviewer(reviewer)
        rationale = _text(rationale, "rationale", required=False)
        report = resource_report(resource_id, inventory)
        if report is None:
            raise KeyError(resource_id)
        if label not in LABELS:
            raise DecisionError(f"Unknown label '{label}'. Choose from {list(LABELS)}.")
        if decision not in DECISIONS:
            raise DecisionError(f"Unknown decision '{decision}'. Choose from {list(DECISIONS)}.")
        lr = report["labels"][label]
        if decision == "override":
            clean = normalise(
                label, value, inventory.attrs.get("label_config") if inventory is not None else None
            )
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
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reviewer": reviewer,
            **review_context(inventory),
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
                (resource_id, label, decision, value, suggested, previous, decided_at,
                 reviewer, inventory_id, model_version)
                VALUES (:resource_id, :label, :decision, :value, :suggested, :previous, :decided_at,
                        :reviewer, :inventory_id, :model_version)
                ON CONFLICT(resource_id, label) DO UPDATE SET
                    decision = excluded.decision,
                    value = excluded.value,
                    suggested = excluded.suggested,
                    previous = excluded.previous,
                    decided_at = excluded.decided_at,
                    reviewer = excluded.reviewer,
                    inventory_id = excluded.inventory_id,
                    model_version = excluded.model_version
                """,
                entry,
            )
            cursor = conn.execute(
                """
                INSERT INTO review_feedback
                (resource_id, label, decision, value, suggested, previous,
                 confidence, source, reason, decided_at, reviewer, inventory_id, model_version)
                VALUES (:resource_id, :label, :decision, :value, :suggested, :previous,
                        :confidence, :source, :reason, :decided_at,
                        :reviewer, :inventory_id, :model_version)
                """,
                feedback,
            )
            if rationale is not None:
                self._append(
                    conn,
                    resource_id,
                    label,
                    "rationale",
                    reviewer,
                    feedback_id=cursor.lastrowid,
                    text=rationale,
                )
        return entry

    def revoke(
        self,
        resource_id: str,
        label: str,
        *,
        reviewer: str | None = None,
        reason: str | None = None,
    ) -> dict | None:
        """Withdraw the current decision and record who did it, when, and what was withdrawn.

        Removing the decision and appending the 'revoke' event to review_feedback happen in one
        transaction. Returns the event, or None (writing nothing) when there is no decision to
        revoke. Invalid input raises DecisionError before anything is read or written.
        """
        reviewer = _reviewer(reviewer)
        if label not in LABELS:
            raise DecisionError(f"Unknown label '{label}'. Choose from {list(LABELS)}.")
        reason = (reason or "").strip() or None
        if reason is not None and len(reason) > MAX_REVOCATION_REASON:
            raise DecisionError(
                f"Keep the revocation reason to {MAX_REVOCATION_REASON} characters or fewer."
            )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM decisions WHERE resource_id = ? AND label = ?", (resource_id, label)
            ).fetchone()
            if current is None:
                return None
            # The feedback row that recorded the decision being revoked (absent for decisions
            # imported from legacy JSON files).
            origin = conn.execute(
                """
                SELECT id FROM review_feedback
                WHERE resource_id = ? AND label = ? AND decision = ? AND decided_at = ?
                  AND reviewer IS ? AND inventory_id IS ? AND model_version IS ?
                ORDER BY id DESC LIMIT 1
                """,
                (
                    resource_id,
                    label,
                    current["decision"],
                    current["decided_at"],
                    current["reviewer"],
                    current["inventory_id"],
                    current["model_version"],
                ),
            ).fetchone()
            event = {
                "resource_id": resource_id,
                "label": label,
                "decision": REVOKE,
                # The approval context being withdrawn.
                "value": current["value"],
                "suggested": current["suggested"],
                "previous": current["previous"],
                "inventory_id": current["inventory_id"],
                "model_version": current["model_version"],
                "revoked_decision": current["decision"],
                "revoked_reviewer": current["reviewer"],
                "revoked_decided_at": current["decided_at"],
                "revoked_feedback_id": origin["id"] if origin else None,
                # Who revoked it, when and (optionally) why.
                "reviewer": reviewer,
                "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "revocation_reason": reason,
            }
            conn.execute(
                "DELETE FROM decisions WHERE resource_id = ? AND label = ?", (resource_id, label)
            )
            cursor = conn.execute(
                """
                INSERT INTO review_feedback
                (resource_id, label, decision, value, suggested, previous, decided_at, reviewer,
                 inventory_id, model_version, revoked_decision, revoked_reviewer,
                 revoked_decided_at, revoked_feedback_id, revocation_reason)
                VALUES (:resource_id, :label, :decision, :value, :suggested, :previous,
                        :decided_at, :reviewer, :inventory_id, :model_version, :revoked_decision,
                        :revoked_reviewer, :revoked_decided_at, :revoked_feedback_id,
                        :revocation_reason)
                """,
                event,
            )
            row = conn.execute(
                "SELECT * FROM review_feedback WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return dict(row)

    def delete(
        self,
        resource_id: str,
        label: str,
        *,
        reviewer: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """Revoke the decision (see revoke()); True if there was one."""
        return self.revoke(resource_id, label, reviewer=reviewer, reason=reason) is not None

    # ── Collaboration: assignments, notes, unresolved questions (#46) ──────
    #
    # Every write appends one attributable row to review_collaboration (triggers refuse UPDATE
    # and DELETE); current state is derived from that history. None of it is read by
    # resolve(), pending_changes(), approved_change_plan() or export(), so it can never approve
    # or change an exported label.

    def _append(
        self,
        conn: sqlite3.Connection,
        resource_id: str,
        label: str,
        event: str,
        reviewer: str,
        *,
        assignee: str | None = None,
        question_id: int | None = None,
        feedback_id: int | None = None,
        text: str | None = None,
    ) -> dict:
        cursor = conn.execute(
            """
            INSERT INTO review_collaboration
            (resource_id, label, event, reviewer, created_at, assignee, question_id,
             feedback_id, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resource_id,
                label,
                event,
                reviewer,
                datetime.now(UTC).isoformat(timespec="seconds"),
                assignee,
                question_id,
                feedback_id,
                text,
            ),
        )
        row = conn.execute(
            "SELECT * FROM review_collaboration WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)

    def collaboration_history(
        self, resource_id: str | None = None, label: str | None = None
    ) -> list[dict]:
        """Append-only collaboration events, oldest first (optionally for one item)."""
        query, args = "SELECT * FROM review_collaboration", []
        if resource_id is not None:
            query += " WHERE resource_id = ? AND label = ?"
            args = [resource_id, label]
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query + " ORDER BY id", args)]

    def collaboration(
        self, *, assignee: str | None = None, unresolved: bool | None = None
    ) -> list[dict]:
        """Current collaboration state per (resource, label), filtered by assignee and
        whether the item has an unresolved question."""
        items = collaboration_state(self.collaboration_history()).values()
        return [
            item
            for item in items
            if (assignee is None or item["assignee"] == assignee)
            and (unresolved is None or bool(item["unresolved_questions"]) == unresolved)
        ]

    def collaboration_item(self, resource_id: str, label: str) -> dict:
        """State and history of one (resource, label); empty state when nothing is recorded."""
        history = self.collaboration_history(resource_id, label)
        state = collaboration_state(history).get((resource_id, label)) or _empty_item(
            resource_id, label
        )
        return state | {"history": history}

    def _target(self, resource_id: str, label: str, inventory: pd.DataFrame | None) -> None:
        if label not in LABELS:
            raise DecisionError(f"Unknown label '{label}'. Choose from {list(LABELS)}.")
        if resource_report(resource_id, inventory) is None:
            raise KeyError(resource_id)

    @staticmethod
    def _current_assignee(conn: sqlite3.Connection, resource_id: str, label: str) -> str | None:
        row = conn.execute(
            """
            SELECT event, assignee FROM review_collaboration
            WHERE resource_id = ? AND label = ? AND event IN ('assign', 'unassign')
            ORDER BY id DESC LIMIT 1
            """,
            (resource_id, label),
        ).fetchone()
        return row["assignee"] if row and row["event"] == "assign" else None

    def assign(
        self,
        resource_id: str,
        label: str,
        assignee: str,
        inventory: pd.DataFrame | None = None,
        *,
        reviewer: str | None = None,
    ) -> dict | None:
        """Assign (or reassign) a label to a reviewer ID. None, writing nothing, when it is
        already assigned to that reviewer. An assignment is never an approval."""
        reviewer = _reviewer(reviewer)
        assignee = _identity(assignee, "assignee")
        self._target(resource_id, label, inventory)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if self._current_assignee(conn, resource_id, label) == assignee:
                return None
            return self._append(conn, resource_id, label, "assign", reviewer, assignee=assignee)

    def unassign(self, resource_id: str, label: str, *, reviewer: str | None = None) -> dict | None:
        """Remove the assignment; None, writing nothing, when there is none."""
        reviewer = _reviewer(reviewer)
        if label not in LABELS:
            raise DecisionError(f"Unknown label '{label}'. Choose from {list(LABELS)}.")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._current_assignee(conn, resource_id, label)
            if current is None:
                return None
            return self._append(conn, resource_id, label, "unassign", reviewer, assignee=current)

    def add_note(
        self,
        resource_id: str,
        label: str,
        text: str,
        inventory: pd.DataFrame | None = None,
        *,
        reviewer: str | None = None,
    ) -> dict:
        """Append a short free-text review note (history only; never exported)."""
        reviewer = _reviewer(reviewer)
        text = _text(text, "note")
        self._target(resource_id, label, inventory)
        with self._connect() as conn:
            return self._append(conn, resource_id, label, "note", reviewer, text=text)

    def ask_question(
        self,
        resource_id: str,
        label: str,
        text: str,
        inventory: pd.DataFrame | None = None,
        *,
        reviewer: str | None = None,
    ) -> dict:
        """Open an unresolved question; the returned event's id is the question ID."""
        reviewer = _reviewer(reviewer)
        text = _text(text, "question")
        self._target(resource_id, label, inventory)
        with self._connect() as conn:
            return self._append(conn, resource_id, label, "question", reviewer, text=text)

    def _question_event(
        self,
        question_id: int,
        event: str,
        text: str | None,
        reviewer: str,
        resource_id: str | None,
        label: str | None,
    ) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            question = conn.execute(
                "SELECT * FROM review_collaboration WHERE id = ? AND event = 'question'",
                (question_id,),
            ).fetchone()
            if question is None or (
                resource_id is not None
                and (question["resource_id"], question["label"]) != (resource_id, label)
            ):
                raise KeyError(question_id)
            resolved = conn.execute(
                "SELECT 1 FROM review_collaboration WHERE question_id = ? AND event = 'resolve'",
                (question_id,),
            ).fetchone()
            if resolved:
                raise DecisionError(f"Question {question_id} is already resolved.")
            return self._append(
                conn,
                question["resource_id"],
                question["label"],
                event,
                reviewer,
                question_id=question_id,
                text=text,
            )

    def answer_question(
        self,
        question_id: int,
        text: str,
        *,
        reviewer: str | None = None,
        resource_id: str | None = None,
        label: str | None = None,
    ) -> dict:
        """Answer an open question (it stays unresolved until resolved). KeyError for an
        unknown question (or one on another item when resource_id/label are given)."""
        reviewer = _reviewer(reviewer)
        text = _text(text, "answer")
        return self._question_event(question_id, "answer", text, reviewer, resource_id, label)

    def resolve_question(
        self,
        question_id: int,
        text: str | None = None,
        *,
        reviewer: str | None = None,
        resource_id: str | None = None,
        label: str | None = None,
    ) -> dict:
        """Mark a question resolved, with an optional resolution note. Resolving is not an
        approval: the label still needs a decision."""
        reviewer = _reviewer(reviewer)
        text = _text(text, "resolution", required=False)
        return self._question_event(question_id, "resolve", text, reviewer, resource_id, label)


def _identity(value: str | None, what: str) -> str:
    value = (value or "").strip()
    if not value or len(value) > 128:
        raise DecisionError(f"Provide the {what} name or ID (1–128 characters).")
    return value


def _reviewer(reviewer: str | None) -> str:
    """The self-reported reviewer ID for a write, falling back to OBS_AGENT_REVIEWER."""
    reviewer = (
        reviewer if reviewer is not None else os.environ.get("OBS_AGENT_REVIEWER", "")
    ).strip()
    if not reviewer or len(reviewer) > 128:
        raise DecisionError("Provide a reviewer name or ID (1–128 characters).")
    return reviewer


def _text(value: str | None, what: str, *, required: bool = True) -> str | None:
    """Validated free text. Error messages never quote it (they can reach logs and clients)."""
    value = (value or "").strip()
    if not value:
        if required:
            raise DecisionError(f"Provide the {what} text.")
        return None
    if len(value) > MAX_NOTE:
        raise DecisionError(f"Keep the {what} to {MAX_NOTE} characters or fewer.")
    return value


def _empty_item(resource_id: str, label: str) -> dict:
    return {
        "resource_id": resource_id,
        "label": label,
        "assignee": None,
        "assigned_by": None,
        "assigned_at": None,
        "notes": [],
        "questions": [],
        "unresolved_questions": 0,
    }


def collaboration_state(history: list[dict]) -> dict[tuple[str, str], dict]:
    """Fold collaboration events into the current state per (resource_id, label)."""
    items: dict[tuple[str, str], dict] = {}
    questions: dict[int, dict] = {}
    for e in history:
        key = (e["resource_id"], e["label"])
        item = items.setdefault(key, _empty_item(*key))
        kind = e["event"]
        by = {"reviewer": e["reviewer"], "created_at": e["created_at"]}
        if kind == "assign":
            item |= {
                "assignee": e["assignee"],
                "assigned_by": e["reviewer"],
                "assigned_at": e["created_at"],
            }
        elif kind == "unassign":
            item |= {"assignee": None, "assigned_by": None, "assigned_at": None}
        elif kind in ("note", "rationale"):
            item["notes"].append(
                {"id": e["id"], "kind": kind, "text": e["text"], "feedback_id": e["feedback_id"]}
                | by
            )
        elif kind == "question":
            questions[e["id"]] = {
                "id": e["id"],
                "text": e["text"],
                "status": "open",
                "asked_by": e["reviewer"],
                "asked_at": e["created_at"],
                "answers": [],
                "resolved_by": None,
                "resolved_at": None,
                "resolution": None,
            }
            item["questions"].append(questions[e["id"]])
        elif kind == "answer" and e["question_id"] in questions:
            question = questions[e["question_id"]]
            question["answers"].append({"text": e["text"]} | by)
            question["status"] = "answered"
        elif kind == "resolve" and e["question_id"] in questions:
            questions[e["question_id"]] |= {
                "status": "resolved",
                "resolved_by": e["reviewer"],
                "resolved_at": e["created_at"],
                "resolution": e["text"],
            }
    for item in items.values():
        item["unresolved_questions"] = sum(q["status"] != "resolved" for q in item["questions"])
    return items


def with_collaboration(rows: list[dict], store: DecisionStore) -> list[dict]:
    """Queue rows (copies) with a per-label summary of assignments and unresolved questions.

    Counts only: note and question text stay in the collaboration routes and history.
    """
    state = collaboration_state(store.collaboration_history())
    out = []
    for r in rows:
        summary = {}
        for label in LABELS:
            item = state.get((r["resource_id"], label))
            if item:
                summary[label] = {
                    "assignee": item["assignee"],
                    "notes": len(item["notes"]),
                    "unresolved_questions": item["unresolved_questions"],
                }
        out.append(r | {"collaboration": summary})
    return out


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


# ── Post-review quality ───────────────────────────────────────────────


def _resolved_feedback_value(entry: dict) -> str | None:
    """Reviewer-selected label value, when the decision establishes one."""
    if entry["decision"] in ("accept", "override"):
        return entry["value"]
    if entry["decision"] == "reject":
        return entry["previous"]
    return None


def _agreement_scores(rows: list[dict]) -> dict:
    """Macro precision/recall against reviewer-resolved labels."""
    if not rows:
        return {"scored": 0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0}
    values = sorted({r["suggested"] for r in rows} | {r["resolved"] for r in rows})
    precision, recall, f1 = [], [], []
    for value in values:
        tp = sum(r["suggested"] == value and r["resolved"] == value for r in rows)
        predicted = sum(r["suggested"] == value for r in rows)
        actual = sum(r["resolved"] == value for r in rows)
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "scored": len(rows),
        "precision": round(sum(precision) / len(precision), 3),
        "recall": round(sum(recall) / len(recall), 3),
        "macro_f1": round(sum(f1) / len(f1), 3),
    }


def feedback_metrics(feedback: list[dict]) -> dict:
    """Summarise model agreement with reviewer decisions, overall and by month.

    An accepted or overridden value is the reviewer's resolved label. A rejected
    suggestion can be scored only when the resource already had a usable label;
    rejecting an untagged suggestion supplies no replacement truth. Revocation events are
    counted ('revoked') but never scored or counted as reviewed.
    """
    # A revocation withdraws an approval; it is not a reviewer-resolved label.
    revocations = [entry for entry in feedback if entry["decision"] == REVOKE]
    feedback = [entry for entry in feedback if entry["decision"] != REVOKE]
    result = {
        "reviewed": len(feedback),
        "revoked": len(revocations),
        "scored": 0,
        "unresolved": 0,
        "abstained": 0,
        "auto_apply_threshold": AUTO_APPLY_THRESHOLD,
        "labels": {},
    }
    for label in LABELS:
        entries = [entry for entry in feedback if entry["label"] == label]
        rows = [
            {
                "suggested": entry["suggested"],
                "resolved": _resolved_feedback_value(entry),
                "period": entry["decided_at"][:7],
                "confidence": entry.get("confidence"),
            }
            for entry in entries
            if entry["suggested"] and _resolved_feedback_value(entry)
        ]
        summary = _agreement_scores(rows)
        # Overrides of a withheld suggestion resolve the label but score nothing.
        abstained = sum(
            1 for entry in entries if not entry["suggested"] and entry.get("source") == "abstain"
        )
        summary["reviewed"] = len(entries)
        summary["revoked"] = sum(entry["label"] == label for entry in revocations)
        summary["abstained"] = abstained
        summary["unresolved"] = len(entries) - len(rows) - abstained
        periods = []
        for period in sorted({row["period"] for row in rows}):
            period_rows = [row for row in rows if row["period"] == period]
            periods.append({"period": period} | _agreement_scores(period_rows))
        summary["periods"] = periods
        buckets = []
        for name, lower, upper in (
            (f"below {AUTO_APPLY_THRESHOLD:.0%}", 0.0, AUTO_APPLY_THRESHOLD),
            (f"at or above {AUTO_APPLY_THRESHOLD:.0%}", AUTO_APPLY_THRESHOLD, 1.01),
        ):
            bucket_rows = [
                row
                for row in rows
                if row["confidence"] is not None and lower <= row["confidence"] < upper
            ]
            buckets.append({"bucket": name} | _agreement_scores(bucket_rows))
        summary["confidence_buckets"] = buckets
        result["labels"][label] = summary
        result["scored"] += summary["scored"]
        result["unresolved"] += summary["unresolved"]
        result["abstained"] += summary["abstained"]
    return result


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


def current_decisions(store: DecisionStore, inventory: pd.DataFrame | None = None) -> list[dict]:
    """Only approvals bound to the snapshot and inference currently being reviewed."""
    context = review_context(inventory)
    return [d for d in store.all() if all(d.get(k) == v for k, v in context.items())]


def pending_changes(
    store: DecisionStore, include_auto: bool = False, inventory: pd.DataFrame | None = None
) -> list[dict]:
    """Resources whose resolved tags differ from what they carry today."""
    inf = get_inference() if inventory is None else _review_data(inventory)[0]
    decisions = {f"{d['resource_id']}|{d['label']}": d for d in current_decisions(store, inventory)}
    changes = []
    for r in reports(inventory):
        final = resolve(r, decisions, include_auto)
        tags = {}
        for label in LABELS:
            value, how = final[label]
            if value is not None and value != r["labels"][label]["current"]:
                tags[label] = (value, how)
        if "team" in tags and tags["team"][0] in inf.cost_center_of:
            # Only export a cost center when this inventory establishes a mapping.
            tags["cost_center"] = (inf.cost_center_of[tags["team"][0]], tags["team"][1])
        if tags:
            tags = {
                label: (_export_value(r["cloud"], r["resource_id"], label, value), how)
                for label, (value, how) in tags.items()
            }
            changes.append(
                {
                    "resource_id": r["resource_id"],
                    "cloud": r["cloud"],
                    "tags": tags,
                    "previous": {lbl: r["labels"][lbl]["raw"] for lbl in LABELS}
                    | {"cost_center": r["cost_center"]["current"]},
                }
            )
    return changes


def _export_value(cloud: str, resource_id: str, label: str, value: str) -> str:
    value = str(value)
    if cloud == "gcp":
        value = value.lower()
        valid = re.fullmatch(r"[\w-]{0,63}", value) is not None
        requirement = "GCP values must be at most 63 letters, numbers, underscores or hyphens"
    else:
        valid = len(value) <= 256 and "\0" not in value
        requirement = "Azure values must be at most 256 characters and contain no NUL characters"
    if not valid:
        raise DecisionError(
            f"{resource_id}: invalid {label}; {requirement}. "
            "Correct the inventory and import again."
        )
    return value


def approved_change_plan(store: DecisionStore, inventory: pd.DataFrame | None = None) -> dict:
    """An auditable, reviewed-only change set for approval before application."""
    decisions = {f"{d['resource_id']}|{d['label']}": d for d in current_decisions(store, inventory)}
    # Approval evidence is the latest decision event per label; revocation events never count
    # (a revoked decision is also absent from `decisions`, so it cannot reach the plan).
    feedback = {
        f"{d['resource_id']}|{d['label']}": {
            k: v for k, v in d.items() if k not in REVOCATION_COLUMNS
        }
        for d in store.feedback()
        if d["decision"] != REVOKE
    }
    changes = pending_changes(store, include_auto=False, inventory=inventory)
    approvals = []
    for change in changes:
        for label in change["tags"]:
            decision_label = "team" if label == "cost_center" else label
            key = f"{change['resource_id']}|{decision_label}"
            if key in decisions and key in feedback and feedback[key] not in approvals:
                approvals.append(feedback[key])
    return {
        "schema_version": 2,
        **review_context(inventory),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reviewed_only": True,
        "changes": [
            {
                "resource_id": change["resource_id"],
                "cloud": change["cloud"],
                "tags": {
                    label: {"previous": change["previous"][label], "new": value, "how": how}
                    for label, (value, how) in change["tags"].items()
                },
            }
            for change in changes
        ],
        "approvals": approvals,
    }


def export(
    store: DecisionStore,
    fmt: str,
    include_auto: bool = False,
    inventory: pd.DataFrame | None = None,
) -> str:
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {list(EXPORT_FORMATS)}.")
    if fmt == "plan":
        return json.dumps(approved_change_plan(store, inventory), indent=2) + "\n"
    changes = pending_changes(store, include_auto, inventory)
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
            group = "subscription_tags" if c["cloud"] == "azure" else "project_labels"
            doc[group][c["resource_id"]] = {k: v for k, (v, _) in c["tags"].items()}
        return json.dumps(doc, indent=2) + "\n"
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by obs-agent label review, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.",
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
            labels = ",".join(f"{k}={v}" for k, (v, _) in c["tags"].items())
            lines.append(
                f"gcloud projects update {shlex.quote(rid)} --update-labels={shlex.quote(labels)}"
            )
    return "\n".join(lines) + "\n"
