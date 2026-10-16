"""Compare cloud-inventory snapshots and return resources that need review."""

from __future__ import annotations

import pandas as pd

from label_config import inventory_config
from label_review import reports
from labeling import LABELS

REQUIRED_COLUMNS = {
    "resource_id",
    "parent",
    "owners",
    "top_services",
    "avg_daily_cost",
    "weekend_ratio",
    "cost_cv",
}
TAG_COLUMNS = (*LABELS, "cost_center")
REASONS = {
    "parent": "Moved in the hierarchy; ownership may have changed.",
    "owners": "Owners changed; the responsible team may have changed.",
    "top_services": "Service footprint changed; the workload may have changed.",
    "avg_daily_cost": "Spend changed materially; usage or ownership may have changed.",
    "weekend_ratio": "Cost shape changed; the environment may have changed.",
    "cost_cv": "Cost volatility changed; the environment may have changed.",
}


def detect_drift(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    spend_change: float = 0.5,
    shape_change: float = 0.2,
) -> list[dict]:
    """Return changed resources and the evidence that warrants a new review.

    The caller owns snapshot collection. New and deleted resources are ignored:
    this helper only detects changes to resources that exist in both snapshots.
    Any zero-to-positive spend change warrants review; positive baselines use spend_change.
    """
    missing = (REQUIRED_COLUMNS - set(previous.columns)) | (REQUIRED_COLUMNS - set(current.columns))
    if missing:
        raise ValueError(f"Snapshots require columns: {sorted(missing)}")

    before = previous.set_index("resource_id", drop=False)
    after = current.set_index("resource_id", drop=False)
    findings = []
    for resource_id in sorted(before.index.intersection(after.index)):
        old, new = before.loc[resource_id], after.loc[resource_id]
        changes = []
        for column in ("parent", "owners", "top_services"):
            old_value = sorted(old[column]) if isinstance(old[column], list) else old[column]
            new_value = sorted(new[column]) if isinstance(new[column], list) else new[column]
            if old_value != new_value:
                changes.append({"field": column, "previous": old_value, "current": new_value})

        old_spend, new_spend = float(old["avg_daily_cost"]), float(new["avg_daily_cost"])
        if (old_spend == 0 and new_spend > 0) or (
            old_spend != 0 and abs(new_spend - old_spend) / old_spend >= spend_change
        ):
            changes.append({"field": "avg_daily_cost", "previous": old_spend, "current": new_spend})
        for column in ("weekend_ratio", "cost_cv"):
            old_value, new_value = float(old[column]), float(new[column])
            if abs(new_value - old_value) >= shape_change:
                changes.append({"field": column, "previous": old_value, "current": new_value})

        if changes:
            findings.append({"resource_id": resource_id, "changes": changes})
    return findings


def _tag(value) -> str | None:
    return None if pd.isna(value) else str(value)


def _summary(row) -> dict:
    return {
        "resource_id": row["resource_id"],
        "cloud": row["cloud"],
        "parent": row["parent"],
        "owners": list(row["owners"]),
        "tags": {column: _tag(row[column]) for column in TAG_COLUMNS},
    }


def compare_snapshots(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    previous_decisions: list[dict] | None = None,
    **thresholds,
) -> dict:
    """New, removed, and changed resources between two imported inventory snapshots.

    Changed resources carry before/after evidence and a reason for renewed review: ownership
    and cost-shape drift (from detect_drift), tag edits made outside the review, and changed
    label suggestions. Earlier decisions are reported for context only; they belong to the
    previous snapshot and are never copied into the current one.
    """
    old_vocab, new_vocab = inventory_config(previous), inventory_config(current)
    before = previous.set_index("resource_id", drop=False)
    after = current.set_index("resource_id", drop=False)
    old_reports = {r["resource_id"]: r for r in reports(previous)}
    new_reports = {r["resource_id"]: r for r in reports(current)}
    signals = {
        f["resource_id"]: f["changes"] for f in detect_drift(previous, current, **thresholds)
    }
    decisions: dict[str, list[dict]] = {}
    for d in previous_decisions or []:
        decisions.setdefault(d["resource_id"], []).append(
            {k: d.get(k) for k in ("label", "decision", "value", "reviewer", "decided_at")}
        )

    changed = []
    for resource_id in sorted(before.index.intersection(after.index)):
        changes = [c | {"reason": REASONS[c["field"]]} for c in signals.get(resource_id, [])]
        for column in TAG_COLUMNS:
            old, new = _tag(before.loc[resource_id, column]), _tag(after.loc[resource_id, column])
            if old != new:
                changes.append(
                    {
                        "field": column,
                        "previous": old,
                        "current": new,
                        "reason": "Tag changed outside this review; confirm the new value.",
                    }
                )
        for label in LABELS:
            old = old_reports[resource_id]["labels"][label]
            new = new_reports[resource_id]["labels"][label]
            if (old["status"], old["suggested"]) != (new["status"], new["suggested"]):
                changes.append(
                    {
                        "field": f"{label} suggestion",
                        "previous": f"{old['status']}: {old['suggested'] or '-'}",
                        "current": f"{new['status']}: {new['suggested'] or '-'}",
                        "reason": "The label suggestion changed; review it again.",
                    }
                )
        if changes:
            changed.append(
                {
                    "resource_id": resource_id,
                    "cloud": after.loc[resource_id, "cloud"],
                    "changes": changes,
                    "needs_attention": new_reports[resource_id]["needs_attention"],
                    "earlier_decisions": decisions.get(resource_id, []),
                }
            )

    added = [_summary(after.loc[i]) for i in sorted(after.index.difference(before.index))]
    removed = [
        _summary(before.loc[i]) | {"earlier_decisions": decisions.get(i, [])}
        for i in sorted(before.index.difference(after.index))
    ]
    for item in added:
        item["needs_attention"] = new_reports[item["resource_id"]]["needs_attention"]
    return {
        "vocabulary_changed": old_vocab != new_vocab,
        "vocabulary_differences": [
            label for label in LABELS if old_vocab[label] != new_vocab[label]
        ],
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": len(before.index.intersection(after.index)) - len(changed),
    }
