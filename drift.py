"""Compare cloud-inventory snapshots and return resources that need review."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "resource_id",
    "parent",
    "owners",
    "top_services",
    "avg_daily_cost",
    "weekend_ratio",
    "cost_cv",
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
        if old_spend and abs(new_spend - old_spend) / old_spend >= spend_change:
            changes.append(
                {"field": "avg_daily_cost", "previous": old_spend, "current": new_spend}
            )
        for column in ("weekend_ratio", "cost_cv"):
            old_value, new_value = float(old[column]), float(new[column])
            if abs(new_value - old_value) >= shape_change:
                changes.append({"field": column, "previous": old_value, "current": new_value})

        if changes:
            findings.append({"resource_id": resource_id, "changes": changes})
    return findings
