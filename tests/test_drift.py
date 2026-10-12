"""Tests for inventory-snapshot drift detection."""

import pandas as pd
import pytest

from drift import detect_drift


def _snapshot(**changes) -> pd.DataFrame:
    row = {
        "resource_id": "sub-a",
        "parent": "mg-payments",
        "owners": ["alice"],
        "top_services": ["AKS", "SQL"],
        "avg_daily_cost": 100.0,
        "weekend_ratio": 0.8,
        "cost_cv": 0.1,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_detects_hierarchy_owner_service_and_cost_drift():
    previous = _snapshot()
    current = _snapshot(
        parent="mg-shared",
        owners=["bob"],
        top_services=["AKS", "Redis"],
        avg_daily_cost=175.0,
        weekend_ratio=0.5,
        cost_cv=0.4,
    )

    findings = detect_drift(previous, current)

    assert findings[0]["resource_id"] == "sub-a"
    assert {change["field"] for change in findings[0]["changes"]} == {
        "parent",
        "owners",
        "top_services",
        "avg_daily_cost",
        "weekend_ratio",
        "cost_cv",
    }


def test_ignores_changes_below_the_review_thresholds():
    previous = _snapshot()
    current = _snapshot(avg_daily_cost=149.0, weekend_ratio=0.61, cost_cv=0.29)

    assert detect_drift(previous, current) == []


def test_requires_snapshot_columns():
    with pytest.raises(ValueError, match="Snapshots require columns"):
        detect_drift(pd.DataFrame([{"resource_id": "sub-a"}]), _snapshot())
