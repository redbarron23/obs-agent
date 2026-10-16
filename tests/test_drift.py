"""Tests for inventory-snapshot drift detection."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from drift import compare_snapshots, detect_drift
from inventory_import import load_inventory_csv
from label_config import load_label_config
from review_workspaces import compare_reviews, load_review, save_review


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


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (0, 1000, True),
        (0, 0.01, True),
        (0, 0, False),
        (100, 0, True),
        (100, 149, False),
        (100, 150, True),
        (100, 51, False),
        (100, 50, True),
    ],
)
def test_spend_drift_handles_zero_baselines_and_relative_thresholds(old, new, expected):
    findings = detect_drift(_snapshot(avg_daily_cost=old), _snapshot(avg_daily_cost=new))
    assert findings == (
        [
            {
                "resource_id": "sub-a",
                "changes": [{"field": "avg_daily_cost", "previous": old, "current": new}],
            }
        ]
        if expected
        else []
    )


# ── Saved-snapshot comparison ──────────────────────────────────────────

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
OLD_ROW = b"azure,az-old-001,mg-payments,alice,alice,AKS,80.00,0.90,0.20,payments,prod,CC-1001\n"
NEW_ROW = b"gcp,gcp-demo-004,folders-data,gus,gus,BigQuery,30.00,0.30,0.40,,,\n"
CURRENT = (
    (SAMPLE + NEW_ROW)
    .replace(b"fatima|gus,fatima,BigQuery|GCS,36.20", b"harry,fatima,BigQuery|GCS,36.20")
    .replace(b"0.92,0.18,payments,prod,CC-1001", b"0.92,0.18,payments,prod,CC-1009")
)


def test_compare_separates_new_removed_and_changed_resources():
    result = compare_snapshots(load_inventory_csv(SAMPLE + OLD_ROW), load_inventory_csv(CURRENT))

    assert [r["resource_id"] for r in result["added"]] == ["gcp-demo-004"]
    assert result["added"][0]["needs_attention"]
    assert [r["resource_id"] for r in result["removed"]] == ["az-old-001"]
    assert result["removed"][0]["tags"] == {
        "team": "payments",
        "env": "prod",
        "cost_center": "CC-1001",
    }
    changed = {r["resource_id"]: r["changes"] for r in result["changed"]}
    assert changed == {
        "az-demo-001": [
            {
                "field": "cost_center",
                "previous": "CC-1001",
                "current": "CC-1009",
                "reason": "Tag changed outside this review; confirm the new value.",
            }
        ],
        "gcp-demo-003": [
            {
                "field": "owners",
                "previous": ["fatima", "gus"],
                "current": ["harry"],
                "reason": "Owners changed; the responsible team may have changed.",
            }
        ],
    }
    assert result["unchanged"] == 3
    assert not result["vocabulary_changed"]


def test_compare_reports_no_drift_for_an_identical_snapshot():
    inventory = load_inventory_csv(SAMPLE)
    result = compare_snapshots(inventory, inventory.copy())
    assert (result["added"], result["removed"], result["changed"]) == ([], [], [])
    assert result["unchanged"] == len(inventory)


def test_saved_review_comparison_reports_new_spend_with_unchanged_suggestions(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    previous = save_review(SAMPLE.replace(b"120.50", b"0.00"), "before.csv")
    current = save_review(SAMPLE.replace(b"120.50", b"1000.00"), "after.csv")
    result = compare_reviews(previous, current)
    resource = next(r for r in result["changed"] if r["resource_id"] == "az-demo-001")
    assert resource["changes"] == [
        {
            "field": "avg_daily_cost",
            "previous": 0.0,
            "current": 1000.0,
            "reason": "Spend changed materially; usage or ownership may have changed.",
        }
    ]


def test_compare_reports_changed_suggestions_and_tags():
    tagged = SAMPLE.replace(
        b"BigQuery|GCS,36.20,0.30,0.42,,,", b"BigQuery|GCS,36.20,0.30,0.42,,dev,"
    )
    result = compare_snapshots(load_inventory_csv(SAMPLE), load_inventory_csv(tagged))
    fields = {c["field"]: c for c in result["changed"][0]["changes"]}
    assert set(fields) == {"env", "env suggestion"}
    assert fields["env suggestion"]["previous"].startswith("untagged: ")
    assert fields["env suggestion"]["current"] == "ok: -"


def test_compare_identifies_a_different_vocabulary():
    config = load_label_config(ROOT / "fixtures/label_config_example.json")
    result = compare_snapshots(load_inventory_csv(SAMPLE), load_inventory_csv(SAMPLE, config))
    assert result["vocabulary_changed"]
    assert result["vocabulary_differences"] == ["team", "env"]


def test_saved_review_comparison_never_transfers_approvals(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    previous_id = save_review(SAMPLE + OLD_ROW, "september.csv")
    current_id = save_review(CURRENT, "october.csv")
    previous, previous_store = load_review(previous_id)
    previous_store.set("gcp-demo-003", "team", "accept", inventory=previous, reviewer="alice")

    result = compare_reviews(previous_id, current_id)

    changed = next(r for r in result["changed"] if r["resource_id"] == "gcp-demo-003")
    assert changed["earlier_decisions"][0] | {"decided_at": None} == {
        "label": "team",
        "decision": "accept",
        "value": "data-platform",
        "reviewer": "alice",
        "decided_at": None,
    }
    assert load_review(current_id)[1].all() == []
    assert len(previous_store.all()) == 1
    with pytest.raises(ValueError, match="two different"):
        compare_reviews(current_id, current_id)


def test_compare_page_links_drift_into_the_current_review_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    monkeypatch.setenv("OBS_AGENT_DECISIONS", str(tmp_path / "decisions.db"))
    previous_id = save_review(SAMPLE + OLD_ROW + NEW_ROW, "september.csv")
    current_id = save_review(CURRENT, "october.csv")
    with sqlite3.connect(tmp_path / f"{previous_id}.db") as conn:
        conn.execute("UPDATE workspace SET created_at = '2026-09-01T00:00:00+00:00'")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page("pages/4_Compare_snapshots.py").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric} == {
        "New": "0",
        "Removed": "1",
        "Changed": "2",
        "Unchanged": "4",
    }
    next(b for b in at.button if b.label.startswith("Review 2 new")).click().run()
    assert at.session_state["review_id"] == current_id
    assert set(at.session_state["drift_filter"]["resource_ids"]) == {
        "az-demo-001",
        "gcp-demo-003",
    }

    at.switch_page("pages/1_Label_review.py").run()
    assert not at.exception
    assert at.sidebar.checkbox[-1].value  # drift filter on by default
    # gcp-demo-004 is untagged but unchanged, so the drift filter hides it.
    assert at.subheader[0].value == "1 resources to review for `team`"
    at.sidebar.checkbox[-1].uncheck().run()
    assert at.subheader[0].value == "2 resources to review for `team`"
