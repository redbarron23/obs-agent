"""Snapshot reuse must save fits without reusing mutable reports or approvals."""

import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

import label_review as lr
import review_context
from inventory_import import load_inventory_csv
from review_workspaces import load_review, save_review

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()


@pytest.fixture
def fitted(monkeypatch):
    monkeypatch.setattr(lr, "_review_cache", OrderedDict(), raising=False)
    fits = []
    original = Pipeline.fit

    def count_fit(self, *args, **kwargs):
        fits.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", count_fit)
    return fits


def test_consumers_and_decisions_reuse_snapshot_but_refresh_approvals(fitted, tmp_path):
    inventory = load_inventory_csv(SAMPLE)
    store = lr.DecisionStore(tmp_path / "review.db")
    lr.reports(inventory)
    initial_fits = len(fitted)
    assert initial_fits > 0

    lr.coverage(inventory)
    lr.review_queue(inventory=inventory.copy())
    assert lr.export(store, "plan", inventory=inventory)
    assert not lr.pending_changes(store, inventory=inventory)
    for label in ("team", "env"):
        store.set("gcp-demo-003", label, "accept", inventory=inventory)
    for fmt in lr.EXPORT_FORMATS:
        assert "gcp-demo-003" in lr.export(store, fmt, inventory=inventory)
    plan = lr.approved_change_plan(store, inventory)
    assert {a["label"] for a in plan["approvals"]} == {"team", "env"}
    assert plan["inventory_id"] == review_context.inventory_id(inventory)
    assert len(fitted) == initial_fits

    # Another reviewer's decisions are never part of the snapshot cache.
    other = lr.DecisionStore(tmp_path / "other.db")
    assert not lr.pending_changes(other, inventory=inventory)
    store.delete("gcp-demo-003", "team")
    assert set(lr.approved_change_plan(store, inventory)["changes"][0]["tags"]) == {"env"}
    assert len(fitted) == initial_fits


@pytest.mark.parametrize("change", ["snapshot", "vocabulary", "version"])
def test_changed_context_refits_and_excludes_old_approvals(fitted, tmp_path, monkeypatch, change):
    inventory = load_inventory_csv(SAMPLE)
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=inventory)
    first = lr.resource_report("gcp-demo-003", inventory)
    initial_fits = len(fitted)
    if change == "snapshot":
        inventory.loc[inventory.resource_id == "gcp-demo-003", "parent"] = "new-private-parent"
    elif change == "vocabulary":
        inventory.attrs["label_config"]["team"]["aliases"]["new-alias"] = "payments"
    else:
        monkeypatch.setattr(review_context, "_implementation_id", lambda: "new-version")

    current = lr.resource_report("gcp-demo-003", inventory)
    assert len(fitted) > initial_fits
    assert lr.current_decisions(store, inventory) == []
    assert not lr.pending_changes(store, inventory=inventory)
    if change == "snapshot":
        assert first["parent"] != current["parent"] == "new-private-parent"


def test_returned_reports_do_not_mutate_cached_snapshot(fitted):
    inventory = load_inventory_csv(SAMPLE)
    first = lr.resource_report("gcp-demo-003", inventory)
    first["owners"].append("injected-owner")
    first["labels"]["team"]["suggested"] = "injected-team"
    again = lr.resource_report("gcp-demo-003", inventory.copy())
    assert "injected-owner" not in again["owners"]
    assert again["labels"]["team"]["suggested"] != "injected-team"


def test_imported_page_fits_once_and_reuses_across_reruns(fitted, monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    review_id = save_review(SAMPLE, "private.csv")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["review_id"] = review_id
    at.run()
    assert not at.exception
    # Two fitted label models plus two out-of-fold fits per label on this fixture.
    assert len(fitted) == 6
    inventory, store = load_review(review_id)
    store.set("gcp-demo-003", "team", "accept", inventory=inventory)
    at.run()
    assert not at.exception
    assert len(fitted) == 6
    assert {m.label: m.value for m in at.metric}["Reviewed"] == "1"
    assert json.loads(lr.export(store, "plan", inventory=inventory))["changes"]


def test_concurrent_consumers_fit_the_same_snapshot_once(fitted):
    inventory = load_inventory_csv(SAMPLE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lr.reports, [inventory.copy() for _ in range(4)]))
    assert all(result == results[0] for result in results)
    assert len(fitted) == 6


def test_cache_evicts_least_recently_used_snapshot(fitted):
    inventories = [
        load_inventory_csv(SAMPLE.replace(b"folders-data", f"private-folder-{i}".encode()))
        for i in range(5)
    ]
    for inventory in inventories[:4]:
        lr.reports(inventory)
    lr.reports(inventories[0])
    lr.reports(inventories[4])
    initial_fits = len(fitted)
    lr.reports(inventories[0])
    assert len(fitted) == initial_fits
    lr.reports(inventories[1])
    assert len(fitted) > initial_fits
    assert len(lr._review_cache) == 4
