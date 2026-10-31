"""Workflow metrics, correlated error logs and the review backlog (workflow_metrics.py)."""

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api
import label_review as lr
import workflow_metrics as wm
from inventory_import import load_inventory_csv
from review_workspaces import load_review, save_review

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "fixtures" / "sanitized_inventory_sample.csv"
REVIEWER = "alice.reviewer@example.com"
FREE_TEXT = "Secret-Free-Text-Override"


@pytest.fixture(autouse=True)
def _reset():
    wm.reset()
    yield
    wm.reset()


@pytest.fixture
def store(tmp_path):
    return lr.DecisionStore(tmp_path / "decisions.db")


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
    monkeypatch.setattr(api, "decisions", store)
    return TestClient(api.app, headers={"X-API-Key": "test-api-key"})


def _suggested_team_resource() -> str:
    return next(
        r["resource_id"]
        for r in lr.review_queue(label="team", status="untagged")
        if r["labels"]["team"]["suggested"]
    )


def _counts() -> dict:
    return {
        (r["operation"], r["surface"], r["outcome"]): r["count"]
        for r in wm.snapshot()["operations"]
    }


def _events(caplog) -> list[dict]:
    return [
        r.workflow
        for r in caplog.records
        if r.name == "obs_agent.workflow" and hasattr(r, "workflow")
    ]


class TestOperation:
    def test_success_invalid_and_error_outcomes(self):
        with wm.operation("export", surface="ui"):
            pass
        with (
            pytest.raises(lr.DecisionError),
            wm.operation("export", surface="ui", invalid=(lr.DecisionError,)),
        ):
            raise lr.DecisionError("bad")
        with pytest.raises(sqlite3.OperationalError), wm.operation("export", surface="ui"):
            raise sqlite3.OperationalError("disk I/O error")
        assert _counts() == {
            ("export", "ui", "success"): 1,
            ("export", "ui", "invalid"): 1,
            ("export", "ui", "error"): 1,
        }
        (durations,) = wm.snapshot()["durations"]
        assert durations["count"] == 3

    def test_labels_are_bounded(self):
        with pytest.raises(ValueError):
            wm.operation("az-prod-0001")
        op = wm.operation("import", surface="sub-1234")
        assert op.surface == "other"

    def test_body_exception_propagates_unchanged(self):
        boom = RuntimeError("caller failure")
        with pytest.raises(RuntimeError) as info, wm.operation("inference"):
            raise boom
        assert info.value is boom

    def test_telemetry_failure_never_fails_the_operation(self, monkeypatch):
        def broken(*args):
            raise RuntimeError("metrics store down")

        monkeypatch.setattr(wm, "_record", broken)
        with wm.operation("export") as op:
            result = "exported"
        assert result == "exported" and op.outcome == "success"
        assert wm.snapshot()["telemetry_errors"] == 1

    def test_error_log_has_type_and_correlation_but_not_message(self, caplog):
        caplog.set_level(logging.INFO, logger="obs_agent.workflow")
        op = wm.operation("decision_write", surface="api")
        with pytest.raises(sqlite3.OperationalError), op:
            raise sqlite3.OperationalError(f"locked while writing {FREE_TEXT}")
        (event,) = _events(caplog)
        assert event == event | {
            "operation": "decision_write",
            "outcome": "error",
            "error_type": "OperationalError",
            "correlation_id": op.correlation_id,
        }
        assert FREE_TEXT not in caplog.text

    def test_correlation_id_sanitised(self):
        assert wm.clean_correlation_id("req-123_abc") == "req-123_abc"
        for unsafe in ("", "a b", "x" * 65, "id\nINJECTED"):
            assert wm.clean_correlation_id(unsafe) != unsafe


class TestInference:
    def test_imported_inference_success_is_timed(self, monkeypatch):
        monkeypatch.setattr(lr, "_review_cache", type(lr._review_cache)())
        inventory = load_inventory_csv(SAMPLE)
        lr.reports(inventory)
        assert _counts()[("inference", "other", "success")] == 1
        lr.reports(inventory)  # cache hit: no second inference
        assert _counts()[("inference", "other", "success")] == 1

    def test_inference_failure_is_counted_and_raised(self, monkeypatch):
        class Broken:
            def __init__(self, inventory):
                raise MemoryError("fit failed")

        monkeypatch.setattr(lr, "LabelInference", Broken)
        monkeypatch.setattr(lr, "_review_cache", type(lr._review_cache)())
        inventory = load_inventory_csv(SAMPLE)
        with pytest.raises(MemoryError):
            lr.reports(inventory)
        assert _counts() == {("inference", "other", "error"): 1}


class TestApi:
    def test_request_id_header_echoed_or_generated(self, client):
        assert (
            client.get("/labels/coverage", headers={"X-Request-ID": "trace-42"}).headers[
                "X-Request-ID"
            ]
            == "trace-42"
        )
        generated = client.get("/labels/coverage", headers={"X-Request-ID": "bad id\r"})
        assert generated.headers["X-Request-ID"] != "bad id\r"

    def test_decision_write_and_export_success(self, client):
        rid = _suggested_team_resource()
        r = client.put(f"/resources/{rid}/labels/team/decision", json={"decision": "accept"})
        assert r.status_code == 200
        assert client.delete(f"/resources/{rid}/labels/team/decision").status_code == 200
        assert client.get("/labels/export?format=csv").status_code == 200
        counts = _counts()
        assert counts[("decision_write", "api", "success")] == 2
        assert counts[("export", "api", "success")] == 1

    def test_invalid_decision_is_correlated_without_free_text(self, client, caplog):
        caplog.set_level(logging.INFO, logger="obs_agent.workflow")
        rid = _suggested_team_resource()
        r = client.put(
            f"/resources/{rid}/labels/team/decision",
            json={"decision": "override", "value": FREE_TEXT, "reviewer": REVIEWER},
            headers={"X-Request-ID": "req-invalid-1"},
        )
        assert r.status_code == 422
        (event,) = [e for e in _events(caplog) if e["operation"] == "decision_write"]
        assert event["outcome"] == "invalid" and event["correlation_id"] == "req-invalid-1"
        assert _counts()[("decision_write", "api", "invalid")] == 1
        for leaked in (FREE_TEXT, REVIEWER, rid):
            assert leaked not in caplog.text

    def test_decision_write_failure_returns_correlated_500(
        self, client, store, monkeypatch, caplog
    ):
        caplog.set_level(logging.INFO, logger="obs_agent.workflow")
        rid = _suggested_team_resource()

        def locked(*args, **kwargs):
            raise sqlite3.OperationalError(f"database is locked: {rid} {REVIEWER}")

        monkeypatch.setattr(store, "set", locked)
        r = client.put(
            f"/resources/{rid}/labels/team/decision",
            json={"decision": "accept", "reviewer": REVIEWER},
        )
        assert r.status_code == 500
        correlation_id = r.headers["X-Request-ID"]
        assert r.json() == {"detail": "Internal server error", "correlation_id": correlation_id}
        events = _events(caplog)
        assert {e["event"] for e in events} == {"workflow_operation", "request_failed"}
        assert all(e["correlation_id"] == correlation_id for e in events)
        assert all(e["error_type"] == "OperationalError" for e in events)
        assert _counts()[("decision_write", "api", "error")] == 1
        for leaked in (rid, REVIEWER, "database is locked"):
            assert leaked not in caplog.text

    def test_export_failure_counted(self, client, monkeypatch):
        def broken(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(lr, "export", broken)
        assert client.get("/labels/export?format=sh").status_code == 500
        assert _counts()[("export", "api", "error")] == 1

    def test_telemetry_failure_does_not_block_review_or_export(self, client, monkeypatch):
        monkeypatch.setattr(wm, "_emit", lambda *a: (_ for _ in ()).throw(RuntimeError("log down")))
        rid = _suggested_team_resource()
        r = client.put(f"/resources/{rid}/labels/team/decision", json={"decision": "accept"})
        assert r.status_code == 200
        r = client.get("/labels/export?format=csv")
        assert r.status_code == 200 and rid in r.text
        assert wm.snapshot()["telemetry_errors"] >= 2

    def test_metrics_endpoint_json_and_prometheus_exclude_sensitive_data(self, client):
        rid = _suggested_team_resource()
        client.put(
            f"/resources/{rid}/labels/team/decision",
            json={"decision": "accept", "reviewer": REVIEWER},
        )
        client.put(
            f"/resources/{rid}/labels/env/decision",
            json={"decision": "override", "value": FREE_TEXT, "reviewer": REVIEWER},
        )
        body = client.get("/ops/metrics").json()
        assert body["backlog"]["reviewed_items"] == 1
        assert body["tracing"]["active"] is False
        text = client.get("/ops/metrics?format=prometheus").text
        assert (
            "obs_agent_workflow_operations_total"
            '{operation="decision_write",surface="api",outcome="success"} 1' in text
        )
        assert "obs_agent_review_backlog_pending_items " in text
        inventory_id = lr.review_context(None)["inventory_id"]
        for leaked in (rid, REVIEWER, FREE_TEXT, inventory_id):
            assert leaked not in json.dumps(body) and leaked not in text

    def test_metrics_stay_readable_when_backlog_fails(self, client, monkeypatch):
        monkeypatch.setattr(wm, "backlog", lambda *a, **k: 1 / 0)
        r = client.get("/ops/metrics")
        assert r.status_code == 200 and r.json()["backlog"] is None


class TestBacklog:
    def test_pending_work_vs_historical_feedback(self, store):
        before = wm.backlog(store)
        assert before["pending_items"] > 0 and before["feedback_actions"] == 0
        assert before["oldest_pending_age_seconds"] is None  # no review start recorded yet
        rid = _suggested_team_resource()
        store.set(rid, "team", "accept")
        after = wm.backlog(store)
        assert after["pending_items"] == before["pending_items"] - 1
        assert after["reviewed_items"] == 1 and after["feedback_actions"] == 1
        store.delete(rid, "team")  # undo: pending again, history kept
        undone = wm.backlog(store)
        assert undone["pending_items"] == before["pending_items"]
        assert undone["reviewed_items"] == 0 and undone["feedback_actions"] == 1

    def test_abstained_labels_are_pending(self, store):
        b = wm.backlog(store)
        abstained = sum(
            r["labels"][label]["action"] == lr.INSUFFICIENT_EVIDENCE
            for r in lr.reports()
            for label in lr.LABELS
        )
        assert b["pending_insufficient_evidence"] == abstained

    def test_age_from_saved_review_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        inventory, store = load_review(save_review(SAMPLE.read_bytes(), "sample.csv"))
        created = wm.backlog(store, inventory)["review_started_at"]
        later = datetime.fromisoformat(created) + timedelta(hours=3)
        b = wm.backlog(store, inventory, now=later)
        assert b["oldest_pending_age_seconds"] == 3 * 3600
        assert b["last_action_age_seconds"] is None

    def test_stale_decisions_are_not_reviewed_work(self, store):
        rid = _suggested_team_resource()
        store.set(rid, "team", "accept")
        with store._connect() as conn:
            conn.execute("UPDATE decisions SET inventory_id = 'older-snapshot'")
        b = wm.backlog(store, now=datetime.now(UTC))
        assert b["reviewed_items"] == 0 and b["stale_decisions"] == 1


class TestPages:
    def test_import_page_counts_invalid_and_successful_imports(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        at.switch_page("pages/2_Import_inventory.py").run()
        header, first, *rest = SAMPLE.read_bytes().split(b"\n")
        broken = b"\n".join([header, first.replace(b"azure", b"aws", 1), *rest])
        at.file_uploader[1].set_value(("bad.csv", broken, "text/csv")).run()
        assert at.error and not at.exception
        assert _counts()[("import", "ui", "invalid")] == 1
        at.file_uploader[1].set_value(("ok.csv", SAMPLE.read_bytes(), "text/csv")).run()
        next(b for b in at.button if b.label == "Save and open review").click().run()
        assert not at.exception
        assert _counts()[("import", "ui", "success")] >= 2  # validation + save

    def test_review_page_counts_export_preparation(self, store, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))
        at = AppTest.from_file(str(ROOT / "pages" / "1_Label_review.py"), default_timeout=60).run()
        assert not at.exception
        assert _counts()[("export", "ui", "success")] == 1

    def test_review_page_export_failure_shows_reference(self, store, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="obs_agent.workflow")
        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))

        def broken(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(lr, "export", broken)
        at = AppTest.from_file(str(ROOT / "pages" / "1_Label_review.py"), default_timeout=60).run()
        assert not at.exception
        (event,) = [e for e in _events(caplog) if e["operation"] == "export"]
        assert event["outcome"] == "error" and event["surface"] == "ui"
        assert any(event["correlation_id"] in e.value for e in at.error)
