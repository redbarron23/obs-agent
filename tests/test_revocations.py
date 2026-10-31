"""
Decision revocations (#64): an undo is an attributable, append-only event.

Runs on the seeded synthetic inventory and the sanitized sample; every database
lives in a pytest temp directory.
"""

import json
import logging
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import label_review as lr
import plan_validation as pv
import real_evaluation
from inventory_import import load_inventory_csv
from label_review import DecisionError, DecisionStore

UNTAGGED = "svc-ingest"  # no team tag; team → data-platform (CC-2002)
INVENTORY_SAMPLE = Path(__file__).parent.parent / "fixtures" / "sanitized_inventory_sample.csv"
REASON = "owner confirmed this belongs to ml"


@pytest.fixture
def store(tmp_path):
    return DecisionStore(tmp_path / "decisions.db")


def _exports(store, inventory=None) -> dict[str, str]:
    return {fmt: lr.export(store, fmt, inventory=inventory) for fmt in lr.EXPORT_FORMATS}


class TestRevoke:
    def test_records_actor_time_reason_and_revoked_context(self, store):
        decision = store.set(UNTAGGED, "team", "accept", reviewer="alice")
        origin = store.feedback()[-1]

        event = store.revoke(UNTAGGED, "team", reviewer=" bob ", reason=f"  {REASON} ")

        assert store.get(UNTAGGED, "team") is None
        assert store.feedback()[-1] == event
        assert event["decision"] == lr.REVOKE
        assert event["reviewer"] == "bob"
        assert event["revocation_reason"] == REASON
        assert event["decided_at"] >= decision["decided_at"]
        assert event["revoked_decision"] == "accept"
        assert event["revoked_reviewer"] == "alice"
        assert event["revoked_decided_at"] == decision["decided_at"]
        assert event["revoked_feedback_id"] == origin["id"]
        for field in ("value", "suggested", "previous", "inventory_id", "model_version"):
            assert event[field] == decision[field]
        assert event["confidence"] is None and event["source"] is None

    def test_history_is_preserved_in_order(self, store):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        store.set(UNTAGGED, "team", "override", "ml", reviewer="alice")
        store.revoke(UNTAGGED, "team", reviewer="bob")
        store.set(UNTAGGED, "team", "accept", reviewer="carol")

        history = store.feedback()
        assert [(h["decision"], h["reviewer"]) for h in history] == [
            ("accept", "alice"),
            ("override", "alice"),
            ("revoke", "bob"),
            ("accept", "carol"),
        ]
        assert [h["id"] for h in history] == sorted(h["id"] for h in history)
        # The revocation withdrew the override, not the first accept.
        assert history[2]["revoked_decision"] == "override"
        assert history[2]["value"] == "ml"
        assert history[2]["revoked_feedback_id"] == history[1]["id"]
        assert history[2]["revocation_reason"] is None
        # Decision rows carry no revocation fields.
        assert all(h["revoked_decision"] is None for h in history if h["decision"] != "revoke")

    def test_set_revoke_reopen_export_then_reapprove(self, store):
        empty = _exports(store)
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        assert UNTAGGED in lr.export(store, "csv")
        store.revoke(UNTAGGED, "team", reviewer="bob", reason=REASON)

        reopened = DecisionStore(store.path)
        assert reopened.all() == [] and lr.current_decisions(reopened) == []
        assert lr.pending_changes(reopened) == []
        exports = _exports(reopened)
        for fmt in ("csv", "sh", "tfvars"):
            assert exports[fmt].split("\n", 2)[-1] == empty[fmt].split("\n", 2)[-1]
            assert UNTAGGED not in exports[fmt]
        plan = json.loads(exports["plan"])
        assert plan["changes"] == [] and plan["approvals"] == []
        assert REASON not in exports["plan"]

        reopened.set(UNTAGGED, "team", "override", "ml", reviewer="carol")
        plan = lr.approved_change_plan(reopened)
        assert [c["resource_id"] for c in plan["changes"]] == [UNTAGGED]
        assert plan["changes"][0]["tags"]["team"] == {
            "previous": None,
            "new": "ml",
            "how": "override",
        }
        [approval] = plan["approvals"]
        assert (approval["decision"], approval["reviewer"], approval["value"]) == (
            "override",
            "carol",
            "ml",
        )
        assert not set(approval) & set(lr.REVOCATION_COLUMNS)
        assert pv.plan_problems(plan) == []

    def test_absent_decision_records_nothing(self, store):
        assert store.revoke(UNTAGGED, "team", reviewer="bob") is None
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        assert store.revoke(UNTAGGED, "team", reviewer="bob") is not None
        assert store.revoke(UNTAGGED, "team", reviewer="bob") is None
        assert not store.delete(UNTAGGED, "team")
        assert [h["decision"] for h in store.feedback()] == ["accept", "revoke"]

    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            ({"label": "team", "reviewer": " "}, "reviewer"),
            ({"label": "team", "reviewer": "x" * 129}, "reviewer"),
            ({"label": "owner", "reviewer": "bob"}, "Unknown label"),
            ({"label": "team", "reviewer": "bob", "reason": "x" * 501}, "500 characters"),
        ],
    )
    def test_invalid_call_changes_nothing(self, store, kwargs, msg):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        before = (store.all(), store.feedback())
        with pytest.raises(DecisionError, match=msg):
            store.revoke(UNTAGGED, **kwargs)
        assert (store.all(), store.feedback()) == before

    def test_reviewer_falls_back_to_environment(self, store, monkeypatch):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        monkeypatch.delenv("OBS_AGENT_REVIEWER")
        with pytest.raises(DecisionError, match="reviewer"):
            store.revoke(UNTAGGED, "team")
        monkeypatch.setenv("OBS_AGENT_REVIEWER", "env-reviewer")
        assert store.revoke(UNTAGGED, "team")["reviewer"] == "env-reviewer"

    def test_failed_event_write_keeps_the_decision(self, store):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        with sqlite3.connect(store.path) as conn:
            conn.execute(
                "CREATE TRIGGER no_revoke BEFORE INSERT ON review_feedback "
                "WHEN NEW.decision = 'revoke' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
            )
        before = (store.all(), store.feedback())
        with pytest.raises(sqlite3.DatabaseError):
            store.revoke(UNTAGGED, "team", reviewer="bob")
        assert (store.all(), store.feedback()) == before
        assert lr.approved_change_plan(store)["approvals"]

    def test_imported_review_revocation_and_plan_validation(self, tmp_path):
        inventory = load_inventory_csv(INVENTORY_SAMPLE)
        store = DecisionStore(tmp_path / "review.db")
        store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
        store.set("gcp-demo-003", "env", "override", "prod", inventory=inventory, reviewer="alice")
        store.revoke("gcp-demo-003", "env", reviewer="bob", reason=REASON)

        plan = lr.approved_change_plan(store, inventory)
        assert set(plan["changes"][0]["tags"]) == {"team", "cost_center"}
        assert [a["label"] for a in plan["approvals"]] == ["team"]
        report = pv.validate_plan(plan, load_inventory_csv(INVENTORY_SAMPLE), source=inventory)
        assert report["status"] == "pass"

    def test_revoked_stale_decision_keeps_its_own_context(self, store, monkeypatch):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        old = store.get(UNTAGGED, "team")
        context = lr.review_context()
        monkeypatch.setattr(
            lr, "review_context", lambda inv=None: context | {"model_version": "v2"}
        )
        assert lr.current_decisions(store) == []  # stale: not exported anyway
        event = store.revoke(UNTAGGED, "team", reviewer="bob")
        assert event["model_version"] == old["model_version"]


class TestMetrics:
    def test_revocations_are_not_scored(self, store):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        scored_before = lr.feedback_metrics(store.feedback())
        store.revoke(UNTAGGED, "team", reviewer="bob")
        metrics = lr.feedback_metrics(store.feedback())
        assert metrics["revoked"] == 1 and metrics["labels"]["team"]["revoked"] == 1
        assert metrics["reviewed"] == scored_before["reviewed"] == 1
        assert metrics["scored"] == scored_before["scored"]
        assert metrics["unresolved"] == scored_before["unresolved"] == 0
        assert metrics["labels"]["team"] | {"revoked": 0} == scored_before["labels"]["team"]
        only_revocation = lr.feedback_metrics(store.feedback()[1:])
        assert only_revocation["scored"] == only_revocation["reviewed"] == 0

    def test_real_evaluation_ignores_revocations(self):
        rows = []
        for i in range(50):
            base = {
                "resource_id": f"r{i}",
                "label": "team",
                "suggested": "ml",
                "previous": None,
                "confidence": 0.99,
                "value": "ml",
                "decided_at": "2026-10-01T00:00:00+00:00",
            }
            rows.append(base | {"decision": "accept"})
            if i % 2:
                rows.append(base | {"decision": "revoke", "confidence": None})
        report = real_evaluation.evaluate_feedback(rows)
        assert report["sample"]["feedback_events"] == 50
        assert report["reviewer_agreement"]["labels"]["team"]["scored"] == 50


class TestMigration:
    def _old_database(self, path: Path) -> None:
        """A database written before reviewer, provenance and revocation columns existed."""
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE decisions (resource_id TEXT NOT NULL, label TEXT NOT NULL, "
                "decision TEXT NOT NULL, value TEXT, suggested TEXT, previous TEXT, "
                "decided_at TEXT NOT NULL, PRIMARY KEY (resource_id, label))"
            )
            conn.execute(
                "CREATE TABLE review_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "resource_id TEXT NOT NULL, label TEXT NOT NULL, decision TEXT NOT NULL, "
                "value TEXT, suggested TEXT, previous TEXT, confidence REAL, source TEXT, "
                "reason TEXT, decided_at TEXT NOT NULL)"
            )
            row = (UNTAGGED, "team", "accept", "data-platform", "data-platform", None)
            conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, '2026-01-02T00:00:00+00:00')", row
            )
            # An earlier decision undone under the old schema: feedback kept, decision gone.
            conn.execute(
                "INSERT INTO review_feedback (resource_id, label, decision, value, suggested, "
                "previous, confidence, source, reason, decided_at) VALUES "
                "(?, 'env', 'override', 'prod', NULL, NULL, NULL, 'rules', 'r', "
                "'2026-01-01T00:00:00+00:00')",
                (UNTAGGED,),
            )
            conn.execute(
                "INSERT INTO review_feedback (resource_id, label, decision, value, suggested, "
                "previous, confidence, source, reason, decided_at) VALUES "
                "(?, ?, ?, ?, ?, ?, 0.8, 'model', 'r', '2026-01-02T00:00:00+00:00')",
                row,
            )

    def test_existing_database_migrates_without_inventing_revocations(self, tmp_path):
        path = tmp_path / "old.db"
        self._old_database(path)

        store = DecisionStore(path)
        DecisionStore(path)  # idempotent

        with sqlite3.connect(path) as conn:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(review_feedback)")}
        assert set(lr.REVOCATION_COLUMNS) <= columns
        history = store.feedback()
        assert [h["decision"] for h in history] == ["override", "accept"]
        assert all(h[c] is None for h in history for c in lr.REVOCATION_COLUMNS)
        assert [d["decision"] for d in store.all()] == ["accept"]
        assert lr.feedback_metrics(history)["revoked"] == 0

        event = store.revoke(UNTAGGED, "team", reviewer="bob")
        assert event["revoked_feedback_id"] == history[1]["id"]
        assert event["revoked_reviewer"] is None  # unknown then; not invented now
        assert [h["decision"] for h in store.feedback()] == ["override", "accept", "revoke"]

    def test_legacy_json_decision_revokes_without_a_feedback_link(self, tmp_path):
        legacy = tmp_path / "decisions.json"
        legacy.write_text(
            json.dumps(
                [
                    {
                        "resource_id": UNTAGGED,
                        "label": "team",
                        "decision": "accept",
                        "value": "data-platform",
                        "suggested": "data-platform",
                        "previous": None,
                        "decided_at": "2025-12-01T00:00:00+00:00",
                    }
                ]
            )
        )
        store = DecisionStore(legacy)
        assert store.feedback() == []
        event = store.revoke(UNTAGGED, "team", reviewer="bob")
        assert event["revoked_feedback_id"] is None
        assert event["revoked_decided_at"] == "2025-12-01T00:00:00+00:00"


class TestApi:
    @pytest.fixture
    def client(self, store, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        monkeypatch.setattr(api, "decisions", store)
        return TestClient(api.app, headers={"X-API-Key": "test-api-key"})

    URL = f"/resources/{UNTAGGED}/labels/team/decision"

    def test_delete_records_reviewer_and_reason(self, client, store, caplog):
        caplog.set_level(logging.DEBUG)
        client.put(self.URL, json={"decision": "accept", "reviewer": "alice"})
        r = client.request("DELETE", self.URL, json={"reviewer": "bob", "reason": REASON})
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] == f"{UNTAGGED}/team"
        assert body["revocation"]["reviewer"] == "bob"
        assert body["revocation"]["revocation_reason"] == REASON
        assert body["revocation"]["revoked_reviewer"] == "alice"
        assert client.get("/labels/decisions").json() == []
        feedback = client.get("/labels/feedback")
        assert feedback.status_code == 200
        assert [f["decision"] for f in feedback.json()] == ["accept", "revoke"]
        assert client.get("/labels/feedback/metrics").json()["revoked"] == 1
        plan = client.get("/labels/export", params={"format": "plan"}).json()
        assert plan["approvals"] == [] and plan["changes"] == []
        # Free-text reasons stay out of logs and metrics.
        assert REASON not in caplog.text
        assert REASON not in client.get("/ops/metrics").text
        assert REASON not in client.get("/ops/metrics", params={"format": "prometheus"}).text

    def test_delete_without_body_uses_configured_reviewer(self, client):
        client.put(self.URL, json={"decision": "accept"})
        r = client.delete(self.URL)
        assert r.status_code == 200
        assert r.json()["revocation"]["reviewer"] == "test-reviewer"

    def test_repeated_delete_is_404_and_records_nothing(self, client, store):
        assert client.delete(self.URL).status_code == 404
        client.put(self.URL, json={"decision": "accept"})
        assert client.delete(self.URL).status_code == 200
        assert client.delete(self.URL).status_code == 404
        assert [f["decision"] for f in store.feedback()] == ["accept", "revoke"]

    def test_invalid_delete_changes_nothing(self, client, store, monkeypatch):
        client.put(self.URL, json={"decision": "accept"})
        before = (store.all(), store.feedback())
        assert client.request("DELETE", self.URL, json={"reason": "x" * 501}).status_code == 422
        assert client.request("DELETE", self.URL, json={"reviewer": ""}).status_code == 422
        monkeypatch.delenv("OBS_AGENT_REVIEWER")
        r = client.delete(self.URL)
        assert r.status_code == 422 and "reviewer" in r.json()["detail"]
        assert (store.all(), store.feedback()) == before

    def test_reapproval_after_revocation(self, client):
        client.put(self.URL, json={"decision": "accept", "reviewer": "alice"})
        client.request("DELETE", self.URL, json={"reviewer": "bob"})
        r = client.put(self.URL, json={"decision": "override", "value": "ml", "reviewer": "carol"})
        assert r.status_code == 200
        plan = client.get("/labels/export", params={"format": "plan"}).json()
        assert [(a["decision"], a["reviewer"]) for a in plan["approvals"]] == [
            ("override", "carol")
        ]


class TestReviewPage:
    def test_page_offers_a_revocation_reason(self, store, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        store.revoke(UNTAGGED, "team", reviewer="bob", reason=REASON)
        page = Path(__file__).parent.parent / "pages" / "1_Label_review.py"
        at = AppTest.from_file(str(page), default_timeout=60).run()
        assert not at.exception
        assert any(t.key == "revoke_reason" for t in at.text_input)
        assert {m.label: m.value for m in at.metric}["Reviewed"] == "0"
