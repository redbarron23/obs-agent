"""
Review assignments, notes and unresolved questions (#46).

Collaboration metadata is attributable, append-only history. It must never change suggestions,
decisions, the reviewed change plan or exports, and its free text must never reach logs,
metrics or plans. Every database lives in a pytest temp directory; nothing calls an LLM.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import label_review as lr
import review_cli
import workflow_metrics as wm
from inventory_import import load_inventory_csv
from label_review import DecisionError, DecisionStore
from review_workspaces import load_review, save_review

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "fixtures" / "sanitized_inventory_sample.csv"
UNTAGGED = "svc-ingest"  # synthetic: no team tag; team -> data-platform
OTHER = "gcp-demo-003"  # sanitized sample: team -> data-platform, env -> dev
NOTE = "ask the data team; owner left in May"
QUESTION = "who pays for the nightly export job?"
ANSWER = "finance analytics, per the budget sheet"
RATIONALE = "owner confirmed this belongs to ml, not data-platform"
SECRETS = (NOTE, QUESTION, ANSWER, RATIONALE)


@pytest.fixture
def store(tmp_path):
    return DecisionStore(tmp_path / "decisions.db")


@pytest.fixture
def frozen(monkeypatch):
    """Fix every timestamp the store and exports write, so outputs compare byte for byte."""

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 1, 9, 30, tzinfo=UTC)

    monkeypatch.setattr(lr, "datetime", Fixed)


def _exports(store, inventory=None) -> dict[str, str]:
    return {fmt: lr.export(store, fmt, inventory=inventory) for fmt in lr.EXPORT_FORMATS}


def _add_everything(store, resource, inventory=None):
    """Every kind of collaboration metadata on the label being decided and on another one."""
    for label in ("team", "env"):
        store.assign(resource, label, "bob", inventory, reviewer="alice")
        store.add_note(resource, label, NOTE, inventory, reviewer="alice")
        q = store.ask_question(resource, label, QUESTION, inventory, reviewer="alice")
        store.answer_question(q["id"], ANSWER, reviewer="bob")
    resolved = store.ask_question(resource, "team", QUESTION, inventory, reviewer="alice")
    store.resolve_question(resolved["id"], ANSWER, reviewer="bob")


class TestAssignments:
    def test_assign_reassign_unassign_with_attributable_history(self, store):
        first = store.assign(UNTAGGED, "team", " bob ", reviewer="alice")
        assert (first["event"], first["assignee"], first["reviewer"]) == ("assign", "bob", "alice")
        assert store.assign(UNTAGGED, "team", "bob", reviewer="carol") is None  # no-op
        store.assign(UNTAGGED, "team", "dana", reviewer="carol")  # reassign
        item = store.collaboration_item(UNTAGGED, "team")
        assert (item["assignee"], item["assigned_by"]) == ("dana", "carol")

        removed = store.unassign(UNTAGGED, "team", reviewer="dana")
        assert removed["assignee"] == "dana"  # who was unassigned
        assert store.unassign(UNTAGGED, "team", reviewer="dana") is None  # nothing to remove
        assert store.collaboration_item(UNTAGGED, "team")["assignee"] is None

        history = store.collaboration_history()
        assert [(e["event"], e["assignee"], e["reviewer"]) for e in history] == [
            ("assign", "bob", "alice"),
            ("assign", "dana", "carol"),
            ("unassign", "dana", "dana"),
        ]
        assert all(e["created_at"] for e in history)

    @pytest.mark.parametrize(
        "call,error",
        [
            (lambda s: s.assign(UNTAGGED, "team", "bob", reviewer=" "), DecisionError),
            (lambda s: s.assign(UNTAGGED, "team", " ", reviewer="alice"), DecisionError),
            (lambda s: s.assign(UNTAGGED, "team", "x" * 129, reviewer="alice"), DecisionError),
            (lambda s: s.assign(UNTAGGED, "owner", "bob", reviewer="alice"), DecisionError),
            (lambda s: s.assign("no-such-resource", "team", "bob", reviewer="a"), KeyError),
            (lambda s: s.add_note(UNTAGGED, "team", " ", reviewer="alice"), DecisionError),
            (lambda s: s.add_note(UNTAGGED, "team", "x" * 1001, reviewer="a"), DecisionError),
            (lambda s: s.ask_question("no-such-resource", "team", "q?", reviewer="a"), KeyError),
            (lambda s: s.answer_question(999, "a", reviewer="alice"), KeyError),
            (lambda s: s.resolve_question(999, reviewer="alice"), KeyError),
        ],
    )
    def test_invalid_calls_write_nothing(self, store, call, error):
        store.assign(UNTAGGED, "team", "bob", reviewer="alice")
        before = store.collaboration_history()
        with pytest.raises(error):
            call(store)
        assert store.collaboration_history() == before

    def test_reviewer_falls_back_to_environment(self, store, monkeypatch):
        monkeypatch.delenv("OBS_AGENT_REVIEWER")
        with pytest.raises(DecisionError, match="reviewer"):
            store.add_note(UNTAGGED, "team", NOTE)
        monkeypatch.setenv("OBS_AGENT_REVIEWER", "env-reviewer")
        assert store.add_note(UNTAGGED, "team", NOTE)["reviewer"] == "env-reviewer"


class TestNotesAndQuestions:
    def test_note_and_question_lifecycle(self, store):
        store.add_note(UNTAGGED, "team", f"  {NOTE} ", reviewer="alice")
        q = store.ask_question(UNTAGGED, "team", QUESTION, reviewer="alice")
        item = store.collaboration_item(UNTAGGED, "team")
        assert [n["text"] for n in item["notes"]] == [NOTE]
        assert item["questions"][0]["status"] == "open" and item["unresolved_questions"] == 1

        store.answer_question(q["id"], ANSWER, reviewer="bob")
        question = store.collaboration_item(UNTAGGED, "team")["questions"][0]
        assert question["status"] == "answered"  # answered is still unresolved
        assert question["answers"][0] | {"created_at": None} == {
            "text": ANSWER,
            "reviewer": "bob",
            "created_at": None,
        }
        assert store.collaboration_item(UNTAGGED, "team")["unresolved_questions"] == 1

        store.resolve_question(q["id"], reviewer="alice")
        item = store.collaboration_item(UNTAGGED, "team")
        assert item["questions"][0]["status"] == "resolved"
        assert item["questions"][0]["resolved_by"] == "alice"
        assert item["unresolved_questions"] == 0
        with pytest.raises(DecisionError, match="already resolved"):
            store.answer_question(q["id"], ANSWER, reviewer="bob")
        with pytest.raises(DecisionError, match="already resolved"):
            store.resolve_question(q["id"], reviewer="bob")
        assert [e["event"] for e in item["history"]] == ["note", "question", "answer", "resolve"]

    def test_question_is_scoped_to_its_label(self, store):
        q = store.ask_question(UNTAGGED, "team", QUESTION, reviewer="alice")
        with pytest.raises(KeyError):
            store.answer_question(
                q["id"], ANSWER, reviewer="bob", resource_id=UNTAGGED, label="env"
            )
        note = store.add_note(UNTAGGED, "team", NOTE, reviewer="alice")
        with pytest.raises(KeyError):  # a note is not a question
            store.resolve_question(note["id"], reviewer="bob")

    def test_history_is_append_only(self, store):
        store.add_note(UNTAGGED, "team", NOTE, reviewer="alice")
        with sqlite3.connect(store.path) as conn:
            for sql in (
                "UPDATE review_collaboration SET text = 'x'",
                "DELETE FROM review_collaboration",
            ):
                with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                    conn.execute(sql)
        assert [e["text"] for e in store.collaboration_history()] == [NOTE]

    def test_state_survives_reopening_the_store(self, store):
        _add_everything(store, UNTAGGED)
        before = store.collaboration()
        assert DecisionStore(store.path).collaboration() == before

    def test_saved_review_reopens_with_its_collaboration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        review_id = save_review(SAMPLE.read_bytes(), "estate.csv")
        inventory, store = load_review(review_id)
        _add_everything(store, OTHER, inventory)
        before = store.collaboration()
        # "Restart": a fresh load of the saved review (as after an app restart) sees it all.
        _, reopened = load_review(review_id)
        assert reopened.collaboration() == before
        assert {i["assignee"] for i in before} == {"bob"}


class TestRationale:
    def test_override_rationale_is_kept_and_linked(self, store):
        decision = store.set(
            UNTAGGED, "team", "override", "ml", reviewer="alice", rationale=RATIONALE
        )
        feedback = store.feedback()[-1]
        assert "rationale" not in decision and RATIONALE not in json.dumps(feedback)
        note = store.collaboration_item(UNTAGGED, "team")["notes"][0]
        assert note["kind"] == "rationale" and note["text"] == RATIONALE
        assert note["feedback_id"] == feedback["id"] and note["reviewer"] == "alice"
        # A later decision without rationale keeps the earlier one in history.
        store.set(UNTAGGED, "team", "accept", reviewer="bob")
        assert [n["text"] for n in store.collaboration_item(UNTAGGED, "team")["notes"]] == [
            RATIONALE
        ]

    def test_invalid_rationale_records_no_decision(self, store):
        with pytest.raises(DecisionError, match="1000 characters"):
            store.set(UNTAGGED, "team", "override", "ml", reviewer="alice", rationale="x" * 1001)
        assert store.all() == [] and store.feedback() == [] and store.collaboration_history() == []


class TestApprovalAndExportBoundary:
    def test_collaboration_alone_never_approves_or_exports(self, store):
        report_before = lr.resource_report(UNTAGGED)
        backlog_before = wm.backlog(store)
        empty = _exports(store)
        _add_everything(store, UNTAGGED)
        # Assigned to, noted on and resolved: still no decision, nothing to apply.
        assert store.all() == [] and lr.current_decisions(store) == []
        assert lr.approved_change_plan(store)["changes"] == []
        assert lr.approved_change_plan(store)["approvals"] == []
        assert lr.pending_changes(store) == []
        exports = _exports(store)
        for fmt in ("csv", "tfvars"):
            assert exports[fmt] == empty[fmt]
        assert lr.resource_report(UNTAGGED) == report_before  # suggestions unchanged
        backlog = wm.backlog(store)
        assert backlog["reviewed_items"] == 0
        assert backlog["pending_items"] == backlog_before["pending_items"]
        assert backlog["feedback_actions"] == 0

    @pytest.mark.parametrize("imported", [False, True])
    def test_plans_and_exports_byte_identical_with_and_without_metadata(
        self, tmp_path, frozen, imported
    ):
        inventory = load_inventory_csv(SAMPLE) if imported else None
        resource = OTHER if imported else UNTAGGED
        plain = DecisionStore(tmp_path / "plain.db")
        rich = DecisionStore(tmp_path / "rich.db")
        _add_everything(rich, resource, inventory)
        for s, rationale in ((plain, None), (rich, RATIONALE)):
            s.set(
                resource, "team", "override", "ml", inventory, reviewer="alice", rationale=rationale
            )
            s.set(resource, "env", "accept", inventory=inventory, reviewer="alice")
        plain_exports, rich_exports = _exports(plain, inventory), _exports(rich, inventory)
        assert rich_exports == plain_exports
        assert json.loads(rich_exports["plan"])["changes"]  # a non-trivial plan
        for text in rich_exports.values():
            assert not any(secret in text for secret in SECRETS)
        assert plain.feedback() == rich.feedback()  # also the anonymized-feedback source

    def test_revoking_after_collaboration_still_removes_approval(self, store):
        store.set(UNTAGGED, "team", "accept", reviewer="alice")
        store.assign(UNTAGGED, "team", "bob", reviewer="alice")
        store.revoke(UNTAGGED, "team", reviewer="bob")
        assert lr.approved_change_plan(store)["changes"] == []
        assert store.collaboration_item(UNTAGGED, "team")["assignee"] == "bob"


class TestQueueFilters:
    def test_assignee_and_unresolved_filters(self, store):
        flagged = [r["resource_id"] for r in lr.review_queue(label="team")][:3]
        a, b, c = flagged
        store.assign(a, "team", "bob", reviewer="alice")
        store.assign(b, "team", "carol", reviewer="alice")
        store.assign(c, "env", "bob", reviewer="alice")  # other label
        q = store.ask_question(b, "team", QUESTION, reviewer="alice")

        def ids(**kw):
            return [r["resource_id"] for r in lr.review_queue(label="team", store=store, **kw)]

        assert ids(assignee="bob") == [a]
        assert ids(assignee="carol") == [b]
        assert ids(unresolved=True) == [b]
        assert ids(assignee="carol", unresolved=True) == [b]
        assert ids(assignee="bob", unresolved=True) == []
        store.answer_question(q["id"], ANSWER, reviewer="carol")
        assert ids(unresolved=True) == [b]  # answered, not resolved
        store.resolve_question(q["id"], reviewer="carol")
        assert ids(unresolved=True) == []
        assert b in ids(unresolved=False)
        # Without filters the queue is unchanged by collaboration metadata.
        assert lr.review_queue(label="team", store=store) == lr.review_queue(label="team")

    def test_filters_need_a_store(self):
        with pytest.raises(ValueError, match="store"):
            lr.review_queue(assignee="bob")

    def test_queue_summary_has_counts_not_text(self, store):
        store.assign(UNTAGGED, "team", "bob", reviewer="alice")
        store.add_note(UNTAGGED, "team", NOTE, reviewer="alice")
        store.ask_question(UNTAGGED, "team", QUESTION, reviewer="alice")
        rows = lr.with_collaboration(lr.review_queue(search=UNTAGGED), store)
        assert rows[0]["collaboration"] == {
            "team": {"assignee": "bob", "notes": 1, "unresolved_questions": 1}
        }
        assert not any(s in json.dumps(rows) for s in SECRETS)
        assert "collaboration" not in lr.all_reports()[0]  # cached reports untouched


class TestMigration:
    def test_pre_46_database_opens_with_no_invented_metadata(self, tmp_path):
        path = tmp_path / "old.db"
        old = DecisionStore(path)
        old.set(UNTAGGED, "team", "accept", reviewer="alice")
        old.set(UNTAGGED, "env", "accept", reviewer="alice")
        old.revoke(UNTAGGED, "env", reviewer="bob", reason="wrong")
        with sqlite3.connect(path) as conn:  # the schema as it was before #46
            conn.execute("DROP TRIGGER review_collaboration_no_update")
            conn.execute("DROP TRIGGER review_collaboration_no_delete")
            conn.execute("DROP TABLE review_collaboration")
        decisions, feedback = old.all(), old.feedback()
        plan = {k: v for k, v in lr.approved_change_plan(old).items() if k != "generated_at"}

        store = DecisionStore(path)
        DecisionStore(path)  # idempotent

        assert store.collaboration_history() == [] and store.collaboration() == []
        assert store.all() == decisions and store.feedback() == feedback
        after = {k: v for k, v in lr.approved_change_plan(store).items() if k != "generated_at"}
        assert after == plan
        assert wm.backlog(store)["pending_assigned_items"] == 0
        store.assign(UNTAGGED, "team", "bob", reviewer="alice")
        assert store.collaboration_item(UNTAGGED, "team")["assignee"] == "bob"


class TestBacklog:
    def test_backlog_counts_assigned_pending_items_and_unresolved_questions(self, store):
        flagged = [r["resource_id"] for r in lr.review_queue(label="team")][:2]
        store.assign(flagged[0], "team", "bob", reviewer="alice")
        store.assign(flagged[1], "team", "bob", reviewer="alice")
        q = store.ask_question(flagged[0], "team", QUESTION, reviewer="alice")
        backlog = wm.backlog(store)
        assert backlog["pending_assigned_items"] == 2
        assert backlog["unresolved_questions"] == 1
        store.set(flagged[1], "team", "accept", reviewer="bob")  # decided: no longer pending
        store.resolve_question(q["id"], reviewer="bob")
        backlog = wm.backlog(store)
        assert backlog["pending_assigned_items"] == 1
        assert backlog["unresolved_questions"] == 0
        assert backlog["reviewed_items"] == 1 and backlog["feedback_actions"] == 1


class TestApi:
    @pytest.fixture
    def client(self, store, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        monkeypatch.setattr(api, "decisions", store)
        return TestClient(api.app, headers={"X-API-Key": "test-api-key"})

    ITEM = f"/resources/{UNTAGGED}/labels/team"

    def test_synthetic_workflow_and_text_stays_out_of_logs_and_metrics(self, client, store, caplog):
        caplog.set_level(logging.DEBUG)
        r = client.put(f"{self.ITEM}/assignment", json={"assignee": "bob", "reviewer": "alice"})
        assert r.status_code == 200 and r.json()["assignee"] == "bob"
        assert client.post(f"{self.ITEM}/notes", json={"text": NOTE}).status_code == 200
        q = client.post(f"{self.ITEM}/questions", json={"text": QUESTION}).json()["questions"][0]
        r = client.post(f"{self.ITEM}/questions/{q['id']}/answer", json={"text": ANSWER})
        assert r.json()["questions"][0]["status"] == "answered"
        r = client.put(
            f"{self.ITEM}/decision",
            json={"decision": "override", "value": "ml", "rationale": RATIONALE},
        )
        assert r.status_code == 200 and "rationale" not in r.json()

        queue = client.get("/resources", params={"assignee": "bob", "unresolved": True}).json()
        assert [i["resource_id"] for i in queue["items"]] == [UNTAGGED]
        assert queue["items"][0]["collaboration"]["team"]["unresolved_questions"] == 1
        assert client.get("/resources", params={"assignee": "nobody"}).json()["total"] == 0
        items = client.get("/labels/collaboration", params={"assignee": "bob"}).json()
        assert [(i["resource_id"], i["label"]) for i in items] == [(UNTAGGED, "team")]
        history = client.get("/labels/collaboration/history").json()
        assert [e["event"] for e in history] == [
            "assign",
            "note",
            "question",
            "answer",
            "rationale",
        ]
        detail = client.get(f"{self.ITEM}/collaboration").json()
        assert detail["history"] == history

        resolved = client.post(f"{self.ITEM}/questions/{q['id']}/resolve", json={"reviewer": "bob"})
        assert resolved.json()["unresolved_questions"] == 0
        assert client.get("/resources", params={"unresolved": True}).json()["total"] == 0
        assert client.request("DELETE", f"{self.ITEM}/assignment").status_code == 200
        assert client.request("DELETE", f"{self.ITEM}/assignment").status_code == 404

        backlog = client.get("/labels/backlog").json()
        assert backlog["unresolved_questions"] == 0 and backlog["reviewed_items"] == 1
        plan = client.get("/labels/export", params={"format": "plan"}).text
        feedback = client.get("/labels/feedback").text
        metrics = (
            client.get("/ops/metrics").text
            + client.get("/ops/metrics", params={"format": "prometheus"}).text
        )
        for secret in SECRETS:
            assert secret not in caplog.text
            assert secret not in metrics
            assert secret not in plan and secret not in feedback
        assert "collaboration_write" in metrics

    def test_errors(self, client, store, monkeypatch):
        q = store.ask_question(UNTAGGED, "team", QUESTION, reviewer="alice")
        assert client.get("/resources/nope/labels/team/collaboration").status_code == 404
        assert (
            client.post("/resources/nope/labels/team/notes", json={"text": NOTE}).status_code == 404
        )
        r = client.post(
            f"/resources/{UNTAGGED}/labels/env/questions/{q['id']}/answer", json={"text": "a"}
        )
        assert r.status_code == 404  # question belongs to team
        assert client.post(f"{self.ITEM}/notes", json={"text": "x" * 1001}).status_code == 422
        monkeypatch.delenv("OBS_AGENT_REVIEWER")
        r = client.post(f"{self.ITEM}/notes", json={"text": NOTE})
        assert r.status_code == 422 and "reviewer" in r.json()["detail"]
        client.post(f"{self.ITEM}/questions/{q['id']}/resolve", json={"reviewer": "bob"})
        r = client.post(
            f"{self.ITEM}/questions/{q['id']}/resolve", json={"reviewer": "bob", "text": NOTE}
        )
        assert r.status_code == 422 and NOTE not in r.text
        assert [e["event"] for e in store.collaboration_history()] == ["question", "resolve"]


class TestSavedReviewApiAndCli:
    @pytest.fixture
    def review(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
        monkeypatch.setattr(api, "decisions", DecisionStore(tmp_path / "synthetic.db"))
        return save_review(SAMPLE.read_bytes(), "estate.csv")

    def _cli(self, capsys, *argv):
        code = review_cli.main(list(argv))
        out = capsys.readouterr()
        return code, out.out, out.err

    def test_api_and_cli_reach_the_same_store(self, review, capsys, frozen):
        client = TestClient(api.app, headers={"X-API-Key": "test-api-key"})
        base = f"/reviews/{review}"
        plan_before = client.get(f"{base}/export", params={"format": "plan"}).text
        item = f"{base}/resources/{OTHER}/labels/team"
        r = client.put(f"{item}/assignment", json={"assignee": "bob", "reviewer": "alice"})
        assert r.status_code == 200

        code, out, _ = self._cli(capsys, "ask", review, OTHER, "team", QUESTION, "--reviewer", "a")
        assert code == 0
        qid = json.loads(out)["questions"][0]["id"]
        assert self._cli(capsys, "note", review, OTHER, "env", NOTE, "--reviewer", "a")[0] == 0
        assert self._cli(capsys, "assign", review, OTHER, "env", "carol", "--reviewer", "a")[0] == 0

        # Same state through the API...
        api_items = client.get(f"{base}/collaboration").json()
        assert {(i["label"], i["assignee"], i["unresolved_questions"]) for i in api_items} == {
            ("team", "bob", 1),
            ("env", "carol", 0),
        }
        # ...and the CLI, including the "assigned to me" and "has open question" filters.
        _, out, _ = self._cli(capsys, "collaboration", review)
        assert json.loads(out) == api_items
        _, out, _ = self._cli(capsys, "queue", review, "--mine", "--reviewer", "bob")
        rows = json.loads(out)
        assert [r["resource_id"] for r in rows] == [OTHER]
        assert rows[0]["collaboration"]["team"]["assignee"] == "bob"
        _, out, _ = self._cli(capsys, "queue", review, "--label", "env", "--assignee", "bob")
        assert json.loads(out) == []
        _, out, _ = self._cli(capsys, "queue", review, "--unresolved")
        assert [r["resource_id"] for r in json.loads(out)] == [OTHER]
        queue = client.get(f"{base}/resources", params={"assignee": "carol", "label": "env"})
        assert [i["resource_id"] for i in queue.json()["items"]] == [OTHER]

        assert self._cli(capsys, "answer", review, str(qid), ANSWER, "--reviewer", "bob")[0] == 0
        assert self._cli(capsys, "resolve", review, str(qid), "--reviewer", "bob")[0] == 0
        _, out, _ = self._cli(capsys, "queue", review, "--unresolved")
        assert json.loads(out) == []
        code, _, err = self._cli(capsys, "resolve", review, str(qid), "--reviewer", "bob")
        assert code == 2 and "already resolved" in err
        code, _, err = self._cli(capsys, "answer", review, "999", "x", "--reviewer", "bob")
        assert code == 2 and "Unknown" in err
        code, _, err = self._cli(capsys, "unassign", review, OTHER, "team", "--reviewer", "bob")
        assert code == 0
        code, _, err = self._cli(capsys, "unassign", review, OTHER, "team", "--reviewer", "bob")
        assert code == 2 and "Not assigned" in err

        # Collaboration never changed the plan; a CLI override keeps its rationale in history.
        assert client.get(f"{base}/export", params={"format": "plan"}).text == plan_before
        code, _, _ = self._cli(
            capsys,
            "decide",
            review,
            OTHER,
            "team",
            "override",
            "--value",
            "ml",
            "--reviewer",
            "alice",
            "--rationale",
            RATIONALE,
        )
        assert code == 0
        _, out, _ = self._cli(
            capsys, "collaboration", review, "--resource", OTHER, "--label", "team"
        )
        detail = json.loads(out)
        assert detail["notes"][-1]["kind"] == "rationale"
        assert detail == client.get(f"{item}/collaboration").json()
        _, out, _ = self._cli(capsys, "collaboration", review, "--history")
        assert [e["event"] for e in json.loads(out)][-1] == "rationale"
        plan = client.get(f"{base}/export", params={"format": "plan"}).text
        assert json.loads(plan)["changes"] and not any(s in plan for s in SECRETS)
        backlog = client.get(f"{base}/backlog").json()
        assert backlog["reviewed_items"] == 1 and backlog["unresolved_questions"] == 0


class TestReviewPage:
    PAGE = ROOT / "pages" / "1_Label_review.py"

    def test_page_shows_and_records_collaboration(self, store, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))
        monkeypatch.setenv("OBS_AGENT_REVIEWER", "bob")
        first = lr.review_queue(label="team")[0]["resource_id"]
        store.assign(first, "team", "bob", reviewer="alice")
        store.ask_question(first, "team", QUESTION, reviewer="alice")

        at = AppTest.from_file(str(self.PAGE), default_timeout=60).run()
        assert not at.exception
        assert any(t.key == "decision_rationale" for t in at.text_input)
        assert {m.label: m.value for m in at.metric}["Reviewed"] == "0"

        next(c for c in at.sidebar.checkbox if c.label == "Assigned to me").check().run()
        assert at.subheader[0].value.startswith("1 resources")
        next(c for c in at.sidebar.checkbox if c.label == "Only unresolved questions").check().run()
        assert at.subheader[0].value.startswith("1 resources")
        assert at.selectbox(key="collab_resource").value == first

        at.text_area(key="collab_text").set_value(NOTE).run()
        at.button(key="collab_note").click().run()
        assert not at.exception
        notes = store.collaboration_item(first, "team")["notes"]
        assert [(n["text"], n["reviewer"]) for n in notes] == [(NOTE, "bob")]
        assert store.all() == []  # a note is not a decision

        at.button(key="collab_resolve").click().run()
        assert not at.exception
        assert store.collaboration_item(first, "team")["unresolved_questions"] == 0
        assert store.all() == []
