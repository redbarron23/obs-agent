"""Policy evidence retained with reviewed label decisions (issue #103).

Offline: retrieval results are built from the committed fixture, and the review-page tests use
the stub retrieval API from test_policy_docs. No real rag service or LLM is called.
"""

import copy
import json
import sqlite3

import pytest

import label_review as lr
import policy_docs
import policy_evidence
import workflow_metrics
from review_workspaces import load_review, save_review
from tests.test_backup_restore import CONFIG, CSV
from tests.test_policy_docs import FIXTURE, REVIEW_PAGE, rag  # noqa: F401  (pytest fixture)

QUESTION = policy_docs.review_question("team", "untagged")


def _result(payload: dict | None = None, question: str = QUESTION) -> dict:
    payload = copy.deepcopy(payload or FIXTURE)
    hits = policy_docs._parse_hits(payload)
    return policy_docs._result(
        "ok" if hits else "no_match",
        "x",
        question=question,
        hits=hits,
        corpus=payload["corpus"],
        freshness=payload["freshness"],
    )


@pytest.fixture
def store(tmp_path):
    s = lr.DecisionStore(tmp_path / "decisions.db")
    s.set("svc-ingest", "team", "accept", reviewer="alice")
    return s


def _attach(store, result=None, hits=slice(0, 1), **kw):
    result = result or _result()
    kw = {"question": result["question"], "reviewer": "alice"} | kw
    return store.attach_policy_evidence(
        "svc-ingest", "team", result, result["hits"][hits], kw.pop("inventory", None), **kw
    )


# ── What is recorded ───────────────────────────────────────────────────


class TestAttach:
    def test_records_excerpt_corpus_and_decision_context(self, store):
        (row,) = _attach(store, retrieved_at="2026-11-01T12:00:00+00:00", reviewer="bob")
        hit = FIXTURE["hits"][0]
        decision = store.get("svc-ingest", "team")
        assert (row["source"], row["section"], row["chunk_id"]) == (
            hit["source"],
            hit["section"],
            hit["chunk_id"],
        )
        assert row["excerpt_text"] == hit["text"]
        assert row["excerpt_sha256"] == policy_evidence.excerpt_sha256(hit["text"])
        assert row["corpus_sha256"] == FIXTURE["corpus"]["corpus_sha256"]
        assert row["ingested_at"] == FIXTURE["corpus"]["ingested_at"]
        assert (row["corpus_provenance"], row["freshness_status"]) == ("available", "current")
        assert row["question"] == QUESTION and row["retrieved_at"].startswith("2026-11-01")
        # The decision it supported, and who attached it.
        assert row["feedback_id"] == store.feedback()[-1]["id"]
        assert (row["decision"], row["decision_reviewer"]) == ("accept", "alice")
        assert row["decided_at"] == decision["decided_at"]
        assert (row["inventory_id"], row["model_version"]) == (
            decision["inventory_id"],
            decision["model_version"],
        )
        assert row["reviewer"] == "bob"
        assert row["policy"] is None and row["policy_status"] is None

    def test_hash_only(self, store):
        (row,) = _attach(store, retain_text=False)
        assert row["excerpt_text"] is None and len(row["excerpt_sha256"]) == 64
        assert any("exact historical text is unavailable" in r for r in _inspect(store)[0])

    def test_policy_metadata_is_kept_when_reported(self, store):
        payload = copy.deepcopy(FIXTURE)
        meta = {"status": "active", "policy_id": "tagging", "version": "3", "supersedes": []}
        payload["hits"][0]["policy"] = meta
        (row,) = _attach(store, _result(payload))
        assert row["policy_status"] == "active" and json.loads(row["policy"]) == meta

    def test_needs_an_active_decision(self, store):
        store.revoke("svc-ingest", "team", reviewer="alice")
        with pytest.raises(lr.DecisionError, match="Save a decision first"):
            _attach(store)
        assert store.policy_evidence() == []

    @pytest.mark.parametrize(
        ("result", "hits", "match"),
        [
            (policy_docs._result("timeout", "x", question=QUESTION), [], "answered retrieval"),
            (_result(), [], "Select at least one"),
            (_result(), [dict(FIXTURE["hits"][0], text="forged")], "retrieval shown"),
        ],
    )
    def test_refuses_unanswered_empty_or_foreign_excerpts(self, store, result, hits, match):
        with pytest.raises(lr.DecisionError, match=match):
            store.attach_policy_evidence(
                "svc-ingest", "team", result, hits, question=QUESTION, reviewer="alice"
            )
        assert store.policy_evidence() == []

    def test_needs_a_reviewer(self, store, monkeypatch):
        monkeypatch.delenv("OBS_AGENT_REVIEWER", raising=False)
        with pytest.raises(lr.DecisionError, match="reviewer"):
            _attach(store, reviewer="")

    def test_append_only(self, store):
        _attach(store)
        with sqlite3.connect(store.path) as conn:
            for sql in ("UPDATE policy_evidence SET reviewer = 'x'", "DELETE FROM policy_evidence"):
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    conn.execute(sql)

    def test_is_not_an_approval(self, store):
        """Evidence never changes decisions, plans, exports or label confidence."""
        before = (
            store.all(),
            store.feedback(),
            lr.approved_change_plan(store)["changes"],
            lr.export(store, "csv"),
            lr.resource_report("svc-ingest"),
        )
        _attach(store, hits=slice(None))
        after = (
            store.all(),
            store.feedback(),
            lr.approved_change_plan(store)["changes"],
            lr.export(store, "csv"),
            lr.resource_report("svc-ingest"),
        )
        assert after == before
        assert FIXTURE["hits"][0]["text"][:40] not in json.dumps(lr.approved_change_plan(store))


# ── Two snapshots ──────────────────────────────────────────────────────


class TestSnapshots:
    def test_saved_review_and_synthetic_store_are_isolated(self, store, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        inventory, saved = load_review(save_review(CSV, "private.csv", CONFIG))
        saved.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="bob")
        result = _result()
        (row,) = saved.attach_policy_evidence(
            "gcp-demo-003",
            "team",
            result,
            result["hits"][:1],
            inventory,
            question=QUESTION,
            reviewer="bob",
        )
        assert row["inventory_id"] == saved.get("gcp-demo-003", "team")["inventory_id"]
        assert row["inventory_id"] != store.get("svc-ingest", "team")["inventory_id"]
        assert store.policy_evidence() == []

    def test_decision_from_another_snapshot_must_be_reviewed_again(self, store, monkeypatch):
        original = lr.review_context()
        monkeypatch.setattr(
            lr, "review_context", lambda inv=None: original | {"inventory_id": "other"}
        )
        with pytest.raises(lr.DecisionError, match="another inventory snapshot"):
            _attach(store)
        assert store.policy_evidence() == []


# ── Reasons to inspect a decision again ────────────────────────────────


def _inspect(store, current=None) -> list[list[str]]:
    return [row["inspect"] for row in store.policy_evidence_review("svc-ingest", "team", current)]


class TestInspection:
    def test_unchanged_evidence_needs_no_inspection(self, store):
        _attach(store)
        assert _inspect(store) == [[]]
        assert _inspect(store, _result()) == [[]]
        assert store.policy_evidence_review("svc-ingest", "team")[0]["active"]

    def test_changed_corpus_and_excerpt_text(self, store):
        _attach(store)
        payload = copy.deepcopy(FIXTURE)
        payload["corpus"]["corpus_sha256"] = "f" * 64
        payload["hits"][0]["text"] += " Amended."
        (reasons,) = _inspect(store, _result(payload))
        assert any("index has changed" in r and "0a50d3028906 → ffffffffffff" in r for r in reasons)
        assert any("text has changed" in r for r in reasons)
        # The retained text is still what the reviewer consulted.
        assert store.policy_evidence()[0]["excerpt_text"] == FIXTURE["hits"][0]["text"]

    def test_removed_excerpt(self, store):
        _attach(store)
        payload = copy.deepcopy(FIXTURE)
        payload["hits"] = payload["hits"][1:]
        (reasons,) = _inspect(store, _result(payload))
        assert any("did not return this excerpt" in r for r in reasons)

    def test_superseded_policy(self, store):
        payload = copy.deepcopy(FIXTURE)
        payload["hits"][0]["policy"] = {"status": "active", "policy_id": "tagging", "version": "3"}
        _attach(store, _result(payload))
        payload["hits"][0]["policy"] = {"status": "superseded", "policy_id": "tagging"}
        (reasons,) = _inspect(store, _result(payload))
        assert reasons == ["The policy is now superseded in the current index."]

    def test_attached_while_not_applicable(self, store):
        payload = copy.deepcopy(FIXTURE)
        payload["hits"][0]["policy"] = {"status": "expired", "policy_id": "tagging", "version": "2"}
        _attach(store, _result(payload))
        assert _inspect(store) == [["When attached, policy tagging@2 was expired."]]

    def test_unavailable_provenance(self, store):
        payload = copy.deepcopy(FIXTURE)
        payload["corpus"] = {"provenance": "unavailable"}
        _attach(store, _result(payload))
        (reasons,) = _inspect(store, _result())
        assert any("revision cannot be identified" in r for r in reasons)
        assert not any("index has changed" in r for r in reasons)

    def test_other_question_is_not_compared(self, store):
        _attach(store)
        payload = copy.deepcopy(FIXTURE)
        payload["corpus"]["corpus_sha256"] = "f" * 64
        assert _inspect(store, _result(payload, question=policy_docs.SCOPE_QUESTION)) == [[]]

    def test_revoked_and_replaced_decisions_keep_their_evidence(self, store):
        _attach(store)
        store.revoke("svc-ingest", "team", reviewer="bob", reason="wrong team")
        (row,) = store.policy_evidence_review("svc-ingest", "team")
        assert not row["active"] and "revoked or replaced" in row["inspect"][0]
        store.set("svc-ingest", "team", "accept", reviewer="carol")
        _attach(store, reviewer="carol")
        old, new = store.policy_evidence_review("svc-ingest", "team")
        assert (old["active"], new["active"]) == (False, True)
        assert new["inspect"] == [] and new["decision_reviewer"] == "carol"


# ── Review page ────────────────────────────────────────────────────────


class TestReviewPage:
    @pytest.fixture
    def page(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest

        self.db = tmp_path / "decisions.db"
        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(self.db))
        monkeypatch.setenv("OBS_AGENT_REVIEWER", "alice")
        workflow_metrics.reset()
        return AppTest.from_file(str(REVIEW_PAGE), default_timeout=60)

    def _show(self, at, resource):
        at.selectbox(key="policy_resource").set_value(resource).run()
        next(b for b in at.button if b.label == "Show policy context").click().run()
        assert not at.exception
        return at

    def test_attach_needs_a_decision_first(self, page, rag):  # noqa: F811
        at = page.run()
        resource = at.selectbox(key="policy_resource").options[0]
        at = self._show(at, resource)
        assert any("Save a decision for this label" in c.value for c in at.caption)
        assert not [b for b in at.button if b.label == "Attach to decision"]

    def test_attach_and_reopen(self, page, rag, monkeypatch):  # noqa: F811
        at = page.run()
        resource = at.selectbox(key="policy_resource").options[0]
        lr.DecisionStore(self.db).set(resource, "team", "override", "payments", reviewer="alice")
        at = self._show(at, resource)
        at.multiselect(key=f"policy_attach_{resource}_team").select(
            at.multiselect(key=f"policy_attach_{resource}_team").options[0]
        ).run()
        next(b for b in at.button if b.label == "Attach to decision").click().run()
        assert not at.exception
        assert any("Attached 1 excerpt(s)" in s.value for s in at.success)
        assert any("Retained policy evidence" in m.value for m in at.markdown)
        (row,) = lr.DecisionStore(self.db).policy_evidence()
        assert (row["resource_id"], row["reviewer"]) == (resource, "alice")
        # One evidence_write operation; nothing about the excerpt reaches the metrics.
        ops = workflow_metrics.snapshot()["operations"]
        evidence_ops = [o for o in ops if o["operation"] == "evidence_write"]
        assert evidence_ops == [
            {"operation": "evidence_write", "surface": "ui", "outcome": "success", "count": 1}
        ]
        assert FIXTURE["hits"][0]["source"] not in json.dumps(workflow_metrics.snapshot())

        # The service later changes its index, then goes away: the evidence stays readable.
        rag.payload["corpus"]["corpus_sha256"] = "f" * 64
        at = self._show(page.run(), resource)
        assert any("index has changed" in w.value for w in at.warning)
        monkeypatch.delenv("OBS_AGENT_RAG_URL")
        at = page.run()
        assert not at.exception
        assert at.selectbox(key="evidence_resource").value == resource
        assert any("Retained policy evidence" in m.value for m in at.markdown)
