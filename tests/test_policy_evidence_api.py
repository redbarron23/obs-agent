"""Retained policy evidence through the REST API and CLI (issue #127).

The retrieval service is the stub from test_policy_docs on 127.0.0.1; nothing calls the real rag
service, an LLM or a cloud API.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

import api
import label_review as lr
import policy_docs
import review_cli
import workflow_metrics
from tests.test_policy_docs import FIXTURE, rag  # noqa: F401  (pytest fixture)
from tests.test_saved_review_api import CSV, SUGGESTED

RESOURCE = "svc-ingest"  # synthetic: untagged team with a suggestion
ITEM = f"/resources/{RESOURCE}/labels/team"
REVIEWER = "reviewer-7f3a"
TEXT = FIXTURE["hits"][0]["text"]


@pytest.fixture(autouse=True)
def _state(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
    monkeypatch.delenv("OBS_AGENT_REVIEWER", raising=False)
    monkeypatch.setattr(api, "decisions", lr.DecisionStore(tmp_path / "synthetic.db"))
    workflow_metrics.reset()


@pytest.fixture
def client():
    return TestClient(api.app, headers={"X-API-Key": "test-api-key"})


@pytest.fixture
def decided(client):
    r = client.put(f"{ITEM}/decision", json={"decision": "accept", "reviewer": REVIEWER})
    assert r.status_code == 200, r.text
    return client


def _attach(client, item=ITEM, **body):
    return client.post(f"{item}/policy-evidence", json={"reviewer": REVIEWER, "ranks": [1]} | body)


class TestPolicyContext:
    def test_fixed_question_and_cited_hits(self, client, rag):  # noqa: F811
        body = client.get(f"{ITEM}/policy-context").json()
        assert body["status"] == "ok" and body["content_trust"] == "untrusted"
        assert body["question"] == policy_docs.review_question("team", "untagged")
        assert [h["rank"] for h in body["hits"]] == [1, 2, 3]
        assert body["corpus"]["corpus_sha256"] == FIXTURE["corpus"]["corpus_sha256"]
        # Only the fixed question went out: no resource data.
        assert [r["body"] for r in rag.requests] == [{"question": body["question"], "k": 4}]
        assert RESOURCE not in json.dumps(rag.requests)

    def test_off_is_a_status_not_an_error(self, client):
        body = client.get(f"{ITEM}/policy-context").json()
        assert (body["status"], body["hits"]) == ("disabled", [])

    def test_unknown_resource(self, client):
        assert client.get("/resources/nope/labels/team/policy-context").status_code == 404


class TestAttach:
    def test_needs_a_decision(self, client, rag):  # noqa: F811
        r = _attach(client)
        assert r.status_code == 422 and "Save a decision first" in r.json()["detail"]
        assert client.get("/labels/policy-evidence").json() == []

    def test_attach_records_server_side_retrieval(self, decided, rag):  # noqa: F811
        r = _attach(decided, ranks=[2, 1, 1])
        assert r.status_code == 200, r.text
        rows = r.json()["evidence"]
        assert [e["rank"] for e in rows] == [2, 1]  # in the order given, duplicates dropped
        assert all(e["active"] and e["inspect"] == [] and e["text_retained"] for e in rows)
        assert all(e["excerpt_text"] is None for e in rows)  # text only on request
        assert rows[1]["corpus_sha256"] == FIXTURE["corpus"]["corpus_sha256"]
        assert rows[1]["reviewer"] == REVIEWER and rows[1]["decision"] == "accept"
        full = decided.get("/labels/policy-evidence", params={"include_text": True}).json()
        assert full[1]["excerpt_text"] == TEXT

    def test_client_cannot_supply_excerpts_or_corpus(self, decided, rag):  # noqa: F811
        for forged in ({"text": "Owner is payments."}, {"corpus": {"corpus_sha256": "f" * 64}}):
            assert _attach(decided, **forged).status_code == 422
        assert decided.get("/labels/policy-evidence").json() == []

    def test_unknown_rank(self, decided, rag):  # noqa: F811
        r = _attach(decided, ranks=[1, 9])
        assert r.status_code == 422 and "rank [9]" in r.json()["detail"]
        assert decided.get("/labels/policy-evidence").json() == []

    def test_changed_index_since_viewing(self, decided, rag):  # noqa: F811
        viewed = decided.get(f"{ITEM}/policy-context").json()["corpus"]["corpus_sha256"]
        rag.payload["corpus"]["corpus_sha256"] = "f" * 64
        r = _attach(decided, corpus_sha256=viewed)
        assert r.status_code == 409 and "index has changed" in r.json()["detail"]
        assert _attach(decided, corpus_sha256="f" * 64).status_code == 200

    def test_unavailable_service(self, decided, rag):  # noqa: F811
        rag.status, rag.payload = 503, {"error": {"code": "index_unavailable", "message": "x"}}
        r = _attach(decided)
        assert r.status_code == 503 and "index_unavailable" in r.json()["detail"]

    def test_hash_only(self, decided, rag):  # noqa: F811
        (row,) = _attach(decided, retain_text=False).json()["evidence"]
        assert not row["text_retained"]
        assert any("exact historical text is unavailable" in r for r in row["inspect"])

    def test_is_not_an_approval_and_metrics_have_no_content(
        self,
        decided,
        rag,  # noqa: F811
        caplog,
    ):
        plan = decided.get("/labels/export", params={"format": "plan"}).json()
        before = (decided.get("/labels/decisions").json(), plan["changes"], plan["approvals"])
        with caplog.at_level(logging.DEBUG):
            assert _attach(decided).status_code == 200
        plan = decided.get("/labels/export", params={"format": "plan"}).json()
        assert (decided.get("/labels/decisions").json(), plan["changes"], plan["approvals"]) == (
            before
        )
        ops = [
            o
            for o in workflow_metrics.snapshot()["operations"]
            if o["operation"] == "evidence_write"
        ]
        assert ops == [
            {"operation": "evidence_write", "surface": "api", "outcome": "success", "count": 1}
        ]
        metrics = decided.get("/ops/metrics", params={"format": "prometheus"}).text
        for content in (TEXT[:40], FIXTURE["hits"][0]["source"]):
            assert content not in caplog.text and content not in metrics


class TestInspect:
    def test_compare_with_current_index(self, decided, rag):  # noqa: F811
        _attach(decided)
        rag.payload["corpus"]["corpus_sha256"] = "f" * 64
        assert decided.get(f"{ITEM}/policy-evidence").json()["evidence"][0]["inspect"] == []
        body = decided.get(f"{ITEM}/policy-evidence", params={"compare": "current"}).json()
        assert any("index has changed" in r for r in body["evidence"][0]["inspect"])
        assert body["compared_with"][0]["corpus_sha256"] == "f" * 64

    def test_compare_while_service_is_down(self, decided, rag, monkeypatch):  # noqa: F811
        _attach(decided)
        monkeypatch.setenv("OBS_AGENT_RAG_URL", "http://127.0.0.1:9")
        body = decided.get(f"{ITEM}/policy-evidence", params={"compare": "current"}).json()
        assert len(body["evidence"]) == 1 and body["evidence"][0]["inspect"] == []
        assert body["compared_with"][0]["status"] == "unreachable"

    def test_revoked_decision_keeps_evidence_as_history(self, decided, rag):  # noqa: F811
        _attach(decided)
        decided.request("DELETE", f"{ITEM}/decision", json={"reviewer": REVIEWER})
        (row,) = decided.get(f"{ITEM}/policy-evidence").json()["evidence"]
        assert not row["active"] and "revoked or replaced" in row["inspect"][0]


class TestSavedReview:
    def test_attach_and_list_per_review(self, client, rag):  # noqa: F811
        review_id = client.post("/reviews", json={"name": "e.csv", "csv": CSV.decode()}).json()[
            "id"
        ]
        item = f"/reviews/{review_id}/resources/{SUGGESTED}/labels/team"
        client.put(f"{item}/decision", json={"decision": "accept", "reviewer": REVIEWER})
        assert client.get(f"{item}/policy-context").json()["status"] == "ok"
        assert _attach(client, item).status_code == 200
        (row,) = client.get(f"/reviews/{review_id}/policy-evidence").json()
        assert row["resource_id"] == SUGGESTED
        assert client.get("/labels/policy-evidence").json() == []  # snapshots stay apart
        assert client.get(f"/reviews/{'0' * 64}/policy-evidence").status_code == 404


class TestCli:
    def test_lists_evidence_offline(self, client, rag, capsys, monkeypatch):  # noqa: F811
        review_id = client.post("/reviews", json={"name": "e.csv", "csv": CSV.decode()}).json()[
            "id"
        ]
        item = f"/reviews/{review_id}/resources/{SUGGESTED}/labels/team"
        client.put(f"{item}/decision", json={"decision": "accept", "reviewer": REVIEWER})
        _attach(client, item, retain_text=False)
        client.put(f"{item}/decision", json={"decision": "reject", "reviewer": REVIEWER})

        def offline(*args, **kwargs):
            raise AssertionError("the CLI must not query the retrieval service")

        monkeypatch.setattr(policy_docs, "retrieve", offline)
        assert review_cli.main(["evidence", review_id]) == 0
        (row,) = json.loads(capsys.readouterr().out)
        assert row["resource_id"] == SUGGESTED and not row["active"]
        assert row["excerpt_text"] is None and not row["text_retained"]
        assert any("revoked or replaced" in r for r in row["inspect"])

        assert review_cli.main(["evidence", review_id, "--resource", SUGGESTED]) == 2
        assert "together" in capsys.readouterr().err
        args = ["evidence", review_id, "--resource", "nope", "--label", "team"]
        assert review_cli.main(args) == 2

    def test_text_only_on_request(self, client, rag, capsys):  # noqa: F811
        review_id = client.post("/reviews", json={"name": "e.csv", "csv": CSV.decode()}).json()[
            "id"
        ]
        item = f"/reviews/{review_id}/resources/{SUGGESTED}/labels/team"
        client.put(f"{item}/decision", json={"decision": "accept", "reviewer": REVIEWER})
        _attach(client, item)
        args = ["evidence", review_id, "--resource", SUGGESTED, "--label", "team"]
        assert review_cli.main(args) == 0
        assert json.loads(capsys.readouterr().out)[0]["excerpt_text"] is None
        assert review_cli.main([*args, "--text"]) == 0
        assert json.loads(capsys.readouterr().out)[0]["excerpt_text"] == TEXT
