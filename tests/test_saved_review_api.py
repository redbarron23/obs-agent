"""Saved private reviews through the REST API and CLI (issue #47).

Every test uses a temporary OBS_AGENT_REVIEW_DIR; nothing calls an LLM or a cloud API.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

import api
import label_review as lr
import review_cli
from inventory_import import REQUIRED_COLUMNS
from review_workspaces import load_review, saved_reviews

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
# Two extra untagged resources whose team is withheld for insufficient evidence.
CSV = (
    SAMPLE
    + b"azure,sub-0a1b2c3d,mg-payments,,,AKS|SQL,110.00,0.90,0.20,,,\n"
    + b"gcp,proj-482913,folders-data,zoe,zoe,BigQuery|GCS,40.00,0.30,0.40,,,\n"
)
ABSTAINED = "sub-0a1b2c3d"
SUGGESTED = "gcp-demo-003"  # team -> data-platform, env -> dev
REVIEWER = "reviewer-7f3a"
UNKNOWN_ID = "0" * 64


@pytest.fixture(autouse=True)
def _private_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
    monkeypatch.delenv("OBS_AGENT_REVIEWER")
    synthetic = lr.DecisionStore(tmp_path / "synthetic.db")
    monkeypatch.setattr(api, "decisions", synthetic)
    return synthetic


@pytest.fixture
def client():
    return TestClient(api.app, headers={"X-API-Key": "test-api-key"})


@pytest.fixture
def review_id(client):
    r = client.post("/reviews", json={"name": "estate.csv", "csv": CSV.decode()})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _decide(client, review_id, resource, label, decision, value=None):
    return client.put(
        f"/reviews/{review_id}/resources/{resource}/labels/{label}/decision",
        json={"decision": decision, "value": value, "reviewer": REVIEWER},
    )


def _plan(text: str) -> dict:
    plan = json.loads(text)
    plan.pop("generated_at")
    return plan


def _cli(capsys, *argv) -> tuple[int, str, str]:
    code = review_cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


class TestApiWorkflow:
    def test_pipeline_submits_csv_and_retrieves_reviewed_plan_without_llm(self, client):
        with (
            patch("api.run", side_effect=AssertionError("no LLM")),
            patch("api.stream_run", side_effect=AssertionError("no LLM")),
        ):
            created = client.post("/reviews", json={"name": "estate.csv", "csv": CSV.decode()})
            assert created.status_code == 200
            summary = created.json()
            rid = summary["id"]
            assert summary["name"] == "estate.csv" and summary["resources"] == 7
            assert client.get("/reviews").json() == [
                {k: summary[k] for k in ("id", "name", "created_at")}
            ]
            assert client.get(f"/reviews/{rid}").json() == summary

            queue = client.get(f"/reviews/{rid}/resources", params={"label": "team"}).json()
            assert queue["items"][0]["resource_id"] == SUGGESTED  # highest score first
            assert {i["resource_id"] for i in queue["items"]} == {
                SUGGESTED,
                ABSTAINED,
                "proj-482913",
            }
            assert _decide(client, rid, SUGGESTED, "team", "accept").status_code == 200
            assert (
                _decide(client, rid, ABSTAINED, "team", "override", "payments").status_code == 200
            )
            plan = _plan(client.get(f"/reviews/{rid}/export", params={"format": "plan"}).text)

        assert plan["reviewed_only"] is True
        assert plan["inventory_id"] == summary["inventory_id"]
        assert {c["resource_id"]: c["tags"]["team"]["new"] for c in plan["changes"]} == {
            SUGGESTED: "data-platform",
            ABSTAINED: "payments",
        }
        assert {a["reviewer"] for a in plan["approvals"]} == {REVIEWER}
        # The synthetic store is untouched by private-review decisions.
        assert api.decisions.all() == []

    def test_matches_library_functions_used_by_the_ui(self, client, review_id):
        inventory, _ = load_review(review_id)
        assert client.get(f"/reviews/{review_id}/coverage").json() == lr.coverage(inventory)
        report = client.get(f"/reviews/{review_id}/resources/{SUGGESTED}/labels").json()
        assert report == lr.resource_report(SUGGESTED, inventory)
        items = client.get(f"/reviews/{review_id}/resources").json()["items"]
        assert items == lr.review_queue(inventory=inventory)

    def test_decisions_feedback_backlog_and_undo(self, client, review_id):
        assert _decide(client, review_id, SUGGESTED, "env", "accept").status_code == 200
        decisions = client.get(f"/reviews/{review_id}/decisions").json()
        assert [(d["resource_id"], d["label"], d["reviewer"]) for d in decisions] == [
            (SUGGESTED, "env", REVIEWER)
        ]
        assert client.get(f"/reviews/{review_id}/feedback").json()[0]["confidence"] is not None
        assert client.get(f"/reviews/{review_id}/feedback/metrics").json()["reviewed"] == 1
        backlog = client.get(f"/reviews/{review_id}/backlog").json()
        assert backlog["reviewed_items"] == 1 and backlog["pending_insufficient_evidence"] == 2
        metrics = client.get("/ops/metrics", params={"review_id": review_id}).json()
        assert metrics["backlog"] == backlog
        url = f"/reviews/{review_id}/resources/{SUGGESTED}/labels/env/decision"
        assert client.delete(url).status_code == 200
        assert client.delete(url).status_code == 404
        assert client.get(f"/reviews/{review_id}/decisions").json() == []

    def test_reviewer_is_required(self, client, review_id):
        url = f"/reviews/{review_id}/resources/{SUGGESTED}/labels/team/decision"
        r = client.put(url, json={"decision": "accept"})
        assert r.status_code == 422 and "reviewer" in r.json()["detail"]
        assert client.get(f"/reviews/{review_id}/decisions").json() == []

    def test_unknown_resource_is_404(self, client, review_id):
        # svc-ingest exists in the synthetic estate, not in this review.
        assert client.get(f"/reviews/{review_id}/resources/svc-ingest/labels").status_code == 404
        assert _decide(client, review_id, "svc-ingest", "team", "accept").status_code == 404


class TestAbstention:
    def test_withheld_labels_behave_as_in_the_ui(self, client, review_id):
        queue = client.get(
            f"/reviews/{review_id}/resources", params={"label": "team", "evidence": "insufficient"}
        ).json()
        assert {i["resource_id"] for i in queue["items"]} == {ABSTAINED, "proj-482913"}
        team = client.get(f"/reviews/{review_id}/resources/{ABSTAINED}/labels").json()["labels"][
            "team"
        ]
        assert team["suggested"] is None and team["action"] == "insufficient evidence"
        assert team["evidence"]["status"] == "missing_signals"
        cov = client.get(f"/reviews/{review_id}/coverage").json()["labels"]["team"]
        assert cov["insufficient_evidence"] == 2
        for decision in ("accept", "reject"):
            r = _decide(client, review_id, ABSTAINED, "team", decision)
            assert r.status_code == 422 and "no team suggestion" in r.json()["detail"]
        # Never exported unreviewed, even with include_auto.
        csv = client.get(
            f"/reviews/{review_id}/export", params={"format": "csv", "include_auto": True}
        ).text
        assert f"{ABSTAINED},azure,team" not in csv
        override = _decide(client, review_id, ABSTAINED, "team", "override", "payments")
        assert override.status_code == 200 and override.json()["suggested"] is None
        plan = _plan(client.get(f"/reviews/{review_id}/export", params={"format": "plan"}).text)
        assert plan["approvals"][0]["source"] == "abstain"


class TestParity:
    def test_ui_api_and_cli_plans_match_for_the_same_decisions(
        self, client, review_id, monkeypatch, capsys
    ):
        assert _decide(client, review_id, SUGGESTED, "team", "accept").status_code == 200
        code, _, _ = _cli(
            capsys, "decide", review_id, ABSTAINED, "team", "override", "--value", "payments",
            "--reviewer", REVIEWER,
        )  # fmt: skip
        assert code == 0
        api_plan = _plan(client.get(f"/reviews/{review_id}/export", params={"format": "plan"}).text)
        code, out, _ = _cli(capsys, "export", review_id, "--format", "plan")
        assert code == 0 and _plan(out) == api_plan

        downloads = {}
        original = MemoryMediaFileStorage.load_and_get_id

        def capture(self, path_or_data, mimetype, kind, filename=None):
            downloads[filename] = path_or_data
            return original(self, path_or_data, mimetype, kind, filename)

        monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", capture)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["review_id"] = review_id
        at.run()
        assert not at.exception
        ui_plan = downloads["approved-label-change-plan.json"]
        ui_plan = ui_plan.decode() if isinstance(ui_plan, bytes) else ui_plan
        assert _plan(ui_plan) == api_plan
        assert len(api_plan["changes"]) == 2

    def test_cli_queue_and_coverage_match_api(self, client, review_id, capsys):
        code, out, _ = _cli(capsys, "queue", review_id, "--evidence", "insufficient")
        assert code == 0
        assert (
            json.loads(out)
            == client.get(
                f"/reviews/{review_id}/resources", params={"evidence": "insufficient"}
            ).json()["items"]
        )
        code, out, _ = _cli(capsys, "coverage", review_id)
        assert json.loads(out) == client.get(f"/reviews/{review_id}/coverage").json()


class TestPersistence:
    def test_imports_survive_a_restart_and_isolate_snapshots(self, client, review_id, tmp_path):
        assert _decide(client, review_id, SUGGESTED, "team", "accept").status_code == 200
        expected = _plan(client.get(f"/reviews/{review_id}/export", params={"format": "plan"}).text)
        # A separate process sees the same saved review and decisions.
        env = os.environ | {"OBS_AGENT_REVIEW_DIR": str(tmp_path / "reviews")}
        done = subprocess.run(
            [sys.executable, "review_cli.py", "export", review_id, "--format", "plan"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr
        assert _plan(done.stdout) == expected
        # Re-submitting the same snapshot and vocabulary reopens the same review.
        lr._review_cache.clear()
        again = client.post("/reviews", json={"name": "renamed.csv", "csv": CSV.decode()})
        assert again.json()["id"] == review_id
        assert len(client.get(f"/reviews/{review_id}/decisions").json()) == 1
        # Changed data or vocabulary is a separate review with no decisions.
        changed = client.post(
            "/reviews",
            json={"name": "changed.csv", "csv": CSV.decode().replace("36.20", "37.20")},
        ).json()["id"]
        vocabulary = json.loads((ROOT / "fixtures/label_config_example.json").read_text())
        other_vocab = client.post(
            "/reviews", json={"name": "v.csv", "csv": CSV.decode(), "vocabulary": vocabulary}
        ).json()["id"]
        assert len({review_id, changed, other_vocab}) == 3
        for rid in (changed, other_vocab):
            assert client.get(f"/reviews/{rid}/decisions").json() == []
            assert (
                _plan(client.get(f"/reviews/{rid}/export", params={"format": "plan"}).text)[
                    "changes"
                ]
                == []
            )

    def test_stale_approvals_are_excluded(self, client, review_id, monkeypatch):
        assert _decide(client, review_id, SUGGESTED, "team", "accept").status_code == 200
        inventory, _ = load_review(review_id)
        original = lr.review_context(inventory)
        monkeypatch.setattr(
            lr, "review_context", lambda inv=None: original | {"model_version": "x"}
        )
        plan = _plan(client.get(f"/reviews/{review_id}/export", params={"format": "plan"}).text)
        assert plan["changes"] == [] and plan["approvals"] == []
        assert client.get(f"/reviews/{review_id}/decisions", params={"current": True}).json() == []
        assert len(client.get(f"/reviews/{review_id}/decisions").json()) == 1


class TestValidation:
    @pytest.mark.parametrize(
        "body, message",
        [
            ({"csv": SAMPLE.decode().replace("azure,", ","), "name": "x"}, "cloud must be"),
            ({"csv": "cloud,resource_id\n", "name": "x"}, "Missing required columns"),
            ({"csv": SAMPLE.decode(), "name": "x", "vocabulary": {"team": {}}}, "Vocabulary"),
            (
                {"csv": SAMPLE.decode(), "name": "x", "column_mapping": {"cloud": "cloud"}},
                "Map a source column",
            ),
        ],
    )
    def test_invalid_import_is_rejected_without_saving(self, client, body, message):
        r = client.post("/reviews", json=body)
        assert r.status_code == 422 and message in r.json()["detail"]
        assert saved_reviews() == []

    def test_empty_body_fields_are_rejected(self, client):
        assert client.post("/reviews", json={"name": "", "csv": "x"}).status_code == 422
        assert client.post("/reviews", json={"name": "x", "csv": ""}).status_code == 422

    def test_column_mapping_is_applied_like_the_import_page(self, client):
        renamed = CSV.replace(b"cloud,resource_id", b"provider,subscription", 1)
        mapping = {f: f for f in REQUIRED_COLUMNS} | {
            "cloud": "provider",
            "resource_id": "subscription",
        }
        r = client.post(
            "/reviews", json={"name": "m.csv", "csv": renamed.decode(), "column_mapping": mapping}
        )
        assert r.status_code == 200
        assert (
            r.json()["inventory_id"]
            == client.post("/reviews", json={"name": "x", "csv": CSV.decode()}).json()[
                "inventory_id"
            ]
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "ABC",
            "0" * 63,
            "0" * 65,
            "A" * 64,
            "g" * 64,
            "..%2F..%2Fetc%2Fpasswd",
            "%2e%2e",
            f"{'0' * 64}%0A",
            f"..%2F{'0' * 64}",
        ],
    )
    def test_invalid_or_traversal_ids_are_rejected(self, client, review_id, bad, tmp_path):
        before = sorted(p.name for p in tmp_path.rglob("*"))
        for path in (f"/reviews/{bad}", f"/reviews/{bad}/resources", f"/reviews/{bad}/export"):
            assert client.get(path).status_code in (404, 422)
        r = client.put(
            f"/reviews/{bad}/resources/{SUGGESTED}/labels/team/decision",
            json={"decision": "accept", "reviewer": REVIEWER},
        )
        assert r.status_code in (404, 405, 422)
        assert client.get("/ops/metrics", params={"review_id": bad}).status_code == 422
        assert sorted(p.name for p in tmp_path.rglob("*")) == before

    def test_unknown_id_never_falls_back_to_synthetic_data(self, client, tmp_path):
        base = f"/reviews/{UNKNOWN_ID}"
        for path in (base, f"{base}/coverage", f"{base}/resources", f"{base}/export"):
            r = client.get(path)
            assert r.status_code == 404 and r.json()["detail"] == "Saved review not found"
        assert client.get(f"{base}/resources/svc-ingest/labels").status_code == 404
        assert _decide(client, UNKNOWN_ID, "svc-ingest", "team", "accept").status_code == 404
        assert client.get("/ops/metrics", params={"review_id": UNKNOWN_ID}).status_code == 404
        assert api.decisions.all() == []
        assert not (tmp_path / "reviews" / f"{UNKNOWN_ID}.db").exists()


def _review_routes(review_id: str) -> list[tuple[str, str]]:
    calls = []
    for route in api.app.routes:
        if not getattr(route, "path", "").startswith("/reviews"):
            continue
        path = route.path.format(review_id=review_id, resource_id=SUGGESTED, label="team")
        calls += [(method, path) for method in route.methods]
    return calls


class TestAuthAndDemo:
    def test_every_review_route_requires_the_api_key(self, review_id):
        calls = _review_routes(review_id)
        assert len(calls) == 13
        for headers in ({}, {"X-API-Key": "wrong"}):
            anonymous = TestClient(api.app, headers=headers)
            for method, path in calls:
                assert anonymous.request(method, path, json={}).status_code == 401, (method, path)
            assert anonymous.get("/ops/metrics", params={"review_id": review_id}).status_code == 401

    def test_unconfigured_key_refuses_review_routes(self, review_id, monkeypatch):
        monkeypatch.delenv("OBS_AGENT_API_KEY")
        anonymous = TestClient(api.app)
        for method, path in _review_routes(review_id):
            assert anonymous.request(method, path, json={}).status_code == 503

    def test_demo_mode_blocks_private_imports_and_reviews(
        self, client, review_id, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setenv("OBS_AGENT_DEMO", "1")
        r = client.post("/reviews", json={"name": "x.csv", "csv": SAMPLE.decode()})
        assert r.status_code == 403 and "demo" in r.json()["detail"]
        for method, path in _review_routes(review_id):
            assert client.request(method, path, json={}).status_code == 403, (method, path)
        assert client.get("/ops/metrics", params={"review_id": review_id}).status_code == 403
        # The synthetic demo API keeps working.
        assert client.get("/labels/coverage").status_code == 200
        source = tmp_path / "x.csv"
        source.write_bytes(SAMPLE)
        code, _, err = _cli(capsys, "import", str(source))
        assert code == 2 and "demo" in err
        assert _cli(capsys, "list")[0] == 2
        monkeypatch.delenv("OBS_AGENT_DEMO")
        assert [r["id"] for r in saved_reviews()] == [review_id]


class TestCli:
    def test_import_list_decide_export(self, capsys, tmp_path):
        source = tmp_path / "estate.csv"
        source.write_bytes(CSV)
        code, out, _ = _cli(capsys, "import", str(source))
        assert code == 0
        rid = json.loads(out)["id"]
        assert json.loads(_cli(capsys, "list")[1])[0]["name"] == "estate.csv"
        assert json.loads(_cli(capsys, "show", rid)[1])["resources"] == 7
        code, out, _ = _cli(capsys, "resource", rid, SUGGESTED)
        assert json.loads(out)["labels"]["team"]["suggested"] == "data-platform"
        code, _, err = _cli(capsys, "decide", rid, SUGGESTED, "team", "accept")
        assert code == 2 and "reviewer" in err
        code, _, err = _cli(capsys, "decide", rid, ABSTAINED, "team", "accept", "--reviewer", "a")
        assert code == 2 and "no team suggestion" in err
        assert _cli(capsys, "decide", rid, SUGGESTED, "team", "accept", "--reviewer", "a")[0] == 0
        assert len(json.loads(_cli(capsys, "decisions", rid, "--current")[1])) == 1
        assert json.loads(_cli(capsys, "backlog", rid)[1])["reviewed_items"] == 1
        queue = json.loads(_cli(capsys, "queue", rid, "--label", "team", "--hide-reviewed")[1])
        assert SUGGESTED not in {r["resource_id"] for r in queue}
        plan_file = tmp_path / "plan.json"
        assert _cli(capsys, "export", rid, "--output", str(plan_file))[0] == 0
        assert json.loads(plan_file.read_text())["changes"][0]["resource_id"] == SUGGESTED
        code, out, _ = _cli(capsys, "export", rid, "--format", "csv")
        assert out.splitlines()[1].startswith(f"{SUGGESTED},gcp,team")
        assert _cli(capsys, "undo", rid, SUGGESTED, "team")[0] == 0
        assert _cli(capsys, "undo", rid, SUGGESTED, "team")[0] == 2

    def test_invalid_input_and_unknown_ids(self, capsys, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(SAMPLE.replace(b"azure,", b","))
        code, _, err = _cli(capsys, "import", str(bad))
        assert code == 2 and "cloud must be" in err
        assert saved_reviews() == []
        code, _, err = _cli(capsys, "show", UNKNOWN_ID)
        assert code == 2 and "Unknown saved review" in err
        with pytest.raises(SystemExit) as exc:
            review_cli.main(["show", "../escape"])
        assert exc.value.code == 2
        assert not (tmp_path / "reviews").exists()


def test_inventory_and_reviewer_text_are_not_logged(client, caplog):
    # INFO is the production level; DEBUG tracebacks are an explicit operator opt-in.
    caplog.set_level(logging.INFO)
    invalid = SAMPLE.decode().replace("azure,", "secret-cloud-zz,")
    assert client.post("/reviews", json={"name": "x", "csv": invalid}).status_code == 422
    rid = client.post("/reviews", json={"name": "x", "csv": CSV.decode()}).json()["id"]
    _decide(client, rid, SUGGESTED, "team", "accept")
    _decide(client, rid, SUGGESTED, "team", "override", "free-text-value-q9")
    assert caplog.records  # the operations were measured
    for secret in ("secret-cloud-zz", "alice|bob", REVIEWER, "free-text-value-q9", "fatima"):
        assert secret not in caplog.text
