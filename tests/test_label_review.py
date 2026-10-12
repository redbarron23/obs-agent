"""
Tests for label review: reports, decisions, exports, the REST endpoints
and the Streamlit review page. Runs on the seeded synthetic inventory;
decisions go to a temporary file so tests never touch real state.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import label_review as lr
from label_review import DecisionError, DecisionStore

UNTAGGED = "svc-ingest"  # no team/env tag; team → data-platform
SUSPECT = "api-login-d-ew1"  # tagged security, really identity
AUTO_ENV = "platform-backend-poc-uc1"  # env untagged, 'poc' → sandbox at 95%
REVIEW_PAGE = Path(__file__).parent.parent / "pages" / "1_Label_review.py"


def _override_team(path: Path, resource_id: str) -> None:
    DecisionStore(path).set(resource_id, "team", "override", "ml")


@pytest.fixture
def store(tmp_path):
    return DecisionStore(tmp_path / "decisions.db")


class TestReports:
    def test_untagged_report(self):
        r = lr.resource_report(UNTAGGED)
        team = r["labels"]["team"]
        assert team["status"] == "untagged"
        assert team["suggested"] == "data-platform"
        assert team["action"] == "needs review"
        assert r["cost_center"]["implied"] == "CC-2002"
        assert r["needs_attention"]

    def test_suspect_report_uses_out_of_fold(self):
        team = lr.resource_report(SUSPECT)["labels"]["team"]
        assert team["status"] == "suspect"
        assert team["current"] == "security"
        assert team["suggested"] == "identity"
        assert team["source"] == "out-of-fold model"

    def test_ok_label_has_no_suggestion(self):
        team = lr.resource_report(AUTO_ENV)["labels"]["team"]
        assert team["status"] == "ok" and team["suggested"] is None

    def test_unknown_resource(self):
        assert lr.resource_report("nope") is None

    def test_coverage_matches_label_tools(self):
        cov = lr.coverage()
        assert cov["total"] == 320
        assert cov["labels"]["team"]["untagged"] == 170
        assert cov["labels"]["team"]["suspect"] == 6

    def test_queue_filters(self):
        suspects = lr.review_queue(label="team", status="suspect")
        assert {r["resource_id"] for r in suspects} >= {SUSPECT}
        assert all(r["labels"]["team"]["status"] == "suspect" for r in suspects)
        gcp = lr.review_queue(cloud="gcp")
        assert gcp and all(r["cloud"] == "gcp" for r in gcp)
        confident = lr.review_queue(label="env", min_confidence=0.9)
        assert all(r["labels"]["env"]["confidence"] >= 0.9 for r in confident)

    def test_queue_sorted_most_confident_first(self):
        conf = [r["labels"]["team"]["confidence"] for r in lr.review_queue(label="team")]
        assert conf == sorted(conf, reverse=True)


class TestDecisions:
    def test_accept_uses_suggestion(self, store):
        d = store.set(UNTAGGED, "team", "accept")
        assert d["value"] == "data-platform" and d["previous"] is None

    def test_override_normalises_value(self, store):
        assert store.set(UNTAGGED, "env", "override", "Production")["value"] == "prod"

    def test_reject_has_no_value(self, store):
        assert store.set(SUSPECT, "team", "reject")["value"] is None

    @pytest.mark.parametrize(
        "args,msg",
        [
            ((UNTAGGED, "team", "override", "martians"), "not a valid team"),
            ((AUTO_ENV, "team", "accept"), "no team suggestion"),
            ((UNTAGGED, "owner", "accept"), "Unknown label"),
            ((UNTAGGED, "team", "maybe"), "Unknown decision"),
        ],
    )
    def test_invalid(self, store, args, msg):
        with pytest.raises(DecisionError, match=msg):
            store.set(*args)

    def test_unknown_resource(self, store):
        with pytest.raises(KeyError):
            store.set("nope", "team", "accept")

    def test_replace_and_delete(self, store):
        store.set(UNTAGGED, "team", "accept")
        store.set(UNTAGGED, "team", "override", "ml")
        assert [d["value"] for d in store.all()] == ["ml"]
        assert store.delete(UNTAGGED, "team")
        assert not store.delete(UNTAGGED, "team")
        assert store.all() == []

    def test_persists_across_instances(self, store):
        store.set(UNTAGGED, "team", "accept")
        assert DecisionStore(store.path).get(UNTAGGED, "team")["value"] == "data-platform"

    def test_feedback_retains_the_model_context_for_each_action(self, store):
        report = lr.resource_report(UNTAGGED)["labels"]["team"]
        store.set(UNTAGGED, "team", "accept")
        store.set(UNTAGGED, "team", "override", "ml")

        feedback = store.feedback()
        assert [entry["decision"] for entry in feedback] == ["accept", "override"]
        assert feedback[0]["suggested"] == report["suggested"]
        assert feedback[0]["confidence"] == report["confidence"]
        assert feedback[0]["reason"] == report["reason"]
        assert feedback[1]["value"] == "ml"

    def test_imports_legacy_json(self, tmp_path):
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
                        "decided_at": "2026-01-01T00:00:00+00:00",
                    }
                ]
            )
        )
        store = DecisionStore(legacy)
        assert store.path == tmp_path / "decisions.db"
        assert store.get(UNTAGGED, "team")["value"] == "data-platform"

    def test_concurrent_writes_do_not_lose_decisions(self, store):
        resource_ids = [r["resource_id"] for r in lr.all_reports()[:4]]

        with ProcessPoolExecutor(max_workers=4) as executor:
            list(executor.map(_override_team, [store.path] * len(resource_ids), resource_ids))
        assert {d["resource_id"] for d in store.all()} == set(resource_ids)


class TestEditorChanges:
    def _df(self, rows):
        return pd.DataFrame(rows, columns=["resource_id", "decision", "override"])

    def test_detects_only_changed_rows(self):
        before = self._df([["a", "", None], ["b", "accept", None], ["c", "", None]])
        after = self._df([["a", "accept", None], ["b", "accept", None], ["c", None, float("nan")]])
        assert lr.editor_changes(before, after) == [("a", "accept", None)]

    def test_override_value_alone_means_override(self):
        before = self._df([["a", "", None]])
        after = self._df([["a", "", "ml"]])
        assert lr.editor_changes(before, after) == [("a", "override", "ml")]

    def test_clearing_a_decision(self):
        before = self._df([["a", "reject", None]])
        after = self._df([["a", "", None]])
        assert lr.editor_changes(before, after) == [("a", None, None)]


class TestExport:
    def test_only_reviewed_without_auto(self, store):
        store.set(UNTAGGED, "team", "accept")
        changes = lr.pending_changes(store, include_auto=False)
        assert [c["resource_id"] for c in changes] == [UNTAGGED]
        assert changes[0]["tags"]["cost_center"] == ("CC-2002", "accept")

    def test_reject_is_not_exported(self, store):
        store.set(SUSPECT, "team", "reject")
        assert lr.pending_changes(store, include_auto=False) == []

    def test_override_beats_auto(self, store):
        store.set(AUTO_ENV, "env", "override", "dev")
        change = next(c for c in lr.pending_changes(store) if c["resource_id"] == AUTO_ENV)
        assert change["tags"]["env"] == ("dev", "override")

    def test_auto_apply_included_by_default(self, store):
        auto = lr.pending_changes(store)
        assert len(auto) > 50
        assert all(how == "auto-apply" for c in auto for _, how in c["tags"].values())

    def test_formats(self, store):
        store.set(UNTAGGED, "team", "accept")  # azure
        store.set(AUTO_ENV, "env", "accept")  # gcp
        csv_out = lr.export(store, "csv", include_auto=False)
        assert csv_out.splitlines()[0] == "resource_id,cloud,label,previous,new,how"
        assert f"{UNTAGGED},azure,team,,data-platform,accept" in csv_out
        sh = lr.export(store, "sh", include_auto=False)
        assert sh.startswith("#!/usr/bin/env bash")
        assert f"az tag update --resource-id /subscriptions/{UNTAGGED} --operation Merge" in sh
        assert f"gcloud projects update {AUTO_ENV} --update-labels=env=sandbox" in sh
        tf = json.loads(lr.export(store, "tfvars", include_auto=False))
        assert tf["subscription_tags"][UNTAGGED]["cost_center"] == "CC-2002"
        assert tf["project_labels"][AUTO_ENV] == {"env": "sandbox"}

    def test_change_plan_is_reviewed_only_and_has_approval_evidence(self, store):
        store.set(UNTAGGED, "team", "accept")

        plan = json.loads(lr.export(store, "plan", include_auto=True))

        assert plan["reviewed_only"] is True
        assert [change["resource_id"] for change in plan["changes"]] == [UNTAGGED]
        assert plan["changes"][0]["tags"]["team"]["new"] == "data-platform"
        assert plan["approvals"][0]["confidence"] is not None

    def test_gcp_values_lowercased(self, store):
        gcp_untagged = next(
            r["resource_id"] for r in lr.review_queue(label="team", status="untagged", cloud="gcp")
        )
        store.set(gcp_untagged, "team", "override", "payments")
        sh = lr.export(store, "sh", include_auto=False)
        assert "cost_center=cc-1001" in sh

    def test_unknown_format(self, store):
        with pytest.raises(ValueError):
            lr.export(store, "xml")


class TestLabelAPI:
    @pytest.fixture
    def client(self, store, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        monkeypatch.setattr(api, "decisions", store)
        return TestClient(api.app, headers={"X-API-Key": "test-api-key"})

    def test_coverage(self, client):
        assert client.get("/labels/coverage").json()["labels"]["team"]["untagged"] == 170

    def test_queue_and_pagination(self, client):
        page = client.get("/resources", params={"label": "team", "limit": 5}).json()
        assert len(page["items"]) == 5 and page["total"] >= 170
        nxt = client.get("/resources", params={"label": "team", "limit": 5, "offset": 5}).json()
        assert page["items"][0]["resource_id"] != nxt["items"][0]["resource_id"]

    def test_queue_rejects_bad_filter(self, client):
        assert client.get("/resources", params={"cloud": "aws"}).status_code == 422

    def test_resource_labels(self, client):
        body = client.get(f"/resources/{UNTAGGED}/labels").json()
        assert body["labels"]["team"]["suggested"] == "data-platform"
        assert client.get("/resources/nope/labels").status_code == 404

    def test_decision_lifecycle(self, client):
        url = f"/resources/{UNTAGGED}/labels/team/decision"
        assert client.put(url, json={"decision": "accept"}).json()["value"] == "data-platform"
        assert len(client.get("/labels/decisions").json()) == 1
        feedback = client.get("/labels/feedback").json()
        assert feedback[0]["suggested"] == "data-platform"
        assert feedback[0]["confidence"] is not None
        assert client.delete(url).status_code == 200
        assert client.delete(url).status_code == 404

    def test_decision_errors(self, client):
        url = f"/resources/{UNTAGGED}/labels/team/decision"
        assert client.put(url, json={"decision": "override", "value": "x"}).status_code == 422
        assert (
            client.put(
                f"/resources/{UNTAGGED}/labels/owner/decision", json={"decision": "accept"}
            ).status_code
            == 422
        )
        assert (
            client.put(
                "/resources/nope/labels/team/decision", json={"decision": "accept"}
            ).status_code
            == 404
        )

    @pytest.mark.parametrize(
        "fmt,media,filename",
        [
            ("csv", "text/csv", "label-changes.csv"),
            ("sh", "text/x-shellscript", "apply-labels.sh"),
            ("tfvars", "application/json", "labels.auto.tfvars.json"),
            ("plan", "application/json", "approved-label-change-plan.json"),
        ],
    )
    def test_export(self, client, fmt, media, filename):
        r = client.get("/labels/export", params={"format": fmt})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(media)
        assert filename in r.headers["content-disposition"]

    def test_requires_api_key_when_set(self, client, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "secret")
        assert client.get("/labels/coverage").status_code == 401
        assert client.get("/labels/coverage", headers={"X-API-Key": "secret"}).status_code == 200


class TestReviewPage:
    @pytest.fixture
    def page(self, store, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))
        return AppTest.from_file(str(REVIEW_PAGE), default_timeout=60)

    def test_renders(self, page):
        at = page.run()
        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Untagged team"] == "170"
        assert metrics["Reviewed"] == "0"

    def test_accept_all_auto(self, page, store):
        at = page.run()
        next(b for b in at.button if b.label.startswith("Accept all")).click().run()
        assert not at.exception
        decided = store.all()
        assert decided and all(d["decision"] == "accept" for d in decided)
        assert {m.label: m.value for m in at.metric}["Reviewed"] == str(len(decided))

    def test_filters(self, page):
        at = page.run()
        at.sidebar.selectbox[0].set_value("suspect").run()
        assert at.subheader[0].value.startswith("6 resources")

    def test_demo_mode_isolates_sessions(self, monkeypatch, tmp_path):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DEMO", "1")
        monkeypatch.chdir(tmp_path)  # a shared decisions file would land here
        first = AppTest.from_file(str(REVIEW_PAGE), default_timeout=60).run()
        next(b for b in first.button if b.label.startswith("Accept all")).click().run()
        assert {m.label: m.value for m in first.metric}["Reviewed"] != "0"
        second = AppTest.from_file(str(REVIEW_PAGE), default_timeout=60).run()
        assert {m.label: m.value for m in second.metric}["Reviewed"] == "0"
        assert not (tmp_path / "label_decisions.json").exists()

    def test_demo_mode_disables_chat(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DEMO", "1")
        at = AppTest.from_file(str(REVIEW_PAGE.parent.parent / "app.py"), default_timeout=60).run()
        assert not at.exception
        assert at.chat_input[0].disabled
        # No LLM controls or local-model warning to confuse demo visitors.
        assert not at.sidebar.selectbox
        assert not at.sidebar.warning

    def test_demo_mode_refuses_forged_chat_input(self, monkeypatch):
        """A chat value sent despite the disabled widget must never be processed.

        Streamlit itself drops values for disabled widgets, and app.py also discards
        the prompt in demo mode; this guards the combined behaviour.
        """
        from unittest.mock import patch

        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DEMO", "1")
        at = AppTest.from_file(str(REVIEW_PAGE.parent.parent / "app.py"), default_timeout=60).run()
        widget = at.chat_input[0]
        # AppTest refuses to set a disabled widget, as a real browser would; a forged
        # websocket message is not a browser, so bypass that check to simulate one.
        with patch.object(type(widget), "_assert_can_interact", lambda self: None):
            widget.set_value("spend your credits").run()
        assert not at.exception
        assert at.session_state["messages"] == []  # nothing echoed, nothing sent to an LLM
        assert not at.chat_message

    def test_demo_mode_from_streamlit_secrets(self, monkeypatch, tmp_path):
        from streamlit.testing.v1 import AppTest

        monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(str(REVIEW_PAGE), default_timeout=60)
        at.secrets["OBS_AGENT_DEMO"] = "1"
        at.run()
        assert not at.exception
        assert at.info and "Demo" in at.info[0].value
