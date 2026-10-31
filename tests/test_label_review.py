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
from inventory_import import load_inventory_csv
from label_review import DecisionError, DecisionStore

UNTAGGED = "svc-ingest"  # no team/env tag; team → data-platform
SUSPECT = "api-login-d-ew1"  # tagged security, really identity
AUTO_ENV = "platform-backend-poc-uc1"  # env untagged, 'poc' → sandbox at 95%
REVIEW_PAGE = Path(__file__).parent.parent / "pages" / "1_Label_review.py"
FEEDBACK_FIXTURE = Path(__file__).parent.parent / "fixtures" / "anonymized_review_feedback.json"
INVENTORY_SAMPLE = Path(__file__).parent.parent / "fixtures" / "sanitized_inventory_sample.csv"


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
        conf = [r["labels"]["team"]["confidence"] or 0 for r in lr.review_queue(label="team")]
        assert conf == sorted(conf, reverse=True)

    def test_imported_inventory_uses_the_same_review_workflow(self, tmp_path):
        inventory = load_inventory_csv(INVENTORY_SAMPLE)
        report = lr.resource_report("gcp-demo-003", inventory)
        assert report["labels"]["team"]["status"] == "untagged"
        assert lr.coverage(inventory)["total"] == 5
        store = DecisionStore(tmp_path / "imported.db")
        store.set("gcp-demo-003", "team", "accept", inventory=inventory)
        assert "gcp-demo-003" in lr.export(store, "csv", include_auto=False, inventory=inventory)


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

    def test_feedback_metrics_tracks_reviewer_agreement_by_month(self):
        feedback = [
            {
                "resource_id": "az-01",
                "label": "team",
                "decision": "accept",
                "value": "payments",
                "suggested": "payments",
                "previous": None,
                "decided_at": "2026-01-05T12:00:00+00:00",
            },
            {
                "resource_id": "az-02",
                "label": "team",
                "decision": "override",
                "value": "identity",
                "suggested": "security",
                "previous": None,
                "decided_at": "2026-02-05T12:00:00+00:00",
            },
            {
                "resource_id": "az-03",
                "label": "team",
                "decision": "reject",
                "value": None,
                "suggested": "security",
                "previous": "identity",
                "decided_at": "2026-02-06T12:00:00+00:00",
            },
            {
                "resource_id": "az-04",
                "label": "env",
                "decision": "reject",
                "value": None,
                "suggested": "prod",
                "previous": None,
                "decided_at": "2026-02-06T12:00:00+00:00",
            },
        ]

        metrics = lr.feedback_metrics(feedback)

        assert metrics["reviewed"] == 4 and metrics["scored"] == 3
        assert metrics["unresolved"] == 1
        assert metrics["auto_apply_threshold"] == 0.9
        assert metrics["labels"]["team"]["precision"] == 0.333
        assert metrics["labels"]["team"]["recall"] == 0.333
        assert metrics["labels"]["team"]["confidence_buckets"][0]["bucket"] == "below 90%"
        assert [p["period"] for p in metrics["labels"]["team"]["periods"]] == [
            "2026-01",
            "2026-02",
        ]

    def test_anonymized_feedback_fixture_matches_metric_contract(self):
        metrics = lr.feedback_metrics(json.loads(FEEDBACK_FIXTURE.read_text()))
        assert metrics["reviewed"] == 8
        assert metrics["scored"] == 7
        assert metrics["unresolved"] == 1
        assert set(metrics["labels"]) == {"team", "env"}

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

    def test_auto_apply_requires_opt_in(self, store):
        assert lr.pending_changes(store) == []
        auto = lr.pending_changes(store, include_auto=True)
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

    @pytest.mark.parametrize("fmt", lr.EXPORT_FORMATS)
    def test_invalid_export_returns_actionable_error(self, client, store, monkeypatch, fmt):
        resource = next(r for r in lr.review_queue(cloud="gcp", label="team", status="untagged"))
        store.set(resource["resource_id"], "team", "override", "payments")
        inference = lr.get_inference()
        monkeypatch.setattr(inference, "cost_center_of", {"payments": "CC-1,env=prod"})
        response = client.get("/labels/export", params={"format": fmt})
        assert response.status_code == 422
        assert "invalid cost_center" in response.json()["detail"]

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

    def test_feedback_metrics(self, client):
        client.put(f"/resources/{UNTAGGED}/labels/team/decision", json={"decision": "accept"})
        report = client.get("/labels/feedback/metrics")
        assert report.status_code == 200
        assert report.json()["labels"]["team"]["scored"] == 1

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
        at.switch_page("pages/3_Cost_chat.py").run()
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
        at.switch_page("pages/3_Cost_chat.py").run()
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


ABSTAINED = "lake-warehouse-d"  # team withheld: conflicting ownership evidence


class TestAbstention:
    """Insufficient evidence: no suggestion, never auto-applied, overrides still work."""

    @pytest.fixture
    def client(self, store, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        monkeypatch.setattr(api, "decisions", store)
        return TestClient(api.app, headers={"X-API-Key": "test-api-key"})

    @pytest.fixture
    def small(self):
        source = INVENTORY_SAMPLE.read_bytes()
        source += b"azure,sub-0a1b2c3d,mg-payments,,,AKS|SQL,110.00,0.90,0.20,,,\n"
        source += b"gcp,proj-482913,folders-data,zoe,zoe,BigQuery|GCS,40.00,0.30,0.40,,,\n"
        return load_inventory_csv(source)

    def test_report_withholds_the_suggestion(self):
        r = lr.resource_report(ABSTAINED)
        team = r["labels"]["team"]
        assert team["status"] == "untagged"
        assert team["suggested"] is None and team["confidence"] is None
        assert team["action"] == lr.INSUFFICIENT_EVIDENCE and team["source"] == "abstain"
        assert team["evidence"]["status"] == "conflicting"
        assert team["evidence"]["withheld"]["value"] == "observability"
        assert "override" in team["reason"]
        assert r["cost_center"]["implied"] is None
        # Supported suggestions say why too.
        assert lr.resource_report(UNTAGGED)["labels"]["team"]["evidence"]["status"] == "supported"

    def test_visible_in_queue_and_coverage(self):
        queue = lr.review_queue(label="team", evidence="insufficient")
        assert ABSTAINED in {r["resource_id"] for r in queue}
        assert all(r["labels"]["team"]["suggested"] is None for r in queue)
        assert ABSTAINED in {r["resource_id"] for r in lr.review_queue(label="team")}
        assert ABSTAINED not in {
            r["resource_id"] for r in lr.review_queue(label="team", min_confidence=0.05)
        }
        cov = lr.coverage()["labels"]["team"]
        assert cov["insufficient_evidence"] == len(queue) == 5
        assert cov["insufficient_evidence_reasons"]["conflicting"] == 5
        assert cov["untagged"] == 170

    def test_reasons_are_distinguished(self, small):
        reports = {r["resource_id"]: r["labels"]["team"] for r in lr.reports(small)}
        assert reports["sub-0a1b2c3d"]["evidence"]["status"] == "missing_signals"
        assert reports["proj-482913"]["evidence"]["status"] == "insufficient_support"
        assert reports["gcp-demo-003"]["evidence"]["status"] == "supported"
        reasons = lr.coverage(small)["labels"]["team"]["insufficient_evidence_reasons"]
        assert reasons == {"missing_signals": 1, "insufficient_support": 1, "conflicting": 0}

    def test_cannot_be_accepted_or_rejected(self, store):
        for decision in ("accept", "reject"):
            with pytest.raises(DecisionError, match="no team suggestion"):
                store.set(ABSTAINED, "team", decision)
        assert store.all() == []

    def test_never_exported_unreviewed(self, store):
        # The resource's env is auto-apply eligible; its withheld team must not ride along.
        change = next(c for c in lr.pending_changes(store, True) if c["resource_id"] == ABSTAINED)
        assert set(change["tags"]) == {"env"}
        assert f"{ABSTAINED},gcp,team" not in lr.export(store, "csv", include_auto=True)
        tfvars = json.loads(lr.export(store, "tfvars", include_auto=True))
        assert set(tfvars["project_labels"][ABSTAINED]) == {"env"}
        assert "team=" not in next(
            line
            for line in lr.export(store, "sh", include_auto=True).splitlines()
            if ABSTAINED in line
        )
        assert json.loads(lr.export(store, "plan", include_auto=True))["changes"] == []

    def test_unreviewed_small_inventory_exports_nothing_abstained(self, store, small):
        changed = {c["resource_id"] for c in lr.pending_changes(store, True, small)}
        assert changed == {"gcp-demo-003"}

    def test_explicit_override_is_exported_with_evidence(self, store):
        store.set(ABSTAINED, "team", "override", "data-platform")
        changes = lr.pending_changes(store)
        assert [c["resource_id"] for c in changes] == [ABSTAINED]
        assert changes[0]["tags"]["team"] == ("data-platform", "override")
        assert changes[0]["tags"]["cost_center"] == ("cc-2002", "override")
        plan = lr.approved_change_plan(store)
        assert plan["approvals"][0]["source"] == "abstain"
        assert plan["approvals"][0]["suggested"] is None
        metrics = lr.feedback_metrics(store.feedback())
        assert metrics["abstained"] == 1 and metrics["unresolved"] == 0

    def test_api(self, client):
        items = client.get("/resources", params={"label": "team", "evidence": "conflicting"})
        assert ABSTAINED in {r["resource_id"] for r in items.json()["items"]}
        team = client.get(f"/resources/{ABSTAINED}/labels").json()["labels"]["team"]
        assert team["action"] == "insufficient evidence"
        assert team["evidence"]["status"] == "conflicting"
        cov = client.get("/labels/coverage").json()["labels"]["team"]
        assert cov["insufficient_evidence"] == 5
        assert (
            client.put(
                f"/resources/{ABSTAINED}/labels/team/decision", json={"decision": "accept"}
            ).status_code
            == 422
        )

    def test_agent_tool_says_why(self):
        from tools import suggest_labels

        out = suggest_labels(ABSTAINED)
        assert "→ no suggestion (insufficient evidence: conflicting" in out

    def test_review_page_shows_insufficient_evidence(self, store, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("OBS_AGENT_DECISIONS", str(store.path))
        at = AppTest.from_file(str(REVIEW_PAGE), default_timeout=60).run()
        assert not at.exception
        assert {m.label: m.value for m in at.metric}["Insufficient evidence"] == "5"
        at.sidebar.checkbox[0].check().run()
        assert not at.exception
        assert at.subheader[0].value.startswith("5 resources")
