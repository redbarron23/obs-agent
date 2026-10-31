"""Pre-delivery validation of reviewed change plans against a fresh inventory (offline)."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

import label_review as lr
import plan_validation as pv
from inventory_import import load_inventory_csv
from label_config import default_config
from review_workspaces import load_review, save_review

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
TARGET = b"gcp,gcp-demo-003,folders-data,fatima|gus,fatima,BigQuery|GCS,36.20,0.30,0.42,,,"
assert TARGET in SAMPLE


def _fresh(replacement: bytes, source: bytes = SAMPLE, config=None):
    return load_inventory_csv(source.replace(TARGET, replacement), config)


@pytest.fixture
def reviewed(tmp_path):
    """The reviewed snapshot and its plan: gcp-demo-003 team accepted (plus its cost center)."""
    inventory = load_inventory_csv(SAMPLE)
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
    plan = lr.approved_change_plan(store, inventory)
    assert plan["changes"][0]["tags"] == {
        "team": {"previous": None, "new": "data-platform", "how": "accept"},
        "cost_center": {"previous": None, "new": "cc-2002", "how": "accept"},
    }
    return inventory, plan


def _codes(report) -> list[str]:
    return [f["code"] for r in report["resources"] for f in r["findings"]] + [
        c["code"] for c in report["context"]
    ]


class TestValid:
    def test_unchanged_context_passes_with_machine_readable_report(self, reviewed):
        source, plan = reviewed
        report = pv.validate_plan(plan, load_inventory_csv(SAMPLE), source=source)
        assert report["status"] == "pass" and not report["renewed_review_required"]
        assert report["fresh_inventory"]["same_as_reviewed"]
        assert report["source_snapshot"]["matches_plan"]
        assert report["errors"] == report["context"] == report["warnings"] == []
        assert report["summary"] == {"resources": 1, "ok": 1, "needs_review": 0, "findings": {}}
        assert report["resources"][0]["tags"]["team"]["state"] == pv.BEFORE
        assert report["plan"]["reviewers"] == ["alice"]
        assert json.loads(json.dumps(report)) == report

    def test_unrelated_changes_elsewhere_still_pass(self, reviewed):
        source, plan = reviewed
        fresh = load_inventory_csv(SAMPLE.replace(b"120.50", b"900.00"))
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "pass"
        assert not report["fresh_inventory"]["same_as_reviewed"]

    def test_without_source_only_before_values_are_checked_and_said_so(self, reviewed):
        _, plan = reviewed
        fresh = _fresh(TARGET.replace(b"folders-data", b"mg-payments"))
        report = pv.validate_plan(plan, fresh)
        assert report["status"] == "pass"
        assert report["source_snapshot"] is None
        assert "no source snapshot" in report["warnings"][0]

    def test_override_of_withheld_suggestion_validates(self, tmp_path):
        source_csv = (
            SAMPLE + b"gcp,proj-482913,folders-data,zoe,zoe,BigQuery|GCS,40.00,0.30,0.40,,,\n"
        )
        inventory = load_inventory_csv(source_csv)
        store = lr.DecisionStore(tmp_path / "review.db")
        store.set("proj-482913", "team", "override", "data-platform", inventory, reviewer="bo")
        plan = lr.approved_change_plan(store, inventory)
        assert plan["approvals"][0]["source"] == "abstain"
        assert plan["approvals"][0]["suggested"] is None
        report = pv.validate_plan(plan, load_inventory_csv(source_csv), source=inventory)
        assert report["status"] == "pass", report


class TestStaleAndApplied:
    def test_stale_previous_value_fails_with_reason(self, reviewed):
        source, plan = reviewed
        fresh = _fresh(TARGET.replace(b",,,", b",payments,,"))
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "fail" and report["renewed_review_required"]
        (resource,) = report["resources"]
        assert resource["resource_id"] == "gcp-demo-003" and resource["status"] == "needs_review"
        (finding,) = resource["findings"]
        assert finding["code"] == "previous_value_changed" and finding["label"] == "team"
        assert finding["previous"] is None and finding["current"] == "payments"
        assert "Review gcp-demo-003 again" in finding["reason"]
        assert resource["tags"]["cost_center"]["state"] == pv.BEFORE

    def test_already_applied_values_fail_using_export_normalisation(self, reviewed):
        source, plan = reviewed
        fresh = _fresh(TARGET.replace(b",,,", b",data-platform,,CC-2002"))
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "fail"
        assert _codes(report) == ["already_applied", "already_applied"]
        assert report["resources"][0]["tags"]["cost_center"] == {
            "previous": None,
            "new": "cc-2002",
            "observed": "CC-2002",
            "state": pv.INTENDED,
        }
        assert "outside this plan" in report["resources"][0]["findings"][0]["reason"]

    def test_azure_values_are_compared_exactly(self):
        plan = {
            "changes": [
                {
                    "resource_id": "gcp-demo-003",
                    "cloud": "azure",
                    "tags": {"team": {"previous": None, "new": "Data-Platform", "how": "accept"}},
                }
            ]
        }
        fresh = _fresh(TARGET.replace(b",,,", b",data-platform,,"))
        assert pv.observe_tags(plan, fresh)[0]["state"] == pv.OTHER


class TestMissingAndConflicting:
    def test_removed_resource_fails(self, reviewed):
        source, plan = reviewed
        fresh = load_inventory_csv(SAMPLE.replace(TARGET + b"\n", b""))
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "fail"
        assert _codes(report) == ["resource_missing"]
        assert "not in the fresh inventory" in report["resources"][0]["findings"][0]["reason"]
        assert report["resources"][0]["tags"]["team"]["state"] == pv.MISSING

    def test_changed_ownership_evidence_conflicts_with_approval(self, reviewed):
        source, plan = reviewed
        fresh = _fresh(TARGET.replace(b"folders-data,fatima|gus", b"mg-payments,alice|bob"))
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "fail"
        findings = report["resources"][0]["findings"]
        assert {(f["code"], f["field"]) for f in findings} == {
            ("evidence_changed", "parent"),
            ("evidence_changed", "owners"),
        }
        assert all("approved the earlier evidence" in f["reason"] for f in findings)

    def test_changed_cloud_fails(self, reviewed):
        source, plan = reviewed
        fresh = _fresh(TARGET.replace(b"gcp,gcp-demo-003", b"azure,gcp-demo-003"))
        report = pv.validate_plan(plan, fresh, source=source)
        assert "cloud_changed" in _codes(report)


class TestContextMismatch:
    def test_different_vocabulary_changes_the_inference_version(self, reviewed):
        source, plan = reviewed
        config = default_config()
        config["team"]["aliases"]["pay"] = "payments"
        fresh = load_inventory_csv(SAMPLE, config)
        report = pv.validate_plan(plan, fresh, source=source)
        assert report["status"] == "fail"
        assert _codes(report) == ["inference_version_changed"]
        assert "vocabulary" in report["context"][0]["message"]

    def test_changed_inference_version_fails(self, reviewed):
        source, plan = reviewed
        plan = copy.deepcopy(plan)
        plan["model_version"] = "sha256:" + "0" * 64
        for approval in plan["approvals"]:
            approval["model_version"] = plan["model_version"]
        for kwargs in ({"source": source}, {}):
            report = pv.validate_plan(plan, load_inventory_csv(SAMPLE), **kwargs)
            assert report["status"] == "fail"
            assert _codes(report) == ["inference_version_changed"]

    def test_wrong_source_snapshot_fails(self, reviewed):
        _, plan = reviewed
        other = load_inventory_csv(SAMPLE.replace(b"120.50", b"121.50"))
        report = pv.validate_plan(plan, load_inventory_csv(SAMPLE), source=other)
        assert report["status"] == "fail"
        assert _codes(report) == ["source_snapshot_mismatch"]

    def test_plan_edited_after_export_fails(self, reviewed):
        source, plan = reviewed
        plan = copy.deepcopy(plan)
        plan["changes"][0]["tags"]["team"]["previous"] = "payments"
        report = pv.validate_plan(plan, load_inventory_csv(SAMPLE), source=source)
        assert "plan_inconsistent_with_source" in _codes(report)


def _drop_approvals(p):
    p["approvals"] = []


def _blank_reviewer(p):
    p["approvals"][0]["reviewer"] = " "


def _other_snapshot(p):
    p["approvals"][0]["inventory_id"] = "0" * 64


def _auto_apply(p):
    p["changes"][0]["tags"]["team"]["how"] = "auto-apply"


def _other_value(p):
    p["approvals"][0]["value"] = "payments"


MALFORMED = {
    "unsupported_schema": lambda p: p.update(schema_version=1),
    "not_reviewed_only": lambda p: p.update(reviewed_only=False),
    "missing_field": lambda p: p.pop("inventory_id"),
    "malformed_plan": lambda p: p.update(changes={}),
    "empty_plan": lambda p: p.update(changes=[]),
    "malformed_change": lambda p: p["changes"][0].update(cloud="aws"),
    "missing_approval": _drop_approvals,
    "missing_provenance": _blank_reviewer,
    "approval_context_mismatch": _other_snapshot,
    "unreviewed_change": _auto_apply,
    "approval_mismatch": _other_value,
}


class TestMalformed:
    @pytest.mark.parametrize("code", MALFORMED)
    def test_rejected_clearly(self, reviewed, code):
        source, plan = reviewed
        plan = copy.deepcopy(plan)
        MALFORMED[code](plan)
        report = pv.validate_plan(plan, load_inventory_csv(SAMPLE), source=source)
        assert report["status"] == "invalid" and report["renewed_review_required"]
        assert code in {e["code"] for e in report["errors"]}
        assert report["resources"] == []
        assert "invalid plan" in pv.summary_text(report)

    @pytest.mark.parametrize("plan", [None, [], "plan", {"schema_version": 2}])
    def test_non_plans_are_rejected(self, plan):
        report = pv.validate_plan(plan, load_inventory_csv(SAMPLE))
        assert report["status"] == "invalid" and report["errors"]

    def test_example_plan_fixture_is_well_formed(self):
        plan = json.loads((ROOT / "fixtures/reviewed_change_plan_example.json").read_text())
        assert pv.plan_problems(plan) == []


def test_validation_never_mutates_inputs(reviewed, tmp_path):
    source, plan = reviewed
    fresh = _fresh(TARGET.replace(b",,,", b",payments,,"))
    before = (copy.deepcopy(plan), source.copy(deep=True), fresh.copy(deep=True))
    pv.validate_plan(plan, fresh, source=source)
    assert plan == before[0]
    assert source.equals(before[1]) and fresh.equals(before[2])
    assert list(tmp_path.iterdir()) == [tmp_path / "review.db"]


class TestCli:
    @pytest.fixture
    def saved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        review_id = save_review(SAMPLE, "october")
        inventory, store = load_review(review_id)
        store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
        plan = tmp_path / "plan.json"
        plan.write_text(lr.export(store, "plan", inventory=inventory))
        return review_id, plan, store.path

    def _digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_pass_and_fail_exit_codes_with_review_id(self, saved, tmp_path, capsys):
        review_id, plan, db = saved
        digest = self._digest(db)
        fresh = tmp_path / "fresh.csv"
        fresh.write_bytes(SAMPLE)
        out = tmp_path / "report.json"
        args = [str(plan), str(fresh), "--review-id", review_id, "--output", str(out)]
        assert pv.main(args) == 0
        assert json.loads(out.read_text())["status"] == "pass"
        assert "PASS" in capsys.readouterr().err
        fresh.write_bytes(SAMPLE.replace(TARGET + b"\n", b""))
        assert pv.main(args) == 1
        assert "resource_missing" in capsys.readouterr().err
        assert self._digest(db) == digest

    def test_source_csv_and_invalid_input(self, saved, tmp_path, capsys):
        _, plan, _ = saved
        csv = tmp_path / "inventory.csv"
        csv.write_bytes(SAMPLE)
        assert pv.main([str(plan), str(csv), "--source", str(csv)]) == 0
        assert json.loads(capsys.readouterr().out)["source_snapshot"]["matches_plan"]
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema_version": 1}')
        assert pv.main([str(bad), str(csv)]) == 2
        assert pv.main([str(tmp_path / "missing.json"), str(csv)]) == 2
        assert "INVALID INPUT" in capsys.readouterr().err
