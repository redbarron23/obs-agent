"""Post-delivery reconciliation of applied change plans against a fresh inventory (offline)."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import label_review as lr
import plan_reconciliation as pr
import plan_validation as pv
from inventory_import import load_inventory_csv
from review_workspaces import load_review, save_review

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
AZ = b"azure,az-demo-001,mg-payments,alice|bob,alice,AKS|SQL,120.50,0.92,0.18,payments,prod,CC-1001"
GCP = b"gcp,gcp-demo-003,folders-data,fatima|gus,fatima,BigQuery|GCS,36.20,0.30,0.42,,,"
assert AZ in SAMPLE and GCP in SAMPLE
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
# Unrelated resources outside the plan, so every observed estate keeps enough labelled values to
# pass import validation whichever planned changes reached it.
PADDING = b"".join(
    b"azure,az-pad-00%d,mg-payments,alice,alice,AKS,10.00,0.90,0.20,payments,prod,CC-1001\n" % i
    for i in (1, 2)
)


def _tags(row: bytes, team: str, env: str, cost_center: str) -> bytes:
    return b",".join(row.split(b",")[:9] + [team.encode(), env.encode(), cost_center.encode()])


def observed(az: bytes = AZ, gcp: bytes | None = GCP, csv: bytes = SAMPLE):
    """The sample inventory with the two planned rows replaced (gcp=None drops gcp-demo-003)."""
    csv = csv.replace(AZ, az)
    csv = csv.replace(GCP + b"\n", b"") if gcp is None else csv.replace(GCP, gcp)
    return load_inventory_csv(csv + PADDING)


APPLIED_AZ = _tags(AZ, "payments", "dev", "CC-1001")
# GCP labels are lowercase; an inventory showing CC-2002 still matches the exported cc-2002.
APPLIED_GCP = _tags(GCP, "data-platform", "prod", "CC-2002")


def _store(tmp_path, inventory):
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
    store.set("gcp-demo-003", "env", "override", "prod", inventory, reviewer="alice")
    store.set("az-demo-001", "env", "override", "dev", inventory, reviewer="bo")
    return store


@pytest.fixture
def reviewed(tmp_path):
    """Reviewed snapshot and a mixed plan: az-demo-001 env, gcp-demo-003 team/env/cost_center."""
    source = load_inventory_csv(SAMPLE)
    plan = lr.approved_change_plan(_store(tmp_path, source), source)
    assert [(c["resource_id"], list(c["tags"])) for c in plan["changes"]] == [
        ("az-demo-001", ["env"]),
        ("gcp-demo-003", ["team", "env", "cost_center"]),
    ]
    assert pv.plan_problems(plan) == []
    return source, plan


def _outcomes(report) -> dict:
    return {
        (r["resource_id"], label): t["outcome"]
        for r in report["resources"]
        for label, t in r["tags"].items()
    }


def _status(report) -> dict:
    return {r["resource_id"]: r["status"] for r in report["resources"]}


class TestOutcomes:
    def test_complete_application(self, reviewed):
        source, plan = reviewed
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, APPLIED_GCP), source=source, now=NOW)
        assert report["status"] == "complete"
        assert set(_outcomes(report).values()) == {pr.APPLIED}
        assert _status(report) == {"az-demo-001": "applied", "gcp-demo-003": "applied"}
        assert report["summary"]["tag_outcomes"] == {
            "applied": 4,
            "not_applied": 0,
            "conflicting": 0,
            "unavailable": 0,
        }
        gcp = report["resources"][1]["tags"]["cost_center"]
        assert gcp == gcp | {"new": "cc-2002", "observed": "CC-2002", "outcome": "applied"}
        assert "Post-delivery reconciliation: COMPLETE" in pr.summary_text(report)

    def test_not_applied_keeps_before_value_without_claiming_a_cause(self, reviewed):
        source, plan = reviewed
        report = pr.reconcile_plan(plan, load_inventory_csv(SAMPLE), source=source, now=NOW)
        assert report["status"] == "incomplete"
        assert set(_outcomes(report).values()) == {pr.NOT_APPLIED}
        assert _status(report) == {"az-demo-001": "not_applied", "gcp-demo-003": "not_applied"}
        env = report["resources"][0]["tags"]["env"]
        assert env["observed"] == "prod" and "does not show whether" in env["reason"]
        assert report["observed_inventory"]["same_as_reviewed"]

    def test_partial_application(self, reviewed):
        source, plan = reviewed
        gcp = _tags(GCP, "data-platform", "", "")
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, gcp), source=source, now=NOW)
        assert report["status"] == "incomplete"
        assert _outcomes(report) == {
            ("az-demo-001", "env"): "applied",
            ("gcp-demo-003", "team"): "applied",
            ("gcp-demo-003", "env"): "not_applied",
            ("gcp-demo-003", "cost_center"): "not_applied",
        }
        assert _status(report) == {"az-demo-001": "applied", "gcp-demo-003": "partially_applied"}
        assert report["summary"]["resource_status"]["partially_applied"] == 1

    def test_conflicting_value_does_not_say_who_set_it(self, reviewed):
        source, plan = reviewed
        gcp = _tags(GCP, "payments", "prod", "CC-2002")
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, gcp), source=source, now=NOW)
        team = report["resources"][1]["tags"]["team"]
        assert team["outcome"] == "conflicting" and team["observed"] == "payments"
        assert "does not show who set this value" in team["reason"]
        assert _status(report)["gcp-demo-003"] == "conflicting"
        text = pr.summary_text(report)
        assert "gcp-demo-003 team [conflicting]" in text and "did not retry delivery" in text

    def test_azure_values_are_compared_exactly(self, reviewed):
        source, plan = reviewed
        az = _tags(AZ, "payments", "DEV", "CC-1001")
        report = pr.reconcile_plan(plan, observed(az, APPLIED_GCP), source=source, now=NOW)
        assert _outcomes(report)[("az-demo-001", "env")] == "conflicting"

    def test_missing_resource_is_unavailable(self, reviewed):
        source, plan = reviewed
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, None), source=source, now=NOW)
        assert _status(report) == {"az-demo-001": "applied", "gcp-demo-003": "unavailable"}
        gcp = report["resources"][1]
        assert {t["outcome"] for t in gcp["tags"].values()} == {"unavailable"}
        assert all(t["observed"] is None for t in gcp["tags"].values())
        assert "not in the observed inventory" in gcp["tags"]["team"]["reason"]
        assert gcp["unplanned_changes"] == []

    def test_resource_under_another_cloud_is_unavailable(self, reviewed):
        source, plan = reviewed
        az = APPLIED_AZ.replace(b"azure,", b"gcp,", 1)
        report = pr.reconcile_plan(plan, observed(az, APPLIED_GCP), source=source, now=NOW)
        resource = report["resources"][0]
        assert resource["status"] == "unavailable" and resource["observed_cloud"] == "gcp"
        assert "under gcp, not azure" in resource["tags"]["env"]["reason"]


class TestUnrelatedTags:
    def test_unchanged_unrelated_tags_and_other_resources_are_not_reported(self, reviewed):
        source, plan = reviewed
        csv = SAMPLE.replace(b"gcp-demo-001,folders-data", b"gcp-demo-001,folders-x")
        csv = csv.replace(b"38.10", b"900.00").replace(b"data-platform,dev,CC-2002", b"x,y,z", 1)
        report = pr.reconcile_plan(
            plan, observed(APPLIED_AZ, APPLIED_GCP, csv), source=source, now=NOW
        )
        assert report["status"] == "complete"
        assert [r["resource_id"] for r in report["resources"]] == ["az-demo-001", "gcp-demo-003"]
        assert all(r["unplanned_changes"] == [] for r in report["resources"])
        assert report["summary"]["unplanned_changes"] == 0
        # The plan touched only env on az-demo-001: its team and cost_center are not in the report.
        assert list(report["resources"][0]["tags"]) == ["env"]

    def test_changed_unplanned_tag_on_a_planned_resource_is_flagged_not_explained(self, reviewed):
        source, plan = reviewed
        az = _tags(AZ, "payments", "dev", "CC-9999")
        report = pr.reconcile_plan(plan, observed(az, APPLIED_GCP), source=source, now=NOW)
        assert report["status"] == "complete"  # every approved change is present
        (change,) = report["resources"][0]["unplanned_changes"]
        assert change | {"reason": None} == {
            "label": "cost_center",
            "reviewed": "CC-1001",
            "observed": "CC-9999",
            "reason": None,
        }
        assert "does not show whether the delivery or someone else" in change["reason"]
        assert "[unplanned_change]" in pr.summary_text(report)

    def test_without_source_unplanned_changes_are_not_checked_and_said_so(self, reviewed):
        _, plan = reviewed
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, APPLIED_GCP), now=NOW)
        assert report["status"] == "complete" and report["source_snapshot"] is None
        assert report["summary"]["unplanned_changes"] is None
        assert "unplanned_changes" not in report["resources"][0]
        assert "were not checked" in pr.summary_text(report)


def _drop_approvals(p):
    p["approvals"] = []


def _blank_reviewer(p):
    p["approvals"][0]["reviewer"] = ""


def _other_value(p):
    p["approvals"][0]["value"] = "payments"


MALFORMED = {
    "unsupported_schema": lambda p: p.update(schema_version=1),
    "not_reviewed_only": lambda p: p.update(reviewed_only=False),
    "malformed_change": lambda p: p["changes"][0].update(cloud="aws"),
    "missing_approval": _drop_approvals,
    "missing_provenance": _blank_reviewer,
    "approval_mismatch": _other_value,
}


class TestInvalid:
    @pytest.mark.parametrize("code", MALFORMED)
    def test_invalid_plans_are_rejected_before_reconciliation(self, reviewed, code):
        source, plan = reviewed
        plan = copy.deepcopy(plan)
        MALFORMED[code](plan)
        report = pr.reconcile_plan(plan, observed(APPLIED_AZ, APPLIED_GCP), now=NOW)
        assert report["status"] == "invalid" and report["resources"] == []
        assert code in {e["code"] for e in report["errors"]}
        assert report["plan"] == {"sha256": pr.plan_digest(plan)}
        assert "invalid plan" in pr.summary_text(report)

    @pytest.mark.parametrize("plan", [None, [], "plan", {"schema_version": 2}])
    def test_non_plans_are_rejected(self, plan):
        report = pr.reconcile_plan(plan, load_inventory_csv(SAMPLE), now=NOW)
        assert report["status"] == "invalid" and report["errors"]

    def test_wrong_source_snapshot_is_rejected(self, reviewed):
        _, plan = reviewed
        other = load_inventory_csv(SAMPLE.replace(b"120.50", b"121.50"))
        report = pr.reconcile_plan(plan, load_inventory_csv(SAMPLE), source=other, now=NOW)
        assert report["status"] == "invalid"
        assert [e["code"] for e in report["errors"]] == ["source_snapshot_mismatch"]

    def test_plan_edited_after_export_is_rejected(self, reviewed):
        source, plan = reviewed
        plan = copy.deepcopy(plan)
        plan["changes"][0]["tags"]["env"]["previous"] = "dev"
        report = pr.reconcile_plan(plan, load_inventory_csv(SAMPLE), source=source, now=NOW)
        assert report["status"] == "invalid"
        assert [e["code"] for e in report["errors"]] == ["plan_inconsistent_with_source"]


class TestAuditable:
    def test_report_links_plan_and_observed_snapshot(self, reviewed):
        source, plan = reviewed
        fresh = observed(APPLIED_AZ, APPLIED_GCP)
        report = pr.reconcile_plan(plan, fresh, source=source, now=NOW)
        assert report["check"] == "post_delivery_reconciliation"
        assert report["reconciled_at"] == "2026-11-01T12:00:00+00:00"
        assert report["plan"] | {"sha256": None} == {
            "sha256": None,
            "inventory_id": plan["inventory_id"],
            "model_version": plan["model_version"],
            "generated_at": plan["generated_at"],
            "resources": 2,
            "tags": 4,
            "approvals": 3,
            "reviewers": ["alice", "bo"],
        }
        assert report["plan"]["sha256"] == pr.plan_digest(plan)
        assert report["observed_inventory"] == {
            "inventory_id": pr.review_context(fresh)["inventory_id"],
            "resources": 7,
            "same_as_reviewed": False,
        }
        assert report["source_snapshot"] == {
            "inventory_id": plan["inventory_id"],
            "matches_plan": True,
        }
        assert "does not show why" in report["limitations"]

    def test_report_is_deterministic(self, reviewed):
        source, plan = reviewed
        reports = [
            json.dumps(
                pr.reconcile_plan(plan, observed(APPLIED_AZ, None), source=source, now=NOW),
                sort_keys=True,
            )
            for _ in range(2)
        ]
        assert reports[0] == reports[1]
        reordered = json.loads(json.dumps(plan), object_hook=lambda d: dict(reversed(d.items())))
        assert pr.plan_digest(reordered) == pr.plan_digest(plan)
        assert pr.plan_digest(plan | {"generated_at": "x"}) != pr.plan_digest(plan)

    def test_reconciliation_never_mutates_inputs(self, reviewed, tmp_path):
        source, plan = reviewed
        fresh = observed(APPLIED_AZ, _tags(GCP, "payments", "", ""))
        before = (copy.deepcopy(plan), source.copy(deep=True), fresh.copy(deep=True))
        digest = hashlib.sha256((tmp_path / "review.db").read_bytes()).hexdigest()
        pr.reconcile_plan(plan, fresh, source=source, now=NOW)
        assert plan == before[0]
        assert source.equals(before[1]) and fresh.equals(before[2])
        assert hashlib.sha256((tmp_path / "review.db").read_bytes()).hexdigest() == digest
        assert list(tmp_path.iterdir()) == [tmp_path / "review.db"]

    def test_pre_delivery_validation_is_unchanged(self, reviewed):
        source, plan = reviewed
        assert pv.validate_plan(plan, load_inventory_csv(SAMPLE), source=source)["status"] == "pass"
        # After delivery the same inventory means "already applied" to the pre-delivery check.
        report = pv.validate_plan(plan, observed(APPLIED_AZ, APPLIED_GCP), source=source)
        assert report["status"] == "fail"
        codes = {f["code"] for r in report["resources"] for f in r["findings"]}
        assert codes == {"already_applied"}


class TestCli:
    @pytest.fixture
    def saved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
        review_id = save_review(SAMPLE, "october")
        inventory, store = load_review(review_id)
        store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
        store.set("gcp-demo-003", "env", "override", "prod", inventory, reviewer="alice")
        store.set("az-demo-001", "env", "override", "dev", inventory, reviewer="bo")
        plan = tmp_path / "plan.json"
        plan.write_text(lr.export(store, "plan", inventory=inventory))
        return review_id, plan, store.path

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_exit_codes_with_saved_review_and_nothing_mutated(self, saved, tmp_path, capsys):
        review_id, plan, db = saved
        fresh = tmp_path / "observed.csv"
        fresh.write_bytes(SAMPLE.replace(AZ, APPLIED_AZ).replace(GCP, APPLIED_GCP))
        digests = {p: self._digest(p) for p in (db, plan, fresh)}
        out = tmp_path / "report.json"
        args = [str(plan), str(fresh), "--review-id", review_id, "--output", str(out)]
        assert pr.main(args) == 0
        report = json.loads(out.read_text())
        assert report["status"] == "complete" and report["source_snapshot"]["matches_plan"]
        assert report["inputs"] == {
            "plan_file": {"name": "plan.json", "sha256": "sha256:" + digests[plan]},
            "observed_file": {"name": "observed.csv", "sha256": "sha256:" + digests[fresh]},
            "review_id": review_id,
        }
        assert "COMPLETE" in capsys.readouterr().err
        assert {p: self._digest(p) for p in (db, plan, fresh)} == digests

        fresh.write_bytes(SAMPLE.replace(GCP + b"\n", b""))
        digests[fresh] = self._digest(fresh)
        assert pr.main(args) == 1
        assert "[unavailable]" in capsys.readouterr().err
        assert {p: self._digest(p) for p in (db, plan, fresh)} == digests

    def test_source_csv_stdout_and_invalid_input(self, saved, tmp_path, capsys):
        _, plan, _ = saved
        csv = tmp_path / "inventory.csv"
        csv.write_bytes(SAMPLE)
        assert pr.main([str(plan), str(csv), "--source", str(csv)]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["summary"]["tag_outcomes"]["not_applied"] == 4
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema_version": 1}')
        assert pr.main([str(bad), str(csv)]) == 2
        assert pr.main([str(tmp_path / "missing.json"), str(csv)]) == 2
        assert "INVALID INPUT" in capsys.readouterr().err
