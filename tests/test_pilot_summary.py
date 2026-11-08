"""The real-world pilot kit (docs/pilot-kit.md, issue #49), on fictional data only."""

import csv
import json
from pathlib import Path

import pytest

import review_cli
from review_workspaces import load_review
from scripts import pilot_summary as ps

ROOT = Path(__file__).parent.parent
PILOT = ROOT / "docs" / "pilot"
SAMPLE = ROOT / "fixtures" / "sanitized_inventory_sample.csv"
VOCAB = ROOT / "fixtures" / "label_config_example.json"


def _write(path: Path, header: str, rows: list[list]) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header.split(","))
        writer.writerows(rows)
    return path


def _header(name: str) -> str:
    return (PILOT / name).read_text().strip()


def _feedback(*rows) -> list[dict]:
    out = []
    for rid, label, decision, value, suggested, previous, reviewer in rows:
        out.append(
            {
                "resource_id": rid,
                "label": label,
                "decision": decision,
                "value": value,
                "suggested": suggested,
                "previous": previous,
                "reviewer": reviewer,
            }
        )
    return out


FEEDBACK = _feedback(
    ("r-1", "team", "accept", "payments", "payments", None, "alice"),
    ("r-1", "env", "override", "prod", "dev", None, "alice"),
    ("r-2", "team", "override", "payments", None, None, "alice"),  # withheld, overridden
    ("r-2", "env", "reject", None, "dev", None, "bob"),  # untagged reject: unresolved
    ("r-3", "team", "accept", "data-platform", "data-platform", None, "bob"),
    ("r-3", "team", "revoke", "data-platform", "data-platform", None, "bob"),  # undone
)
REVIEW_LOG = [
    ["r-1", "team", "alice", 40, "accept", "no", ""],
    ["r-1", "env", "alice", 20, "override", "no", "unclear_reason"],
    ["r-2", "team", "alice", 200, "override", "yes", "missing_owner"],
    ["r-2", "env", "bob", 30, "reject", "no", ""],
    ["r-3", "team", "bob", 60, "unresolved", "yes", "missing_owner"],
]


@pytest.fixture
def logs(tmp_path):
    return {
        "review": _write(tmp_path / "review.csv", _header("review-log.csv"), REVIEW_LOG),
        "baseline": _write(
            tmp_path / "baseline.csv",
            _header("manual-baseline-log.csv"),
            [["r-1", "team", "carol", 120, "payments"], ["r-1", "env", "carol", 90, "prod"]],
        ),
        "re_review": _write(
            tmp_path / "re.csv",
            _header("re-review-log.csv"),
            [
                ["r-1", "team", "carol", "payments", ""],
                ["r-2", "team", "carol", "data-platform", "first"],
            ],
        ),
    }


def _summary(logs, **kw):
    return ps.summarise(
        FEEDBACK,
        ps.load_review_log(logs["review"]),
        ps.load_baseline_log(logs["baseline"]),
        "2 of 3, random.sample, seed 1",
        ps.load_re_review_log(logs["re_review"]),
        **kw,
    )


class TestSummary:
    def test_reports_effort_usefulness_stability_and_friction(self, logs):
        s = _summary(logs)
        assert s["scope"] == {"resources": 3, "resource_labels": 5, "reviewers": 2}
        assert s["time_with_obs_agent"]["median_seconds_per_resource"] == 60.0
        team, env = s["usefulness"]["team"], s["usefulness"]["env"]
        assert (team["accept"], team["override"], team["unresolved"]) == (1, 1, 1)
        assert team["decided_without_a_suggestion"] == 1 and team["needed_missing_evidence"] == 2
        assert env["reject"] == 1 and env["unresolved_for_scoring"] == 1
        assert s["friction"] == {"unclear_reason": 1, "missing_owner": 2}
        base = s["manual_baseline"]
        assert base["resources_compared"] == 1 and base["manual"]["total_minutes"] == 3.5
        assert base["median_time_ratio"] == round(60 / 210, 2)
        stability = s["decision_stability"]
        assert (stability["re_reviewed"], stability["agreed"]) == (2, 1)
        assert stability["disagreements"] == [
            {
                "resource_id": "r-2",
                "label": "team",
                "first": "payments",
                "second": "data-platform",
                "resolution": "first",
            }
        ]
        assert "not an unbiased production accuracy estimate" in s["agreement_is_not_accuracy"]

    def test_optional_parts_can_be_left_out(self, logs):
        s = ps.summarise(FEEDBACK, ps.load_review_log(logs["review"]))
        assert "manual_baseline" not in s and "decision_stability" not in s

    def test_log_must_match_the_saved_review(self, logs, tmp_path):
        rows = [r[:] for r in REVIEW_LOG]
        rows[0][4] = "override"
        bad = _write(tmp_path / "bad.csv", _header("review-log.csv"), rows)
        with pytest.raises(ps.PilotInputError, match="r-1/team was override"):
            ps.summarise(FEEDBACK, ps.load_review_log(bad))
        missing = _write(tmp_path / "short.csv", _header("review-log.csv"), REVIEW_LOG[1:])
        with pytest.raises(ps.PilotInputError, match="not in the review log"):
            ps.summarise(FEEDBACK, ps.load_review_log(missing))

    def test_re_review_must_be_independent(self, logs, tmp_path):
        same = _write(
            tmp_path / "same.csv", _header("re-review-log.csv"), [["r-1", "team", "alice", "x", ""]]
        )
        with pytest.raises(ps.PilotInputError, match="different reviewer"):
            ps.summarise(
                FEEDBACK,
                ps.load_review_log(logs["review"]),
                re_review_log=(ps.load_re_review_log(same)),
            )

    def test_baseline_needs_a_sampling_description(self, logs):
        with pytest.raises(ps.PilotInputError, match="sampled"):
            ps.summarise(
                FEEDBACK,
                ps.load_review_log(logs["review"]),
                ps.load_baseline_log(logs["baseline"]),
                "  ",
            )

    @pytest.mark.parametrize(
        ("column", "value", "match"),
        [
            (3, "0", "seconds"),
            (4, "approve", "outcome"),
            (5, "maybe", "missing_evidence"),
            (6, "slow", "friction"),
            (1, "owner", "label"),
        ],
    )
    def test_invalid_rows(self, tmp_path, column, value, match):
        rows = [r[:] for r in REVIEW_LOG]
        rows[0][column] = value
        path = _write(tmp_path / "log.csv", _header("review-log.csv"), rows)
        with pytest.raises(ps.PilotInputError, match=match):
            ps.load_review_log(path)

    def test_no_free_text_columns(self, tmp_path):
        path = _write(
            tmp_path / "log.csv",
            _header("review-log.csv") + ",notes",
            [REVIEW_LOG[0] + ["owner is jane@example.com"]],
        )
        with pytest.raises(ps.PilotInputError, match="unexpected column"):
            ps.load_review_log(path)


def test_templates_match_the_script():
    expected = {
        "review-log.csv": "resource_id,label,reviewer,seconds,outcome,missing_evidence,friction",
        "manual-baseline-log.csv": "resource_id,label,reviewer,seconds,resolved_value",
        "re-review-log.csv": "resource_id,label,reviewer,resolved_value,resolution",
    }
    for name, header in expected.items():
        assert _header(name) == header
    from inventory_import import REQUIRED_COLUMNS

    assert set(_header("inventory-template.csv").split(",")) >= set(REQUIRED_COLUMNS)


def test_end_to_end_with_the_cli(tmp_path, monkeypatch, capsys):
    """The runbook's commands on the fictional sample: import, decide, export, summarise."""
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
    assert review_cli.main(["import", str(SAMPLE), "--vocabulary", str(VOCAB)]) == 0
    review_id = json.loads(capsys.readouterr().out)["id"]
    inventory, store = load_review(review_id)
    store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
    store.set("gcp-demo-003", "env", "override", "prod", inventory, reviewer="alice")
    assert review_cli.main(["feedback", review_id]) == 0
    feedback = tmp_path / "feedback.json"
    feedback.write_text(capsys.readouterr().out)
    log = _write(
        tmp_path / "review-log.csv",
        _header("review-log.csv"),
        [
            ["gcp-demo-003", "team", "alice", 45, "accept", "no", ""],
            ["gcp-demo-003", "env", "alice", 30, "override", "no", "unclear_reason"],
        ],
    )
    out = tmp_path / "summary.json"
    args = ["--feedback", str(feedback), "--review-log", str(log), "--out", str(out)]
    assert ps.main(args) == 0
    summary = json.loads(out.read_text())
    assert summary["usefulness"]["env"]["override"] == 1
    assert summary["time_with_obs_agent"]["median_seconds_per_resource"] == 75.0

    assert ps.main(["--feedback", str(log), "--review-log", str(log)]) == 2
    assert "error:" in capsys.readouterr().err


def test_workspace_is_git_ignored():
    assert "pilot-workspace/" in (ROOT / ".gitignore").read_text().splitlines()
