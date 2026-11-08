"""Smoke test for the packaged agent skill (skills/cloud-label-review).

Runs triage.py the way SKILL.md tells an agent to, so a renamed import, parameter or output
field breaks CI instead of the skill.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SKILL = ROOT / "skills" / "cloud-label-review"
TRIAGE = SKILL / "scripts" / "triage.py"
SAMPLE = ROOT / "fixtures" / "sanitized_inventory_sample.csv"
VOCAB = ROOT / "fixtures" / "label_config_example.json"
# Two untagged rows whose team is withheld for insufficient evidence (as in test_saved_review_api).
WITHHELD = (
    b"azure,sub-0a1b2c3d,mg-payments,,,AKS|SQL,110.00,0.90,0.20,,,\n"
    b"gcp,proj-482913,folders-data,zoe,zoe,BigQuery|GCS,40.00,0.30,0.40,,,\n"
)


def _triage(*args, cwd=ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRIAGE), *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )


def test_frontmatter():
    text = (SKILL / "SKILL.md").read_text()
    front = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert front and front.group(1) == "cloud-label-review"
    assert len(front.group(2)) <= 1024


@pytest.mark.parametrize("vocab", [[], ["--vocab", str(VOCAB)]], ids=["default", "vocab"])
def test_triage_output_has_the_fields_the_skill_describes(tmp_path, vocab):
    out = tmp_path / "triage.json"
    run = _triage("--csv", str(SAMPLE), *vocab, "--out", str(out))
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())
    team = result["coverage"]["labels"]["team"]
    assert {"untagged", "suspect", "auto_apply_ready", "insufficient_evidence"} <= set(team)
    (item,) = result["queue"]
    assert item["needs_attention"] and item["cost_center"]["implied"]
    label = item["labels"]["team"]
    assert {
        "status",
        "action",
        "suggested",
        "confidence",
        "source",
        "reason",
        "alternatives",
    } <= set(label)
    assert {"status", "supporting", "conflicting", "withheld", "reason"} <= set(label["evidence"])
    assert label["action"] == "auto-apply" and label["evidence"]["status"] == "supported"


def test_withheld_labels_have_no_suggestion(tmp_path):
    csv = tmp_path / "inventory.csv"
    csv.write_bytes(SAMPLE.read_bytes() + WITHHELD)
    run = _triage("--csv", str(csv))
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    withheld = [
        q["labels"]["team"]
        for q in result["queue"]
        if q["labels"]["team"]["action"] == "insufficient evidence"
    ]
    assert len(withheld) == result["coverage"]["labels"]["team"]["insufficient_evidence"] == 2
    for label in withheld:
        assert label["suggested"] is None and label["evidence"]["withheld"]["value"]
        assert label["evidence"]["status"] in ("missing_signals", "insufficient_support")


def test_runs_from_outside_the_checkout_and_writes_nothing_else(tmp_path):
    """Installed elsewhere (e.g. ~/.claude/skills/), it falls back to the bundled checkout."""
    run = _triage("--csv", str(SAMPLE), "--min-confidence", "0.95", cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["queue"] == []  # 0.929 is below 0.95
    assert list(tmp_path.iterdir()) == []


def test_invalid_input_exits_2(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("cloud,resource_id\nazure,sub-1\n")
    run = _triage("--csv", str(bad))
    assert run.returncode == 2 and run.stderr.startswith("error:")
