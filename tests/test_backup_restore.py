"""Backup/restore round trip for decisions and saved reviews (docs/backup-restore.md)."""

import json
import sqlite3
import stat
from pathlib import Path

import pandas as pd
import pytest

import label_review as lr
from review_context import review_context
from review_workspaces import load_review, save_review
from scripts import review_backup

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
CSV = (
    SAMPLE.replace(b"payments", b"platform-engineering")
    .replace(b"data-platform", b"customer-success")
    .replace(b",prod,", b",production,")
    .replace(b",dev,", b",development,")
)
CONFIG = {
    "team": {
        "values": ["platform-engineering", "customer-success"],
        "aliases": {"platform": "platform-engineering"},
    },
    "env": {
        "values": ["production", "development"],
        "aliases": {"prod": "production", "dev": "development"},
        "name_tokens": {"production": ["prod"], "development": ["dev"]},
    },
}


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A populated installation: the main decision database plus one saved review."""
    decisions = tmp_path / "live" / "label_decisions.db"
    review_dir = tmp_path / "live" / "reviews"
    monkeypatch.setenv("OBS_AGENT_DECISIONS", str(decisions))
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(review_dir))
    main = lr.DecisionStore()
    main.set("svc-ingest", "team", "accept", reviewer="alice")
    review_id = save_review(CSV, "private.csv", CONFIG)
    inventory, store = load_review(review_id)
    store.set("gcp-demo-003", "team", "accept", inventory=inventory, reviewer="alice")
    store.set("gcp-demo-003", "team", "override", "Platform", inventory, reviewer="bob")
    store.set("gcp-demo-003", "env", "accept", inventory=inventory, reviewer="bob")
    store.delete("gcp-demo-003", "env")  # undone: history stays, decision goes
    return decisions, review_dir, review_id


def _state(decisions: Path, review_dir: Path, review_id: str, monkeypatch) -> dict:
    monkeypatch.setenv("OBS_AGENT_DECISIONS", str(decisions))
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(review_dir))
    inventory, store = load_review(review_id)
    main = lr.DecisionStore()
    return {
        "inventory": inventory,
        "decisions": store.all(),
        "feedback": store.feedback(),
        "main_decisions": main.all(),
        "main_feedback": main.feedback(),
        "plans": review_backup.plans(decisions, review_dir),
    }


def test_round_trip_restores_review_vocabulary_decisions_and_plan(live, tmp_path, monkeypatch):
    decisions, review_dir, review_id = live
    before = _state(decisions, review_dir, review_id, monkeypatch)
    backup = tmp_path / "backup"
    manifest = review_backup.backup(backup, decisions, review_dir)
    assert set(manifest["files"]) == {"label_decisions.db", f"reviews/{review_id}.db"}
    assert manifest["files"][f"reviews/{review_id}.db"]["rows"] == {
        "decisions": 1,
        "review_feedback": 3,
        "workspace": 1,
    }
    assert review_backup.verify(backup) == []

    isolated = tmp_path / "restored"
    restored = review_backup.restore(backup, isolated / "label_decisions.db", isolated / "reviews")
    for path in restored:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((isolated / "reviews").stat().st_mode) == 0o700
    after = _state(isolated / "label_decisions.db", isolated / "reviews", review_id, monkeypatch)

    assert after["inventory"].attrs == before["inventory"].attrs  # vocabulary
    assert after["inventory"].attrs["label_config"]["team"]["values"] == CONFIG["team"]["values"]
    pd.testing.assert_frame_equal(after["inventory"], before["inventory"])
    for key in ("decisions", "feedback", "main_decisions", "main_feedback"):
        assert after[key] == before[key]
    # Same code and dependencies: same effective changes and approval evidence.
    assert after["plans"] == before["plans"]
    plan = after["plans"][f"reviews/{review_id}.db"]
    assert plan["changes"][0]["tags"]["team"]["new"] == "platform-engineering"
    assert plan["approvals"][0]["reviewer"] == "bob"
    assert "generated_at" not in plan
    assert after["plans"]["label_decisions.db"]["approvals"][0]["reviewer"] == "alice"


def test_changed_inference_context_needs_renewed_approval(live, tmp_path, monkeypatch):
    decisions, review_dir, review_id = live
    backup = tmp_path / "backup"
    review_backup.backup(backup, decisions, review_dir)
    isolated = tmp_path / "restored"
    review_backup.restore(backup, isolated / "label_decisions.db", isolated / "reviews")
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(isolated / "reviews"))
    inventory, store = load_review(review_id)
    original = review_context(inventory)
    monkeypatch.setattr(lr, "review_context", lambda inv=None: original | {"model_version": "new"})
    plan = lr.approved_change_plan(store, inventory)
    assert plan["changes"] == [] and plan["approvals"] == []
    assert len(store.all()) == 1 and len(store.feedback()) == 3  # history kept, not transferred


def test_backup_of_running_store_has_only_committed_rows(live, tmp_path):
    decisions, review_dir, review_id = live
    writer = sqlite3.connect(review_dir / f"{review_id}.db", isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")  # an application write in progress
    writer.execute(
        "INSERT INTO decisions (resource_id, label, decision, decided_at) "
        "VALUES ('gcp-demo-001', 'env', 'reject', '2026-01-01')"
    )
    try:
        manifest = review_backup.backup(tmp_path / "backup", decisions, review_dir)
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    assert manifest["files"][f"reviews/{review_id}.db"]["rows"]["decisions"] == 1


def test_verify_and_restore_refuse_damaged_backups_and_existing_state(live, tmp_path):
    decisions, review_dir, review_id = live
    backup = tmp_path / "backup"
    review_backup.backup(backup, decisions, review_dir)
    with pytest.raises(SystemExit, match="already exists"):
        review_backup.backup(backup, decisions, review_dir)
    with pytest.raises(SystemExit, match="overwrite"):
        review_backup.restore(backup, decisions, review_dir)

    manifest = json.loads((backup / "manifest.json").read_text())
    manifest["files"]["label_decisions.db"]["rows"]["review_feedback"] = 99
    (backup / "manifest.json").write_text(json.dumps(manifest))
    (backup / f"reviews/{review_id}.db").unlink()
    errors = review_backup.verify(backup)
    assert any("row counts" in e for e in errors)
    assert f"reviews/{review_id}.db: missing" in errors
    with pytest.raises(SystemExit, match="not restoring"):
        review_backup.restore(backup, tmp_path / "new.db", tmp_path / "new-reviews")
    assert not (tmp_path / "new.db").exists()


def test_cli_uses_configured_paths(live, tmp_path, capsys):
    review_backup.main(["backup", str(tmp_path / "backup")])
    assert "Backed up 2 database(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_:
        review_backup.main(["verify", str(tmp_path / "backup")])
    assert exit_.value.code == 0
