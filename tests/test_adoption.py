"""Adoption workflow: custom vocabulary, persistence, and approval provenance."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api
import label_review as lr
from inventory_import import InventoryImportError, load_inventory_csv
from label_config import LabelConfigError, load_label_config
from labeling import Rules, clean, normalise
from review_context import review_context
from review_workspaces import load_review, save_review, saved_reviews

ROOT = Path(__file__).parent.parent
SAMPLE = (ROOT / "fixtures/sanitized_inventory_sample.csv").read_bytes()
CUSTOM_CSV = (
    SAMPLE.replace(b"payments", b"platform-engineering")
    .replace(b"data-platform", b"customer-success")
    .replace(b",prod,", b",production,")
    .replace(b",dev,", b",development,")
)
CUSTOM_CONFIG = {
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
def imported():
    config = load_label_config(json.dumps(CUSTOM_CONFIG).encode())
    return load_inventory_csv(CUSTOM_CSV, config)


def test_custom_vocabulary_survives_inference_override_and_export(imported, tmp_path):
    assert normalise("team", "Platform", imported.attrs["label_config"]) == "platform-engineering"
    report = lr.resource_report("gcp-demo-003", imported)
    assert report["labels"]["team"]["suggested"] in CUSTOM_CONFIG["team"]["values"]
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "override", "Platform", imported, reviewer="alice")
    plan = lr.approved_change_plan(store, imported)
    assert plan["changes"][0]["tags"]["team"]["new"] == "platform-engineering"
    assert plan["approvals"][0]["reviewer"] == "alice"
    assert plan["approvals"][0]["inventory_id"] == plan["inventory_id"]
    assert plan["approvals"][0]["model_version"] == plan["model_version"]


@pytest.mark.parametrize("fmt", lr.EXPORT_FORMATS)
@pytest.mark.parametrize("value", ["CC-2,env=prod", "CC=2", "CC 2", "x" * 64])
def test_invalid_gcp_cost_center_cannot_be_exported(tmp_path, fmt, value):
    source = CUSTOM_CSV.replace(b"CC-2002", f'"{value}"'.encode())
    imported = load_inventory_csv(source, CUSTOM_CONFIG)
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    with pytest.raises(lr.DecisionError, match="gcp-demo-003.*cost_center"):
        lr.export(store, fmt, inventory=imported)


def test_gcp_export_formats_share_destination_values(imported, tmp_path):
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    plan = json.loads(lr.export(store, "plan", inventory=imported))
    assert plan["changes"][0]["tags"]["cost_center"]["new"] == "cc-2002"
    assert ",cost_center,,cc-2002,accept" in lr.export(store, "csv", inventory=imported)
    assert "cost_center=cc-2002" in lr.export(store, "sh", inventory=imported)
    assert (
        json.loads(lr.export(store, "tfvars", inventory=imported))["project_labels"][
            "gcp-demo-003"
        ]["cost_center"]
        == "cc-2002"
    )


@pytest.mark.parametrize("values", [["CC-1", "CC-2"], ["CC-1", "CC-1", "CC-2"]])
def test_conflicting_cost_centers_remain_visible_without_export(imported, tmp_path, values):
    # Repeat one labelled row to exercise both a tie and a majority mapping.
    if len(values) == 3:
        extra = imported.iloc[[2]].copy()
        extra["resource_id"] = "gcp-extra"
        imported = pd.concat([imported, extra], ignore_index=True)
    imported.loc[imported.team == "customer-success", "cost_center"] = values
    report = lr.resource_report("gcp-demo-003", imported)
    assert report["cost_center"]["implied"] is None
    assert report["cost_center"]["conflicting"] == ["CC-1", "CC-2"]
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    assert set(lr.approved_change_plan(store, imported)["changes"][0]["tags"]) == {"team"}


@pytest.mark.parametrize(
    "cloud,value", [("gcp", "Équipe_2"), ("gcp", "x" * 63), ("azure", "X" * 256)]
)
def test_valid_destination_values_are_preserved(imported, tmp_path, cloud, value):
    imported.loc[imported.team == "customer-success", "cost_center"] = value
    imported.loc[imported.resource_id == "gcp-demo-003", "cloud"] = cloud
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    tags = lr.approved_change_plan(store, imported)["changes"][0]["tags"]
    assert tags["cost_center"]["new"] == (value.lower() if cloud == "gcp" else value)


def test_oversized_azure_tag_is_rejected(imported, tmp_path):
    imported.loc[imported.team == "customer-success", "cost_center"] = "X" * 257
    imported.loc[imported.resource_id == "gcp-demo-003", "cloud"] = "azure"
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    with pytest.raises(lr.DecisionError, match="Azure values must be at most 256"):
        lr.export(store, "sh", inventory=imported)


def test_review_page_reports_invalid_export_values(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    source = CUSTOM_CSV.replace(b"CC-2002", b'"CC-2002,env=prod"')
    review_id = save_review(source, "invalid-cost-center.csv", CUSTOM_CONFIG)
    inventory, store = load_review(review_id)
    store.set("gcp-demo-003", "team", "accept", inventory=inventory)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["review_id"] = review_id
    at.run()
    assert not at.exception
    assert "gcp-demo-003: invalid cost_center" in at.error[0].value
    assert not at.get("download_button")


def test_custom_environment_rules_do_not_emit_demo_values(imported):
    rules = Rules(clean(imported), "env")
    assert rules.suggest({"resource_id": "svc-prod"}).label == "production"
    config = imported.attrs["label_config"]
    config["env"]["name_tokens"] = {}
    assert Rules(clean(imported), "env").suggest({"resource_id": "svc-prod"}).label is None


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"team": {}, "env": {}},
        {"team": {"values": ["x", "x"]}, "env": {"values": ["prod", "dev"]}},
        {
            "team": {"values": ["x", "y"], "aliases": {"z": "missing"}},
            "env": {"values": ["prod", "dev"]},
        },
        {
            "team": {"values": ["x", "y"]},
            "env": {"values": ["prod", "dev"], "name_tokens": {"prod": ["p"], "dev": ["p"]}},
        },
    ],
)
def test_invalid_vocabulary_is_rejected(bad):
    with pytest.raises(LabelConfigError):
        load_label_config(json.dumps(bad).encode())


@pytest.mark.parametrize(
    "source",
    [
        b"",
        SAMPLE.replace(b"azure,", b","),
        SAMPLE.replace(b"120.50,", b"inf,"),
        SAMPLE.replace(b"120.50,", b"-1,"),
    ],
)
def test_invalid_inventory_is_rejected(source):
    with pytest.raises(InventoryImportError):
        load_inventory_csv(source)


def test_saved_review_reopens_inventory_vocabulary_and_decisions(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "private reviews"))
    review_id = save_review(CUSTOM_CSV, "private.csv", CUSTOM_CONFIG)
    df, store = load_review(review_id)
    store.set("gcp-demo-003", "team", "accept", inventory=df, reviewer="alice")
    assert save_review(CUSTOM_CSV, "second-name.csv", CUSTOM_CONFIG) == review_id
    reopened, reopened_store = load_review(review_id)
    assert reopened.attrs == df.attrs
    pd.testing.assert_frame_equal(df, reopened)
    assert reopened_store.get("gcp-demo-003", "team")["reviewer"] == "alice"
    assert lr.approved_change_plan(reopened_store, reopened)["changes"]
    assert saved_reviews()[0]["id"] == review_id
    changed_id = save_review(
        CUSTOM_CSV.replace(b"120.50,", b"121.50,"), "changed.csv", CUSTOM_CONFIG
    )
    assert changed_id != review_id
    assert load_review(changed_id)[1].all() == []
    with pytest.raises(ValueError, match="Invalid saved review ID"):
        load_review("../escape")


def test_old_snapshot_or_model_approvals_are_not_exported(imported, tmp_path, monkeypatch):
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    changed = imported.copy()
    changed.loc[0, "parent"] = "different-parent"
    assert lr.approved_change_plan(store, changed)["changes"] == []
    assert lr.current_decisions(store, changed) == []
    original_context = review_context(imported)
    monkeypatch.setattr(
        lr, "review_context", lambda inventory: original_context | {"model_version": "new"}
    )
    assert lr.approved_change_plan(store, imported)["changes"] == []
    assert len(store.all()) == 1  # preserve the history


def test_legacy_sqlite_is_migrated_without_inventing_provenance(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE decisions (resource_id TEXT, label TEXT, decision TEXT, "
            "value TEXT, suggested TEXT, previous TEXT, decided_at TEXT, "
            "PRIMARY KEY (resource_id, label))"
        )
        conn.execute(
            "INSERT INTO decisions VALUES ('svc-ingest', 'team', 'accept', "
            "'data-platform', 'data-platform', NULL, '2026-01-01')"
        )
    store = lr.DecisionStore(path)
    assert store.all()[0]["reviewer"] is None
    assert lr.approved_change_plan(store)["changes"] == []
    store.set("svc-ingest", "team", "accept", reviewer="alice")
    assert lr.approved_change_plan(store)["approvals"][0]["reviewer"] == "alice"


def test_missing_reviewer_is_rejected_without_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("OBS_AGENT_REVIEWER")
    store = lr.DecisionStore(tmp_path / "review.db")
    with pytest.raises(lr.DecisionError, match="reviewer"):
        store.set("svc-ingest", "team", "accept")
    assert not store.all() and not store.feedback()


def test_override_without_cost_center_mapping_still_exports(imported, tmp_path):
    imported.loc[imported.team == "platform-engineering", "cost_center"] = None
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "override", "platform-engineering", imported)
    tags = lr.approved_change_plan(store, imported)["changes"][0]["tags"]
    assert "team" in tags and "cost_center" not in tags


def test_api_records_reviewer_and_defaults_to_reviewed_only(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_AGENT_API_KEY", "test-key")
    monkeypatch.delenv("OBS_AGENT_REVIEWER")
    monkeypatch.setattr(api, "decisions", lr.DecisionStore(tmp_path / "api.db"))
    client = TestClient(api.app, headers={"X-API-Key": "test-key"})
    url = "/resources/svc-ingest/labels/team/decision"
    assert client.put(url, json={"decision": "accept"}).status_code == 422
    result = client.put(url, json={"decision": "accept", "reviewer": "alice"})
    assert result.status_code == 200 and result.json()["reviewer"] == "alice"
    assert client.get("/labels/feedback").json()[0]["inventory_id"] == result.json()["inventory_id"]
    assert all("svc-ingest" in row for row in client.get("/labels/export").text.splitlines()[1:])


def test_application_lands_on_review_without_provider_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_DECISIONS", str(tmp_path / "decisions.db"))
    monkeypatch.setenv("OBS_AGENT_PROVIDER", "invalid-provider")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "🏷️ Label review"
    assert not at.chat_input


def test_reopened_review_renders_custom_override_choices(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    review_id = save_review(CUSTOM_CSV, "custom.csv", CUSTOM_CONFIG)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["review_id"] = review_id
    at.run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Need attention"] == "1"
    # Choosing a saved snapshot works in a new browser session too.
    reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    reopened.switch_page("pages/2_Import_inventory.py").run()
    assert not reopened.exception
    next(b for b in reopened.button if b.label == "Open saved review").click().run()
    assert reopened.session_state["review_id"] == review_id
    reopened.switch_page("pages/1_Label_review.py").run()
    assert not reopened.exception
    assert {m.label: m.value for m in reopened.metric}["Need attention"] == "1"


def test_downloadable_vocabulary_matches_inventory_template():
    config = load_label_config(ROOT / "fixtures/label_config_example.json")
    assert len(load_inventory_csv(SAMPLE, config)) == 5


def test_custom_canonical_names_are_not_stripped():
    config = CUSTOM_CONFIG | {"team": {"values": ["team-platform", "other-team"], "aliases": {}}}
    assert normalise("team", "team-platform", config) == "team-platform"
    assert normalise("team", "other-team", config) == "other-team"


def test_export_preserves_unrecognized_original_tag(imported, tmp_path):
    imported.loc[imported.resource_id == "gcp-demo-003", "team"] = "legacy-unknown-team"
    store = lr.DecisionStore(tmp_path / "review.db")
    store.set("gcp-demo-003", "team", "accept", inventory=imported)
    assert (
        lr.approved_change_plan(store, imported)["changes"][0]["tags"]["team"]["previous"]
        == "legacy-unknown-team"
    )


def test_import_retains_leading_zero_identifiers():
    inventory = load_inventory_csv(SAMPLE.replace(b"az-demo-001", b"00001"))
    assert inventory.iloc[0]["resource_id"] == "00001"


def test_malformed_json_owner_list_is_rejected():
    with pytest.raises(InventoryImportError, match="invalid JSON list"):
        load_inventory_csv(SAMPLE.replace(b"alice|bob", b"[broken"))
