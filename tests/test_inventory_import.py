from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from inventory_import import (
    InventoryImportError,
    apply_column_mapping,
    csv_columns,
    load_inventory_csv,
    suggest_mapping,
)
from review_workspaces import load_review, save_review, saved_reviews

ROOT = Path(__file__).parent.parent

SAMPLE = Path(__file__).parent.parent / "fixtures" / "sanitized_inventory_sample.csv"

CSV = (
    b"cloud,resource_id,parent,owners,created_by,top_services,avg_daily_cost,weekend_ratio,"
    b"cost_cv,team,env,cost_center\n"
    b"azure,az-001,mg-platform,alice|bob,alice,AKS|SQL,120,0.9,0.2,identity,prod,CC-4001\n"
    b"azure,az-002,mg-platform,alice|bob,bob,AKS|KeyVault,110,0.8,0.2,identity,prod,CC-4001\n"
    b'gcp,gcp-001,folders/data,"[""fatima""]",fatima,BigQuery|GCS,42,0.3,0.4,data-platform,dev,CC-2002\n'
    b"gcp,gcp-002,folders/data,fatima|gus,gus,Dataflow|GCS,38,0.3,0.4,data-platform,dev,CC-2002\n"
    b"gcp,gcp-003,folders/data,fatima|gus,fatima,BigQuery|GCS,36,0.3,0.4,,,\n"
)


def test_loads_sanitized_inventory():
    df = load_inventory_csv(CSV)
    assert list(df["owners"].iloc[0]) == ["alice", "bob"]
    assert list(df["top_services"].iloc[2]) == ["BigQuery", "GCS"]
    assert {"billing_account", "created_at", "region"} <= set(df.columns)


def test_checked_in_sample_is_valid():
    df = load_inventory_csv(SAMPLE)
    assert len(df) == 5
    assert set(df["cloud"]) == {"azure", "gcp"}
    assert df["team"].isna().sum() == 1


def test_rejects_missing_columns():
    with pytest.raises(InventoryImportError, match="Missing required columns"):
        load_inventory_csv(b"cloud,resource_id\nazure,az-001\n")


# ── Row-level validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b"azure,az-002", b"aws,az-002", r"found aws at row 3\."),
        (b"gcp,gcp-003,", b"gcp,az-001,", r"repeated at rows 2, 6\."),
        (b"gcp,gcp-003,", b"gcp,,", r"blank at row 6\."),
        (b",38,0.3", b",n/a,0.3", r"avg_daily_cost .* invalid at row 5\."),
        (b"alice|bob,bob", b"[bob,bob", r"invalid JSON list \(row 3\)"),
        (b"identity,prod,CC-4001\naz", b"identity,dev,CC-4001\naz", r"env .* found dev=3, prod=1"),
    ],
)
def test_validation_names_the_rows_to_correct(old, new, message):
    assert old in CSV
    with pytest.raises(InventoryImportError, match=message):
        load_inventory_csv(CSV.replace(old, new, 1))


# ── Column mapping ─────────────────────────────────────────────────────

EXPORT = (
    b"Provider,Subscription ID,Management Group,Owner,Creator,Services,Daily Cost,"
    b"Weekend Ratio,Cost CV,Team,Environment,Cost Centre,Notes\n"
    + b"\n".join(CSV.splitlines()[1:])
    .replace(b",CC-4001", b",CC-4001,keep out")
    .replace(b",CC-2002", b",CC-2002,")
    .replace(b",,,", b",,,,")
    + b"\n"
)
EXPORT_MAPPING = {
    "cloud": "Provider",
    "resource_id": "Subscription ID",
    "parent": "Management Group",
    "owners": "Owner",
    "created_by": "Creator",
    "top_services": "Services",
    "avg_daily_cost": "Daily Cost",
    "weekend_ratio": "Weekend Ratio",
    "cost_cv": "Cost CV",
    "team": "Team",
    "env": "Environment",
    "cost_center": "Cost Centre",
}


def test_renamed_columns_are_suggested_mapped_and_match_a_direct_import():
    assert suggest_mapping(csv_columns(EXPORT)) == EXPORT_MAPPING
    mapped = load_inventory_csv(apply_column_mapping(EXPORT, EXPORT_MAPPING))
    pd.testing.assert_frame_equal(mapped, load_inventory_csv(CSV))
    assert "Notes" not in mapped


def test_unmapped_required_fields_are_named_not_invented():
    mapping = EXPORT_MAPPING | {"owners": None, "avg_daily_cost": None}
    with pytest.raises(InventoryImportError, match="unmapped: avg_daily_cost, owners"):
        apply_column_mapping(EXPORT, mapping)


def test_one_source_column_cannot_fill_two_fields():
    mapping = EXPORT_MAPPING | {"created_by": "Owner"}
    with pytest.raises(InventoryImportError, match="Owner is mapped to created_by, owners"):
        apply_column_mapping(EXPORT, mapping)


def test_mapping_rejects_columns_absent_from_the_csv():
    with pytest.raises(InventoryImportError, match="not in the CSV: Region"):
        apply_column_mapping(EXPORT, EXPORT_MAPPING | {"region": "Region"})


def test_ambiguous_headers_are_left_for_the_reviewer():
    assert "resource_id" not in suggest_mapping(["Subscription", "Project", "Owner"])


def test_mapping_keeps_blank_cost_data_blank():
    blank = EXPORT.replace(b",120,0.9", b",,0.9")
    with pytest.raises(InventoryImportError, match="avg_daily_cost .* at row 2"):
        load_inventory_csv(apply_column_mapping(blank, EXPORT_MAPPING))


def test_mapping_keeps_unrecognized_labels_visible():
    legacy = EXPORT.replace(b"0.4,,,,", b"0.4,legacy-team,,,", 1)
    inventory = load_inventory_csv(apply_column_mapping(legacy, EXPORT_MAPPING))
    assert inventory.loc[inventory.resource_id == "gcp-003", "team"].item() == "legacy-team"


def _import_page(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page("pages/2_Import_inventory.py").run()
    return at


def test_import_page_maps_saves_and_reopens_a_renamed_csv(monkeypatch, tmp_path):
    at = _import_page(monkeypatch, tmp_path)
    at.file_uploader[1].set_value(("export.csv", EXPORT, "text/csv")).run()
    assert not at.exception and not at.error
    assert "Map your columns" in [h.value for h in at.subheader]
    next(b for b in at.button if b.label == "Save and open review").click().run()
    assert not at.exception
    (review,) = saved_reviews()
    inventory, _ = load_review(review["id"])
    pd.testing.assert_frame_equal(inventory, load_inventory_csv(CSV))
    at.switch_page("pages/1_Label_review.py").run()
    assert not at.exception


def test_import_page_explains_an_incomplete_mapping(monkeypatch, tmp_path):
    at = _import_page(monkeypatch, tmp_path)
    at.file_uploader[1].set_value(("export.csv", EXPORT, "text/csv")).run()
    next(s for s in at.selectbox if s.label == "owners").set_value("— not mapped —").run()
    assert "unmapped: owners" in at.error[0].value
    assert not [b for b in at.button if b.label == "Save and open review"]


def test_schema_compatible_csv_skips_mapping(monkeypatch, tmp_path):
    at = _import_page(monkeypatch, tmp_path)
    at.file_uploader[1].set_value(("inventory.csv", CSV, "text/csv")).run()
    assert "Map your columns" not in [h.value for h in at.subheader]
    next(b for b in at.button if b.label == "Save and open review").click().run()
    assert saved_reviews()[0]["id"] == save_review(CSV, "inventory.csv")
