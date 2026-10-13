from pathlib import Path

import pytest

from inventory_import import InventoryImportError, load_inventory_csv

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
