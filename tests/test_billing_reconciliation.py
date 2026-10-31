"""
Summary billing totals must reconcile with the daily records (issue #60).

These tests run against the real seeded data in data.py, so they also pin
the headline numbers the docs and evals quote.
"""

from datetime import date

import pytest

import data
from data import (
    AZURE_DETAILS_DF,
    AZURE_SUMMARY_DF,
    GCP_DETAILS_DF,
    GCP_SUMMARY_DF,
    REPORTING_END,
    REPORTING_START,
    REPORTING_WINDOW_DAYS,
    in_reporting_window,
)

AZURE_COLUMNS = {
    "ingested_gb_30d": "ingested_gb",
    "overage_gb_30d": "overage_gb",
    "estimated_overage_cost_30d": "estimated_overage_cost",
}


def test_reporting_window_is_latest_30_days():
    assert REPORTING_WINDOW_DAYS == 30
    assert REPORTING_END == date(2026, 6, 1)
    assert REPORTING_START == date(2026, 5, 3)
    for df in (AZURE_DETAILS_DF, GCP_DETAILS_DF):
        assert df["day"].max() == REPORTING_END
        assert in_reporting_window(df)["day"].nunique() == REPORTING_WINDOW_DAYS


@pytest.mark.parametrize("sub", data.AZURE_SUBSCRIPTIONS)
@pytest.mark.parametrize("summary_col,daily_col", AZURE_COLUMNS.items())
def test_azure_summary_reconciles_with_daily(sub, summary_col, daily_col):
    daily = AZURE_DETAILS_DF[AZURE_DETAILS_DF["subscription_id"] == sub]
    latest = daily.sort_values("day").tail(REPORTING_WINDOW_DAYS)
    summary = AZURE_SUMMARY_DF.set_index("subscription_id").loc[sub, summary_col]
    assert summary == pytest.approx(latest[daily_col].sum(), abs=0.005)


@pytest.mark.parametrize("proj", data.GCP_PROJECTS)
def test_gcp_summary_reconciles_with_daily(proj):
    daily = GCP_DETAILS_DF[GCP_DETAILS_DF["project_id"] == proj]
    latest = daily.sort_values("day").tail(REPORTING_WINDOW_DAYS)
    summary = GCP_SUMMARY_DF.set_index("project_id").loc[proj, "total_cost"]
    assert summary == pytest.approx(latest["cost"].sum(), abs=0.005)


def test_azure_cost_matches_overage_rate():
    expected = (AZURE_DETAILS_DF["overage_gb"] * data.AZURE_OVERAGE_RATE_PER_GB).round(2)
    assert (AZURE_DETAILS_DF["estimated_overage_cost"] - expected).abs().max() <= 0.011
    assert (AZURE_DETAILS_DF["overage_gb"] <= AZURE_DETAILS_DF["ingested_gb"]).all()


def test_every_entity_summarised_in_one_currency():
    assert list(AZURE_SUMMARY_DF["subscription_id"]) == data.AZURE_SUBSCRIPTIONS
    assert list(GCP_SUMMARY_DF["project_id"]) == data.GCP_PROJECTS
    for df in (AZURE_SUMMARY_DF, AZURE_DETAILS_DF, GCP_SUMMARY_DF, GCP_DETAILS_DF):
        assert set(df["currency"]) == {data.CURRENCY}


def test_seeded_headline_numbers():
    """The figures quoted in docs/agent.md and api.py."""
    az = AZURE_SUMMARY_DF.set_index("subscription_id")["estimated_overage_cost_30d"]
    gcp = GCP_SUMMARY_DF.set_index("project_id")["total_cost"]
    assert az.idxmax() == "sub-a1b2c3d4"
    assert az.idxmin() == "sub-q7r8s9t0"
    assert gcp.idxmax() == "project-alpha"
    assert gcp.idxmin() == "project-echo"
    assert az["sub-a1b2c3d4"] == pytest.approx(20109.04)
    assert gcp["project-alpha"] == pytest.approx(13069.13)
    assert az.sum() == pytest.approx(24070.49)
    assert gcp.sum() == pytest.approx(22407.78)


def test_built_in_spikes():
    az = AZURE_DETAILS_DF[AZURE_DETAILS_DF["subscription_id"] == "sub-a1b2c3d4"]
    gcp = GCP_DETAILS_DF[GCP_DETAILS_DF["project_id"] == "project-bravo"]
    assert az.iloc[30]["day"] == date(2026, 5, 3)
    assert az["estimated_overage_cost"].idxmax() == az.index[30]
    assert gcp.iloc[20]["day"] == date(2026, 4, 23)
    assert gcp["cost"].idxmax() == gcp.index[20]


def test_tools_state_window_and_currency():
    from tools import compare_cross_cloud, get_azure_top_overages, get_gcp_top_projects

    for out in (get_azure_top_overages(), get_gcp_top_projects(), compare_cross_cloud()):
        assert "2026-05-03 to 2026-06-01 (30 days)" in out
        assert "USD" in out


def test_cross_cloud_totals_match_daily_window():
    from tools import compare_cross_cloud

    az = in_reporting_window(AZURE_DETAILS_DF)["estimated_overage_cost"].sum()
    gcp = in_reporting_window(GCP_DETAILS_DF)["cost"].sum()
    out = compare_cross_cloud()
    assert f"Azure total overage cost:  ${az:,.2f}" in out
    assert f"GCP total logging cost:    ${gcp:,.2f}" in out
    assert f"${az + gcp:,.2f}" in out
