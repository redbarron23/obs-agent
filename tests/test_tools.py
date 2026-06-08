"""
Unit tests for all 5 tool functions in tools.py.

These tests use the fixtures from conftest.py — small DataFrames with
known values — and patch the module-level DataFrames that tools.py imports
from data.py.  No LLM API calls are made.
"""

from unittest.mock import patch

import pytest

# ── Fixtures that patch tools.py's data sources ────────────────────────


@pytest.fixture
def patch_azure_summary(azure_summary):
    with patch("tools.AZURE_SUMMARY_DF", azure_summary):
        yield


@pytest.fixture
def patch_azure_details(azure_details):
    with patch("tools.AZURE_DETAILS_DF", azure_details):
        yield


@pytest.fixture
def patch_gcp_summary(gcp_summary):
    with patch("tools.GCP_SUMMARY_DF", gcp_summary):
        yield


@pytest.fixture
def patch_gcp_details(gcp_details):
    with patch("tools.GCP_DETAILS_DF", gcp_details):
        yield


# ── Tests ──────────────────────────────────────────────────────────────


class TestGetAzureTopOverages:
    def test_top_1(self, patch_azure_summary):
        from tools import get_azure_top_overages
        result = get_azure_top_overages(n=1)
        assert "sub-a" in result
        assert "5000" in result
        assert "sub-b" not in result

    def test_top_3(self, patch_azure_summary):
        from tools import get_azure_top_overages
        result = get_azure_top_overages(n=3)
        assert "sub-a" in result
        assert "sub-b" in result
        assert "sub-c" in result
        assert "sub-d" not in result  # only 3 returned

    def test_filters_zero_overage(self, patch_azure_summary):
        """Subscriptions with $0 overage should be excluded."""
        from tools import get_azure_top_overages
        result = get_azure_top_overages(n=5)
        assert "sub-e" not in result

    def test_all_zero_overage(self, patch_azure_summary):
        """When all are zero, return the 'none found' message."""
        from tools import get_azure_top_overages
        # Temporarily patch to all-zero
        import pandas as pd
        zero_df = pd.DataFrame([
            {"subscription_id": "sub-a", "ingested_gb_30d": 10, "overage_gb_30d": 0, "estimated_overage_cost_30d": 0},
        ])
        with patch("tools.AZURE_SUMMARY_DF", zero_df):
            result = get_azure_top_overages()
        assert "No Azure subscriptions with overage found" in result

    def test_columns_present(self, patch_azure_summary):
        """Output must contain the expected column headers."""
        from tools import get_azure_top_overages
        result = get_azure_top_overages(n=2)
        assert "subscription_id" in result
        assert "estimated_overage_cost_30d" in result
        assert "ingested_gb_30d" in result
        assert "overage_gb_30d" in result


class TestGetGcpTopProjects:
    def test_top_1(self, patch_gcp_summary):
        from tools import get_gcp_top_projects
        result = get_gcp_top_projects(n=1)
        assert "proj-alpha" in result
        assert "8000" in result
        assert "proj-bravo" not in result

    def test_top_3(self, patch_gcp_summary):
        from tools import get_gcp_top_projects
        result = get_gcp_top_projects(n=3)
        assert "proj-alpha" in result
        assert "proj-bravo" in result
        assert "proj-charlie" in result
        assert "proj-delta" not in result

    def test_filters_zero_cost(self, patch_gcp_summary):
        from tools import get_gcp_top_projects
        result = get_gcp_top_projects(n=5)
        assert "proj-echo" not in result

    def test_all_zero_cost(self, patch_gcp_summary):
        from tools import get_gcp_top_projects
        import pandas as pd
        zero_df = pd.DataFrame([
            {"project_id": "proj-alpha", "total_cost": 0, "currency": "USD"},
        ])
        with patch("tools.GCP_SUMMARY_DF", zero_df):
            result = get_gcp_top_projects()
        assert "No GCP projects with costs found" in result

    def test_currency_present(self, patch_gcp_summary):
        from tools import get_gcp_top_projects
        result = get_gcp_top_projects(n=2)
        assert "USD" in result


class TestGetDailyTrend:
    def test_azure_existing_subscription(self, patch_azure_details):
        from tools import get_daily_trend
        result = get_daily_trend("azure", "sub-a")
        # The result doesn't include the subscription_id in the table
        # (it's grouped by day), but it should contain the data.
        assert "2026-05-01" in result
        assert "2026-05-04" in result  # last day of fixture data
        assert "150.0" in result  # cost on day 1
        assert "800.0" in result  # spike on day 3

    def test_azure_nonexistent_subscription(self, patch_azure_details):
        from tools import get_daily_trend
        result = get_daily_trend("azure", "sub-unknown")
        assert "No Azure data found" in result

    def test_gcp_existing_project(self, patch_gcp_details):
        from tools import get_daily_trend
        result = get_daily_trend("gcp", "proj-bravo")
        # Result doesn't include project_id in table, but should have data
        assert "2026-05-01" in result
        assert "100.0" in result
        assert "450.0" in result  # spike

    def test_gcp_nonexistent_project(self, patch_gcp_details):
        from tools import get_daily_trend
        result = get_daily_trend("gcp", "proj-unknown")
        assert "No GCP data found" in result

    def test_unknown_platform(self, patch_azure_details):
        from tools import get_daily_trend
        result = get_daily_trend("aws", "something")
        assert "Unknown platform" in result

    def test_azure_case_insensitive(self, patch_azure_details):
        """Platform should match case-insensitively."""
        from tools import get_daily_trend
        result_upper = get_daily_trend("Azure", "sub-a")
        result_lower = get_daily_trend("azure", "sub-a")
        assert result_upper == result_lower

    def test_daily_columns_azure(self, patch_azure_details):
        from tools import get_daily_trend
        result = get_daily_trend("azure", "sub-a")
        assert "ingested_gb" in result
        assert "overage_gb" in result
        assert "estimated_overage_cost" in result

    def test_daily_columns_gcp(self, patch_gcp_details):
        from tools import get_daily_trend
        result = get_daily_trend("gcp", "proj-bravo")
        assert "cost" in result


class TestFindSpikes:
    def test_detects_azure_spike(self, patch_azure_details):
        """sub-a spikes 400% on day 3 (150→160→800)."""
        from tools import find_spikes
        result = find_spikes(threshold_pct=200)
        assert "[Azure]" in result
        assert "sub-a" in result
        assert "800.00" in result

    def test_detects_gcp_spike(self, patch_gcp_details):
        """proj-bravo spikes 350% on day 2 (100→450)."""
        from tools import find_spikes
        result = find_spikes(threshold_pct=200)
        assert "[GCP]" in result
        assert "proj-bravo" in result

    def test_no_spikes_above_threshold(self, patch_azure_details, patch_gcp_details):
        """High threshold should return 'none found'."""
        from tools import find_spikes
        result = find_spikes(threshold_pct=9999)
        assert "No spikes above" in result

    def test_lower_threshold_catches_more(self, patch_azure_details, patch_gcp_details):
        """Lower threshold should detect the spike that higher threshold misses."""
        from tools import find_spikes
        high = find_spikes(threshold_pct=9999)
        low = find_spikes(threshold_pct=50)
        assert "No spikes above" in high
        assert "No spikes above" not in low

    def test_spike_format(self, patch_azure_details):
        """Spike output should include percentage and dollar amount."""
        from tools import find_spikes
        result = find_spikes(threshold_pct=200)
        assert "+" in result  # percentage sign
        assert "$" in result    # dollar amount

    def test_azure_and_gcp_both_reported(self, patch_azure_details, patch_gcp_details):
        from tools import find_spikes
        result = find_spikes(threshold_pct=200)
        assert "[Azure]" in result
        assert "[GCP]" in result


class TestCompareCrossCloud:
    def test_azure_total(self, patch_azure_summary, patch_gcp_summary):
        """Azure total = 5000+2000+800+100+0 = 7900."""
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "Azure total overage cost" in result
        assert "7,900" in result or "7900" in result

    def test_gcp_total(self, patch_azure_summary, patch_gcp_summary):
        """GCP total = 8000+4000+1000+200+0 = 13200."""
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "GCP total logging cost" in result
        assert "13,200" in result or "13200" in result

    def test_grand_total(self, patch_azure_summary, patch_gcp_summary):
        """7900 + 13200 = 21100."""
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "Grand total" in result
        assert "21,100" in result or "21100" in result

    def test_breakdown_azure(self, patch_azure_summary, patch_gcp_summary):
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "sub-a" in result
        assert "sub-b" in result
        assert "sub-e" in result  # even $0 is shown in the breakdown

    def test_breakdown_gcp(self, patch_azure_summary, patch_gcp_summary):
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "proj-alpha" in result
        assert "proj-echo" in result

    def test_all_major_sections_present(self, patch_azure_summary, patch_gcp_summary):
        from tools import compare_cross_cloud
        result = compare_cross_cloud()
        assert "Cross-Cloud Cost Summary" in result
        assert "Azure subscription breakdown" in result
        assert "GCP project breakdown" in result
        assert "=" in result
        assert "─" in result
