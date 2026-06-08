"""
Shared test fixtures — tiny, hand-crafted DataFrames with known values.

These are used instead of the full data.py output so tests are fast,
predictable, and don't depend on numpy random state.
"""

import pytest
import pandas as pd
from datetime import date


@pytest.fixture
def azure_summary():
    """5 Azure subscriptions with a clear cost ranking."""
    return pd.DataFrame([
        {"subscription_id": "sub-a", "ingested_gb_30d": 100, "overage_gb_30d": 10, "estimated_overage_cost_30d": 5000},
        {"subscription_id": "sub-b", "ingested_gb_30d": 50, "overage_gb_30d": 5, "estimated_overage_cost_30d": 2000},
        {"subscription_id": "sub-c", "ingested_gb_30d": 20, "overage_gb_30d": 2, "estimated_overage_cost_30d": 800},
        {"subscription_id": "sub-d", "ingested_gb_30d": 5, "overage_gb_30d": 0, "estimated_overage_cost_30d": 100},
        {"subscription_id": "sub-e", "ingested_gb_30d": 1, "overage_gb_30d": 0, "estimated_overage_cost_30d": 0},
    ])


@pytest.fixture
def azure_details():
    """Daily detail for sub-a with a known spike on day 3."""
    return pd.DataFrame([
        {"subscription_id": "sub-a", "day": date(2026, 5, 1), "ingested_gb": 3.0, "overage_gb": 0.3, "estimated_overage_cost": 150.0},
        {"subscription_id": "sub-a", "day": date(2026, 5, 2), "ingested_gb": 3.1, "overage_gb": 0.4, "estimated_overage_cost": 160.0},
        {"subscription_id": "sub-a", "day": date(2026, 5, 3), "ingested_gb": 5.0, "overage_gb": 2.0, "estimated_overage_cost": 800.0},  # spike
        {"subscription_id": "sub-a", "day": date(2026, 5, 4), "ingested_gb": 2.9, "overage_gb": 0.2, "estimated_overage_cost": 140.0},
        # sub-b, no spikes
        {"subscription_id": "sub-b", "day": date(2026, 5, 1), "ingested_gb": 1.5, "overage_gb": 0.1, "estimated_overage_cost": 50.0},
        {"subscription_id": "sub-b", "day": date(2026, 5, 2), "ingested_gb": 1.6, "overage_gb": 0.1, "estimated_overage_cost": 55.0},
        {"subscription_id": "sub-b", "day": date(2026, 5, 3), "ingested_gb": 1.4, "overage_gb": 0.1, "estimated_overage_cost": 52.0},
        {"subscription_id": "sub-b", "day": date(2026, 5, 4), "ingested_gb": 1.5, "overage_gb": 0.1, "estimated_overage_cost": 53.0},
    ])


@pytest.fixture
def gcp_summary():
    """5 GCP projects with a clear cost ranking."""
    return pd.DataFrame([
        {"project_id": "proj-alpha", "total_cost": 8000, "currency": "USD"},
        {"project_id": "proj-bravo", "total_cost": 4000, "currency": "USD"},
        {"project_id": "proj-charlie", "total_cost": 1000, "currency": "USD"},
        {"project_id": "proj-delta", "total_cost": 200, "currency": "USD"},
        {"project_id": "proj-echo", "total_cost": 0, "currency": "USD"},
    ])


@pytest.fixture
def gcp_details():
    """Daily detail for proj-bravo with a known spike on day 2."""
    return pd.DataFrame([
        {"project_id": "proj-alpha", "day": date(2026, 5, 1), "cost": 200.0},
        {"project_id": "proj-alpha", "day": date(2026, 5, 2), "cost": 210.0},
        {"project_id": "proj-alpha", "day": date(2026, 5, 3), "cost": 190.0},
        # proj-bravo spikes on day 2
        {"project_id": "proj-bravo", "day": date(2026, 5, 1), "cost": 100.0},
        {"project_id": "proj-bravo", "day": date(2026, 5, 2), "cost": 450.0},  # spike
        {"project_id": "proj-bravo", "day": date(2026, 5, 3), "cost": 110.0},
    ])
