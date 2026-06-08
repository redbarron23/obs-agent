"""
Synthetic cloud billing data generator.

Produces deterministic, realistic-looking multi-cloud cost data so the
agent and eval harness work out of the box with no external CSVs.

The seed is fixed so evals pass reliably — every run produces the same
subscription IDs, project IDs, costs, and spike patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Seeded RNG for reproducibility ─────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Azure subscriptions ────────────────────────────────────────────────
AZURE_SUBSCRIPTIONS = [
    "sub-a1b2c3d4",
    "sub-e5f6g7h8",
    "sub-i9j0k1l2",
    "sub-m3n4o5p6",
    "sub-q7r8s9t0",
]

# ── GCP projects ───────────────────────────────────────────────────────
GCP_PROJECTS = [
    "project-alpha",
    "project-bravo",
    "project-charlie",
    "project-delta",
    "project-echo",
]

# ── Date range ─────────────────────────────────────────────────────────
END_DATE = datetime(2026, 6, 1)
START_DATE = END_DATE - timedelta(days=59)  # 60 days
DAYS = pd.date_range(START_DATE, END_DATE, freq="D")


def generate_azure_summary() -> pd.DataFrame:
    """Returns the per-subscription 30-day overage summary."""
    # One subscription dominates; others have moderate to near-zero overage.
    base_costs = [18400, 3200, 850, 120, 0]
    base_ingested = [520, 95, 28, 5, 2]
    base_overage = [180, 30, 8, 1, 0]

    # Add some noise so the exact ranking is robust but ordering is clear.
    ingested_noise = RNG.integers(-5, 6, size=len(base_costs))
    overage_noise = RNG.integers(-2, 3, size=len(base_costs))

    rows = []
    for i, sub in enumerate(AZURE_SUBSCRIPTIONS):
        rows.append({
            "subscription_id": sub,
            "ingested_gb_30d": max(0, base_ingested[i] + ingested_noise[i]),
            "overage_gb_30d": max(0, base_overage[i] + overage_noise[i]),
            "estimated_overage_cost_30d": max(0, base_costs[i] + RNG.integers(-200, 201)),
        })
    return pd.DataFrame(rows)


def generate_azure_details() -> pd.DataFrame:
    """Returns daily per-subscription detail with a built-in spike."""
    records = []
    for sub in AZURE_SUBSCRIPTIONS:
        day_cost = RNG.normal(loc=5, scale=1.5, size=len(DAYS)).clip(0.5)
        day_ingested = RNG.normal(loc=8, scale=2, size=len(DAYS)).clip(0.5)
        day_overage = RNG.normal(loc=0.5, scale=0.3, size=len(DAYS)).clip(0)

        # Give the top-cost subscription a spike on day 30
        if sub == "sub-a1b2c3d4":
            spike_idx = 30
            day_cost[spike_idx] = 45.0
            day_overage[spike_idx] = 3.2

        for idx, day in enumerate(DAYS):
            records.append({
                "subscription_id": sub,
                "day": day.date(),
                "ingested_gb": round(day_ingested[idx], 2),
                "overage_gb": round(day_overage[idx], 2),
                "estimated_overage_cost": round(day_cost[idx], 2),
            })
    return pd.DataFrame(records)


def generate_gcp_summary() -> pd.DataFrame:
    """Returns per-project total cost summary."""
    base_costs = [12500, 6700, 2100, 430, 0]
    rows = []
    for i, proj in enumerate(GCP_PROJECTS):
        rows.append({
            "project_id": proj,
            "total_cost": max(0, base_costs[i] + int(RNG.integers(-300, 301))),
            "currency": "USD",
        })
    return pd.DataFrame(rows)


def generate_gcp_details() -> pd.DataFrame:
    """Returns daily per-project cost detail with a built-in spike."""
    records = []
    for proj in GCP_PROJECTS:
        day_cost = RNG.exponential(scale=3, size=len(DAYS)).clip(0.1)

        # Give project-bravo a spike on day 20
        if proj == "project-bravo":
            spike_idx = 20
            day_cost[spike_idx] = 28.0

        for idx, day in enumerate(DAYS):
            records.append({
                "project_id": proj,
                "day": day.date(),
                "cost": round(day_cost[idx], 2),
            })
    return pd.DataFrame(records)


# ── Public in-memory data ──────────────────────────────────────────────
# These are imported by tools.py instead of reading real CSVs.

AZURE_SUMMARY_DF = generate_azure_summary()
AZURE_DETAILS_DF = generate_azure_details()
GCP_SUMMARY_DF = generate_gcp_summary()
GCP_DETAILS_DF = generate_gcp_details()
