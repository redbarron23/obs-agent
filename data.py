"""
Synthetic cloud billing data generator.

Produces deterministic, realistic-looking multi-cloud cost data so the
agent and eval harness work out of the box with no external CSVs.

The seed is fixed so evals pass reliably — every run produces the same
subscription IDs, project IDs, costs, and spike patterns.

The daily detail records are the single source of truth.  The per-entity
summaries are *derived* from them for an explicit reporting window (the
latest ``REPORTING_WINDOW_DAYS`` days, ending ``END_DATE``), so a summary
total always equals the sum of the matching daily rows.  All amounts are
in ``CURRENCY``.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Seeded RNG for reproducibility ─────────────────────────────────────
RNG = np.random.default_rng(seed=42)

CURRENCY = "USD"

# Flat synthetic price for Azure log ingestion beyond a subscription's
# allowance.  A single rate keeps cost and overage GB consistent; it is not
# a pricing engine.
AZURE_OVERAGE_RATE_PER_GB = 2.30

# ── Azure subscriptions ────────────────────────────────────────────────
# Mean daily (ingested GB, overage GB); the difference is the daily
# allowance.  One subscription dominates; the last one is near zero.
AZURE_PROFILES = {
    "sub-a1b2c3d4": (300.0, 260.0),
    "sub-e5f6g7h8": (60.0, 45.0),
    "sub-i9j0k1l2": (20.0, 12.0),
    "sub-m3n4o5p6": (6.0, 1.7),
    "sub-q7r8s9t0": (2.0, 0.05),
}
AZURE_SUBSCRIPTIONS = list(AZURE_PROFILES)

# ── GCP projects ───────────────────────────────────────────────────────
# Mean daily logging cost.
GCP_PROFILES = {
    "project-alpha": 420.0,
    "project-bravo": 220.0,
    "project-charlie": 70.0,
    "project-delta": 14.0,
    "project-echo": 0.5,
}
GCP_PROJECTS = list(GCP_PROFILES)

# ── Date range ─────────────────────────────────────────────────────────
END_DATE = datetime(2026, 6, 1)
START_DATE = END_DATE - timedelta(days=59)  # 60 days
DAYS = pd.date_range(START_DATE, END_DATE, freq="D")

# ── Reporting window used by every summary ─────────────────────────────
REPORTING_WINDOW_DAYS = 30
REPORTING_END = END_DATE.date()
REPORTING_START = (END_DATE - timedelta(days=REPORTING_WINDOW_DAYS - 1)).date()

# Built-in spikes (index into DAYS) and their size relative to the baseline.
AZURE_SPIKE = ("sub-a1b2c3d4", 30)
GCP_SPIKE = ("project-bravo", 20)
SPIKE_MULTIPLIER = 4.5


def reporting_window_label() -> str:
    """Human-readable reporting window, e.g. '2026-05-03 to 2026-06-01 (30 days)'."""
    return f"{REPORTING_START} to {REPORTING_END} ({REPORTING_WINDOW_DAYS} days)"


def in_reporting_window(df: pd.DataFrame) -> pd.DataFrame:
    """Daily rows that fall inside the reporting window."""
    return df[(df["day"] >= REPORTING_START) & (df["day"] <= REPORTING_END)]


def generate_azure_details() -> pd.DataFrame:
    """Returns daily per-subscription detail with a built-in spike."""
    records = []
    for sub, (ingested_mean, overage_mean) in AZURE_PROFILES.items():
        # Ingestion is the subscription's daily allowance plus that day's overage.
        allowance = ingested_mean - overage_mean
        day_overage = RNG.normal(loc=1.0, scale=0.1, size=len(DAYS)).clip(0.5) * overage_mean

        if sub == AZURE_SPIKE[0]:
            day_overage[AZURE_SPIKE[1]] = overage_mean * SPIKE_MULTIPLIER

        day_ingested = allowance + day_overage

        for idx, day in enumerate(DAYS):
            overage_gb = round(float(day_overage[idx]), 2)
            records.append(
                {
                    "subscription_id": sub,
                    "day": day.date(),
                    "ingested_gb": round(float(day_ingested[idx]), 2),
                    "overage_gb": overage_gb,
                    "estimated_overage_cost": round(overage_gb * AZURE_OVERAGE_RATE_PER_GB, 2),
                    "currency": CURRENCY,
                }
            )
    return pd.DataFrame(records)


def generate_azure_summary(details: pd.DataFrame) -> pd.DataFrame:
    """Per-subscription totals over the reporting window, summed from *details*."""
    window = in_reporting_window(details)
    summary = (
        window.groupby("subscription_id", sort=False)[
            ["ingested_gb", "overage_gb", "estimated_overage_cost"]
        ]
        .sum()
        .round(2)
        .reset_index()
        .rename(
            columns={
                "ingested_gb": "ingested_gb_30d",
                "overage_gb": "overage_gb_30d",
                "estimated_overage_cost": "estimated_overage_cost_30d",
            }
        )
    )
    summary["currency"] = CURRENCY
    return summary


def generate_gcp_details() -> pd.DataFrame:
    """Returns daily per-project cost detail with a built-in spike."""
    records = []
    for proj, cost_mean in GCP_PROFILES.items():
        day_cost = RNG.lognormal(mean=0.0, sigma=0.25, size=len(DAYS)) * cost_mean

        if proj == GCP_SPIKE[0]:
            day_cost[GCP_SPIKE[1]] = cost_mean * SPIKE_MULTIPLIER

        for idx, day in enumerate(DAYS):
            records.append(
                {
                    "project_id": proj,
                    "day": day.date(),
                    "cost": round(float(day_cost[idx]), 2),
                    "currency": CURRENCY,
                }
            )
    return pd.DataFrame(records)


def generate_gcp_summary(details: pd.DataFrame) -> pd.DataFrame:
    """Per-project total cost over the reporting window, summed from *details*."""
    window = in_reporting_window(details)
    summary = (
        window.groupby("project_id", sort=False)["cost"]
        .sum()
        .round(2)
        .reset_index()
        .rename(columns={"cost": "total_cost"})
    )
    summary["currency"] = CURRENCY
    return summary


# ── Public in-memory data ──────────────────────────────────────────────
# These are imported by tools.py instead of reading real CSVs.

AZURE_DETAILS_DF = generate_azure_details()
GCP_DETAILS_DF = generate_gcp_details()
AZURE_SUMMARY_DF = generate_azure_summary(AZURE_DETAILS_DF)
GCP_SUMMARY_DF = generate_gcp_summary(GCP_DETAILS_DF)
