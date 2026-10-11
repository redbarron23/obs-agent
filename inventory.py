"""
Synthetic cloud inventory with partially missing / messy labels.

Simulates the real-world tagging problem: a few hundred Azure
subscriptions and GCP projects, where only about half carry a usable
`team` / `env` / `cost_center` label and some of those are wrong.

Every row keeps its *true* labels (true_team, true_env, true_cost_center)
next to the *observed* ones (team, env, cost_center).  Real billing exports
only have the observed columns; the true ones exist here so label
inference can be scored against ground truth.

Uses its own seeded RNG so it never shifts the numbers in data.py.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(seed=7)

N_RESOURCES = 320

# ── Teams ──────────────────────────────────────────────────────────────
# Each team has a home in the hierarchy, a pool of owners, a naming
# vocabulary and a service footprint.  Sizes are deliberately uneven.
TEAMS = {
    "payments": {
        "weight": 0.20, "cost_center": "CC-1001",
        "azure_parent": "mg-payments", "gcp_parent": "folders/payments",
        "owners": ["alice", "bob", "chen", "dana"],
        "prefixes": ["pay", "payments", "checkout", "billing"],
        "components": ["api", "ledger", "gateway", "worker", "db"],
        "services": ["AKS", "SQL", "KeyVault", "GKE", "CloudSQL", "PubSub"],
    },
    "data-platform": {
        "weight": 0.18, "cost_center": "CC-2002",
        "azure_parent": "mg-data", "gcp_parent": "folders/data-platform",
        "owners": ["erik", "fatima", "gus", "hana"],
        "prefixes": ["data", "dp", "lake", "etl"],
        "components": ["ingest", "warehouse", "airflow", "stream", "catalog"],
        "services": ["Synapse", "DataFactory", "Storage", "BigQuery", "Dataflow", "GCS"],
    },
    "ml": {
        "weight": 0.14, "cost_center": "CC-2003",
        "azure_parent": "mg-data", "gcp_parent": "folders/ml",
        "owners": ["ivan", "jia", "fatima"],
        "prefixes": ["ml", "ai", "models"],
        "components": ["training", "inference", "features", "notebooks"],
        "services": ["AzureML", "GPU-VM", "Storage", "VertexAI", "GCS", "BigQuery"],
    },
    "web": {
        "weight": 0.16, "cost_center": "CC-3001",
        "azure_parent": "mg-digital", "gcp_parent": "folders/digital",
        "owners": ["kofi", "lena", "marco", "nina"],
        "prefixes": ["web", "www", "shop", "frontend"],
        "components": ["app", "cdn", "search", "cms", "bff"],
        "services": ["AppService", "FrontDoor", "Redis", "CloudRun", "CloudCDN", "Memorystore"],
    },
    "identity": {
        "weight": 0.10, "cost_center": "CC-4001",
        "azure_parent": "mg-platform", "gcp_parent": "folders/platform",
        "owners": ["omar", "priya"],
        "prefixes": ["idp", "auth", "iam", "sso"],
        "components": ["login", "tokens", "directory", "mfa"],
        "services": ["AKS", "KeyVault", "SQL", "GKE", "SecretManager", "CloudSQL"],
    },
    "observability": {
        "weight": 0.10, "cost_center": "CC-4002",
        "azure_parent": "mg-platform", "gcp_parent": "folders/platform",
        "owners": ["quinn", "raj", "omar"],
        "prefixes": ["obs", "logging", "metrics", "monitor"],
        "components": ["collector", "grafana", "loki", "alerts"],
        "services": ["LogAnalytics", "Monitor", "Storage", "CloudLogging", "CloudMonitoring", "GCS"],
    },
    "security": {
        "weight": 0.07, "cost_center": "CC-4003",
        "azure_parent": "mg-security", "gcp_parent": "folders/security",
        "owners": ["sara", "tom"],
        "prefixes": ["sec", "soc", "siem"],
        "components": ["scanner", "sentinel", "vault", "audit"],
        "services": ["Sentinel", "LogAnalytics", "Defender", "SCC", "CloudLogging", "SecretManager"],
    },
    "finance-apps": {
        "weight": 0.05, "cost_center": "CC-5001",
        "azure_parent": "mg-corporate", "gcp_parent": "folders/corporate",
        "owners": ["uma", "victor"],
        "prefixes": ["fin", "erp", "reporting"],
        "components": ["sap", "reports", "budget"],
        "services": ["VM", "SQL", "PowerBI", "ComputeEngine", "BigQuery", "CloudSQL"],
    },
}

TEAM_NAMES = list(TEAMS)

# Name parts with no team signal, used by everyone.
SHARED_PREFIXES = ["core", "platform", "api", "svc", "cloud"]
SHARED_COMPONENTS = ["backend", "service", "jobs", "infra"]

# Catch-all parents that carry no team signal (the "legacy" mess).
SHARED_PARENTS = {"azure": ["mg-shared", "mg-legacy"], "gcp": ["folders/shared", "folders/legacy"]}

# ── Environments ───────────────────────────────────────────────────────
ENVS = ["prod", "staging", "dev", "sandbox"]
ENV_WEIGHTS = [0.35, 0.20, 0.30, 0.15]
ENV_NAME_TOKENS = {
    "prod": ["prod", "prd", "p"],
    "staging": ["stg", "staging", "uat"],
    "dev": ["dev", "d", "test"],
    "sandbox": ["sbx", "sandbox", "poc"],
}
# How each env spends money: (mean daily $, weekend multiplier, noise sd)
ENV_COST_SHAPE = {
    "prod": (120.0, 0.95, 0.10),
    "staging": (40.0, 0.60, 0.20),
    "dev": (25.0, 0.25, 0.30),
    "sandbox": (6.0, 0.15, 0.80),
}

REGIONS = {"azure": ["westeurope", "northeurope", "germanywestcentral", "eastus"],
           "gcp": ["europe-west1", "europe-west3", "europe-west4", "us-central1"]}
REGION_SHORT = {"westeurope": "weu", "northeurope": "neu", "germanywestcentral": "gwc",
                "eastus": "eus", "europe-west1": "ew1", "europe-west3": "ew3",
                "europe-west4": "ew4", "us-central1": "uc1"}

# Spellings people actually type, used to make observed labels messy.
TEAM_VARIANTS = {
    "payments": ["Payments", "team-payments", "PAYMENTS", "payment"],
    "data-platform": ["Data Platform", "data_platform", "dataplatform", "DP"],
    "ml": ["ML", "machine-learning", "team-ml"],
    "web": ["Web", "web-team", "digital-web"],
    "identity": ["Identity", "IAM", "team-identity"],
    "observability": ["Observability", "o11y", "obs"],
    "security": ["Security", "SecOps", "sec"],
    "finance-apps": ["Finance Apps", "finance", "fin-apps"],
}
ENV_VARIANTS = {
    "prod": ["Prod", "production", "prd", "PROD"],
    "staging": ["stg", "Staging", "uat"],
    "dev": ["Dev", "development", "DEV"],
    "sandbox": ["sbx", "Sandbox", "poc"],
}

END_DATE = datetime(2026, 6, 1)
N_DAYS = 60


def _pick(seq):
    return seq[RNG.integers(len(seq))]


def _resource_name(cloud: str, team: dict, env: str, region: str) -> str:
    """Name built from team vocabulary — or an opaque one, as in real estates."""
    roll = RNG.random()
    if roll < 0.15:
        # Opaque: GUID-ish subscription or auto-generated project id
        if cloud == "azure":
            return f"sub-{RNG.integers(16**8):08x}"
        return f"proj-{RNG.integers(10**6):06d}"
    if roll < 0.22:
        # Generic name with no team vocabulary
        return f"{_pick(['legacy', 'shared', 'app', 'migration'])}-{RNG.integers(1, 40)}"
    if env == "sandbox" and RNG.random() < 0.6:
        return f"sbx-{_pick(team['owners'])}-{RNG.integers(1, 9)}"
    # A third of names start with a prefix every team uses ("core-", "api-").
    prefix = _pick(SHARED_PREFIXES) if RNG.random() < 0.33 else _pick(team["prefixes"])
    component = _pick(team["components"]) if RNG.random() < 0.75 else _pick(SHARED_COMPONENTS)
    parts = [prefix, component]
    # Not everyone puts the env in the name.
    if RNG.random() < 0.75:
        parts.append(_pick(ENV_NAME_TOKENS[env]))
    if RNG.random() < 0.5:
        parts.append(REGION_SHORT[region])
    return "-".join(parts)


def _cost_stats(env: str) -> dict:
    """60 days of spend shaped by env, reduced to the features a model sees."""
    mean, weekend_mult, noise = ENV_COST_SHAPE[env]
    scale = mean * RNG.lognormal(0, 0.6)
    days = pd.date_range(END_DATE - timedelta(days=N_DAYS - 1), END_DATE, freq="D")
    weekend = days.dayofweek >= 5
    cost = scale * np.where(weekend, weekend_mult, 1.0)
    cost = (cost * RNG.lognormal(0, noise, size=len(days))).clip(0.01)
    weekday_mean = cost[~weekend].mean()
    return {
        "avg_daily_cost": round(float(cost.mean()), 2),
        "weekend_ratio": round(float(cost[weekend].mean() / weekday_mean), 3),
        "cost_cv": round(float(cost.std() / cost.mean()), 3),
    }


def _observed(true_value: str, variants: dict, choices: list[str],
              p_missing: float, p_variant: float, p_wrong: float):
    """What the tag actually says: missing, misspelt, wrong, or correct."""
    roll = RNG.random()
    if roll < p_missing:
        return None
    roll = RNG.random()
    if roll < p_wrong:
        return _pick([c for c in choices if c != true_value])
    if roll < p_wrong + p_variant:
        return _pick(variants[true_value])
    return true_value


def generate_inventory(n: int = N_RESOURCES) -> pd.DataFrame:
    weights = np.array([t["weight"] for t in TEAMS.values()])
    weights = weights / weights.sum()
    rows = []
    seen_ids: set[str] = set()

    for _ in range(n):
        team_name = TEAM_NAMES[RNG.choice(len(TEAM_NAMES), p=weights)]
        team = TEAMS[team_name]
        env = ENVS[RNG.choice(len(ENVS), p=ENV_WEIGHTS)]
        cloud = "azure" if RNG.random() < 0.5 else "gcp"
        region = _pick(REGIONS[cloud])

        name = _resource_name(cloud, team, env, region)
        while name in seen_ids:  # keep resource_id unique
            name = f"{name}-{RNG.integers(2, 99)}"
        seen_ids.add(name)

        # Most sit under the team's own parent; reorgs leave some under another
        # team's parent, and older ones in a catch-all.
        parent_key = "azure_parent" if cloud == "azure" else "gcp_parent"
        roll = RNG.random()
        if roll < 0.65:
            parent = team[parent_key]
        elif roll < 0.80:
            parent = TEAMS[_pick([t for t in TEAM_NAMES if t != team_name])][parent_key]
        else:
            parent = _pick(SHARED_PARENTS[cloud])

        # Owners mostly from the team, sometimes a borrowed engineer.
        k = int(RNG.integers(1, 4))
        owners = list(RNG.choice(team["owners"], size=min(k, len(team["owners"])), replace=False))
        if RNG.random() < 0.30:
            other = TEAMS[_pick([t for t in TEAM_NAMES if t != team_name])]
            owners.append(_pick(other["owners"]))

        # Services: cloud-appropriate half of the team profile + one random.
        own = team["services"][:3] if cloud == "azure" else team["services"][3:]
        services = sorted(set(RNG.choice(own, size=2, replace=False)) | {
            _pick(_pick(list(TEAMS.values()))["services"][:3] if cloud == "azure"
                  else _pick(list(TEAMS.values()))["services"][3:])
        })

        created_at = END_DATE - timedelta(days=int(RNG.integers(30, 365 * 6)))
        age_years = (END_DATE - created_at).days / 365

        # Older resources are more likely untagged (tagging policy came late),
        # so the labelled half is NOT a random sample of the estate.
        p_missing = min(0.85, 0.20 + 0.11 * age_years)

        observed_team = _observed(team_name, TEAM_VARIANTS, TEAM_NAMES, p_missing, 0.12, 0.05)
        # cost_center is tagged together with team (same tagging moment).
        if observed_team is None:
            cost_center = None
        else:
            cost_center = TEAMS.get(observed_team, team)["cost_center"]

        rows.append({
            "resource_id": name,
            "cloud": cloud,
            "parent": parent,
            "billing_account": "ba-azure-01" if cloud == "azure" else "ba-gcp-01",
            "owners": sorted(owners),
            "created_by": owners[0],
            "created_at": created_at.date(),
            "region": region,
            "top_services": services,
            **_cost_stats(env),
            "true_team": team_name,
            "true_env": env,
            "true_cost_center": team["cost_center"],
            "team": observed_team,
            "env": _observed(env, ENV_VARIANTS, ENVS, p_missing, 0.12, 0.05),
            "cost_center": cost_center,
        })

    return pd.DataFrame(rows)


INVENTORY_DF = generate_inventory()

# Columns a real billing/inventory export would have (no ground truth).
OBSERVED_COLUMNS = [c for c in INVENTORY_DF.columns if not c.startswith("true_")]
