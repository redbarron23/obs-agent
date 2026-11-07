"""Export a reproducible fictional estate and separate evaluation truth; no cloud/LLM calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import inventory
from inventory_import import COLUMN_ORDER, load_inventory_csv
from label_config import default_config
from labeling import normalise

EXPECTATIONS = {
    "background": "Review evidence; the base profile can contain noisy or missing signals.",
    "missing_labels": "Use supporting evidence for suggestions; human review required.",
    "shared_ownership": "Abstain on a single team; investigate shared ownership.",
    "unknown_ownership": "Abstain on team despite the simulator knowing its hidden owner.",
    "ownership_change": "After the change, inspect the stale tag/hierarchy against updated owners.",
}


def _csv(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _inventory_csv(frame):
    public = frame.loc[:, list(COLUMN_ORDER)].copy()
    for column in ("owners", "top_services"):
        public[column] = public[column].map(json.dumps)
    return _csv(public)


def _remove_singletons(frame, vocabulary):
    """Keep importable training support without inventing extra labelled examples."""
    removed = {}
    for label in ("team", "env"):
        normalized = frame[label].map(lambda value, name=label: normalise(name, value, vocabulary))
        counts = normalized.value_counts()
        singletons = counts[counts == 1].index
        mask = normalized.isin(singletons)
        removed[label] = frame.loc[mask, "resource_id"].tolist()
        frame.loc[mask, label] = None
        if label == "team":
            frame.loc[mask, "cost_center"] = None
    return removed


def build_pack(*, rows=320, seed=7, profile="harsh"):
    if not 100 <= rows <= 10_000:
        raise ValueError("rows must be between 100 and 10000")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if profile not in inventory.HARDNESS:
        raise ValueError("profile must be clean, harsh or reorg")
    before = inventory.make_estate(hardness=profile, seed=seed, n=rows)
    rng = np.random.default_rng(seed)
    count = max(1, rows // 10)
    selected = rng.permutation(rows)[: count * 4].reshape(4, count)
    cases = pd.Series("background", index=before.index)
    for name, indices in zip(list(EXPECTATIONS)[1:], selected, strict=True):
        cases.loc[indices] = name
    for index in selected[0]:
        before.loc[index, ["team", "env", "cost_center"]] = None
    for name, indices in (("shared_ownership", selected[1]), ("unknown_ownership", selected[2])):
        for index in indices:
            cloud = before.at[index, "cloud"]
            # Surrogates reveal neither the hidden team nor the scenario name.
            before.at[index, "resource_id"] = f"estate-{seed}-{cloud}-{index:06d}"
            before.at[index, "parent"] = inventory.SHARED_PARENTS[cloud][0]
            before.at[index, "team"] = None
            before.at[index, "cost_center"] = None
            before.at[index, "top_services"] = (
                ["KeyVault"] if cloud == "azure" else ["CloudLogging"]
            )
            if name == "shared_ownership":
                teams = rng.choice(inventory.TEAM_NAMES, size=2, replace=False)
                before.at[index, "owners"] = [inventory.TEAMS[t]["owners"][0] for t in teams]
                before.at[index, "created_by"] = before.at[index, "owners"][0]
                before.at[index, "true_team"] = None
                before.at[index, "true_cost_center"] = None
            else:
                before.at[index, "owners"] = []
                before.at[index, "created_by"] = None
    # IDs must also remain unique after replacing opaque scenario names.
    if not before.resource_id.is_unique:
        raise ValueError("generated resource identifiers collide; choose another seed")
    after = before.copy(deep=True)
    changes = []
    for index in selected[3]:
        old = before.at[index, "true_team"]
        new = inventory.TEAM_NAMES[
            (inventory.TEAM_NAMES.index(old) + 1) % len(inventory.TEAM_NAMES)
        ]
        team = inventory.TEAMS[new]
        # The old tags, name, services and hierarchy intentionally remain stale.
        before.at[index, "team"] = old
        before.at[index, "cost_center"] = inventory.TEAMS[old]["cost_center"]
        after.at[index, "team"] = old
        after.at[index, "cost_center"] = inventory.TEAMS[old]["cost_center"]
        after.at[index, "true_team"] = new
        after.at[index, "true_cost_center"] = team["cost_center"]
        after.at[index, "owners"] = list(team["owners"][:2])
        after.at[index, "created_by"] = team["owners"][0]
        changes.append(before.at[index, "resource_id"])
    vocabulary = default_config()
    files = {"vocabulary.json": (json.dumps(vocabulary, indent=2, sort_keys=True) + "\n").encode()}
    truth = []
    withheld = {}
    for snapshot, frame in (("before", before), ("after", after)):
        withheld[snapshot] = _remove_singletons(frame, vocabulary)
        public = _inventory_csv(frame)
        # Use the production import boundary before writing any files.
        load_inventory_csv(public, config=vocabulary)
        files[f"inventory-{snapshot}.csv"] = public
        gold = frame[["resource_id", "true_team", "true_env", "true_cost_center"]].copy()
        gold.insert(0, "snapshot", snapshot)
        gold["scenario"] = cases
        gold["expected_team_action"] = cases.map(
            {
                "background": "review_evidence",
                "missing_labels": "review_evidence",
                "shared_ownership": "abstain",
                "unknown_ownership": "abstain",
                "ownership_change": "review_evidence"
                if snapshot == "before"
                else "investigate_change",
            }
        )
        truth.append(gold)
    files["truth.csv"] = _csv(pd.concat(truth, ignore_index=True))
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "seed": seed,
        "rows_per_snapshot": rows,
        "profile": profile,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
        "fictional_entities": {
            "teams": {
                name: {
                    key: settings[key]
                    for key in ("cost_center", "owners", "azure_parent", "gcp_parent")
                }
                for name, settings in inventory.TEAMS.items()
            },
            "environments": inventory.ENVS,
        },
        "baseline_date": inventory.END_DATE.date().isoformat(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inventory_generator_sha256": hashlib.sha256(
            Path(inventory.__file__).read_bytes()
        ).hexdigest(),
        "scenarios": {
            name: {"rows": int((cases == name).sum()), "expectation": text}
            for name, text in EXPECTATIONS.items()
        },
        "changed_ownership_resources": changes,
        "singleton_observations_withheld": withheld,
        "limitations": [
            "Fictional data; production distributions and user benefit remain unvalidated.",
            "Snapshots change ownership evidence/truth and support withholding; "
            "not billing history.",
            "Shared ownership has no single true team; blank truth is not a negative class.",
            "Unknown ownership has hidden truth but no ownership evidence; expect abstention.",
            "Singleton observations are blanked to satisfy importer support requirements.",
            "Seeds share generator logic; they are not independent "
            "organisation or real-world validation.",
            "Never give truth.csv or this scenario manifest to inference as resource evidence.",
        ],
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in files.items()
        },
    }
    files["scenario_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New directory; existing paths are never overwritten",
    )
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=inventory.HARDNESS, default="harsh")
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("output path already exists; choose a new directory")
        files = build_pack(rows=args.rows, seed=args.seed, profile=args.profile)
        args.out.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            with (args.out / name).open("xb") as target:
                target.write(content)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Synthetic pack not completed: {exc}\n")
    print(
        f"Created {args.out}: two fictional snapshots of {args.rows} resources; truth is separate."
    )


if __name__ == "__main__":
    main()
