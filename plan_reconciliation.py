"""Read-only post-delivery reconciliation of an applied change plan against a fresh inventory.

    uv run python plan_reconciliation.py PLAN.json OBSERVED.csv
        [--source SOURCE.csv | --review-id ID] [--vocabulary VOCAB.json] [--output REPORT.json]

A reviewed plan and a successful delivery job do not show which label changes reached the estate.
After the team's own delivery process has applied a plan, export the inventory again and run this
check: for every planned tag it reports whether the observed value is the approved value
(``applied``), still the reviewed before-value (``not_applied``), something else
(``conflicting``), or cannot be observed because the resource is absent or listed under another
cloud (``unavailable``). Values are compared as the executable exports write them (GCP labels
lowercased), using the same plan checks and tag observation as the pre-delivery check in
``plan_validation``.

A snapshot shows values, not causes: the report never says why a change is missing or who set a
conflicting value. The check never changes tags, decisions, approvals, snapshots or plans, and it
never retries delivery.

Exit status: 0 every planned tag applied, 1 some tag not applied, conflicting or unavailable,
2 invalid plan or unreadable input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from plan_validation import (
    BEFORE,
    INTENDED,
    MISSING,
    PLAN_LABELS,
    _problem,
    _raw,
    _saved_review,
    observe_tags,
    plan_problems,
)
from review_context import review_context

REPORT_VERSION = 1

# Per-tag outcomes.
APPLIED, NOT_APPLIED, CONFLICTING, UNAVAILABLE = (
    "applied",
    "not_applied",
    "conflicting",
    "unavailable",
)
OUTCOMES = (APPLIED, NOT_APPLIED, CONFLICTING, UNAVAILABLE)
# Per-resource status adds the mix of applied and not-yet-applied tags.
PARTIALLY_APPLIED = "partially_applied"
RESOURCE_STATUSES = (APPLIED, PARTIALLY_APPLIED, NOT_APPLIED, CONFLICTING, UNAVAILABLE)

_STATE_OUTCOME = {INTENDED: APPLIED, BEFORE: NOT_APPLIED, MISSING: UNAVAILABLE}

LIMITATIONS = (
    "This report compares values in one inventory snapshot with the approved plan. It does not "
    "show why a change is absent, whether delivery is still in progress, or who set a "
    "conflicting value; check the delivery system's own logs and change history. obs-agent "
    "changed nothing and did not retry delivery."
)


def plan_digest(plan) -> str:
    """Content identity of a plan: SHA-256 of its canonical JSON (key order and spacing ignored)."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _reason(outcome: str, obs: dict, cloud_mismatch: bool) -> str:
    label, new, previous = obs["label"], obs["new"], obs["previous"]
    if outcome == APPLIED:
        return f"{label} holds the approved value {new!r}."
    if outcome == NOT_APPLIED:
        return (
            f"{label} still holds the reviewed before-value {previous!r}, not the approved "
            f"{new!r}. The snapshot does not show whether delivery skipped, failed, reverted or "
            "has not yet finished this change."
        )
    if outcome == CONFLICTING:
        return (
            f"{label} holds {obs['observed']!r}, which is neither the reviewed before-value "
            f"{previous!r} nor the approved {new!r}. The snapshot does not show who set this "
            "value or when."
        )
    if cloud_mismatch:
        return (
            f"The observed inventory lists {obs['resource_id']} under {obs['observed_cloud']}, "
            f"not {obs['cloud']}, so the approved {label} change cannot be confirmed. It may be "
            "a different resource with the same ID or an export error."
        )
    return (
        f"{obs['resource_id']} is not in the observed inventory, so the approved {label} change "
        "cannot be confirmed. It may have been deleted, renamed or left out of the export."
    )


def _resource_status(outcomes: list[str]) -> str:
    if UNAVAILABLE in outcomes:
        return UNAVAILABLE
    if CONFLICTING in outcomes:
        return CONFLICTING
    if all(o == APPLIED for o in outcomes):
        return APPLIED
    if all(o == NOT_APPLIED for o in outcomes):
        return NOT_APPLIED
    return PARTIALLY_APPLIED


def _source_errors(plan: dict, source: pd.DataFrame) -> list[dict]:
    """The supplied reviewed snapshot must be the plan's, with the plan's before-values."""
    if review_context(source)["inventory_id"] != plan["inventory_id"]:
        return [
            _problem(
                "source_snapshot_mismatch",
                "The supplied source snapshot is not the inventory this plan was reviewed on "
                "(different inventory_id). Supply the saved snapshot that produced the plan.",
            )
        ]
    return [
        _problem(
            "plan_inconsistent_with_source",
            f"{obs['resource_id']} {obs['label']}: the plan records previous "
            f"{obs['previous']!r} but the reviewed snapshot holds {obs['observed']!r}. The plan "
            "was edited or produced elsewhere.",
            resource_id=obs["resource_id"],
            label=obs["label"],
        )
        for obs in observe_tags(plan, source)
        if obs["state"] != BEFORE
    ]


def _unplanned_changes(change: dict, source_row, observed_row) -> list[dict]:
    """Tags on a planned resource that the plan did not touch but that differ from review."""
    out = []
    for label in PLAN_LABELS:
        if label in change["tags"]:
            continue
        reviewed, observed = _raw(source_row[label]), _raw(observed_row[label])
        if reviewed != observed:
            out.append(
                {
                    "label": label,
                    "reviewed": reviewed,
                    "observed": observed,
                    "reason": f"{label} was not part of this plan but changed from {reviewed!r} "
                    f"to {observed!r} since review. The snapshot does not show whether the "
                    "delivery or someone else made this change.",
                }
            )
    return out


def reconcile_plan(
    plan,
    observed: pd.DataFrame,
    *,
    source: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict:
    """Machine-readable reconciliation of *plan* against the *observed* post-delivery inventory.

    ``source`` is the saved snapshot the plan was reviewed on. With it, the report also checks the
    plan belongs to that snapshot and lists unplanned changes to the planned resources' other
    tags. ``now`` fixes the report timestamp (default: the current UTC time). Given the same
    inputs and ``now`` the report is identical. Nothing passed in is modified.
    """
    now = now or datetime.now(UTC)
    report = {
        "report_version": REPORT_VERSION,
        "check": "post_delivery_reconciliation",
        "reconciled_at": now.astimezone(UTC).isoformat(timespec="seconds"),
        "status": "invalid",
        "plan": None,
        "observed_inventory": None,
        "source_snapshot": None,
        "errors": plan_problems(plan),
        "resources": [],
        "summary": {},
        "limitations": LIMITATIONS,
    }
    try:
        report["plan"] = {"sha256": plan_digest(plan)}
    except (TypeError, ValueError):
        pass
    if not report["errors"] and source is not None:
        report["errors"] = _source_errors(plan, source)
    if report["errors"]:
        report["summary"] = {"errors": len(report["errors"])}
        return report

    observed_context = review_context(observed)
    report["plan"] |= {
        "inventory_id": plan["inventory_id"],
        "model_version": plan["model_version"],
        "generated_at": plan["generated_at"],
        "resources": len(plan["changes"]),
        "tags": sum(len(c["tags"]) for c in plan["changes"]),
        "approvals": len(plan["approvals"]),
        "reviewers": sorted({a["reviewer"] for a in plan["approvals"]}),
    }
    report["observed_inventory"] = {
        "inventory_id": observed_context["inventory_id"],
        "resources": len(observed),
        "same_as_reviewed": observed_context["inventory_id"] == plan["inventory_id"],
    }
    if source is not None:
        report["source_snapshot"] = {"inventory_id": plan["inventory_id"], "matches_plan": True}

    observations = observe_tags(plan, observed)
    observed_rows = observed.set_index("resource_id", drop=False)
    source_rows = None if source is None else source.set_index("resource_id", drop=False)
    for change in plan["changes"]:
        rid, cloud = change["resource_id"], change["cloud"]
        obs = [o for o in observations if o["resource_id"] == rid]
        present = obs[0]["state"] != MISSING
        cloud_mismatch = present and obs[0]["observed_cloud"] != cloud
        tags = {}
        for o in obs:
            outcome = UNAVAILABLE if cloud_mismatch else _STATE_OUTCOME.get(o["state"], CONFLICTING)
            tags[o["label"]] = {
                "previous": o["previous"],
                "new": o["new"],
                "observed": o["observed"],
                "outcome": outcome,
                "reason": _reason(outcome, o, cloud_mismatch),
            }
        resource = {
            "resource_id": rid,
            "cloud": cloud,
            "observed_cloud": obs[0]["observed_cloud"],
            "status": _resource_status([t["outcome"] for t in tags.values()]),
            "tags": tags,
        }
        if source_rows is not None:
            resource["unplanned_changes"] = (
                _unplanned_changes(change, source_rows.loc[rid], observed_rows.loc[rid])
                if present and not cloud_mismatch
                else []
            )
        report["resources"].append(resource)

    outcomes = [t["outcome"] for r in report["resources"] for t in r["tags"].values()]
    report["status"] = "complete" if all(o == APPLIED for o in outcomes) else "incomplete"
    report["summary"] = {
        "resources": len(report["resources"]),
        "tags": len(outcomes),
        "tag_outcomes": {o: outcomes.count(o) for o in OUTCOMES},
        "resource_status": {
            s: sum(r["status"] == s for r in report["resources"]) for s in RESOURCE_STATUSES
        },
        "unplanned_changes": (
            None
            if source is None
            else sum(len(r["unplanned_changes"]) for r in report["resources"])
        ),
    }
    return report


def summary_text(report: dict) -> str:
    """A short human-readable version of the report."""
    lines = [f"Post-delivery reconciliation: {report['status'].upper()}"]
    for e in report["errors"]:
        lines.append(f"  invalid plan [{e['code']}]: {e['message']}")
    if report["status"] != "invalid":
        s = report["summary"]
        counts = ", ".join(f"{n} {o.replace('_', ' ')}" for o, n in s["tag_outcomes"].items())
        lines.append(f"  {s['tags']} planned tag(s) on {s['resources']} resource(s): {counts}.")
        for r in report["resources"]:
            for label, t in r["tags"].items():
                if t["outcome"] != APPLIED:
                    lines.append(f"  {r['resource_id']} {label} [{t['outcome']}]: {t['reason']}")
            for u in r.get("unplanned_changes", []):
                lines.append(f"  {r['resource_id']} {u['label']} [unplanned_change]: {u['reason']}")
        if s["unplanned_changes"] is None:
            lines.append(
                "  Unplanned changes to other tags were not checked; pass --review-id or --source."
            )
    lines.append(f"  {report['limitations']}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    parser.add_argument("plan", type=Path, help="approved change plan JSON that was delivered")
    parser.add_argument("observed", type=Path, help="inventory CSV exported after delivery")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--source", type=Path, help="the inventory CSV the plan was reviewed on")
    group.add_argument("--review-id", help="saved review the plan was exported from")
    parser.add_argument(
        "--vocabulary",
        type=Path,
        help="vocabulary JSON for the CSVs (default: the saved review's, else the built-in one)",
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here, not stdout")
    args = parser.parse_args(argv)

    from inventory_import import InventoryImportError, load_inventory_csv
    from label_config import LabelConfigError, load_label_config

    try:
        plan_bytes, observed_bytes = args.plan.read_bytes(), args.observed.read_bytes()
        plan = json.loads(plan_bytes)
        config = load_label_config(args.vocabulary) if args.vocabulary else None
        source = None
        if args.review_id:
            csv, saved_config = _saved_review(args.review_id)
            config = config or saved_config
            source = load_inventory_csv(csv, saved_config)
        elif args.source:
            source = load_inventory_csv(args.source, config)
        observed = load_inventory_csv(observed_bytes, config)
    except (
        OSError,
        ValueError,
        sqlite3.Error,
        InventoryImportError,
        LabelConfigError,
    ) as exc:
        print(f"Post-delivery reconciliation: INVALID INPUT\n  {exc}", file=sys.stderr)
        return 2

    report = reconcile_plan(plan, observed, source=source)
    report["inputs"] = {
        "plan_file": {"name": args.plan.name, "sha256": _sha256(plan_bytes)},
        "observed_file": {"name": args.observed.name, "sha256": _sha256(observed_bytes)},
        "review_id": args.review_id,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    print(summary_text(report), file=sys.stderr)
    return {"complete": 0, "incomplete": 1}.get(report["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
