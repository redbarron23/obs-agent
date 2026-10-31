"""Read-only pre-delivery check of a reviewed change plan against a fresh inventory export.

    uv run python plan_validation.py PLAN.json FRESH.csv [--source SOURCE.csv | --review-id ID]
                                     [--vocabulary VOCAB.json] [--output REPORT.json]

Resources can change between review and delivery. This check compares the plan's recorded
before-values, approval evidence and inference version with a newly exported inventory (and,
optionally, the saved snapshot the plan was reviewed on) and says which approvals no longer
describe the estate. It never changes tags, approvals, decision databases or snapshots, and it
never carries an approval over to the fresh inventory: a failed check means "review again".

Exit status: 0 pass, 1 fail (renewed review needed), 2 invalid plan or unreadable input.

The building blocks are separate so post-delivery reconciliation (#65) can reuse them:
``plan_problems`` validates schema and approval provenance, and ``observe_tags`` classifies each
planned tag in an inventory as still at its before-value, at the intended value, at some other
value, or missing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from drift import REASONS, detect_drift
from review_context import review_context

REPORT_VERSION = 1
PLAN_SCHEMA_VERSION = 2
REVIEWED_HOW = ("accept", "override")
PLAN_LABELS = ("team", "env", "cost_center")
# What a reviewer approved for each exported tag; cost_center follows the team decision.
APPROVED_BY = {"team": "team", "env": "env", "cost_center": "team"}

# Observed states of one planned tag in an inventory (shared with post-delivery reconciliation).
BEFORE, INTENDED, OTHER, MISSING = "before", "intended", "other", "missing"


# ── Plan schema and approval provenance ───────────────────────────────


def _problem(code: str, message: str, **extra) -> dict:
    return {"code": code, "message": message} | extra


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def destination_value(cloud: str, value) -> str | None:
    """The value as an executable export writes it (GCP labels are lowercased)."""
    if value is None:
        return None
    value = str(value)
    return value.lower() if cloud == "gcp" else value


def plan_problems(plan) -> list[dict]:
    """Reasons the plan cannot be trusted as a reviewed approval artifact; empty means usable.

    Checks the schema, that every exported tag is covered by a recorded reviewer decision for the
    same value, and that each approval is bound to the plan's snapshot and inference version.
    """
    if not isinstance(plan, dict):
        return [_problem("malformed_plan", "The plan must be a JSON object.")]
    problems = []
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        problems.append(
            _problem(
                "unsupported_schema",
                f"schema_version must be {PLAN_SCHEMA_VERSION}; "
                f"found {plan.get('schema_version')!r}. Export the plan again from obs-agent.",
            )
        )
    if plan.get("reviewed_only") is not True:
        problems.append(
            _problem(
                "not_reviewed_only",
                "reviewed_only must be true; convenience exports with unreviewed suggestions "
                "cannot be validated for delivery.",
            )
        )
    for field in ("inventory_id", "model_version", "generated_at"):
        if not _text(plan.get(field)):
            problems.append(_problem("missing_field", f"The plan has no {field}.", field=field))
    changes, approvals = plan.get("changes"), plan.get("approvals")
    if not isinstance(changes, list) or not isinstance(approvals, list):
        problems.append(_problem("malformed_plan", "changes and approvals must both be lists."))
        return problems
    if not changes:
        problems.append(_problem("empty_plan", "The plan contains no changes to deliver."))

    by_key: dict[tuple[str, str], dict] = {}
    for i, approval in enumerate(approvals):
        where = f"approvals[{i}]"
        if not isinstance(approval, dict):
            problems.append(_problem("malformed_approval", f"{where} must be an object."))
            continue
        missing = [
            f
            for f in ("resource_id", "label", "decision", "reviewer", "decided_at")
            if not _text(approval.get(f))
        ]
        if missing:
            problems.append(
                _problem(
                    "missing_provenance",
                    f"{where} lacks {', '.join(missing)}; every approval needs the resource, "
                    "label, decision, reviewer and decision time.",
                    resource_id=approval.get("resource_id"),
                )
            )
            continue
        for field in ("inventory_id", "model_version"):
            if approval.get(field) != plan.get(field):
                problems.append(
                    _problem(
                        "approval_context_mismatch",
                        f"{where} ({approval['resource_id']} {approval['label']}) was recorded "
                        f"against {field} {approval.get(field)!r}, not the plan's. Approvals "
                        "cannot be transferred between snapshots or inference versions.",
                        resource_id=approval["resource_id"],
                        label=approval["label"],
                    )
                )
        by_key[(approval["resource_id"], approval["label"])] = approval

    seen = set()
    for i, change in enumerate(changes):
        where = f"changes[{i}]"
        if (
            not isinstance(change, dict)
            or not _text(change.get("resource_id"))
            or change.get("cloud") not in ("azure", "gcp")
            or not isinstance(change.get("tags"), dict)
            or not change["tags"]
        ):
            problems.append(
                _problem(
                    "malformed_change",
                    f"{where} needs a resource_id, a cloud of azure or gcp, and tags.",
                )
            )
            continue
        rid, cloud = change["resource_id"], change["cloud"]
        if rid in seen:
            problems.append(
                _problem("malformed_change", f"{rid} appears more than once.", resource_id=rid)
            )
        seen.add(rid)
        for label, tag in change["tags"].items():
            if label not in PLAN_LABELS or not isinstance(tag, dict):
                problems.append(
                    _problem(
                        "malformed_change",
                        f"{rid}: unknown or malformed tag {label!r}.",
                        resource_id=rid,
                    )
                )
                continue
            if (
                set(tag) != {"previous", "new", "how"}
                or not _text(tag["new"])
                or not (tag["previous"] is None or isinstance(tag["previous"], str))
            ):
                problems.append(
                    _problem(
                        "malformed_change",
                        f"{rid} {label}: each tag needs previous (text or null), new and how.",
                        resource_id=rid,
                        label=label,
                    )
                )
                continue
            if tag["how"] not in REVIEWED_HOW:
                problems.append(
                    _problem(
                        "unreviewed_change",
                        f"{rid} {label} was resolved by {tag['how']!r}, not a reviewer decision.",
                        resource_id=rid,
                        label=label,
                    )
                )
                continue
            approval = by_key.get((rid, APPROVED_BY[label]))
            if approval is None:
                problems.append(
                    _problem(
                        "missing_approval",
                        f"{rid} {label} has no approval evidence for the "
                        f"{APPROVED_BY[label]} decision.",
                        resource_id=rid,
                        label=label,
                    )
                )
            elif approval["decision"] != tag["how"]:
                problems.append(
                    _problem(
                        "approval_mismatch",
                        f"{rid} {label}: the plan says {tag['how']!r} but the approval records "
                        f"{approval['decision']!r}.",
                        resource_id=rid,
                        label=label,
                    )
                )
            elif (
                label != "cost_center"
                and destination_value(cloud, approval.get("value")) != tag["new"]
            ):
                problems.append(
                    _problem(
                        "approval_mismatch",
                        f"{rid} {label}: the plan would write {tag['new']!r} but the reviewer "
                        f"approved {approval.get('value')!r}.",
                        resource_id=rid,
                        label=label,
                    )
                )
    return problems


# ── Observing planned tags in an inventory ────────────────────────────


def _raw(value) -> str | None:
    """A tag cell as the plan records it (label_review uses the same conversion)."""
    return None if pd.isna(value) else str(value)


def observe_tags(plan: dict, inventory: pd.DataFrame) -> list[dict]:
    """Each planned tag's value in *inventory*, classified against the plan.

    ``state`` is ``before`` (still the recorded previous value), ``intended`` (already the planned
    value, compared as an executable export writes it), ``other`` (neither) or ``missing`` (the
    resource is absent). Expects a plan that passed ``plan_problems``.
    """
    rows = inventory.set_index("resource_id", drop=False)
    out = []
    for change in plan["changes"]:
        rid, cloud = change["resource_id"], change["cloud"]
        row = rows.loc[rid] if rid in rows.index else None
        for label, tag in change["tags"].items():
            item = {
                "resource_id": rid,
                "cloud": cloud,
                "label": label,
                "previous": tag["previous"],
                "new": tag["new"],
            }
            if row is None:
                out.append(item | {"observed": None, "observed_cloud": None, "state": MISSING})
                continue
            observed = _raw(row[label])
            if observed == tag["previous"]:
                state = BEFORE
            elif destination_value(cloud, observed) == tag["new"]:
                state = INTENDED
            else:
                state = OTHER
            out.append(
                item | {"observed": observed, "observed_cloud": row["cloud"], "state": state}
            )
    return out


# ── Pre-delivery validation ───────────────────────────────────────────


def _renew(resource_id: str) -> str:
    return f" Review {resource_id} again on a fresh snapshot before delivery."


def _tag_finding(obs: dict) -> dict | None:
    rid, label = obs["resource_id"], obs["label"]
    detail = {
        "label": label,
        "previous": obs["previous"],
        "new": obs["new"],
        "current": obs["observed"],
    }
    if obs["state"] == INTENDED:
        return detail | {
            "code": "already_applied",
            "reason": f"{label} already carries the planned value {obs['new']!r}, so it was "
            "changed outside this plan after review. Confirm who applied it rather than "
            "delivering the plan again." + _renew(rid),
        }
    if obs["state"] == OTHER:
        return detail | {
            "code": "previous_value_changed",
            "reason": f"{label} was {obs['previous']!r} when reviewed but is now "
            f"{obs['observed']!r}; the approval no longer describes the resource and delivering "
            "it would overwrite a change made since review." + _renew(rid),
        }
    return None


def _evidence_findings(source: pd.DataFrame, fresh: pd.DataFrame, ids: set[str]) -> dict:
    found = {}
    for item in detect_drift(source, fresh):
        if item["resource_id"] in ids:
            found[item["resource_id"]] = [
                {
                    "code": "evidence_changed",
                    "field": c["field"],
                    "previous": c["previous"],
                    "current": c["current"],
                    "reason": REASONS[c["field"]]
                    + " The reviewer approved the earlier evidence."
                    + _renew(item["resource_id"]),
                }
                for c in item["changes"]
            ]
    return found


def validate_plan(plan, fresh: pd.DataFrame, *, source: pd.DataFrame | None = None) -> dict:
    """Machine-readable pre-delivery report for *plan* against the *fresh* inventory.

    ``source`` is the saved snapshot the plan was reviewed on. With it, the check also confirms
    the plan belongs to that snapshot and flags ownership or cost-shape drift on planned
    resources; without it, only the recorded tag before-values can be compared. Nothing passed
    in is modified.
    """
    report = {
        "report_version": REPORT_VERSION,
        "check": "pre_delivery",
        "status": "invalid",
        "renewed_review_required": True,
        "plan": None,
        "fresh_inventory": None,
        "source_snapshot": None,
        "errors": plan_problems(plan),
        "context": [],
        "warnings": [],
        "resources": [],
        "summary": {},
    }
    if report["errors"]:
        report["summary"] = {"errors": len(report["errors"])}
        return report

    fresh_context = review_context(fresh)
    report["plan"] = {
        "inventory_id": plan["inventory_id"],
        "model_version": plan["model_version"],
        "generated_at": plan["generated_at"],
        "resources": len(plan["changes"]),
        "tags": sum(len(c["tags"]) for c in plan["changes"]),
        "approvals": len(plan["approvals"]),
        "reviewers": sorted({a["reviewer"] for a in plan["approvals"]}),
    }
    report["fresh_inventory"] = fresh_context | {
        "resources": len(fresh),
        "same_as_reviewed": fresh_context["inventory_id"] == plan["inventory_id"],
    }

    if source is not None:
        source_context = review_context(source)
        report["source_snapshot"] = source_context | {
            "matches_plan": source_context["inventory_id"] == plan["inventory_id"],
        }
        if source_context["inventory_id"] != plan["inventory_id"]:
            report["context"].append(
                _problem(
                    "source_snapshot_mismatch",
                    "The supplied source snapshot is not the inventory this plan was reviewed "
                    "on (different inventory_id). Supply the saved snapshot that produced the "
                    "plan, or review the fresh inventory and export a new plan.",
                )
            )
        else:
            for obs in observe_tags(plan, source):
                if obs["state"] != BEFORE:
                    report["context"].append(
                        _problem(
                            "plan_inconsistent_with_source",
                            f"{obs['resource_id']} {obs['label']}: the plan records previous "
                            f"{obs['previous']!r} but the reviewed snapshot holds "
                            f"{obs['observed']!r}. The plan was edited or produced elsewhere; "
                            "export it again from the saved review.",
                            resource_id=obs["resource_id"],
                            label=obs["label"],
                        )
                    )
        if source_context["model_version"] != plan["model_version"]:
            report["context"].append(
                _problem(
                    "inference_version_changed",
                    "The inference implementation, settings or dependencies have changed since "
                    "the plan was reviewed (the source snapshot no longer yields the plan's "
                    "model_version). Review the resources again with the current version.",
                )
            )
    if fresh_context["model_version"] != plan["model_version"] and not any(
        c["code"] == "inference_version_changed" for c in report["context"]
    ):
        vocabulary = source is not None and report["source_snapshot"]["matches_plan"]
        report["context"].append(
            _problem(
                "inference_version_changed",
                (
                    "The fresh inventory was loaded with a different vocabulary than the "
                    "reviewed snapshot, so its inference version differs from the plan's. "
                    "Load it with the review's vocabulary, or review it again."
                )
                if vocabulary
                else "The fresh inventory's inference version (vocabulary, implementation, "
                "settings or dependencies) differs from the plan's model_version. Approvals "
                "made under another version need renewed review.",
            )
        )

    observations = observe_tags(plan, fresh)
    ids = {c["resource_id"] for c in plan["changes"]}
    ownership = (
        _evidence_findings(source, fresh, ids)
        if source is not None and report["source_snapshot"]["matches_plan"]
        else {}
    )
    if source is None and not report["fresh_inventory"]["same_as_reviewed"]:
        report["warnings"].append(
            "The fresh inventory differs from the reviewed snapshot and no source snapshot was "
            "supplied, so only tag before-values were compared; ownership and cost-shape drift "
            "were not checked. Pass the saved review (--review-id or --source) for a full check."
        )

    for change in plan["changes"]:
        rid = change["resource_id"]
        obs = [o for o in observations if o["resource_id"] == rid]
        findings = []
        if obs[0]["state"] == MISSING:
            findings.append(
                {
                    "code": "resource_missing",
                    "reason": f"{rid} is not in the fresh inventory; it may have been deleted, "
                    "renamed or left out of the export. Confirm the resource exists before "
                    "delivering any change to it." + _renew(rid),
                }
            )
        else:
            if obs[0]["observed_cloud"] != change["cloud"]:
                findings.append(
                    {
                        "code": "cloud_changed",
                        "previous": change["cloud"],
                        "current": obs[0]["observed_cloud"],
                        "reason": f"{rid} was reviewed as a {change['cloud']} resource but the "
                        f"fresh inventory lists it under {obs[0]['observed_cloud']}; the exported "
                        "commands would target the wrong provider." + _renew(rid),
                    }
                )
            findings += [f for f in map(_tag_finding, obs) if f]
            findings += ownership.get(rid, [])
        report["resources"].append(
            {
                "resource_id": rid,
                "cloud": change["cloud"],
                "status": "needs_review" if findings else "ok",
                "tags": {
                    o["label"]: {k: o[k] for k in ("previous", "new", "observed", "state")}
                    for o in obs
                },
                "findings": findings,
            }
        )

    codes: dict[str, int] = {}
    for item in report["context"] + [f for r in report["resources"] for f in r["findings"]]:
        codes[item["code"]] = codes.get(item["code"], 0) + 1
    failed = bool(report["context"]) or any(r["findings"] for r in report["resources"])
    report["status"] = "fail" if failed else "pass"
    report["renewed_review_required"] = failed
    report["summary"] = {
        "resources": len(report["resources"]),
        "ok": sum(r["status"] == "ok" for r in report["resources"]),
        "needs_review": sum(r["status"] == "needs_review" for r in report["resources"]),
        "findings": codes,
    }
    return report


def summary_text(report: dict) -> str:
    """A short human-readable version of the report."""
    lines = [f"Pre-delivery check: {report['status'].upper()}"]
    for e in report["errors"]:
        lines.append(f"  invalid plan [{e['code']}]: {e['message']}")
    for c in report["context"]:
        lines.append(f"  [{c['code']}] {c['message']}")
    for w in report["warnings"]:
        lines.append(f"  warning: {w}")
    for r in report["resources"]:
        for f in r["findings"]:
            subject = f.get("label") or f.get("field") or ""
            lines.append(f"  {r['resource_id']} {subject} [{f['code']}]: {f['reason']}")
    if report["status"] == "pass":
        s = report["summary"]
        lines.append(
            f"  {s['ok']} planned resource(s) still match their reviewed before-values. "
            "Approvals remain bound to the reviewed snapshot; nothing was changed."
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────


def _saved_review(review_id: str) -> tuple[bytes, dict]:
    """A saved review's inventory and vocabulary, opened read-only."""
    from review_workspaces import _path

    path = _path(review_id)
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT csv, config FROM workspace WHERE id = 1").fetchone()
    if row is None:
        raise ValueError("Saved review has no inventory.")
    return row[0], json.loads(row[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    parser.add_argument("plan", type=Path, help="reviewed change plan JSON (format=plan export)")
    parser.add_argument("fresh", type=Path, help="freshly exported inventory CSV")
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
        plan = json.loads(args.plan.read_text())
        config = load_label_config(args.vocabulary) if args.vocabulary else None
        source = None
        if args.review_id:
            csv, saved_config = _saved_review(args.review_id)
            config = config or saved_config
            source = load_inventory_csv(csv, saved_config)
        elif args.source:
            source = load_inventory_csv(args.source, config)
        fresh = load_inventory_csv(args.fresh, config)
    except (
        OSError,
        ValueError,
        sqlite3.Error,
        InventoryImportError,
        LabelConfigError,
    ) as exc:
        print(f"Pre-delivery check: INVALID INPUT\n  {exc}", file=sys.stderr)
        return 2

    report = validate_plan(plan, fresh, source=source)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    print(summary_text(report), file=sys.stderr)
    return {"pass": 0, "fail": 1}.get(report["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
