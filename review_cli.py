"""Command-line access to saved private reviews: import, queue, decide, export.

    uv run python review_cli.py import inventory.csv [--vocabulary vocab.json] [--name NAME]
    uv run python review_cli.py list
    uv run python review_cli.py queue REVIEW_ID [--label team] [--evidence insufficient]
    uv run python review_cli.py decide REVIEW_ID RESOURCE team accept --reviewer alice
    uv run python review_cli.py export REVIEW_ID --format plan --output plan.json

It calls the same workspace, validation, inference, decision and export functions as the Label
review UI and the REST API's /reviews routes, on the saved reviews in OBS_AGENT_REVIEW_DIR
(default review-workspaces/). No LLM, network or cloud credentials are used. Saved reviews are
refused when OBS_AGENT_DEMO=1, and an unknown review ID is an error, never synthetic data.

Exit status: 0 success, 2 invalid input, unknown review or refused decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABELS = ("team", "env")
EVIDENCE = ("insufficient", "supported", "missing_signals", "insufficient_support", "conflicting")


def _review_id(value: str) -> str:
    from review_workspaces import is_review_id

    if not is_review_id(value):
        raise argparse.ArgumentTypeError("a saved review ID is 64 lowercase hex digits")
    return value


def _score(value: str) -> float:
    score = float(value)
    if not 0 <= score <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Saved private reviews from the command line (no LLM, no cloud credentials).",
        epilog="Output is JSON on stdout; errors go to stderr.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    imp = commands.add_parser("import", help="validate and save an inventory CSV")
    imp.add_argument("csv", type=Path, help="sanitized Azure/GCP inventory CSV")
    imp.add_argument("--vocabulary", type=Path, help="team/env vocabulary JSON")
    imp.add_argument(
        "--mapping", type=Path, help="JSON {import field: source column} for other headers"
    )
    imp.add_argument("--name", help="display name (default: the CSV file name)")

    commands.add_parser("list", help="list saved reviews, newest first")

    def review(name: str, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.add_argument("review_id", type=_review_id, help="saved review ID")
        return sub

    review("show", "one review's summary and approval context")
    review("coverage", "untagged / suspect / insufficient-evidence counts")

    queue = review("queue", "resources needing review, highest model score first")
    queue.add_argument("--label", choices=LABELS)
    queue.add_argument("--status", choices=("untagged", "suspect"))
    queue.add_argument("--cloud", choices=("azure", "gcp"))
    queue.add_argument("--min-confidence", type=_score, default=0.0)
    queue.add_argument("--max-confidence", type=_score, default=1.0)
    queue.add_argument("--search", help="substring of resource_id")
    queue.add_argument("--evidence", choices=EVIDENCE)
    queue.add_argument("--hide-reviewed", action="store_true", help="omit decided resources")

    resource = review("resource", "suggestions for one resource")
    resource.add_argument("resource_id")

    decide = review("decide", "record accept / reject / override")
    decide.add_argument("resource_id")
    decide.add_argument("label", choices=LABELS)
    decide.add_argument("decision", choices=("accept", "reject", "override"))
    decide.add_argument("--value", help="value to apply (override only)")
    decide.add_argument("--reviewer", help="reviewer name or ID (default: OBS_AGENT_REVIEWER)")

    undo = review("undo", "revoke a decision (recorded in feedback)")
    undo.add_argument("resource_id")
    undo.add_argument("label", choices=LABELS)
    undo.add_argument("--reviewer", help="reviewer name or ID (default: OBS_AGENT_REVIEWER)")
    undo.add_argument("--reason", help="optional reason, kept in the review history only")

    decisions = review("decisions", "recorded decisions")
    decisions.add_argument(
        "--current", action="store_true", help="only decisions bound to this snapshot"
    )
    review("feedback", "append-only reviewer actions with model context")
    review("backlog", "pending review work and its age")

    export = review("export", "resolved label changes")
    export.add_argument("--format", choices=("csv", "sh", "tfvars", "plan"), default="plan")
    export.add_argument(
        "--include-auto",
        action="store_true",
        help="also include unreviewed auto-apply suggestions (not for plan)",
    )
    export.add_argument("--output", type=Path, help="write here instead of stdout")
    return parser


def _queue(args, inventory, store) -> list[dict]:
    from label_review import current_decisions, review_queue

    rows = review_queue(
        label=args.label,
        status=args.status,
        cloud=args.cloud,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence,
        search=args.search,
        inventory=inventory,
        evidence=args.evidence,
    )
    if args.hide_reviewed:
        # As in the UI: a resource is hidden once every flagged label it shows is decided.
        decided = {(d["resource_id"], d["label"]) for d in current_decisions(store, inventory)}
        labels = [args.label] if args.label else list(LABELS)
        rows = [
            r
            for r in rows
            if any(
                r["labels"][lbl]["status"] != "ok" and (r["resource_id"], lbl) not in decided
                for lbl in labels
            )
        ]
    return rows


def run(args) -> object:
    import label_review
    import review_workspaces as rw
    import workflow_metrics

    if args.command == "import":
        vocabulary = args.vocabulary.read_bytes() if args.vocabulary else None
        mapping = json.loads(args.mapping.read_text()) if args.mapping else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("--mapping must be a JSON object of import field to source column.")
        return rw.import_review(
            args.csv.read_bytes(), args.name or args.csv.name, vocabulary, mapping
        )
    rw.require_private_reviews()
    if args.command == "list":
        return rw.saved_reviews()

    inventory, store = rw.load_review(args.review_id)
    if args.command == "show":
        return rw.review_summary(args.review_id, inventory)
    if args.command == "coverage":
        return label_review.coverage(inventory)
    if args.command == "queue":
        return _queue(args, inventory, store)
    if args.command in ("resource", "decide"):
        report = label_review.resource_report(args.resource_id, inventory)
        if report is None:
            raise ValueError("Unknown resource in this saved review.")
        if args.command == "resource":
            return report
        with workflow_metrics.operation("decision_write", invalid=(label_review.DecisionError,)):
            return store.set(
                args.resource_id,
                args.label,
                args.decision,
                args.value,
                inventory,
                reviewer=args.reviewer,
            )
    if args.command == "undo":
        with workflow_metrics.operation("decision_write", invalid=(label_review.DecisionError,)):
            event = store.revoke(
                args.resource_id, args.label, reviewer=args.reviewer, reason=args.reason
            )
        if event is None:
            raise ValueError("Decision not found.")
        return {"deleted": f"{args.resource_id}/{args.label}", "revocation": event}
    if args.command == "decisions":
        return label_review.current_decisions(store, inventory) if args.current else store.all()
    if args.command == "feedback":
        return store.feedback()
    if args.command == "backlog":
        return workflow_metrics.backlog(store, inventory)
    # export
    with workflow_metrics.operation("export", invalid=(label_review.DecisionError,)):
        return label_review.export(store, args.format, args.include_auto, inventory)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import sqlite3

    from review_workspaces import PrivateReviewsDisabled

    try:
        result = run(args)
    except (OSError, ValueError, sqlite3.Error, PrivateReviewsDisabled) as exc:
        # ValueError covers invalid CSV/vocabulary, unknown reviews and refused decisions.
        # The message goes to the caller's terminal only; nothing is logged.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = result if isinstance(result, str) else json.dumps(result, indent=2, default=str) + "\n"
    output = getattr(args, "output", None)
    if output:
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
