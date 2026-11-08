"""Summarise a real-world label-review pilot (issue #49) from its private logs.

    uv run python scripts/pilot_summary.py \
        --feedback pilot-workspace/feedback.json \
        --review-log pilot-workspace/review-log.csv \
        [--baseline-log pilot-workspace/manual-baseline-log.csv --baseline-sampling "..."] \
        [--re-review-log pilot-workspace/re-review-log.csv] \
        [--out pilot-workspace/pilot-summary.json]

Inputs (templates and column meanings in docs/pilot/; runbook in docs/pilot-kit.md):

  feedback.json        `review_cli.py feedback REVIEW_ID`, the same file real_evaluation.py reads
  review-log.csv       one row per resource label reviewed with obs-agent: seconds, outcome,
                       whether the ownership decision needed evidence the inventory lacked,
                       and a friction category
  manual-baseline-log  optional: the same work done by hand, without suggestions
  re-review-log.csv    optional: a second reviewer's independent decisions on a subset

It reports time per resource (obs-agent vs manual), usefulness (accepts, overrides, rejects,
unresolved and withheld labels, decisions that needed missing evidence), decision stability
between independent reviewers, and friction counts. It reads only surrogate IDs, labels,
values, durations and fixed categories: there is no free-text column, so the output can go into
the sanitized summary. Reviewer agreement here is not production accuracy (see
docs/real-evaluation.md).

Exit status: 0 success, 2 invalid input.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

LABELS = ("team", "env")
OUTCOMES = ("accept", "reject", "override", "unresolved")
FRICTION = (
    "",
    "import",
    "vocabulary",
    "unclear_reason",
    "missing_owner",
    "conflicting_evidence",
    "ui",
    "other",
)
RESOLUTIONS = ("", "first", "second", "other", "unresolved")
YES_NO = ("yes", "no")


class PilotInputError(ValueError):
    pass


def _rows(path: Path, columns: tuple[str, ...]) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise PilotInputError(f"{path.name}: missing column(s) {', '.join(missing)}")
        extra = [c for c in reader.fieldnames or [] if c not in columns]
        if extra:
            # No free-text columns: the summary must stay sanitized.
            raise PilotInputError(f"{path.name}: unexpected column(s) {', '.join(extra)}")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise PilotInputError(f"{path.name}: no rows")
    return rows


def _choice(row: dict, column: str, allowed: tuple[str, ...], where: str) -> str:
    value = row[column]
    if value not in allowed:
        shown = ", ".join(a or "(blank)" for a in allowed)
        raise PilotInputError(f"{where}: {column} must be one of {shown}")
    return value


def _seconds(row: dict, where: str) -> float:
    try:
        seconds = float(row["seconds"])
    except ValueError:
        seconds = -1.0
    if not 0 < seconds <= 3600:
        raise PilotInputError(f"{where}: seconds must be a number between 0 and 3600")
    return seconds


def _key(row: dict, where: str) -> tuple[str, str]:
    if not row["resource_id"]:
        raise PilotInputError(f"{where}: resource_id is empty")
    _choice(row, "label", LABELS, where)
    return row["resource_id"], row["label"]


def _per_resource(rows: list[dict]) -> dict[str, float]:
    """Total seconds per resource, across its labels."""
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["resource_id"]] = totals.get(row["resource_id"], 0.0) + row["_seconds"]
    return totals


def _timing(totals: dict[str, float]) -> dict:
    values = sorted(totals.values())
    p90 = values[min(len(values) - 1, round(0.9 * (len(values) - 1)))]
    return {
        "resources": len(values),
        "median_seconds_per_resource": round(statistics.median(values), 1),
        "p90_seconds_per_resource": round(p90, 1),
        "total_minutes": round(sum(values) / 60, 1),
    }


def _final_decisions(feedback: list[dict]) -> dict[tuple[str, str], dict]:
    """The last action per resource label; a revoke leaves it undecided."""
    final: dict[tuple[str, str], dict] = {}
    for entry in feedback:
        key = (entry["resource_id"], entry["label"])
        if entry["decision"] == "revoke":
            final.pop(key, None)
        else:
            final[key] = entry
    return final


def _resolved(entry: dict | None) -> str | None:
    if entry is None:
        return None
    if entry["decision"] in ("accept", "override"):
        return entry["value"]
    if entry["decision"] == "reject":
        return entry["previous"]
    return None


def load_review_log(path: Path) -> list[dict]:
    rows = _rows(
        path,
        ("resource_id", "label", "reviewer", "seconds", "outcome", "missing_evidence", "friction"),
    )
    seen = set()
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        key = _key(row, where)
        if key in seen:
            raise PilotInputError(f"{where}: {key[0]}/{key[1]} is logged twice")
        seen.add(key)
        if not row["reviewer"]:
            raise PilotInputError(f"{where}: reviewer is empty")
        row["_seconds"] = _seconds(row, where)
        _choice(row, "outcome", OUTCOMES, where)
        _choice(row, "missing_evidence", YES_NO, where)
        _choice(row, "friction", FRICTION, where)
    return rows


def load_baseline_log(path: Path) -> list[dict]:
    rows = _rows(path, ("resource_id", "label", "reviewer", "seconds", "resolved_value"))
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        _key(row, where)
        row["_seconds"] = _seconds(row, where)
    return rows


def load_re_review_log(path: Path) -> list[dict]:
    rows = _rows(path, ("resource_id", "label", "reviewer", "resolved_value", "resolution"))
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        _key(row, where)
        if not row["reviewer"]:
            raise PilotInputError(f"{where}: reviewer is empty")
        _choice(row, "resolution", RESOLUTIONS, where)
    return rows


def summarise(
    feedback: list[dict],
    review_log: list[dict],
    baseline_log: list[dict] | None = None,
    baseline_sampling: str | None = None,
    re_review_log: list[dict] | None = None,
) -> dict:
    final = _final_decisions(feedback)
    logged = {(r["resource_id"], r["label"]): r for r in review_log}

    # The review log and the saved review must describe the same decisions.
    for key, row in logged.items():
        decided = final.get(key)
        expected = decided["decision"] if decided else "unresolved"
        if row["outcome"] != expected:
            raise PilotInputError(
                f"review log says {key[0]}/{key[1]} was {row['outcome']}, but the saved review "
                f"records {expected}"
            )
    if missing := sorted(set(final) - set(logged)):
        raise PilotInputError(
            f"{len(missing)} decided label(s) are not in the review log, e.g. "
            f"{missing[0][0]}/{missing[0][1]}"
        )

    usefulness = {}
    for label in LABELS:
        rows = [r for r in review_log if r["label"] == label]
        counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
        withheld = sum(
            1
            for r in rows
            if (e := final.get((r["resource_id"], label))) is not None and not e.get("suggested")
        )
        usefulness[label] = {
            "reviewed": len(rows),
            **counts,
            "override_rate": round(counts["override"] / len(rows), 3) if rows else None,
            "rejection_rate": round(counts["reject"] / len(rows), 3) if rows else None,
            "decided_without_a_suggestion": withheld,
            # A rejected suggestion on an untagged label leaves no resolved value (as in
            # real_evaluation.py): it counts as unresolved for scoring.
            "unresolved_for_scoring": counts["unresolved"]
            + sum(
                1
                for r in rows
                if r["outcome"] == "reject"
                and _resolved(final.get((r["resource_id"], label))) is None
            ),
            "needed_missing_evidence": sum(r["missing_evidence"] == "yes" for r in rows),
        }

    friction = {
        category: sum(r["friction"] == category for r in review_log)
        for category in FRICTION
        if category
    }
    summary: dict = {
        "scope": {
            "resources": len({r["resource_id"] for r in review_log}),
            "resource_labels": len(review_log),
            "reviewers": len({r["reviewer"] for r in review_log}),
        },
        "time_with_obs_agent": _timing(_per_resource(review_log)),
        "usefulness": usefulness,
        "friction": {k: v for k, v in friction.items() if v},
        "agreement_is_not_accuracy": (
            "Reviewer agreement and decision stability describe this sample's human decisions. "
            "They are not an unbiased production accuracy estimate; report real_evaluation.py's "
            "output beside them and keep the distinction in the summary."
        ),
    }

    if baseline_log is not None:
        if not baseline_sampling or not baseline_sampling.strip():
            raise PilotInputError("Describe how the manual baseline was sampled.")
        baseline = _per_resource(baseline_log)
        tool = _per_resource(review_log)
        shared = sorted(set(baseline) & set(tool))
        if not shared:
            raise PilotInputError("The manual baseline shares no resources with the review log.")
        tool_shared = _timing({r: tool[r] for r in shared})
        manual = _timing({r: baseline[r] for r in shared})
        manual_median = manual["median_seconds_per_resource"]
        summary["manual_baseline"] = {
            "sampling": baseline_sampling.strip(),
            "resources_compared": len(shared),
            "manual": manual,
            "obs_agent_same_resources": tool_shared,
            "median_time_ratio": (
                round(tool_shared["median_seconds_per_resource"] / manual_median, 2)
                if manual_median
                else None
            ),
        }

    if re_review_log is not None:
        compared, agreed, disagreements = 0, 0, []
        for row in re_review_log:
            key = (row["resource_id"], row["label"])
            if key not in logged:
                raise PilotInputError(f"re-review of {key[0]}/{key[1]} is not in the review log")
            first_reviewer = (final.get(key) or {}).get("reviewer")
            if row["reviewer"] in (logged[key]["reviewer"], first_reviewer):
                raise PilotInputError(
                    f"re-review of {key[0]}/{key[1]} must be by a different reviewer"
                )
            first = _resolved(final.get(key))
            second = row["resolved_value"] or None
            compared += 1
            if first == second:
                agreed += 1
            else:
                disagreements.append(
                    {
                        "resource_id": key[0],
                        "label": key[1],
                        "first": first,
                        "second": second,
                        "resolution": row["resolution"] or "not recorded",
                    }
                )
        summary["decision_stability"] = {
            "re_reviewed": compared,
            "agreed": agreed,
            "agreement": round(agreed / compared, 3),
            "disagreements": disagreements,
            "unresolved_disagreements": sum(
                d["resolution"] in ("not recorded", "unresolved") for d in disagreements
            ),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--review-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--baseline-sampling", help="how the manual-baseline subset was chosen")
    parser.add_argument("--re-review-log", type=Path)
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    try:
        feedback = json.loads(args.feedback.read_text())
        if not isinstance(feedback, list):
            raise PilotInputError("feedback must be a JSON list (review_cli.py feedback output)")
        result = summarise(
            feedback,
            load_review_log(args.review_log),
            load_baseline_log(args.baseline_log) if args.baseline_log else None,
            args.baseline_sampling,
            load_re_review_log(args.re_review_log) if args.re_review_log else None,
        )
    except (OSError, ValueError, KeyError) as exc:
        # Messages name surrogate IDs and file names only; nothing is logged.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
