"""Evaluate an approved, anonymized 50–100-resource label-review sample.

The input is a JSON export of reviewer feedback, not a cloud inventory. It contains
only surrogate IDs, suggestions, confidence, and reviewer-resolved values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from label_review import feedback_metrics
from labeling import AUTO_APPLY_THRESHOLD, BEST_STRATEGY, LABELS, evaluate

MIN_RESOURCES = 50
MAX_RESOURCES = 100
DEFAULT_CANDIDATES = (0.90, 0.95, 0.98)


def _resolved(entry: dict) -> str | None:
    if entry["decision"] in ("accept", "override"):
        return entry["value"]
    if entry["decision"] == "reject":
        return entry["previous"]
    return None


def _scored(entries: list[dict], label: str) -> list[dict]:
    return [
        entry
        for entry in entries
        if entry["label"] == label
        and entry.get("suggested")
        and _resolved(entry) is not None
        and isinstance(entry.get("confidence"), (int, float))
    ]


def _threshold_result(
    rows: list[dict], threshold: float, min_support: int, min_precision: float
) -> dict:
    reviewed = [entry for entry in rows if entry["confidence"] >= threshold]
    agreement = (
        sum(entry["suggested"] == _resolved(entry) for entry in reviewed) / len(reviewed)
        if reviewed
        else 0.0
    )
    return {
        "support": len(reviewed),
        "precision": round(agreement, 3),
        "eligible": len(reviewed) >= min_support and agreement >= min_precision,
    }


def evaluate_feedback(
    entries: list[dict],
    *,
    min_support: int = 20,
    min_precision: float = 0.98,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATES,
) -> dict:
    """Score a review sample and recommend a threshold only when evidence supports it."""
    required = {"resource_id", "label", "decision", "suggested", "previous", "confidence"}
    missing = [str(index) for index, entry in enumerate(entries) if required - set(entry)]
    if missing:
        raise ValueError(
            f"Feedback entries missing required fields at indexes: {', '.join(missing)}."
        )
    resource_count = len({entry["resource_id"] for entry in entries})
    if not MIN_RESOURCES <= resource_count <= MAX_RESOURCES:
        raise ValueError(
            f"Use a representative sample of {MIN_RESOURCES}–{MAX_RESOURCES} unique resources; "
            f"received {resource_count}."
        )

    evidence = {label: _scored(entries, label) for label in LABELS}
    thresholds = {
        label: {
            f"{candidate:.0%}": _threshold_result(rows, candidate, min_support, min_precision)
            for candidate in candidates
        }
        for label, rows in evidence.items()
    }
    recommendation = next(
        (
            candidate
            for candidate in candidates
            if all(thresholds[label][f"{candidate:.0%}"]["eligible"] for label in LABELS)
        ),
        None,
    )
    return {
        "sample": {
            "resources": resource_count,
            "feedback_events": len(entries),
            "min_support": min_support,
            "min_precision": min_precision,
        },
        "reviewer_agreement": feedback_metrics(entries),
        "threshold_evidence": thresholds,
        "recommended_auto_apply_threshold": recommendation,
        "recommendation": (
            f"Use {recommendation:.0%} for a monitored pilot."
            if recommendation is not None
            else (
                "Defer auto-apply: no candidate has sufficient observed precision and support "
                "for both labels."
            )
        ),
    }


def synthetic_baseline() -> dict:
    """Comparable synthetic metrics, explicitly not a production gate."""
    scores = evaluate()
    return {
        label: {
            "strategy": BEST_STRATEGY[label],
            **scores[label][BEST_STRATEGY[label]],
        }
        for label in LABELS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path, help="Approved anonymized reviewer-feedback JSON")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=0.98)
    args = parser.parse_args()
    entries = json.loads(args.feedback.read_text())
    report = evaluate_feedback(
        entries, min_support=args.min_support, min_precision=args.min_precision
    )
    report["synthetic_baseline_not_a_production_gate"] = synthetic_baseline()
    report["current_configured_threshold"] = AUTO_APPLY_THRESHOLD
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
