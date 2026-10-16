import pytest

from label_review import feedback_metrics
from real_evaluation import evaluate_feedback


def _sample(*, team_correct: bool = True) -> list[dict]:
    rows = []
    for index in range(50):
        for label, value in (("team", "payments"), ("env", "prod")):
            correct = team_correct or label == "env"
            rows.append(
                {
                    "resource_id": f"surrogate-{index:03d}",
                    "label": label,
                    "decision": "accept" if correct else "override",
                    "value": value if correct else "web",
                    "suggested": value,
                    "previous": None,
                    "confidence": 0.96,
                    "source": "model",
                    "reason": "test",
                    "decided_at": "2026-10-13T10:00:00+00:00",
                }
            )
    return rows


def test_recommends_lowest_supported_threshold():
    report = evaluate_feedback(_sample(), min_support=20, min_precision=0.98)
    assert report["recommended_auto_apply_threshold"] == 0.9
    assert report["threshold_evidence"]["team"]["90%"]["support"] == 50


def test_defers_when_observed_precision_is_too_low():
    report = evaluate_feedback(_sample(team_correct=False), min_support=20, min_precision=0.98)
    assert report["recommended_auto_apply_threshold"] is None
    assert report["recommendation"].startswith("Defer")


def test_rejects_nonrepresentative_sample_size():
    with pytest.raises(ValueError, match="50–100"):
        evaluate_feedback(_sample()[:20])


def test_repeated_reviews_do_not_create_threshold_support():
    rows = _sample()
    for row in rows[2:]:
        row["confidence"] = 0.2
    rows += [dict(row) for _ in range(19) for row in rows[:2]]

    report = evaluate_feedback(rows)

    assert report["recommended_auto_apply_threshold"] is None
    assert report["threshold_evidence"]["team"]["90%"]["support"] == 1
    assert report["threshold_evidence"]["env"]["90%"]["support"] == 1
    assert report["sample"]["feedback_events"] == 138
    assert report["sample"]["unique_resource_labels"] == 100
    assert report["sample"]["duplicate_events"] == 38
    assert report["reviewer_agreement"]["reviewed"] == 100
    assert feedback_metrics(rows)["reviewed"] == 138


def test_repeated_matching_reviews_count_once_even_with_different_review_metadata():
    rows = _sample()
    repeated = [row | {"reviewer": "second-reviewer", "decided_at": "2026-10-16"} for row in rows]
    report = evaluate_feedback(rows + repeated)
    assert report["threshold_evidence"]["team"]["90%"]["support"] == 50
    assert report["recommended_auto_apply_threshold"] == 0.9


@pytest.mark.parametrize(
    "change",
    [
        {"decision": "override", "value": "web"},
        {"confidence": 0.99},
        {"suggested": "web"},
        {"previous": "web"},
        {"decision": "reject", "value": None},
    ],
)
def test_conflicting_reviews_require_an_explicit_resolved_outcome(change):
    rows = _sample()
    with pytest.raises(ValueError, match="Conflicting feedback.*surrogate-000.*team"):
        evaluate_feedback(rows + [rows[0] | change])


@pytest.mark.parametrize("field", ["inventory_id", "model_version"])
def test_different_evaluation_contexts_cannot_be_pooled(field):
    rows = [row | {"inventory_id": "snapshot-a", "model_version": "version-a"} for row in _sample()]
    # Distinct resources still cannot supply evidence for different contexts in one report.
    rows[-1][field] = "different-context"
    with pytest.raises(ValueError, match="one inventory and inference version"):
        evaluate_feedback(rows)


def test_feedback_with_and_without_provenance_cannot_be_pooled():
    rows = _sample()
    rows[0].update(inventory_id="snapshot-a", model_version="version-a")
    with pytest.raises(ValueError, match="one inventory and inference version"):
        evaluate_feedback(rows)


@pytest.mark.parametrize("field", ["inventory_id", "model_version"])
def test_partial_provenance_is_rejected(field):
    rows = [row | {field: "partial-context"} for row in _sample()]
    with pytest.raises(ValueError, match="both inventory_id and model_version"):
        evaluate_feedback(rows)


def test_one_known_context_retains_unique_evidence():
    rows = [row | {"inventory_id": "snapshot-a", "model_version": "version-a"} for row in _sample()]
    report = evaluate_feedback(rows + [dict(rows[0])])
    assert report["sample"]["evaluation_context"] == {
        "inventory_id": "snapshot-a",
        "model_version": "version-a",
    }
    assert report["threshold_evidence"]["team"]["90%"]["support"] == 50
