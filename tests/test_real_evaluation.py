import pytest

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
