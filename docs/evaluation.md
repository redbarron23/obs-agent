# Evaluation and observability

## Label-inference quality

`inventory.py` keeps ground truth beside deliberately incomplete and messy observed tags. The label
tests pin quality floors against that ground truth, including auto-apply precision on the adversarial
estate. The report also includes expected calibration error (ECE), the gap between stated confidence
and observed accuracy; lower is better.

```bash
uv run labeling.py --hardness both
uv run pytest tests/test_labeling.py -q
```

Synthetic evaluation is necessary but insufficient. In production, validate the model with a
reviewed sample of unlabelled resources, then use reviewer feedback to monitor calibration and
retrain only when enough representative data exists.

For a repeatable 50–100-resource production-validation process, including anonymisation and a
threshold recommendation that defers when evidence is insufficient, use the
[real-world evaluation pack](real-evaluation.md).
Changing the expedited-review threshold also requires the documented
[approval process](deployment.md#confidence-threshold-changes).

## Anonymized review fixture and post-review metrics

[`fixtures/anonymized_review_feedback.json`](../fixtures/anonymized_review_feedback.json) is a
small, hand-curated production-style sample with opaque resource IDs and no people, account IDs,
or billing data. It exercises the feedback-reporting contract without claiming to be a production
benchmark. Replace it with an approved, anonymized export before drawing conclusions about a real
estate.

Every review action retains the original suggestion and the reviewer outcome. `GET
/labels/feedback/metrics` reports macro precision, recall, and F1 against the reviewer-resolved
value, grouped by label and month. An accept or override supplies that value; a rejection of an
already-tagged resource uses its prior label. Rejecting an untagged suggestion is **unresolved**,
not a false label, because no replacement truth was supplied.

The Label review page shows the same report in its **Review quality** panel, including monthly
trends and a below/at-or-above-90% confidence split. Watch whether high-confidence agreement
falls relative to lower-confidence review: that is a practical signal to revisit the threshold.

These are reviewer-agreement metrics, not an unbiased production benchmark: reviewers choose what
to inspect, and repeated decisions are retained as historical events. Track the `reviewed`,
`scored`, and `unresolved` counts with the scores; do not compare periods with too little reviewed
data.

## Cost-agent evaluations

The agent has three complementary layers:

| Layer | Question | CI | Provider key |
|---|---|---:|---:|
| Unit and integration tests | Does code behave as written? | Yes | No |
| Eval dry-run | Do tools still expose the needed facts? | Yes | No |
| Live evaluation | Does a real model choose and use tools correctly? | No | Yes |

```bash
# Deterministic, offline evaluation.
uv run evals.py --dry-run

# End-to-end evaluation against a configured provider.
uv run evals.py
uv run evals.py --provider anthropic --model claude-haiku-4-5-20251001
```

The 12 cases cover rankings, trends, spikes, cross-cloud summaries, label coverage, suggestions,
and suspected wrong labels. They use fact-presence checks rather than exact text because correct
agent answers can be phrased differently.

## MLflow

MLflow is an optional dependency group for traces and comparable live-evaluation runs. The
dedicated `mlops` CI job installs it; ordinary test jobs, the default Docker image, and the hosted
demo do not. Start with the [MLflow walkthrough](mlflow.md) for the vocabulary, an offline trace
exercise, UI inspection, evaluation limits, and troubleshooting.

```bash
uv sync --locked --group mlops
OBS_AGENT_MLFLOW=1 uv run --group mlops agent.py -q "Which team does svc-ingest belong to?"
uv run --group mlops evals_mlflow.py --provider deepseek
uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Traces record agent, model, and tool spans. `evals_mlflow.py` scores both facts-present and
tools-called, so the fact-presence check alone cannot hide a skipped required tool. These checks
do not establish complete semantic correctness. Agent and evaluation commands above call a real
provider; hosted providers can incur charges.

For test commands, limitations, and the CI matrix, see [Testing](testing.md).
