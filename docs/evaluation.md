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

MLflow is an optional dependency group for traces and comparable live-evaluation runs. It is not
installed in CI, the Docker image, or the hosted demo.

```bash
uv sync --group mlops
OBS_AGENT_MLFLOW=1 uv run agent.py -q "Which team does svc-ingest belong to?"
uv run evals_mlflow.py --provider deepseek
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Traces record agent, model, and tool spans. `evals_mlflow.py` scores both facts-present and
tools-called, so a model cannot pass merely by guessing the right answer.

For test commands, limitations, and the CI matrix, see [Testing](testing.md).
