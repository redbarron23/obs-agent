# MLflow in obs-agent: tracing and evaluation

MLflow helps answer two questions in this project: what happened inside an agent request, and
how a provider/model performed across the evaluation questions. This guide starts with those
concepts, follows the implementation, and includes an offline exercise you can inspect in the UI.

The repository currently locks MLflow to **3.17.0** in the optional `mlops` dependency group.
This integration covers the optional cost agent and its evaluations. The local label-inference
models, review decisions, and approval fingerprints use the separate workflows described in
[Architecture](architecture.md) and [Decision storage](decision-storage.md).

## 1. Understand the objects

| Object | Meaning | Example in this repository |
| --- | --- | --- |
| Experiment | A named collection of related work | `obs-agent` by default |
| Run | One recorded evaluation execution | One provider/model answering the 12 evaluation questions |
| Parameter | A setting recorded for a run | Provider, requested model, case count, Git commit |
| Metric | A numerical result | Mean `facts_present` or `tools_called` score |
| Trace | The execution record of one request | Answering a question about Azure overage |
| Span | One operation inside a trace | An LLM call or a tool execution |

A run lets you compare evaluations; a trace lets you investigate a request. Starting a run or
logging a metric does not create a trace. A standalone traced request does not need an active run.
See the [MLflow tracing FAQ](https://mlflow.org/docs/latest/genai/tracing/faq/).

For a question that needs one tool, the application trace can look like this:

```text
agent                                      AGENT: root span, entire request
├── llm: <provider>/<model>                 CHAT_MODEL: choose a tool
├── get_azure_top_overages                  TOOL: execute the tool
└── llm: <provider>/<model>                 CHAT_MODEL: produce the answer
```

The tree is one trace. Each operation has its own timing and recorded inputs/outputs. Child spans
refer to their parent, so you can follow the request through the operations it triggered.
Provider SDK autologging may add further nested spans.

## 2. Follow the code

| File | What to read |
| --- | --- |
| [observability.py](../observability.py) | Enablement, destination, experiment, optional span helpers |
| [agent.py](../agent.py) | Root agent span, model-call spans, tool spans, recorded outputs |
| [evals_mlflow.py](../evals_mlflow.py) | Evaluation dataset, scorers, run parameters, result summary |
| [tests/test_observability.py](../tests/test_observability.py) | Offline examples with a mocked provider and temporary tracking database |

`observability.setup()` checks whether tracing is requested, imports MLflow if available, selects
the tracking destination and experiment, and attempts SDK autologging. Configuration is checked
once per process. The manual spans in `agent.py` give the application a consistent trace structure
across provider paths, including paths SDK autologging may not cover.

`observability.span()` wraps `mlflow.start_span()` when tracing is active. Its callers record:

- Agent span: question, final answer, requested provider/model, and reported answering model.
- Model span: message count and last content, completion reason, requested tools, reported model,
  and token usage where available. Some provider/streaming paths do not supply all these fields.
- Tool span: tool arguments and returned output.

When tracing is disabled, the helper yields `None`; `set_outputs(None, ...)` does nothing.
The same agent code can therefore run without MLflow installed.
See [MLflow manual tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/).

## 3. Configure the destination

Run these commands from the repository root:

```bash
uv sync --locked --group mlops
```

Keep `--group mlops` on the `uv run` commands below so uv includes the optional dependency group.

| Setting | Application behavior |
| --- | --- |
| Neither enablement setting supplied | Tracing is off |
| `OBS_AGENT_MLFLOW=1` | Request tracing; default destination is `sqlite:///mlflow.db` |
| `MLFLOW_TRACKING_URI` set | Request tracing and use that destination |
| `OBS_AGENT_MLFLOW_EXPERIMENT` | Override the default experiment name `obs-agent` |

`sqlite:///mlflow.db` is a SQLite file relative to the process's working directory. The application
and UI must use the same destination. The default `mlflow.db` and `mlruns/` paths are ignored by
git; custom storage paths need their own handling. MLflow state is separate from the SQLite
databases holding label-review approvals.

## 4. Create and inspect a trace without an LLM

This exercise calls a real deterministic tool on synthetic billing data. It creates two manual
spans and checks their relationship. It does not contact an LLM provider or measure model quality.

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db OBS_AGENT_MLFLOW_EXPERIMENT=obs-agent-learning \
uv run --group mlops python - <<'PY'
import mlflow
import observability
from tools import get_azure_top_overages

assert observability.setup()
with observability.span("learning-agent", "AGENT", inputs={"question": "Top Azure overage?"}) as root:
    with observability.span("get_azure_top_overages", "TOOL", inputs={"n": 1}) as tool:
        result = get_azure_top_overages(n=1)
        observability.set_outputs(tool, result)
    observability.set_outputs(root, {"answer": result})

trace_id = observability.last_trace_id()
mlflow.flush_trace_async_logging()
trace = mlflow.get_trace(trace_id)
spans = {span.name: span for span in trace.data.spans}
assert spans["get_azure_top_overages"].parent_id == spans["learning-agent"].span_id
assert "subscription_id" in spans["get_azure_top_overages"].outputs
print(f"Recorded trace: {trace_id}")
PY
```

Open the UI in the same repository directory:

```bash
uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Visit `http://127.0.0.1:5000`, select **obs-agent-learning**, and open **Traces**. Select the trace
ID printed above. Find the root operation and its tool child, inspect the tool's `n=1` input and
output, and compare their durations. The root output should contain the same tool result.
This exercise has no evaluation run, so an empty Runs view is expected. Stop the UI with Ctrl+C.

For the application's mocked agent-to-provider integration, run:

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run --group mlops pytest tests/test_observability.py -q
```

Those tests use temporary databases, which are separate from the learning trace in `mlflow.db`.
The dedicated `mlops` CI job installs this dependency group and runs those tests. Ordinary Python
test jobs, the default Docker image, and the hosted demo omit MLflow.

## 5. Compare evaluation runs

`evals_mlflow.py` starts a named run for a provider/model, records its settings and Git commit,
and calls `mlflow.genai.evaluate()` with the 12 questions and two custom scorers. Each question
produces a trace that the scorer can inspect. See
[MLflow agent evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/).

| Scorer | What passes | What a pass does not establish |
| --- | --- | --- |
| `facts_present` | Answer contains the expected fact strings | Complete semantic correctness or a sound explanation |
| `tools_called` | Trace contains all tools in one accepted tool set | Correct arguments, correct use of results, call order, or absence of extra tool calls |

Inspect both aggregate scores and failed-question traces. A correct-looking answer with a missing
required tool is a different failure from a correct tool call followed by a wrong answer.

The following commands are **opt-in live evaluations**, unlike the offline exercise. Hosted
providers can incur API charges; Ollama needs the local server and model available.

```bash
# Local provider: no hosted API charge.
uv run --group mlops evals_mlflow.py --provider ollama --model qwen2.5:7b

# Hosted provider: calls a paid API using the configured key.
uv run --group mlops evals_mlflow.py --provider deepseek
```

The runner requests tracing automatically. With no tracking URI override, it uses `mlflow.db`;
with no experiment override, it uses `obs-agent`. Open the corresponding experiment in the UI to
compare runs and inspect traces. For a useful comparison, retain the dataset, provider/model,
Git commit, and scorer definitions; changed questions or checks can change scores independently
of model quality.

## 6. Investigate failures and recognize limits

| Symptom | First check |
| --- | --- |
| No traces | Was tracing enabled before the process configured itself, and is MLflow installed? |
| UI looks empty | Same database/server, working directory, experiment, time filter, and Traces view? |
| Standalone request has no run | Expected: a trace does not require an evaluation run |
| Slow answer | Which model/tool span dominates the trace duration? |
| Wrong answer | Tool name and arguments, tool output, then final answer |
| Token/model details missing | Whether that provider and streaming path returned those fields |
| Tests skip tracing checks | Include the `mlops` group |

Changing enablement settings after `setup()` has run requires restarting the process. `reset()`
exists for isolated tests, not as a per-request production configuration switch.

There is a known initialization failure in
[#59](https://github.com/redbarron23/obs-agent/issues/59): an unavailable tracking store can raise
before the agent loop runs. Setup marks itself checked before finishing; subsequent calls can
then continue with tracing inactive. The intended fix is to preserve agent availability, report
degraded tracing, and retain application errors. This guide documents current behavior; it does
not implement that fix.

Trace inputs and outputs can contain questions, tool arguments, answers, and SDK-captured content.
Choose storage access, retention, and content-capture/redaction policies before using private data.
The current cost agent uses synthetic data, but a question can still contain sensitive text.

A trace shows recorded execution. Scorers assess particular properties of that execution. Neither
replaces representative production validation or monitoring of the primary label-review service.
