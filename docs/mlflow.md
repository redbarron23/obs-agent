# MLflow in obs-agent: tracing and evaluation

MLflow helps answer two questions in this project: what happened inside an agent request, and
how a provider/model performed across the evaluation questions. This guide starts with those
concepts, follows the implementation, and includes an offline exercise you can inspect in the UI.

The repository currently locks MLflow to **3.17.2** in the optional `mlops` dependency group.
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
the tracking destination and experiment, and attempts SDK autologging. A successful configuration
is kept for the process; failures are retried a bounded number of times (section 6). The manual
spans in `agent.py` give the application a consistent trace structure across provider paths,
including paths SDK autologging may not cover.

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

Tracing is best-effort and never fails an agent request. If MLflow is missing, `setup()` warns
and tracing stays off. If the tracking store is unreachable (for example `set_experiment` raises
`ConnectionError`), `setup()` logs a warning naming the error type and a redacted destination
(no credentials, path, query or request content), and that request runs untraced. Setup is not
marked complete: a later request retries after `RETRY_INTERVAL_S` (60 s), and after
`MAX_SETUP_ATTEMPTS` (3) failures tracing is disabled until the process restarts.
`observability.status()` reports `active`, `attempts` and `disabled_reason`. A span that fails
to record is skipped with a single warning. Exceptions raised by the agent itself still propagate.

Trace inputs and outputs can contain questions, tool arguments, answers, and SDK-captured content.
Choose storage access, retention, and content-capture/redaction policies before using private data.
The current cost agent uses synthetic data, but a question can still contain sensitive text.

A trace shows recorded execution. Scorers assess particular properties of that execution. Neither
replaces representative production validation or monitoring of the primary label-review service.

## 7. Choose MLflow, Jaeger, or both

| Mode | Enablement before process startup | Recorded data and destination |
| --- | --- | --- |
| Both off (default) | Unset `OBS_AGENT_MLFLOW`, `MLFLOW_TRACKING_URI`, and `OTEL_TRACES_EXPORTER` | Neither application tracing path records spans |
| MLflow only | Set `OBS_AGENT_MLFLOW=1` or `MLFLOW_TRACKING_URI`; leave `OTEL_TRACES_EXPORTER` unset | Agent/model/tool content and SDK detail go to the MLflow tracking store |
| Jaeger only | Set `OTEL_TRACES_EXPORTER=otlp` and the HTTP/protobuf endpoint; leave both MLflow enablement settings unset | Metadata-only policy caller spans go to Jaeger; RAG exports its own spans |
| Both | Enable both paths and install both optional groups | Separate MLflow agent traces and Jaeger policy-request traces; no content forwarding from MLflow to Jaeger by default |

For both paths in a local Python process, install with
`uv sync --locked --group mlops --group telemetry`. The combined Docker simulation includes
OpenTelemetry but does not install or enable MLflow. Follow the
[Jaeger setup](local-rag-stack.md#opt-in-to-jaeger-tracing) for the actual endpoint and startup
commands; starting a collector alone does not activate tracing.

### Content capture and export routing

**Enabling MLflow opts into content capture.** Manual spans retain questions, last-message content,
tool arguments/results and final answers; SDK autologging can retain complete model requests and
responses, including policy excerpts returned to the model. Exception messages may also be
recorded by MLflow. This is not a metadata-only or automatically redacted mode. Use fictional or
sanitized inputs unless content capture, access and retention are explicitly approved for the
selected MLflow store. Do not put secrets or reviewer identities into questions or tool data.
The default remains off, and enabling Jaeger alone does not enable MLflow.

Jaeger caller spans retain only operation/outcome, HTTP status and timing. RAG additionally permits
configured service/model metadata, usage, source paths and retrieval distances. They exclude raw
questions, documents, answers, reviewer data, authorization headers and exception text. Source
paths can still identify private documents; see the
[local data boundary](local-rag-stack.md#sampling-storage-and-local-access).

MLflow 3.17.2 recognizes standard OTLP endpoint variables independently of the policy exporter.
Without a guard, setting a Jaeger endpoint can reroute content-bearing MLflow spans there, even
when the policy exporter's `OTEL_TRACES_EXPORTER` is `none`. `observability.setup()` therefore
defaults **`MLFLOW_ENABLE_OTLP_EXPORTER=false`** before configuring tracing. MLflow keeps its
tracking destination, and the policy exporter still uses its own OTLP endpoint. This is a routing
boundary, not a content-redaction filter.

An explicit `MLFLOW_ENABLE_OTLP_EXPORTER=true` is preserved as an advanced opt-in to exporting
MLflow content to OTLP. It can replace MLflow tracking export; MLflow's separate dual-export option
can send to both destinations. These routing overrides are outside the tested separate-backend
mode: review data approval and destination behavior before using them. Keep the setting false
for the metadata-only Jaeger experiment, and restart after changing configuration. Agent evaluation
runs/metrics retain their existing MLflow destination and scorer behavior.

### Context, duplicate spans and failure behavior

The supported separate-backend mode keeps MLflow's default isolated provider/context:
`MLFLOW_USE_DEFAULT_TRACER_PROVIDER=true` and
`MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT=false` (the SDK defaults). The policy client owns a private
OpenTelemetry provider and does not replace the process-global provider. Do not switch MLflow to
the global provider or enable context propagation and expect the same isolation guarantees.

The offline coexistence test runs the real agent, OpenAI SDK autologging and production policy
client with a temporary MLflow database and in-process HTTP/export substitutes. It verifies all
four modes, unchanged answers, no duplicate application operations, provider/context isolation,
and content exclusion from policy exports and captured diagnostics. It deliberately confirms that
MLflow does contain fictional question, answer and excerpt sentinels, while credential sentinels
remain absent. This validates the content boundary; it does not certify arbitrary third-party
autologging integrations or private workloads.

One agent/model/tool span per operation is intentional. A nested SDK span adds request/response
detail and is not a second application model call. The Jaeger caller and RAG server spans likewise
measure different boundaries. In separate-backend mode, the policy caller starts its own trace;
its W3C context connects it to RAG, not to the isolated MLflow agent trace. Do not assume that the
two UIs share trace IDs or add parent and child durations as separate elapsed time.

Existing failure tests cover missing optional packages, disabled tracing, unavailable MLflow
configuration/span recording and policy SDK/export failures. Requests continue without those trace
records, while application exceptions remain failures. Backend downtime does not guarantee replay
of dropped spans. Run the offline checks with:

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run --group mlops pytest tests/test_observability.py tests/test_policy_tracing.py -q
```
