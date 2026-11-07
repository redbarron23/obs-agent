# Simulate obs-agent and RAG locally

The standalone `compose.rag.yml` starts the real obs-agent API, Streamlit review UI and
rag-observability retrieval API. It indexes the bundled fictional policies/runbook, preserves local
state in dedicated volumes, and needs no cloud credentials or LLM provider key. It does not forward
provider keys from your regular Compose `.env` file or mount your private review directory.

## Start

Requirements: Docker with a running daemon and Docker Compose 2.24 or newer. Use Docker Desktop
or your existing Colima environment. First build/start downloads CPU model libraries and the local
embedding model, so allow several minutes and enough disk space for the images.

From the obs-agent checkout:

```bash
docker compose -f compose.rag.yml up -d --build --wait --wait-timeout 600
docker compose -f compose.rag.yml ps
```

| Interface | Address |
| --- | --- |
| Label review UI | <http://127.0.0.1:8501> |
| obs-agent API documentation | <http://127.0.0.1:8000/docs> |
| RAG health | <http://127.0.0.1:8765/health> (requires the bearer key below) |

All published ports bind to host loopback. Stop another stack using these ports before starting.
This is a local simulation with fixed demo keys: `local-obs-simulation` for obs-agent's `X-API-Key`
and `local-rag-simulation` for RAG's bearer header. Use only fictional/sanitized data in this stack;
these public demo values are not deployment credentials.

By default the RAG build uses the pinned Git source `19ebcdd` (retrieval contract 1.1.0). obs-agent
builds from this checkout. The RAG image copies application modules and the nine bundled sample
documents explicitly; it never copies `data/uploads/`. To iterate on your local RAG checkout:

```bash
RAG_REPO_PATH=../rag-observability docker compose -f compose.rag.yml up -d --build --wait --wait-timeout 600
```

The override is relative to this Compose file, or can be an absolute path. Use a sanitized checkout;
Docker receives that directory as its build context. Record `git rev-parse HEAD` in each checkout
when comparing runs. Rebuild the RAG image after editing its source or corpus.

## Try the integrated review

1. Open **Label review** at port 8501 and use the synthetic inventory.
2. Enter a reviewer name, choose a label/resource, and press **Show policy context** below the
   review table. Inspect the cited source/section, corpus identity and freshness beside the
   resource evidence. Policy text does not change label confidence or approvals.
3. Review a suggestion and download its reviewed plan. Decisions persist in the stack's own
   `review-state` volume, shared by the UI and API.

No model server is started for cost chat. The review-page policy action and checks below make no
LLM calls. Optional generated chat needs a separately configured provider/model; it is not part
of this simulation's startup or smoke checks.

## Run the real-service smoke check

```bash
docker compose -f compose.rag.yml --profile checks run --rm --no-deps smoke
```

The check runs inside the obs-agent image and uses its production `policy_docs.retrieve` client
against the real RAG service. It verifies obs-agent health and inventory access, tagging-policy
source/section attribution, indexed corpus identity/current source freshness, an empty retrieval,
and rejection of a wrong bearer key. It uses no provider key and records no approvals.
Start the stack first and keep `--no-deps` on the check commands: Compose must not recreate the
API container while the other services still share its network namespace.
The `local-rag` CI job builds this same stack, runs the normal check, stops RAG and runs the outage
check, then removes the disposable containers/volumes. It records the obs-agent revision and pinned
RAG source. Normal unit tests remain independent of Docker and the second repository.

## Opt in to Jaeger tracing

The default stack leaves tracing off. To inspect retrieval, stop the existing stack without
removing volumes, then start it with both files (the full cycle preserves the shared network):

```bash
docker compose -f compose.rag.yml down
docker compose -f compose.rag.yml -f compose.telemetry.yml up -d --build --wait --wait-timeout 600
docker compose -f compose.rag.yml -f compose.telemetry.yml --profile checks run --rm --no-deps --entrypoint python smoke /checks/telemetry.py
```

Open <http://127.0.0.1:16686> and select **obs-agent-policy**. The check makes one request through
obs-agent's production client and verifies a new `obs.policy.retrieve` → `rag.question` → `rag.retrieve` trace, parentage,
tagging-policy source and distances, and absence of question/excerpt text and credentials.
Avoid other policy requests during the check. No generation span or paid provider call is expected.
The image installs the optional HTTP exporter; a `RAG_REPO_PATH` override must include
`requirements-otel.txt` and the tracing implementation.

Keep both `-f` arguments on commands while tracing is enabled. Test collector failure and recovery:

```bash
docker compose -f compose.rag.yml -f compose.telemetry.yml stop jaeger
docker compose -f compose.rag.yml -f compose.telemetry.yml --profile checks run --rm --no-deps smoke
docker compose -f compose.rag.yml -f compose.telemetry.yml start jaeger
docker compose -f compose.rag.yml -f compose.telemetry.yml --profile checks run --rm --no-deps --entrypoint python smoke /checks/telemetry.py
```

Retrieval must still work during the collector outage. Recovery verifies a fresh trace; buffered
spans can be dropped and replay is not guaranteed. CI runs these same checks. Jaeger keeps traces
in memory, separately from saved reviews. Its UI binds to loopback; OTLP stays inside the Compose
network at `http://jaeger:4318/v1/traces`. Exported metadata includes source paths, distances and
configured service information. Use only the bundled fictional corpus in this simulation.
`OBS_JAEGER_UI_PORT` changes the UI port; `OBS_RAG_TRACE_SERVICE` and `OBS_POLICY_TRACE_SERVICE`
change the distinct provider and caller service names.

To turn tracing off, run `docker compose -f compose.rag.yml -f compose.telemetry.yml down`, then
start with `compose.rag.yml` alone. Do not add `-v` if you want to keep reviews and the index.
See the [RAG telemetry guide](https://github.com/redbarron23/rag-observability/blob/main/docs/telemetry.md)
for span interpretation. Cross-repository trace propagation is coordinated in
[#112](https://github.com/redbarron23/obs-agent/issues/112).

### Sampling, storage and local access

For agent tracing alongside Jaeger, read the
[MLflow/Jaeger mode and content-capture guide](mlflow.md#7-choose-mlflow-jaeger-or-both).
Enabling metadata-only policy traces does not enable MLflow content capture.

The telemetry override uses `parentbased_always_on`: new root traces are recorded, while a valid
parent's sampled flag is respected. Caller and RAG services receive the same standard SDK settings.
An unsampled caller still propagates trace context, so RAG does not start recording its children
independently. Sampling chooses what to record; Jaeger's storage policy chooses what to retain.

| Experiment | `OTEL_TRACES_SAMPLER` | `OTEL_TRACES_SAMPLER_ARG` |
| --- | --- | --- |
| Small local exercise (default) | `parentbased_always_on` | `1.0` (unused) |
| Export no spans, including sampled-parent requests | `always_off` | `1.0` (unused) |
| Sample about 10% of new roots; respect existing parent decisions | `parentbased_traceidratio` | `0.1` |

For the ratio experiment, set these in the shell before starting the stack:

```bash
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
docker compose -f compose.rag.yml -f compose.telemetry.yml up -d --build --wait --wait-timeout 600
```

Compose recreates services whose environment changed; SDK configuration is read once per process.
Ratio sampling is not a promise of exactly one trace per ten requests. A sampled parent keeps its
children even at ratio zero; `always_off` drops spans regardless of the parent. Unset both variables
to restore the default, then repeat the startup command. The exact-trace telemetry smoke check
requires the default sampling settings; a sampled-out trace is intentionally absent, not a
retrieval failure.

Jaeger 2.22.0 uses its embedded all-in-one configuration with one disposable in-memory store,
capped at **100,000 traces**; older traces are evicted as the store fills. The Compose override
does not replace this configuration. A trace-count limit is not a byte limit: span count and
attribute sizes also affect memory. No persistent trace volume is configured, and stopping,
restarting or recreating Jaeger loses its
traces. Saved reviews, the RAG index and model cache live in separate volumes and survive.
See the [pinned embedded configuration](https://github.com/jaegertracing/jaeger/blob/v2.22.0/cmd/jaeger/internal/all-in-one.yaml)
and [memory storage documentation](https://www.jaegertracing.io/docs/2.22/storage/memory/).

Local policy: use fictional data, limit an exercise to 1,000 requests or one hour (whichever comes
first), and clear trace state at the end. This is an operator cleanup policy, not an automatic TTL.
The memory backend does not provide a time-based retention guarantee; do not use this setup as a
long-lived archive. For larger workloads, establish a memory budget and storage/retention policy;
[#117](https://github.com/redbarron23/obs-agent/issues/117) tracks measured overhead and backpressure.

Inspect effective sampler settings, Jaeger command arguments and host bindings without starting
services (the output also contains the stack's public demo keys):

```bash
docker compose -f compose.rag.yml -f compose.telemetry.yml --profile checks config
```

Only the Jaeger UI is published, at `127.0.0.1:16686` by default. OTLP remains on the internal
Compose network; no ingestion port is published to the host. Loopback is a local development
boundary, not user authentication. Exported source paths and service metadata can identify private
documents even without their contents; private use needs an approved access and storage policy.

To clear only trace state while keeping the application and review/index/cache volumes:

```bash
docker compose -f compose.rag.yml -f compose.telemetry.yml restart jaeger
```

This discards in-memory traces, but ongoing application traffic or queued exports can immediately
create new ones. For a whole-stack shutdown, use both Compose files with `down` **without `-v`**.
Never remove application volumes just to clear traces.

### Measure caller export backpressure

Run the controlled check without Docker, an embedding model or an LLM:

```bash
uv run --group telemetry python -m scripts.telemetry_load_check > telemetry-load-results.json
```

It runs tracing-disabled, healthy-collector and slow-collector scenarios in separate processes.
Each sends the same fictional question through the production policy client 256 times with four
workers, after nine warm-up requests. Fixture retrieval injects a fixed **40 ms** delay in every
mode; the local OTLP/HTTP sink injects **1 second per batch** only in slow mode, exceeding the
**0.5-second** exporter deadline. Nothing is exported to a user's configured endpoint: each worker
clears inherited OTLP configuration and binds its own sink on a random loopback port.

The JSON report records Python/SDK versions, platform, request counts, p50/p95/max request latency,
elapsed time, observed queue occupancy, accepted/undelivered spans, flush and shutdown durations,
and fresh-span recovery. It excludes questions, documents, credentials and exception text.
Warm-up spans are flushed before counters reset. Slow mode additionally primes export with 16
requests, checks one original application exception, and generates 80 synthetic spans to test
shutdown with a stalled exporter and a nonempty queue. These phases are reported separately from
the timed 256-request workload.

Pass criteria are explicit: identical retrieval evidence (apart from freshness check timestamps),
every timed request below five seconds, workload below 60 seconds, flush/shutdown each below five
seconds, queue capacity 64, and batches at most 16 spans. Healthy mode must deliver every generated
span; disabled mode must issue no export requests. The fixture's slow mode must fill the queue,
preserve the original application exception and deliver exactly one new span after recovery.
Each worker has an outer **90-second process deadline**, including initialization and cleanup.
These are generous controlled-check availability budgets, not production latency objectives or
an assertion that tracing has zero overhead.

Run against the existing **fictional** combined-stack corpus after starting the telemetry stack:

```bash
docker compose -f compose.rag.yml -f compose.telemetry.yml --profile checks run --rm --no-deps --entrypoint python smoke /checks/telemetry-load.py --live-rag > telemetry-stack-load-results.json
```

This reuses the smoke image and the production HTTP client/provider. It varies only the smoke
caller's tracing; the RAG service keeps its configured tracing throughout. The sink confirms OTLP
receipt, not Jaeger indexing; the existing distributed-trace smoke check verifies real Jaeger
storage and parentage separately. The fixture check runs in the Python 3.12 CI job; the real-service
version runs in the existing isolated combined-stack job. CI retains their sanitized JSON artifacts
for seven days. The check never stops a user's collector or removes review/index/cache volumes.

### SDK queue, timeout and shutdown limits

The caller uses `BatchSpanProcessor` and the OTLP HTTP/protobuf exporter. The pinned RAG provider
uses the same SDK pattern, but its workload/memory bounds are not measured by the caller probe.
The telemetry override forwards these standard settings to API, UI, smoke and RAG:

| Setting | Stack default | Controlled check |
| --- | --- | --- |
| `OTEL_BSP_MAX_QUEUE_SIZE` | 2048 spans | 64 spans |
| `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | 512 spans | 16 spans |
| `OTEL_BSP_SCHEDULE_DELAY` | 5000 ms | 20 ms |
| `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` | 2 seconds | 0.5 seconds |

Set queue and batch sizes together: the batch size must not exceed the queue capacity. Restart or
recreate application processes after changing settings. On overflow, the SDK drops older queued
spans instead of blocking application requests. One batch can be in flight outside the queue;
this is a span-count bound, not a byte/RSS limit. Actual memory also depends on concurrent active
spans, attribute sizes, serialization and the exporter/HTTP stack.

In locked SDK **1.45.2**, `OTEL_BSP_EXPORT_TIMEOUT` is read but does not bound calls to the exporter,
and `force_flush(timeout_millis=...)` does not enforce its supplied timeout. The OTLP HTTP exporter
has its own deadline; do not confuse those controls. Provider shutdown uses the SDK's default
30-second worker-join allowance, not an `OTEL_BSP_EXPORT_TIMEOUT` shutdown setting. The experiment
measures actual flush/shutdown times and kills a worker exceeding its outer deadline rather than
assuming those APIs enforce a shorter deadline. The queue probe inspects locked SDK internals and
must be revisited on SDK upgrades. See the
[locked batch processor](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.2/opentelemetry-sdk/src/opentelemetry/sdk/_shared_internal/__init__.py).

`undelivered_spans` combines queue eviction and failed/timed-out exports; it does not assign an
exact cause to each missing span. Recovery proves export of a fresh span, not replay of old batches.
These short experiments do not certify RAG throughput, production memory budgets, long-running
process leaks, model latency or general network/DNS failure behavior.

Observed fixture run on 2026-11-07: macOS 26.6.2 arm64, Python 3.12.15, SDK/exporter 1.45.2,
working tree based on `915f59c`. The fixed 40 ms retrieval delay is included in the timings below.
All 256 requests per mode returned unchanged evidence, and the original exception was preserved.

| Caller mode | p95 request latency | Observed queue occupancy | Accepted / generated spans | Shutdown |
| --- | --- | --- | --- | --- |
| Disabled | 45.49 ms | 0 | 0 / 0 | No provider |
| Healthy sink | 46.08 ms | 4 / 64 | 257 / 257 | 0.0008 s |
| Slow sink | 46.18 ms | 64 / 64 | 0 / 273 | 2.51 s with 64 queued spans |

Slow-mode flush took 2.14 seconds; recovery exported exactly one fresh span. The 273 undelivered
spans are from the main/priming/error phases, before recovery and the separate 80-span shutdown
probe. Differences in a single short run are observations, not statistically established overhead
or a real embedding/provider benchmark. CI artifacts retain subsequent runs with their environment.

The real combined-stack run on the same date also passed: pinned RAG `19ebcdda`, Linux aarch64
under Colima, Python 3.12.16 and caller SDK/exporter 1.45.2. All 256 requests per mode returned
unchanged evidence. Disabled/healthy/slow caller p95 latencies were 107.22/62.63/54.86 ms; the
healthy sink received all 257 spans, and slow mode reached 64 queued spans, flushed in 2.44 seconds
and shut down in 2.52 seconds with 64 spans queued. Exactly one fresh span arrived after recovery.
The ordinary Jaeger smoke check separately verified the three linked caller/question/retrieve
spans. These modes ran sequentially against a warmed service, so the lower later timings do not
show that tracing makes retrieval faster. Provider tracing remained enabled throughout.

## Check review persistence automatically

From the checkout, run:

```bash
python3 scripts/review_persistence_check.py
```

This needs Docker Compose 2.24.4 or newer and Python 3.11+. It builds only the API service
from this checkout, imports the bundled synthetic CSV through the production CLI, and saves
accept, reject and override decisions with a test reviewer. It captures their snapshot/model
context and reviewed change plan, runs Compose `down`/`up` without deleting volumes, and verifies
that a new API container reads exactly the same decisions and approval evidence. Missing or
altered decisions fail the check; the rejected resource must not appear as an exported change.

The check uses a generated project/image name, no published ports and its own review volume.
It does not stop or change your running simulation. No RAG download, cloud credentials or LLM
are needed. On success or failure it prints service diagnostics and removes its disposable
containers, volumes and image. Command/startup waits are bounded. The `review-persistence` CI job
runs the same command; this checks storage across container recreation. The separate
`browser-review` job checks browser editing, reopening and exports through Chromium.

## Simulate a RAG outage

```bash
docker compose -f compose.rag.yml stop rag
docker compose -f compose.rag.yml --profile checks run --rm --no-deps smoke --rag-down
```

The UI and obs-agent API remain running. Press **Show policy context** again: it should report that
documentation is unavailable while ordinary review and export still work. Existing displayed
excerpts may remain until you request fresh context. The outage check verifies that the inventory
is still accessible and the production RAG client reports a connection failure/timeout.

Restore retrieval, then repeat the normal smoke check:

```bash
docker compose -f compose.rag.yml start rag
docker compose -f compose.rag.yml logs -f rag
# Once RAG reports it is serving, leave the log view with Ctrl-C:
docker compose -f compose.rag.yml --profile checks run --rm --no-deps smoke
```

## Networking and state

The API container owns one network namespace; RAG and the UI join it, so obs-agent reaches RAG at
`http://127.0.0.1:8765`. This preserves the application's loopback-only HTTP rule and uses RAG's
normal bearer authentication. Ports for all three services are published on the API container;
stopping only RAG therefore leaves the review UI/API reachable. Recreating the API container
requires recreating its namespace-sharing services together with a full `down`/`up` cycle.

The project name is `obs-rag-local`, with separate `review-state`, `rag-state` and `model-cache`
volumes. RAG ingests on startup and reuses unchanged embeddings; model downloads are cached.
No local working-copy inventory or index is mounted into either application.

```bash
# Stop the simulation while retaining reviews, index and model cache:
docker compose -f compose.rag.yml down

# Optional disposable reset: deletes this simulation's reviews, index and model cache:
docker compose -f compose.rag.yml down -v
```

For failures, use `docker compose -f compose.rag.yml logs api ui rag`. If startup fails during a
download, correct the network access and start again. Kubernetes and paid generation evaluations
are separate follow-ups. The pinned and weekly/manual provider jobs share the contract, freshness and outage checks
described below.

## Contract, stale-source and latest-provider checks

The pinned PR job and the separately named **RAG latest-provider compatibility** workflow call
one reusable stack job. Both run real retrieval/authentication, a controlled contract break,
stale-source propagation/restoration, RAG outage and optional Jaeger recovery checks. Every run
owns a disposable Compose project. The default provider pin is never advanced automatically.

The contract check forwards a real provider response over a temporary loopback HTTP server and
removes its required `hits` field. The production client must return `invalid_response` with no
accepted hits. The unchanged normal smoke command must exit nonzero on that response, while
obs-agent health/inventory still respond. Run against your already started fictional stack:

```bash
docker compose -f compose.rag.yml --profile checks run --rm --no-deps smoke --contract-break
```

For freshness, CI edits only the disposable container's tagging-standard source after indexing.
`smoke --stale` waits up to 20 seconds for the provider's brief freshness cache, then requires
`stale` on an otherwise successful retrieval. It restores the exact original bytes and verifies
`current` again. No checkout or user corpus is edited. A failed job still removes its project and
volumes. The mutation/restoration commands live in `.github/workflows/rag-stack.yml`; run them
only in a disposable simulation, since a failed interrupted manual check can leave its source changed.

The latest-provider workflow runs weekly on Monday at 07:00 UTC and accepts manual dispatch.
A blank `provider_sha` resolves the provider's default branch once; an explicit value must be a
full 40-character SHA. The job summary and logs record consumer and exact provider revisions.
Its result is separate from the required pinned PR job; regressions fail visibly.

```bash
gh workflow run rag-latest.yml --repo redbarron23/obs-agent
# Reproduce a failing provider revision:
gh workflow run rag-latest.yml --repo redbarron23/obs-agent -f provider_sha=<full-sha>
# Locally, in a separate checkout/project with no other stack on the fixed ports:
RAG_REPO_PATH=https://github.com/redbarron23/rag-observability.git#<full-sha> COMPOSE_PROJECT_NAME=obs-rag-repro docker compose -f compose.rag.yml up -d --build --wait --wait-timeout 600
COMPOSE_PROJECT_NAME=obs-rag-repro docker compose -f compose.rag.yml --profile checks run --rm --no-deps smoke
COMPOSE_PROJECT_NAME=obs-rag-repro docker compose -f compose.rag.yml down -v
```

Replace `<full-sha>` before running. A build/model-download/startup failure occurs before the
smoke checks; inspect those step logs and service health first. A contract failure occurs after
startup and reports `invalid_response`, missing source/provenance or an unexpected outcome.
A newer provider failure should be reproduced at its recorded SHA, fixed or reported upstream,
and checked again. Promote a passing SHA by editing the Compose pin in a normal PR, reviewing
the provider's contract policy and passing the pinned suite. Never bypass a failing compatibility
check by treating incompatible data as an empty retrieval.

## Follow one policy request across services

With `compose.telemetry.yml`, API, UI and smoke clients export metadata-only caller spans and send
W3C `traceparent`/`tracestate` to RAG. The provider validates context after authentication and joins
the same trace. Select **obs-agent-policy** in Jaeger or open the exact ID printed by the smoke
check. You should see the caller with a RAG question child and retrieval grandchild, served by
**obs-rag-retrieval**. No generation is requested by this workflow.

The caller records only outcome, HTTP status and duration. It never records question, excerpt,
reviewer, inventory, auth header, endpoint URL or exception text. Only W3C trace context is
propagated; baggage is excluded. Trace headers cannot authenticate a request, approve a label,
or enter saved approval records/exports. Invalid context starts an independent provider trace.

The caller duration includes network waiting and the provider's work. The question includes
retrieval. Do not add all three as separate elapsed time. A caller timeout, RAG outage or rejected
bearer key can leave only the caller span; inspect its `rag.client.outcome` and HTTP status first.
Collector problems leave ordinary retrieval/review results unchanged. Tracing remains off in the
base stack, uses a private OTel provider and does not enable MLflow content capture.

For CLI/local development install `uv sync --group telemetry` and set standard OTel exporter
variables before the process starts; `OTEL_TRACES_EXPORTER=none`, an unset value or
`OTEL_SDK_DISABLED=true` disables these spans. Only OTLP HTTP/protobuf and console are supported.
The Docker image includes the optional exporter but activates it only through explicit configuration.
Configuration is cached per process; restart after changes. The
[RAG learning guide](https://github.com/redbarron23/rag-observability/blob/main/docs/telemetry.md#incoming-distributed-trace-context)
explains parentage, absent children, sampling and interpretation limits.
