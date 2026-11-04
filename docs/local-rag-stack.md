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
runs the same command; this checks storage across container recreation, while browser editing
is tracked separately in [#107](https://github.com/redbarron23/obs-agent/issues/107).

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
are separate follow-ups. This stack and its pinned CI job support
[#102](https://github.com/redbarron23/obs-agent/issues/102); scheduled checks against future RAG
revisions and contract-upgrade scenarios can extend it later.

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
