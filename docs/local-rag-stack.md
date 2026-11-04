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

By default the RAG build uses the pinned Git source `fb7ee91` (retrieval contract 1.1.0). obs-agent
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

Open <http://127.0.0.1:16686> and select **obs-rag-retrieval**. The check makes one request through
obs-agent's production client and verifies a new `rag.question` → `rag.retrieve` trace, parentage,
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
`OBS_JAEGER_UI_PORT` changes the UI port; `OBS_RAG_TRACE_SERVICE` changes the service name.

To turn tracing off, run `docker compose -f compose.rag.yml -f compose.telemetry.yml down`, then
start with `compose.rag.yml` alone. Do not add `-v` if you want to keep reviews and the index.
See the [RAG telemetry guide](https://github.com/redbarron23/rag-observability/blob/main/docs/telemetry.md)
for span interpretation. Cross-repository trace propagation is tracked separately in
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
