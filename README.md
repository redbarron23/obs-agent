# obs-agent — Tag Inference & Review for Multi-Cloud Cost Estates

[![CI](https://github.com/redbarron23/obs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/redbarron23/obs-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Live demo](https://img.shields.io/badge/live%20demo-obs--agent.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://obs-agent.streamlit.app/Label_review)

**About half of a typical cloud estate has missing or wrong cost tags, so nobody can say which
team owns the spend.** obs-agent infers the missing `team` / `env` / `cost_center` tags across
Azure subscriptions and GCP projects, flags the wrong ones, puts a human in the loop to approve
the changes, and exports them ready to apply as `az` / `gcloud` commands or Terraform.

**[Try the live demo](https://obs-agent.streamlit.app/Label_review)**: no install or API key,
synthetic data, and your review decisions stay private to your browser session. A sleeping app
takes about 30 seconds to wake up.

![Reviewing inferred tags and exporting them](docs/label-review.gif)

| On a synthetic estate of 320 subscriptions / projects, scored against ground truth | |
|---|---|
| Untagged `team` labels inferred correctly | **98%** (macro-F1 0.98) |
| Precision of the suggestions confident enough to auto-apply (≥90%) | **100%** team · **99%** env |
| Wrong existing tags caught by the mislabel check | 11 of 16, with 11 of 13 flags correct |

These are an upper bound: the synthetic owners map more neatly onto teams than a real estate's
would ([details](#how-it-works)).

- **Tag inference** (the core): cleaning, a rules baseline and a scikit-learn model over names,
  folders / management groups, owners, services and spend shape — with a reason and a confidence
  for every suggestion.
- **Human in the loop**: a review queue in the web UI and REST API; accept, reject or override;
  export as an `az` / `gcloud` script, Terraform tfvars or CSV.
- **Second feature — cost Q&A agent**: *"Which Azure subscription has the highest overage?"*
  answered by tool calls over billing data. See [docs/agent.md](docs/agent.md).
- **Engineering**: ground-truth eval harness, 200+ tests, CI on Python 3.10–3.12,
  FastAPI + Streamlit, Docker Compose, uv.

## The problem

Tag inference is the whole point of this project, so it is worth being precise about the problem
it solves.

Cloud estates are full of subscriptions and projects that were created before a tagging policy
existed. Their `team` / `env` tags are missing, spelled three different ways, or plainly wrong
after a reorg moved a folder. The result is that finance can see a large bill but cannot
attribute it: nobody knows whether `svc-ingest` belongs to Data Platform or Observability, so
nobody can charge it back.

Re-tagging by hand does not scale — a few hundred resources, each needing someone who remembers
its history. So `labeling.py` treats it as a supervised learning problem instead: **train on the
resources that *are* correctly tagged, predict the ones that are not**, and hand the reviewable
suggestions to a human.

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/redbarron23/obs-agent.git && cd obs-agent
uv sync                                     # install uv: https://docs.astral.sh/uv/

# 2. Score the inference strategies against ground truth
uv run labeling.py

# 3. Review and export the suggestions in the web UI
uv run streamlit run app.py                 # then open the "Label review" page

# 4. (Second feature) ask the cost agent a question — needs Ollama or a provider key
uv run agent.py -q "Which Azure subscription has the highest overage?"
```

Steps 2 and 3 need no API key and no LLM: label inference is plain scikit-learn, and the review
page is a UI over its output. The agent in step 4 is the optional second feature
([docs/agent.md](docs/agent.md)).

## How it works

Typical estates have ~half their Azure subscriptions and GCP projects missing a usable
`team` / `env` tag. `labeling.py` treats this as supervised learning: train on the correctly
tagged resources, predict the rest.

1. **Clean** — normalise spellings (`Prod`, `prd` → `prod`; `o11y` → `observability`). This alone
   recovers ~6% of "unlabelled" resources.
2. **Rules baseline** — inherit team from the parent management group / folder or name prefix
   when ≥80% of labelled siblings agree; env from name tokens (`-prd-`, `-uat-`, `sbx-`).
3. **Model** — logistic regression over name tokens, parent, owners, creator, services and spend
   shape (weekend/weekday ratio, volatility). Every prediction comes with a confidence and the
   features that drove it.
4. **Mislabel check** — out-of-fold predictions flag existing tags the model confidently disagrees
   with.

The synthetic inventory (`inventory.py`, 320 resources) keeps the *true* labels next to the
observed ones, so every strategy is scored against ground truth (`uv run labeling.py`):

| Label | Strategy | Coverage | Accuracy | Macro-F1 | Auto-apply (≥0.9) | Auto-apply precision |
|---|---|---|---|---|---|---|
| team | rules | 41% | 94% | 0.81 | 32% | 94% |
| team | **model** | 100% | 98% | 0.98 | 37% | 100% |
| env | rules | 55% | 100% | 1.00 | 55% | 100% |
| env | **hybrid** | 100% | 81% | 0.79 | 58% | 99% |

Takeaways: rules are precise but cover under half, and parent-folder inheritance is *confidently
wrong* after reorgs, so team uses the model alone. Env is easy when the name says so and hard when
it doesn't (spend alone can't separate dev from staging), so it uses whichever of rule and model is
more confident. The mislabel check caught 6/10 wrong team tags and 5/6 wrong env tags, with 6/6
and 5/7 of its flags correct.

Two agent tools expose this: `get_label_coverage` and `suggest_labels`:

```
Label suggestions for svc-ingest [azure, parent mg-security, owners hana]:
  team untagged (raw value: none)
    → suggest team=data-platform (60% confidence, needs review; via model: strongest signals: creator=hana (+1.27), owner=hana (+0.86), svc=DataFactory (+0.53))
    alternatives: observability 12%, security 8%
    → implies cost_center=CC-2002
```

The synthetic data is cleaner than a real estate (owners map neatly onto teams), so treat these
numbers as an upper bound. On real data, untagged resources skew older, so measure on a reviewed
sample of them and not only on held-out tagged ones.

## Review & export

Predictions are only useful once someone applies them. The **Label review** page in the web UI,
and the matching REST endpoints, turn suggestions into tag changes you can apply:

1. **Review queue**: every untagged or suspect label, most confident first, filterable by label,
   status, cloud, confidence and name.
2. **Decide**: per row choose `accept` (apply the suggestion), `reject` (keep today's tag) or
   `override` (pick the right value). "Accept all auto-apply" handles the ≥90% suggestions in one
   click.
3. **Export**: download the resolved changes as CSV, as a shell script of `az tag update` /
   `gcloud projects update --update-labels` commands, or as a Terraform `labels.auto.tfvars.json`
   map. Overrides win over accepts, accepts over auto-apply; rejected suggestions are never
   exported, and `cost_center` follows the team.

Decisions are stored in a SQLite database (`OBS_AGENT_DECISIONS`, default `label_decisions.db`,
git-ignored). The API and UI share it safely across processes, so a decision made in one shows up
in the other; in Docker Compose both containers mount the `label-state` volume. On first use, a
legacy sibling `label_decisions.json` file is imported automatically.

The same data over HTTP, with no LLM call:

```bash
curl localhost:8000/labels/coverage
curl "localhost:8000/resources?label=team&status=suspect"
curl localhost:8000/resources/svc-ingest/labels
curl -X PUT localhost:8000/resources/svc-ingest/labels/team/decision \
     -H 'content-type: application/json' -d '{"decision": "accept"}'
curl "localhost:8000/labels/export?format=sh&include_auto=false" -o apply-labels.sh
```

## Interfaces

### Web UI (Streamlit)

```bash
uv run streamlit run app.py
```

The sidebar has two pages. **Label review** is the main one: the review queue, per-row decisions
and the export buttons described above. The **chat** page drives the cost Q&A agent with
provider/model selection, conversation history and example prompts ([docs/agent.md](docs/agent.md)).

### REST API (FastAPI)

Both the label review and the agent are available as a standalone HTTP service, so other apps
(dashboards, chat bots, pipelines) can call them without importing any Python:

```bash
uv run uvicorn api:app --reload          # http://localhost:8000/docs for interactive OpenAPI docs
```

Label endpoints — the ones the review UI is built on, and the ones that need no LLM:

| Endpoint | Purpose |
|---|---|
| `GET /labels/coverage` | Untagged / suspect / auto-apply-ready counts per label |
| `GET /resources` | Review queue; filters `label`, `status`, `cloud`, `min_confidence`, `max_confidence`, `search`, plus `limit` / `offset` |
| `GET /resources/{id}/labels` | Current tags, suggestions, confidence, reasons and alternatives for one resource |
| `PUT` / `DELETE /resources/{id}/labels/{label}/decision` | Record (`accept` / `reject` / `override` + `value`) or undo a review decision |
| `GET /labels/decisions` | All recorded decisions |
| `GET /labels/export?format=csv\|sh\|tfvars` | Resolved changes as a download; `include_auto=false` exports reviewed decisions only |

Agent endpoints (see [docs/agent.md](docs/agent.md) for request/response detail and the SSE
streaming format):

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question", "provider"?, "model"?, "session_id"?}` → `{"answer", "session_id", "tools_called", ...}` |
| `POST /ask/stream` | Same request, answer streamed as server-sent events (`token` events, then `done` with `tools_called`, or `error`) |
| `GET /providers` | Supported providers and default models |
| `GET /health` | Liveness probe (always open) |
| `DELETE /sessions/{id}` | Forget a conversation |

```bash
curl -X PUT localhost:8000/resources/svc-ingest/labels/team/decision \
  -H 'content-type: application/json' -d '{"decision": "accept"}'

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which Azure subscription has the highest overage?", "session_id": "demo"}'
```

#### Errors

| Status | Meaning |
|---|---|
| `401` | Missing or wrong `X-API-Key` (only when `OBS_AGENT_API_KEY` is set) |
| `404` | `DELETE /sessions/{id}` for an unknown session |
| `422` | Invalid body (empty question, over 4000 chars) or unknown provider |
| `502` | The LLM backend failed: bad key, no credits, Ollama not running |

All error bodies are `{"detail": "..."}`. Interactive docs with schemas and examples are at `/docs`
(Swagger UI) and `/redoc`; the raw spec is at `/openapi.json`.

#### Running it in production

- **Sessions are in memory** (LRU, 200 max) and lost on restart. Run a single replica, or put
  sticky sessions (by `session_id`) in front. For multiple replicas, move `SessionStore` to Redis.
- **Set `OBS_AGENT_API_KEY`** whenever the port is reachable by anything but you. `/health` stays
  open for load-balancer probes.
- **Behind a gateway** (AWS API Gateway, Azure APIM, nginx): the service is plain HTTP/JSON plus
  SSE. Disable response buffering and use a long idle timeout on `/ask/stream`, or tokens arrive
  in one burst. Local models can take a minute for a multi-tool question.
- **Provider keys** go in environment variables (`.env` with Compose, a secret store elsewhere),
  never in requests.
- **Request size and cost:** questions are capped at 4000 characters, but there is no built-in
  rate limit. Add one at the gateway.

### Docker

```bash
cp .env.example .env              # add keys for hosted providers (optional)
ollama serve                      # the cost agent's default provider runs on the host
docker compose up --build         # API on :8000, Streamlit UI on :8501
```

Containers reach the host's Ollama through `host.docker.internal` (override with `OLLAMA_HOST`).
The image installs dependencies with uv, runs as a non-root user, and the API has a health check.

### Hosted demo (Streamlit Community Cloud)

Live at **https://obs-agent.streamlit.app** (Label review: [/Label_review](https://obs-agent.streamlit.app/Label_review)).
The Label review page needs no LLM, so it can be hosted for free:

1. Open [share.streamlit.io](https://share.streamlit.io) → **Create app** → repository
   `redbarron23/obs-agent`, branch `main`, main file `app.py`.
2. **Advanced settings**: Python **3.12**, and under Secrets add `OBS_AGENT_DEMO = "1"`.
3. Deploy. Dependencies install from `uv.lock`.

With `OBS_AGENT_DEMO=1` each visitor's decisions are private to their browser session and
discarded when it ends, and the chat page is switched off with a pointer to Label review (there
is no LLM in the demo).

## Feature 2: the cost Q&A agent

The agent answers cost questions in plain English over the same billing data, using tool calls —
with Ollama (local, free), DeepSeek, Claude or OpenAI:

![obs-agent answering a cost question via tool calls](docs/demo.gif)

```
"Which Azure subscription has the highest overage cost?"
"Show me the top 3 most expensive GCP projects"
"Compare costs across Azure and GCP"
```

It is fully documented, including the agent loop, the tool set, the synthetic billing data,
streaming, conversation memory, the CLI, provider support and sample conversations, in
**[docs/agent.md](docs/agent.md)**.

## Evaluation

### Label inference scored against ground truth

`uv run labeling.py` prints the strategy comparison table shown in [How it works](#how-it-works).
The inventory keeps true labels beside observed ones, so every strategy — rules, model, hybrid and
the mislabel check — is scored honestly. `tests/test_labeling.py` pins accuracy and precision
floors against that ground truth, so a regression in the model fails CI rather than quietly
changing the README's numbers.

### Agent evals

```bash
# Deterministic dry-run — checks tool outputs directly, no API key needed
uv run evals.py --dry-run

# Live evals against the LLM (needs the provider's API key; default: Ollama)
uv run evals.py
uv run evals.py --provider anthropic --model claude-haiku-4-5-20251001

# Verbose output for every case
uv run evals.py --dry-run -v
```

Dry-run mode calls the tools directly and verifies ground-truth identifiers appear in the output —
fully deterministic, suitable for CI. Live mode runs the full agent loop and checks that the LLM's
answer contains the expected facts.

This runs 12 eval cases with known-correct answers. Covers all seven tools:

- Azure top overage, GCP top project, GCP top 3
- Daily trends (Azure and GCP)
- Spike detection (Azure and GCP)
- Multi-cloud summary
- Label coverage, a team suggestion, a suspected-wrong tag
- `label-count-and-team`: count plus team in one question (the issue #5 regression)

#### Latest results

| Mode | Cases | Passed | Needs API key |
|---|---|---|---|
| Deterministic dry-run (`--dry-run`) | 12 | 12 (100%) | No |
| Live, DeepSeek (`--provider deepseek`, `deepseek-chat`) | 12 | 12 (100%) | Yes |
| Live, Ollama (`--provider ollama`, `qwen2.5:7b`) | 12 | 10 (83%): fails `suspect-label`, `label-count-and-team` | No |

The dry-run result is reproducible and runs in CI on every push. Live results for Claude and OpenAI
are not yet published.

Because LLM output is non-deterministic, evals check for **presence of correct facts**
(`must_contain`) rather than exact-string matching — the right testing strategy for AI systems.

### Tracing and eval runs with MLflow (optional)

MLflow is an optional dependency group, so CI, the Docker image and the hosted demo never install
it. Turn it on with an environment variable:

```bash
uv sync --group mlops
OBS_AGENT_MLFLOW=1 uv run agent.py -q "Which team does svc-ingest belong to?"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

**Tracing.** Every question becomes one trace: an `agent` span, an `llm: <provider>/<model>` span
per model call (stop reason, tool calls, tokens, served-by model), and a span per tool call with its
inputs and output. MLflow's OpenAI autolog adds SDK-level detail inside the `llm` spans for
DeepSeek, OpenAI and Ollama. The `llm` span is the agent's own, because MLflow's Anthropic autolog
does not patch the beta endpoint or `messages.stream` that Claude Opus 5.5 and streaming use. Set
`MLFLOW_TRACKING_URI` to send traces to a remote server instead of the local `mlflow.db`.

**Eval runs.** `evals_mlflow.py` runs the same 12 cases through `mlflow.genai.evaluate`, one MLflow
run per provider and model, with two scorers:

- `facts_present`: the same fact check as `evals.py`
- `tools_called`: reads the trace and checks the agent called every tool the question needs (the
  tools its dry run uses). A model that guesses can pass the fact check by luck, but not this one.

```bash
uv run evals_mlflow.py --provider deepseek
uv run evals_mlflow.py --provider ollama
```

Recorded runs (2026-10-11):

| Run | `facts_present` | `tools_called` | Failed cases |
|---|---|---|---|
| `deepseek/deepseek-chat` | 100% | 100% | none |
| `ollama/qwen2.5:7b` | 83% | 92% | `gcp-top-3` (skipped the tool), `label-count-and-team` (wrong facts) |

The local model fails different cases from run to run, which is why each run is recorded rather than
reported once. The trace check also caught a flaw in the evals themselves: both models answered
`multi-cloud-summary` with two specific tools instead of `compare_cross_cloud`, which is equally
valid, so cases can now list `accepted_tools`.

## Architecture

The tag inference pipeline is the spine of the project: a synthetic inventory with ground truth,
an inference module that produces scored suggestions, and a review layer that turns the ones a
human accepts into applyable tag changes.

```mermaid
flowchart TD
  subgraph datalayer["Synthetic inventory"]
    inv["inventory.py<br/>320 resources<br/>true + observed labels<br/>seeded RNG (7)"]
  end

  subgraph infer["Inference — labeling.py"]
    clean["clean()<br/>normalise spellings"]
    rules["Rules<br/>parent / prefix inheritance<br/>env name tokens"]
    model["Model<br/>logistic regression<br/>+ per-prediction explanation"]
    suspect["suspect_labels()<br/>out-of-fold mislabel check"]
  end

  subgraph reviewlayer["Review & export — label_review.py"]
    queue["review_queue()<br/>untagged + suspect<br/>most confident first"]
    decide["DecisionStore (SQLite)<br/>accept / reject / override"]
    export["export()<br/>csv · az+gcloud sh · tfvars"]
  end

  subgraph surfaces["Interfaces"]
    page["Streamlit<br/>Label review page"]
    api["FastAPI<br/>/labels, /resources"]
    tool["agent tools<br/>get_label_coverage, suggest_labels"]
  end

  inv --> clean
  clean --> rules
  clean --> model
  rules --> queue
  model --> queue
  model --> suspect
  suspect --> queue
  queue --> decide
  decide --> export
  queue --> page
  queue --> api
  decide --> page
  decide --> api
  export --> page
  export --> api
  model --> tool
  suspect --> tool

  classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
  classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
  classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
  classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
  class clean,rules,model,suspect toneAmber
  class queue,decide,export toneMint
  class page,api,tool toneBlue
  class inv toneRose
```

### The inference pipeline

| Module | Role |
|---|---|
| `inventory.py` | Synthetic estate with true *and* observed labels, so inference can be scored against ground truth |
| `labeling.py` | Cleaning, rules baseline, logistic-regression model, out-of-fold mislabel check, evaluation |
| `label_review.py` | Review reports, decision store, and CSV / `az`+`gcloud` / tfvars export |

`BEST_STRATEGY` in `labeling.py` picks rules for `env` and the model for `team`, and the reason is
written next to it: parent-folder inheritance is precise but confidently wrong after reorgs, so
`team` is left to the model, while `env` name tokens are near-perfect. That choice is a result, not
a default — the eval table is what justifies it.

The agent side of the repo (`agent.py`, `tools.py`, `data.py`, `app.py`, `api.py`) is documented in
[docs/agent.md](docs/agent.md), including its architecture diagram.

### File layout

```
obs-agent/
├── labeling.py           # Tag inference: cleaning, rules, model, mislabel check, evaluation
├── inventory.py          # Synthetic inventory with true vs observed team/env labels
├── label_review.py       # Review reports, decision store, CSV / az+gcloud / tfvars export
├── pages/
│   └── 1_Label_review.py # Streamlit page: review queue, decisions, exports
├── agent.py              # Agent loop + CLI (REPL / --question) + streaming + provider abstraction
├── tools.py              # 7 tool functions + LLM tool definitions + dispatch table
├── data.py               # Deterministic synthetic billing data generator (seeded RNG, no CSVs)
├── app.py                # Streamlit web UI (chat page + host for the Label review page)
├── api.py                # FastAPI REST service (/ask, /ask/stream, sessions, label review, API key)
├── evals.py              # 12 eval cases with ground-truth checks against the synthetic data
├── evals_mlflow.py       # Live evals as MLflow runs: fact + trace-based tool scorers
├── observability.py      # Optional MLflow tracing (agent / llm / tool spans); no-op when off
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_agent.py
│   ├── test_api.py
│   ├── test_evals.py
│   ├── test_labeling.py
│   ├── test_label_review.py
│   ├── test_model_config.py
│   └── test_tools.py
├── docs/
│   └── agent.md          # The cost Q&A agent, in full
├── .github/workflows/
│   └── ci.yml            # Unit tests + eval dry-run + Docker build/smoke test
├── Dockerfile
├── docker-compose.yml    # API on :8000 + Streamlit UI on :8501
├── .env.example
├── pyproject.toml        # Dependencies (single source of truth)
├── uv.lock               # Exact locked versions (commit with pyproject.toml)
├── .python-version       # Local Python version for uv
├── LICENSE
└── README.md
```

## Design decisions

### Why a synthetic inventory with true labels?

Real billing exports contain sensitive subscription IDs and project names, and — more importantly
for this project — they do **not** contain ground truth. You cannot score an inference method
against a label you do not have. Keeping a *true* label beside every *observed* one means the
rules, the model and the mislabel check can all be measured honestly, and the eval is fully
reproducible. The same reasoning applies to the agent's billing data ([docs/agent.md](docs/agent.md)).

### Why fact-presence checks instead of exact matching?

Most software lets you write deterministic tests. Agent behaviour is non-deterministic — the same
question can get different phrasings. The eval harness in this project shows the right testing
strategy for this world: check that *correct facts* appear in the output, not that the exact
wording matches.

### Why a provider abstraction?

Building API-agnostic agents is a practical skill for production systems where provider lock-in is
a concern. The `Provider` class in `agent.py` shows how to switch between Anthropic and
OpenAI-compatible backends without changing the tool logic or agent loop.

## What this project demonstrates

| Concept | Implementation |
|---|---|
| **Supervised inference** | Train on correctly tagged resources, predict the untagged ones |
| **Leakage-free evaluation** | Out-of-fold predictions so a resource never votes on itself |
| **Honest metrics** | Ground-truth inventory, pinned accuracy/precision floors in CI |
| **Explainable predictions** | Per-suggestion confidence, driving features and alternatives |
| **Human in the loop** | Review queue, accept/reject/override, resolution order in one place |
| **Applyable output** | `az` / `gcloud` script and Terraform tfvars, not just a CSV |
| **Agent loop** | Provider-agnostic request → dispatch → feed-back cycle until `end_turn` |
| **Streaming** | Token-by-token output through the full tool-use loop (Anthropic + OpenAI SSE) |
| **Conversation memory** | Multi-turn REPL with history pruning at 20 messages |
| **Provider abstraction** | `Provider` class wrapping Anthropic and OpenAI-compatible APIs |
| **Tool definitions** | JSON Schema input specs that the model calls dynamically |
| **Evals** | Fact-presence checks (`must_contain`) instead of brittle exact-string matching |
| **Deterministic test data** | Seeded RNG for reproducible dev, test, and CI |

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for Python, environments and dependencies (used locally, in CI
  and in the Dockerfile)
- **Label inference and review**: no LLM, no API key, no extra services
- The cost agent (optional) additionally needs **Ollama** (default, no key) or a
  **DeepSeek / OpenAI / Anthropic** key: `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

### Dependencies

`pyproject.toml` is the single source of truth and `uv.lock` pins exact versions; both are
committed. Runtime deps are under `[project] dependencies`, test tools under the `dev` dependency
group (installed by `uv sync`, left out of the Docker image).

```bash
uv sync                      # create/update .venv from uv.lock (Python version from .python-version)
uv run <command>             # run anything inside that env, no activation needed
uv add <package>             # add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>       # add a test/dev dependency
uv lock --upgrade            # bump locked versions within the declared ranges
```

CI runs `uv sync --locked`, which fails if `uv.lock` is out of date with `pyproject.toml`. Commit
both files together.

## Testing

Unit tests and deterministic eval dry-runs:

```bash
uv run pytest tests/ -v

# Skip live LLM evals (default in CI)
uv run pytest tests/ -v -m "not integration"

# Run live evals (requires a configured, reachable provider)
OBS_AGENT_RUN_LIVE_EVALS=1 uv run pytest tests/ -v -m integration
```

CI runs on every push/PR via GitHub Actions: unit tests and the eval dry-run on Python 3.10–3.12,
plus a Docker job that builds the image and smoke-tests `/health`.

### Test strategy

**Tool tests** (`tests/test_tools.py`, 34 tests) patch the module-level DataFrames with small,
hand-crafted fixtures containing known values:

```python
# conftest.py — fixtures with transparent data
@pytest.fixture
def azure_summary():
    return pd.DataFrame(
        [
            {"subscription_id": "sub-a", "estimated_overage_cost_30d": 5000},
            {"subscription_id": "sub-b", "estimated_overage_cost_30d": 2000},
            ...,
        ]
    )


# test_tools.py — verifies logic, not API behaviour
def test_top_1(self, patch_azure_summary):
    result = get_azure_top_overages(n=1)
    assert "sub-a" in result
    assert "sub-b" not in result  # only top 1
```

**Agent tests** (`tests/test_agent.py`, 41 tests) mock the Provider class so no API calls are made:

```python
# Helper builds a fake Anthropic response with known content
resp = make_anthropic_response(
    tool_name="get_azure_top_overages",
    tool_input={"n": 1},
)

with patch("agent.Provider.create", return_value=resp):
    answer, messages = run("Which Azure sub is top?")

assert "sub-a" in answer
```

Both API shapes (Anthropic and OpenAI-compatible) are tested via helper factories that build
correctly-shaped fake responses. The streaming path also has dedicated tests for
tool-call-through-streaming.

### Test groups

| Group | Tests | What's verified |
|---|---|---|
| Label inference (`test_labeling.py`) | 34 | Tag normalisation, rules, model, accuracy/precision floors vs ground truth, label tools |
| Label review (`test_label_review.py`) | 45 | Reports, decision validation and persistence, table-edit diffing, export resolution and formats, label endpoints + auth, review page (headless) |
| Tool logic (`test_tools.py`) | 34 | Top-N ranking, zero filtering, spike thresholds / platform filter / limit, cross-cloud totals, edge cases |
| Agent loop (`test_agent.py`) | 16 | Simple answer, single/multi tool call, conversation memory, history pruning, empty/long questions, safety rails |
| Provider abstraction | 10 | Stop reason mapping + content block parsing for Anthropic and OpenAI shapes |
| Streaming | 7 | Streaming with/without tools, OpenAI-compatible streaming, stop-reason normalisation |
| Message formatting | 3 | Canonical ↔ provider message conversion |
| CLI parsing | 8 | All flags, defaults, combinations |
| REST API (`test_api.py`) | 16 | `/ask`, `/ask/stream` (SSE + errors), sessions, API-key auth, OpenAPI contents |
| Model config (`test_model_config.py`) | 24 | Env defaults and fail-fast, CLI/API/UI wiring, Claude adaptive-thinking request, thinking-block replay, refusal and fallback reporting |
| Evals (`test_evals.py`) | 17 | 12 deterministic dry-run cases, the full dry-run, matcher checks and an opt-in live run |

## References

- [scikit-learn — Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — the inference and out-of-fold mislabel check
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — the macro-F1 and precision metrics used above
- [Anthropic — Tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) and [OpenAI — Function calling](https://platform.openai.com/docs/guides/function-calling) — the two tool-call shapes the provider abstraction unifies
- [Google Cloud — Labeling projects](https://cloud.google.com/resource-manager/docs/creating-managing-labels) and [Azure — Tag resources](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/tag-resources) — the tag semantics the export targets
- [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/) — NVIDIA blog on evaluation patterns
- [Evaluating RAG: A Comprehensive Guide to Metrics and Methods](https://www.rungalileo.io/blog/evaluating-rag-a-comprehensive-guide-to-metrics-and-methods) — Galileo guide covering dual-attribution eval (source + fact checks)
- [LangSmith RAG Evaluation](https://docs.smith.langchain.com/faq/evaluation/eval_rag) — production-grade RAG eval patterns (correctness, faithfulness, relevance)

## License

MIT — see [LICENSE](./LICENSE).

## Related projects

### [rag-observability](https://github.com/redbarron23/rag-observability) — RAG over Observability Docs

A standalone Retrieval-Augmented Generation system that answers natural-language questions about
multi-cloud observability architecture, coverage targets, and monitoring standards. Built on real
domain documentation.

```
Q: What is the coverage target for Tier 1 production resources?
A: The production Tier 1 coverage target is 85% within 4 months and 100%
   within 9 months (source: coverage-targets.md, section 'Tier 1').
```

**Pipeline:** `.md` documents → heading-aware chunking → `all-MiniLM-L6-v2` embeddings → ChromaDB
vector store → cosine similarity retrieval → Claude with cited generation.

**Key differences from obs-agent:**

| Concern | obs-agent | rag-observability |
|---|---|---|
| **Pattern** | Supervised inference + tool-use agent | RAG (retrieval + generation) |
| **Data** | Synthetic inventory + billing DataFrames | Real documentation (.md files) |
| **LLM role** | Optional: answers cost questions via tools | Answers from retrieved context |
| **Memory** | Multi-turn conversation (20 msg history) | Stateless per query |
| **Eval focus** | Ground-truth accuracy + fact presence | Source attribution + fact presence |
| **Providers** | Ollama, DeepSeek, OpenAI, Anthropic | Anthropic, DeepSeek |

**References:**
- [ChromaDB](https://www.trychroma.com/products/chromadb) — vector database
- [RAG 101: Demystifying RAG Pipelines](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/) — NVIDIA
- [Sentence Transformers](https://www.sbert.net/) — embedding models (`all-MiniLM-L6-v2`)
- [Evaluating RAG: Metrics and Methods](https://www.rungalileo.io/blog/evaluating-rag-a-comprehensive-guide-to-metrics-and-methods) — Galileo
- [LangSmith RAG Evaluation](https://docs.smith.langchain.com/faq/evaluation/eval_rag) — production eval patterns
