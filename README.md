# obs-agent — Cloud Label Review

[![CI](https://github.com/redbarron23/obs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/redbarron23/obs-agent/actions/workflows/ci.yml)

**Turn missing or suspect Azure/GCP labels into a reviewed change plan.** Import an inventory,
inspect ownership evidence, accept or override suggestions, and export the before/after changes.
Label review runs locally without an LLM or cloud credentials. Optional cost chat uses synthetic data.

![Label review workflow](docs/label-review.gif)

## Quick start

Requirements: Python **3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/redbarron23/obs-agent.git
cd obs-agent
uv sync --locked
uv run streamlit run app.py
```

Open the local URL printed by Streamlit. **Label review** is the landing page. Enter your reviewer
name or ID, inspect a suggestion, save a decision, and download **Reviewed change plan**.
Start with the bundled synthetic inventory; no API keys or Ollama setup are needed.

For disposable demo decisions, run `OBS_AGENT_DEMO=1 uv run streamlit run app.py`.
See the [five-minute walkthrough](docs/five-minute-demo.md).

## Use your own inventory

1. Open **Import / reopen inventory** and download the CSV and vocabulary examples.
2. Upload your sanitized Azure subscription/GCP project snapshot. If its headers differ from the
   CSV schema, map them on the page. Supply your own team/environment values and aliases in the
   optional vocabulary JSON.
3. Select **Save and open review**, then review suggestions and export a plan.
4. Return to **Import / reopen inventory** to resume a saved snapshot after a browser or app restart.
5. Next cycle, save the new snapshot and open **Compare snapshots** to review only what changed.

To hand work to a colleague, assign a label, add a note, or open a question below the review
table, then filter by **Assigned to me** or **Only unresolved questions**. These never approve a
change ([details](docs/decision-storage.md#assignments-notes-and-questions)).

Snapshots, vocabularies, and approvals persist locally in `review-workspaces/` (configurable with
`OBS_AGENT_REVIEW_DIR`). Keep this private directory outside version control and back it up.
Saved reviews are also available through the REST API's `/reviews` routes and `review_cli.py`
([example](docs/api.md#saved-reviews-api-and-cli)); optional chat uses the synthetic estate.
The public demo disables private imports. See [Import inventory](docs/importing.md).

## What you get

- Missing/suspect `team` and `env` suggestions with evidence, confidence, and alternatives.
- A SQLite decision history with reviewer identity, timestamps, inventory fingerprint, and inference version.
- Reviewed-only JSON change plans and CSV, shell, or Terraform exports. A cost center accompanies
  a team change only when the inventory establishes an unambiguous mapping.
- Explicit opt-in to include unreviewed high-confidence suggestions in convenience exports.

An [example exported plan](fixtures/reviewed_change_plan_example.json) shows the full schema.
For example, an accepted suggestion produces this change:

```json
{
  "resource_id": "gcp-demo-003",
  "cloud": "gcp",
  "tags": {
    "team": {"previous": null, "new": "data-platform", "how": "accept"},
    "cost_center": {"previous": null, "new": "cc-2002", "how": "accept"}
  }
}
```

The complete plan includes the approval evidence and snapshot/version identifiers. Changed inputs
or inference versions require a fresh review. obs-agent exports plans; it never applies cloud changes.
Reviewer identifiers are self-reported locally; authenticate reviewers at your deployment boundary.

## Run with Docker

```bash
cp .env.example .env
# Generate a key, then paste it into OBS_AGENT_API_KEY in .env:
openssl rand -hex 32
docker compose up --build
```

Open `http://localhost:8501` for review and `http://localhost:8000/docs` for the API.
Both bind to loopback. The named volume retains decisions and imported reviews. Ollama is needed
only for optional cost chat; Compose reaches the host automatically. See [Interfaces](docs/api.md).

## How it works

```mermaid
flowchart LR
    I[Inventory + vocabulary] --> M[Rules and label inference]
    M --> R[Human review with evidence]
    R --> D[(SQLite approvals + snapshot/version)]
    D --> P[Reviewed change plan]
```

Label inference uses observed labels and ownership signals; reviewers resolve uncertain suggestions.
Plans bind approvals to their inventory and inference version. Optional cost chat is a separate
synthetic-data workflow. See [Architecture](docs/architecture.md).

## Quality and limits

| Synthetic hold-out result | Value |
| --- | ---: |
| Team-label accuracy | 98% |
| Team-label macro-F1 | 0.98 |
| Team-label accuracy with abstention (97% coverage) | 99% |
| Environment-label accuracy (hybrid) | 81% |

These results use the default team model and environment hybrid strategy on the bundled synthetic
inventory. Environment rules alone score 100% on the 55% of examples they cover; the hybrid covers
100% of examples at 81% accuracy. Where ownership evidence is missing, unsupported or conflicting,
team inference abstains instead of suggesting a value: 3% of untagged resources on this estate, 23%
on the harsh one (where accuracy on the rest rises from 85% to 94%). These are not production
performance claims. Each imported label needs at least two recognized values with two labelled
resources per value. Validate on a representative reviewed sample before relying on confidence
thresholds. See [Evaluation](docs/evaluation.md).

## Documentation

| Topic | Guide |
| --- | --- |
| End-to-end synthetic workflow | [Five-minute demo](docs/five-minute-demo.md) |
| CSV schema, vocabulary, saved reviews | [Import inventory](docs/importing.md) |
| Inference, review, and drift | [Label inference](docs/label-inference.md) |
| Optional cost Q&A | [Agent guide](docs/agent.md) |
| Optional cited tagging-policy context (RAG) | [Policy context](docs/policy-context.md) |
| Run obs-agent and RAG together locally | [Compose simulation](docs/local-rag-stack.md) |
| REST API, Docker, hosted demo | [Interfaces](docs/api.md) |
| Quality and calibration | [Evaluation](docs/evaluation.md) |
| Agent tracing and evaluation walkthrough | [MLflow guide](docs/mlflow.md) |
| Evaluate a real reviewed sample | [Real-world evaluation pack](docs/real-evaluation.md) |
| Deployment and operations | [Deployment guardrails](docs/deployment.md) |
| Workflow metrics, error correlation, alerts | [Operations](docs/operations.md) |
| Components and data flow | [Architecture](docs/architecture.md) |
| Tests and local checks | [Testing](docs/testing.md) |
| Reviewer's guide | [Reviewing this project](docs/reviewing.md) |
| Persistence and approval provenance | [Decision storage](docs/decision-storage.md) |
| Backup, restore and recovery checks | [Backup and restore](docs/backup-restore.md) |
| Source material | [References](docs/references.md) |

## Use as an agent skill

[`skills/cloud-label-review/`](skills/cloud-label-review/SKILL.md) packages the label-review
workflow as a `SKILL.md` for Claude Code and other agents that support the format. The agent
builds the inventory CSV and vocabulary with you, runs a read-only triage, and summarises the
suggestions; approvals still happen in the Label review page. It never applies tags or records
decisions.

```bash
# Claude Code: install for your user, then ask it to "review the labels in inventory.csv"
cp -r skills/cloud-label-review ~/.claude/skills/

# Or run the triage directly from this checkout
uv run python skills/cloud-label-review/scripts/triage.py \
    --csv fixtures/sanitized_inventory_sample.csv --vocab fixtures/label_config_example.json
```

## Application and library

This repository is the runnable Streamlit/FastAPI application. Clone it or run the Docker image;
its flat modules are intentionally not an installable Python package.
The separate [tag-inference library](https://github.com/obs-tools/tag-inference) is the library
entry point for embedding inference in another Python project. This application currently uses its
own local inference modules; it does not depend on that library.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q -m "not integration"
uv run evals.py --dry-run
```

Follow the [open-source roadmap](https://github.com/users/redbarron23/projects/2).

### Testing the RAG integration

Put a test in the repo that owns the behavior it checks and normally fixes its failures.
For the planned [rag-observability integration](https://github.com/redbarron23/obs-agent/issues/90),
obs-agent is the caller and RAG provides the documentation API:

| Behavior being checked | Test owner |
| --- | --- |
| API request validation, response fields, provenance, and retrieval without generation | rag-observability: provider contract tests |
| Finding the policy section that supports a question | rag-observability: retrieval evaluations |
| Handling RAG responses, source references, no matches, timeouts, and errors | obs-agent: consumer tests with a stub RAG service |
| A real obs-agent tool retrieves evidence from a real RAG service | obs-agent: cross-repo smoke test |

Build the checks alongside the integration, in this order:

1. Add provider contract tests with the [RAG HTTP API](https://github.com/redbarron23/rag-observability/issues/13).
2. Add offline consumer tests when implementing the documentation tool in obs-agent. Keep these
   in ordinary CI; assert source attribution, safe failure handling, and selected-review isolation.
3. Once both sides exist, add one real-service smoke test here: ingest a tiny fictional Markdown
   corpus, start RAG, call it through the obs-agent tool, and verify the supporting source/section
   and corpus identity survive. This retrieval check needs no paid generation call.

Pin the other repo to an explicit commit or release and record both revisions in test output.
Run the real-service check separately from each repo's normal independent suite, initially on
manual request and before integration releases. A scheduled check against the provider's latest
revision can detect upcoming incompatibilities, but should remain separate from the pinned check.
Generate the test's index from its fixture; do not depend on a developer's private corpus or index.

The existing pytest `integration` marker is reserved for opt-in live LLM tests; do not reuse it
for offline consumer tests. Keep any paid generation evaluation explicitly opt-in. Avoid duplicating
the same end-to-end suite in both repos. A separate integration repo is unnecessary unless a team
starts owning a shared deployment across several services.

## License

MIT. See [LICENSE](LICENSE).
