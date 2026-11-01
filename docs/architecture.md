# Architecture and design decisions

## System map

```mermaid
flowchart TD
  inventory["Synthetic or saved imported snapshots"] --> inference["labeling.py\nclean · rules · model · mislabel check"]
  inference --> review["label_review.py\nqueue · decisions · change plan"]
  review --> ui["Streamlit label review"]
  review --> api["FastAPI"]
  inventory --> drift["drift.py\nrenewed-review findings"]
  agent["agent.py"] --> provider["providers.py"]
  agent --> tools["tools.py"]
  tools --> billing["Synthetic billing data"]
```

The label-inference flow and cost-agent flow share a user interface but are independent: tag review
needs no LLM, while cost Q&A uses an LLM only to choose and summarise deterministic tools.

## Key modules

| Module | Responsibility |
|---|---|
| `inventory_import.py` / `label_config.py` | CSV validation and per-inventory vocabularies |
| `review_workspaces.py` / `review_context.py` | Saved snapshots and approval fingerprints |
| `review_cli.py` | Command-line access to saved reviews (import, queue, decide, export) |
| `plan_validation.py` | Read-only pre-delivery check of a reviewed plan against a fresh inventory |
| `plan_reconciliation.py` | Read-only post-delivery reconciliation of an applied plan against a fresh inventory |
| `inventory.py` | Seeded synthetic inventory with true and observed labels |
| `labeling.py` | Cleaning, rules, logistic regression, explanations, and scoring |
| `label_review.py` | Review decisions, append-only feedback, change plans, and exports |
| `drift.py` | Snapshot comparison for renewed review |
| `agent.py` / `providers.py` | Bounded tool-use loop and provider abstraction |
| `tools.py` / `data.py` | Deterministic cost-analysis functions and billing data |
| `api.py` / `pages/` | REST and Streamlit surfaces |

## Design decisions

### Synthetic data with ground truth

Real billing exports are sensitive and lack authoritative labels for every resource. Synthetic true
labels make inference, thresholds, and degradation scenarios measurable and reproducible without
claiming that the result transfers automatically to a real estate.

### Human approval before application

Inference is a recommendation, not a mutation. The reviewed change plan creates a clear trust
boundary between model output and a team's Terraform, CI, or shell-based delivery process.

### Reuse within a snapshot

Imported inference and reports are kept in a process-local cache of the four most recently used
snapshots. The inventory and inference-version fingerprints include data, vocabulary, and inference
settings, so changed contexts refit. Report callers receive independent copies; decisions and
approval history are read from SQLite on each call. The cache is shared by this trusted-team
installation, retains private snapshot data in memory until eviction or process exit, and is not
tenant-isolated storage. Restarting or evicting an entry only requires recomputing its inference.

### Provider abstraction for the agent

The cost agent keeps message formats and tool-loop logic separate from provider SDK details. It can
switch between Anthropic and OpenAI-compatible providers without changing tool implementations.

### Tests check facts, not phrasing

Agent answers vary in wording. Evaluations therefore test for the facts a correct answer must
contain and, where useful, whether the expected tool path was used.
