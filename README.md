# obs-agent — Multi-Cloud Cost Triage Agent

A Claude agent that answers questions about cloud logging costs across **Azure** and **GCP** by calling tools over billing/overage data, instead of requiring you to dig through spreadsheets by hand.

Built as a demonstration of the AI engineering agent pattern — tool definitions, tool dispatch, a conversational REPL, a CLI for scripting, and an eval harness with ground-truth checks against deterministic synthetic data.

## What it does

Ask questions in plain English:

```
"Which Azure subscription has the highest overage cost?"
"Show me the top 3 most expensive GCP projects"
"Give me a summary of the top overage on Azure and the top cost on GCP"
"Has subscription sub-a1b2c3d4 had any cost spikes recently?"
```

The agent decides which tools to call, runs them against the data, and summarises the findings — concisely, and grounded in actual numbers rather than guesses.

## Quick start

```bash
# 1. Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...

# 3. Interactive REPL
python agent.py

# 4. Or ask a single question (scriptable)
python agent.py -q "Which Azure subscription has the highest overage?" --verbose
```

### Use a different model

```bash
python agent.py -q "Show me the top 3 GCP projects" --model claude-sonnet-4-6
```

## Architecture

```
data.py     — Deterministic synthetic data generator (seeded RNG, no external files)
tools.py    — Data-access functions + Claude tool definitions + dispatch table
agent.py    — Agent loop (Claude API tool-use cycle) + CLI (REPL / --question flag)
evals.py    — Eval harness with ground-truth checks against the synthetic data
```

```
You ──question──▶  agent.py  ──tools──▶  tools.py  ──data──▶  data.py
                    │                                              │
                    │  (Claude API loop)                          (in-memory
                    │   tool_use ↔ end_turn                       DataFrames)
                    │
                    ▼
               answer (printed)
```

### The agent loop (`agent.py`)

The standard Claude tool-use pattern:
1. Send the user's question + tool definitions to the model
2. If the model requests a tool, execute it via `TOOL_DISPATCH` and feed the result back
3. Repeat until the model produces a final answer

### The tools (`tools.py`)

| Tool | Purpose |
|---|---|
| `get_azure_top_overages` | Top N Azure subscriptions by estimated 30-day overage cost |
| `get_gcp_top_projects` | Top N GCP projects by total logging cost |
| `get_daily_trend` | Day-by-day cost/ingestion trend for a given subscription or project |
| `find_spikes` | Detects days where cost jumped >X% vs. the previous day (both clouds) |

Each tool returns a clean formatted string (not a raw DataFrame) so the model gets compact, readable results.

### The data (`data.py`)

Generates 60 days of realistic multi-cloud billing data with a fixed random seed:
- **5 Azure subscriptions** with varying ingestion/overage levels — one dominant, one near-zero
- **5 GCP projects** with different cost profiles
- **Built-in cost spikes**: `sub-a1b2c3d4` spikes on day 30, `project-bravo` spikes on day 20
- **Deterministic**: every run produces the same IDs, costs, and patterns, so evals always pass

No external CSVs required. The project runs entirely from synthetic in-memory data.

## Run the evals

```bash
python evals.py
```

This runs a fixed set of questions with known-correct answers and checks that the agent's response contains the expected identifiers. The eval harness covers all four tools:

- Top overage / top project queries
- Daily trends for both clouds
- Spike detection (Azure and GCP)
- Multi-cloud summary

## What this project demonstrates

| Concept | Implementation |
|---|---|
| **Agent loop** | Anthropic tool-use API: request → dispatch tool → feed back result → repeat until `end_turn` |
| **Tool definitions** | JSON Schema input specs that the model calls as needed, not a fixed pipeline |
| **Tool dispatch** | A `TOOL_DISPATCH` dict mapping tool names to Python functions |
| **CLI with scripting** | `python agent.py -q "..."` for automation; REPL for interactive use |
| **Evals for non-deterministic code** | Fact-presence checks (`must_contain`) instead of brittle exact-string matching |
| **Deterministic test data** | Seeded RNG means reproducible data for dev, test, and CI |

## Why this matters for AI engineering

Most software lets you write deterministic tests. Agent behaviour is non-deterministic — the same question can get different phrasings. The eval harness in this project shows the right testing strategy for this world: check that *correct facts* appear in the output, not that the exact wording matches.

The synthetic data generator means the project is self-contained, reproducible, and immediately runnable — no external APIs, cloud credentials, or CSV files needed (other than the Anthropic API key).

## Requirements

- Python 3.10+
- Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- Dependencies: `anthropic`, `pandas`, `numpy`
