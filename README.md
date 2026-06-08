# obs-agent — Multi-Cloud Cost Triage Agent

An LLM agent that answers questions about cloud logging costs across **Azure** and **GCP** by calling tools over billing/overage data, instead of requiring you to dig through spreadsheets by hand.

Supports **Anthropic** (Claude) and **DeepSeek** as LLM providers — switch between them with a single `--provider` flag.

Built as a demonstration of the AI engineering agent pattern — tool definitions, tool dispatch, streaming output, a conversational REPL with multi-turn memory, a scripting CLI, a web UI, and an eval harness with ground-truth checks against deterministic synthetic data.

## Quick start

```bash
# 1. Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# 2. Interactive REPL (with conversation memory)
python agent.py

# 3. Or ask a single question (scriptable)
python agent.py -q "Which Azure subscription has the highest overage?" --verbose

# 4. Or launch the web UI
streamlit run app.py
```

## Features

### Streaming output

Watch tokens arrive in real-time instead of waiting for the full response:

```bash
python agent.py -q "Compare costs across Azure and GCP" --stream
```

Streaming works through the full tool-use loop — if the model calls a tool mid-stream, it prints text up to that point, executes the tool, and continues streaming.

### Multi-turn conversation memory

The REPL and web UI remember previous turns. Ask follow-up questions naturally:

```
You: Which Azure subscription has the highest overage?
Agent: sub-a1b2c3d4 at $18,411...

You: Show me its daily trend
Agent: (remembers which subscription you meant and fetches the trend)
```

The message history is pruned at 20 turns to stay within context limits.

### Cross-cloud cost comparison tool

A single tool that sums costs across both clouds and provides per-subscription and per-project breakdowns:

```
Cross-Cloud Cost Summary (30 days)
========================================
Azure total overage cost:  $23,059.00
GCP total logging cost:    $21,705.00
────────────────────────────────────────
Grand total:               $44,764.00
```

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens a chat interface in your browser with provider/model selection, conversation history, and example prompts. Supports multi-turn memory natively.

### CLI with scripting

```bash
# Single answer (for scripts, CI, or demos)
python agent.py -q "Show me the top 3 GCP projects"

# Stream mode
python agent.py -q "Any cost spikes?" --stream

# Verbose mode (see tool calls)
python agent.py -q "Compare costs" --verbose

# Specific model
python agent.py -q "Show me spikes" --model claude-sonnet-4-6
```

## Providers

### Anthropic (default)

```bash
export ANTHROPIC_API_KEY=sk-...
python agent.py -q "Which Azure subscription has the highest overage?"
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY=sk-...
python agent.py --provider deepseek -q "Which Azure subscription has the highest overage?"
```

Provider default models:
- **Anthropic**: `claude-sonnet-4-6`
- **DeepSeek**: `deepseek-chat`

## What it does

Ask questions in plain English:

```
"Which Azure subscription has the highest overage cost?"
"Show me the top 3 most expensive GCP projects"
"Compare costs across Azure and GCP"
"Has subscription sub-a1b2c3d4 had any cost spikes recently?"
"What's the daily trend for project-bravo?"
```

The agent decides which tools to call, runs them against the data, and summarises the findings — concisely, and grounded in actual numbers rather than guesses.

## Architecture

```
app.py      — Streamlit web UI (chat interface with provider/model selection)
agent.py    — Agent loop + CLI (REPL / --question) + streaming + provider abstraction
tools.py    — 5 tool functions + LLM tool definitions + dispatch table
data.py     — Deterministic synthetic data generator (seeded RNG, CSVs not needed)
evals.py    — 8 eval cases with ground-truth checks against the synthetic data
```

```
You ──question──▶  agent.py  ──tools──▶  tools.py  ──data──▶  data.py
                    │                                              │
                    │  (LLM API loop)                             (in-memory
                    │   tool_use ↔ end_turn                       DataFrames)
                    │   streaming supported
                    │   conversation memory persists
                    │
                    ▼
               answer (printed / streamed / rendered in web UI)
```

### The agent loop (`agent.py`)

The standard tool-use pattern:
1. Send the user's question + tool definitions to the model
2. If the model requests a tool, execute it via `TOOL_DISPATCH` and feed the result back
3. Repeat until the model produces a final answer

Supports both **batch** (full response at once) and **streaming** (token-by-token) modes. The `Provider` class abstracts over Anthropic and OpenAI-compatible (DeepSeek) APIs, handling differences in request format, response shape, and stop-reason naming.

### The tools (`tools.py`)

| Tool | Purpose |
|---|---|
| `get_azure_top_overages` | Top N Azure subscriptions by estimated 30-day overage cost |
| `get_gcp_top_projects` | Top N GCP projects by total logging cost |
| `get_daily_trend` | Day-by-day cost/ingestion trend for a given subscription or project |
| `find_spikes` | Detects days where cost jumped >X% vs. the previous day (both clouds) |
| `compare_cross_cloud` | Total cost comparison across Azure and GCP with per-sub/project breakdown |

Each tool returns a clean formatted string so the model gets compact, readable results.

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

This runs 8 eval cases with known-correct answers and checks that the agent's response contains the expected identifiers. Covers all five tools:

- Azure top overage, GCP top project, GCP top 3
- Daily trends (Azure and GCP)
- Spike detection (Azure and GCP)
- Multi-cloud summary

## What this project demonstrates

| Concept | Implementation |
|---|---|
| **Agent loop** | Provider-agnostic request → dispatch → feed-back cycle until `end_turn` |
| **Streaming** | Token-by-token output through the full tool-use loop (Anthropic + OpenAI SSE) |
| **Conversation memory** | Multi-turn REPL with history pruning at 20 messages |
| **Provider abstraction** | `Provider` class wrapping Anthropic and OpenAI-compatible APIs |
| **Tool definitions** | JSON Schema input specs that the model calls dynamically |
| **Tool dispatch** | `TOOL_DISPATCH` dict mapping tool names to Python functions |
| **CLI** | REPL for interactive use + `--question` / `--stream` / `--model` / `--provider` flags |
| **Web UI** | Streamlit chat interface with provider/model selection |
| **Evals** | Fact-presence checks (`must_contain`) instead of brittle exact-string matching |
| **Deterministic test data** | Seeded RNG for reproducible dev, test, and CI |

## Why this matters for AI engineering

Most software lets you write deterministic tests. Agent behaviour is non-deterministic — the same question can get different phrasings. The eval harness in this project shows the right testing strategy for this world: check that *correct facts* appear in the output, not that the exact wording matches.

The `Provider` abstraction shows how to build API-agnostic agents that can switch between LLM backends without changing the tool logic or agent loop — a practical skill for production systems where provider lock-in is a concern.

The streaming implementation handles the tool-use edge case gracefully: if the model starts answering but then decides it needs a tool, the stream pauses, executes the tool, and continues — all transparent to the user.

## Requirements

- Python 3.10+
- **Anthropic**: `ANTHROPIC_API_KEY` environment variable
- **DeepSeek**: `DEEPSEEK_API_KEY` environment variable
- Dependencies: `anthropic`, `openai`, `streamlit`, `pandas`, `numpy`
