# obs-agent — Multi-Cloud Cost Triage Agent

An LLM-powered agent that answers natural-language questions about cloud logging costs across **Azure** and **GCP**. Instead of digging through spreadsheets, you ask: *"Which Azure subscription has the highest overage?"* — and the agent calls tools, gathers data, and gives you a concise answer.

Supports **Anthropic (Claude)** and **DeepSeek** as LLM providers — switch with a single `--provider` flag.

Built as a demonstration of the AI engineering agent pattern: tool definitions, tool dispatch, streaming output, a conversational REPL with multi-turn memory, a scripting CLI, a web UI, and an eval harness with ground-truth checks against deterministic synthetic data.

## What it does

Ask questions in plain English and get answers grounded in actual numbers:

```
"Which Azure subscription has the highest overage cost?"
"Show me the top 3 most expensive GCP projects"
"Compare costs across Azure and GCP"
"Has subscription sub-a1b2c3d4 had any cost spikes recently?"
"What's the daily trend for project-bravo?"
```

The agent decides which tools to call, executes them against the billing data, and summarises the findings — concisely, with actual dollar figures.

## Quick start

```bash
# 1. Clone and set up
git clone <repo-url> && cd obs-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-...

# 3. Interactive REPL (with conversation memory)
python agent.py

# 4. Or ask a single question (scriptable)
python agent.py -q "Which Azure subscription has the highest overage?" --verbose

# 5. Or launch the web UI
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

Message history prunes at 20 turns to stay within context limits.

### Cross-cloud cost comparison

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

Opens a chat interface in your browser with provider/model selection, conversation history, and example prompts. Multi-turn memory works natively.

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

### Provider support

| Provider | Default model | Env variable |
|---|---|---|
| **Anthropic** (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| **DeepSeek** | `deepseek-chat` | `DEEPSEEK_API_KEY` |

Switch providers:
```bash
python agent.py --provider deepseek -q "Which Azure subscription has the highest overage?"
```

The provider abstraction (`Provider` class in `agent.py`) wraps both APIs behind a common interface, so tool logic and the agent loop work identically regardless of backend.

## Architecture

```
app.py      — Streamlit web UI (chat interface with provider/model selection)
agent.py    — Agent loop + CLI (REPL / --question) + streaming + provider abstraction
tools.py    — 5 tool functions + LLM tool definitions + dispatch table
data.py     — Deterministic synthetic data generator (seeded RNG, no CSVs needed)
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
| `find_spikes` | Detects days where cost jumped >X% vs the previous day (both clouds) |
| `compare_cross_cloud` | Total cost comparison across Azure and GCP with per-sub/project breakdown |

Each tool returns a clean formatted string so the model gets compact, readable results. Tools are pure functions operating on in-memory DataFrames — no network calls, no side effects.

### The data (`data.py`)

Generates 60 days of realistic multi-cloud billing data with a fixed random seed:

- **5 Azure subscriptions** with varying ingestion/overage levels — one dominant, one near-zero
- **5 GCP projects** with different cost profiles
- **Built-in cost spikes**: `sub-a1b2c3d4` spikes on day 30, `project-bravo` spikes on day 20
- **Deterministic**: every run produces the same IDs, costs, and patterns, so evals always pass

No external CSVs required. The project runs entirely from synthetic in-memory data. To use real billing data, swap the DataFrames in `data.py` — the rest of the code doesn't care where the data comes from.

## Run the evals

```bash
python evals.py
```

This runs 8 eval cases with known-correct answers and checks that the agent's response contains the expected identifiers. Covers all five tools:

- Azure top overage, GCP top project, GCP top 3
- Daily trends (Azure and GCP)
- Spike detection (Azure and GCP)
- Multi-cloud summary

Because LLM output is non-deterministic, evals check for **presence of correct facts** (`must_contain`) rather than exact-string matching — the right testing strategy for AI systems.

## Design decisions

### Why synthetic in-memory data?

Real billing CSVs contain sensitive subscription IDs and project names. Using deterministic synthetic data means:

- The project runs with zero setup beyond `pip install`
- Evals are reproducible — they pass or fail deterministically
- No risk of committing credentials or PII to version control
- Anyone can clone and run it immediately

### Why fact-presence checks instead of exact matching?

Most software lets you write deterministic tests. Agent behaviour is non-deterministic — the same question can get different phrasings. The eval harness in this project shows the right testing strategy for this world: check that *correct facts* appear in the output, not that the exact wording matches.

### Why a provider abstraction?

Building API-agnostic agents is a practical skill for production systems where provider lock-in is a concern. The `Provider` class in `agent.py` shows how to switch between Anthropic and OpenAI-compatible backends without changing the tool logic or agent loop.

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

## Requirements

- Python 3.10+
- **Anthropic**: `ANTHROPIC_API_KEY` environment variable (for Claude)
- **DeepSeek**: `DEEPSEEK_API_KEY` environment variable (optional)

### Dependencies

```
anthropic>=0.50.0
openai>=1.0.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
```

## Related resources

- [How to Transition from Software Engineering to AI Engineering](../How-to-Transition-from-Software-Engineering-to-AI-Engineering.md) — notes on the skills gap and learning approach
- [Skills & Initiatives](../skills-initiatives.md) — tracking evals, RAG, LLMOps, and portfolio work
- [AI Engineering Skills Checklist](../AI/AI-Engineering-Skills-Checklist.pdf) — comprehensive skill taxonomy used to guide this project

## License

MIT — see [LICENSE](./LICENSE).
