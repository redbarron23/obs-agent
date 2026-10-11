# Testing

The test suite is fast, offline and deterministic. That is also its main limitation, and this
document is explicit about where the line falls — a green suite here means **the code behaves as
written**, not that **the model answers well**.

## Three layers, three different questions

| Layer | Count | Question it answers | Runs in CI? | Needs an API key? |
|---|---|---|---|---|
| Unit + integration tests | 232 | Does the code do what it says? | Yes | No |
| Eval dry-run | 12 | Do the tool outputs still contain the facts the answers depend on? | Yes | No |
| Live evals | 12 | Does the model actually use those facts to answer correctly? | **No** | Yes |

The 232 unit tests run in CI on every push and need no provider. The 12 live evals do not run in CI,
and that gap is deliberate — see [What the mocks cannot catch](#what-the-mocks-cannot-catch).

## Layer 1 — unit and integration tests (offline)

```bash
uv run pytest tests/ -v                    # everything except live evals
uv run pytest tests/ -v -m "not integration"   # same thing, explicit
```

Everything under `tests/` runs without a provider. There are two sub-styles:

**Patching the data.** `test_tools.py` replaces the module-level DataFrames with small fixtures
whose values are known by construction, then asserts on logic:

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

These tests are strong precisely because the data is hand-built: a ranking bug cannot hide behind
realistic noise.

**Mocking the provider.** `test_agent.py` patches `Provider.create` so no network call happens, then
drives the loop with a scripted response:

```python
resp = make_anthropic_response(
    tool_name="get_azure_top_overages",
    tool_input={"n": 1},
)

with patch("agent.Provider.create", return_value=resp):
    answer, messages = run("Which Azure sub is top?")

assert "sub-a" in answer
```

Both API shapes (Anthropic and OpenAI-compatible) are built by helper factories, so the loop is
exercised against each. The streaming path has its own tests, including tool calls arriving through
the stream.

**What this layer cannot tell you.** The scripted response is written by hand, so the test asserts
that the loop *dispatches* a tool call and *reassembles* the answer. It does not assert that a real
model would have chosen that tool, with those arguments, or phrased a correct answer. A change that
made the prompts subtly worse would pass every test in this layer.

## Layer 2 — the eval dry-run (offline)

`evals.py` holds 12 cases, each pairing a question with a list of strings a correct answer must
contain. In dry-run mode the **tools** are called for real and the model's job is stubbed:

```bash
uv run evals.py --dry-run     # 12/12 must pass; this is what CI runs
```

So a dry-run case catches a genuine class of regression — a tool that stops returning the fact an
answer depends on, a renamed field, a changed sort order — without needing a model. `test_evals.py`
runs the whole dry-run and each case individually.

It does **not** catch a model that ignores the tool output. That is the whole reason layer 3 exists.

## Layer 3 — live evals (opt-in, costs money)

```bash
OBS_AGENT_RUN_LIVE_EVALS=1 uv run pytest tests/ -v -m integration
```

This is the same 12 cases, but the real model answers end-to-end. The suite skips itself unless
`OBS_AGENT_RUN_LIVE_EVALS=1` **and** the key for the active provider is set, so it can never fire by
accident or run up a bill in CI. It also writes to MLflow when the `mlops` group is installed
(`OBS_AGENT_MLFLOW=1`), so runs are comparable over time.

Because these calls cost money and depend on a provider's availability, they are a **release check,
not a per-commit gate**. Run them before changing a prompt, a tool schema, the system prompt, or the
default model.

## What the mocks cannot catch

Stated plainly, because this is the honest weakness of a suite this size:

| A change to… | Caught by unit tests? | Caught by dry-run? | Caught by live evals? |
|---|---|---|---|
| Tool logic (ranking, filters, thresholds) | ✅ | ✅ | ✅ |
| Tool return shape / field names | ✅ | ✅ | ✅ |
| Loop plumbing (dispatch, history, streaming) | ✅ | — | ✅ |
| Provider request params (thinking, tokens) | ✅ | — | — |
| A tool description or prompt that misleads the model | ❌ | ❌ | ✅ |
| The model picking the wrong tool, or misreading output | ❌ | ❌ | ✅ |
| Answer quality, tone, or hedging | ❌ | ❌ | Partial |

The bottom three rows are the reason the live eval layer exists. No amount of mocking substitutes
for them, because the variable under test *is* the model.

### On small local models

The default Ollama model (`qwen2.5:7b`) is unreliable on questions that combine several facts. On
the regression case `label-count-and-team` it produced the right count but invented a team, and it
scores **10/12** across the set. That is a finding about the model, not a bug to be fixed, and it is
why the CLI, the API (`warning` field) and the UI sidebar all flag small local models rather than
pretending they are equivalent. `deepseek-chat` scores 12/12.

## The regression wall

`tests/test_labeling.py` is the one place where tests assert against **quality**, not just
behaviour: model accuracy and auto-apply precision have floors pinned against the synthetic estate's
ground truth. A change that quietly degrades inference fails CI rather than silently changing the
numbers in the README.

Since [the harsh estate](../README.md#how-it-works) landed, those floors are checked on the messy
estate too — including that auto-apply precision stays ≥95% when the data is adversarial, and that
the clean estate stays byte-identical so the published metrics keep meaning the same thing.

## CI

`uv run pytest -q -m "not integration"` and `uv run evals.py --dry-run` run on every push and PR
across Python 3.10, 3.11 and 3.12, alongside a Docker job that builds the image and smoke-tests
`/health`. Live evals are never run in CI.

## Running things

```bash
uv run pytest tests/ -q                          # unit + integration tests
uv run pytest tests/ -q -m "not integration"     # same, explicit
uv run pytest tests/test_labeling.py -v          # one file
uv run evals.py --dry-run                        # deterministic eval check
OBS_AGENT_RUN_LIVE_EVALS=1 uv run pytest tests/ -m integration   # live evals (costs money)
OBS_AGENT_MLFLOW=1 uv run evals.py               # live evals with MLflow tracing
```
