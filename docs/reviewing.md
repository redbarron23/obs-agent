# Reviewing this project

Notes for anyone peer-reviewing obs-agent — a colleague, an interviewer, or your future self
returning after six months. It says what the project is, what is worth your attention, and where
the sharp edges are.

**You do not need to read this before opening the repo.** The README and `docs/` carry the
argument on their own. This page exists to point your review budget at the parts that matter.

## Orientation

This is two features in one repo:

1. **ML tag inference** (the main one) — infers missing `team` / `env` / `cost_center` tags on
   Azure subscriptions and GCP projects, flags tags that look wrong, and puts suggestions through
   a human review queue that exports `az` / `gcloud` / Terraform changes. No LLM involved.
2. **Cost Q&A agent** (secondary) — a tool-calling LLM agent that answers billing questions. See
   [docs/agent.md](agent.md).

Roughly 7k lines of Python. The argument lives in `README.md` and `docs/`; the code is in flat
top-level modules (`labeling.py`, `providers.py`, `tools.py`, `api.py`, …).

**Published benchmarks use synthetic data.** The UI also supports saved, private imports with
per-inventory vocabularies, also reachable by review ID through the API and CLI. The cost agent remains synthetic. Every number in the README is measured against a generated inventory
that carries true labels alongside observed ones. That is what makes honest scoring possible, and
it is also the project's main limit — see [Judging the numbers](#judging-the-numbers).

## Where your attention is worth the most

In rough priority order. These are the load-bearing claims; a bug in any of them invalidates
something the README asserts.

| Look at | Why it matters |
|---|---|
| `labeling.py` — the mislabel check | `StratifiedKFold` + `cross_val_predict` (line ~299). This is the one place where a subtle mistake would quietly inflate several headline numbers at once. Out-of-fold prediction is easy to get wrong in a way that still looks plausible. |
| `tools.py` + `evals.py` | Whether the eval cases actually test what they claim. `evals.py` pairs each question with ground-truth identifiers (`must_contain`) drawn from `data.py`. Check that those pairs are correct, not just that they pass. |
| `api.py` — auth and decision persistence | The only stateful, user-facing surface. `X-API-Key` enforcement (line ~82) and review decisions persisted to SQLite via `OBS_AGENT_DECISIONS`. Look for anything a hostile caller could do. |
| `providers.py` | Split out of `agent.py` recently. A second pair of eyes is cheap right now: it holds the Anthropic / OpenAI-compatible abstraction, message formatting and model config. |
| `inventory.py` — the two estates | `clean` is pinned byte-identical by tests; `harsh` is the adversarial one. Worth checking the harsh estate is genuinely harder rather than merely noisier. |

### The highest-value checks

Try an imported snapshot with your own vocabulary, save a review, restart, and reopen it.
Check that the export carries the correct before/after changes and reviewer evidence. Then change
an input or inference version and verify that earlier approvals cannot be exported as current.

`AUTO_APPLY_THRESHOLD = 0.9` is expedited-review eligibility, not permission to write cloud tags.
Test whether high-confidence suggestions survive messy ownership signals. Synthetic precision
alone does not justify a production threshold. The reviewed plan must always exclude unreviewed
suggestions, even when convenience exports explicitly include them.

## Judging the numbers

The README is deliberately explicit about this, but it bears repeating:

- **The 98% team accuracy is a synthetic benchmark.** Synthetic owners map more neatly onto teams than
  real ones would. The harsh estate exists to measure that gap: team accuracy falls **98% → 85%**
  while auto-apply precision holds at 100%. With insufficient-evidence abstention, the review flow
  suggests a team for 77% of harsh untagged resources at 94% accuracy; read accuracy with coverage.
- **The synthetic-data ceiling is unresolved.** Whether the approach transfers to a real estate is
  still open. The honest next step would be a small hand-labelled real sample (even ~50 resources)
  — that is a known gap, not an oversight.
- **Live LLM evals cost money** and are not run in CI. Offline: unit and integration tests plus a 12-case
  deterministic dry-run. The live results in the testing guide (DeepSeek 12/12, Ollama 10/12) are
  recorded runs, not a per-commit gate. See [docs/testing.md](testing.md) for what the mocks
  cannot catch.

## Not worth your time

Please do not spend your review budget on these — they are handled or deliberately out of scope:

- **Formatting, import order, quote style** — `ruff` enforces it (`uv run ruff check .`).
- **File length.** These are flat top-level modules on purpose: the Dockerfile, the Streamlit
  `pages/` directory and the uv flat-module layout all depend on it. Prose was moved into `docs/`
  rather than code being moved into packages.
- **README length.** It was trimmed from 703 lines by moving the API reference, testing
  internals and references into `docs/`. Further detail belongs in `docs/`, not the README.
- **Missing type hints on internal helpers**, `#` comment density, test naming.

## Running it

```bash
uv sync                                   # environment (uv only, never pip)
uv run labeling.py                        # strategy comparison table
uv run pytest -q -m "not integration"     # offline, no API key
uv run evals.py --dry-run                 # 12 deterministic eval cases, no API key
uv run streamlit run app.py               # the review UI
```

Live LLM evals need a provider and cost money; ask before running them.

If you find something, send it over with the input that reproduces it. Findings that contradict a
claim in the README are the priority — a docs fix is the wrong response to a broken claim.
