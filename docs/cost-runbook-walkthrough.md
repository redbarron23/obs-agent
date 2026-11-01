# Cost findings with cited runbook steps (fictional walkthrough)

The cost chat can answer "which project has the largest logging spike, and what does our runbook
say to investigate?" in one provider loop: the cost tools supply the billing facts and
`search_observability_docs` (the retrieval tool from [Policy context](policy-context.md)) supplies
cited runbook steps. There is no second retrieval stack, no second LLM call and no new top-level
tool. Everything here is fictional: the billing data is obs-agent's seeded synthetic estate and the
runbook is rag-observability's invented Contoso Cloud corpus.

This is read-only cost triage. obs-agent never changes retention, sampling, exclusion filters or
any cloud resource.

## What comes from where

| Part of the answer | Source | Rule |
| --- | --- | --- |
| Billing facts: amounts, rankings, spike days, trends | `find_spikes`, `get_daily_trend`, `get_azure_top_overages`, `get_gcp_top_projects`, `compare_cross_cloud` over the synthetic demo billing data (`data.py`, seed 42), in USD | Quoted exactly as the tools report them. Summary tools total the reporting window `2026-05-03 to 2026-06-01 (30 days)`; `find_spikes` and `get_daily_trend` read every daily record from 2026-04-03 to 2026-06-01, so a spike's date is always given |
| Suggested next steps | Runbook excerpts, each cited as `source § section` | Recommendations, not findings. Listed causes are hypotheses; a spike never proves a cause; no change is guaranteed to save money, and savings must be verified in the billing data |
| Conflicts between documents | Same heading returned from more than one source | Disclosed, never resolved |
| Missing guidance (no match, or the service is down) | `search_observability_docs` status | The billing facts stay; the answer says documentation guidance is unavailable |

Numbers in a document are never used as billing facts. Imported inventory cost-shape fields
(`avg_daily_cost`, `weekend_ratio`, `cost_cv`) are label-inference features, not billing history;
the chat uses only the bundled synthetic data ([#50](https://github.com/redbarron23/obs-agent/issues/50)
covers any imported data in chat). Real billing analysis would need a separate, approved import
path.

These rules are added to the system prompt (`COST_RUNBOOK_PROMPT` in `providers.py`) and to every
`search_observability_docs` result (`policy_docs.format_for_llm()`, plus the billing scope line
from `tools.billing_scope()`) only when `OBS_AGENT_RAG_URL` is set. With it unset, the prompt, the
seven tools and their output are unchanged.

## The runbook

The corpus document is rag-observability's
[`data/runbooks/logging-cost-spike.md`](https://github.com/redbarron23/rag-observability/blob/main/data/runbooks/logging-cost-spike.md),
added in [redbarron23/rag-observability#74](https://github.com/redbarron23/rag-observability/pull/74).
Its stable headings, which obs-agent cites as sections, are:

| Section | Content |
| --- | --- |
| `Detection and triage` | Confirm the spike, period and currency in the billing data; priority |
| `Investigation: Azure Log Analytics` | `Usage` table by `DataType` and day, `_BilledSize` by `_ResourceId`, daily cap warnings, pricing and commitment tier, DCR changes |
| `Investigation: GCP Cloud Logging` | `logging.googleapis.com/billing/bytes_ingested` by `resource_type` and `log`, Logs Storage, billing export, sink and exclusion changes |
| `Likely causes (hypotheses)` | Onboarding, debug logging, retry storms, removed filters, duplicate sinks, new audit logs |
| `Remediation options` | Retention, sampling, exclusion filters, commitment tiers; "verify savings; savings are not guaranteed" |
| `Verifying savings` | Compare at least 14 days before and after in the same billing data |
| `Commitment tiers` | Deliberately conflicts with `programme/COST_SMOOTHING.md` § `Commitment tiers` |
| `Escalation` | Owning team, FinOps, observability team |

## Run it

```bash
# 1. rag-observability: build the index (includes the runbook) and serve it on loopback
cd rag-observability
python ingest.py
python api.py                      # http://127.0.0.1:8765, no generation, no provider key
# optional: RAG_API_KEY=<key> python api.py   (then export the same key below)

# 2. obs-agent: point the chat at it
cd ../obs-agent
export OBS_AGENT_RAG_URL=http://127.0.0.1:8765
# export RAG_API_KEY=<key>         # only if the service was started with one
uv run agent.py                    # or: uv run streamlit run app.py  (Cost chat page)
```

The chat itself needs an LLM provider (see [Agent](agent.md)); the retrieval service never does.
To check the retrieval half without any LLM:

```bash
OBS_AGENT_RAG_URL=http://127.0.0.1:8765 uv run python -c "
from tools import find_spikes, search_observability_docs
print(find_spikes(threshold_pct=200, platform='gcp'))
print(search_observability_docs('What does our runbook say to investigate for a GCP logging cost spike?'))"
```

## Example

Ask:

> Which GCP project has the largest logging spike, and what does our runbook say to investigate?

Expected tool facts (`find_spikes` with `threshold_pct=200, platform="gcp"`; spikes are ranked by
that day's cost):

```
[GCP]   project-bravo on 2026-04-23: +405% ($990.00)
```

Expected cited sections (vector retrieval, `k=4`, with the corpus from PR #74):

| Rank | Citation |
| --- | --- |
| 1, 2 | `runbooks/logging-cost-spike.md § Investigation: GCP Cloud Logging` |
| 3 | `runbooks/logging-cost-spike.md § Logging Cost Spike Runbook` |
| 4 | `runbooks/logging-cost-spike.md § Detection and triage` |

A good answer has two labelled parts. **Billing facts**: project-bravo had the costliest GCP spike,
+405% to $990.00 on 2026-04-23, in the synthetic demo billing data (USD); that day is before the
30-day reporting window that the summary tools total. **Suggested next steps**: chart
`bytes_ingested` by `resource_type` and `log`, check the billing export and recent sink or
exclusion changes, each cited to `runbooks/logging-cost-spike.md § Investigation: GCP Cloud
Logging`, with possible causes offered as hypotheses and no promise of savings.

Two more questions show the other paths:

| Question | Docs question the model sends | Expected result |
| --- | --- | --- |
| sub-a1b2c3d4 spiked on Azure. Should we buy a Log Analytics commitment tier now? | When should we buy a commitment tier after a logging cost spike? | `[Azure] sub-a1b2c3d4 on 2026-05-03: +331% ($2691.00)`; `runbooks/logging-cost-spike.md § Commitment tiers` and `programme/COST_SMOOTHING.md § Commitment tiers`, flagged "may conflict: Commitment tiers". The answer states both positions (three stable months and a median-sized tier, against two months and 100 GB per day) and does not choose |
| How much did project-bravo spend on logging on 23 April, according to our docs? | How much did project-bravo spend on logging on 23 April? | `status: no_match`. The answer gives the billing facts from the tools and says documentation guidance is unavailable |

Stop `api.py`, or point `OBS_AGENT_RAG_URL` at a closed port, and ask again: the result is
`status: unreachable`, and the answer keeps the spike and says documentation guidance is
unavailable.

The `corpus_sha256`, `ingested_at` and distances depend on your checkout and ingestion time; the
fixtures in `fixtures/` were captured from this corpus (`corpus_sha256=931f94bb…`, 9 documents,
66 chunks).

## Tests and evals

- `uv run evals.py --dry-run` runs three combined stub-retrieval cases (`cost-runbook-supported`,
  `cost-runbook-no-match`, `cost-runbook-conflict`). Each runs the real cost tool and a stubbed
  retrieval from `fixtures/rag_retrieve_logging_cost_runbook.json`,
  `fixtures/rag_retrieve_no_match.json` or `fixtures/rag_retrieve_commitment_tier_conflict.json`,
  and checks the tool facts, the stated period and the cited sources, not wording.
- `tests/test_cost_runbooks.py` covers the prompt on and off, the billing scope line, the
  recommendation rules, conflict disclosure, no match and failures keeping the cost findings, one
  provider loop with no extra LLM call, and that a document cannot change any cost tool's output.
- rag-observability's `evals.py` has the matching retrieval cases:
  `runbook-gcp-logging-investigation`, `conflict-commitment-tier-after-spike` and
  `unanswerable-logging-spend-amount`.
