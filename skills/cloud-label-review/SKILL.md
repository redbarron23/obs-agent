---
name: cloud-label-review
description: Find missing or suspect team/env labels on Azure subscriptions and GCP projects, suggest values with evidence and confidence, and hand a human a reviewed change plan. Use when someone asks to audit cloud tags/labels, fix untagged cost, prepare a tagging clean-up, or attribute cloud spend to teams. Read-only, never applies tags.
---

# Cloud label review (obs-agent)

Turn an Azure/GCP inventory into label suggestions a human can approve. Uses
[obs-agent](https://github.com/redbarron23/obs-agent): rules plus a logistic-regression
model infer `team` and `env`, flag existing tags the model disputes, and derive `cost_center`
from the team.

## Hard rules

- **Never apply tags.** No `az tag`, `gcloud ... update-labels`, Terraform apply, or similar.
  This skill produces suggestions and, through the UI, a reviewed change plan. Applying it is the
  user's own delivery process.
- **Never record decisions on the user's behalf.** Approvals carry a reviewer identity; only a
  person enters them. Don't run `review_cli.py decide`, `undo`, `assign`, `note`, `ask`,
  `answer` or `resolve`, and don't call the REST API's `PUT`/`DELETE …/decision` or
  `POST …/policy-evidence`, even if the user hands you a reviewer name. Point them to the Label
  review page or those commands to do it themselves.
- Treat `"action": "auto-apply"` in the output as "eligible for expedited human review" (confidence
  at or above 0.90), not as permission to apply.
- **Disputed existing tags (`status: "suspect"`) always need a human**, whatever the confidence.
- **A withheld label is an unknown, not a low-confidence guess.** When `action` is
  `"insufficient evidence"`, `suggested` is null on purpose: the ownership evidence is too weak to
  suggest anything. Don't present `evidence.withheld.value` (the model's raw pick) as a
  suggestion or recommend it; only a person who confirms ownership can set the label, with an
  explicit override.
- Keep inventories and `review-workspaces/` out of version control. They name owners and spend.

## Steps

1. **Get obs-agent.** If not already in a checkout:
   `git clone https://github.com/redbarron23/obs-agent.git && cd obs-agent && uv sync --locked`
   (needs Python 3.11+ and [uv](https://docs.astral.sh/uv/)). No API keys or cloud credentials needed.

2. **Build the inventory CSV.** One row per Azure subscription or GCP project. Required columns:
   `cloud` (`azure`/`gcp`), `resource_id`, `parent` (management group / folder), `owners`,
   `created_by`, `top_services` (lists as `a|b` or JSON), `avg_daily_cost`, `weekend_ratio`,
   `cost_cv`, `team`, `env`, `cost_center` (blank when missing). Full schema and column mapping:
   `docs/importing.md`. If the user exports from their own tooling, ask them for the export rather
   than querying their cloud accounts yourself, and remove anything sensitive the model doesn't need.

3. **Write a vocabulary JSON** for their organisation (`fixtures/label_config_example.json` shows
   the shape): exactly `team` and `env`, each with lowercase `values` and optional `aliases`;
   `env` may add `name_tokens`. Without it the synthetic demo vocabulary is used, which is wrong
   for a real estate.

4. **Check there's enough signal.** Each label needs at least two recognised values with at least
   two labelled resources per value, or the import fails. If coverage is thin, say so: the
   suggestions will be weak.

5. **Run the headless triage** (read-only):

   ```bash
   uv run python skills/cloud-label-review/scripts/triage.py \
       --csv inventory.csv --vocab vocabulary.json --out triage.json
   ```

   Run it from the obs-agent checkout. If the skill is installed elsewhere (for example
   `~/.claude/skills/`), use that path to `triage.py` instead; it uses the checkout it's run from.
   It writes nothing else: no saved review, no decisions. `--min-confidence 0.5` hides queue
   entries below that model score.

   Output: `coverage` and `queue`.
   - `coverage`: totals, plus per label the `untagged`, `suspect`, `auto_apply_ready` and
     `insufficient_evidence` counts, with `insufficient_evidence_reasons`.
   - `queue`: resources with `needs_attention`, most confident first. Each label has:
     - `status`: `ok`, `untagged` or `suspect`.
     - `action`: `auto-apply` (≥ 0.90, still needs a human), `needs review`,
       `insufficient evidence`, or null when the label is fine.
     - `suggested`, `confidence`, `source` (`rules`/`model`/`out-of-fold model`), `alternatives`.
     - `reason`: the model's strongest features. These can be coincidences such as `cloud=` or
       a name prefix.
     - `evidence`: the ownership evidence check. `status` is `supported`, `missing_signals`,
       `insufficient_support` or `conflicting`. `supporting` and `conflicting` list owner and
       name signals seen on labelled resources, `reason` explains the result, and `withheld`
       records what was not suggested.
   - `cost_center.implied` is set only when the team-to-cost-center mapping is unambiguous;
     `conflicting` lists the clashes.

6. **Summarise for the user.** Lead with the counts, then group the queue:
   - High-confidence untagged suggestions (fast to review), with the owner signals from
     `evidence.supporting`.
   - Low-confidence untagged ones: show the top alternative and the `reason`, so they know
     what evidence to check.
   - Withheld labels (`insufficient evidence`): say why, using `evidence.reason`. Typical causes
     are no owner or creator recorded (`missing_signals`), or owners not seen on other labelled
     resources (`insufficient_support`). Say what would resolve it: record an owner, or confirm
     ownership and override. Don't name the withheld value.
   - Suspect existing tags: the current value, the model's value, and why they disagree.
   - Cost-center conflicts.
   Judge ownership from `evidence`, not `reason`. If a suggestion's `reason` is only
   `cloud=`/`prefix=`/`tok=` features and `evidence.supporting` has no `owner=` signal, say the
   suggestion rests on naming, not ownership.

7. **Hand off to human review.** `uv run streamlit run app.py`, then **Import / reopen inventory**,
   upload the same CSV and vocabulary, **Save and open review**, enter a reviewer name, accept,
   reject or override, and download the **Reviewed change plan**. It contains only reviewed
   changes with before/after values and approval evidence. CSV, shell and Terraform exports are
   available for their delivery process.

   If the user would rather not upload again, and agrees to keep a private copy, you can save the
   review for them. This is not a decision:
   `uv run python review_cli.py import inventory.csv --vocabulary vocabulary.json` prints a review
   ID. They then pick it under **Import / reopen inventory → Reopen a saved review → Open saved review**, or decide with
   `review_cli.py decide` themselves. The saved copy lives in `review-workspaces/` (or
   `OBS_AGENT_REVIEW_DIR`), contains the inventory, and must stay out of version control.

8. **Next cycle:** save the new snapshot and use **Compare snapshots** to review only what changed
   (drift in owners, services or spend re-opens a label).

## Limits to state honestly

- Published accuracy (team 98%, env 81% hybrid) is on synthetic data, not a production claim.
  Recommend validating on a reviewed sample before trusting the 0.90 threshold:
  `docs/real-evaluation.md`.
- Only `team`, `env` and derived `cost_center`, at subscription/project level.
- The model learns from existing tags, so it inherits their bias. Resources unlike anything
  labelled get weak suggestions.
