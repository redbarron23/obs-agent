# Real-world pilot kit

A step-by-step runbook for the 50–100-resource label-review pilot
([#49](https://github.com/redbarron23/obs-agent/issues/49)). It wraps the existing
[real-world evaluation pack](real-evaluation.md), which defines sampling, anonymisation and the
agreement report. This kit adds what the pilot also has to measure: time per resource against a
manual baseline, decision stability between independent reviewers, usefulness and friction. It
ends with a sanitized summary and two decisions: expedited review, and the next feature.

Everything here runs locally. No LLM, cloud credentials or network are needed.

| File | Use |
| --- | --- |
| [`pilot/sanitization-checklist.md`](pilot/sanitization-checklist.md) | Sign-off before anyone opens the sample |
| [`pilot/inventory-template.csv`](pilot/inventory-template.csv) | The inventory columns obs-agent reads ([importing](importing.md)) |
| [`pilot/review-log.csv`](pilot/review-log.csv) | One row per resource label reviewed with obs-agent |
| [`pilot/manual-baseline-log.csv`](pilot/manual-baseline-log.csv) | The same work done by hand, on a subset |
| [`pilot/re-review-log.csv`](pilot/re-review-log.csv) | A second reviewer's independent decisions, on a subset |
| [`pilot/pilot-summary-template.md`](pilot/pilot-summary-template.md) | The sanitized write-up |
| `scripts/pilot_summary.py` | Turns the logs and the saved review's feedback into the numbers |

## 0. Set up a private workspace

Work in `pilot-workspace/` at the root of a checkout (git ignores it), or in another approved
location. Copy the templates in, and point the saved reviews at it:

```bash
mkdir -m 700 pilot-workspace
cp docs/pilot/*.csv docs/pilot/sanitization-checklist.md pilot-workspace/
export OBS_AGENT_REVIEW_DIR="$PWD/pilot-workspace/reviews"
```

Agree the roles first: a **sample owner** who draws and sanitizes the sample and holds the mapping;
at least two **reviewers** who know the estate's ownership; and a **second-person checker** for
sanitization. One person can hold more than one role, except that the re-reviewer must not be the
person who made the first decision.

Optional rehearsal: run the whole kit once on a [fictional estate pack](synthetic-estates.md) to
practise. Keep those results out of the pilot record.

## 1. Draw and sanitize the sample

Follow [`sanitization-checklist.md`](pilot/sanitization-checklist.md). In short: draw 50–100
subscriptions or projects before anyone sees suggestions, spanning clouds, teams, environments and
easy and ambiguous cases. Keep only the template's columns, replace identifiers with stable
surrogates, and have a second person check at least 10 rows. Write the estate's vocabulary as
`pilot-workspace/vocabulary.json` (shape: `fixtures/label_config_example.json`).

## 2. Import

```bash
uv run python review_cli.py import pilot-workspace/inventory.csv \
    --vocabulary pilot-workspace/vocabulary.json --name pilot    # prints REVIEW_ID
uv run python review_cli.py coverage "$REVIEW_ID"
```

The import page (`uv run streamlit run app.py`, **Import / reopen inventory**) works too. If the
import fails or needs a column mapping, note it: import problems are a `friction` category.

Importing the same CSV and vocabulary again returns the **same** review ID. A second import is not
a fresh copy for the re-reviewer (see step 5).

## 3. Review with obs-agent and log the time

Reviewers open the saved review (**Import / reopen inventory → Reopen a saved review**) and
decide each `team` and `env` label: accept, reject or override, with their own reviewer ID. Leave
a label undecided only when it can't be resolved.

For each resource label, add one row to `review-log.csv`:

| Column | Values |
| --- | --- |
| `resource_id`, `label` | The surrogate ID; `team` or `env` |
| `reviewer` | The reviewer ID used for the decision |
| `seconds` | Time spent on this label, from opening the resource to saving (0–3600) |
| `outcome` | `accept`, `reject`, `override`, or `unresolved` (no decision saved). It must match the saved review |
| `missing_evidence` | `yes` if the ownership decision needed evidence the inventory didn't have (an owner, a cost centre, a conversation), else `no` |
| `friction` | Blank, or the main obstacle: `import`, `vocabulary`, `unclear_reason`, `missing_owner`, `conflicting_evidence`, `ui`, `other` |

The logs have no free-text column on purpose, so the summary stays sanitized. If a reviewer
wants to explain a decision, use the review's own rationale or notes, which stay in the private
review database.

## 4. Manual baseline

Before reviewers see obs-agent's suggestions for them, label a subset by hand (10–20 resources is
enough), in the way the team normally would, and log it in `manual-baseline-log.csv`: `seconds`
and the `resolved_value` per label. Choose the subset at random from the sample and write down
how (for example "15 of 80, `random.sample`, seed 3"). You pass that description to the summary.

Doing the baseline first, or with a reviewer who doesn't review those resources with obs-agent,
avoids timing a person who has already seen the answer.

## 5. Independent re-review

A second reviewer re-decides a subset (10–20 resource labels, including some hard ones) without
seeing the first decisions. Because a second import returns the same saved review, give them
the read-only triage output for their subset instead:

```bash
uv run python skills/cloud-label-review/scripts/triage.py \
    --csv pilot-workspace/inventory.csv --vocab pilot-workspace/vocabulary.json \
    --out pilot-workspace/triage.json
```

They log one row per label in `re-review-log.csv`: `reviewer` (theirs) and the `resolved_value`
they would set. Leave `resolved_value` blank for unresolved. After comparing, record how each
disagreement was settled in `resolution`: `first`, `second`, `other` or `unresolved`. A changed
decision is made in the saved review by a reviewer, as usual.

## 6. Run the numbers

```bash
uv run python review_cli.py feedback "$REVIEW_ID" > pilot-workspace/feedback.json
uv run real_evaluation.py pilot-workspace/feedback.json > pilot-workspace/evaluation.json
uv run python scripts/pilot_summary.py \
    --feedback pilot-workspace/feedback.json \
    --review-log pilot-workspace/review-log.csv \
    --baseline-log pilot-workspace/manual-baseline-log.csv \
    --baseline-sampling "15 of 80, random.sample, seed 3" \
    --re-review-log pilot-workspace/re-review-log.csv \
    --out pilot-workspace/pilot-summary.json
```

`real_evaluation.py` reports reviewer agreement and the confidence thresholds
([how to read it](real-evaluation.md#3-set-the-production-policy-from-evidence)).
`pilot_summary.py` checks that the review log matches the saved review, then reports:

- **time:** median and p90 seconds per resource with obs-agent and, on the same resources, by hand;
- **usefulness per label:** accepts, overrides, rejections and their rates; labels unresolved for
  scoring (including a rejected suggestion on an untagged label); labels decided without a
  suggestion (withheld for insufficient evidence); decisions that needed missing evidence;
- **decision stability:** agreement between the two reviewers, each disagreement and its
  resolution;
- **friction:** counts per category.

It exits with status 2 and names the row when a log is inconsistent, for example when an outcome
doesn't match the saved review or a re-review is by the same reviewer.

Back up the workspace with `scripts/review_backup.py` ([backup and restore](backup-restore.md)) once
the review is complete.

## 7. Write the summary and decide

Fill in [`pilot-summary-template.md`](pilot/pilot-summary-template.md) from the two JSON reports.
Keep reviewer agreement separate from accuracy: the sample's reviewers are the reference, not
independently established ground truth.

Two decisions close the pilot:

- **Expedited review.** Adopt a threshold only if `real_evaluation.py` finds one with enough
  per-label support and agreement. Otherwise keep review-only. Too little evidence never lowers
  a threshold.
- **Next feature.** Use the usefulness and friction results. For example, many `missing_evidence`
  decisions argue for better ownership signals before chat over reviews
  ([#50](https://github.com/redbarron23/obs-agent/issues/50)); the sample's export format can seed
  the billing import ([#104](https://github.com/redbarron23/obs-agent/issues/104)).

Only the sanitized summary leaves the workspace. The inventory, mapping, logs, saved review and
backups stay in the approved environment.
