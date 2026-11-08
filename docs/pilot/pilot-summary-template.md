# Label-review pilot summary (sanitized)

Fill this in from `pilot-summary.json` (`scripts/pilot_summary.py`) and the
`real_evaluation.py` report. Use surrogate IDs only. Runbook: [pilot kit](../pilot-kit.md).

## Sample

- Resources / resource labels / reviewers:
- How the sample was drawn:
- Sanitization approved by, and second-person check:
- obs-agent commit, `inventory_id`, `model_version`:

## Reviewer agreement (not accuracy)

- `real_evaluation.py` macro precision / recall / F1 per label:
- Per-label support, and threshold results at 0.90 / 0.95 / 0.98:
- State plainly: these measure agreement with this sample's reviewers. They are not an unbiased
  production accuracy estimate.

## Usefulness

| | team | env |
| --- | --- | --- |
| Reviewed | | |
| Accepted / overridden / rejected | | |
| Unresolved for scoring | | |
| Decided without a suggestion (withheld) | | |
| Needed evidence the inventory lacked | | |

- Ownership decisions that needed missing evidence, and what evidence was missing:

## Effort

- Median and p90 seconds per resource with obs-agent:
- Manual baseline: how it was sampled, resources compared, median and p90, time ratio:
- Main import and review friction (counts by category), and what caused it:

## Decision stability

- Re-reviewed resource labels, agreement:
- Disagreements and how each was resolved; any left unresolved:

## Decisions

- Expedited review (auto-apply threshold): adopt, keep review-only, or defer, with the evidence.
  Insufficient evidence never lowers a threshold.
- Recommended next feature, and why (link the open issue):
- Sign-off: reviewer, sample owner, date.
