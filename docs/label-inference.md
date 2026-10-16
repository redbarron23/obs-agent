# Label inference, review, and drift

## The problem

Cloud estates accumulate resources that predate a tagging policy, inherit old organisational names,
or have tags that became wrong after a reorganisation. `obs-agent` infers `team` and `env`, derives
`cost_center` from the selected team, and makes every uncertain result reviewable.

It differs from a policy engine: enforcement can identify a missing tag, but it cannot know which
value belongs there. This project predicts a likely value and exposes the evidence behind it.

## Inference pipeline

`labeling.py` trains only on resources with a usable observed label, then predicts the rest.

1. **Clean:** normalise variants such as `Prod` and `prd`, or `o11y` and `observability`.
2. **Rules:** inherit from a parent or name prefix only with at least 80% agreement across at least
   three labelled siblings; environment-name tokens are also high-precision evidence.
3. **Model:** logistic regression consumes resource-name tokens, hierarchy, creator and owners,
   service footprint, cloud, average spend, weekend ratio, and cost volatility.
4. **Mislabel check:** out-of-fold predictions flag existing tags that the model confidently
   disagrees with, without allowing a resource to vote on its own label.

Each prediction includes confidence, the strongest weighted features, and alternative values. Team
uses the model because hierarchy is unreliable after a reorganisation; environment uses whichever
of the rule and model is more confident.

## Ground-truth evaluation

The synthetic inventory stores hidden `true_*` values alongside observed tags. That makes the
evaluation reproducible and lets CI measure coverage, accuracy, macro-F1, auto-apply precision,
top-three recall, and expected calibration error (ECE).

| Label | Strategy | Coverage | Accuracy | Macro-F1 | Auto-apply precision |
|---|---:|---:|---:|---:|---:|
| team | rules | 41% | 94% | 0.81 | 94% |
| team | model | 100% | 98% | 0.98 | 100% |
| env | rules | 55% | 100% | 1.00 | 100% |
| env | hybrid | 100% | 81% | 0.79 | 99% |

Run the report with:

```bash
uv run labeling.py
uv run labeling.py --hardness both
uv run labeling.py --hardness reorg
```

The clean estate is deliberately tidy. The `harsh` scenario adds shared vocabulary, missing owners,
opaque names, and weak hierarchy. The `reorg` scenario isolates stale parents while preserving useful
names and ownership. Measure production confidence on a reviewed sample of currently unlabelled
resources; training only on already-tagged resources reproduces their selection bias.

## Review and safe handoff

The Label review page and API list missing or suspect tags, most confident first. A reviewer can
accept the recommendation, reject it, or override it with an allowed value.

### Confidence and deferral policy

`AUTO_APPLY_THRESHOLD` is **0.90**. Confidence is the model or rule probability for its proposed
value; it is not proof that the label is correct.

| Result | Policy |
| --- | --- |
| Untagged prediction at or above 90% | Eligible for expedited review. It may be bulk-accepted, but it is not sent to a provider automatically. |
| Untagged prediction below 90% | Deferred: it requires an explicit accept, reject, or override before export. |
| Existing tag that the model disputes | Always deferred, whatever its confidence. |
| No useful prediction | Leave the tag unchanged and investigate outside the tool. |

Exports default to reviewed decisions only. A user can deliberately include high-confidence,
unreviewed suggestions in a convenience export, but the reviewed change plan never includes them.
Before production use, validate the threshold on a representative reviewed sample and raise it or
disable expedited review if its precision falls below the team’s accepted risk level.

Every action is retained with the original suggestion, confidence, source, explanation, and
timestamp, reviewer ID, inventory fingerprint, and inference version. Imported vocabularies
control cleaning, environment-name rules, and override choices; see [Import inventory](importing.md).
A changed snapshot or inference version requires fresh approvals. Download a **Reviewed change plan** before any application step: it contains only
accepted or overridden changes, their before/after diff, and the associated approval evidence.

The plan is an approval artifact, not an apply command. After approval, export a CSV, Terraform
`tfvars`, or shell script for the team's existing delivery process. `obs-agent` does not have cloud
credentials and never mutates Azure or GCP directly.

## Drift detection

`drift.detect_drift(previous, current)` compares two snapshots of the same inventory. It returns
existing resources whose parent, owners, service footprint, average daily cost, weekend ratio, or
cost volatility changed enough to require a renewed label review.

Any increase from zero to positive average daily cost requires review. For a positive baseline,
the default spend threshold remains a relative change of at least 50%; zero-to-zero is unchanged.

```python
from drift import detect_drift

review_queue = detect_drift(last_month_inventory, current_inventory)
```

Snapshot collection belongs to the caller. The helper does not read cloud accounts, alter tags, or
decide that a drifted tag is wrong; it only makes review work visible.

`drift.compare_snapshots(previous, current)` builds on it for whole snapshots: it also returns new
and removed resources, tag edits, changed label suggestions, and a vocabulary-mismatch flag.
`review_workspaces.compare_reviews(previous_id, current_id)` runs it on two saved reviews and attaches
the earlier review's decisions for context without copying them. The **Compare snapshots** page uses
it; see [Importing](importing.md#compare-with-an-earlier-snapshot).
