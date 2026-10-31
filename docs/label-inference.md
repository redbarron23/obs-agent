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

Each prediction includes a model score (reported as `confidence`), the strongest weighted features,
and alternative values. Team uses the model because hierarchy is unreliable after a reorganisation;
environment uses whichever of the rule and model scores higher.

## Insufficient evidence

A model score can be high while the resource itself carries little or contradictory ownership
evidence. Before a `team` suggestion reaches the review flow, `LabelInference.assess()` checks the
resource's ownership features — owners, creator, name tokens and parent — against how often each was
seen on labelled resources. The first condition that holds withholds the suggestion:

| Evidence status | Condition | What the reviewer sees |
|---|---|---|
| `missing_signals` | No owner, creator or meaningful name token; only hierarchy, services and spend remain. | Record an owner, or override. |
| `insufficient_support` | None of the resource's owners, creator or name tokens appears on at least 2 labelled resources with the predicted team. | Confirm ownership, then override. |
| `conflicting` | The resource has decisive evidence (a feature seen on at least 2 labelled resources, at least 80% with one team) and none of it points to the predicted team. | The conflicting features and counts; resolve with the owners, then override. |
| `supported` | None of the above. | The suggestion, with the supporting features. |

The bars are the import minimum (two labelled resources per value) and the inheritance rules' 80%
agreement. They use features and training counts only, never the model score, so there is no
universal confidence cutoff: a low-scoring prediction with supporting evidence is still suggested,
and a high-scoring one without it is not. Parent counts towards a conflict but cannot support a
prediction on its own, because hierarchy goes stale after reorganisations. `env` is not checked:
name tokens and spend shape are its evidence.

An abstained label stays `untagged` with `suggested` and `confidence` set to null, `action:
"insufficient evidence"`, `source: "abstain"`, an actionable `reason`, and an `evidence` object
naming the status, the conflicting features and the withheld value (context only). It cannot be
accepted, rejected or auto-applied, implies no cost center, and is never part of an unreviewed
export. An explicit override with an allowed value is still recorded and exported as usual.
Coverage reports `insufficient_evidence` with a count per status, and the queue can be filtered by
evidence status (`evidence=insufficient` in the API, **Only insufficient evidence** in the UI).

## Ground-truth evaluation

The synthetic inventory stores hidden `true_*` values alongside observed tags. That makes the
evaluation reproducible and lets CI measure coverage, accuracy, macro-F1, auto-apply precision,
top-three recall, and expected calibration error (ECE).

| Label | Strategy | Coverage | Accuracy | Macro-F1 | Auto-apply precision |
|---|---:|---:|---:|---:|---:|
| team | rules | 41% | 94% | 0.81 | 94% |
| team | model | 100% | 98% | 0.98 | 100% |
| team | model with abstention | 97% | 99% | 0.99 | 100% |
| env | rules | 55% | 100% | 1.00 | 100% |
| env | hybrid | 100% | 81% | 0.79 | 99% |

The `abstain` row scores what reviewers see: the team model with insufficient-evidence cases
withheld. Accuracy is measured on the suggestions made, so read it with coverage. The report also
states how often the withheld value would have been right:

| Estate | Team model: coverage / accuracy | With abstention: coverage / accuracy | Withheld (by status) | Withheld value right |
|---|---:|---:|---|---:|
| clean | 100% / 98% | 97% / 99% | 5 of 170 (5 conflicting) | 60% |
| harsh | 100% / 85% | 77% / 94% | 49 of 212 (17 missing, 18 unsupported, 14 conflicting) | 53% |
| reorg | 100% / 98% | 97% / 100% | 5 of 160 (1 unsupported, 4 conflicting) | 40% |

Abstention trades coverage for accuracy where evidence is weak; it does not change the
expedited-review slice, whose precision was already 100% on these estates.

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

`AUTO_APPLY_THRESHOLD` is **0.90**. "Confidence" in reports and the API is a model score: the
logistic-regression probability or the rule's agreement share for its proposed value. It ranks
suggestions, but it is not a calibrated probability that the label is correct (ECE on the
clean and harsh synthetic estates is 17–18% for the team model) and should not be read as one until it has been calibrated
on a representative reviewed estate.

| Result | Policy |
| --- | --- |
| Untagged prediction with a model score at or above 90% | Eligible for expedited review. It may be bulk-accepted, but it is not sent to a provider automatically. |
| Untagged prediction below 90% | Deferred: it requires an explicit accept, reject, or override before export. |
| Existing tag that the model disputes | Always deferred, whatever its score. |
| Insufficient evidence (abstained) | No suggestion. Never expedited or exported unreviewed; only an explicit override resolves it. |

Exports default to reviewed decisions only. A user can deliberately include high-scoring,
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
