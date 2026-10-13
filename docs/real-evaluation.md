# Real-world evaluation pack

Synthetic metrics show that the pipeline works on the deliberately constructed inventory. They do
not establish production performance. This pack measures agreement between suggestions and
human-resolved labels on a representative, approved sample of **50–100 real resources**.

## 1. Build an approved anonymous sample

Sample resources across Azure and GCP, teams, environments, age, cost bands, and both easy and
ambiguous-looking names. Include missing tags and suspected-wrong tags; do not select only
high-confidence predictions. Draw the sample before reviewers see suggestions.

Do this in an approved private workspace:

1. Export only the fields needed to make a label decision. Keep the original export and any
   identifier mapping in the approved environment; never add either to this repository.
2. Replace subscription, project, folder, resource, owner, creator, and billing-account values
   with stable random surrogates. Remove email addresses, URLs, ticket IDs, free-text notes, and
   exact timestamps. Bucket costs and regions if they are needed for review.
3. Verify that a surrogate cannot be reversed without the separately stored mapping. Have a second
   person spot-check 10 rows for identifiers and sensitive free text.
4. Review each `team` and `env` suggestion independently, then record the resolved value as
   accept, reject-with-existing-label, or override. An untagged rejection is unresolved and cannot
   score the model.

The tool intentionally consumes reviewer feedback only: surrogate resource ID, label, decision,
suggested value, resolved value, confidence, and timestamp. It does not need cloud credentials,
an inventory export, or a mapping table.

## 2. Run the pack

Save the approved feedback JSON outside git, then run:

```bash
uv run real_evaluation.py /approved/path/reviewer-feedback.json
```

The command rejects samples outside 50–100 unique resources. It reports overall reviewer-agreement
macro precision/recall/F1, the synthetic baseline for context, and high-confidence observed
precision at 90%, 95%, and 98% confidence.

## 3. Set the production policy from evidence

By default, a candidate threshold needs at least 20 resolved examples **for both labels** and at
least 98% observed agreement. The first candidate meeting both conditions is recommended for a
monitored pilot. If none qualifies, the result explicitly says to defer auto-apply; keep the
review-only workflow instead.

These defaults are a conservative starting policy, not a claim about this repository's accuracy.
Set `--min-support` and `--min-precision` only with the owning team's documented risk tolerance.
After a pilot, rerun the pack on new, representative feedback and lower the threshold only with
fresh supporting evidence.

The JSON report includes `synthetic_baseline_not_a_production_gate` so the two results can be
compared without conflating them. A release record should retain the anonymisation approval, sample
selection notes, command output, chosen threshold, reviewer sign-off, and rollback owner.
