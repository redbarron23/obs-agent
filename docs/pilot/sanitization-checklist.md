# Pilot sanitization checklist

Complete this in the approved private workspace before anyone opens the sample in obs-agent. Keep
the signed copy with the pilot record, not in this repository. Method:
[Real-world evaluation pack](../real-evaluation.md#1-build-an-approved-anonymous-sample).

Sample: ______________ Approved by: ______________ Date: __________

## Selection (before anyone sees suggestions)

- [ ] 50–100 Azure subscriptions and/or GCP projects, drawn before running inference.
- [ ] Spans clouds, teams, environments, age and cost bands.
- [ ] Includes untagged labels, suspected-wrong tags and ambiguous names, not only easy cases.
- [ ] How the sample was drawn is written down (random, stratified by team, and so on).

## Fields

- [ ] Only the inventory columns in `inventory-template.csv`; every other export column removed.
- [ ] `resource_id`, `parent`, `owners`, `created_by` replaced with stable random surrogates
      (the same person or folder gets the same surrogate everywhere).
- [ ] Surrogates keep any naming tokens the review needs (for example `prod`), and nothing that
      identifies the organisation.
- [ ] No email addresses, URLs, ticket IDs, billing-account IDs, free-text notes or exact
      timestamps.
- [ ] Costs bucketed or rounded if exact spend is sensitive; `weekend_ratio` and `cost_cv` kept.
- [ ] `team`, `env`, `cost_center` values use the estate's own vocabulary
      (`vocabulary.json`, shaped like `fixtures/label_config_example.json`).

## Mapping and storage

- [ ] The surrogate-to-real mapping is stored separately from the sample, with restricted access.
- [ ] A surrogate cannot be reversed without that mapping.
- [ ] The sample, vocabulary, logs, saved review and backups stay in the approved workspace
      (for example `pilot-workspace/`, which git ignores). None of it is committed.

## Second-person check

- [ ] A second person spot-checked at least 10 rows for identifiers and sensitive text.
      Checked by: ______________ Rows checked: ____ Issues found and fixed: ____

## Logs

- [ ] Review, baseline and re-review logs use surrogate IDs, numbers and the fixed categories
      only. Private notes, if any, are kept outside these logs.
