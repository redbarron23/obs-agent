# Deployment guardrails

This checklist turns the local review workflow into a production service without changing its core
boundary: obs-agent recommends and exports labels; it does not hold cloud credentials or mutate
Azure or GCP.

## Before the first production deployment

### API keys and service secrets

- [ ] Store provider keys and `OBS_AGENT_API_KEY` in the deployment platform's secret manager, not
  in git, images, `.env` files, Streamlit secrets committed to a repo, logs, or exports.
- [ ] Give each environment its own key and the minimum provider scope. Disable chat entirely when
  it is not needed.
- [ ] Set an owner and a documented rotation cadence. Rotate immediately after a suspected leak,
  staff departure, or provider incident.
- [ ] During rotation, deploy the replacement secret, run a health check, revoke the old secret,
  and record the date, owner, and affected environment. Never print either key while diagnosing a
  failed rotation.

### Persistent decisions and backups

How and where decisions are stored is described in [Decision storage](decision-storage.md).
The backup, verify and restore procedure, with its operator checklist, is in
[Backup and restore](backup-restore.md).

- [ ] Set `OBS_AGENT_DECISIONS` to a file on a persistent, access-controlled volume.
  The default local `label_decisions.db` is suitable only for local development.
- [ ] Set `OBS_AGENT_REVIEW_DIR` to a private persistent directory and back up the imported review
  databases too. They contain the uploaded inventories as well as approvals.
- [ ] Authenticate reviewers at the gateway or application boundary and bind the recorded reviewer
  ID to the authenticated subject; identifiers are self-reported locally and a shared API key does not identify a person. Retain authentication logs.
- [ ] Back up the SQLite decision database and every saved-review database with
  `scripts/review_backup.py` (SQLite online backup, safe while running) before deployments and on a
  scheduled cadence that meets the team's recovery-point objective. Encrypt backups and restrict restore access to the service
  owner and designated operators.
- [ ] Define backup retention, recovery-point objective (maximum acceptable lost decisions), and
  recovery-time objective (maximum acceptable review outage) with the data owner.
- [ ] Test a restore into an isolated environment at least once per quarter. Confirm both current
  decisions and the append-only `review_feedback` history are present and that the approved plans
  reproduce (`review_backup.py plans`); record the test result.

### Audit exports and retention

- [ ] Save each approved change plan with its approval evidence before applying an exported CSV,
  Terraform file, or shell script through the normal delivery system.
- [ ] Immediately before delivery, check the plan against a freshly exported inventory (see
  [Pre-delivery plan validation](#pre-delivery-plan-validation)). Deliver only on `pass`.
- [ ] Store exports in an access-controlled audit location. Retain them for the period required by
  the organisation's finance, security, and records policies; assign an owner for deletion when
  that period expires.
- [ ] Keep immutable metadata alongside each export: generation time, environment, commit/tag,
  decision-database backup identifier, approver, and a checksum of the artifact.
- [ ] Do not place cloud credentials, raw billing exports, unnecessary personal data, or reviewer free text in
  an audit export. Use stable reviewer IDs rather than email addresses when possible.

## Pre-delivery plan validation

Resources can change between review and delivery. Before the delivery system applies an export
derived from a reviewed plan, export the inventory again and run the read-only check:

```bash
uv run python plan_validation.py plan.json fresh.csv --review-id <saved-review-id> --output report.json
# or: --source reviewed.csv [--vocabulary vocab.json] when the snapshot is not a saved review
# in the image: docker compose exec api python plan_validation.py ...
```

The JSON report (`report_version` 1) lists every planned resource with its recorded before-value,
planned value, observed value, and findings with reasons. Exit status is 0 for `pass`, 1 for `fail`
(renewed review needed), and 2 for an invalid plan or unreadable input.

| Outcome | Detected when |
| --- | --- |
| `invalid` plan | Wrong schema version, not reviewed-only, malformed changes, or a tag without a matching approval (reviewer, decision time, value, and the plan's `inventory_id`/`model_version`) |
| `previous_value_changed` | A planned tag is neither its recorded before-value nor the planned value |
| `already_applied` | A planned tag already has the planned value (GCP compared lowercased, as exports write it) |
| `resource_missing` | A planned resource is absent from the fresh inventory |
| `cloud_changed` / `evidence_changed` | The resource moved cloud, or (with the source snapshot) its parent, owners, services or cost shape drifted since review |
| `inference_version_changed` | The fresh inventory's vocabulary or the inference implementation no longer yields the plan's `model_version` |
| `source_snapshot_mismatch` / `plan_inconsistent_with_source` | The supplied source is not the reviewed snapshot, or the plan's before-values do not match it |

Unrelated changes elsewhere in the estate do not fail the check. Without `--review-id`/`--source`
only tag before-values are compared, and the report warns about it. The check never changes tags,
decisions, snapshots or plans, and never moves an approval to the fresh inventory: on `fail`, import
the fresh inventory, review the flagged resources again, and export a new plan.

## Confidence-threshold changes

`AUTO_APPLY_THRESHOLD` controls only expedited review eligibility. It never authorizes direct
provider changes. A lower threshold increases the number of suggestions that may be bulk-reviewed,
so treat every change as a production policy change.

Before changing it:

- [ ] Run the [real-world evaluation pack](real-evaluation.md) on a newly sampled, approved,
  anonymized 50–100-resource review set. Synthetic scores alone are not evidence for a change.
- [ ] Record the old and proposed threshold, sample-selection notes, resolved-example counts by
  label, observed precision, macro precision/recall/F1, and the evaluator output.
- [ ] Obtain approval from the service owner and the team accountable for tagging/FinOps policy.
  Include security or compliance approval when required by the environment's change policy.
- [ ] Make the change in a reviewed pull request, update the documented policy, run the full test
  suite, and deploy first to a monitored pilot.
- [ ] Set a rollback owner and criterion before deployment—for example, observed reviewer agreement
  below the approved floor or a material increase in overrides. Revert the threshold rather than
  silently changing reviewer behaviour.

Use this change record in the approval ticket:

| Field | Record |
| --- | --- |
| Environment / release | |
| Previous → proposed threshold | |
| Evaluation sample and date | |
| Per-label support and observed precision | |
| Approval links and approvers | |
| Pilot window and monitoring owner | |
| Rollback criterion and owner | |

## Ongoing operations

- [ ] Scrape `/ops/metrics?format=prometheus` (or collect the structured `obs_agent.workflow`
  log lines) and set the alerts described in [Operations](operations.md).
- [ ] Review decision-database backup jobs and restore-test evidence on the same cadence as
  production access reviews.
- [ ] Review audit-export retention and access at least annually or when policy changes.
- [ ] Monitor reviewer agreement and unresolved decisions. Pause expedited review if the approved
  threshold no longer meets its evidence-based policy.
- [ ] Rehearse the service's incident path: revoke exposed keys, stop exports, preserve audit
  evidence, restore decisions if needed, and communicate the review impact.
