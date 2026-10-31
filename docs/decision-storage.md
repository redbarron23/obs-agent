# Decision storage

Reviewer decisions (accept / reject / override) are stored in a local SQLite file by
`DecisionStore` in `label_review.py`. The REST API and the Streamlit **Label review** page both use
it, and every export (CSV, shell script, tfvars, approved change plan) is resolved from it. Only
SQLite is supported; there is no other backend.

## Where the file lives

`DecisionStore(path=None)` picks its file in this order:

1. the `path` argument, if one is passed;
2. the `OBS_AGENT_DECISIONS` environment variable;
3. `label_decisions.db` in the process's working directory.

Missing parent directories are created, but the process needs write access to them. Relative paths
resolve from the working directory, so use an absolute path anywhere but a local checkout.
`label_decisions.db` is in `.gitignore`.

| Where it runs | File |
| --- | --- |
| Local checkout (`uv run streamlit run app.py`, `uv run uvicorn api:app`) | `./label_decisions.db`, or `OBS_AGENT_DECISIONS` |
| Docker image / `docker compose` | `/app/state/label_decisions.db` (set in the `Dockerfile`) |
| Saved imported review | `review-workspaces/<snapshot-and-vocabulary-hash>.db`, or `OBS_AGENT_REVIEW_DIR` |
| Public demo (`OBS_AGENT_DEMO=1`) | `<tmp>/obs-agent-<id>.db`, one per browser session |

`<tmp>` is `tempfile.gettempdir()`. Demo mode always uses a disposable store and synthetic data.
Locally, a selected saved review uses its own database; otherwise the page uses `DecisionStore()`. The API always uses `DecisionStore()`, so it never sees decisions made
against an imported inventory or in a demo session.

Saved imports persist across browser/app restarts and can be reopened on the import page.
Their databases also contain a `workspace` table with the original CSV, vocabulary JSON, name, and
creation time. Each snapshot/vocabulary gets a separate database. All users of this installation
share the saved-review directory; use it only for one trusted team. Back up this directory along
with the synthetic decision database (see [Backup and restore](backup-restore.md)).

The app does not delete the public demo's per-session temp files. A session that ends simply stops using its
file, and the operating system's temp cleanup removes it eventually. Don't rely on these files for
anything you need to keep.

## Docker

The image creates `/app/state`, owned by the non-root `appuser`, and sets
`OBS_AGENT_DECISIONS=/app/state/label_decisions.db`. `docker-compose.yml` mounts the named volume
`label-state` there for both the `api` and `ui` services, so they share one decision history and it
survives container restarts and rebuilds. Imported reviews use `/app/state/reviews` in the same volume.

To use a different location, override `OBS_AGENT_DECISIONS` and mount a volume at that path. The
directory must be writable by `appuser`.

## What is stored

Two tables are created on first use.

**`decisions`** holds the current decision for each `(resource_id, label)` pair, which is the
table's primary key:

| Column | Meaning |
| --- | --- |
| `decision` | `accept`, `reject` or `override` |
| `value` | The resolved value: the suggestion for `accept`, the reviewer's value for `override`, `NULL` for `reject` |
| `suggested` | The model's suggestion at decision time |
| `previous` | The resource's normalized tag value at decision time; exported diffs use its raw value |
| `decided_at` | UTC timestamp (ISO 8601, seconds) |
| `reviewer` | Reviewer name or ID entered in the UI/request, or `OBS_AGENT_REVIEWER` |
| `inventory_id` | SHA-256 fingerprint of the inventory snapshot |
| `model_version` | Fingerprint of inference code, scikit-learn/NumPy/SciPy versions, vocabulary, strategy, and threshold |

A new decision for the same pair replaces the old one (`INSERT ... ON CONFLICT DO UPDATE`).

**`review_feedback`** is append-only. Every successful `set()` call adds a row with the same fields
plus the model snapshot the reviewer saw: `confidence`, `source` and `reason`. Every successful
revocation adds a `revoke` row (see [Revoking a decision](#revoking-a-decision)). It feeds the
reviewer-agreement metrics (`GET /labels/feedback/metrics` and the quality panel on Label review)
and the approval evidence in the approved change plan export.

Missing provenance and revocation columns are added automatically to existing databases (an
additive `ALTER TABLE ... ADD COLUMN` inside the same transaction that opens the store; nothing is
rewritten or dropped). Historical rows retain null values; they are not backfilled with an invented
reviewer/version, and decisions undone before this change gain no invented revocation. Only decisions whose
inventory and model fingerprints match the current review are used for exports. Re-review after
input, vocabulary, inference code, or model-version changes. A reviewed plan (schema version 2)
contains these fingerprints at the top level and on its approval evidence.

Reviewer names are self-reported; they do not prove an authenticated person approved the change.
Put an authenticated reviewer boundary in front of a production service and retain its access logs.

Before writing, `set()` checks that the resource exists, that the label and decision are known,
that a reviewer identifier is present, that an override value is allowed for that label, and that `accept` / `reject` have a suggestion to
act on. A rejected call writes nothing to either table.

## Revoking a decision

Undoing a decision (`DELETE /resources/{id}/labels/{label}/decision`, or clearing it on the review
page) is a revocation. `DecisionStore.revoke()` removes the row from `decisions` and appends a
`revoke` event to `review_feedback` in one transaction: if either write fails, neither happens.

| Column on a `revoke` row | Meaning |
| --- | --- |
| `reviewer`, `decided_at` | Who revoked it (self-reported, as for decisions) and when (UTC) |
| `revocation_reason` | Optional free-text reason (at most 500 characters) |
| `revoked_decision`, `revoked_reviewer`, `revoked_decided_at` | The decision being withdrawn and who made it when |
| `value`, `suggested`, `previous`, `inventory_id`, `model_version` | That decision's approval context |
| `revoked_feedback_id` | The `review_feedback` row that recorded it (null for legacy JSON imports) |

`confidence`, `source` and `reason` are null on `revoke` rows; the linked row has the model context.
The API takes `{"reviewer": ..., "reason": ...}` as an optional JSON body on `DELETE` (the reviewer
falls back to `OBS_AGENT_REVIEWER`) and returns the event. The review page uses the reviewer field
and an optional reason box.

- A missing reviewer, unknown label, or overlong reason is rejected before anything is written.
- Revoking a decision that does not exist (including a second revoke) writes nothing: the store
  returns `None` and the API returns 404.
- A revoked decision is not in `decisions`, so it never reaches exports, the approved change plan
  or plan validation. A plan's approval evidence is the latest non-`revoke` row for each label.
- Recording a new decision after a revocation works like any other decision and appends a new row.
- Reviewer-agreement metrics never score `revoke` rows: they are reported as `revoked`, not as
  `reviewed`, and `real_evaluation.py` ignores them.
- The reason is review history only. It is returned by `GET /labels/feedback` and kept in backups,
  but never written to logs or workflow metrics, and the approved change plan omits revocation
  columns. Treat it like reviewer IDs when sharing exports of the feedback history.

## Legacy JSON files

Decisions used to be stored as JSON. When the SQLite file does not exist yet, `DecisionStore` looks
for a JSON file to import into `decisions`:

- If the configured path ends in `.json` (for example `OBS_AGENT_DECISIONS=decisions.json`), the
  store uses `decisions.db` alongside it and imports `decisions.json`.
- Otherwise it imports a JSON file with the same name as the database: `label_decisions.json` next
  to `label_decisions.db`.

The import runs once, when the database is created, and fills only `decisions`. Legacy files
carried no model snapshot, so they add no `review_feedback` history.

## Concurrency

Each operation opens its own connection with a 5-second busy timeout, so the API and UI can write
to the same file. This relies on SQLite file locking. Keep the file on a local disk or a
block-storage volume; network file shares (NFS, SMB / Azure Files) often don't support those locks
reliably.

## Production

[Deployment guardrails](deployment.md#persistent-decisions-and-backups) has the checklist. In
short: put the file on a persistent, access-controlled volume, back it up on a schedule, and test
restores. A restore must bring back both `decisions` and `review_feedback`, since the feedback table
is the audit trail. [Backup and restore](backup-restore.md) has the procedure: an online backup of
the decision database and every saved review, plus a restore check that the approved plans
reproduce.
