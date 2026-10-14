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
| Label review with an imported inventory | `<tmp>/obs-agent-import-<id>.db`, one per browser session |
| Public demo (`OBS_AGENT_DEMO=1`) | `<tmp>/obs-agent-<id>.db`, one per browser session |

`<tmp>` is `tempfile.gettempdir()`. The Label review page chooses the store at the top of
`pages/1_Label_review.py`: an imported inventory takes precedence over demo mode, and otherwise it
uses `DecisionStore()`. The API always uses `DecisionStore()`, so it never sees decisions made
against an imported inventory or in a demo session.

The app does not delete the per-session temp files. A session that ends simply stops using its
file, and the operating system's temp cleanup removes it eventually. Don't rely on these files for
anything you need to keep.

## Docker

The image creates `/app/state`, owned by the non-root `appuser`, and sets
`OBS_AGENT_DECISIONS=/app/state/label_decisions.db`. `docker-compose.yml` mounts the named volume
`label-state` there for both the `api` and `ui` services, so they share one decision history and it
survives container restarts and rebuilds.

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
| `previous` | The resource's tag value at decision time |
| `decided_at` | UTC timestamp (ISO 8601, seconds) |

A new decision for the same pair replaces the old one (`INSERT ... ON CONFLICT DO UPDATE`).

**`review_feedback`** is append-only. Every successful `set()` call adds a row with the same fields
plus the model snapshot the reviewer saw: `confidence`, `source` and `reason`. It feeds the
reviewer-agreement metrics (`GET /labels/feedback/metrics` and the quality panel on Label review)
and the approval evidence in the approved change plan export.

Before writing, `set()` checks that the resource exists, that the label and decision are known,
that an override value is allowed for that label, and that `accept` / `reject` have a suggestion to
act on. A rejected call writes nothing to either table.

Undoing a decision (`DELETE /resources/{id}/labels/{label}/decision`, or clearing it on the review
page) removes the row from `decisions` only. The `review_feedback` rows stay, and the undo itself
is not recorded there.

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
is the audit trail.
