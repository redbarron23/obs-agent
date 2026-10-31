# Backup and restore

A complete review workspace is a set of SQLite files: the main decision database and one database
per saved review. This guide shows how to back them up consistently, restore them into an isolated
directory, and check that the restore reproduces the reviewed change plans. `tests/test_backup_restore.py`
runs the same round trip offline in CI.

Where the files live and what their tables hold is described in [Decision storage](decision-storage.md).

## What a backup must contain

| File | Path setting | Contains |
| --- | --- | --- |
| Main decision database | `OBS_AGENT_DECISIONS` (default `./label_decisions.db`; Docker `/app/state/label_decisions.db`) | `decisions` and `review_feedback` for the synthetic estate and the REST API |
| One database per saved review | `OBS_AGENT_REVIEW_DIR/<64 hex>.db` (default `./review-workspaces/`; Docker `/app/state/reviews/`) | `workspace` (uploaded inventory CSV, vocabulary JSON, name, creation time), `decisions`, `review_feedback` |

Back up every file together so that inventories, vocabularies, active decisions and the append-only
feedback history come from the same point in time. Revoked decisions are absent from `decisions`
and recorded as `revoke` events in `review_feedback`; both tables are restored as they were.

Not included, by design: the public demo's per-session temp databases, exported files (CSV, shell,
tfvars, plans), `.env` secrets and API keys, MLflow data, and the in-memory review cache, which is
rebuilt on first use.

The saved-review databases contain uploaded inventories and reviewer identifiers. Treat each
backup as being as sensitive as the live review directory.

## Back up a running installation

Don't copy a live `.db` file with `cp`, `tar` or `rsync`: a copy taken while the API or UI writes
can mix pages from before and after a transaction. `scripts/review_backup.py` uses SQLite's online
backup API instead. Each copy contains only committed transactions, and the app can keep serving
reviews while it runs. A filesystem or volume snapshot that captures the whole directory atomically
is also safe, because SQLite is crash-consistent. Copying files one by one is not.

```bash
uv run python scripts/review_backup.py backup backups/$(date -u +%Y%m%dT%H%M%SZ)
```

The script reads `OBS_AGENT_DECISIONS` and `OBS_AGENT_REVIEW_DIR`, or `--decisions` and
`--review-dir`, the same way the app does. It opens the sources read-only and refuses to write into
an existing backup directory. It writes:

- `label_decisions.db` and `reviews/<id>.db`, each with mode `0600`, in a directory with mode `0700`;
- `manifest.json`, with each file's SHA-256, `PRAGMA integrity_check` result, source path and row counts
  for `decisions`, `review_feedback` and `workspace`.

Each database is copied as one consistent snapshot, but the files are copied one after another. A
decision recorded during the backup can therefore appear in a later file and not an earlier one.
The decisions in each file are still internally consistent. For a single point in time across all
files, take the backup during a quiet period or stop writes first.

The script verifies the backup before it exits. To check it again later, for example after it has
been copied to off-site storage:

```bash
uv run python scripts/review_backup.py verify backups/<stamp>
```

`verify` exits non-zero if a file is missing, a checksum or row count differs from the manifest, a
saved review has no inventory or vocabulary, `integrity_check` fails, or a file is readable by group
or others.

Also record the approved change plans at backup time. You compare them against the restore later:

```bash
uv run python scripts/review_backup.py plans > backups/<stamp>.plans.json
```

`plans` prints the reviewed-only plan for the main database and for each saved review, without
`generated_at`.

### Docker Compose

The image sets both path variables and has no copy of the script, so pipe the script in. `backup`,
`verify` and `restore` use only the standard library.

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker compose exec -T api python - backup /tmp/obs-backup < scripts/review_backup.py
docker compose exec -T api python - plans < scripts/review_backup.py > "backups/$stamp.plans.json"
docker compose cp api:/tmp/obs-backup "backups/$stamp"
docker compose exec api rm -rf /tmp/obs-backup
chmod -R go-rwx "backups/$stamp"
```

## Restore into an isolated directory

Test every restore in a new directory before you rely on it. `restore` verifies the backup first.
It then copies the files to the configured paths with mode `0600` (review directory `0700`) and
refuses to overwrite any existing file.

```bash
restore=$(mktemp -d)
export OBS_AGENT_DECISIONS="$restore/label_decisions.db" OBS_AGENT_REVIEW_DIR="$restore/reviews"
uv run python scripts/review_backup.py restore backups/<stamp>
uv run python scripts/review_backup.py plans > "$restore/plans.json"
diff backups/<stamp>.plans.json "$restore/plans.json" && echo "plans reproduce"
uv run streamlit run app.py   # optional: Import inventory → Open saved review
```

Use the same commit and locked dependencies (`uv sync --locked`) as the source installation. With
those, the diff must be empty: the same effective changes, the same approval evidence (reviewer,
decision, model snapshot, `inventory_id`, `model_version`) and the same fingerprints. Opening a
saved review in the UI should show the uploaded inventory, its custom vocabulary in the override
choices, and its active decisions.

For a Docker installation, run the check in the same image. The restore goes to `/tmp` inside a
one-off container; the live volume is mounted but not written:

```bash
docker compose run --rm --no-deps -T \
  -v "$PWD/backups/<stamp>:/backup:ro" -v "$PWD/scripts:/scripts:ro" \
  -e OBS_AGENT_DECISIONS=/tmp/r/label_decisions.db -e OBS_AGENT_REVIEW_DIR=/tmp/r/reviews \
  api sh -c 'python /scripts/review_backup.py restore /backup >&2 &&
             python /scripts/review_backup.py plans' > restored.plans.json
diff "backups/<stamp>.plans.json" restored.plans.json && echo "plans reproduce"
```

The container's `appuser` must be able to read the bind-mounted backup. On Linux, give the backup
to that UID (`docker compose run --rm --no-deps api id -u`), or use a group it belongs to.

### Restores do not carry approvals to new inference

An approval applies only to the inventory snapshot and inference context it was made against (see
[Decision storage](decision-storage.md#what-is-stored)). If the restored installation runs different
inference code, scikit-learn/NumPy/SciPy versions, strategy, threshold or vocabulary, its
`model_version` differs. The restored decisions and feedback remain available for audit, but the
approved plan contains no changes until reviewers approve again. A non-empty `plans` diff after a
restore means that re-review is required. It is not a restore failure, and you should not edit the
database to make the fingerprints match.

## Restore into production

1. Stop the writers: `docker compose stop api ui`, or stop the local processes.
2. Move the current state aside rather than deleting it. `restore` refuses to overwrite existing files.
3. Run `restore` with the production `OBS_AGENT_DECISIONS` and `OBS_AGENT_REVIEW_DIR`.
4. Start the application, run `plans`, and diff it against the plans recorded with the backup.

With Docker Compose:

```bash
docker compose stop api ui
docker compose run --rm --no-deps \
  -v "$PWD/backups/<stamp>:/backup:ro" -v "$PWD/scripts:/scripts:ro" api sh -c '
    set -e; old=/app/state/pre-restore-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 "$old"
    if [ -e /app/state/label_decisions.db ]; then mv /app/state/label_decisions.db "$old/"; fi
    if [ -e /app/state/reviews ]; then mv /app/state/reviews "$old/"; fi
    python /scripts/review_backup.py restore /backup'
docker compose up -d
```

Decisions recorded after the backup was taken are lost. That gap is what the recovery-point
objective below bounds.

## Operator checklist

- [ ] **Scope:** back up the main decision database and every file in the saved-review directory
  together, using `backup` or an atomic volume snapshot, never a plain file copy of live databases.
- [ ] **Freshness:** schedule backups at least as often as the recovery-point objective allows, and
  before every deployment. Alert when the newest verified backup is older than the RPO.
- [ ] **Verification:** run `verify` after every transfer to backup storage, and keep each backup's
  `plans` output with it.
- [ ] **Retention:** agree a retention period with the data owner. Saved reviews contain uploaded
  inventories and reviewer IDs, so delete expired backups as well as rotating them.
- [ ] **Encryption and access:** encrypt backups at rest and in transit (an encrypted bucket or
  `age`/`gpg`). Restrict read and restore access to the service owner and designated operators.
  Keep file modes `0600` and directory modes `0700`, and audit access to backup storage.
- [ ] **Recovery objectives:** record the RPO (how many decisions you can afford to lose) and the
  RTO (how long review can be unavailable) for each deployment. Time each restore test against the
  RTO.
- [ ] **Restore test:** at least once a quarter, restore into an isolated directory as described
  above. Record the backup identifier, manifest checksum, `verify` result, `plans` diff result,
  elapsed time, and operator.
- [ ] **After an inference change:** expect restored approvals not to export until they are reviewed
  again. Plan re-review time into the RTO for restores across code or dependency upgrades.
