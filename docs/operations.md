# Operations: workflow metrics and error correlation

The review workflow (import → inference → decision write → export) is measured by
`workflow_metrics.py`. It uses only the standard library, so it works in the Docker image and CI
without MLflow. Agent tracing is separate and optional; see the [MLflow guide](mlflow.md).

Scope: one trusted-team deployment. Counters live in each process's memory and reset on restart.
This is not a monitoring platform. Scrape or ship them to the one you already run.

## What is measured

Every workflow step increments `obs_agent_workflow_operations_total{operation, surface, outcome}`
and records its duration in `obs_agent_workflow_operation_duration_seconds{operation, surface}`.
Labels are fixed sets. Nothing from the inventory, reviewer or request becomes a label.

| Label | Values |
| --- | --- |
| `operation` | `import`, `inference`, `decision_write`, `export` |
| `surface` | `api` (REST API), `ui` (Streamlit pages), `other` (CLI, agent tool, tests) |
| `outcome` | `success`, `invalid` (input was rejected: bad CSV or vocabulary, bad decision value, invalid export value), `error` (anything else; investigate) |

| Operation | Where it is recorded |
| --- | --- |
| `import` | Import page: CSV/vocabulary validation, then saving the review workspace (two operations per saved import) |
| `inference` | Fitting suggestions for a snapshot (`label_review`), once per snapshot and inference version. Cache hits are not counted. Includes the abstention checks. |
| `decision_write` | `PUT`/`DELETE …/decision` and the review page's Save and Accept-all buttons, one per decision |
| `export` | `GET /labels/export` per request. The review page prepares every download on each render, so `ui` export counts are renders, not downloads. |

`obs_agent_workflow_telemetry_errors_total` counts measurements that could not be recorded.
Telemetry failures are swallowed: they never fail an import, review or export.

## Review backlog

`GET /labels/backlog` (and the `backlog` block of `/ops/metrics`) describes the snapshot under
review. *Pending work* is current state. *Feedback actions* are history. They are reported
separately on purpose.

| Field | Definition |
| --- | --- |
| `pending_items` | (resource, label) pairs that are `untagged` or `suspect` and have no decision bound to the current snapshot and inference version. Labels withheld for insufficient evidence count, because they still need a reviewer's override. |
| `pending_resources` | Distinct resources with at least one pending item |
| `pending_auto_apply_ready` / `pending_insufficient_evidence` | Subsets of `pending_items` by suggested action |
| `reviewed_items` | Decisions bound to the current snapshot and inference version |
| `stale_decisions` | Decisions made against an earlier snapshot or inference version. They are not exported and must be reviewed again. |
| `feedback_actions` | All rows in the append-only `review_feedback` history, including superseded and earlier-snapshot actions and `revoke` events. Not backlog. |
| `review_started_at` | When the saved review workspace was created. For the default (synthetic) store, the first feedback action for the current snapshot. Null if neither exists. |
| `oldest_pending_age_seconds` | Now minus `review_started_at` while `pending_items > 0`. `0` when nothing is pending; null when the start is unknown. |
| `last_action_age_seconds` | Time since the most recent feedback action for the current snapshot. A large value with pending work means review has stalled. |

`/labels/backlog` serves the synthetic inventory and `OBS_AGENT_DECISIONS`. For a saved review use
`GET /reviews/{review_id}/backlog`, `GET /ops/metrics?review_id=...`, or
`uv run python review_cli.py backlog REVIEW_ID`.

## Retrieving the measurements

```bash
# JSON: counters, duration summaries, backlog, MLflow tracing status
curl -s -H "X-API-Key: $OBS_AGENT_API_KEY" http://localhost:8000/ops/metrics

# Prometheus text exposition, for a scraper
curl -s -H "X-API-Key: $OBS_AGENT_API_KEY" "http://localhost:8000/ops/metrics?format=prometheus"
```

`/ops/metrics` needs the API key, like every route except `/health`. Configure the scraper with
the key as an `X-API-Key` header.

The Streamlit UI runs in its own process and has no metrics endpoint. Each operation there writes
one JSON log line to the `obs_agent.workflow` logger:

```json
{"correlation_id": "4f1c9a0e2b7d4c11", "duration_ms": 812.4, "event": "workflow_operation",
 "operation": "inference", "outcome": "success", "surface": "ui"}
```

Failures log at WARNING (`invalid`) or ERROR (`error`) and always reach stderr. Successes log at
INFO, so set `OBS_AGENT_LOG_LEVEL=INFO` to see them, or `DEBUG` to add tracebacks. Count or chart
the UI's lines in your log pipeline.

## Correlating a failure

- **API:** every response has an `X-Request-ID` header. A client may send its own (1–64 letters,
  digits, `.`, `_` or `-`), which is echoed back; otherwise one is generated. An unexpected error
  returns `500 {"detail": "Internal server error", "correlation_id": "…"}`.
- **UI:** an unexpected failure shows `Reference: <id>` next to the error.
- **Server:** search the logs for that ID. The `workflow_operation` line names the step and
  `error_type`. A `request_failed` line names the route template and HTTP method.

```bash
docker compose logs api ui | grep 4f1c9a0e2b7d4c11
```

Routine log lines carry the operation, surface, outcome, duration, correlation ID and exception
**type** only. They never contain resource IDs, owners, reviewer names, override values, inventory
fingerprints, credentials, raw CSV content or exception messages. Tracebacks can quote those, so
they are written only at `DEBUG`. Enable `DEBUG` briefly to diagnose a reproduced failure, then
turn it off.

## Diagnosing the main failure cases

| Symptom | Likely cause | Check |
| --- | --- | --- |
| `import` `invalid` | Malformed CSV, unmapped column or bad vocabulary. The UI shows the reason to the user. | Usually not an operator issue. A sudden rise after an upstream export change means the source format changed. |
| `import` `error` | Review directory not writable or full; SQLite failure | `OBS_AGENT_REVIEW_DIR` permissions and free space |
| Slow `inference` | Large inventory or an overloaded host. It runs once per new snapshot, so the first page load after an import is slowest. | The `inference` duration histogram; host CPU and memory |
| `inference` `error` | Out of memory, or an inventory the model cannot fit | Reproduce with `OBS_AGENT_LOG_LEVEL=DEBUG` for the traceback |
| `decision_write` `error` | SQLite `database is locked` (contention beyond the 5 s busy timeout), read-only or full volume | `OBS_AGENT_DECISIONS` volume; other writers; [Backup and restore](backup-restore.md) |
| `export` `invalid` | A resolved value fails Azure/GCP tag rules. The message names the resource to the user. | Correct the inventory and import again |
| `export` `error` | Decision database unreadable | As for `decision_write` |
| `telemetry_errors_total` rising | The logging handler is failing (for example, a full disk or a broken log shipper) | The process's log destination. The workflow itself is unaffected. |

## Choosing alert thresholds

Volumes in a single-team deployment are small, so alert on counts over a window, not on rates.
Start with these values and adjust them after two to four weeks of baseline data:

| Alert | Starting point | Rationale |
| --- | --- | --- |
| Any `outcome="error"` for `decision_write` or `export` | ≥ 1 in 15 min, page or ticket | Lost or blocked approvals. There should normally be none. |
| `import` or `inference` `error` | ≥ 1 in 1 h, ticket | Blocks a new review but loses no data |
| `inference` p95 duration | Above 3× the observed baseline p95 (for example, 30 s) | Catches resource starvation. Set it from your inventory size, not from synthetic data. |
| `invalid` ratio | More than 50 % of `import` attempts over a day | An upstream export format change, not a user mistake |
| `oldest_pending_age_seconds` | Above the team's review service level (for example, 7 days) with `last_action_age_seconds` above 2 days | Review has stalled. Make this a ticket, not a page. |
| `pending_items` | No static threshold. Watch the trend after each import. | Backlog size depends on the estate and the untagged rate. |
| `telemetry_errors_total` increasing | ≥ 1 in 1 h, ticket | The measurements themselves are becoming unreliable |

Counters reset on restart, so use `increase()`-style queries that tolerate resets. Run one API
replica. With more replicas, each one reports its own counters and you sum them in the
monitoring system.
