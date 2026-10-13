# Importing a sanitized inventory

The local **Import inventory** page accepts a CSV snapshot from Azure and GCP so the existing
label-review workflow can run without cloud credentials, discovery, or tag writes. The public
demo intentionally disables imports: run the app locally for private data.

Start the app with `uv run streamlit run app.py`, choose **Import a sanitized inventory**, upload
the CSV, then open **Label review**. The upload is kept only for the current Streamlit browser
session. Decisions use a temporary local SQLite file; exports are plans for a reviewer to inspect,
not provider mutations.

## CSV schema

Every row is an Azure subscription or GCP project. These columns are required:

| Column | Notes |
| --- | --- |
| `cloud` | `azure` or `gcp` |
| `resource_id` | Non-empty, unique subscription/project identifier |
| `parent` | Management group or folder path |
| `owners`, `top_services` | JSON string list, or `|`-separated values |
| `created_by` | Creator identifier |
| `avg_daily_cost`, `weekend_ratio`, `cost_cv` | Numeric cost-shape signals |
| `team`, `env`, `cost_center` | Current tag values; blank tags are reviewed |

`billing_account`, `created_at`, and `region` are accepted as optional context columns. The
importer rejects invalid clouds, duplicate resource IDs, invalid list fields, and non-numeric cost
signals.

The model needs enough examples to safely fit and evaluate a local review snapshot: each of
`team` and `env` must contain at least two recognized values, with two labelled resources for each
value. Values use the same supported vocabulary and aliases as the bundled workflow. See
[`fixtures/sanitized_inventory_sample.csv`](../fixtures/sanitized_inventory_sample.csv) for a tiny
two-cloud example with one deliberately missing label.

This import path is deliberately separate from the REST API and agent tools. It does not change
their synthetic-data contract or grant the application access to cloud accounts.
