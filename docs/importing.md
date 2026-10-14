# Importing and reopening an inventory

The local **Import / reopen inventory** page accepts a sanitized CSV of Azure subscriptions and
GCP projects. It needs no cloud credentials and never writes tags. The public demo disables imports.

```bash
uv run streamlit run app.py
```

1. Open **Import / reopen inventory**. Download the CSV example/template and vocabulary example.
2. Upload your CSV and, optionally, a vocabulary JSON matching your own organisation.
3. Inspect the validation summary and data preview, then select **Save and open review**.
4. Open **Label review**, enter your reviewer name or ID, and accept, reject, or override suggestions.
5. Download **Reviewed change plan** and inspect it before handing changes to your delivery system.

## CSV schema

Each row is an Azure subscription or GCP project. These columns are required:

| Column | Notes |
| --- | --- |
| `cloud` | `azure` or `gcp` |
| `resource_id` | Non-empty, unique subscription/project identifier |
| `parent` | Management group or folder path |
| `owners`, `top_services` | JSON string list, or `|`-separated values |
| `created_by` | Creator identifier |
| `avg_daily_cost`, `weekend_ratio`, `cost_cv` | Finite, non-negative numeric cost-shape signals |
| `team`, `env`, `cost_center` | Current tag values; missing tags are left blank |

`billing_account`, `created_at`, and `region` are optional context columns.
The importer rejects missing columns/clouds/IDs, duplicate IDs, invalid lists, and invalid numbers.

For **both** team and env, provide at least two recognized values with at least two labelled
resources per value. The model trains on these observed examples; review their quality first.
The [CSV example](../fixtures/sanitized_inventory_sample.csv) has five resources and one missing-label row.

## Your own vocabulary

Without JSON, the synthetic demo's vocabulary and aliases are used. For another estate, upload a
small JSON file. It must contain exactly `team` and `env`; canonical values are unique lowercase
names using letters, digits, and hyphens, starting with a letter.

```json
{
  "team": {
    "values": ["platform-engineering", "customer-success"],
    "aliases": {"platform": "platform-engineering", "cs": "customer-success"}
  },
  "env": {
    "values": ["development", "production"],
    "aliases": {"dev": "development", "prod": "production"},
    "name_tokens": {"development": ["dev"], "production": ["prod"]}
  }
}
```

Aliases are optional and map normalized lowercase input names to canonical values. Environment
name tokens are optional, map a canonical environment to name tokens, and must be unambiguous.
If omitted, the imported model uses learned signals without the demo's environment-name rules.
Unrecognized CSV tag values are treated as needing review. Override choices use your vocabulary.
See the [downloadable example](../fixtures/label_config_example.json), which matches the CSV template.

## Save and resume

Saving stores the original CSV, its vocabulary, and its decisions in a local SQLite review database.
The same snapshot and vocabulary reopen the same review; changed data or vocabulary create a separate
review so approvals cannot transfer accidentally. Closing the browser or restarting the app does
not delete saved reviews. Use **Open saved review** on the import page to continue.

Files live in `review-workspaces/`, or `OBS_AGENT_REVIEW_DIR`; Docker uses `/app/state/reviews` on
its persistent volume. They contain private inventory data: restrict access, back up the directory,
and keep it outside git. All users of one local app installation can reopen its saved reviews;
this is a single trusted-team workspace, not tenant-isolated storage. Delete a review's database
through your normal local file management when its retention period ends.

The API and agent still use synthetic data. Saving/importing does not give either surface access to
private inventories or cloud accounts. See [Decision storage](decision-storage.md) for approval
provenance and [the example plan](../fixtures/reviewed_change_plan_example.json) for the export.
