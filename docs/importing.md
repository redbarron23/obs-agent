# Importing and reopening an inventory

The local **Import / reopen inventory** page accepts a sanitized CSV of Azure subscriptions and
GCP projects. It needs no cloud credentials and never writes tags. The public demo disables imports.

```bash
uv run streamlit run app.py
```

1. Open **Import / reopen inventory**. Download the CSV example/template and vocabulary example.
2. Upload your CSV and, optionally, a vocabulary JSON matching your own organisation. If the
   headers differ from the schema below, [map your columns](#map-your-own-column-names) on the page.
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
Each error names the spreadsheet rows to correct (the header is row 1).

For **both** team and env, provide at least two recognized values with at least two labelled
resources per value. The model trains on these observed examples; review their quality first.
The [CSV example](../fixtures/sanitized_inventory_sample.csv) has five resources and one missing-label row.

## Map your own column names

You do not need to rename columns before importing. When a CSV lacks any required column name, the
page shows **Map your columns**: one selector per field, pre-filled where a header matches the
field name or a common synonym (for example `Subscription ID`, `Environment`, `Cost Centre`).
Ambiguous headers are left unmapped. Check every selector before saving.

For example, this export:

```csv
Provider,Subscription ID,Management Group,Owner,Creator,Services,Daily Cost,Weekend Ratio,Cost CV,Team,Environment,Cost Centre,Notes
azure,az-001,mg-platform,alice|bob,alice,AKS|SQL,120,0.9,0.2,identity,prod,CC-4001,keep out
```

maps `Provider` → `cloud`, `Subscription ID` → `resource_id`, `Management Group` → `parent`,
`Owner` → `owners`, `Creator` → `created_by`, `Services` → `top_services`, `Daily Cost` →
`avg_daily_cost`, `Weekend Ratio` → `weekend_ratio`, `Cost CV` → `cost_cv`, `Team` → `team`,
`Environment` → `env` and `Cost Centre` → `cost_center`. `Notes` is listed under **Not imported**
and is dropped.

Mapping only renames and selects columns; values pass through unchanged and get the same validation
as a direct import. Every required field must come from a real column, and each source column can
fill one field. Blank owners, services, or cost-shape values are not filled in: a blank cost value
fails validation with its row number. Unrecognized tag values stay visible as needing review.
Saving stores the mapped CSV, so the review reopens without remapping. A CSV that already uses the
schema's column names skips this step.

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

Cost centers are inferred only when all non-missing observations for a team agree. Conflicting
values are shown in the review table; a team change still exports, without an inferred cost center.
All export formats use the same destination values: GCP values are lowercased and limited to 63
letters, numbers, underscores or hyphens; Azure values retain their case and are limited to 256
characters. Invalid outgoing values block the export with the affected resource and label so you
can correct the inventory and import it again. Raw before-values remain unchanged in the plan.

Saving stores the original CSV, its vocabulary, and its decisions in a local SQLite review database.
The same snapshot and vocabulary reopen the same review; changed data or vocabulary create a separate
review so approvals cannot transfer accidentally. Closing the browser or restarting the app does
not delete saved reviews. Use **Open saved review** on the import page to continue.

Files live in `review-workspaces/`, or `OBS_AGENT_REVIEW_DIR`; Docker uses `/app/state/reviews` on
its persistent volume. They contain private inventory data: restrict access, back up the directory,
and keep it outside git. All users of one local app installation can reopen its saved reviews;
this is a single trusted-team workspace, not tenant-isolated storage. Delete a review's database
through your normal local file management when its retention period ends.

## Compare with an earlier snapshot

For a recurring review, save each new export as its own snapshot, then open **Compare snapshots**.
It defaults to the newest saved review against the one before it and lists:

- **New** and **Removed** resources, with their parent, owners, and current tags.
- **Changed** resources, one row per change with before/after values and a reason for renewed
  review: parent, owners, or service footprint changes, material spend or cost-shape changes
  (the `drift.detect_drift` thresholds), tag edits made outside the review, and changed
  `team`/`env` suggestions. Resources that did not change are counted, not listed.

**Review new or changed resources** opens the current snapshot in **Label review** with a sidebar
filter limited to them. A changed resource whose tags still look correct has no suggestion to
decide, so it does not appear in the queue; the comparison table is its record.

Earlier decisions are shown next to changed and removed resources for context only. They stay in
the earlier snapshot's database and are never copied into the current review: approve again
against the current snapshot. If the two snapshots use different vocabularies, the page says so,
because tag and suggestion changes may then reflect the vocabulary rather than the estate.

The REST API (`/reviews`) and `review_cli.py` reach the same saved reviews by ID; see
[Saved reviews: API and CLI](api.md#saved-reviews-api-and-cli). The cost agent still uses synthetic
data. No surface gets access to cloud accounts. See [Decision storage](decision-storage.md) for approval
provenance and [the example plan](../fixtures/reviewed_change_plan_example.json) for the export.
