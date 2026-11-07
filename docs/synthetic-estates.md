# Generate a fictional cloud estate

Build a repeatable test pack for inventory import, label review and snapshot comparison.
This reuses `inventory.make_estate()` without changing the bundled inventory, published synthetic
benchmarks or model configuration. It makes no cloud, hosted-model or telemetry requests.

From the obs-agent checkout, with the locked environment installed:

```bash
uv sync --locked
uv run python -m scripts.synthetic_estate \
  --out output/synthetic-estates/example --rows 320 --seed 7 --profile harsh
```

If you already have a generated ZIP, extract it and use the five files inside its estate folder.
You do not need to regenerate the data. Upload each inventory CSV and its vocabulary separately;
the ZIP, hidden truth and manifest are not review-app inputs.

Use a **new output directory** each time. Existing directories, files and symlinks are refused;
they are never overwritten. Inputs are validated before the directory is created. If a filesystem
failure interrupts writing, the new directory can be incomplete: the manifest is written last.
Inspect/remove only that incomplete output, then rerun into a new directory.

Rows must be 100–10,000 and the seed non-negative. Profiles `clean`, `harsh` and `reorg` reuse the
existing generator's distributions; controlled cases below are added to each profile. The default
is `harsh`. The same parameters, source files and locked library versions reproduce the same bytes.
A new seed varies the simulated estate; it does not establish independent real-world validation.

## Files and boundaries

| File | Purpose | Give to the review app? |
| --- | --- | --- |
| `inventory-before.csv` | First fictional Azure subscription/GCP project snapshot | Yes |
| `inventory-after.csv` | Same resource IDs after controlled ownership changes | Yes |
| `vocabulary.json` | Valid team/environment values, aliases and environment name tokens | Yes, with each CSV |
| `truth.csv` | Hidden labels, scenario and expected team action, keyed by snapshot/resource ID | No: evaluator only |
| `scenario_manifest.json` | Seed, profile, baseline date, library/source hashes, entity relationships, scenario counts, changed IDs and file checksums | No: evaluator only |

Public CSVs contain only the supported import columns, never `true_*`, scenario or expected-action
columns. Missing truth for a shared resource means **no single owner**, not an extra classification
label. The manifest's fictional team directory relates owners, Azure/GCP parents and cost centres;
those are scenario-author definitions, not discovered facts about a customer.

The baseline uses the existing fictional teams and owner names. Spend fields are summary signals
for the generator's fixed historical window, not a real billing feed. The second snapshot changes
ownership evidence/truth, not the spend period, environment or resource population. The manifest
records the fixed baseline date; creation dates do not follow today's date.

## Scenarios

Each controlled group has `floor(rows / 10)` rows. The rest retain the chosen base profile.
For 320 rows, there are 32 rows per controlled case and 192 background rows.

| Case | Input change | Evaluator's expectation |
| --- | --- | --- |
| Background | Existing profile's clean/noisy naming, ownership, hierarchy and observed labels | Inspect whatever evidence is available; correct inference is not guaranteed |
| Missing labels | Remove observed team/environment/cost-centre tags while retaining other evidence | Suggest only with supporting evidence; retain human review |
| Shared ownership | Two different teams' owner contacts, shared hierarchy, generic services, blank team tag | Abstain on a single team; investigate shared ownership |
| Unknown ownership | Opaque ID, no owners/creator, shared hierarchy, generic services, blank team tag | Abstain even though the simulator retains hidden truth |
| Ownership change | In the second snapshot change hidden team/cost centre and owner evidence; retain old tags/name/hierarchy | Detect changed evidence and investigate stale tags; never transfer an approval |

The unknown/shared IDs encode neither team nor scenario. They deliberately remove name evidence.
The old hierarchy in an ownership-change case can be misleading by design; inconsistent signals
are part of the exercise, not broken referential IDs.

The importer requires at least two recognised values per label, with at least two observations
for every observed value. A singleton observation is blanked rather than inventing another labelled
example. The manifest records the affected IDs. Both CSVs pass the production importer before
export; an estate with insufficient remaining support is rejected before writing. Withholding
observations is fixture compatibility, not advice to delete a customer's rare labels.

## Try the review cycle

To keep this exercise separate from other reviews and a Docker UI on port 8501:

```bash
OBS_AGENT_DEMO=0 \
OBS_AGENT_REVIEW_DIR=output/synthetic-estates/reviews \
OBS_AGENT_DECISIONS=output/synthetic-estates/demo-decisions.db \
uv run streamlit run app.py --server.port 8502
```

1. Open port 8502, choose **Import / reopen inventory**, and upload `inventory-before.csv` plus
   `vocabulary.json`. Save it with a clearly fictional name.
2. Inspect the queue. If trying accept/reject/override, enter your own test reviewer identity and
   choose the decisions yourself. Generating a pack never creates approvals.
3. Import `inventory-after.csv` with the same vocabulary, then open **Compare snapshots**.
4. Use the evaluator manifest to check that every listed ownership-change resource is reported
   as changed. Additional rows can have changed suggestions because the fitted model is different.
5. Only after reviewing the app output, compare it with `truth.csv`. Keep truth and scenario
   metadata out of model features, tool evidence and any answer prompt.

The generator starts no services and changes no existing reviews, cloud tags or RAG corpus.
Stop this optional Streamlit process with Ctrl-C when finished. Generated data and this separate
review directory under `output/synthetic-estates/` are ignored by git.

## Evaluator reference: the seed-7 sample

Read this section **after recording the app's suggestions**. These are construction facts and
expected actions from the generated pack, not observed prediction-quality results. They apply to
`--rows 320 --seed 7 --profile harsh`; use your own manifest/truth for other parameters.
The captured sample uses NumPy 2.5.4, pandas 3.1.0 and the fixed baseline date 2026-06-01.

| Group | Rows in each snapshot | Example resource ID | Expected team action |
| --- | --- | --- | --- |
| Background | 192 | Use the manifest/truth to identify rows | Review available evidence |
| Missing labels | 32 | `legacy-12` | Review available evidence |
| Shared ownership | 32 | `estate-7-gcp-000033` | Abstain on a single team |
| Unknown ownership | 32 | `estate-7-azure-000005` | Abstain despite hidden truth being available to the evaluator |
| Ownership change | 32 | `app-vault-gwc` | Review before; investigate changed ownership after |

Both snapshots have the same 320 resource IDs. There are 640 truth rows, keyed by snapshot and
resource ID. In this sample, no singleton observations needed withholding. The before/after
comparison must find all 32 IDs in `changed_ownership_resources`; it can also report changed
suggestions for other rows because inference is refitted. It should report no added/removed IDs.

The manifest records SHA-256 and byte size for the two CSVs, vocabulary and truth. To inspect file
hashes on macOS, run from the extracted estate folder:

```bash
shasum -a 256 inventory-before.csv inventory-after.csv vocabulary.json truth.csv
```

Compare each result with the corresponding `files` entry in `scenario_manifest.json`. The
manifest also records generator-source hashes and library versions. This detects changed files;
it does not establish that data matches a real organisation or that its expected actions are right.

### Record a review before interpreting the score

Keep a small worksheet outside the inference inputs. For each inspected resource, record snapshot,
resource ID, displayed team suggestion (or abstention), confidence, evidence/reason, and your own
explanation. Only then join it to `truth.csv` by **both** snapshot and resource ID.

| Question | Evidence to retain |
| --- | --- |
| Did the model make a supported suggestion? | Displayed suggestion, confidence and supporting owner/name evidence |
| Was it confidently wrong? | Wrong non-abstaining suggestion against known single-team truth, with the chosen confidence threshold recorded in advance |
| Did it handle absent/conflicting ownership safely? | Abstention or unsupported suggestion on the unknown/shared cases; do not replace blank shared truth with a team |
| Did comparison find the controlled changes? | The 32 manifest IDs and their changed owner fields; record extra suggestion changes separately |
| Did the workflow help the reviewer? | Your time, confusing steps and missing evidence; one fictional exercise is not a real-user study |

Report inspected counts and per-scenario denominators alongside every rate. Coverage is the fraction
receiving a suggestion; correctness among suggestions and abstention are separate quantities.
Do not claim an accuracy percentage across shared-owner rows with no single true team, or count a
safe abstention as an ordinary wrong classification. A model can also guess the hidden team
correctly while lacking evidence: record that as an unsupported guess, not a supported success.

Feynman check: explain why “the simulator knows the owner” does not mean “the model should name
the owner.” Answer: only the evaluator sees hidden truth; the model must work from supplied evidence
and abstain where that evidence is missing. Generating more rows cannot resolve that distinction.

## What these tests establish

Run the fixture checks without Docker, cloud keys or a collector:

```bash
uv run pytest tests/test_synthetic_estate.py -q
```

They verify importability, truth separation, repeatability, unchanged demo/RNG state, hidden-owner
and shared-owner semantics, stable IDs, cost-centre truth relationships, new-directory safety and
real saved-review/comparison behavior. Scenario expectations describe what the system should do;
the generator tests do **not** certify that its suggestions meet every expectation.

When assessing model behavior, report confident mistakes, abstentions, supported suggestion
coverage and per-scenario counts. Do not count abstaining on an unknown/shared owner as an ordinary
wrong classification or score blank shared truth as a team. Define acceptance criteria before
running a model comparison and retain failures.

These fixtures share generator logic, vocabulary and naming assumptions with existing model
experiments. Extra seeds or larger row counts do not remove that circularity. Independently authored
cases, different naming conventions and eventually an approved real pilot are still needed to
measure robustness or customer usefulness. See [real evaluation](real-evaluation.md).
