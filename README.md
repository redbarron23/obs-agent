# obs-agent — Tag Inference & Review for Multi-Cloud Cost Estates

[![CI](https://github.com/jhourihane/obs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jhourihane/obs-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jhourihane/obs-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/jhourihane/obs-agent)

**obs-agent finds untagged cloud resources, infers likely ownership, and produces auditable, human-reviewed change plans.** It supports synthetic AWS, Azure, and GCP billing inventories so the entire workflow is safe to explore locally.

| Synthetic hold-out result | Value |
| --- | ---: |
| Team-label accuracy | 98% |
| Team-label macro-F1 | 0.98 |
| Environment-label accuracy | 100% |

These figures are from the bundled synthetic inventory—not a promise of production accuracy. See [evaluation](docs/evaluation.md) for methodology and limits.

## Quick start

```bash
git clone https://github.com/jhourihane/obs-agent.git
cd obs-agent
uv sync
uv run streamlit run app.py
```

Open the dashboard, inspect an untagged resource, adjust its proposed labels if needed, record your decision, and export a reviewed-only plan.

## What it does

- Generates a deterministic multi-cloud cost inventory, including missing labels and a deliberate reorganisation scenario.
- Infers `team` and `environment` labels from billing, resource, ownership, and parent-child signals.
- Shows evidence, confidence, calibration, coverage, and drift signals before anyone acts.
- Captures accept/reject/override feedback in a local SQLite audit trail.
- Exports only reviewed accepted or overridden label changes—never applies provider changes directly.

## Documentation

| Topic | Guide |
| --- | --- |
| Try the end-to-end synthetic workflow | [Five-minute demo](docs/five-minute-demo.md) |
| How label inference and review work | [Label inference](docs/label-inference.md) |
| Ask questions over the cost estate | [Agent guide](docs/agent.md) |
| REST endpoints and exports | [API reference](docs/api.md) |
| Quality, calibration, and evaluation | [Evaluation](docs/evaluation.md) |
| Components and data flow | [Architecture](docs/architecture.md) |
| Tests and local checks | [Testing](docs/testing.md) |
| Reviewer workflow | [Reviewing labels](docs/reviewing.md) |
| Reading and source material | [References](docs/references.md) |

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

## Test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q -m "not integration"
uv run evals.py --dry-run
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Related project

The reusable tag-inference library is maintained at [obs-tools/tag-inference](https://github.com/obs-tools/tag-inference).
