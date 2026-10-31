"""Headless label triage for an Azure/GCP inventory CSV (read-only).

Run from an obs-agent checkout:

    uv run python skills/cloud-label-review/scripts/triage.py \
        --csv inventory.csv [--vocab vocabulary.json] [--out triage.json]

Prints coverage and the review queue (most confident first) as JSON. It never
records decisions, writes tags, or contacts a cloud provider.
"""

import argparse
import json
import sys
from pathlib import Path

# Prefer the checkout we're run from, so the skill also works when installed
# outside the repo (e.g. ~/.claude/skills/); fall back to the bundled location.
REPO = Path.cwd() if (Path.cwd() / "labeling.py").exists() else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from inventory_import import InventoryImportError, load_inventory_csv  # noqa: E402
from label_config import LabelConfigError, load_label_config  # noqa: E402
from label_review import coverage, review_queue  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, type=Path, help="inventory CSV in the obs-agent schema")
    ap.add_argument("--vocab", type=Path, help="optional team/env vocabulary JSON")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = ap.parse_args()

    try:
        config = load_label_config(args.vocab) if args.vocab else None
        inventory = load_inventory_csv(args.csv, config)
        result = {
            "coverage": coverage(inventory),
            "queue": review_queue(inventory=inventory, min_confidence=args.min_confidence),
        }
    except (InventoryImportError, LabelConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
