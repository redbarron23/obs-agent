"""Fail when repository-local Markdown links point to a missing file."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def local_link_errors(root: Path) -> list[str]:
    """Return broken relative Markdown links in README.md and docs/ files."""
    sources = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    errors = []
    for source in sources:
        for target in LINK.findall(source.read_text()):
            target = target.strip().strip("<>").split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            path = unquote(parsed.path)
            if not path:
                continue
            destination = root / path.lstrip("/") if path.startswith("/") else source.parent / path
            try:
                destination.resolve().relative_to(root.resolve())
            except ValueError:
                errors.append(f"{source.relative_to(root)}: link escapes repository: {target}")
                continue
            if not destination.exists():
                errors.append(f"{source.relative_to(root)}: missing link target: {target}")
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = local_link_errors(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Markdown links: OK")


if __name__ == "__main__":
    main()
