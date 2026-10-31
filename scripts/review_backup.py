"""Back up, verify and restore the main decision database and every saved review.

    uv run python scripts/review_backup.py backup DEST
    uv run python scripts/review_backup.py verify DEST
    uv run python scripts/review_backup.py restore DEST   # into empty OBS_AGENT_* paths
    uv run python scripts/review_backup.py plans          # approved plans, for before/after diffs

Source and restore paths come from OBS_AGENT_DECISIONS and OBS_AGENT_REVIEW_DIR (the app's
defaults otherwise), or --decisions / --review-dir. Backups use SQLite's online backup API, so
they are consistent while the API or UI is running. backup, verify and restore use only the
standard library, so they also run inside the Docker image: `python - backup ... < this file`.
See docs/backup-restore.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = "manifest.json"
DECISIONS_FILE = "label_decisions.db"
REVIEWS_DIR = "reviews"
REVIEW_NAME = re.compile(r"[0-9a-f]{64}\.db")


def default_decisions() -> Path:
    path = Path(os.environ.get("OBS_AGENT_DECISIONS", "label_decisions.db"))
    return path.with_suffix(".db") if path.suffix == ".json" else path  # as DecisionStore


def default_review_dir() -> Path:
    return Path(os.environ.get("OBS_AGENT_REVIEW_DIR", "review-workspaces"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(path: Path) -> dict:
    """Integrity result and row counts, read without creating or changing the file."""
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {
            t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for t in ("decisions", "review_feedback", "workspace")
            if t in tables
        }
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    return {"integrity": integrity, "rows": counts}


def _private_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))


def _online_copy(source: Path, target: Path) -> None:
    """Consistent snapshot of a possibly live database (committed transactions only)."""
    _private_file(target)
    src = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    target.chmod(0o600)


def _sources(decisions: Path, review_dir: Path) -> list[tuple[str, Path]]:
    found = [(DECISIONS_FILE, decisions)] if decisions.is_file() else []
    if review_dir.is_dir():
        for path in sorted(review_dir.iterdir()):
            if REVIEW_NAME.fullmatch(path.name):
                found.append((f"{REVIEWS_DIR}/{path.name}", path))
    return found


def backup(dest: Path, decisions: Path, review_dir: Path) -> dict:
    if dest.exists():
        raise SystemExit(f"{dest} already exists; choose a new backup directory.")
    dest.mkdir(parents=True, mode=0o700)
    dest.chmod(0o700)
    files = {}
    for name, source in _sources(decisions, review_dir):
        target = dest / name
        _online_copy(source, target)
        files[name] = _summary(target) | {"sha256": _sha256(target), "source": str(source)}
    manifest = {
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    _private_file(dest / MANIFEST)
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    errors = verify(dest)
    if errors:
        raise SystemExit("Backup failed verification:\n" + "\n".join(errors))
    return manifest


def verify(dest: Path) -> list[str]:
    """Problems with a backup or restored copy; an empty list means it checks out."""
    manifest = json.loads((dest / MANIFEST).read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = dest / name
        if not path.is_file():
            errors.append(f"{name}: missing")
            continue
        if _sha256(path) != expected["sha256"]:
            errors.append(f"{name}: checksum mismatch")
        actual = _summary(path)
        if actual["integrity"] != "ok":
            errors.append(f"{name}: integrity_check returned {actual['integrity']}")
        if actual["rows"] != expected["rows"]:
            errors.append(f"{name}: row counts {actual['rows']} != {expected['rows']}")
        if name.startswith(f"{REVIEWS_DIR}/") and actual["rows"].get("workspace") != 1:
            errors.append(f"{name}: no saved inventory and vocabulary (workspace table)")
        if path.stat().st_mode & 0o077:
            errors.append(f"{name}: readable by group or others; chmod 600")
    return errors


def restore(dest: Path, decisions: Path, review_dir: Path) -> list[Path]:
    """Copy a verified backup to empty target paths; never overwrites existing state."""
    errors = verify(dest)
    if errors:
        raise SystemExit("Backup failed verification; not restoring:\n" + "\n".join(errors))
    names = json.loads((dest / MANIFEST).read_text())["files"]
    targets = {
        name: decisions if name == DECISIONS_FILE else review_dir / Path(name).name
        for name in names
    }
    clashes = [str(t) for t in targets.values() if t.exists()]
    if clashes:
        raise SystemExit("Refusing to overwrite existing state:\n" + "\n".join(clashes))
    if any(name != DECISIONS_FILE for name in names):
        review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        review_dir.chmod(0o700)
    for name, target in targets.items():
        _private_file(target)
        shutil.copyfile(dest / name, target)
        if _sha256(target) != names[name]["sha256"]:
            raise SystemExit(f"{target}: restored copy does not match the backup")
    return list(targets.values())


def plans(decisions: Path, review_dir: Path) -> dict:
    """Approved change plans without generation timestamps, keyed by database.

    Needs the project modules (run from the checkout, or with /app as the working directory in
    the image). Two runs against the same
    state, code and dependencies print identical JSON.
    """
    script = Path(globals().get("__file__", ""))
    # The checkout this script lives in, then the working directory (/app in the image).
    sys.path[:0] = [str(script.resolve().parents[1]) if script.is_file() else "", str(Path.cwd())]
    os.environ["OBS_AGENT_REVIEW_DIR"] = str(review_dir)
    import label_review as lr
    import review_workspaces

    def strip(plan: dict) -> dict:
        return {k: v for k, v in plan.items() if k != "generated_at"}

    out = {}
    if decisions.is_file():
        out[DECISIONS_FILE] = strip(lr.approved_change_plan(lr.DecisionStore(decisions)))
    for review in review_workspaces.saved_reviews():
        inventory, store = review_workspaces.load_review(review["id"])
        out[f"{REVIEWS_DIR}/{review['id']}.db"] = strip(lr.approved_change_plan(store, inventory))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("backup", "verify", "restore", "plans"))
    parser.add_argument("backup_dir", nargs="?", type=Path)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    decisions = args.decisions or default_decisions()
    review_dir = args.review_dir or default_review_dir()
    if args.command != "plans" and args.backup_dir is None:
        parser.error(f"{args.command} needs a backup directory")
    if args.command == "backup":
        manifest = backup(args.backup_dir, decisions, review_dir)
        print(f"Backed up {len(manifest['files'])} database(s) to {args.backup_dir}")
    elif args.command == "verify":
        errors = verify(args.backup_dir)
        print("\n".join(errors) or f"{args.backup_dir}: OK")
        raise SystemExit(1 if errors else 0)
    elif args.command == "restore":
        restored = restore(args.backup_dir, decisions, review_dir)
        print(f"Restored {len(restored)} database(s):\n" + "\n".join(map(str, restored)))
    else:
        print(json.dumps(plans(decisions, review_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
