"""Validated per-inventory vocabularies; synthetic defaults remain unchanged."""

import json
import re
from pathlib import Path

from inventory import ENV_NAME_TOKENS
from labeling import ALIASES, ALLOWED, LABELS


class LabelConfigError(ValueError):
    """A vocabulary or alias cannot be used safely."""


def default_config() -> dict:
    return {
        label: {"values": sorted(ALLOWED[label]), "aliases": dict(ALIASES[label])}
        | ({"name_tokens": ENV_NAME_TOKENS} if label == "env" else {})
        for label in LABELS
    }


def load_label_config(source: bytes | str | Path) -> dict:
    try:
        config = json.loads(source if isinstance(source, bytes) else Path(source).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelConfigError(f"Could not read vocabulary JSON: {exc}") from exc
    if not isinstance(config, dict) or set(config) != set(LABELS):
        raise LabelConfigError("Vocabulary must contain exactly 'team' and 'env'.")
    for label, settings in config.items():
        if not isinstance(settings, dict) or set(settings) - {"values", "aliases", "name_tokens"}:
            raise LabelConfigError(f"{label}: use values, aliases, and env name_tokens only.")
        values = settings.get("values")
        if (
            not isinstance(values, list)
            or len(values) < 2
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", v) for v in values
            )
            or len(set(values)) != len(values)
        ):
            raise LabelConfigError(f"{label}: provide at least two unique lowercase label values.")
        aliases = settings.setdefault("aliases", {})
        if not isinstance(aliases, dict) or any(
            not isinstance(k, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", k)
            or v not in values
            or (k in values and k != v)
            for k, v in aliases.items()
        ):
            raise LabelConfigError(f"{label}: aliases must map lowercase names to allowed values.")
        tokens = settings.setdefault("name_tokens", {}) if label == "env" else {}
        if label == "team" and "name_tokens" in settings:
            raise LabelConfigError("name_tokens is supported only for env.")
        if not isinstance(tokens, dict) or any(
            env not in values
            or not isinstance(names, list)
            or any(not isinstance(n, str) or not re.fullmatch(r"[a-z][a-z0-9]*", n) for n in names)
            for env, names in tokens.items()
        ):
            raise LabelConfigError("env: name_tokens must map allowed values to lowercase tokens.")
        names = [name for group in tokens.values() for name in group]
        if len(names) != len(set(names)):
            raise LabelConfigError("env: a name token cannot name more than one environment.")
    return config


def inventory_config(inventory=None) -> dict:
    return (
        default_config()
        if inventory is None
        else inventory.attrs.get("label_config", default_config())
    )
