"""Keep changed evidence and failed worker processes distinguishable from successful load checks."""

from types import SimpleNamespace

import pytest

from scripts.telemetry_load_check import check, comparable


def test_only_freshness_check_time_is_ignored():
    first = {
        "status": "ok",
        "hits": [{"text": "fictional"}],
        "freshness": {"status": "current", "checked_at": "before"},
    }
    second = {**first, "freshness": {**first["freshness"], "checked_at": "after"}}
    assert comparable(first) == comparable(second)
    assert first["freshness"]["checked_at"] == "before"
    assert comparable(first) != comparable({**second, "hits": []})
    assert comparable(first) != comparable({**second, "freshness": {"status": "stale"}})


def test_failed_worker_is_not_reported_as_passing(monkeypatch):
    monkeypatch.setattr(
        "scripts.telemetry_load_check.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="availability budget exceeded"),
    )
    with pytest.raises(
        AssertionError,
        match="disabled process failed.*",
    ):
        check()
