"""Exercise the persistence assertions with real CLI decisions and disposable storage."""

import json
import subprocess

import pytest

import review_cli
from scripts import review_persistence_check as check


@pytest.fixture
def cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    monkeypatch.delenv("OBS_AGENT_DEMO", raising=False)
    monkeypatch.chdir(check.ROOT)

    def call(*args):
        assert review_cli.main(list(args)) == 0
        return json.loads(capsys.readouterr().out)

    return call


def test_fresh_cli_reads_preserve_decisions_and_export(cli):
    review_id = check.seed(cli)
    before = check.snapshot(cli, review_id)
    assert check.snapshot(cli, review_id) == before
    assert {d["decision"] for d in before["decisions"]} == {"accept", "reject", "override"}


def test_missing_decision_fails_instead_of_reseeding(cli):
    review_id = check.seed(cli)
    cli("undo", review_id, "az-demo-001", "team", "--reviewer", check.REVIEWER)
    with pytest.raises(AssertionError, match="saved decisions missing"):
        check.snapshot(cli, review_id)


def test_changed_override_fails(cli):
    review_id = check.seed(cli)
    cli(
        "decide",
        review_id,
        "az-demo-002",
        "team",
        "override",
        "--value",
        "payments",
        "--reviewer",
        check.REVIEWER,
    )
    with pytest.raises(AssertionError):
        check.snapshot(cli, review_id)


def test_changed_export_approval_fails(cli):
    review_id = check.seed(cli)

    def damaged_export(*args):
        result = cli(*args)
        if args[0] == "export":
            result["approvals"][0]["reviewer"] = "wrong-reviewer"
        return result

    with pytest.raises(AssertionError, match="approval reviewer"):
        check.snapshot(damaged_export, review_id)


@pytest.mark.parametrize("fail", [False, True])
def test_compose_isolation_and_cleanup_even_on_failure(monkeypatch, fail):
    calls = []
    containers = iter(["before", "after"])

    def docker(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] > 0
        if args[:2] == ["docker", "compose"]:
            project = args[args.index("-p") + 1]
            assert project.startswith("obs-review-test-")
            override = check.Path(args[args.index("-f") + 3])
            assert "ports: !reset []" in override.read_text()
        output = next(containers) if "ps" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    def seed(_):
        if fail:
            raise AssertionError("simulated write failure")
        return "review-id"

    monkeypatch.setattr(check.subprocess, "run", docker)
    monkeypatch.setattr(check, "seed", seed)
    monkeypatch.setattr(check, "snapshot", lambda *_: {"saved": True})
    if fail:
        with pytest.raises(AssertionError, match="simulated write failure"):
            check.run()
    else:
        check.run()
        restart_down = [args for args in calls if args[-1] == "down"]
        assert len(restart_down) == 1
    assert calls[-2][-3:] == ["down", "-v", "--remove-orphans"]
    assert calls[-1][:3] == ["docker", "image", "rm"]
