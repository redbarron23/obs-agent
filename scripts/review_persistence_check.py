"""Check saved reviews across isolated Compose container recreation (no RAG or LLM)."""

import json
import subprocess
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "compose-persistence-test"
EXPECTED = {
    "gcp-demo-003": ("accept", "data-platform"),
    "az-demo-001": ("reject", None),
    "az-demo-002": ("override", "data-platform"),
}


def seed(cli):
    review = cli("import", "fixtures/review_persistence_inventory.csv")
    review_id = review["id"]
    assert cli("decisions", review_id) == [], "test state must start empty"
    for resource, (decision, value) in EXPECTED.items():
        args = ["decide", review_id, resource, "team", decision, "--reviewer", REVIEWER]
        if decision == "override":
            args += ["--value", value]
        cli(*args)
    return review_id


def snapshot(cli, review_id):
    summary = cli("show", review_id)
    decisions = cli("decisions", review_id, "--current")
    assert len(decisions) == 3, "saved decisions missing or duplicated"
    assert {d["resource_id"] for d in decisions} == set(EXPECTED), "wrong resources"
    for decision in decisions:
        assert (decision["decision"], decision["value"]) == EXPECTED[decision["resource_id"]]
        assert decision["label"] == "team" and decision["reviewer"] == REVIEWER
        for key in ("inventory_id", "model_version"):
            assert decision[key] == summary[key], f"decision {key} differs from inventory"
    plan = cli("export", review_id, "--format", "plan")
    assert plan["reviewed_only"] is True
    approved = {r for r, (d, _) in EXPECTED.items() if d != "reject"}
    assert {c["resource_id"] for c in plan["changes"]} == approved, "wrong exported changes"
    for change in plan["changes"]:
        decision, value = EXPECTED[change["resource_id"]]
        assert change["tags"]["team"]["new"] == value
        assert change["tags"]["team"]["how"] == decision
    assert len(plan["approvals"]) == 2, "approval evidence missing or duplicated"
    evidence = {d["resource_id"]: d for d in decisions}
    assert {a["resource_id"] for a in plan["approvals"]} == approved
    for approval in plan["approvals"]:
        for key, value in evidence[approval["resource_id"]].items():
            assert approval[key] == value, f"approval {key} differs from saved decision"
    for key in ("inventory_id", "model_version"):
        assert plan[key] == summary[key], f"plan {key} differs from inventory"
    del plan["generated_at"]
    return {
        "summary": summary,
        "decisions": sorted(decisions, key=lambda d: d["resource_id"]),
        "plan": plan,
    }


def run():
    project = "obs-review-test-" + uuid.uuid4().hex[:12]
    image = project + ":test"
    with tempfile.TemporaryDirectory(prefix="obs-review-test-") as directory:
        override = Path(directory) / "compose.yml"
        override.write_text(f"services:\n  api:\n    ports: !reset []\n    image: {image}\n")
        base = [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "-p",
            project,
            "-f",
            str(ROOT / "compose.rag.yml"),
            "-f",
            str(override),
        ]

        def compose(*args, timeout=120, check=True):
            return subprocess.run(
                base + list(args),
                check=check,
                timeout=timeout,
                text=True,
                capture_output=True,
            )

        def cli(*args):
            return json.loads(compose("exec", "-T", "api", "python", "review_cli.py", *args).stdout)

        print(f"Isolated persistence check: {project}", flush=True)
        try:
            compose("up", "-d", "--build", "--wait", "--wait-timeout", "120", "api", timeout=900)
            review_id = seed(cli)
            before = snapshot(cli, review_id)
            old_container = compose("ps", "-q", "api").stdout.strip()
            assert old_container, "API container not found"
            compose("down")
            compose("up", "-d", "--wait", "--wait-timeout", "120", "api", timeout=180)
            new_container = compose("ps", "-q", "api").stdout.strip()
            assert new_container and old_container != new_container, "API was not recreated"
            assert snapshot(cli, review_id) == before, "saved review or plan changed after restart"
            print(
                "PASS: accept/reject/override decisions and reviewed plan survived Compose down/up."
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(exc.stdout)
            print(exc.stderr)
            raise
        finally:
            try:
                logs = compose("logs", "--no-color", "api", check=False, timeout=30)
                print(logs.stdout)
                print(logs.stderr)
            finally:
                try:
                    compose("down", "-v", "--remove-orphans")
                finally:
                    subprocess.run(["docker", "image", "rm", image], check=False, timeout=60)


if __name__ == "__main__":
    run()
