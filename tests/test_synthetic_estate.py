"""Validate the exported simulation through real import and review boundaries."""

import hashlib
import json
from io import BytesIO

import pandas as pd
import pytest

import inventory
from inventory_import import COLUMN_ORDER, load_inventory_csv
from scripts.synthetic_estate import build_pack, main


@pytest.fixture(scope="module")
def pack():
    return build_pack(seed=11)


def frames(pack):
    config = json.loads(pack["vocabulary.json"])
    return [
        load_inventory_csv(pack[f"inventory-{name}.csv"], config) for name in ("before", "after")
    ]


def test_pack_is_reproducible_and_leaves_demo_unchanged(pack):
    baseline = inventory.INVENTORY_DF.copy(deep=True)
    rng_state = json.dumps(inventory.RNG.bit_generator.state, sort_keys=True)
    assert build_pack(seed=11) == pack
    assert build_pack(seed=12)["inventory-before.csv"] != pack["inventory-before.csv"]
    pd.testing.assert_frame_equal(inventory.INVENTORY_DF, baseline)
    assert json.dumps(inventory.RNG.bit_generator.state, sort_keys=True) == rng_state


def test_inputs_are_importable_and_truth_is_separate(pack):
    before, after = frames(pack)
    assert len(before) == len(after) == 320
    assert list(before.resource_id) == list(after.resource_id)
    assert before.resource_id.is_unique
    for name in ("before", "after"):
        raw = pd.read_csv(BytesIO(pack[f"inventory-{name}.csv"]))
        assert set(raw.columns) == set(COLUMN_ORDER)
        assert not any(c.startswith("true_") or "scenario" in c for c in raw.columns)
    truth = pd.read_csv(BytesIO(pack["truth.csv"]))
    assert len(truth) == 640 and not truth.duplicated(["snapshot", "resource_id"]).any()
    assert set(truth.resource_id) == set(before.resource_id)
    known = truth[truth.true_team.notna()]
    assert all(
        inventory.TEAMS[row.true_team]["cost_center"] == row.true_cost_center
        for row in known.itertuples()
    )
    manifest = json.loads(pack["scenario_manifest.json"])
    assert sum(s["rows"] for s in manifest["scenarios"].values()) == 320
    for name, info in manifest["files"].items():
        assert info["sha256"] == hashlib.sha256(pack[name]).hexdigest()
        assert info["bytes"] == len(pack[name])


def test_missing_and_ambiguous_cases_have_honest_expectations(pack):
    before, _ = frames(pack)
    before = before.set_index("resource_id")
    truth = pd.read_csv(BytesIO(pack["truth.csv"]))
    truth = truth[truth.snapshot == "before"].set_index("resource_id")
    for key, row in truth.iterrows():
        observed = before.loc[key]
        if row.scenario == "missing_labels":
            assert pd.isna(observed.team) and pd.isna(observed.env)
        elif row.scenario == "unknown_ownership":
            assert observed.owners == [] and pd.isna(observed.created_by)
            assert pd.isna(observed.team) and pd.notna(row.true_team)
            assert row.expected_team_action == "abstain"
        elif row.scenario == "shared_ownership":
            assert len(observed.owners) == 2 and pd.isna(row.true_team)
            assert pd.isna(row.true_cost_center) and row.expected_team_action == "abstain"


def test_later_snapshot_changes_ownership_without_rekeying_resources(pack):
    before, after = [f.set_index("resource_id") for f in frames(pack)]
    truth = pd.read_csv(BytesIO(pack["truth.csv"]))
    old = truth[truth.snapshot == "before"].set_index("resource_id")
    new = truth[truth.snapshot == "after"].set_index("resource_id")
    changed = set(json.loads(pack["scenario_manifest.json"])["changed_ownership_resources"])
    assert len(changed) == 32
    for key in changed:
        assert old.loc[key, "true_team"] != new.loc[key, "true_team"]
        team = inventory.TEAMS[new.loc[key, "true_team"]]
        assert new.loc[key, "true_cost_center"] == team["cost_center"]
        assert after.loc[key, "owners"] == team["owners"][:2]
        assert new.loc[key, "expected_team_action"] == "investigate_change"
    pd.testing.assert_frame_equal(
        before.drop(columns=["owners", "created_by"]), after.drop(columns=["owners", "created_by"])
    )


def test_cli_writes_new_pack_and_refuses_overwrite(tmp_path, capsys):
    out = tmp_path / "estate"
    main(["--out", str(out), "--seed", "11"])
    original = {p.name: p.read_bytes() for p in out.iterdir()}
    assert "truth is separate" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(out)])
    assert exc.value.code == 2
    assert {p.name: p.read_bytes() for p in out.iterdir()} == original


@pytest.mark.parametrize("args", [{"rows": 0}, {"rows": 10_001}, {"seed": -1}, {"profile": "bad"}])
def test_invalid_generation_parameters_are_rejected(args):
    with pytest.raises(ValueError):
        build_pack(**args)


def test_cli_validation_creates_no_output(tmp_path):
    out = tmp_path / "estate"
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--rows", "1"])
    assert not out.exists()


@pytest.mark.parametrize("profile", ["clean", "harsh", "reorg"])
def test_profiles_work_with_small_estates(profile):
    result = build_pack(rows=100, seed=23, profile=profile)
    assert json.loads(result["scenario_manifest.json"])["rows_per_snapshot"] == 100


def test_saved_review_roundtrip_detects_controlled_ownership_changes(pack, tmp_path, monkeypatch):
    import review_workspaces as rw

    monkeypatch.setenv("OBS_AGENT_DEMO", "0")
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", str(tmp_path / "reviews"))
    summaries = [
        rw.import_review(pack[f"inventory-{name}.csv"], name, pack["vocabulary.json"])
        for name in ("before", "after")
    ]
    old, new = (summary["id"] for summary in summaries)
    assert old != new
    for key in (old, new):
        frame, store = rw.load_review(key)
        assert len(frame) == 320 and store.all() == []
        assert "true_team" not in frame and "scenario" not in frame
    difference = rw.compare_reviews(old, new)
    assert not difference["added"] and not difference["removed"]
    expected = set(json.loads(pack["scenario_manifest.json"])["changed_ownership_resources"])
    assert expected <= {r["resource_id"] for r in difference["changed"]}
    assert all(not row["earlier_decisions"] for row in difference["changed"])


def test_cli_refuses_dangling_symlink(tmp_path):
    out = tmp_path / "estate"
    destination = tmp_path / "missing"
    out.symlink_to(destination, target_is_directory=True)
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(out)])
    assert exc.value.code == 2 and out.is_symlink() and not destination.exists()
