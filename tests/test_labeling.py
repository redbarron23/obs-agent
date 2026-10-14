"""
Tests for label inference (labeling.py) and its agent tools.

Unit tests use tiny hand-built inventories; the quality tests run the full
evaluation on the seeded synthetic inventory and assert floors, so a change
that quietly degrades predictions fails CI.
"""

import pandas as pd
import pytest

from labeling import (
    LabelInference,
    Rules,
    clean,
    evaluate,
    expected_calibration_error,
    name_tokens,
    normalise,
)


def _row(
    resource_id,
    parent,
    owners,
    team=None,
    env=None,
    true_team="payments",
    true_env="prod",
    services=("AKS",),
    cost=100.0,
    weekend=0.95,
):
    return {
        "resource_id": resource_id,
        "cloud": "azure",
        "parent": parent,
        "billing_account": "ba",
        "owners": list(owners),
        "created_by": owners[0],
        "created_at": None,
        "region": "westeurope",
        "top_services": list(services),
        "avg_daily_cost": cost,
        "weekend_ratio": weekend,
        "cost_cv": 0.1,
        "true_team": true_team,
        "true_env": true_env,
        "true_cost_center": "CC-1",
        "team": team,
        "env": env,
        "cost_center": "CC-1" if team else None,
    }


@pytest.fixture
def tiny_inventory():
    rows = (
        [
            _row(f"pay-api-prod-{i}", "mg-payments", ["alice"], team="payments", env="prod")
            for i in range(4)
        ]
        + [
            _row(
                f"web-app-dev-{i}",
                "mg-digital",
                ["kofi"],
                team="web",
                env="dev",
                true_team="web",
                true_env="dev",
                services=("AppService",),
                cost=20,
                weekend=0.2,
            )
            for i in range(4)
        ]
        + [
            _row("pay-ledger-prod-x", "mg-payments", ["alice"]),  # untagged
            _row(
                "web-cms-dev-x",
                "mg-digital",
                ["kofi"],
                true_team="web",
                true_env="dev",
                services=("AppService",),
                cost=20,
                weekend=0.2,
            ),  # untagged
        ]
    )
    return pd.DataFrame(rows)


class TestNormalise:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Payments", "payments"),
            ("team-payments", "payments"),
            ("data_platform", "data-platform"),
            ("Data Platform", "data-platform"),
            ("o11y", "observability"),
            ("finance", "finance-apps"),
            ("nonsense", None),
            (None, None),
            (float("nan"), None),
        ],
    )
    def test_team(self, raw, expected):
        assert normalise("team", raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Prod", "prod"),
            ("production", "prod"),
            ("uat", "staging"),
            ("DEV", "dev"),
            ("poc", "sandbox"),
            ("live", None),
        ],
    )
    def test_env(self, raw, expected):
        assert normalise("env", raw) == expected

    def test_clean_keeps_none_not_nan(self, tiny_inventory):
        df = clean(tiny_inventory)
        assert df["team_clean"].iloc[-1] is None


class TestNameTokens:
    def test_drops_opaque_ids(self):
        assert name_tokens("sub-a695feba") == []
        assert name_tokens("proj-013469") == []

    def test_keeps_words(self):
        assert name_tokens("pay-api-prod-weu") == ["pay", "api", "prod", "weu"]


class TestRules:
    def test_inherits_from_parent(self, tiny_inventory):
        df = clean(tiny_inventory)
        rules = Rules(df[df["team_clean"].notna()], "team")
        s = rules.suggest(df.iloc[-2])
        assert s.label == "payments"
        assert s.source.startswith("rule:")

    def test_env_from_name_token(self, tiny_inventory):
        df = clean(tiny_inventory)
        rules = Rules(df[df["env_clean"].notna()], "env")
        assert rules.suggest(df.iloc[-1]).label == "dev"

    def test_no_rule_for_unknown_parent(self, tiny_inventory):
        df = clean(tiny_inventory)
        rules = Rules(df[df["team_clean"].notna()], "team")
        row = df.iloc[-1].copy()
        row["parent"] = "mg-unknown"
        row["resource_id"] = "zzz-thing"
        assert rules.suggest(row).label is None


class TestModel:
    def test_predicts_untagged(self, tiny_inventory):
        inf = LabelInference(tiny_inventory)
        for _, row in inf.unlabelled("team").iterrows():
            s = inf.suggest(row, "team", "model")
            assert s.label == row["true_team"]
            assert 0 < s.confidence <= 1
            assert "strongest signals" in s.reason


class TestCalibration:
    def test_expected_calibration_error_is_zero_for_matching_confidence(self):
        predicted = pd.Series(["a", "b", "c", "d"])
        confidence = pd.Series([0.5, 0.5, 1.0, 1.0])
        truth = pd.Series(["a", "x", "c", "d"])

        assert expected_calibration_error(predicted, confidence, truth, bins=2) == 0.0


class TestQuality:
    """Floors on the seeded synthetic inventory — guards against regressions."""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return evaluate()

    def test_roughly_half_unlabelled(self, results):
        assert 140 <= results["team"]["n_unlabelled"] <= 200

    def test_team_model_beats_rules_coverage(self, results):
        assert results["team"]["model"]["coverage"] == 1.0
        assert results["team"]["rules"]["coverage"] < 0.6

    def test_team_accuracy_floor(self, results):
        assert results["team"]["model"]["accuracy"] >= 0.9
        assert results["team"]["model"]["macro_f1"] >= 0.85

    def test_env_hybrid_beats_model(self, results):
        assert results["env"]["hybrid"]["accuracy"] > results["env"]["model"]["accuracy"]

    def test_auto_apply_is_precise(self, results):
        for label in ("team", "env"):
            assert results[label]["model"]["auto_precision"] >= 0.95

    def test_model_confidence_is_calibrated_on_the_clean_estate(self, results):
        for label in ("team", "env"):
            assert results[label]["model"]["ece"] <= 0.2

    @pytest.mark.parametrize("label,floor", [("team", 0.9), ("env", 0.65)])
    def test_mislabel_flags_are_mostly_right(self, results, label, floor):
        # env is lower: without an env token in the name, spend shape alone
        # cannot reliably tell dev from staging.
        m = results[label]["mislabels"]
        assert m["flagged"] > 0
        assert m["flagged_and_wrong"] / m["flagged"] >= floor


class TestLabelTools:
    def test_coverage(self):
        from tools import get_label_coverage

        out = get_label_coverage(limit=3)
        assert "team: 170 unlabelled" in out
        assert "nan" not in out

    def test_suggest_untagged(self):
        from tools import suggest_labels

        out = suggest_labels("svc-ingest")
        assert "suggest team=data-platform" in out
        assert "cost_center=CC-2002" in out

    def test_suspected_wrong_uses_out_of_fold(self):
        from tools import suggest_labels

        out = suggest_labels("api-login-d-ew1")
        assert "SUSPECTED WRONG" in out
        assert "team=identity" in out

    def test_unknown_resource(self):
        from tools import suggest_labels

        assert "No Azure subscription or GCP project found" in suggest_labels("nope")

    def test_registered_with_agent(self):
        from tools import TOOL_DEFINITIONS, TOOL_DISPATCH

        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert {"get_label_coverage", "suggest_labels"} <= names <= set(TOOL_DISPATCH)


class TestEstateHardness:
    """The two synthetic estates: clean is published, harsh is the stress test.

    The clean estate is what the README's metrics describe, so it must not
    move. The harsh one exists to show how the pipeline degrades on data it
    was not tuned for, and to keep that story honest in CI.
    """

    def test_clean_estate_is_the_default(self):
        from inventory import INVENTORY_DF, make_estate

        pd.testing.assert_frame_equal(INVENTORY_DF, make_estate("clean"))

    def test_clean_estate_shape_is_stable(self):
        # Pinned because evals and tools assert against these exact values.
        from inventory import INVENTORY_DF

        assert len(INVENTORY_DF) == 320
        assert INVENTORY_DF["team"].isna().sum() == 170
        assert INVENTORY_DF["env"].isna().sum() == 168

    def test_unknown_hardness_is_rejected(self):
        from inventory import generate_inventory

        with pytest.raises(ValueError):
            generate_inventory(hardness="nope")

    def test_harsh_estate_is_messier(self):
        from inventory import INVENTORY_DF, make_estate

        harsh = make_estate("harsh")
        assert len(harsh) == len(INVENTORY_DF)
        # More untagged, and more resources with no recorded owner.
        assert harsh["team"].isna().sum() > INVENTORY_DF["team"].isna().sum()
        assert (harsh["owners"].map(len) == 0).sum() > 0
        assert (INVENTORY_DF["owners"].map(len) == 0).sum() == 0

    def test_harsh_estate_keeps_the_same_schema(self):
        from inventory import INVENTORY_DF, make_estate

        harsh = make_estate("harsh")
        # Same columns, same order as the published estate.
        assert list(harsh.columns) == list(INVENTORY_DF.columns)
        # Ground truth is still present, so the estate is still scoreable.
        assert harsh["true_team"].notna().all()
        assert harsh["true_env"].notna().all()

    def test_reorg_estate_shuffles_parents_but_keeps_schema(self):
        from inventory import INVENTORY_DF, TEAMS, make_estate

        def home_parent(row):
            parent_key = "azure_parent" if row["cloud"] == "azure" else "gcp_parent"
            return TEAMS[row["true_team"]][parent_key]

        reorg = make_estate("reorg")
        expected_parent = reorg.apply(home_parent, axis=1)
        clean_parent = INVENTORY_DF.apply(home_parent, axis=1)
        reorg_mismatch = (reorg["parent"] != expected_parent).mean()
        clean_mismatch = (INVENTORY_DF["parent"] != clean_parent).mean()
        assert reorg_mismatch > clean_mismatch
        assert list(reorg.columns) == list(INVENTORY_DF.columns)

    def test_estates_are_independent(self):
        # Building the harsh estate must not disturb the published one.
        from inventory import INVENTORY_DF, make_estate

        before = INVENTORY_DF.copy()
        make_estate("harsh")
        pd.testing.assert_frame_equal(INVENTORY_DF, before)


class TestDegradation:
    """The harsh estate must actually be harder, and the report must say so."""

    @pytest.fixture(scope="class")
    @classmethod
    def both(cls):
        from inventory import INVENTORY_DF, make_estate

        return evaluate(INVENTORY_DF), evaluate(make_estate("harsh"))

    def test_team_model_degrades_but_stays_useful(self, both):
        clean, harsh = both
        # Accuracy drops, but the model is still well above chance and covers
        # every resource — that is the point of using it over rules.
        assert harsh["team"]["model"]["accuracy"] < clean["team"]["model"]["accuracy"]
        assert harsh["team"]["model"]["accuracy"] >= 0.75
        assert harsh["team"]["model"]["coverage"] == 1.0

    def test_rules_collapse_on_the_harsh_estate(self, both):
        clean, harsh = both
        # Parent/prefix inheritance depends on the hierarchy being tidy, which
        # is exactly what the harsh estate breaks.
        assert harsh["team"]["rules"]["accuracy"] < clean["team"]["rules"]["accuracy"]
        assert harsh["team"]["rules"]["coverage"] < clean["team"]["rules"]["coverage"]

    def test_auto_apply_stays_precise_on_the_harsh_estate(self, both):
        # The safety property that matters in production: suggestions the
        # pipeline is confident about remain trustworthy, even on messy data.
        _, harsh = both
        assert harsh["team"]["model"]["auto_precision"] >= 0.95

    def test_degradation_report_renders(self, both, capsys=None):
        from labeling import format_degradation

        text = format_degradation(*both)
        assert "Degradation: clean → harsh estate" in text
        assert "team/model" in text


class TestSuspectOnHarsh:
    def test_mislabel_check_weakens_on_harsh_estate(self):
        # A real finding: with overlapping vocabulary the out-of-fold check
        # flags far fewer genuine mislabels, so it must not be trusted blindly.
        from inventory import INVENTORY_DF, make_estate
        from labeling import evaluate

        clean = evaluate(INVENTORY_DF)
        harsh = evaluate(make_estate("harsh"))
        clean_caught = clean["team"]["mislabels"]["flagged_and_wrong"]
        harsh_caught = harsh["team"]["mislabels"]["flagged_and_wrong"]
        assert clean_caught >= 4
        assert harsh_caught <= clean_caught
