"""
Label inference for untagged Azure subscriptions / GCP projects.

Supervised learning in the textbook shape:

    labelled resources  ──► fit ──► model w
    unlabelled resources ─────────► model w ──► predicted label (+ confidence)

Three strategies, scored against inventory.py's ground truth:

  rules   — inherit the label from the parent (management group / folder)
            or from resources sharing the name prefix; env from name tokens.
  model   — logistic regression over name tokens, parent, owners,
            services and cost shape.
  hybrid  — whichever of rule and model is more confident.

It also flags labelled resources whose tag the model strongly disagrees
with (likely mislabels), using out-of-fold predictions so a resource never
votes on itself.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from inventory import ENV_NAME_TOKENS, ENVS, INVENTORY_DF, TEAM_NAMES

LABELS = ("team", "env")

# Best strategy per label, from evaluate(): parent/prefix inheritance is
# confidently wrong after reorgs, so team is left to the model; env name
# tokens are near-perfect, so env uses whichever source is more confident.
BEST_STRATEGY = {"team": "model", "env": "hybrid"}

# Confidence above which a suggestion is eligible for expedited human review.
AUTO_APPLY_THRESHOLD = 0.9
CALIBRATION_BINS = 10

# ── Cleaning ───────────────────────────────────────────────────────────
# Built by eyeballing value_counts() of the raw tags — the unglamorous
# first step that recovers a surprising share of "unlabelled" resources.

TEAM_ALIASES = {
    "payment": "payments",
    "dataplatform": "data-platform",
    "dp": "data-platform",
    "machine-learning": "ml",
    "digital-web": "web",
    "iam": "identity",
    "o11y": "observability",
    "obs": "observability",
    "secops": "security",
    "sec": "security",
    "finance": "finance-apps",
    "fin-apps": "finance-apps",
}
ENV_ALIASES = {
    "production": "prod",
    "prd": "prod",
    "stg": "staging",
    "uat": "staging",
    "development": "dev",
    "sbx": "sandbox",
    "poc": "sandbox",
}
ALLOWED = {"team": set(TEAM_NAMES), "env": set(ENVS)}
ALIASES = {"team": TEAM_ALIASES, "env": ENV_ALIASES}


def normalise(label: str, value, config: dict | None = None) -> str | None:
    """Map a raw tag value onto the allowed vocabulary, or None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = re.sub(r"[\s_]+", "-", str(value).strip().lower())
    if config is None or (
        key not in config[label]["values"] and key not in config[label]["aliases"]
    ):
        key = re.sub(r"^team-|-team$", "", key)
    aliases = ALIASES[label] if config is None else config[label]["aliases"]
    allowed = ALLOWED[label] if config is None else config[label]["values"]
    key = aliases.get(key, key)
    return key if key in allowed else None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Add <label>_clean columns (None = treat as unlabelled)."""
    df = df.copy()
    for label in LABELS:
        df[f"{label}_clean"] = pd.Series(
            [normalise(label, v, df.attrs.get("label_config")) for v in df[label]],
            index=df.index,
            dtype=object,
        )
    return df


# ── Features ───────────────────────────────────────────────────────────

_OPAQUE = re.compile(r"^[0-9a-f]{6,}$|^\d+$")


def name_tokens(resource_id: str) -> list[str]:
    """Meaningful tokens of a resource name (drops ids, hashes, numbers)."""
    return [
        t
        for t in resource_id.lower().split("-")
        if t and not _OPAQUE.match(t) and t not in ("sub", "proj")
    ]


# Who owns a resource says which team it belongs to, not whether it is prod.
_ENV_IGNORES = ("owner=", "creator=", "parent=")


def features(row, label: str = "team") -> dict:
    f = _all_features(row)
    if label == "env":
        f = {k: v for k, v in f.items() if not k.startswith(_ENV_IGNORES)}
    return f


def _all_features(row) -> dict:
    tokens = name_tokens(row["resource_id"])
    f = {f"tok={t}": 1 for t in tokens}
    if tokens:
        f[f"prefix={tokens[0]}"] = 1
    f[f"parent={row['parent']}"] = 1
    f[f"cloud={row['cloud']}"] = 1
    f[f"creator={row['created_by']}"] = 1
    for o in row["owners"]:
        f[f"owner={o}"] = 1
    for s in row["top_services"]:
        f[f"svc={s}"] = 1
    f["log_daily_cost"] = float(np.log1p(row["avg_daily_cost"]))
    f["weekend_ratio"] = float(row["weekend_ratio"])
    f["cost_cv"] = float(row["cost_cv"])
    return f


def _new_model():
    return make_pipeline(
        DictVectorizer(sparse=True),
        LogisticRegression(C=2.0, max_iter=5000, class_weight="balanced"),
    )


# ── Rules baseline ─────────────────────────────────────────────────────


@dataclass
class Suggestion:
    label: str | None
    confidence: float
    source: str  # "rule:parent" | "rule:name" | "model" | "none"
    reason: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


NO_SUGGESTION = Suggestion(None, 0.0, "none", "no rule fired")


class Rules:
    """Majority-vote inheritance learned from the labelled resources."""

    MIN_SHARE = 0.8
    MIN_SUPPORT = 3

    def __init__(self, train: pd.DataFrame, label: str):
        self.label = label
        config = train.attrs.get("label_config")
        tokens = ENV_NAME_TOKENS if config is None else config["env"]["name_tokens"]
        self.env_tokens = {tok: env for env, names in tokens.items() for tok in names}
        y = train[f"{label}_clean"]
        self.by_parent = self._votes(train["parent"], y)
        prefixes = [(name_tokens(r) or [""])[0] for r in train["resource_id"]]
        self.by_prefix = self._votes(pd.Series(prefixes, index=train.index), y)

    @staticmethod
    def _votes(keys: pd.Series, y: pd.Series) -> dict[str, tuple[str, float, int]]:
        out = {}
        for key, group in y.groupby(keys):
            top, count = Counter(group).most_common(1)[0]
            out[key] = (top, count / len(group), len(group))
        return out

    def _vote(self, table, key, kind) -> Suggestion:
        if key not in table:
            return NO_SUGGESTION
        top, share, n = table[key]
        if share < self.MIN_SHARE or n < self.MIN_SUPPORT:
            return NO_SUGGESTION
        return Suggestion(
            top,
            round(share, 3),
            f"rule:{kind}",
            f"{share:.0%} of {n} labelled resources with {kind} '{key}' are {self.label}={top}",
        )

    def suggest(self, row) -> Suggestion:
        if self.label == "env":
            # Name tokens are a direct, high-precision env signal.
            for tok in name_tokens(row["resource_id"]):
                if tok in self.env_tokens:
                    env = self.env_tokens[tok]
                    return Suggestion(env, 0.95, "rule:name", f"name token '{tok}' means env={env}")
            return NO_SUGGESTION
        prefix = (name_tokens(row["resource_id"]) or [""])[0]
        candidates = [
            self._vote(self.by_parent, row["parent"], "parent"),
            self._vote(self.by_prefix, prefix, "name prefix"),
        ]
        return max(candidates, key=lambda s: s.confidence)


# ── Model ──────────────────────────────────────────────────────────────


class Model:
    def __init__(self, train: pd.DataFrame, label: str):
        self.label = label
        X = [features(r, label) for _, r in train.iterrows()]
        self.pipe = _new_model().fit(X, train[f"{label}_clean"])
        self.vec, self.clf = self.pipe.named_steps.values()

    def suggest(self, row) -> Suggestion:
        x = features(row, self.label)
        proba = self.pipe.predict_proba([x])[0]
        order = np.argsort(proba)[::-1]
        classes = self.clf.classes_
        best = classes[order[0]]
        return Suggestion(
            best,
            round(float(proba[order[0]]), 3),
            "model",
            self._explain(x, order[0]),
            [(classes[i], round(float(proba[i]), 3)) for i in order[:3]],
        )

    def _explain(self, x: dict, class_idx: int, k: int = 3) -> str:
        """Top features pushing toward the predicted class (coef × value)."""
        # Binary problems store one coefficient row, signed toward classes_[1].
        coef = (
            self.clf.coef_[0] * (1 if class_idx == 1 else -1)
            if self.clf.coef_.shape[0] == 1
            else self.clf.coef_[class_idx]
        )
        vocab = self.vec.vocabulary_
        contrib = [(name, coef[vocab[name]] * val) for name, val in x.items() if name in vocab]
        top = [f"{n} (+{c:.2f})" for n, c in sorted(contrib, key=lambda t: -t[1])[:k] if c > 0]
        return "strongest signals: " + (", ".join(top) if top else "none")


# ── Fitted bundle ──────────────────────────────────────────────────────


class LabelInference:
    """Rules + model for every label, fitted on one inventory snapshot."""

    def __init__(self, inventory: pd.DataFrame):
        self.df = clean(inventory)
        self.rules: dict[str, Rules] = {}
        self.models: dict[str, Model] = {}
        for label in LABELS:
            train = self.df[self.df[f"{label}_clean"].notna()]
            self.rules[label] = Rules(train, label)
            self.models[label] = Model(train, label)
        # cost_center follows team: learn the mapping from labelled rows.
        tagged = self.df.dropna(subset=["team_clean", "cost_center"])
        self.cost_center_of = (
            tagged.groupby("team_clean")["cost_center"].agg(lambda s: s.mode().iat[0]).to_dict()
        )

    def suggest(self, row, label: str, strategy: str = "hybrid") -> Suggestion:
        if strategy == "rules":
            return self.rules[label].suggest(row)
        model = self.models[label].suggest(row)
        if strategy == "model":
            return model
        rule = self.rules[label].suggest(row)
        return rule if rule.confidence > model.confidence else model

    def row(self, resource_id: str):
        match = self.df[self.df["resource_id"] == resource_id]
        return None if match.empty else match.iloc[0]

    def unlabelled(self, label: str) -> pd.DataFrame:
        return self.df[self.df[f"{label}_clean"].isna()]


def suspect_labels(df: pd.DataFrame, label: str, min_confidence: float = 0.8) -> pd.DataFrame:
    """Labelled rows where out-of-fold predictions confidently disagree with the tag."""
    df = clean(df)
    train = df[df[f"{label}_clean"].notna()]
    X = [features(r, label) for _, r in train.iterrows()]
    y = train[f"{label}_clean"]
    n_splits = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=0)
    proba = cross_val_predict(_new_model(), X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(y.unique()))
    out = train.assign(predicted=classes[proba.argmax(1)], confidence=proba.max(1).round(3))
    if label == "env":
        # Same hybrid as for predictions: an env token in the name outranks
        # a less confident model ("dp-airflow-prod" tagged prod is fine).
        rules = Rules(train, label)
        for idx, row in out.iterrows():
            r = rules.suggest(row)
            if r.label is not None and r.confidence > row["confidence"]:
                out.loc[idx, ["predicted", "confidence"]] = [r.label, r.confidence]
    return out[(out["predicted"] != y) & (out["confidence"] >= min_confidence)]


@lru_cache(maxsize=1)
def get_inference() -> LabelInference:
    """The fitted bundle for the shared synthetic inventory (built once)."""
    return LabelInference(INVENTORY_DF)


@lru_cache(maxsize=2)
def get_suspects(label: str) -> pd.DataFrame:
    return suspect_labels(INVENTORY_DF, label)


# ── Evaluation against ground truth ────────────────────────────────────


def expected_calibration_error(
    predicted: pd.Series,
    confidence: pd.Series,
    truth: pd.Series,
    bins: int = CALIBRATION_BINS,
) -> float:
    """Return the confidence-weighted gap between predicted and observed accuracy."""
    covered = predicted.notna()
    if not covered.any():
        return 0.0
    conf = confidence[covered]
    correct = (predicted[covered] == truth[covered]).astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        members = (conf > lower) & (conf <= upper)
        if members.any():
            ece += members.mean() * abs(conf[members].mean() - correct[members].mean())
    return float(ece)


def evaluate(inventory: pd.DataFrame = INVENTORY_DF) -> dict:
    """Score every strategy on the unlabelled resources.

    coverage        share of unlabelled resources that got any suggestion
    accuracy        correct / suggested
    macro_f1        F1 averaged over classes (small teams count equally),
                    on the suggested rows
    auto_coverage   share with confidence >= AUTO_APPLY_THRESHOLD
    auto_precision  accuracy on that auto-applied slice
    ece             expected calibration error (lower is better)
    top3            true label among the model's top 3 (model only)
    """
    inf = LabelInference(inventory)
    results: dict = {}
    for label in LABELS:
        test = inf.unlabelled(label)
        truth = test[f"true_{label}"]
        results[label] = {"n_unlabelled": len(test)}
        for strategy in ("rules", "model", "hybrid"):
            sugg = [inf.suggest(r, label, strategy) for _, r in test.iterrows()]
            pred = pd.Series([s.label for s in sugg], index=test.index)
            conf = pd.Series([s.confidence for s in sugg], index=test.index)
            covered = pred.notna()
            auto = covered & (conf >= AUTO_APPLY_THRESHOLD)
            m = {
                "coverage": covered.mean(),
                "accuracy": (pred[covered] == truth[covered]).mean() if covered.any() else 0.0,
                "macro_f1": f1_score(truth[covered], pred[covered], average="macro")
                if covered.any()
                else 0.0,
                "auto_coverage": auto.mean(),
                "auto_precision": (pred[auto] == truth[auto]).mean() if auto.any() else 0.0,
                "ece": expected_calibration_error(pred, conf, truth),
            }
            if strategy == "model":
                m["top3"] = float(
                    np.mean(
                        [
                            t in [c for c, _ in s.alternatives]
                            for s, t in zip(sugg, truth, strict=False)
                        ]
                    )
                )
            results[label][strategy] = {k: round(float(v), 3) for k, v in m.items()}

        # Mislabel detection: how many flagged tags are genuinely wrong?
        sus = suspect_labels(inventory, label)
        cleaned = clean(inventory)
        wrong = cleaned[f"{label}_clean"].notna() & (
            cleaned[f"{label}_clean"] != cleaned[f"true_{label}"]
        )
        caught = (sus[f"{label}_clean"] != sus[f"true_{label}"]).sum()
        results[label]["mislabels"] = {
            "actually_wrong": int(wrong.sum()),
            "flagged": len(sus),
            "flagged_and_wrong": int(caught),
        }
    return results


def format_report(results: dict) -> str:
    lines = []
    for label, r in results.items():
        lines += [
            f"\n{label.upper()}  ({r['n_unlabelled']} unlabelled after cleaning)",
            f"  {'strategy':8} {'coverage':>9} {'accuracy':>9} {'macroF1':>8} "
            f"{'auto≥' + str(AUTO_APPLY_THRESHOLD):>8} {'auto-prec':>10} {'ECE':>6} {'top3':>6}",
        ]
        for strategy in ("rules", "model", "hybrid"):
            m = r[strategy]
            top3 = f"{m['top3']:.0%}" if "top3" in m else "-"
            lines.append(
                f"  {strategy:8} {m['coverage']:>9.0%} {m['accuracy']:>9.0%} "
                f"{m['macro_f1']:>8.2f} {m['auto_coverage']:>8.0%} "
                f"{m['auto_precision']:>10.0%} {m['ece']:>6.1%} {top3:>6}"
            )
        ml = r["mislabels"]
        lines.append(
            f"  mislabel check: {ml['flagged']} flagged, {ml['flagged_and_wrong']} "
            f"truly wrong (of {ml['actually_wrong']} wrong tags in the data)"
        )
    return "\n".join(lines)


def format_degradation(clean: dict, harsh: dict) -> str:
    """Side-by-side comparison of the two estates, as a degradation table.

    This is the honest version of the headline metrics: the clean estate is
    what the pipeline was tuned against, the harsh one is closer to a real,
    ungoverned estate. The delta is what to expect before measuring on your
    own data.
    """
    lines = ["", "Degradation: clean → harsh estate", "=".rjust(58, "=")]
    lines.append(f"  {'label/strategy':20} {'metric':16} {'clean':>8} {'harsh':>8} {'delta':>8}")
    for label in LABELS:
        strategy = BEST_STRATEGY[label]
        c, h = clean[label][strategy], harsh[label][strategy]
        for key in ("coverage", "accuracy", "macro_f1", "auto_coverage", "auto_precision", "ece"):
            delta = h[key] - c[key]
            if key == "macro_f1":
                c_s, h_s, d_s = f"{c[key]:.2f}", f"{h[key]:.2f}", f"{delta:+.2f}"
            else:
                c_s, h_s, d_s = f"{c[key]:.0%}", f"{h[key]:.0%}", f"{delta:+.0%}"
            lines.append(f"  {label + '/' + strategy:20} {key:16} {c_s:>8} {h_s:>8} {d_s:>8}")
    lines += ["", "  mislabel check (flags that were genuinely wrong):"]
    for label in LABELS:
        c, h = clean[label]["mislabels"], harsh[label]["mislabels"]
        cf = f"{c['flagged_and_wrong']}/{c['flagged']}" if c["flagged"] else "n/a"
        hf = f"{h['flagged_and_wrong']}/{h['flagged']}" if h["flagged"] else "n/a"
        lines.append(f"    {label:20} clean {cf:>8}   harsh {hf:>8}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score label inference against ground truth.")
    ap.add_argument(
        "--hardness",
        choices=("clean", "harsh", "reorg", "both"),
        default="clean",
        help="which synthetic estate to score (default: clean, the published one)",
    )
    args = ap.parse_args()

    if args.hardness in ("clean", "both"):
        raw = INVENTORY_DF
        print(f"Inventory (clean): {len(raw)} resources")
        for label in LABELS:
            raw_ok = raw[label].isin(ALLOWED[label]).mean()
            clean_ok = clean(raw)[f"{label}_clean"].notna().mean()
            print(f"  {label}: {raw_ok:.0%} usable as-is → {clean_ok:.0%} usable after cleaning")
        print(format_report(evaluate()))

    if args.hardness in ("harsh", "both"):
        from inventory import make_estate

        harsh_df = make_estate("harsh")
        print(f"\n\nInventory (harsh): {len(harsh_df)} resources")
        for label in LABELS:
            raw_ok = harsh_df[label].isin(ALLOWED[label]).mean()
            clean_ok = clean(harsh_df)[f"{label}_clean"].notna().mean()
            print(f"  {label}: {raw_ok:.0%} usable as-is → {clean_ok:.0%} usable after cleaning")
        print(format_report(evaluate(harsh_df)))

    if args.hardness == "reorg":
        from inventory import make_estate

        reorg_df = make_estate("reorg")
        print(f"Inventory (reorg): {len(reorg_df)} resources")
        print(format_report(evaluate(reorg_df)))

    if args.hardness == "both":
        print(format_degradation(evaluate(INVENTORY_DF), evaluate(make_estate("harsh"))))
