"""Label review page: work through untagged / suspect tags and export the result.

Opened from the sidebar of the main app (`uv run streamlit run app.py`).
Decisions are shared with the REST API through the same SQLite database
(OBS_AGENT_DECISIONS, default label_decisions.db).
"""

import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from label_config import inventory_config
from label_review import (
    DecisionError,
    DecisionStore,
    coverage,
    current_decisions,
    demo_enabled,
    editor_changes,
    export,
    feedback_metrics,
    pending_changes,
    review_queue,
)
from labeling import AUTO_APPLY_THRESHOLD
from review_context import review_context
from review_workspaces import load_review

st.set_page_config(page_title="Label review — obs-agent", page_icon="🏷️", layout="wide")

DEMO = demo_enabled()
imported_inventory = None if DEMO else st.session_state.get("imported_inventory")
if DEMO:
    # Public demo: each browser session gets its own throwaway decisions file,
    # so visitors never see or overwrite each other's choices.
    st.session_state.setdefault("demo_id", uuid.uuid4().hex)
    demo_file = f"obs-agent-{st.session_state.demo_id}.db"
    store = DecisionStore(Path(tempfile.gettempdir()) / demo_file)
elif st.session_state.get("review_id"):
    imported_inventory, store = load_review(st.session_state.review_id)
    st.session_state["imported_inventory"] = imported_inventory
elif imported_inventory is not None:
    st.error("Save or reopen this inventory on the Import page before reviewing.")
    st.stop()
else:
    store = DecisionStore()
st.session_state.setdefault("editor_version", 0)


def _refresh():
    # A new editor key drops stale in-table edits after decisions change.
    st.session_state.editor_version += 1
    st.rerun()


# ── Header ─────────────────────────────────────────────────────────────

st.title("🏷️ Label review")
st.caption(
    "Suggested `team` / `env` tags for untagged Azure subscriptions and GCP projects, "
    f"plus existing tags the model thinks are wrong. Suggestions with a model score "
    f"≥{AUTO_APPLY_THRESHOLD:.0%} are eligible for expedited review; everything else needs an "
    "explicit human decision. The score ranks suggestions; it is not a calibrated probability. "
    "Where ownership evidence is missing, unsupported or conflicting, no team is suggested: "
    "the row says why, and only an explicit override resolves it."
)

if imported_inventory is not None:
    st.info(
        "Reviewing your imported inventory. Its snapshot, vocabulary, and decisions are kept in a "
        "saved local review database. Reopen it from Import / reopen inventory after a restart.",
        icon="📥",
    )
elif DEMO:
    st.info(
        "Demo on synthetic data: your decisions are private to this browser session "
        "and are discarded when it ends. Try accepting a few, then download the export.",
        icon="🧪",
    )

reviewer = st.text_input(
    "Reviewer name or ID",
    key="reviewer",
    value="demo-reviewer" if DEMO else os.environ.get("OBS_AGENT_REVIEWER", ""),
    max_chars=128,
    help="Recorded with each approval. This local identifier is self-reported.",
).strip()
if not reviewer:
    st.info("Enter your reviewer name or ID before saving or bulk-accepting decisions.")
context = review_context(imported_inventory)
with st.expander("Review snapshot"):
    st.code(f"Inventory: {context['inventory_id']}\nInference: {context['model_version']}")
valid_decisions = current_decisions(store, imported_inventory)
if len(valid_decisions) < len(store.all()):
    st.warning(
        "Earlier decisions lack matching snapshot/version evidence. Review them again to export."
    )
cov = coverage(imported_inventory)
decided = {f"{d['resource_id']}|{d['label']}": d for d in valid_decisions}
cols = st.columns(7)
cols[0].metric("Need attention", cov["needs_attention"], help=f"of {cov['total']} resources")
cols[1].metric("Untagged team", cov["labels"]["team"]["untagged"])
cols[2].metric("Untagged env", cov["labels"]["env"]["untagged"])
cols[3].metric("Suspect tags", sum(c["suspect"] for c in cov["labels"].values()))
cols[4].metric("Auto-apply ready", sum(c["auto_apply_ready"] for c in cov["labels"].values()))
cols[5].metric(
    "Insufficient evidence",
    sum(c["insufficient_evidence"] for c in cov["labels"].values()),
    help="Untagged labels with no suggestion: "
    + ", ".join(
        f"{reason.replace('_', ' ')} {n}"
        for reason, n in cov["labels"]["team"]["insufficient_evidence_reasons"].items()
    ),
)
cols[6].metric("Reviewed", len(decided))

quality = feedback_metrics(store.feedback())
with st.expander("Review quality", expanded=bool(quality["reviewed"])):
    st.caption(
        "Reviewer-agreement metrics compare each suggestion with the reviewer-resolved value. "
        "They are useful for monitoring, not an unbiased production benchmark."
    )
    qcols = st.columns(4)
    qcols[0].metric("Review actions", quality["reviewed"])
    qcols[1].metric("Scored", quality["scored"])
    qcols[2].metric("Unresolved", quality["unresolved"])
    qcols[3].metric("Expedited-review threshold", f"{quality['auto_apply_threshold']:.0%}")

    if not quality["scored"]:
        st.info(
            "No reviewer-resolved labels yet. Accept, override, or reject an existing tag "
            "to score it."
        )
    else:
        for label_name, summary in quality["labels"].items():
            mcols = st.columns(3)
            mcols[0].metric(f"{label_name} precision", f"{summary['precision']:.1%}")
            mcols[1].metric(f"{label_name} recall", f"{summary['recall']:.1%}")
            mcols[2].metric(f"{label_name} macro-F1", f"{summary['macro_f1']:.2f}")

        trend = [
            {
                "month": period["period"],
                "label": label_name,
                **{k: period[k] for k in ("scored", "precision", "recall", "macro_f1")},
            }
            for label_name, summary in quality["labels"].items()
            for period in summary["periods"]
        ]
        if trend:
            st.caption("Monthly reviewer-agreement trend")
            st.dataframe(pd.DataFrame(trend), hide_index=True, width="stretch")

        buckets = [
            {"label": label_name, **bucket}
            for label_name, summary in quality["labels"].items()
            for bucket in summary["confidence_buckets"]
        ]
        st.caption("Model-score buckets (the threshold is the high-score drift check)")
        st.dataframe(pd.DataFrame(buckets), hide_index=True, width="stretch")

# ── Filters ────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
label = st.sidebar.radio("Label", ["team", "env"], horizontal=True)
status = st.sidebar.selectbox("Status", ["untagged or suspect", "untagged", "suspect"])
cloud = st.sidebar.selectbox("Cloud", ["both", "azure", "gcp"])
lo, hi = st.sidebar.slider(
    "Model score",
    0.0,
    1.0,
    (0.0, 1.0),
    step=0.05,
    help="Labels withheld for insufficient evidence have no score and count as 0.",
)
only_insufficient = st.sidebar.checkbox("Only insufficient evidence", value=False)
search = st.sidebar.text_input("Resource contains")
hide_reviewed = st.sidebar.checkbox("Hide reviewed", value=False)
drift_filter = st.session_state.get("drift_filter")
drifted = None
if drift_filter and drift_filter["review_id"] == st.session_state.get("review_id"):
    if st.sidebar.checkbox(f"Only new or changed since {drift_filter['since']}", value=True):
        drifted = set(drift_filter["resource_ids"])

rows = review_queue(
    label=label,
    status=None if status.startswith("untagged or") else status,
    cloud=None if cloud == "both" else cloud,
    min_confidence=lo,
    max_confidence=hi,
    search=search or None,
    inventory=imported_inventory,
    evidence="insufficient" if only_insufficient else None,
)

records = []
for r in rows:
    lr = r["labels"][label]
    d = decided.get(f"{r['resource_id']}|{label}")
    if hide_reviewed and d:
        continue
    if drifted is not None and r["resource_id"] not in drifted:
        continue
    records.append(
        {
            "resource_id": r["resource_id"],
            "cloud": r["cloud"],
            "parent": r["parent"],
            "owners": ", ".join(r["owners"]),
            "status": lr["status"],
            "current": lr["current"] or "",
            "suggested": lr["suggested"],
            "confidence": lr["confidence"],
            "action": lr["action"],
            "evidence": (lr["evidence"] or {}).get("status", "").replace("_", " "),
            "why": f"{lr['source']}: {lr['reason']}",
            "alternatives": ", ".join(
                f"{a['value']} {a['confidence']:.0%}" for a in lr["alternatives"]
            ),
            "cost_center": (
                "Conflicting: " + ", ".join(r["cost_center"]["conflicting"])
                if r["cost_center"]["conflicting"]
                else r["cost_center"]["implied"] or "No mapping"
            ),
            "decision": d["decision"] if d else "",
            "override": d["value"] if d and d["decision"] == "override" else None,
        }
    )
df = pd.DataFrame(records)

# ── Review table ───────────────────────────────────────────────────────

st.subheader(f"{len(df)} resources to review for `{label}`")

if df.empty:
    st.success("Nothing matches these filters.")
else:
    edited = st.data_editor(
        df,
        key=f"editor-{label}-{st.session_state.editor_version}",
        hide_index=True,
        width="stretch",
        height=min(600, 38 + 35 * len(df)),
        disabled=[c for c in df.columns if c not in ("decision", "override")],
        # Editable columns sit next to the suggestion so no sideways scrolling.
        column_order=[
            "resource_id",
            "suggested",
            "confidence",
            "decision",
            "override",
            "current",
            "status",
            "action",
            "evidence",
            "cloud",
            "parent",
            "owners",
            "alternatives",
            "cost_center",
            "why",
        ],
        column_config={
            "resource_id": st.column_config.TextColumn("Resource", pinned=True),
            "confidence": st.column_config.ProgressColumn(
                "Model score", min_value=0.0, max_value=1.0, format="percent"
            ),
            "why": st.column_config.TextColumn("Why", width="large"),
            "decision": st.column_config.SelectboxColumn(
                "Decision",
                options=["", "accept", "reject", "override"],
                help="accept = apply the suggestion · reject = keep today's tag · "
                "override = apply the value in the next column (the only choice when no "
                "value is suggested)",
            ),
            "override": st.column_config.SelectboxColumn(
                "Override value",
                options=sorted(inventory_config(imported_inventory)[label]["values"]),
            ),
        },
    )

    left, right = st.columns([1, 3])
    if left.button("Save decisions", icon=":material/save:", type="primary", disabled=not reviewer):
        errors, saved = [], 0
        for rid, decision, override in editor_changes(df, edited):
            try:
                if decision:
                    store.set(rid, label, decision, override, imported_inventory, reviewer=reviewer)
                else:
                    store.delete(rid, label)
                saved += 1
            except DecisionError as exc:
                errors.append(f"{rid}: {exc}")
        for e in errors:
            st.error(e)
        if saved and not errors:
            st.toast(f"Saved {saved} decision(s)")
            _refresh()

    auto = df[(df["action"] == "auto-apply") & (df["decision"] == "")]
    if right.button(
        f"Accept all {len(auto)} auto-apply suggestions shown",
        icon=":material/done_all:",
        disabled=auto.empty or not reviewer,
    ):
        for rid in auto["resource_id"]:
            store.set(rid, label, "accept", inventory=imported_inventory, reviewer=reviewer)
        st.toast(f"Accepted {len(auto)} suggestion(s)")
        _refresh()

# ── Export ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Export")
include_auto = st.checkbox(
    f"Include unreviewed auto-apply suggestions (≥{AUTO_APPLY_THRESHOLD:.0%})",
    value=False,
    help="Off by default: export only what a reviewer accepted or overrode.",
)
try:
    changes = pending_changes(store, include_auto, imported_inventory)
except DecisionError as exc:
    st.error(str(exc))
    st.stop()
st.caption(
    f"{len(changes)} resource(s) would change. Overrides win over accepts, "
    "accepts over auto-apply; rejected suggestions are never exported."
)

c1, c2, c3, c4 = st.columns(4)
c1.download_button(
    "Reviewed change plan",
    export(store, "plan", inventory=imported_inventory),
    "approved-label-change-plan.json",
    "application/json",
    help="Reviewed decisions only, with the before/after diff and approval evidence.",
    icon=":material/fact_check:",
    disabled=not pending_changes(store, include_auto=False, inventory=imported_inventory),
)
c2.download_button(
    "CSV",
    export(store, "csv", include_auto, imported_inventory),
    "label-changes.csv",
    "text/csv",
    icon=":material/download:",
    disabled=not changes,
)
c3.download_button(
    "az / gcloud script",
    export(store, "sh", include_auto, imported_inventory),
    "apply-labels.sh",
    "text/x-shellscript",
    icon=":material/download:",
    disabled=not changes,
)
c4.download_button(
    "Terraform tfvars",
    export(store, "tfvars", include_auto, imported_inventory),
    "labels.auto.tfvars.json",
    "application/json",
    icon=":material/download:",
    disabled=not changes,
)

if changes:
    with st.expander("Preview changes"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "resource_id": c["resource_id"],
                        "cloud": c["cloud"],
                        "label": k,
                        "previous": c["previous"][k],
                        "new": v,
                        "how": how,
                    }
                    for c in changes
                    for k, (v, how) in c["tags"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
