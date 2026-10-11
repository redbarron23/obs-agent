"""Label review page: work through untagged / suspect tags and export the result.

Opened from the sidebar of the main app (`uv run streamlit run app.py`).
Decisions are shared with the REST API through the same DecisionStore file
(OBS_AGENT_DECISIONS, default label_decisions.json).
"""

import pandas as pd
import streamlit as st

from label_review import (
    DecisionError,
    DecisionStore,
    coverage,
    editor_changes,
    export,
    pending_changes,
    review_queue,
)
from labeling import ALLOWED, AUTO_APPLY_THRESHOLD

st.set_page_config(page_title="Label review — obs-agent", page_icon="🏷️", layout="wide")

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
    f"plus existing tags the model thinks are wrong. Suggestions at ≥{AUTO_APPLY_THRESHOLD:.0%} "
    "confidence are safe to auto-apply; everything else needs a human."
)

cov = coverage()
decided = {f"{d['resource_id']}|{d['label']}": d for d in store.all()}
cols = st.columns(6)
cols[0].metric("Need attention", f"{cov['needs_attention']} / {cov['total']}")
cols[1].metric("Untagged team", cov["labels"]["team"]["untagged"])
cols[2].metric("Untagged env", cov["labels"]["env"]["untagged"])
cols[3].metric("Suspect tags", sum(c["suspect"] for c in cov["labels"].values()))
cols[4].metric("Auto-apply ready", sum(c["auto_apply_ready"] for c in cov["labels"].values()))
cols[5].metric("Reviewed", len(decided))

# ── Filters ────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
label = st.sidebar.radio("Label", ["team", "env"], horizontal=True)
status = st.sidebar.selectbox("Status", ["untagged or suspect", "untagged", "suspect"])
cloud = st.sidebar.selectbox("Cloud", ["both", "azure", "gcp"])
lo, hi = st.sidebar.slider("Confidence", 0.0, 1.0, (0.0, 1.0), step=0.05)
search = st.sidebar.text_input("Resource contains")
hide_reviewed = st.sidebar.checkbox("Hide reviewed", value=False)

rows = review_queue(
    label=label,
    status=None if status.startswith("untagged or") else status,
    cloud=None if cloud == "both" else cloud,
    min_confidence=lo,
    max_confidence=hi,
    search=search or None,
)

records = []
for r in rows:
    lr = r["labels"][label]
    d = decided.get(f"{r['resource_id']}|{label}")
    if hide_reviewed and d:
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
            "why": f"{lr['source']}: {lr['reason']}",
            "alternatives": ", ".join(
                f"{a['value']} {a['confidence']:.0%}" for a in lr["alternatives"]
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
            "cloud",
            "parent",
            "owners",
            "alternatives",
            "why",
        ],
        column_config={
            "resource_id": st.column_config.TextColumn("Resource", pinned=True),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="percent"
            ),
            "why": st.column_config.TextColumn("Why", width="large"),
            "decision": st.column_config.SelectboxColumn(
                "Decision",
                options=["", "accept", "reject", "override"],
                help="accept = apply the suggestion · reject = keep today's tag · "
                "override = apply the value in the next column",
            ),
            "override": st.column_config.SelectboxColumn(
                "Override value", options=sorted(ALLOWED[label])
            ),
        },
    )

    left, right = st.columns([1, 3])
    if left.button("💾 Save decisions", type="primary"):
        errors, saved = [], 0
        for rid, decision, override in editor_changes(df, edited):
            try:
                if decision:
                    store.set(rid, label, decision, override)
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
    if right.button(f"✅ Accept all {len(auto)} auto-apply suggestions shown", disabled=auto.empty):
        for rid in auto["resource_id"]:
            store.set(rid, label, "accept")
        st.toast(f"Accepted {len(auto)} suggestion(s)")
        _refresh()

# ── Export ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Export")
include_auto = st.checkbox(
    f"Include unreviewed auto-apply suggestions (≥{AUTO_APPLY_THRESHOLD:.0%})",
    value=True,
    help="Off: export only what a reviewer accepted or overrode.",
)
changes = pending_changes(store, include_auto)
st.caption(
    f"{len(changes)} resource(s) would change. Overrides win over accepts, "
    "accepts over auto-apply; rejected suggestions are never exported."
)

c1, c2, c3 = st.columns(3)
c1.download_button(
    "⬇️ CSV",
    export(store, "csv", include_auto),
    "label-changes.csv",
    "text/csv",
    disabled=not changes,
)
c2.download_button(
    "⬇️ az / gcloud script",
    export(store, "sh", include_auto),
    "apply-labels.sh",
    "text/x-shellscript",
    disabled=not changes,
)
c3.download_button(
    "⬇️ Terraform tfvars",
    export(store, "tfvars", include_auto),
    "labels.auto.tfvars.json",
    "application/json",
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
