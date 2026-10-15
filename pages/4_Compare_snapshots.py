"""Compare two saved reviews and send changed resources back to human review."""

import pandas as pd
import streamlit as st

from label_review import demo_enabled
from review_workspaces import compare_reviews, load_review, saved_reviews

st.set_page_config(page_title="Compare snapshots — obs-agent", page_icon="🔀", layout="wide")
st.title("🔀 Compare snapshots")
st.caption(
    "See what changed since an earlier saved review: new and removed resources, tag edits, "
    "ownership and cost-shape drift, and changed label suggestions."
)

if demo_enabled():
    st.warning("Snapshot comparison is disabled in the public demo. Run obs-agent locally.")
    st.stop()

reviews = saved_reviews()
if len(reviews) < 2:
    st.info("Save at least two inventory snapshots on Import / reopen inventory to compare them.")
    st.stop()

labels = {r["id"]: f"{r['name']} · {r['created_at']} · {r['id'][:8]}" for r in reviews}
ids = list(labels)
left, right = st.columns(2)
# saved_reviews() is newest first: default to the latest snapshot versus the one before it.
previous_id = left.selectbox("Previous snapshot", ids, index=1, format_func=labels.get)
current_id = right.selectbox("Current snapshot", ids, index=0, format_func=labels.get)
if previous_id == current_id:
    st.error("Choose two different saved reviews to compare.")
    st.stop()

result = compare_reviews(previous_id, current_id)

if result["vocabulary_changed"]:
    st.warning(
        "These snapshots use different vocabularies ("
        + ", ".join(result["vocabulary_differences"])
        + "). Tag and suggestion changes may reflect the vocabulary rather than the estate.",
        icon="⚠️",
    )
st.info(
    "Earlier decisions stay with the snapshot they were made on. They are shown for context "
    "and are never copied into the current review.",
    icon="🔒",
)

cols = st.columns(4)
cols[0].metric("New", len(result["added"]))
cols[1].metric("Removed", len(result["removed"]))
cols[2].metric("Changed", len(result["changed"]))
cols[3].metric("Unchanged", result["unchanged"])


def _decisions(items: list[dict]) -> str:
    return "; ".join(
        f"{d['label']} {d['decision']}"
        + (f" → {d['value']}" if d["value"] else "")
        + (f" by {d['reviewer']}" if d["reviewer"] else "")
        for d in items
    )


def _text(value) -> str:
    if isinstance(value, list):
        return ", ".join(value)
    return "" if value is None else str(value)


st.subheader("Changed")
if result["changed"]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "resource_id": item["resource_id"],
                    "field": change["field"],
                    "previous": _text(change["previous"]),
                    "current": _text(change["current"]),
                    "reason": change["reason"],
                    "in review queue": item["needs_attention"],
                    "earlier decisions (not carried over)": _decisions(item["earlier_decisions"]),
                }
                for item in result["changed"]
                for change in item["changes"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.success("No existing resource changed enough to need renewed review.")


def _summaries(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resource_id": item["resource_id"],
                "cloud": item["cloud"],
                "parent": item["parent"],
                "owners": ", ".join(item["owners"]),
                **{k: v or "" for k, v in item["tags"].items()},
            }
            | (
                {"earlier decisions": _decisions(item["earlier_decisions"])}
                if "earlier_decisions" in item
                else {"in review queue": item["needs_attention"]}
            )
            for item in items
        ]
    )


st.subheader("New")
if result["added"]:
    st.dataframe(_summaries(result["added"]), hide_index=True, width="stretch")
else:
    st.caption("No new resources.")

st.subheader("Removed")
if result["removed"]:
    st.dataframe(_summaries(result["removed"]), hide_index=True, width="stretch")
else:
    st.caption("No removed resources.")

flagged = [i["resource_id"] for i in result["changed"] + result["added"]]
if flagged and st.button(
    f"Review {len(flagged)} new or changed resource(s) in the current snapshot", type="primary"
):
    inventory, _ = load_review(current_id)
    st.session_state["review_id"] = current_id
    st.session_state["imported_inventory"] = inventory
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["drift_filter"] = {
        "review_id": current_id,
        "since": labels[previous_id],
        "resource_ids": flagged,
    }
    st.success(
        "Current snapshot opened. Label review filters to these resources; resources whose tags "
        "look correct have no suggestion to decide and do not appear in the queue."
    )
    st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")
