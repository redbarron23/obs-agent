"""Validate an inventory and save or reopen a local review workspace."""

from pathlib import Path

import streamlit as st

from inventory_import import REQUIRED_COLUMNS, InventoryImportError, load_inventory_csv
from label_config import LabelConfigError, load_label_config
from label_review import demo_enabled
from labeling import LABELS, normalise
from review_workspaces import load_review, review_directory, save_review, saved_reviews

st.set_page_config(page_title="Import inventory — obs-agent", page_icon="📥", layout="wide")
st.title("📥 Import / reopen inventory")
st.caption(
    "Review Azure subscriptions and GCP projects locally. No cloud credentials or tag writes."
)

if demo_enabled():
    st.warning(
        "Inventory import is disabled in the public demo. Run obs-agent locally for private data."
    )
    st.stop()

st.info(
    f"Saved reviews persist on this machine in {review_directory()}. "
    "They contain the uploaded inventory, vocabulary, and decisions. "
    "The API and optional chat continue to use synthetic data."
)


def open_review(review_id):
    inventory, _ = load_review(review_id)
    st.session_state["review_id"] = review_id
    st.session_state["imported_inventory"] = inventory
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


reviews = saved_reviews()
if reviews:
    st.subheader("Reopen a saved review")
    choices = {r["id"]: f"{r['name']} · {r['created_at']} · {r['id'][:8]}" for r in reviews}
    chosen = st.selectbox("Saved review", list(choices), format_func=choices.get)
    if st.button("Open saved review"):
        open_review(chosen)
        st.success("Saved inventory and decisions reopened.")
        st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")

st.subheader("Import a new snapshot")
fixtures = Path(__file__).resolve().parent.parent / "fixtures"
st.download_button(
    "Download CSV example / template",
    (fixtures / "sanitized_inventory_sample.csv").read_bytes(),
    "inventory-template.csv",
    "text/csv",
)
st.download_button(
    "Download vocabulary example",
    (fixtures / "label_config_example.json").read_bytes(),
    "label-vocabulary.json",
    "application/json",
)
with st.expander("Required CSV columns"):
    st.code(", ".join(sorted(REQUIRED_COLUMNS)), language=None)
    st.caption(
        "Use JSON string lists or | separators for owners and top_services. "
        "Each label needs two recognized values with at least two examples each."
    )

vocabulary = st.file_uploader("Vocabulary JSON (optional; otherwise use demo values)", type="json")
uploaded = st.file_uploader("Sanitized Azure/GCP inventory CSV", type="csv")
if uploaded:
    try:
        config = load_label_config(vocabulary.getvalue()) if vocabulary else None
        inventory = load_inventory_csv(uploaded.getvalue(), config)
    except (InventoryImportError, LabelConfigError) as exc:
        st.error(str(exc))
    else:
        st.success(
            f"Validated {len(inventory)} resources across {inventory.cloud.nunique()} clouds."
        )
        for label in LABELS:
            recognized = (
                inventory[label]
                .map(lambda v, name=label: normalise(name, v, inventory.attrs["label_config"]))
                .notna()
            )
            unknown = inventory[label].notna() & ~recognized
            st.caption(
                f"{label}: {recognized.sum()} recognized, "
                f"{inventory[label].isna().sum()} missing, {unknown.sum()} unrecognized."
            )
            if unknown.any():
                st.warning(
                    f"Unrecognized {label} values need review: "
                    + ", ".join(sorted(set(inventory.loc[unknown, label].astype(str))))
                )
        st.dataframe(inventory, hide_index=True, width="stretch")
        st.caption(
            "Save explicitly to retain this inventory and its reviews after closing the browser."
        )
        if st.button("Save and open review", type="primary"):
            review_id = save_review(uploaded.getvalue(), uploaded.name, config)
            open_review(review_id)
            st.success("Saved locally. Suggestions are ready for human review.")
            st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")
