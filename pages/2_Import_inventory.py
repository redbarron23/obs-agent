"""Validate an inventory and save or reopen a local review workspace."""

import hashlib
from pathlib import Path

import streamlit as st

from inventory_import import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    InventoryImportError,
    apply_column_mapping,
    csv_columns,
    load_inventory_csv,
    suggest_mapping,
)
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
        "Each label needs two recognized values with at least two examples each. "
        "Other column names can be mapped to these fields after upload."
    )

vocabulary = st.file_uploader("Vocabulary JSON (optional; otherwise use demo values)", type="json")
uploaded = st.file_uploader("Sanitized Azure/GCP inventory CSV", type="csv")
UNMAPPED = "— not mapped —"


def map_columns(source: bytes) -> bytes | None:
    """Return the CSV in the import schema, asking for a column mapping when it differs."""
    columns = csv_columns(source)
    if REQUIRED_COLUMNS <= set(columns):
        return source
    st.subheader("Map your columns")
    st.caption(
        "This CSV's headers differ from the import schema. Choose the source column for each "
        "field. Suggestions are guesses: check each one. Unmapped optional fields stay blank; "
        "required fields must come from your data."
    )
    guess = suggest_mapping(columns)
    digest = hashlib.sha256(source).hexdigest()[:12]
    mapping = {}
    grid = st.columns(3)
    fields = sorted(REQUIRED_COLUMNS) + sorted(OPTIONAL_COLUMNS)
    for i, field in enumerate(fields):
        options = [UNMAPPED, *columns]
        choice = grid[i % 3].selectbox(
            field + ("" if field in REQUIRED_COLUMNS else " (optional)"),
            options,
            index=options.index(guess[field]) if field in guess else 0,
            key=f"map-{digest}-{field}",
        )
        mapping[field] = None if choice == UNMAPPED else choice
    ignored = sorted(set(columns) - {c for c in mapping.values() if c})
    if ignored:
        st.caption("Not imported: " + ", ".join(ignored))
    return apply_column_mapping(source, mapping)


if uploaded:
    try:
        config = load_label_config(vocabulary.getvalue()) if vocabulary else None
        normalized = map_columns(uploaded.getvalue())
        inventory = load_inventory_csv(normalized, config)
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
            # Store the mapped CSV so the saved review reopens without the mapping.
            review_id = save_review(normalized, uploaded.name, config)
            open_review(review_id)
            st.success("Saved locally. Suggestions are ready for human review.")
            st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")
