"""Read-only CSV validation for a sanitized cloud inventory."""

import streamlit as st

from inventory_import import REQUIRED_COLUMNS, InventoryImportError, load_inventory_csv
from label_review import demo_enabled

st.set_page_config(page_title="Import inventory — obs-agent", page_icon="📥", layout="wide")
st.title("📥 Import a sanitized inventory")
st.caption(
    "Local validation only: this page never connects to Azure or GCP and never changes tags."
)

if demo_enabled():
    st.warning(
        "Inventory import is disabled in the public demo. Run obs-agent locally for private data."
    )
    st.stop()

with st.expander("Required CSV columns"):
    st.code(", ".join(sorted(REQUIRED_COLUMNS)), language=None)
    st.caption("Use JSON string lists or `|`-delimited text for `owners` and `top_services`.")

uploaded = st.file_uploader("Sanitized Azure/GCP inventory CSV", type="csv")
if uploaded:
    try:
        inventory = load_inventory_csv(uploaded.getvalue())
    except InventoryImportError as exc:
        st.error(str(exc))
    else:
        st.session_state["imported_inventory"] = inventory
        st.session_state.pop("import_id", None)
        st.success(f"Validated {len(inventory)} resource(s); stored only for this browser session.")
        st.dataframe(inventory, hide_index=True, width="stretch")
        st.caption(
            "The file has enough labelled support for inference. Review suggestions before "
            "exporting a change plan."
        )
        st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")
