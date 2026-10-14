"""Review-first Streamlit application. Run with uv run streamlit run app.py."""

import streamlit as st

st.set_page_config(page_title="obs-agent — Label review", page_icon="🏷️", layout="wide")
st.sidebar.title("🏷️ obs-agent")
st.sidebar.caption("Infer labels · review evidence · export a plan")
page = st.navigation(
    [
        st.Page("pages/1_Label_review.py", title="Label review", icon="🏷️", default=True),
        st.Page("pages/2_Import_inventory.py", title="Import / reopen inventory", icon="📥"),
        st.Page("pages/3_Cost_chat.py", title="Optional cost chat", icon="💬"),
    ]
)
page.run()
