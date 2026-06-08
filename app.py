"""Streamlit web UI for obs-agent.

Usage
-----
    streamlit run app.py

Then open the URL shown in the terminal (typically http://localhost:8501).

You can pass the provider and model as query parameters:
    http://localhost:8501/?provider=deepseek&model=deepseek-chat

Environment variables:
    ANTHROPIC_API_KEY  — required for Anthropic (default provider)
    DEEPSEEK_API_KEY   — required for DeepSeek
"""

import os
import streamlit as st
from agent import run

# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="obs-agent — Multi-Cloud Cost Triage",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────

st.sidebar.title("☁️ obs-agent")
st.sidebar.caption("Multi-Cloud Cost Triage Agent")

provider = st.sidebar.selectbox(
    "Provider",
    options=["anthropic", "deepseek", "ollama"],
    index=0,
    help="LLM provider to use for answering questions. Ollama runs locally (no API key).",
)

model = st.sidebar.text_input(
    "Model (optional)",
    placeholder="e.g. claude-sonnet-4-6, deepseek-chat, or llama3.2",
    help="Leave blank for provider default.",
)

st.sidebar.divider()
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown("- Which Azure subscription has the highest overage cost?")
st.sidebar.markdown("- Show me the top 3 GCP projects")
st.sidebar.markdown("- Compare costs across Azure and GCP")
st.sidebar.markdown("- Any cost spikes above 200%?")
st.sidebar.markdown("- What's the daily trend for sub-a1b2c3d4?")

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About"):
    st.markdown(
        "Answers questions about cloud logging costs across "
        "**Azure** and **GCP** by calling tools over synthetic "
        "billing data. Built with Claude/DeepSeek tool-use API."
    )

# ── API key check ──────────────────────────────────────────────────────

if provider == "ollama":
    st.sidebar.info("🟢 Ollama — runs locally, no API key needed")
elif provider == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
    st.sidebar.error("❌ `DEEPSEEK_API_KEY` not set in environment.")
elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    st.sidebar.error("❌ `ANTHROPIC_API_KEY` not set in environment.")
else:
    st.sidebar.success(f"✅ {provider.title()} API key detected")

# ── Main chat ──────────────────────────────────────────────────────────

st.title("☁️ Multi-Cloud Cost Triage")
st.caption(
    f"Provider: **{provider}**"
    + (f"  ·  Model: **{model}**" if model else "")
)

# Initialise conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about cloud costs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response (with conversation memory)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                answer, history = run(
                    prompt,
                    provider=provider,
                    model=model if model else None,
                    verbose=True,
                    messages=st.session_state.get("_agent_history"),
                )
                st.session_state["_agent_history"] = history
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
