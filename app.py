"""Streamlit web UI for obs-agent.

Usage
-----
    streamlit run app.py

Then open the URL shown in the terminal (typically http://localhost:8501).

You can pass the provider and model as query parameters:
    http://localhost:8501/?provider=deepseek&model=deepseek-chat

Environment variables:
    DEEPSEEK_API_KEY   — required for DeepSeek (default provider)
    ANTHROPIC_API_KEY  — required for Anthropic
"""

import os
import streamlit as st
from agent import run, stream_run

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
    index=1,
    help="LLM provider to use for answering questions. DeepSeek is the default; Ollama runs locally (no API key).",
)

model = st.sidebar.text_input(
    "Model (optional)",
    placeholder="e.g. claude-sonnet-4-6, deepseek-chat, or llama3.2",
    help="Leave blank for provider default.",
)

use_streaming = st.sidebar.checkbox(
    "Stream responses",
    value=True,
    help="Show tokens as they arrive instead of waiting for the full answer.",
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
    + ("  ·  Streaming" if use_streaming else "")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about cloud costs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            model_arg = model if model else None
            history = st.session_state.get("_agent_history")

            if use_streaming:
                holder: dict = {}
                st.write_stream(stream_run(
                    prompt,
                    provider=provider,
                    model=model_arg,
                    verbose=True,
                    messages=history,
                    result_holder=holder,
                ))
                answer = holder["answer"]
                st.session_state["_agent_history"] = holder["messages"]
            else:
                with st.spinner("Agent is thinking..."):
                    answer, updated = run(
                        prompt,
                        provider=provider,
                        model=model_arg,
                        verbose=True,
                        messages=history,
                    )
                st.markdown(answer)
                st.session_state["_agent_history"] = updated

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
