"""Streamlit interface for Minimi, a web-search research assistant."""

import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError

from agent import run_agent

load_dotenv()

st.set_page_config(
    page_title="Minimi | Web Search Assistant",
    page_icon="🔎",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(ellipse at 85% 0%, #eef0ff 0, transparent 38%),
                    #f7f8fc;
    }
    .block-container { max-width: 900px; padding-top: 2.5rem; }
    [data-testid="stChatMessage"] {
        border: 1px solid #e6e9f2;
        border-radius: 16px;
        background: #ffffff;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #dfe3ed;
        border-radius: 14px;
        background: #ffffff;
    }
    .eyebrow {
        color: #5263df;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_timestamp(value: str) -> str:
    timestamp = datetime.fromisoformat(value)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**[{source['id']}] {source['title']}**")
            st.link_button("Open website", source["url"])
            st.caption(source["content"])


st.markdown('<div class="eyebrow">AI WEB SEARCH ASSISTANT</div>', unsafe_allow_html=True)
st.title("Minimi")
st.write(
    "Ask a question, explore the web, and get a concise answer with links to its sources."
)

with st.sidebar:
    st.subheader("Configuration")
    openai_api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Used for chat and follow-up understanding.",
        key="openai_api_key",
    )
    tavily_api_key = st.text_input(
        "Tavily API key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Used only for the agent's web search tool.",
        key="tavily_api_key",
    )
    model = st.text_input(
        "OpenAI model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="A smaller model is recommended for faster answers.",
        key="openai_model",
    )
    st.caption(
        "The agent has one external tool: web search. "
        "Credentials are not written to the project files."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(format_timestamp(message["timestamp"]))
        render_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about anything..."):
    question_time = datetime.now().astimezone().isoformat(timespec="seconds")
    user_message = {
        "role": "user",
        "content": question,
        "timestamp": question_time,
    }
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)
        st.caption(format_timestamp(question_time))

    answer_placeholder = st.empty()
    streamed_answer: list[str] = []

    def show_token(token: str) -> None:
        streamed_answer.append(token)
        answer_placeholder.markdown("".join(streamed_answer))

    def reset_stream() -> None:
        streamed_answer.clear()
        answer_placeholder.empty()

    with st.chat_message("assistant"):
        with st.status("Searching the web and preparing an answer...", expanded=False) as status:
            try:
                answer, sources = run_agent(
                    history=st.session_state.messages,
                    openai_api_key=openai_api_key,
                    tavily_api_key=tavily_api_key,
                    model=model,
                    on_token=show_token,
                    on_reset=reset_stream,
                )
                answer_placeholder.markdown(answer)
                status.update(label="Answer ready", state="complete")
                answer_time = datetime.now().astimezone().isoformat(timespec="seconds")
                st.caption(format_timestamp(answer_time))
                render_sources(sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": answer_time,
                        "sources": sources,
                    }
                )
            except (OpenAIError, requests.RequestException, ValueError, RuntimeError) as error:
                answer_placeholder.empty()
                status.update(label="Request failed", state="error")
                st.error(f"Could not complete the request: {error}")
