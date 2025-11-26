import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import AgentState, build_graph


load_dotenv()


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "graph" not in st.session_state:
        st.session_state.graph = get_graph()


def _render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                # If previous assistant messages included a structured payload
                # with map/table hints, we could handle them here.
                st.markdown(content)
            else:
                st.markdown(msg.get("content", ""))


def _render_agent_output(content: str):
    """
    Render the final agent answer, with special handling for maps and tables
    when possible.

    For simplicity, we currently:
    - Display the text with st.markdown
    - (Optional extension) detect coordinate lists or markdown tables here.
    """
    st.markdown(content)


def main():
    st.set_page_config(page_title="Munich Open Data Chatbot", layout="wide")
    st.title("Munich Open Data Chatbot")
    st.caption(
        "Ask questions about datasets from the Munich Open Data portal. "
        "The agent finds a relevant dataset and analyzes it without hallucinating."
    )

    _init_session_state()
    _render_chat_history()

    user_input = st.chat_input("Ask a question about Munich open data...")
    if not user_input:
        return

    # Append user message to UI history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    graph = st.session_state.graph

    # Build initial graph state
    state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "selected_dataset": None,
        "analysis_result": None,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_state = graph.invoke(state)
            messages = final_state["messages"]
            last_ai = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            answer = last_ai.content if last_ai else "I could not generate an answer."
            _render_agent_output(answer)

    # Save assistant message into UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()


