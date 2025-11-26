from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .tools import analyze_csv, find_dataset, query_geospatial


class AgentState(TypedDict):
    """
    LangGraph state for the Munich open data agent.

    Fields:
    - messages: chat messages (user + assistant)
    - selected_dataset: dataset metadata chosen from the catalog
    - analysis_result: result from CSV or geospatial tool
    """

    messages: List[BaseMessage]
    selected_dataset: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]


def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def node_lookup_dataset(state: AgentState) -> AgentState:
    """Router/lookup node: choose a dataset from the catalog."""
    user_query = _get_last_user_message(state)
    lookup = find_dataset(user_query)
    match = lookup.get("match")

    new_state = dict(state)
    new_state["selected_dataset"] = match
    return new_state


def node_execute_tool(state: AgentState) -> AgentState:
    """Execute the appropriate tool based on the dataset format."""
    dataset = state.get("selected_dataset")
    if not dataset:
        new_state = dict(state)
        new_state["analysis_result"] = None
        return new_state

    selected_resource = dataset.get("selected_resource") or {}
    fmt = (selected_resource.get("format") or "").upper()
    url = selected_resource.get("url")
    user_query = _get_last_user_message(state)

    if not url:
        analysis_result = {
            "error": "no_resource_url",
            "error_message": "I found a dataset, but it has no usable file URL.",
        }
    elif fmt == "CSV":
        analysis_result = analyze_csv(url, user_query)
    elif fmt in {"GEOJSON", "WFS", "JSON"}:
        # Step 1: get a small preview + schema so the LLM can see columns and sample rows.
        preview_sql = "SELECT * FROM geo LIMIT 200"
        preview_result = query_geospatial(url, preview_sql)

        # If loading the dataset failed, surface the error as-is.
        if preview_result.get("error"):
            analysis_result = preview_result
        else:
            # Step 2: ask the LLM to generate a focused DuckDB SQL query over `geo`.
            #
            # We give it:
            # - the user question
            # - the column names
            # - a small markdown preview
            # and instruct it to return ONLY a valid SQL query.
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            system_msg = AIMessage(
                content=(
                    "You are a data analyst writing DuckDB SQL over a single table "
                    "named `geo` that comes from a geospatial dataset.\n"
                    "- Use the available columns only.\n"
                    "- When the question involves locations or distances, you may use "
                    "DuckDB spatial functions such as ST_Distance, ST_Within, "
                    "and ST_Point where appropriate.\n"
                    "- If an aggregate (COUNT, AVG, etc.) answers the question well, "
                    "prefer that over returning many raw rows.\n"
                    "- Always return a single SQL query that DuckDB can execute, and "
                    "nothing else (no explanations or comments)."
                )
            )

            columns = preview_result.get("columns", [])
            preview_md = preview_result.get("preview_markdown", "")
            user_msg = HumanMessage(
                content=(
                    f"User question:\n{user_query}\n\n"
                    f"Available columns in table `geo`:\n{columns}\n\n"
                    f"Sample rows (markdown table):\n{preview_md}\n\n"
                    "Write a DuckDB SQL query over the table `geo` that best answers "
                    "the user question, using only the available columns. "
                    "Return ONLY the SQL query."
                )
            )

            sql_resp = llm.invoke([system_msg, user_msg])
            sql_query = (sql_resp.content or "").strip()

            # Very lightweight safety: fall back to a generic preview if the
            # model did not actually return a SELECT over `geo`.
            if "select" not in sql_query.lower() or " from geo" not in sql_query.lower():
                sql_query = "SELECT * FROM geo LIMIT 500"

            analysis_result = query_geospatial(url, sql_query)
    else:
        analysis_result = {
            "error": "unsupported_format",
            "error_message": (
                f"I found the dataset, but the format '{fmt}' is not supported "
                "by this agent yet."
            ),
        }

    new_state = dict(state)
    new_state["analysis_result"] = analysis_result
    return new_state


def _build_system_prompt() -> str:
    return (
        "You are a data analyst working with the City of Munich Open Data Portal.\n"
        "You MUST NOT hallucinate datasets or values. If the data you see does not\n"
        "contain enough information to answer the question, say so explicitly and\n"
        "suggest what data would be needed.\n\n"
        "You receive:\n"
        "- The user question.\n"
        "- The chosen dataset metadata (title, description, resources).\n"
        "- A preview of the relevant file (CSV or geospatial) as a markdown table.\n\n"
        "Instructions:\n"
        "- Use ONLY the data you see in the preview and metadata.\n"
        "- If the analysis_result has an 'error', clearly explain the issue.\n"
        "- When the preview includes latitude/longitude coordinates, you may refer to\n"
        "  the existence of points on a map, but do not fabricate extra points.\n"
        "- When suitable, present tabular results as a small markdown table.\n"
    )


def node_generate_answer(state: AgentState) -> AgentState:
    """Final response node: use the LLM to generate an answer for the user."""
    user_query = _get_last_user_message(state)
    dataset = state.get("selected_dataset")
    analysis = state.get("analysis_result")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    system_prompt = _build_system_prompt()

    dataset_text = "No dataset was found."
    if dataset:
        dataset_text = (
            f"Dataset title: {dataset.get('title')}\n"
            f"Description: {dataset.get('description')}\n"
        )
        res = dataset.get("selected_resource") or {}
        if res:
            dataset_text += (
                f"Selected resource name: {res.get('name')}\n"
                f"Format: {res.get('format')}\n"
                f"URL: {res.get('url')}\n"
            )

    analysis_text = "No analysis result."
    if analysis:
        if analysis.get("error"):
            analysis_text = (
                f"Tool reported an error: {analysis.get('error')}\n"
                f"Error message: {analysis.get('error_message')}\n"
                f"Exception (if any): {analysis.get('exception')}\n"
            )
        else:
            if analysis.get("kind") == "csv":
                analysis_text = (
                    "CSV analysis preview (markdown table):\n"
                    f"{analysis.get('preview_markdown')}\n"
                )
            elif analysis.get("kind") == "geospatial":
                analysis_text = (
                    "Geospatial query preview (markdown table):\n"
                    f"{analysis.get('preview_markdown')}\n"
                )

    messages: List[BaseMessage] = [
        AIMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"User question:\n{user_query}\n\n"
                f"Chosen dataset metadata:\n{dataset_text}\n\n"
                f"Analysis result:\n{analysis_text}\n\n"
                "Now answer the user's question as well as possible using ONLY this "
                "information. Do not invent columns or values that are not visible "
                "here. If the file link is broken or no data is available, explain "
                "that clearly to the user."
            )
        ),
    ]

    llm_resp = llm.invoke(messages)

    new_state = dict(state)
    # Append the final assistant message to the conversation history
    original_messages: List[BaseMessage] = list(state["messages"])
    original_messages.append(AIMessage(content=llm_resp.content))
    new_state["messages"] = original_messages
    return new_state


def build_graph():
    """
    Construct and return the compiled LangGraph workflow.

    Flow:
    Start -> lookup_dataset -> execute_tool -> generate_answer -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("lookup_dataset", node_lookup_dataset)
    graph.add_node("execute_tool", node_execute_tool)
    graph.add_node("generate_answer", node_generate_answer)

    graph.set_entry_point("lookup_dataset")
    graph.add_edge("lookup_dataset", "execute_tool")
    graph.add_edge("execute_tool", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()



