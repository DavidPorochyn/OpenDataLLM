Project Specification: Munich Open Data Chatbot
1. Project Overview
We are building a Local Agentic RAG Chatbot that allows users to query the Munich Open Data Portal using natural language. The data is heterogeneous (CSV, WFS, GeoJSON, etc.) and often messy.

The system will not hallucinate. It will function as an intelligent router: finding the right dataset, inspecting its format, and dynamically generating code (SQL or Python) to answer the user's specific question.

2. Tech Stack & Constraints
Language: Python 3.10+

UI Framework: Streamlit (for easy rendering of maps and chat).

Orchestration: LangGraph (for stateful, multi-step agent workflows).

Database (Vector): ChromaDB (local) or FAISS.

Database (Analytical): DuckDB (with spatial extension) for handling geospatial WFS/CSV data on the fly.

LLM Integration: langchain-openai (GPT-4o) or langchain-anthropic.

3. Data Source Reference (CKAN API)
The portal runs on CKAN. We will not scrape HTML; we will use the JSON API.

Base URL: https://opendata.muenchen.de/api/3/action/

Endpoint 1 (List IDs): GET /package_list -> Returns list of strings (IDs).

Endpoint 2 (Get Metadata): GET /package_show?id={package_id}

Key JSON Fields to Extract:

result.title (Dataset Name)

result.notes (Description)

result.resources[]: A list of files.

url: The download link.

format: The file type (usually "CSV", "WFS", "KML", "JSON").

name: Name of the specific file.

4. Architecture Description
A. The "Smart Catalog" (Ingestion)
We cannot pre-download 5,000 datasets. We will index metadata only.

Script: ingest_catalog.py

Logic: Fetch all packages -> Filter for usable formats (CSV, WFS, GeoJSON) -> Create a text summary of each dataset -> Embed into ChromaDB.

B. The Agent Workflow (LangGraph)
The agent acts as a finite state machine.

Router Node: Analyzes user input.

Retriever Node: Queries ChromaDB to find the most relevant dataset ID and URL.

Format Handler Node:

If CSV: Download to Pandas DataFrame -> Pass to Python REPL tool for analysis.

If WFS/GeoJSON: Load into DuckDB -> Generate SQL to answer query.

Response Node: Generates the final text + suggests a UI component (Map/Table).

5. Implementation Phases (Cursor Instructions)
Phase 1: Data Ingestion & Indexing
Create src/ingestion.py.

Implement a robust fetcher that hits package_list and then batches package_show.

Critical: Clean the metadata. Convert German text to English summaries if necessary for better embeddings, or use a multilingual embedding model (text-embedding-3-small works well for both).

Store results in a local ChromaDB collection named munich_data_catalog.

Phase 2: The Tooling Layer
Create src/tools.py.

Tool 1: find_dataset(query: str) -> Searches ChromaDB, returns the Dataset Title, Description, and Resource URLs.

Tool 2: analyze_csv(url: str, user_query: str) -> Uses pandas to load the URL directly. Giving the LLM access to df.head(), it should write code to answer the query.

Tool 3: query_geospatial(url: str, sql_query: str) -> Uses duckdb.

Instruction: DuckDB can query remote files: SELECT * FROM ST_Read('https://...').

Phase 3: The LangGraph Agent
Create src/graph.py.

Define the State TypedDict: messages, selected_dataset_url, dataset_format, analysis_result.

Build the graph: Start -> Lookup Dataset -> Choose Tool -> Execute Tool -> Generate Answer -> End.

Phase 4: The Streamlit UI
Create app.py.

Setup a standard chat interface (st.chat_message).

Magic Feature: If the Agent output contains a list of coordinates, intercept it and render st.map. If it contains a markdown table, render st.dataframe.

6. Specific Coding Guidelines
Error Handling: The Open Data portal might have broken links. If a tool fails (404), the Agent must catch the error and tell the user "I found the dataset, but the file link is broken."

WFS Handling: When handling WFS resources, prefer requesting outputFormat=geojson if available, as DuckDB reads GeoJSON natively.

DuckDB Spatial: Ensure duckdb.install_extension('spatial') and duckdb.load_extension('spatial') are called on init.

Directory Structure Goal
text
/
├── .env (OPENAI_API_KEY)
├── SPECIFICATION.md
├── requirements.txt
├── app.py (Streamlit entry point)
└── src/
    ├── __init__.py
    ├── ingestion.py (Crawler)
    ├── vector_store.py (ChromaDB wrapper)
    ├── tools.py (Pandas/DuckDB tools)
    └── agent.py (LangGraph logic)