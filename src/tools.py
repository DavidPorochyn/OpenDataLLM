from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import duckdb
import pandas as pd

from .vector_store import get_chroma_client, search_datasets


def _select_best_resource(resources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Choose the best resource from a CKAN package for analysis.

    Preference order: CSV > GEOJSON > WFS > JSON.
    """
    if not resources:
        return None

    priority = {"CSV": 0, "GEOJSON": 1, "WFS": 2, "JSON": 3}

    def score(res: Dict[str, Any]) -> Tuple[int, str]:
        fmt = (res.get("format") or "").upper()
        return priority.get(fmt, 99), res.get("name") or res.get("url") or ""

    return sorted(resources, key=score)[0]


def find_dataset(query: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Search ChromaDB for the most relevant dataset for the given natural language query.

    Returns a structure containing:
    - match: The top-matching dataset metadata (or None)
    - alternatives: Other candidate datasets
    """
    client = get_chroma_client()
    hits = search_datasets(client, query, n_results=n_results)

    if not hits:
        return {"match": None, "alternatives": []}

    primary = hits[0]
    alternatives = hits[1:]

    # `resources` is stored in Chroma metadata as a JSON string; deserialize it.
    raw_resources = primary.get("resources") or "[]"
    if isinstance(raw_resources, str):
        try:
            primary_res = json.loads(raw_resources)
        except json.JSONDecodeError:
            primary_res = []
    else:
        primary_res = raw_resources or []
    best_res = _select_best_resource(primary_res)

    return {
        "match": {
            "id": primary.get("id"),
            "title": primary.get("title"),
            "description": primary.get("description"),
            "resources": primary_res,
            "selected_resource": best_res,
        },
        "alternatives": alternatives,
    }


def analyze_csv(url: str, user_query: str) -> Dict[str, Any]:
    """
    Load a CSV into a Pandas DataFrame and return a structured summary
    that the LLM can use to answer the user_query.
    """
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        return {
            "kind": "csv",
            "url": url,
            "user_query": user_query,
            "error": "download_failed",
            "error_message": (
                "I found the dataset, but the file link is broken or unreadable."
            ),
            "exception": repr(exc),
        }

    preview = df.head(20)
    info = {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "row_count": len(df),
    }

    return {
        "kind": "csv",
        "url": url,
        "user_query": user_query,
        "info": info,
        "preview_markdown": preview.to_markdown(index=False),
    }


def _init_duckdb_spatial(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure DuckDB spatial extension is installed and loaded.
    """
    duckdb.install_extension("spatial")
    duckdb.load_extension("spatial")
    conn.execute("SET allow_unsigned_extensions=true;")


def query_geospatial(url: str, sql_query: str) -> Dict[str, Any]:
    """
    Execute a geospatial SQL query against a remote file using DuckDB spatial.

    Convention:
    - The caller should refer to the data as the table/view `geo`.
    - This function will create `geo` as `ST_Read('{url}')` before running the query.
    """
    conn = duckdb.connect()
    try:
        _init_duckdb_spatial(conn)
        conn.execute("CREATE OR REPLACE VIEW geo AS SELECT * FROM ST_Read(?);", [url])

        result_df = conn.execute(sql_query).fetch_df()
    except Exception as exc:
        return {
            "kind": "geospatial",
            "url": url,
            "sql_query": sql_query,
            "error": "query_failed",
            "error_message": (
                "I found the dataset, but the file link is broken or the SQL query "
                "could not be executed."
            ),
            "exception": repr(exc),
        }
    finally:
        conn.close()

    preview = result_df.head(200)

    # Heuristic extraction of coordinates if present
    coords = None
    lat_cols = [c for c in preview.columns if c.lower() in ("lat", "latitude", "y")]
    lon_cols = [c for c in preview.columns if c.lower() in ("lon", "longitude", "x")]
    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        coords = preview[[lat_col, lon_col]].rename(
            columns={lat_col: "lat", lon_col: "lon"}
        )

    result: Dict[str, Any] = {
        "kind": "geospatial",
        "url": url,
        "sql_query": sql_query,
        "preview_markdown": preview.to_markdown(index=False),
        "columns": list(preview.columns),
        "row_count": len(preview),
    }

    if coords is not None:
        result["coordinates"] = coords.to_dict(orient="records")

    return result



