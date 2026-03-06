from __future__ import annotations
from typing import List, Optional, Tuple

import psycopg2
import streamlit as st


def get_db_params():
    # Uses your existing config loader
    from vectordb.config import config as read_config
    return read_config(filename="vectordb/database.ini")


def run_query(sql: str, params: Tuple = None) -> Tuple[Optional[List[Tuple]], Optional[List[str]]]:
    """
    Fresh connection per query to avoid locks.
    Adds timeouts + (optional) IVF probes.
    """
    try:
        conn = psycopg2.connect(**get_db_params())
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '2s';")
            cur.execute("SET statement_timeout = '30s';")
            # Improves recall if you used IVFFLAT; harmless otherwise
            try:
                cur.execute("SET ivfflat.probes = 10;")
            except Exception:
                pass

            cur.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
        conn.close()
        return rows, cols
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None, None