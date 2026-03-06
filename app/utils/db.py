from __future__ import annotations
from typing import List, Optional, Tuple

import os
import psycopg2
import streamlit as st


def get_db_params() -> dict:
    """
    Returns DB connection params. Priority:
    1. Environment variables (DB_HOST, DB_NAME, DB_PORT, DB_USER, DB_PASSWORD)
    2. Streamlit secrets (same keys)
    3. vectordb/database.ini (local dev fallback)
    """
    # 1. Environment variables
    if os.environ.get("DB_HOST"):
        return {
            "host":     os.environ["DB_HOST"],
            "dbname":   os.environ.get("DB_NAME", "postgres"),
            "port":     int(os.environ.get("DB_PORT", 5432)),
            "user":     os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
        }

    # 2. Streamlit secrets
    try:
        s = st.secrets
        if "DB_HOST" in s:
            return {
                "host":     s["DB_HOST"],
                "dbname":   s.get("DB_NAME", "postgres"),
                "port":     int(s.get("DB_PORT", 5432)),
                "user":     s["DB_USER"],
                "password": s["DB_PASSWORD"],
            }
        # Also support nested [postgresql] section in secrets.toml
        if "postgresql" in s:
            pg = s["postgresql"]
            return {
                "host":     pg["host"],
                "dbname":   pg.get("database", "postgres"),
                "port":     int(pg.get("port", 5432)),
                "user":     pg["user"],
                "password": pg["password"],
            }
    except Exception:
        pass

    # 3. Local database.ini fallback
    try:
        from vectordb.config import config as read_config
        return read_config(filename="vectordb/database.ini")
    except Exception:
        pass

    raise RuntimeError(
        "No database configuration found. "
        "Set DB_HOST/DB_USER/DB_PASSWORD/DB_NAME/DB_PORT as environment variables "
        "or Streamlit secrets, or provide vectordb/database.ini locally."
    )


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