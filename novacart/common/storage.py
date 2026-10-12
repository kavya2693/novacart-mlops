"""Thin storage layer over SQLite (local) / Postgres (prod).

We deliberately keep this tiny and dependency-free: pandas.read_sql +
sqlite3/psycopg. Swapping SQLite for Postgres is a one-line env change
(NOVACART_DB_URL) — the SQL is standard.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from novacart.common.config import settings


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite:///")


@contextmanager
def connect() -> Iterator["sqlite3.Connection"]:
    """Yield a DB connection. SQLite locally; Postgres via psycopg if configured."""
    url = settings.db_url
    if _is_sqlite(url):
        path = Path(url.replace("sqlite:///", "", 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    else:  # pragma: no cover - exercised only in docker/postgres
        import psycopg  # imported lazily so local runs need no driver

        conn = psycopg.connect(url)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def write_df(name: str, df: pd.DataFrame) -> None:
    """Replace a whole table with a DataFrame (batch load, not row-by-row)."""
    with connect() as conn:
        df.to_sql(name, conn, if_exists="replace", index=False)


def read_sql(query: str, params: tuple | None = None) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql(query, conn, params=params)


def table_exists(name: str) -> bool:
    if _is_sqlite(settings.db_url):
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        return not read_sql(q, (name,)).empty
    q = "SELECT to_regclass(%s) AS t"  # pragma: no cover
    return read_sql(q, (name,)).iloc[0, 0] is not None  # pragma: no cover
