# -*- coding: utf-8 -*-
"""DuckDB helpers for forecast_observations-style tables."""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np


def _sql_ident(name: str) -> str:
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    if not ident.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def preview_forecast_duckdb(
    db_path: str,
    table: str = "forecast_observations",
    order_column: str = "observation_index",
    limit: int = 12,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Peek at rows like the web ``/data`` page: column names, total count, and a page of records.

    Requires: pip install duckdb pandas
    """
    import duckdb

    table = _sql_ident(table)
    order_column = _sql_ident(order_column)
    limit = int(limit)
    offset = int(offset)
    if limit < 1 or offset < 0:
        raise ValueError("limit must be >= 1 and offset >= 0")

    con = duckdb.connect(db_path, read_only=True)
    try:
        total = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        cur = con.execute(
            f"SELECT * FROM {table} ORDER BY {order_column} LIMIT ? OFFSET ?",
            [limit, offset],
        )
        col_names = [d[0] for d in cur.description or []]
        raw_rows = cur.fetchall()
        rows = [dict(zip(col_names, row)) for row in raw_rows]
        return {
            "db_path": db_path,
            "table": table,
            "total_rows": total,
            "limit": limit,
            "offset": offset,
            "columns": col_names,
            "rows": rows,
        }
    finally:
        con.close()


def load_series_from_duckdb_file(
    db_path: str,
    table: str = "forecast_observations",
    order_column: str = "observation_index",
    value_column: str = "normalized_close",
    max_rows: int = 0,
):
    """
    Same idea as horizon/db.py — use if you upload ``normalized_forecast_series.duckdb``.
    Requires: pip install duckdb
    """
    import duckdb

    table = _sql_ident(table)
    order_column = _sql_ident(order_column)
    value_column = _sql_ident(value_column)
    con = duckdb.connect(db_path, read_only=True)
    try:
        if max_rows > 0:
            q = f"""
            SELECT {value_column} AS v FROM {table}
            ORDER BY {order_column} DESC
            LIMIT ?
            """
            df = con.execute(q, [max_rows]).df()
            s = df["v"].to_numpy(dtype=np.float32)[::-1].copy()
        else:
            q = f"SELECT {value_column} AS v FROM {table} ORDER BY {order_column}"
            s = con.execute(q).df()["v"].to_numpy(dtype=np.float32)
        return s
    finally:
        con.close()


def load_forecast_series_and_bounds(
    db_path: str | PathLike[str],
    table: str = "forecast_observations",
    order_column: str = "observation_index",
    norm_column: str = "normalized_close",
    price_column: str = "price_inr",
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Load ordered ``normalized_close`` and matching min/max for norm and ``price_inr`` for denorm.

    Requires: pip install duckdb pandas
    """
    import duckdb

    path = Path(db_path)
    table = _sql_ident(table)
    order_column = _sql_ident(order_column)
    norm_column = _sql_ident(norm_column)
    price_column = _sql_ident(price_column)

    con = duckdb.connect(str(path), read_only=True)
    try:
        q = f"""
        SELECT {norm_column} AS n, {price_column} AS p
        FROM {table}
        ORDER BY {order_column}
        """
        df = con.execute(q).df()
    finally:
        con.close()

    if len(df) == 0:
        raise RuntimeError(f"No rows in {table}.")

    norm = df["n"].to_numpy(dtype=np.float32)
    price = df["p"].to_numpy(dtype=np.float64)
    n_lo, n_hi = float(np.min(norm)), float(np.max(norm))
    p_lo, p_hi = float(np.min(price)), float(np.max(price))
    return norm, n_lo, n_hi, p_lo, p_hi
