"""Tests for src.data.database connection lifecycle."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.data.database import get_connection, get_db_connection


def test_get_db_connection_closes_on_success(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with get_db_connection(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_get_db_connection_closes_on_error(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with pytest.raises(RuntimeError):
        with get_db_connection(db) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            raise RuntimeError("boom")

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_sqlite_connection_context_manager_does_not_close(tmp_path: Path) -> None:
    """Document why callers must not use `with get_connection()` for lifecycle."""
    db = tmp_path / "test.db"
    conn = get_connection(db)
    with conn:
        conn.execute("CREATE TABLE t (x INTEGER)")

    # sqlite3 only commits/rolls back; connection stays open.
    conn.execute("SELECT 1")
    conn.close()
