"""Local SQLite map of employee email -> ticket keys.

Written on every ticket creation. Used as a fallback lookup when the platform's search
does not return a ticket (e.g. label search disabled or index lag right after creation).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class EmailIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ticket_owners ("
                " email TEXT NOT NULL, ticket_key TEXT NOT NULL, created_at TEXT NOT NULL,"
                " PRIMARY KEY (email, ticket_key))"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # `with sqlite3.connect()` commits but never closes; an open handle locks the file on Windows.
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, email: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ticket_owners VALUES (?, ?, ?)",
                (email.lower(), key, datetime.now(timezone.utc).isoformat()),
            )

    def remove(self, email: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ticket_owners WHERE email = ? AND ticket_key = ?", (email.lower(), key))

    def keys_for(self, email: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticket_key FROM ticket_owners WHERE email = ? ORDER BY created_at DESC",
                (email.lower(),),
            ).fetchall()
        return [r[0] for r in rows]
