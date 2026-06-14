"""SQLite database for workspace/account storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .workspace import build_default_workspace, normalize_workspace_payload


class DatabaseManager:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "web" / "trading_data.db"
        self.db_path = str(db_path)
        self._init_tables()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stage2_workspace (
                    workspace_name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def get_stage2_workspace(self, workspace_name: str = "default") -> Optional[dict]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM stage2_workspace WHERE workspace_name = ?",
                (workspace_name,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def save_stage2_workspace(self, workspace: dict, workspace_name: str = "default") -> dict:
        now = datetime.now(timezone.utc).isoformat()
        normalized = normalize_workspace_payload(workspace)
        data_json = json.dumps(normalized, ensure_ascii=False)
        with self.get_connection() as conn:
            existing = conn.execute(
                "SELECT created_at FROM stage2_workspace WHERE workspace_name = ?",
                (workspace_name,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                "INSERT OR REPLACE INTO stage2_workspace (workspace_name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (workspace_name, data_json, created_at, now),
            )
        return {
            "workspace_name": workspace_name,
            "created_at": created_at,
            "updated_at": now,
            **normalized,
        }


_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
