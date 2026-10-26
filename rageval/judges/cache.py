"""SQLite cache for judge responses, so re-running an evaluation doesn't pay twice."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class JudgeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(provider: str, model: str, system: str, user: str, schema: dict) -> str:
        payload = json.dumps([provider, model, system, user, schema], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM judge_cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, response: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, response) VALUES (?, ?)",
                (key, json.dumps(response)),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
