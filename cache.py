from __future__ import annotations

import hashlib
import json
import time
from threading import RLock
from typing import Any


class PredictionCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = RLock()

    def _key(self, *, match_id: str, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(f"{match_id}:{normalized}".encode("utf-8")).hexdigest()

    def get(self, *, match_id: str, payload: dict[str, Any]) -> Any | None:
        key = self._key(match_id=match_id, payload=payload)
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            timestamp, value = entry
            if time.time() - timestamp > self.ttl_seconds:
                self._store.pop(key, None)
                return None
            return value

    def set(self, *, match_id: str, payload: dict[str, Any], value: Any) -> None:
        key = self._key(match_id=match_id, payload=payload)
        with self._lock:
            self._store[key] = (time.time(), value)
