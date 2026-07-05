from __future__ import annotations

import time
from threading import RLock


class RateLimiter:
    def __init__(self, limit_per_minute: int = 60) -> None:
        self.limit_per_minute = limit_per_minute
        self._window_seconds = 60.0
        self._requests: list[float] = []
        self._lock = RLock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            self._requests = [t for t in self._requests if now - t < self._window_seconds]
            if len(self._requests) >= self.limit_per_minute:
                return False
            self._requests.append(now)
            return True
