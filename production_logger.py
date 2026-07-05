import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np


class ProductionLogger:
    def __init__(self, db_path: str = "production_log.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS live_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    match_id TEXT,
                    engine_prob_a REAL,
                    market_prob_a REAL,
                    alpha REAL,
                    latency_ms REAL,
                    circuit_breaker_status TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    timestamp TEXT,
                    match_id TEXT,
                    metadata TEXT,
                    metrics TEXT
                )
                """
            )
            conn.commit()

    def log_signal(self, match_id, engine_prob_a, market_prob_a, alpha, latency_ms, status="OK"):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO live_signals
                (timestamp, match_id, engine_prob_a, market_prob_a, alpha, latency_ms, circuit_breaker_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (datetime.utcnow().isoformat(), match_id, engine_prob_a, market_prob_a, alpha, latency_ms, status),
            )
            conn.commit()

    def log_event(self, event_type: str, match_id: Optional[str] = None, metadata: Optional[dict] = None, metrics: Optional[dict[str, float]] = None) -> int:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (event_type, timestamp, match_id, metadata, metrics)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    datetime.utcnow().isoformat(),
                    match_id,
                    json.dumps(metadata or {}),
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_rolling_stats(self, hours: int = 24) -> dict[str, float]:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT metrics FROM events WHERE timestamp >= ? AND metrics IS NOT NULL",
                (cutoff,),
            )
            rows = cursor.fetchall()
        values: dict[str, list[float]] = {}
        for (metrics_json,) in rows:
            try:
                metrics = json.loads(metrics_json)
            except json.JSONDecodeError:
                continue
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    values.setdefault(key, []).append(float(value))
        result: dict[str, float] = {}
        for key, arr in values.items():
            if arr:
                array = np.array(arr, dtype=float)
                result[f"{key}_mean"] = float(np.mean(array))
                result[f"{key}_std"] = float(np.std(array))
                result[f"{key}_min"] = float(np.min(array))
                result[f"{key}_max"] = float(np.max(array))
        return result

    def check_alerts(self, thresholds: Optional[dict[str, tuple[float, float]]] = None) -> list[dict]:
        thresholds = thresholds or {'brier_mean': (0.65, 0.70)}
        stats = self.get_rolling_stats(hours=6)
        alerts = []
        for metric, (warning, critical) in thresholds.items():
            if f"{metric}_mean" in stats:
                value = stats[f"{metric}_mean"]
                if value >= critical:
                    alerts.append({'level': 'CRITICAL', 'metric': metric, 'value': value, 'threshold': critical})
                elif value >= warning:
                    alerts.append({'level': 'WARNING', 'metric': metric, 'value': value, 'threshold': warning})
        return alerts
