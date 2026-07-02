import sqlite3
from datetime import datetime

class ProductionLogger:
    def __init__(self, db_path="production_log.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
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
            ''')
            conn.commit()

    def log_signal(self, match_id, engine_prob_a, market_prob_a, alpha, latency_ms, status="OK"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO live_signals 
                (timestamp, match_id, engine_prob_a, market_prob_a, alpha, latency_ms, circuit_breaker_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), match_id, engine_prob_a, market_prob_a, alpha, latency_ms, status))
            conn.commit()
