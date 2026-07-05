import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from live_api_connector import LiveAPIConnector
from unified_engine_v3 import UnifiedEngineV3


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FUOL live mode runner')
    parser.add_argument('--config', default='config_v3/config_v3.json')
    parser.add_argument('--match-id', default='FRA_SEN_WC26')
    parser.add_argument('--ws-url', default='ws://localhost:5001')
    parser.add_argument('--max-messages', type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv('API_KEY', 'NOT_FOUND')
    print(f"[+] Iniciando Sistema de Producción. API Key cargada: {'SI' if api_key != 'NOT_FOUND' else 'NO'}")

    config = load_config(args.config)
    print(f"[+] Cargando configuración desde {args.config}")

    print("[+] Levantando Mock WebSocket Server en localhost:5001...")
    mock_server_process = subprocess.Popen([sys.executable, 'mock_websocket_server.py'])
    time.sleep(2)

    hist_a = [
        {"date": "2024-01-01", "home": "FRANCIA", "away": "SENEGAL", "gh": 2, "ga": 0, "minute": 10},
        {"date": "2024-02-01", "home": "FRANCIA", "away": "MAROC", "gh": 1, "ga": 1, "minute": 30},
    ]
    hist_b = [
        {"date": "2024-01-15", "home": "SENEGAL", "away": "CAMERUN", "gh": 1, "ga": 1, "minute": 45},
        {"date": "2024-02-15", "home": "SENEGAL", "away": "TOGO", "gh": 2, "ga": 0, "minute": 60},
    ]

    engine = UnifiedEngineV3(
        team_a='FRANCIA',
        team_b='SENEGAL',
        matches_a=hist_a,
        matches_b=hist_b,
        venue='H',
        team_confederations={'FRANCIA': 'UEFA', 'SENEGAL': 'CAF'},
        config=config,
    )

    connector = LiveAPIConnector(engine, ws_url=args.ws_url, match_id=args.match_id, max_messages=args.max_messages)

    try:
        connector.run()
    finally:
        mock_server_process.terminate()

    print("\n[+] Extrayendo log de transacciones de production_log.db...")
    with sqlite3.connect('production_log.db') as conn:
        df = pd.read_sql_query('SELECT * FROM live_signals ORDER BY id DESC LIMIT 5', conn)
        print(df.to_string(index=False))
