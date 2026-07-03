import numpy as np
from dotenv import load_dotenv
import os
from unified_engine import UnifiedEngine, run_prediction
from live_api_connector import LiveAPIConnector
import sqlite3
import pandas as pd

import subprocess
import time

# 1. Cargar entorno
load_dotenv()
api_key = os.getenv("API_KEY", "NOT_FOUND")
print(f"[+] Iniciando Sistema de Producción. API Key cargada: {'SI' if api_key != 'NOT_FOUND' else 'NO'}")

# Levantar Servidor Mock en background
print("[+] Levantando Mock API Server en localhost:5000...")
mock_server_process = subprocess.Popen(["python", "mock_api_server.py"])
time.sleep(2) # Esperar a que el servidor encienda

# 2. Configurar el Motor para un partido específico (ej. Francia vs Senegal)
FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]])
}

hist_a = [{"gf": 2.5, "gc": 0.8, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}]
hist_b = [{"gf": 1.2, "gc": 1.2, "res": "D"}, {"gf": 1.0, "gc": 1.5, "res": "L"}]

engine = SupremePredictionEngine("Francia", "Senegal", hist_a, hist_b, FORMACIONES)

# 3. Iniciar el Orquestador HTTP
connector = LiveAPIConnector(engine, match_id="FRA_SEN_WC26")

try:
    # 4. Ejecutar el Bucle (Limitamos a 15 iteraciones para la demo)
    connector.run_live_loop(iterations=15, interval=1)
finally:
    # Apagar el servidor mock al terminar
    mock_server_process.terminate()

# 5. Reporte Final de la Base de Datos SQLite
print("\n[+] Extrayendo log de transacciones de production_log.db...")
with sqlite3.connect("production_log.db") as conn:
    df = pd.read_sql_query("SELECT * FROM live_signals ORDER BY id DESC LIMIT 5", conn)
    print(df.to_string(index=False))
