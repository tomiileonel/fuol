import sqlite3
import numpy as np
import datetime
from pathlib import Path

class ModelTelemetry:
    def __init__(self, db_name="supreme_predictions.db"):
        self.db_path = Path(db_name)
        self._initialize_db()

    def _initialize_db(self):
        """Crea el esquema relacional si no existe."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Añadimos columnas de odds_implied
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    team_a TEXT,
                    team_b TEXT,
                    prob_1 REAL,
                    prob_x REAL,
                    prob_2 REAL,
                    lambda_adj REAL,
                    mu_adj REAL,
                    implied_prob_1 REAL DEFAULT NULL,
                    implied_prob_x REAL DEFAULT NULL,
                    implied_prob_2 REAL DEFAULT NULL,
                    actual_res TEXT DEFAULT NULL, 
                    actual_gf INTEGER DEFAULT NULL,
                    actual_gc INTEGER DEFAULT NULL
                )
            ''')
            # Intentar añadir columnas si la tabla ya existía de antes (sin romper)
            try:
                cursor.execute('ALTER TABLE predictions ADD COLUMN implied_prob_1 REAL DEFAULT NULL')
                cursor.execute('ALTER TABLE predictions ADD COLUMN implied_prob_x REAL DEFAULT NULL')
                cursor.execute('ALTER TABLE predictions ADD COLUMN implied_prob_2 REAL DEFAULT NULL')
            except sqlite3.OperationalError:
                pass # Columnas ya existen
            conn.commit()

    def log_prediction(self, team_a, team_b, engine_output):
        """Registra la salida vectorial del SupremePredictionEngine."""
        p1, px, p2 = engine_output["1X2"]
        lam, mu = engine_output["Tasas Finales (Interferencia)"]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, team_a, team_b, prob_1, prob_x, prob_2, lambda_adj, mu_adj)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.datetime.now().isoformat(), team_a, team_b, p1, px, p2, lam, mu))
            conn.commit()
            print(f"[Telemetry] Predicción guardada: {team_a} vs {team_b}")

    def log_actual_result(self, team_a, team_b, goals_a, goals_b):
        """Actualiza la base de datos con la realidad empírica tras el partido."""
        if goals_a > goals_b: res = '1'
        elif goals_a == goals_b: res = 'X'
        else: res = '2'

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Actualiza el último registro que coincida con los equipos
            cursor.execute('''
                UPDATE predictions 
                SET actual_res = ?, actual_gf = ?, actual_gc = ?
                WHERE match_id = (
                    SELECT match_id FROM predictions 
                    WHERE team_a = ? AND team_b = ? AND actual_res IS NULL
                    ORDER BY match_id DESC LIMIT 1
                )
            ''', (res, goals_a, goals_b, team_a, team_b))
            conn.commit()
            print(f"[Telemetry] Resultado real inyectado: {team_a} {goals_a}-{goals_b} {team_b}")

    def normalize_market_odds(self, odd_1, odd_x, odd_2):
        """Limpia el Overround (Vig) de las casas de apuestas para extraer probabilidad pura."""
        sum_inv = (1.0/odd_1) + (1.0/odd_x) + (1.0/odd_2)
        p1 = (1.0/odd_1) / sum_inv
        px = (1.0/odd_x) / sum_inv
        p2 = (1.0/odd_2) / sum_inv
        return p1, px, p2

    def log_market_odds(self, team_a, team_b, odd_1, odd_x, odd_2):
        """Extrae la cuota limpia y la acopla al último registro de ese partido."""
        p1, px, p2 = self.normalize_market_odds(odd_1, odd_x, odd_2)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions 
                SET implied_prob_1 = ?, implied_prob_x = ?, implied_prob_2 = ?
                WHERE match_id = (
                    SELECT match_id FROM predictions 
                    WHERE team_a = ? AND team_b = ? AND actual_res IS NULL
                    ORDER BY match_id DESC LIMIT 1
                )
            ''', (p1, px, p2, team_a, team_b))
            conn.commit()
            print(f"[Telemetry] Market Odds (Cleaned) inyectadas para {team_a} vs {team_b}")

    def calculate_metrics(self):
        """Extrae el Log-Loss, Brier Score, Alpha y Volatilidad (Outliers)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT prob_1, prob_x, prob_2, actual_res, 
                       implied_prob_1, implied_prob_x, implied_prob_2,
                       team_a, team_b
                FROM predictions 
                WHERE actual_res IS NOT NULL
            ''')
            completed_matches = cursor.fetchall()

        if not completed_matches:
            return "Insuficientes datos empíricos para calcular métricas."

        N = len(completed_matches)
        log_loss_sum = 0
        brier_sum = 0
        market_brier_sum = 0
        hits = 0
        
        engine_briers = []
        outlier_data = []

        for p1, px, p2, actual, mp1, mpx, mp2, team_a, team_b in completed_matches:
            # Vector de probabilidades predichas
            preds = np.array([p1, px, p2])
            
            # Vector del resultado real (One-Hot Encoding)
            y_true = np.zeros(3)
            if actual == '1': y_true[0] = 1
            elif actual == 'X': y_true[1] = 1
            elif actual == '2': y_true[2] = 1

            # Log-Loss: - sum(y * log(p))
            preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
            log_loss_sum -= np.sum(y_true * np.log(preds_clipped))

            # Brier Score (Modelo)
            match_brier = np.sum((preds - y_true)**2)
            brier_sum += match_brier
            engine_briers.append(match_brier)
            outlier_data.append((match_brier, team_a, team_b, actual))
            
            # Brier Score (Mercado)
            if mp1 is not None and mpx is not None and mp2 is not None:
                market_preds = np.array([mp1, mpx, mp2])
                market_brier_sum += np.sum((market_preds - y_true)**2)
            else:
                market_brier_sum += match_brier # Fallback

            # Hit Rate (Precisión absoluta)
            if np.argmax(preds) == np.argmax(y_true):
                hits += 1

        outlier_data.sort(key=lambda x: x[0], reverse=True)
        worst_outlier = outlier_data[0] if outlier_data else None

        # Correlación Alpha-Volatilidad: Calcular Alpha en partidos de alta dificultad
        median_brier = np.median(engine_briers)
        high_vol_brier_sum = 0
        high_vol_market_sum = 0
        high_vol_count = 0
        
        # Iteramos nuevamente para calcular el Alpha en la mitad "difícil" de los partidos
        for (p1, px, p2, actual, mp1, mpx, mp2, team_a, team_b) in completed_matches:
            preds = np.array([p1, px, p2])
            y_true = np.zeros(3)
            if actual == '1': y_true[0] = 1
            elif actual == 'X': y_true[1] = 1
            elif actual == '2': y_true[2] = 1
            match_brier = np.sum((preds - y_true)**2)
            
            if match_brier >= median_brier:
                high_vol_brier_sum += match_brier
                if mp1 is not None:
                    market_preds = np.array([mp1, mpx, mp2])
                    high_vol_market_sum += np.sum((market_preds - y_true)**2)
                else:
                    high_vol_market_sum += match_brier
                high_vol_count += 1
                
        high_vol_alpha = (high_vol_market_sum / high_vol_count) - (high_vol_brier_sum / high_vol_count) if high_vol_count > 0 else 0

        # Trigger de "Shutdown": Últimos 3 partidos con Brier > 0.40
        shutdown_trigger = False
        if len(engine_briers) >= 3:
            last_3 = engine_briers[-3:] # chronologically ordered in SQLite
            if all(b > 0.40 for b in last_3):
                shutdown_trigger = True

        metrics = {
            "Partidos Evaluados": N,
            "Log-Loss": log_loss_sum / N,
            "Brier Score": brier_sum / N,
            "Market Brier": market_brier_sum / N,
            "Alpha (Brier Diff)": (market_brier_sum / N) - (brier_sum / N),
            "Hit Rate (%)": (hits / N) * 100,
            "Brier Volatilidad": np.std(engine_briers),
            "Peor Outlier": worst_outlier,
            "Alpha Alta Volatilidad": high_vol_alpha,
            "Shutdown Alert": shutdown_trigger
        }
        
        self._print_audit_report(metrics)
        return metrics

    def _print_audit_report(self, metrics):
        print(f"\n{'='*55}")
        print(f" AUDITORÍA DE RENDIMIENTO (CÁLCULO DE ALPHA)")
        print(f"{'='*55}")
        print(f" Muestra Empírica : {metrics['Partidos Evaluados']} partidos")
        print(f" Hit Rate         : {metrics['Hit Rate (%)']:.1f}%")
        print(f" Log-Loss         : {metrics['Log-Loss']:.4f}")
        print(f" Brier Score (Engine) : {metrics['Brier Score']:.4f}")
        print(f" Brier Score (Mercado): {metrics['Market Brier']:.4f}")
        print(f" Brier Volatilidad    : ±{metrics['Brier Volatilidad']:.4f} (Riesgo/Varianza)")
        
        outlier = metrics['Peor Outlier']
        if outlier:
            print(f" Peor Outlier     : {outlier[1]} vs {outlier[2]} (Error Brier: {outlier[0]:.4f} | Realidad: {outlier[3]})")
            
        print(f"-------------------------------------------------------")
        alpha = metrics['Alpha (Brier Diff)']
        if alpha > 0:
            print(f" ALPHA GLOBAL     : +{alpha:.4f} (Ganamos al Mercado)")
        else:
            print(f" ALPHA GLOBAL     : {alpha:.4f} (Mercado es superior)")
            
        alpha_vol = metrics['Alpha Alta Volatilidad']
        if alpha_vol > 0:
            print(f" ALPHA ALTA VOL.  : +{alpha_vol:.4f} (Edge estable en caos)")
        else:
            print(f" ALPHA ALTA VOL.  : {alpha_vol:.4f} (Pérdida de ventaja en caos)")

        if metrics['Shutdown Alert']:
            print(f"\n [!!!] ALERTA DE RIESGO: Drawdown Crítico Detectado")
            print(f" [!!!] SHUTDOWN TRIGGER ACTIVADO: Suspenda Operaciones")

        print(f"{'='*55}\n")

    def synchronize_knowledge_base(self, team_name, base_matches_list):
        """
        Lee la realidad empírica almacenada en SQLite y la inyecta dinámicamente 
        en el historial del equipo. Cierra el ciclo de actualización Bayesiana 
        y recalibra el autovector de PageRank.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Extraer los partidos empíricos donde el equipo jugó como A (Local/Izquierda)
            cursor.execute('''
                SELECT timestamp, team_b, actual_gf, actual_gc, actual_res
                FROM predictions
                WHERE team_a = ? AND actual_res IS NOT NULL
            ''', (team_name,))
            home_matches = cursor.fetchall()
            
            # 2. Extraer los partidos empíricos donde el equipo jugó como B (Visitante/Derecha)
            cursor.execute('''
                SELECT timestamp, team_a, actual_gc, actual_gf, actual_res
                FROM predictions
                WHERE team_b = ? AND actual_res IS NOT NULL
            ''', (team_name,))
            away_matches = cursor.fetchall()

        # Clonar la base histórica estática para no mutar el estado global accidentalmente
        dynamic_matches = base_matches_list.copy()
        
        # 3. Formatear e inyectar partidos como Local
        for timestamp, opp, gf, gc, res in home_matches:
            parsed_res = "W" if res == '1' else ("D" if res == 'X' else "L")
            dynamic_matches.append({
                "date": timestamp.split("T")[0], 
                "opponent": opp, 
                "venue": "N", # Asumimos neutralidad en el Mundial
                "comp": "WC 2026 Telemetry", 
                "gf": gf, 
                "gc": gc, 
                "res": parsed_res
            })
            
        # 4. Formatear e inyectar partidos como Visitante (invirtiendo la lógica del resultado)
        for timestamp, opp, gf, gc, res in away_matches:
            parsed_res = "W" if res == '2' else ("D" if res == 'X' else "L")
            dynamic_matches.append({
                "date": timestamp.split("T")[0], 
                "opponent": opp, 
                "venue": "N", 
                "comp": "WC 2026 Telemetry", 
                "gf": gf, 
                "gc": gc, 
                "res": parsed_res
            })

        # 5. Ordenar cronológicamente para que la ventana deslizante Bayesiana (los últimos 7 partidos)
        # capture estrictamente la forma actual del equipo durante el transcurso del torneo.
        # Filtramos posibles fechas nulas antes de ordenar
        dynamic_matches.sort(key=lambda x: x.get("date", "1970-01-01"))
        
        return dynamic_matches
