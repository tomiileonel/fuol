"""
run_prediction_cli.py — Interfaz Inmutable para Predicciones
Uso: python run_prediction_cli.py "Equipo A" "Equipo B"
Este script NO modifica el repositorio. Solo invoca el motor y imprime el JSON.
"""
import sys
import json
import warnings
import pandas as pd

# Silenciar warnings de numpy/scipy para salida limpia
warnings.filterwarnings('ignore')

# Importar el motor blindado (No tocar)
from unified_engine import UnifiedEngine
from data_pipeline import DataPipeline

def main():
    if len(sys.argv) < 3:
        print("Error: Uso -> python run_prediction_cli.py 'Equipo A' 'Equipo B'")
        sys.exit(1)

    team_a = sys.argv[1].upper()
    team_b = sys.argv[2].upper()

    print(f"Cargando histórico global para {team_a} vs {team_b}...")
    pipeline = DataPipeline()
    
    try:
        df, _ = pipeline.prepare_data()
    except Exception as e:
        print(f"Error crítico cargando data_pipeline: {e}")
        sys.exit(1)

    # Filtrar historial estrictamente anterior a hoy (evita look-ahead)
    df['date'] = pd.to_datetime(df['date'])
    today = pd.Timestamp.now().normalize()
    hist_df = df[df['date'] < today]

    matches_a = pipeline.get_team_history(hist_df, team_a)
    matches_b = pipeline.get_team_history(hist_df, team_b)

    if not matches_a or len(matches_a) < 5:
        print(f"Error: Sin historial suficiente para {team_a} ({len(matches_a)} partidos).")
        sys.exit(1)
    if not matches_b or len(matches_b) < 5:
        print(f"Error: Sin historial suficiente para {team_b} ({len(matches_b)} partidos).")
        sys.exit(1)

    print("Ejecutando inferencia Dixon-Coles + Bayesianos...")
    
    # Instanciar motor
    engine = UnifiedEngine(
        team_a=team_a,
        team_b=team_b,
        matches_a=matches_a,
        matches_b=matches_b,
        venue='N', # Asumir neutral si no se especifica
        half_life=365.0 # Hiperparámetro de producción estabilizado
    )

    pred = engine.predict()

    # Limpiar la matriz para que el JSON sea legible
    if 'score_matrix' in pred:
        pred['score_matrix'] = pred['score_matrix'].tolist()
    if 'top_5_scores' in pred:
        pass # Ya es serializable

    # Imprimir resultado en formato JSON estructurado
    print("\n=== RESULTADO DEL MOTOR FUOL ===")
    print(json.dumps(pred, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
