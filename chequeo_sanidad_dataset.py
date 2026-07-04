"""
Chequeo de sanidad para el dataset de Kaggle ANTES de usarlo en el backtest.
 
Esto no reemplaza mirar el archivo -- es un chequeo rápido y objetivo,
del mismo tipo que usamos para detectar el CSV inventado. Corré esto
apenas descargues el archivo, antes de adaptarlo a cargar_historial().
 
Uso:
    python chequeo_sanidad_dataset.py results.csv
"""
import csv
import sys
from collections import Counter
from datetime import datetime
 
 
def main():
    if len(sys.argv) < 2:
        print("Uso: python chequeo_sanidad_dataset.py <archivo.csv>")
        return
 
    path = sys.argv[1]
 
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
 
    print(f"Archivo: {path}")
    print(f"Columnas detectadas: {reader.fieldnames}")
    print(f"Total de filas: {len(rows)}")
 
    if len(rows) < 1000:
        print("⚠️  Menos de 1000 filas totales -- este NO parece ser el dataset")
        print("   completo de Kaggle (que trae ~40,000-48,000 partidos). Puede")
        print("   estar truncado o ser un archivo distinto al esperado.")
 
    # Intentar detectar la columna de fecha (puede llamarse 'date', 'Date', etc.)
    date_col = next((c for c in reader.fieldnames if c.lower() == "date"), None)
    home_col = next((c for c in reader.fieldnames if "home" in c.lower() and "team" in c.lower()), None)
    away_col = next((c for c in reader.fieldnames if "away" in c.lower() and "team" in c.lower()), None)
    hscore_col = next((c for c in reader.fieldnames if "home" in c.lower() and "score" in c.lower()), None)
    ascore_col = next((c for c in reader.fieldnames if "away" in c.lower() and "score" in c.lower()), None)
 
    print(f"\nColumnas identificadas -> fecha: {date_col}, local: {home_col}, "
          f"visitante: {away_col}, goles local: {hscore_col}, goles visitante: {ascore_col}")
 
    if not all([date_col, home_col, away_col, hscore_col, ascore_col]):
        print("⚠️  No pude identificar automáticamente todas las columnas esperadas.")
        print("   Revisá el nombre real de las columnas arriba y ajustá cargar_historial()")
        print("   a mano -- esto es normal, cada versión del dataset nombra distinto.")
        return
 
    # Rango de fechas
    fechas = []
    for r in rows:
        try:
            fechas.append(datetime.fromisoformat(r[date_col][:10]))
        except (ValueError, KeyError):
            continue
    if fechas:
        print(f"\nRango de fechas: {min(fechas).date()} a {max(fechas).date()}")
 
    # Nombres de equipo únicos -- si aparece algo tipo "Opponent_N", es la
    # misma señal de datos sintéticos que ya cazamos antes.
    equipos = set()
    for r in rows:
        equipos.add(r[home_col])
        equipos.add(r[away_col])
    sospechosos = [e for e in equipos if e.lower().startswith("opponent") or e.lower().startswith("team_")]
    print(f"\nEquipos únicos detectados: {len(equipos)}")
    if sospechosos:
        print(f"❌ Nombres sospechosos de ser placeholders sintéticos: {sospechosos[:10]}")
    else:
        print("✅ Ningún nombre con patrón de placeholder sintético (Opponent_N, Team_N, etc.)")
 
    # Marcadores fuera de rango histórico plausible.
    # El máximo real conocido en un partido de selecciones absolutas ronda
    # los 31-0 (Australia-Samoa Americana, 2001, clasificación) -- un caso
    # extremo documentado. Cualquier cosa por encima de ~15 goles de un lado
    # amerita mirar la fila a mano, no descartarla automáticamente.
    marcadores_altos = []
    for r in rows:
        try:
            hs, aws = int(r[hscore_col]), int(r[ascore_col])
            if hs > 15 or aws > 15:
                marcadores_altos.append((r[date_col], r[home_col], hs, aws, r[away_col]))
        except (ValueError, KeyError):
            continue
    print(f"\nPartidos con marcador >15 goles de un lado: {len(marcadores_altos)}")
    if marcadores_altos:
        print("   (esto es ESPERABLE en un dataset real de 150 años -- hay goleadas")
        print("   históricas documentadas en clasificatorios entre selecciones muy")
        print("   dispares. No es automáticamente sospechoso como lo era el 9-4 en")
        print("   una muestra de 5 partidos -- acá simplemente mostrá 2-3 ejemplos")
        print("   y confirmá a ojo que tengan sentido, ej. un rival minúsculo real.")
        for ejemplo in marcadores_altos[:3]:
            print(f"     {ejemplo}")
 
    print("\n=== Resumen ===")
    print("Si viste ✅ en nombres de equipo y el rango de fechas cubre décadas")
    print("(no solo unos pocos días), el dataset pasa el chequeo de sanidad básico.")
    print("Próximo paso: adaptar cargar_historial() en fase1_baseline.py con los")
    print("nombres de columna reales que se identificaron arriba.")
 
 
if __name__ == "__main__":
    main()
