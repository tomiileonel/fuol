"""
El heurístico de verificar_tipo_coincidencia.py (gf/gc espejado) tiene
falsos positivos estructurales: cualquier par de partidos con marcador
simétrico en la misma fecha (ej. dos 0-0 distintos, o cualquier par
donde gf_a==gc_b y gc_a==gf_b por casualidad) se marca como "mismo
partido" sin serlo. Esto es especialmente probable en empates, que son
el resultado más fácil de coincidir por azar.

Este script vuelve al CSV crudo y confirma por NOMBRE DE RIVAL -- no por
marcador -- cuáles de las fechas candidatas son un partido ARGENTINA vs
FRANCE real. Es la única fuente de verdad; el heurístico anterior era
una aproximación necesaria porque cargar_historial() descarta el nombre
del rival, pero no reemplaza esta confirmación.
"""
import csv


def buscar_partidos_en_fecha(csv_path: str, fecha: str) -> list[dict]:
    """Todas las filas del CSV crudo en una fecha dada, con equipos originales."""
    filas = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] == fecha:
                filas.append(row)
    return filas


def main():
    csv_path = "results.csv"

    # Las 19 fechas candidatas reportadas por verificar_tipo_coincidencia.py.
    fechas_candidatas = [
        "1930-07-15", "1965-06-03", "1971-01-08", "1971-01-13", "1972-06-25",
        "1974-05-18", "1977-06-26", "1978-06-02", "1978-06-06", "1986-03-26",
        "1986-06-05", "1986-06-25", "2007-02-07", "2009-02-11", "2011-03-29",
        "2014-11-18", "2018-06-30", "2018-11-16", "2022-12-18"
    ]

    confirmados_arg_vs_fra = []
    descartados_falso_positivo = []

    for fecha in fechas_candidatas:
        filas = buscar_partidos_en_fecha(csv_path, fecha)
        es_directo = any(
            {row["home_team"].strip().upper(), row["away_team"].strip().upper()}
            == {"ARGENTINA", "FRANCE"}
            for row in filas
        )
        if es_directo:
            confirmados_arg_vs_fra.append(fecha)
        else:
            descartados_falso_positivo.append(fecha)
            print(f"⚠️  {fecha}: NO es ARG vs FRA directo. Partidos reales ese día:")
            for row in filas:
                print(f"     {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}")

    print(f"\n{'='*70}")
    print(f"Confirmados como ARG vs FRA directo: {len(confirmados_arg_vs_fra)} de {len(fechas_candidatas)} candidatos")
    print(f"Descartados como falso positivo del heurístico gf/gc: {len(descartados_falso_positivo)}")

    if descartados_falso_positivo:
        print(f"\n🛑 El heurístico de marcador espejado tuvo {len(descartados_falso_positivo)}")
        print("   falso(s) positivo(s). El número real de self-leaks (Tipo A) es")
        print(f"   {len(confirmados_arg_vs_fra)}, no 19. Recalcular el impacto esperado")
        print("   en el RPS agregado con el número corregido antes de aceptar la")
        print("   causalidad completa del salto 0.2301 -> 0.2326.")
    else:
        print("\n✅ Los 19 (o los que se hayan pasado en la lista) se confirman")
        print("   como partidos ARG vs FRA directos por nombre de rival, no solo")
        print("   por marcador espejado. El heurístico anterior no tuvo falsos")
        print("   positivos en esta muestra.")


if __name__ == "__main__":
    main()
