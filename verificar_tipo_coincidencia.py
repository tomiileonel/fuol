"""
Las 192 coincidencias de fecha entre ARGENTINA y FRANCE pueden ser de dos
tipos MUY distintos en severidad, y el conteo agregado no los distingue:

  (A) Argentina y Francia jugaron ENTRE SÍ ese día. En ese caso, el
      partido de FRANCE en train_b bajo el filtro viejo (<=) es el
      resultado del MISMO partido que test_m está por predecir -- el
      motor tendría el resultado exacto que intenta pronosticar
      filtrado hacia su propio training set. Esto no es un data leak
      sutil, es el peor caso posible.

  (B) Argentina jugó contra un tercero el mismo día que Francia jugó
      contra otro tercero (coincidencia de fecha FIFA/Mundial). Esto es
      leak real pero mucho más leve: Francia-vs-tercero no revela el
      resultado ARG-vs-tercero.

Este script separa (A) de (B) y cuenta cuántas de las 192 coincidencias
son cada tipo, y además restringe el conteo a folds que el walk-forward
realmente evalúa como test_m (k >= min_train_size), porque coincidencias
en partidos anteriores al arranque del walk-forward no pueden haber
afectado el RPS agregado en absoluto.
"""
from fase1_baseline import cargar_historial


def main():
    csv_path = "results.csv"
    min_train_size = 10  # mismo default usado en fase1_baseline.py / fase2_barrido.py

    matches_a = cargar_historial(csv_path, "ARGENTINA")
    matches_b = cargar_historial(csv_path, "FRANCE")

    fechas_b_set = {m['date'] for m in matches_b}

    # Folds que el walk-forward realmente evalúa como test_m.
    evaluados = matches_a[min_train_size:]
    coincidencias_evaluadas = [m for m in evaluados if m['date'] in fechas_b_set]

    print(f"Partidos de ARGENTINA evaluados como test_m (k>={min_train_size}): {len(evaluados)}")
    print(f"De esos, con fecha coincidente con algún partido de FRANCE: {len(coincidencias_evaluadas)}")

    tipo_a_directo = []  # Argentina vs Francia, mismo partido
    tipo_b_indirecto = []  # cada uno vs un tercero, misma fecha

    for m in coincidencias_evaluadas:
        # ¿Fue directamente contra Francia? cargar_historial no guarda el
        # nombre del rival, así que inferimos por gf/gc espejados: si
        # existe un partido de FRANCE en esa fecha con gf/gc invertidos
        # exactos respecto al de ARGENTINA, es fuertemente sugestivo de
        # que es el mismo enfrentamiento visto desde el otro lado.
        candidatos_b = [b for b in matches_b if b['date'] == m['date']]
        es_mismo_partido = any(
            b['gf'] == m['gc'] and b['gc'] == m['gf'] for b in candidatos_b
        )
        if es_mismo_partido:
            tipo_a_directo.append(m)
        else:
            tipo_b_indirecto.append(m)

    print(f"\n--- Desglose ---")
    print(f"Tipo A (probable ARG vs FRA directo, mismo resultado espejado): {len(tipo_a_directo)}")
    print(f"Tipo B (fecha compartida, rivales distintos):                  {len(tipo_b_indirecto)}")

    print(f"\n{'='*70}")
    if tipo_a_directo:
        print(f"🛑 Hay {len(tipo_a_directo)} caso(s) de Tipo A. Esto es más grave que un")
        print("   'leak leve de fecha compartida': significa que, bajo el filtro")
        print("   viejo (<=), el motor tenía el resultado EXACTO del partido que")
        print("   estaba por predecir metido en su propio training set. Confirmar")
        print("   manualmente estas fechas contra el CSV crudo antes de aceptar")
        print("   cualquier cifra de RPS que las incluya sin este fix.")
        for m in tipo_a_directo:
            print(f"     {m['date']}  gf={m['gf']} gc={m['gc']}")
    else:
        print("✅ No se detectaron casos de Tipo A por este heurístico de gf/gc")
        print("   espejado. Las coincidencias parecen ser Tipo B (fecha compartida,")
        print("   rivales distintos) -- leak real pero de severidad menor, y el")
        print("   conteo relevante para el salto de RPS es el filtrado por")
        print("   min_train_size, no el de 192 sobre el historial completo.")

    print(f"\nNota: el conteo de coincidencias RELEVANTE para el walk-forward es")
    print(f"{len(coincidencias_evaluadas)} (folds realmente evaluados), no 192")
    print(f"(coincidencias en todo el historial, incluyendo partidos de 1911-1989")
    print(f"que min_train_size ya excluye de la evaluación).")


if __name__ == "__main__":
    main()
