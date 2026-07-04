"""
Verificación puntual: ¿el fix de train_b (<= -> <) explica por sí solo
el salto de baseline 0.2301 -> 0.2326?

El fix solo puede afectar folds donde algún partido de team_b comparte
fecha EXACTA con el test_date de team_a (antes esos partidos se incluían
en train_b, ahora se excluyen). Si el conteo de esas coincidencias es
chico, un salto de RPS de 0.0025 no puede explicarse solo por este
mecanismo -- hay que buscar otra causa antes de aceptar 0.2326 como el
"verdadero piso out-of-sample".
"""
from fase1_baseline import cargar_historial


def main():
    csv_path = "results.csv"
    matches_a = cargar_historial(csv_path, "ARGENTINA")
    matches_b = cargar_historial(csv_path, "FRANCE")

    fechas_a = [m['date'] for m in matches_a]
    fechas_b_set = {m['date'] for m in matches_b}

    coincidencias = [f for f in fechas_a if f in fechas_b_set]

    print(f"Partidos de ARGENTINA: {len(matches_a)}")
    print(f"Partidos de FRANCE:    {len(matches_b)}")
    print(f"Fechas de ARGENTINA que coinciden con alguna fecha de FRANCE: {len(coincidencias)}")

    if coincidencias:
        print("\nFechas coincidentes (muestra hasta 10):")
        for f in coincidencias[:10]:
            print(f"  {f}")

    print(f"\n{'='*70}")
    if len(coincidencias) <= 10:
        print(f"⚠️  Solo {len(coincidencias)} fechas coincidentes de {len(matches_a)} partidos.")
        print("   El fix de train_b (<= -> <) SOLO puede alterar el resultado en")
        print("   folds donde el test_date de un partido coincide con la fecha de")
        print("   un partido de FRANCE. Con este conteo tan chico, un salto de RPS")
        print("   de 0.0025 agregado sobre >1000 partidos NO es explicable por este")
        print("   mecanismo únicamente. Hay otra causa moviendo el baseline --")
        print("   revisar si cambió algo más entre las dos corridas (versión del")
        print("   CSV, min_train_size por defecto, algún otro parámetro) antes de")
        print("   aceptar 0.2326 como el verdadero piso out-of-sample.")
    else:
        print(f"{len(coincidencias)} coincidencias es una fracción mayor de lo esperado")
        print("del dataset -- en ese caso sí es plausible que el fix por sí solo")
        print("explique buena parte del salto. Aun así, confirmar que ningún otro")
        print("parámetro cambió entre corridas.")


if __name__ == "__main__":
    main()
