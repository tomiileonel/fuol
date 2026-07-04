import json
import sys
import numpy as np
from pathlib import Path

def bootstrap_paired_rps(baseline: dict, candidato: dict, n_boot=1000, seed=42) -> dict:
    """
    Realiza un bootstrap pareado (muestreo con reemplazo) sobre la diferencia de RPS.
    Empareja los partidos por fecha para asegurar que se comparen exactamente los mismos.
    """
    # Encontrar las fechas que existen en ambos
    common_dates = sorted(list(set(baseline.keys()) & set(candidato.keys())))
    
    if len(common_dates) < 30:
        print(f"⚠️  Pocos partidos para un test estadístico robusto (N={len(common_dates)}). El resultado puede ser ruidoso.")
        
    if len(common_dates) < len(baseline) or len(common_dates) < len(candidato):
        print(f"⚠️  Folds perdidos detectados. Baseline={len(baseline)}, Candidato={len(candidato)}, Intersección={len(common_dates)}")
        
    if not common_dates:
        return {'cruza_cero': True, 'ci95_lo': 0.0, 'ci95_hi': 0.0, 'veredicto': "Sin datos"}
        
    # RPS menor es mejor. Diferencia positiva = baseline > candidato = candidato ES MEJOR
    # Pero standard is baseline - candidato
    diffs = np.array([baseline[d] - candidato[d] for d in common_dates])
    
    rng = np.random.default_rng(seed)
    boot_means = []
    
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
        
    boot_means.sort()
    
    ci95_lo = boot_means[int(n_boot * 0.025)]
    ci95_hi = boot_means[int(n_boot * 0.975)]
    
    cruza_cero = (ci95_lo <= 0.0 <= ci95_hi)
    
    if cruza_cero:
        veredicto = "➖ El intervalo de confianza cruza cero: no hay diferencia estadísticamente significativa."
    elif ci95_lo > 0:
        veredicto = "✅ El intervalo NO cruza cero y es positivo: el candidato es estadísticamente MEJOR que el baseline."
    else:
        veredicto = "❌ El intervalo NO cruza cero y es negativo: el candidato es estadísticamente PEOR que el baseline."
        
    return {
        'cruza_cero': cruza_cero,
        'ci95_lo': round(float(ci95_lo), 5),
        'ci95_hi': round(float(ci95_hi), 5),
        'veredicto': veredicto,
        'n_matches': len(common_dates)
    }

if __name__ == "__main__":
    path = Path("fase3_confirmacion_resultado.json")
    if not path.exists():
        print("❌ Archivo fase3_confirmacion_resultado.json no encontrado.")
        sys.exit(1)
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    baseline = data.get("rps_by_match_baseline", {})
    candidato = data.get("rps_by_match_candidato", {})
    
    if not baseline or not candidato:
        print("❌ No se encontraron los datos crudos 'rps_by_match' en el JSON.")
        sys.exit(1)
        
    print(f"Calculando bootstrap pareado (B=1000) entre baseline y candidato...")
    res = bootstrap_paired_rps(baseline, candidato)
    
    print(f"\nResultados (N = {res['n_matches']} partidos pareados):")
    print(f"Intervalo de confianza 95% para la mejora de RPS: [{res['ci95_lo']}, {res['ci95_hi']}]")
    print(res['veredicto'])
