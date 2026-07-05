"""
statistical_validation.py
==========================
Refuerzo de la suite de validación estadística (fase1/2/3, test_significancia,
detección de leaks) con matemática de mayor rigor.

CONTEXTO
--------
El sistema ya implementa: RPS baseline (fase1), barrido de hiperparámetros
(fase2), validación out-of-sample (fase3), bootstrap pareado (test_significancia)
y detección forense de leaks por coincidencia de fecha/rival. Esto es sólido
como protocolo, pero tiene 3 vacíos matemáticos comunes en este tipo de
pipelines que este módulo cierra:

  1. PROBLEMA DE COMPARACIONES MÚLTIPLES: si fase2_barrido.py prueba, por
     ejemplo, 20 valores de half_life y se queda con el mejor por RPS, la
     probabilidad de que AL MENOS UNO parezca "significativamente mejor"
     que el baseline por puro azar (falso positivo) NO es el 5% nominal,
     es MUCHO más alta. Esto es el problema clásico de "p-hacking"
     involuntario en optimización de hiperparámetros. Se corrige con
     Holm-Bonferroni (o control de FDR de Benjamini-Hochberg).

  2. CALIBRACIÓN PROBABILÍSTICA: el sistema mide RPS (que evalúa ranking
     ordinal de resultados) pero no parece medir si las probabilidades
     están CALIBRADAS (si el modelo dice "60% de probabilidad" para un
     conjunto de partidos, ¿gana realmente cerca del 60% de las veces?).
     Un modelo puede tener buen RPS relativo y aun así estar mal calibrado
     en probabilidad absoluta -- esto es CRÍTICO para el paper_trader
     porque el Kelly Criterion asume que las probabilidades son correctas
     en valor absoluto, no solo en orden relativo.

  3. DETECCIÓN DE LEAKS: la suite actual usa coincidencia de fecha/rival
     (heurística directa, útil pero limitada a leaks conocidos a priori).
     Se añade un test de permutación GENERAL: si romper aleatoriamente la
     estructura temporal del dataset y re-correr el backtest da un RPS
     estadísticamente indistinguible del RPS real, eso es evidencia de
     que el modelo no está usando información temporal genuina (podría
     estar filtrando información de otra forma no contemplada por los
     scripts forenses existentes, ej. leakage vía un feature derivado).

FUNDAMENTO MATEMÁTICO
----------------------
### 3.1 Corrección de comparaciones múltiples (Holm-Bonferroni)

Dado un conjunto de m p-valores {p_1, ..., p_m} (uno por cada
hiperparámetro/variante probada contra el baseline), ordenados
ascendentemente p_(1) <= p_(2) <= ... <= p_(m):

    Rechazar H0_(i) si p_(i) <= alpha / (m - i + 1)   para el primer i
    donde esto falla, se detienen todos los rechazos posteriores.

Esto controla la tasa de error familiar (FWER) exactamente al nivel
alpha, y es UNIFORMEMENTE MÁS POTENTE que Bonferroni simple (menos
conservador, mismo control de error). Alternativa: Benjamini-Hochberg
controla la tasa de falsos descubrimientos (FDR) en vez de FWER, más
apropiada si se acepta cierta tolerancia a falsos positivos a cambio de
más potencia estadística (más relevante si fase2_barrido prueba
decenas de configuraciones).

### 3.2 Calibración: Reliability Diagram + Expected Calibration Error (ECE)

Se discretizan las probabilidades predichas en B bins (ej. B=10):
    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    acc(b)  = frecuencia empírica de aciertos en el bin b
    conf(b) = probabilidad promedio predicha en el bin b
    n_b     = número de observaciones en el bin b

Además se implementa el test de Hosmer-Lemeshow adaptado a resultado
multinomial (1X2), que da un p-valor formal para H0: "el modelo está
bien calibrado" (en vez de solo un número descriptivo como ECE).

### 3.3 Test de permutación para validación de estructura temporal

H0: el orden temporal de los partidos no aporta información al modelo
(si esto fuera cierto, el sistema NO debería depender de time_decay, y
cualquier mejora atribuida a modelar el tiempo sería espuria).

Procedimiento:
    1. Calcular RPS_real del modelo con el orden temporal genuino.
    2. Repetir N_perm veces: permutar aleatoriamente las fechas (romper
       el orden cronológico manteniendo los resultados), re-correr el
       walk-forward backtest, registrar RPS_perm.
    3. p-valor = P(RPS_perm <= RPS_real) bajo H0 (proporción empírica).

Si el p-valor es alto (ej. > 0.10), es evidencia de que el "walk-forward"
no está genuinamente aprovechando SOLO información pasada -- señal de
alerta de posible leak temporal no detectado por los scripts forenses
basados en heurísticas de fecha exacta.

### 3.4 Bootstrap pareado con corrección de sesgo (BCa)

El bootstrap pareado existente probablemente usa percentiles simples.
Se añade BCa (Bias-Corrected and accelerated), que corrige:
  (a) el sesgo de la distribución bootstrap respecto al estimador
      puntual real, y
  (b) la asimetría (skewness) de esa distribución,
dando intervalos de confianza más precisos que percentiles simples,
especialmente relevante con RPS (que está acotado en [0,1] y su
distribución bootstrap suele ser asimétrica cerca de los extremos).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass


# ----------------------------------------------------------------------
# 3.1 Corrección de comparaciones múltiples
# ----------------------------------------------------------------------
def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    p_values: {nombre_variante: p_valor} -- ej. cada valor de half_life
    probado en fase2_barrido.py contra el baseline.

    Devuelve, por variante, si se rechaza H0 (mejora real) tras la
    corrección, y el umbral ajustado que tuvo que superar.

    CRÍTICO para fase2_barrido.py: si se prueban 20 half_life y 3 dan
    p < 0.05 individualmente, SIN esta corrección se reportarían como
    "3 mejoras significativas" cuando en realidad, con 20 pruebas, se
    espera ~1 falso positivo solo por azar (20 * 0.05 = 1).
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    stop = False
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        reject = (not stop) and (p <= threshold)
        if not reject:
            stop = True
        out[name] = {
            "p_value": p,
            "adjusted_threshold": threshold,
            "significant_after_correction": reject,
            "rank": i + 1,
        }
    return out


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    Control de False Discovery Rate (FDR) en vez de FWER. Más potente
    (menos conservador) que Holm-Bonferroni; apropiado cuando se
    exploran MUCHAS configuraciones (ej. grid search de half_life x rho)
    y se acepta que una fracción controlada de los "descubrimientos"
    sea falsa, a cambio de no perder mejoras reales por exceso de
    conservadurismo.

    Rechazar H0_(i) si p_(i) <= (i/m) * alpha, para el i más grande que
    cumple la desigualdad; todos los p_(1..i) también se rechazan.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    thresholds = [(i + 1) / m * alpha for i in range(m)]

    largest_i = -1
    for i in range(m - 1, -1, -1):
        if items[i][1] <= thresholds[i]:
            largest_i = i
            break

    out = {}
    for i, (name, p) in enumerate(items):
        out[name] = {
            "p_value": p,
            "bh_threshold": thresholds[i],
            "significant_after_correction": i <= largest_i,
            "rank": i + 1,
        }
    return out


# ----------------------------------------------------------------------
# 3.2 Calibración probabilística
# ----------------------------------------------------------------------
@dataclass
class CalibrationResult:
    ece: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray   # probabilidad promedio predicha por bin
    bin_accuracy: np.ndarray     # frecuencia empírica de acierto por bin
    bin_counts: np.ndarray
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    well_calibrated: bool


def expected_calibration_error(predicted_probs: np.ndarray, outcomes: np.ndarray,
                                n_bins: int = 10) -> CalibrationResult:
    """
    predicted_probs: array de probabilidades predichas para el evento
        "ocurre" (ej. P(home_win) para cada partido).
    outcomes: array binario (1 si el evento ocurrió, 0 si no).

    Implementa ECE + test de Hosmer-Lemeshow para dar significancia
    estadística formal a la calibración (no solo un número descriptivo).

    Hosmer-Lemeshow:
        HL = sum_b [ (O_b - E_b)^2 / (E_b * (1 - E_b/n_b)) ]

        O_b = aciertos observados en el bin b
        E_b = suma de probabilidades predichas en el bin b (aciertos esperados)
        n_b = número de observaciones en el bin b

    HL ~ chi2(n_bins - 2) bajo H0 de buena calibración (aprox. estándar
    en la literatura de calibración logística, aplicable aquí porque el
    mecanismo generador -- comparar conteo observado vs esperado por bin
    -- es idéntico).
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    assert predicted_probs.shape == outcomes.shape

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted_probs, bin_edges[1:-1]), 0, n_bins - 1)

    bin_confidence = np.zeros(n_bins)
    bin_accuracy = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    hl_stat = 0.0
    n_valid_bins_for_hl = 0

    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        bin_counts[b] = n_b
        if n_b == 0:
            continue
        conf_b = predicted_probs[mask].mean()
        acc_b = outcomes[mask].mean()
        bin_confidence[b] = conf_b
        bin_accuracy[b] = acc_b

        O_b = outcomes[mask].sum()
        E_b = predicted_probs[mask].sum()
        denom = E_b * (1 - E_b / n_b) if n_b > 0 else 0
        if denom > 1e-9:
            hl_stat += (O_b - E_b) ** 2 / denom
            n_valid_bins_for_hl += 1

    n_total = len(predicted_probs)
    ece = float(np.sum(bin_counts / n_total * np.abs(bin_accuracy - bin_confidence)))

    df = max(n_valid_bins_for_hl - 2, 1)
    hl_pvalue = float(1 - stats.chi2.cdf(hl_stat, df))

    return CalibrationResult(
        ece=ece, bin_edges=bin_edges, bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy, bin_counts=bin_counts,
        hosmer_lemeshow_stat=float(hl_stat), hosmer_lemeshow_pvalue=hl_pvalue,
        well_calibrated=hl_pvalue > 0.05,
    )


# ----------------------------------------------------------------------
# 3.3 Test de permutación para validar dependencia temporal genuina
# ----------------------------------------------------------------------
def temporal_permutation_test(compute_rps_fn, dates: np.ndarray, n_perm: int = 500,
                               seed: int = 42) -> dict:
    """
    compute_rps_fn: función que recibe un array de fechas (mismo largo
        que el dataset) y devuelve el RPS del walk-forward backtest
        usando ESE orden temporal. Debe ser provista por el caller
        (envuelve fase1_baseline.py / el motor real), porque este módulo
        no tiene acceso al dataset ni al motor de predicción -- solo
        implementa la lógica estadística del test.

    dates: fechas/orden temporal genuino del dataset.

    H0: el orden temporal no aporta información (el modelo funcionaría
    igual con fechas permutadas al azar). Rechazar H0 es la situación
    DESEADA: confirma que el walk-forward genuinamente depende de la
    estructura temporal real y no hay leak oculto que haga que
    cualquier orden dé resultados similares.
    """
    rng = np.random.default_rng(seed)
    rps_real = compute_rps_fn(dates)

    rps_perm = np.empty(n_perm)
    for k in range(n_perm):
        perm_dates = rng.permutation(dates)
        rps_perm[k] = compute_rps_fn(perm_dates)

    # RPS más bajo = mejor (es un score de error). H0 rechazada si el
    # RPS real es significativamente MEJOR (menor) que la distribución
    # bajo permutación aleatoria.
    p_value = float(np.mean(rps_perm <= rps_real))

    return {
        "rps_real": float(rps_real),
        "rps_perm_mean": float(rps_perm.mean()),
        "rps_perm_std": float(rps_perm.std()),
        "p_value": p_value,
        "temporal_structure_confirmed": p_value < 0.05,
        "interpretation": (
            "El modelo depende genuinamente del orden temporal real "
            "(rechaza H0 correctamente)." if p_value < 0.05 else
            "ALERTA: el RPS real no es distinguible de permutaciones "
            "aleatorias de fecha. Investigar posible leak no capturado "
            "por los scripts forenses basados en heurísticas de fecha "
            "exacta (verificar_coincidencias_fecha.py, etc.)."
        ),
    }


# ----------------------------------------------------------------------
# 3.4 Bootstrap pareado BCa (Bias-Corrected and accelerated)
# ----------------------------------------------------------------------
def paired_bootstrap_bca(scores_model_a: np.ndarray, scores_model_b: np.ndarray,
                          n_boot: int = 10_000, alpha: float = 0.05,
                          seed: int = 42) -> dict:
    """
    scores_model_a, scores_model_b: arrays de RPS por partido (pareados,
    mismo partido evaluado por ambos modelos) -- ej. modelo baseline vs
    modelo con nuevo half_life.

    Devuelve el IC BCa para la diferencia media (A - B), más robusto que
    percentiles simples cuando la distribución bootstrap de la
    diferencia es asimétrica (común en RPS, acotado en [0,1]).

    Método (Efron & Tibshirani, 1993):
      1. theta_hat = estimador puntual (diferencia de medias real)
      2. Generar B réplicas bootstrap: theta*_b
      3. Bias correction: z0 = Phi^{-1}( proporción de theta*_b < theta_hat )
      4. Acceleration: a, vía jackknife de la métrica
      5. Percentiles ajustados:
            alpha_1 = Phi( z0 + (z0 + z_{alpha/2}) / (1 - a*(z0 + z_{alpha/2})) )
            alpha_2 = Phi( z0 + (z0 + z_{1-alpha/2}) / (1 - a*(z0 + z_{1-alpha/2})) )
         y el IC final son los percentiles alpha_1, alpha_2 de la
         distribución bootstrap (no alpha/2, 1-alpha/2 directos como en
         percentil simple).
    """
    rng = np.random.default_rng(seed)
    diffs = scores_model_a - scores_model_b
    n = len(diffs)
    theta_hat = diffs.mean()

    # Paso 2: réplicas bootstrap
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    # Paso 3: bias correction z0
    prop_less = np.mean(boot_means < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)  # evitar z=+-inf
    z0 = stats.norm.ppf(prop_less)

    # Paso 4: acceleration via jackknife
    jack_means = np.empty(n)
    for i in range(n):
        jack_means[i] = np.delete(diffs, i).mean()
    jack_mean_overall = jack_means.mean()
    num = np.sum((jack_mean_overall - jack_means) ** 3)
    den = 6.0 * (np.sum((jack_mean_overall - jack_means) ** 2) ** 1.5)
    a_hat = num / den if den > 1e-12 else 0.0

    # Paso 5: percentiles ajustados
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def _adjusted_percentile(z):
        num = z0 + z
        denom = 1 - a_hat * num
        return stats.norm.cdf(z0 + num / denom) if abs(denom) > 1e-9 else np.nan

    p_lo = _adjusted_percentile(z_lo)
    p_hi = _adjusted_percentile(z_hi)

    ci_lo = np.percentile(boot_means, 100 * p_lo) if not np.isnan(p_lo) else np.nan
    ci_hi = np.percentile(boot_means, 100 * p_hi) if not np.isnan(p_hi) else np.nan

    # p-valor bootstrap bilateral: proporción de réplicas que cruzan 0
    # en la dirección opuesta al signo del estimador puntual
    if theta_hat >= 0:
        p_value = 2 * np.mean(boot_means <= 0)
    else:
        p_value = 2 * np.mean(boot_means >= 0)
    p_value = float(min(p_value, 1.0))

    return {
        "mean_diff_a_minus_b": float(theta_hat),
        "ci_bca": (float(ci_lo), float(ci_hi)),
        "p_value_bootstrap": p_value,
        "significant": bool(ci_lo > 0 or ci_hi < 0),
        "bias_correction_z0": float(z0),
        "acceleration_a": float(a_hat),
        "interpretation": (
            f"Modelo A {'mejora' if theta_hat < 0 else 'empeora'} el RPS "
            f"respecto a B en promedio {abs(theta_hat):.5f} "
            f"({'diferencia estadísticamente sólida' if (ci_lo > 0 or ci_hi < 0) else 'no se puede descartar que sea ruido de muestra'})."
        ),
    }