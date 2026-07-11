"""
validation_core.py
===================
Refuerzo matemático del protocolo de validación (Fase 1/2/3, test_significancia.py,
suite de detección de leaks) del sistema FUOL.

FRENTE 2 — VALIDACIÓN ESTADÍSTICA Y DETECCIÓN DE LEAKS
--------------------------------------------------------
El resumen del repo describe un protocolo de 3 fases (baseline → barrido →
confirmación out-of-sample) con bootstrap pareado para significancia. Esto
es metodológicamente correcto en su estructura general, pero tiene UN
problema matemático de fondo, muy común y muy grave en investigación
cuantitativa: **el problema de comparaciones múltiples (multiple testing)**.

Si fase2_barrido.py prueba, digamos, 20 valores de half_life y elige el
mejor por RPS, y LUEGO test_significancia.py testea "¿es significativa la
mejora del ganador vs baseline?" usando bootstrap pareado estándar, el
resultado está sesgado: se está preguntando "¿es significativo el máximo
de 20 variables aleatorias?", no "¿es significativo ESTE valor específico?".
Esto es el mismo error que "p-hacking" en investigación clínica. La Fase 3
(out-of-sample) mitiga esto PARCIALMENTE, pero no lo resuelve del todo si
la Fase 3 solo valida "generaliza sí/no" sin corregir el nivel de
significancia esperado bajo la hipótesis nula de múltiples comparaciones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ============================================================================
# 2.1 — CORRECCIÓN DE COMPARACIONES MÚLTIPLES (Holm-Bonferroni / FDR)
# ============================================================================

@dataclass
class MultipleComparisonResult:
    n_hypotheses: int
    raw_p_values: list
    adjusted_p_values: list
    significant_mask: list
    method: str
    alpha: float


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05
                                 ) -> MultipleComparisonResult:
    """
    Corrección de Holm-Bonferroni (1979): control FWER (family-wise error
    rate) menos conservador que Bonferroni simple, pero igual de válido
    (uniformly more powerful, misma garantía de error tipo I).

    Aplicación directa a fase2_barrido.py: si se barren N valores de
    half_life (o rho, o cualquier hiperparámetro), y se calcula un p-valor
    de mejora vs baseline para CADA uno (no solo para el ganador), esta
    corrección dice cuáles mejoras sobreviven controlando la probabilidad
    de al menos un falso positivo en TODA la familia de tests.

    Algoritmo (ordenar ascendente, comparar contra alpha/(n-i)):
        p_(1) <= alpha/n       -> rechazar H0_(1), seguir
        p_(2) <= alpha/(n-1)   -> rechazar H0_(2), seguir
        ...
        primer no-rechazo detiene el proceso (todos los siguientes
        se declaran no significativos, es un procedimiento step-down)
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    significant = [False] * n

    running_max = 0.0
    stop = False
    for rank, (original_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        adj_p = min(1.0, p * (n - rank))
        running_max = max(running_max, adj_p)  # monotonicidad step-down
        adjusted[original_idx] = running_max
        if not stop and p <= threshold:
            significant[original_idx] = True
        else:
            stop = True  # a partir de aquí, ningún test se declara significativo

    return MultipleComparisonResult(
        n_hypotheses=n,
        raw_p_values=list(p_values),
        adjusted_p_values=adjusted,
        significant_mask=significant,
        method="Holm-Bonferroni (FWER)",
        alpha=alpha,
    )


def benjamini_hochberg_fdr(p_values: list[float], q: float = 0.10
                             ) -> MultipleComparisonResult:
    """
    Alternativa a Holm cuando N es grande (ej. barrer 50+ combinaciones de
    hiperparámetros): controla False Discovery Rate en vez de FWER. Es
    menos conservador, apropiado cuando el costo de un falso positivo
    aislado es bajo (se filtra igual en Fase 3 out-of-sample) pero se
    quiere más poder estadístico para no descartar mejoras reales.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    significant = [False] * n

    # Procesar de mayor a menor rank para monotonicidad correcta (BH step-up)
    min_so_far = 1.0
    for rank in range(n - 1, -1, -1):
        original_idx, p = indexed[rank]
        bh_value = p * n / (rank + 1)
        min_so_far = min(min_so_far, bh_value)
        adjusted[original_idx] = min_so_far

    threshold_rank = -1
    for rank, (original_idx, p) in enumerate(indexed):
        if p <= (rank + 1) / n * q:
            threshold_rank = rank
    for rank in range(threshold_rank + 1):
        significant[indexed[rank][0]] = True

    return MultipleComparisonResult(
        n_hypotheses=n,
        raw_p_values=list(p_values),
        adjusted_p_values=adjusted,
        significant_mask=significant,
        method="Benjamini-Hochberg (FDR)",
        alpha=q,
    )


# ============================================================================
# 2.2 — BOOTSTRAP PAREADO CON CORRECCIÓN DE SESGO (BCa)
# ============================================================================
#
# El resumen indica "bootstrap pareado" en test_significancia.py. El
# bootstrap percentil simple (asumido por defecto) tiene sesgo conocido
# cuando la distribución del estadístico (diferencia de RPS) es asimétrica,
# lo cual es EXACTAMENTE lo esperado en diferencias de RPS entre modelos
# cuando uno de los dos comete errores grandes ocasionales (colas pesadas).
# BCa (bias-corrected and accelerated, Efron 1987) corrige esto.

def bootstrap_bca_paired(
    baseline_scores: np.ndarray,
    challenger_scores: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """
    Intervalo de confianza BCa para la diferencia media pareada
    (challenger - baseline) de RPS por partido.

    A diferencia del bootstrap percentil ingenuo, BCa corrige:
      1) Sesgo (bias correction, z0): si la distribución bootstrap está
         desplazada respecto al estimador puntual original.
      2) Aceleración (acceleration, a): si la varianza del estadístico
         cambia según el valor del parámetro (heterocedasticidad), común
         cuando hay partidos "sorpresa" (Cisnes Negros) que dominan el
         error de ciertos hiperparámetros más que otros.

    Retorna también el p-valor implícito (fracción de la distribución
    bootstrap que cruza cero), más informativo que solo "significativo/no".
    """
    rng = np.random.default_rng(random_seed)
    diffs = challenger_scores - baseline_scores  # RPS: menor es mejor,
                                                   # diff negativa = challenger mejor
    n = len(diffs)
    observed_diff = np.mean(diffs)

    bootstrap_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(diffs[idx])

    # Bias correction z0
    prop_less = np.mean(bootstrap_diffs < observed_diff)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)  # evitar inf en ppf
    z0 = stats.norm.ppf(prop_less)

    # Acceleration a (vía jackknife)
    jackknife_diffs = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jackknife_diffs[i] = np.mean(diffs[mask])
    jackknife_mean = np.mean(jackknife_diffs)
    numerator = np.sum((jackknife_mean - jackknife_diffs) ** 3)
    denominator = 6.0 * (np.sum((jackknife_mean - jackknife_diffs) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    alpha_level = 1 - confidence
    z_lower = stats.norm.ppf(alpha_level / 2)
    z_upper = stats.norm.ppf(1 - alpha_level / 2)

    def bca_percentile(z):
        num = z0 + z
        denom = 1 - a * (z0 + z)
        adjusted_z = z0 + num / denom
        return stats.norm.cdf(adjusted_z)

    p_lower = np.clip(bca_percentile(z_lower), 0, 1)
    p_upper = np.clip(bca_percentile(z_upper), 0, 1)

    ci_lower = np.percentile(bootstrap_diffs, 100 * p_lower)
    ci_upper = np.percentile(bootstrap_diffs, 100 * p_upper)

    # p-valor bilateral implícito: fracción de la distribución bootstrap
    # que no cruza cero, en la dirección opuesta al efecto observado
    p_value = 2 * min(
        np.mean(bootstrap_diffs >= 0),
        np.mean(bootstrap_diffs <= 0),
    )
    p_value = min(p_value, 1.0)

    return {
        "observed_diff_rps": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence,
        "p_value_two_sided": p_value,
        "bias_correction_z0": z0,
        "acceleration_a": a,
        "significant": (ci_lower > 0) or (ci_upper < 0),
        "interpretation": (
            "Challenger significativamente MEJOR (RPS menor)" if ci_upper < 0 else
            "Challenger significativamente PEOR (RPS mayor)" if ci_lower > 0 else
            "No hay evidencia suficiente de diferencia real"
        ),
    }


# ============================================================================
# 2.3 — FORMALIZACIÓN DEL DATA LEAK COMO CONTAMINACIÓN DE σ-ÁLGEBRA
# ============================================================================
#
# El resumen describe una suite forense empírica de detección de leaks
# (coincidencias de fecha, self-leak por rival). Esto es correcto pero
# ad-hoc (detecta síntomas conocidos). Se agrega una definición matemática
# GENERAL de leak temporal, basada en teoría de filtración de información
# (la misma formalización que sustenta walk-forward validation en series
# de tiempo financieras), que permite verificar CUALQUIER feature del
# pipeline, no solo los casos ya conocidos.

@dataclass
class LeakageAuditResult:
    feature_name: str
    is_leaked: bool
    violating_rows: int
    total_rows: int
    max_lookahead_violation_days: float
    detail: str


def audit_temporal_leakage(
    feature_timestamps: np.ndarray,
    prediction_timestamps: np.ndarray,
    feature_name: str = "unnamed_feature",
) -> LeakageAuditResult:
    """
    Verificación formal de la condición de no-anticipación (non-anticipating
    / adaptedness a la filtración F_t, en el lenguaje de procesos
    estocásticos): para que una predicción en tiempo t sea válida, TODA
    la información usada para construirla debe tener timestamp <= t.

    Esto es la generalización matemática exacta de lo que
    verificar_coincidencias_fecha.py y confirmar_self_leak_por_rival.py
    verifican empíricamente caso por caso. Aplicando esta función a
    CUALQUIER columna candidata a feature (no solo resultados de partidos
    del mismo día, sino odds de mercado, lesiones reportadas, ratings
    actualizados, etc.) generaliza la detección a features futuras que
    aún no se sabe que puedan filtrar información.

    feature_timestamps: momento en que cada valor de feature estuvo
        REALMENTE disponible (ej. cuándo se publicó la noticia, no cuándo
        ocurrió el evento).
    prediction_timestamps: momento del kickoff de cada partido a predecir.

    Uso: aplicar a cada columna del dataset antes de fase1_baseline.py,
    no solo reactivamente cuando el RPS "se ve sospechosamente bueno".
    """
    violations = feature_timestamps > prediction_timestamps
    n_violations = int(violations.sum())

    if n_violations == 0:
        return LeakageAuditResult(
            feature_name=feature_name,
            is_leaked=False,
            violating_rows=0,
            total_rows=len(feature_timestamps),
            max_lookahead_violation_days=0.0,
            detail="Feature cumple adaptedness a F_t: sin lookahead detectado.",
        )

    lookahead_days = (feature_timestamps[violations] - prediction_timestamps[violations])
    max_violation = float(np.max(lookahead_days)) if len(lookahead_days) else 0.0

    return LeakageAuditResult(
        feature_name=feature_name,
        is_leaked=True,
        violating_rows=n_violations,
        total_rows=len(feature_timestamps),
        max_lookahead_violation_days=max_violation,
        detail=(
            f"{n_violations}/{len(feature_timestamps)} filas usan información "
            f"posterior al momento de predicción. Máxima violación: "
            f"{max_violation:.2f} unidades de tiempo hacia el futuro. "
            f"Esta feature DEBE excluirse o recalcularse con snapshot histórico "
            f"antes de cualquier backtest."
        ),
    )


def measure_leak_impact_on_rps(
    rps_with_leak: np.ndarray,
    rps_without_leak: np.ndarray,
) -> dict:
    """
    Generaliza medir_impacto_real_self_leak.py: cuantifica el impacto de
    un leak específico sobre el RPS vía la misma maquinaria BCa de 2.2,
    en vez de una comparación de medias simple. Un leak "pequeño en
    promedio" puede ser altamente significativo si es sistemático
    (baja varianza del efecto), lo cual bootstrap simple detecta mejor
    que un t-test que asume normalidad.
    """
    result = bootstrap_bca_paired(rps_without_leak, rps_with_leak)
    inflated_performance = result["observed_diff_rps"] < 0
    return {
        **result,
        "leak_inflates_performance": inflated_performance,
        "recommendation": (
            "CRÍTICO: el leak infla artificialmente el rendimiento reportado. "
            "Todo resultado de Fase 1/2 previo a este fix debe descartarse."
            if inflated_performance and result["significant"]
            else "El leak detectado no altera significativamente el RPS reportado, "
                 "pero debe corregirse igual por rigor metodológico."
        ),
    }


# ============================================================================
# 2.4 — WALK-FORWARD CON PURGING Y EMBARGO (Prado, 2018)
# ============================================================================
#
# Complemento directo a WalkForwardPipeline y a la
# división cronológica de fase3_confirmacion.py. Una partición cronológica
# simple en 2 mitades es válida pero deja un problema sutil: si hay
# features con ventanas móviles (medias de forma reciente, Elo con
# decaimiento), el primer partido de la Mitad 2 puede seguir usando datos
# calculados con información que "toca" el final de la Mitad 1
# (contaminación de borde). "Purging" y "embargo" (estándar en backtesting
# financiero, López de Prado 2018) resuelven esto de forma general.

def walk_forward_splits_with_purge_embargo(
    timestamps: np.ndarray,
    n_splits: int,
    embargo_fraction: float = 0.01,
    feature_window_days: float = 30.0,
) -> list[dict]:
    """
    Genera splits walk-forward donde:
      - PURGING: se eliminan del set de entrenamiento las observaciones
        cuya ventana de features se solapa con el período de test
        (evita que un partido de train "vea" indirectamente resultados
        de test a través de un feature de ventana móvil).
      - EMBARGO: se añade una brecha temporal extra después del set de
        test antes de reanudar entrenamiento en el próximo split, para
        prevenir fugas por autocorrelación serial residual.

    Esto generaliza fase3_confirmacion.py (que usa 2 mitades) a K splits
    walk-forward, aumentando la potencia estadística del test de
    significancia (más pares de comparación para el bootstrap de 2.2)
    sin sacrificar la validez temporal.
    """
    t_min, t_max = timestamps.min(), timestamps.max()
    total_span = t_max - t_min
    split_span = total_span / n_splits
    embargo_span = total_span * embargo_fraction

    splits = []
    for i in range(1, n_splits):
        test_start = t_min + i * split_span
        test_end = t_min + (i + 1) * split_span

        purge_boundary = test_start - feature_window_days
        train_mask = timestamps < purge_boundary
        test_mask = (timestamps >= test_start) & (timestamps < test_end)

        embargo_end = test_end + embargo_span
        next_train_eligible_mask = timestamps >= embargo_end

        splits.append({
            "split_index": i,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "next_train_eligible_mask": next_train_eligible_mask,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "purge_boundary": purge_boundary,
            "embargo_end": embargo_end,
        })
    return splits