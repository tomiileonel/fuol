# FUOL-SKILL: SISTEMA DE PREDICCIÓN Y ANÁLISIS DE FÚTBOL

> Documentación alineada al código real del repositorio.
> Versión: 2.x (motor unificado, sin módulos v3/hawkes/quantum)

---

## 1. IDENTIDAD Y PROPÓSITO

FUOL (Football Unified Optimization Layer) es un sistema cuantitativo de
predicción deportiva y trading algorítmico en fútbol. Integra un motor
estadístico basado en Dixon-Coles + inferencia Bayesiana Gamma-Poisson
con priors de Elo, un pipeline de datos con decaimiento temporal
walk-forward, gestión de riesgo con Kelly multi-resultado, y una API
FastAPI para consultar predicciones y ejecutar paper trading.

**Filosofía 0% inversión**: no se implementan flujos que requieran
dinero real, pagos de APIs comerciales ni operaciones financieras reales.
Solo se usan fuentes gratuitas, públicas o de demostración.

---

## 2. ARQUITECTURA REAL DEL SISTEMA

### 2.1 Núcleo estadístico

- **`unified_engine.py`**: motor unificado (`UnifiedEngine`). Orquesta:
  - `TimeWeighter`: decaimiento exponencial con `half_life` optimizable
    por Leave-One-Out CV.
  - `BayesianGoalRates`: inferencia Gamma-Poisson con prior informativo
    desde Elo.
  - `DixonColes`: corrección ρ estimada por MLE con prior Bayesiano.
  - `WalkForwardBacktester`: evaluación walk-forward con Brier, Log-Loss,
    RPS (Ranked Probability Score) y ECE (calibration error).
- **`advanced_dixon_coles.py`**: extensión con MLE conjunta, matriz de
  información de Fisher para IC de parámetros, y cópula de Frank
  discreta opcional.
- **`probability_calibrator.py`**: calibración de probabilidades.
- **`feature_engine.py`**: `EloRegistry` dinámico con Margin of Victory
  y `FeatureEngineer.build_features` (Elo + EWMA con opponent-adjusted
  goals).

### 2.2 Pipeline de datos

- **`data_pipeline.py`**: ingesta del CSV `results.csv` (~50k partidos
  internacionales 1872–presente), limpieza, ordenado cronológico,
  enriquecimiento con Elo y EWMA.
- **`cache.py`**, **`rate_limit.py`**: utilidades de caché y rate-limit.
- **`external_data_provider.py`**, **`sync_external_data.py`**: ingesta
  opcional de fuentes externas gratuitas vía `EXTERNAL_DATA_URL`.

### 2.3 Gestión de riesgo

- **`kelly_risk_engine.py`**:
  - `kelly_binary`: Kelly clásico (referencia).
  - `kelly_multi_outcome`: Kelly simultáneo 1X2 vía optimización cóncava
    (SLSQP) con restricción de bankroll.
  - `kelly_robust_binary` (alias retrocompatible `kelly_bayesian_binary`):
    Kelly robusto por incertidumbre epistémica usando cuantil inferior de
    la posterior.
  - `ruin_probability` / `optimal_fractional_kelly`: teoría de la ruina
    del jugador vía aproximación GBM, para fijar fracción de Kelly
    derivada de un umbral explícito de P(ruina).
- **`risk_core.py`**, **`paper_trader.py`**: motor de paper trading.
- **`settlement_engine.py`**: liquidación de resultados post-partido.

### 2.4 API y monitoreo

- **`api_server.py`**: FastAPI con endpoint `POST /api/predict` y
  frontend estático en `frontend/`.
- **`production_monitor.py`**, **`production_logger.py`**,
  **`performance_tracker.py`**: telemetría y logging.
- **`tournament_simulator.py`**, **`tournament_dashboard.py`**: Monte
  Carlo sobre torneos completos.

### 2.5 Validación y prevención de fugas de datos

- **`walk_forward_pipeline.py`**: walk-forward con embargo temporal
  estricto (anti data-leakage).
- **`statistical_validation.py`**, **`validation_core.py`**: tests de
  significancia.
- **`chequeo_sanidad_dataset.py`**, **`verificar_*`**: auditoría del
  dataset.

---

## 3. PROTOCOLO DE EJECUCIÓN

Cuando se solicita un análisis de partido, el pipeline es:

### FASE 1: INGESTIÓN Y VALIDACIÓN
1. Identificar equipos, torneo y fecha.
2. Cargar histórico del CSV y validar cantidad mínima de partidos (≥5).
3. Integrar contexto externo si está disponible (lesiones, clima).

### FASE 2: CÁLCULO DEL BASELINE
1. Calcular `λ_local` y `λ_visitante` vía `BayesianGoalRates` con prior
   desde `EloRegistry.expected_goal_ratio`.
2. Aplicar decaimiento temporal (`TimeWeighter` con `half_life` óptimo).
3. Construir matriz de marcadores vía `AdvancedDixonColes.score_matrix`.
4. Extraer 1X2 y top-k marcadores exactos.

### FASE 3: CALIBRACIÓN (OPCIONAL)
1. Si hay un `ProbabilityCalibrator` ajustado, re-calibrar el 1X2.

### FASE 4: VALIDACIÓN WALK-FORWARD
1. Si hay ≥8 partidos por equipo, ejecutar `WalkForwardBacktester`.
2. Reportar Brier, Log-Loss, RPS, Hit Rate y ECE.

### FASE 5: EVALUACIÓN DE RIESGO Y VALOR (KELLY)
1. Ingerir cuotas provistas por el usuario.
2. Calcular EV por resultado: `EV_i = p_i * o_i - 1`.
3. Si algún EV > 0, ejecutar `kelly_multi_outcome` para obtener stakes
   óptimos sujetos a `MAX_KELLY_STAKE`.
4. Si la posterior de `p` está disponible, ejecutar `kelly_robust_binary`
   para ajustar por incertidumbre epistémica.
5. Verificar fracción contra `optimal_fractional_kelly` con umbral de
   P(ruina) ≤ 1%.
6. Si todos los EV ≤ 0: **NO BET — EV NEGATIVO**.

---

## 4. FORMATO DE RESPUESTA

```markdown
# REPORTE DE PREDICCIÓN FUOL: [Local] vs [Visitante]

## 1. PARÁMETROS BASE
- λ local: [valor]  |  μ visitante: [valor]
- Probabilidades 1X2: 1=[X%]  X=[Y%]  2=[Z%]
- ρ (Dixon-Coles): [valor]
- half_life (LOO-CV): [días]

## 2. TOP-5 MARCADORES EXACTOS
1. [marcador] — [probabilidad]
...

## 3. VALIDACIÓN WALK-FORWARD
- N eval: [n]
- Brier: [valor]  |  Log-Loss: [valor]  |  RPS: [valor]
- Hit rate: [%]  |  ECE: [valor]

## 4. RIESGO Y VALOR (KELLY)
- Mercado: [1X2 | Over/Under | BTTS]
- Cuotas: [o1, oX, o2]
- EV por resultado: [ev1, evX, ev2]
- Stake recomendado (Kelly multi): [s1, sX, s2]
- Fracción de Kelly (P(ruina) ≤ 1%): [%]
- Veredicto: [PLAY | NO PLAY]
```

---

## 5. LEYES INMUTABLES

1. **El modelo es soberano**: si el cálculo dicta EV+, se recomienda; si
   dicta EV-, no se apuesta. Sin emociones.
2. **Cero garantías**: jamás "seguro" o "garantizado". Hablar en términos
   de distribución de probabilidad y valor esperado.
3. **Walk-forward o nada**: una métrica sin validación walk-forward con
   embargo temporal no es una métrica, es sobreajuste.
4. **Robustez > optimismo**: cuando haya incertidumbre en `p`, usar
   cuantil inferior, no la media plug-in.

---

## 6. MÓDULOS DECLARADOS PERO NO IMPLEMENTADOS

Los siguientes nombres aparecen en versiones anteriores de la skill pero
**NO existen en el repositorio**. Cualquier referencia a ellos debe
tratarse como código por escribir, no como funcionalidad disponible:

- `unified_engine_v3.py` / `UnifiedEngineV3`
- `hawkes_goals_process.py` / `HawkesGoalsProcess`
- `quantum_match_state.py`
- `dirichlet_process_mixture.py`
- `integrate_v3.py`
- `calibrate_v3.py`

Si se quiere extender el sistema con estos modelos, crear el módulo y
agregar tests antes de referenciarlo desde la skill.
