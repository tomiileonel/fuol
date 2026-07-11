# FUOL (Football Unified Optimization Layer)

FUOL es un sistema cuantitativo hiper-optimizado para predicción deportiva y *trading* algorítmico en fútbol. Integra un motor estadístico bayesiano basado en Dixon-Coles, un pipeline de datos *Walk-Forward* purgado, y una gestión de riesgo avanzada con el Criterio de Kelly (Ajuste Robusto).

El motor compite a niveles de precisión equiparables a los modelos comerciales de las casas de apuestas profesionales (RPS < 0.20 en fútbol internacional de élite).

## 1. Filosofía y Política del Proyecto
Este proyecto sigue una estricta **filosofía de 0% inversión**: no se implementarán flujos que requieran dinero real, pagos de APIs comerciales ni operaciones financieras. Funciona exclusivamente como un simulador de *paper trading* y detector de *value* estadístico utilizando fuentes gratuitas y públicas.

---

## 2. Arquitectura Matemática

### 2.1. Motor Estadístico (UnifiedEngine & Dixon-Coles)
El corazón de FUOL utiliza una adaptación avanzada del modelo de **Dixon-Coles (1997)**:
- **Base Poisson**: Estima la tasa de goles esperados ($\lambda$ para el local, $\mu$ para la visita) asumiendo independencia, basándose en la jerarquía (diferencia de Elo) acumulada durante los últimos 4 años.
- **Ajuste Rho ($\rho$)**: Corrige la limitación clásica de Poisson inyectando dependencia en marcadores de baja anotación (0-0, 1-1, 1-0, 0-1), vital para predecir empates en torneos eliminatorios cerrados (ej. Mundiales).
- **Time Decay**: Incorpora el decaimiento temporal para calcular probabilidades cronológicas *In-Play* minuto a minuto basándose en los goles esperados restantes.

### 2.2. Gestión de Riesgo (Kelly Robusto)
El módulo `kelly_risk_engine.py` no usa la versión ingenua (plug-in) del criterio de Kelly. Para proteger el bankroll contra la **incertidumbre epistémica** (el ancho del Intervalo de Confianza del 90% para $\lambda$ y $\mu$), el sistema utiliza un **Kelly Robusto** tomando el *cuantil 25* de la distribución posterior. 
Esto garantiza que FUOL devuelva `NO BET` (0% stake) cuando el Valor Esperado (EV) es marginalmente positivo pero matemáticamente inseguro.

---

## 3. Validación y Pipeline de Backtesting

### 3.1. Walk-Forward Pipeline con Purge & Embargo
Para prevenir cualquier fuga de datos (*data leakage*) o autocorrelación, el sistema evalúa los modelos usando `WalkForwardPipeline`:
- **Training Window**: 4 años (1460 días).
- **Test Window**: 1 mes (30 días).
- Recreación estricta de variables en cada pliegue (*fold*) temporal sin sesgo retrospectivo (*look-ahead bias*).

### 3.2. Calibración (Optuna) y Bootstrap Pareado
Los hiperparámetros del sistema fueron encontrados usando optimización bayesiana y validados mediante un Bootstrap Pareado ($B=10,000$ resamples) para garantizar la significancia estadística frente al modelo base:
- `lambda_scale = 0.4031`
- `prior_strength = 7.21`
- `half_life = 365.0` días

### 3.3. Métricas Oficiales (Backtest 2020-2026)
Evaluación fuera de muestra (*out-of-sample*) sobre 6.5 años (6118 partidos internacionales):
- **Hit Rate (1X2)**: 57.8%
- **Brier Score**: 0.5403
- **Ranked Probability Score (RPS)**: **0.1821** *(El RPS < 0.20 certifica la viabilidad profesional del modelo)*

---

## 4. Herramientas CLI (Inferencia en vivo)

El repositorio incluye herramientas por línea de comandos para utilizar el motor calibrado en tiempo real.

### Predicción de Partidos Pre-Match
Predice un partido construyendo dinámicamente el *history_cache* de los últimos 4 años.
```bash
# Para torneos en cancha neutral (ej. Mundial)
python predict_match.py --home Argentina --away Spain --neutral

# Para partidos con localía tradicional
python predict_match.py --home England --away France
```

### Dinámica In-Play (Time Decay)
Calcula cómo las probabilidades evolucionan con el tiempo asumiendo un empate parcial, mostrando el acantilado estadístico de los minutos 45 y 60.
```bash
python live_probabilities.py
```

---

## 5. Instalación y Uso de la API

### Requisitos
- Python 3.10+
- (Opcional) MongoDB para persistencia del Paper Trader.

### Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q  # Correr tests automatizados (33/33)
```

### Levantar API / Web Server
```bash
python api_server.py
```

---

## 6. Limitaciones Conocidas (Próximas Iteraciones)
A pesar de su calibración competitiva, la versión actual de FUOL no modela:
- **Factor Élite Individual (Cola Pesada)**: El modelo subestima la varianza ofensiva cuando participan delanteros generacionales (ej. Haaland, Mbappé, Messi) capaces de quebrar la independencia de Poisson.
- **xG (Goles Esperados Reales)**: Se basa exclusivamente en resultados históricos y Elo, no procesa topología de campo ni métricas avanzadas (StatsBomb).
- **Alineaciones y Lesiones**: Trata al equipo como un ente unificado, sin impactar el $\lambda$ ante la ausencia de figuras clave de último minuto.
