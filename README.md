# FUOL

FUOL (Football Unified Optimization Layer) es un sistema cuantitativo para predicción deportiva y trading algorítmico en fútbol. Integra un motor estadístico basado en Dixon-Coles, un pipeline de datos, gestión de riesgo con Kelly y una API para consultar predicciones y ejecutar paper trading.

## Política del proyecto

Este proyecto sigue una filosofía de 0% inversión: no se implementarán flujos que requieran dinero real, pagos de APIs comerciales ni operaciones financieras reales. Cuando se integre datos externos, solo se usarán fuentes gratuitas, públicas o de demostración.

## Rendimiento y Validación (Walk-Forward)

El motor ha sido calibrado mediante optimización bayesiana (Optuna) y validado estadísticamente con un bootstrap pareado frente al baseline. A continuación, el rendimiento en un riguroso backtest *walk-forward* (evaluando solo datos fuera de muestra sin look-ahead bias) para el período post-pandemia completo.

### Backtest 2020-2026 (6 Temporadas)
- **Partidos evaluados**: 6118
- **Hit Rate (1X2)**: 57.8%
- **Brier Score**: 0.5403
- **Ranked Probability Score (RPS)**: 0.1821

*Nota: Un RPS por debajo de 0.20 en fútbol internacional de élite (selecciones) indica que el motor se encuentra compitiendo a niveles de precisión equiparables a los modelos de casas de apuestas profesionales.*

### Validación estadística
Los hiperparámetros (`lambda_scale=0.4031`, `prior_strength=7.21`, `half_life=365`) fueron calibrados con Optuna y validados con bootstrap pareado (B=10000) contra el baseline sobre 989 partidos out-of-sample:

| Métrica | Baseline | Optimizado | Delta | IC 95% |
|---------|----------|------------|-------|--------|
| RPS Holdout | 0.1824 | 0.1765 | -3.23% | [0.00319, 0.00861] |

## Características principales

- Motor estadístico unificado para predicción 1X2.
- Pipeline de datos con enriquecimiento y extracción de contexto.
- Gestión de riesgo con Kelly multi-resultado y fraccionamiento ajustado por ruina.
- API FastAPI con frontend estático.
- Soporte para backtesting y validación temporal.

## Requisitos

- Python 3.10+
- MongoDB (opcional para persistencia, el sistema cuenta con fallbacks mock)

## Instalación

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install pytest
```

## Variables de entorno

No hay un archivo `.env.example` en la raíz. Las plantillas de configuración
viven en [`examples/`](examples/). Copie la que mejor se ajuste a su caso:

```bash
# Linux / macOS
cp examples/example_config.env .env

# Windows
copy examples\example_config.env .env
```

Para datos externos gratuitos, puede definir:

```bash
EXTERNAL_DATA_URL=https://example.com/fixtures
EXTERNAL_DATA_API_KEY=opcional
```

## Ejecución

Iniciar la API:

```bash
python api_server.py
```

Ejecutar pruebas:

```bash
pytest -q
```

## Estructura relevante

- [api_server.py](api_server.py): API FastAPI (sirve el frontend desde `frontend/`).
- [kelly_risk_engine.py](kelly_risk_engine.py): motor de riesgo y Kelly.
- [unified_engine.py](unified_engine.py): motor estadístico principal.
- [data_pipeline.py](data_pipeline.py): enriquecimiento de datos.
- [paper_trader.py](paper_trader.py): paper trading.
- `frontend/`: UI web principal (servida por `api_server.py`).
- `static/`: dashboard alternativo "360°" (no montado por defecto; copiar a `frontend/` si se quiere usar).
