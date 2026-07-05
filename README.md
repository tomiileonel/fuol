# FUOL

FUOL (Football Unified Optimization Layer) es un sistema cuantitativo para predicción deportiva y trading algorítmico en fútbol. Integra un motor estadístico basado en Dixon-Coles, un pipeline de datos, gestión de riesgo con Kelly y una API para consultar predicciones y ejecutar paper trading.

## Política del proyecto

Este proyecto sigue una filosofía de 0% inversión: no se implementarán flujos que requieran dinero real, pagos de APIs comerciales ni operaciones financieras reales. Cuando se integre datos externos, solo se usarán fuentes gratuitas, públicas o de demostración.

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
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

## Variables de entorno

Copie el archivo [.env.example](.env.example) a .env y ajuste los valores según su entorno.

Para datos externos gratuitos, puede definir:

```bash
EXTERNAL_DATA_URL=https://example.com/fixtures
EXTERNAL_DATA_API_KEY=opcional
```

```bash
copy .env.example .env
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

- [api_server.py](api_server.py): API FastAPI.
- [kelly_risk_engine.py](kelly_risk_engine.py): motor de riesgo y Kelly.
- [unified_engine.py](unified_engine.py): motor estadístico principal.
- [data_pipeline.py](data_pipeline.py): enriquecimiento de datos.
- [paper_trader.py](paper_trader.py): paper trading.
