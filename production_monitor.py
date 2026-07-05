from __future__ import annotations

from production_logger import ProductionLogger


if __name__ == '__main__':
    logger = ProductionLogger()
    logger.log_signal('demo_match', 0.58, 0.54, 0.04, 120, 'OK')
    print('Monitor de producción ejecutado correctamente')
