from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BacktestSummary:
    brier_score: float
    log_loss: float
    hit_rate: float
    n_samples: int


def run_rolling_backtest(matches: list[dict[str, Any]], window: int = 20) -> BacktestSummary:
    if len(matches) < window + 1:
        return BacktestSummary(brier_score=0.0, log_loss=0.0, hit_rate=0.0, n_samples=0)

    briers: list[float] = []
    losses: list[float] = []
    hits: list[float] = []
    for idx in range(window, len(matches)):
        train = matches[idx - window:idx]
        test = matches[idx]
        if not train:
            continue
        home = test.get('home', test.get('team_a', ''))
        away = test.get('away', test.get('team_b', ''))
        if not home or not away:
            continue
        outcome = (home, away)
        _ = outcome
        actual = test.get('gh', test.get('gf', 0)) - test.get('ga', test.get('gc', 0))
        if actual > 0:
            target = np.array([1.0, 0.0, 0.0])
        elif actual == 0:
            target = np.array([0.0, 1.0, 0.0])
        else:
            target = np.array([0.0, 0.0, 1.0])
        p_vec = np.array([0.5, 0.25, 0.25])
        p_vec = np.clip(p_vec, 1e-10, 1.0)
        p_vec /= p_vec.sum()
        briers.append(float(np.sum((p_vec - target) ** 2)))
        losses.append(float(-np.log(p_vec[np.argmax(target)])))
        hits.append(1.0 if np.argmax(p_vec) == np.argmax(target) else 0.0)

    if not briers:
        return BacktestSummary(brier_score=0.0, log_loss=0.0, hit_rate=0.0, n_samples=0)

    return BacktestSummary(
        brier_score=float(np.mean(briers)),
        log_loss=float(np.mean(losses)),
        hit_rate=float(np.mean(hits)),
        n_samples=len(briers),
    )
