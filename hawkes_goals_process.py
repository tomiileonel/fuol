from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class GoalSequence(Protocol):
    def minutes(self) -> NDArray[np.floating]: ...


@dataclass(frozen=True, slots=True)
class HawkesParameters:
    mu: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 1e-9 else 0.0

    @property
    def expected_goals_90min(self) -> float:
        n = self.branching_ratio
        if n >= 1.0 or n < 0:
            return self.mu * 90.0
        return (self.mu * 90.0) / (1.0 - n)


class HawkesGoalsProcess:
    BOUNDS: tuple[tuple[float, float], ...] = (
        (1e-6, 0.15),
        (1e-6, 0.08),
        (0.01, 2.0),
    )

    def __init__(self, match_duration_min: float = 90.0) -> None:
        self._T = match_duration_min

    @staticmethod
    def intensity(
        t: float | NDArray[np.floating],
        events: NDArray[np.floating],
        params: HawkesParameters,
    ) -> NDArray[np.floating]:
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        mu, alpha, beta = params.mu, params.alpha, params.beta
        dt = t_arr[:, None] - events[None, :]
        mask = dt > 0
        contributions = np.where(mask, alpha * np.exp(-beta * np.where(mask, dt, 0.0)), 0.0)
        return mu + contributions.sum(axis=1)

    def _neg_log_likelihood(self, theta: NDArray[np.floating], events: NDArray[np.floating]) -> float:
        mu, alpha, beta = theta
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e12

        params = HawkesParameters(mu=mu, alpha=alpha, beta=beta)
        if params.branching_ratio >= 0.99:
            return 1e12

        lambda_at_events = self.intensity(events, events, params)
        lambda_at_events = np.maximum(lambda_at_events, 1e-12)
        log_intensity_sum = np.log(lambda_at_events).sum()
        compensator = mu * self._T + (alpha / beta) * (1.0 - np.exp(-beta * (self._T - events))).sum()
        nll = -log_intensity_sum + compensator
        return float(nll) if np.isfinite(nll) else 1e12

    def fit(self, goal_minutes: list[float] | NDArray[np.floating]) -> HawkesParameters:
        events = np.asarray(goal_minutes, dtype=float)
        events = events[(events > 0) & (events <= self._T)]
        if len(events) < 2:
            return HawkesParameters(mu=0.0147, alpha=0.008, beta=0.35)

        events.sort()
        theta0 = np.array([0.015, 0.01, 0.3])
        result = optimize.minimize(
            fun=self._neg_log_likelihood,
            x0=theta0,
            args=(events,),
            method='L-BFGS-B',
            bounds=self.BOUNDS,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9},
        )

        if not result.success:
            n_goals = len(events)
            mu_hat = n_goals / self._T
            return HawkesParameters(mu=mu_hat, alpha=0.005, beta=0.3)

        mu, alpha, beta = result.x
        return HawkesParameters(mu=float(mu), alpha=float(alpha), beta=float(beta))

    def simulate(self, params: HawkesParameters, rng: np.random.Generator | None = None) -> NDArray[np.floating]:
        rng = rng or np.random.default_rng()
        events: list[float] = []
        t = 0.0
        lambda_star = params.mu + params.alpha * 10
        while t < self._T:
            u1 = rng.uniform()
            dt = -np.log(u1) / lambda_star
            t += dt
            if t >= self._T:
                break
            events_arr = np.array(events) if events else np.array([])
            current_lambda = self.intensity(np.array([t]), events_arr, params)[0]
            u2 = rng.uniform()
            if u2 <= current_lambda / lambda_star:
                events.append(t)
        return np.array(events, dtype=float)

    def goal_distribution(self, params: HawkesParameters, max_goals: int = 8, n_simulations: int = 10_000) -> NDArray[np.floating]:
        counts = np.zeros(max_goals + 1, dtype=float)
        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            trajectory = self.simulate(params, rng)
            n_goals = min(len(trajectory), max_goals)
            counts[n_goals] += 1
        return counts / n_simulations
