from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchPhase:
    start_minute: float
    end_minute: float
    description: str
    goal_rates: tuple[float, float]
    transition_matrix: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class LivePrediction:
    current_state: str
    minutes_elapsed: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals_remaining: float
    expected_away_goals_remaining: float


class ContinuousTimeMarkovChain:
    STATES = [
        'balanced',
        'home_dominant',
        'away_dominant',
        'home_leads_1',
        'away_leads_1',
        'home_leads_2',
        'away_leads_2',
    ]

    def __init__(self) -> None:
        self.n_states = len(self.STATES)
        self.phases: list[MatchPhase] = []
        self._calibrated = False

    def calibrate_from_matches(self, matches: list[dict], n_phases: int = 5) -> None:
        if not matches:
            self._calibrate_default()
            return

        all_goal_minutes = []
        for m in matches:
            goals = m.get('goals_by_minute', [])
            all_goal_minutes.extend(goals)

        if not all_goal_minutes:
            self._calibrate_default()
            return

        phase_boundaries = np.linspace(0, 90, n_phases + 1)
        self.phases = []

        for i in range(n_phases):
            start = phase_boundaries[i]
            end = phase_boundaries[i + 1]
            goals_in_phase = [g for g in all_goal_minutes if start <= g < end]
            n_goals = len(goals_in_phase)
            duration = end - start
            rate = n_goals / (duration * max(1, len(matches))) if len(matches) > 0 else 0.015

            Q = np.zeros((self.n_states, self.n_states))
            Q[0, 1] = rate * 0.6
            Q[0, 2] = rate * 0.4
            Q[1, 3] = rate * 1.2
            Q[2, 4] = rate * 1.2
            Q[3, 5] = rate * 0.8
            Q[4, 6] = rate * 0.8
            for j in range(self.n_states):
                Q[j, j] = -Q[j, :].sum()

            self.phases.append(MatchPhase(
                start_minute=start,
                end_minute=end,
                description=f'Phase {i+1} ({start:.0f}-{end:.0f} min)',
                goal_rates=(rate, rate),
                transition_matrix=Q,
            ))

        self._calibrated = True

    def _calibrate_default(self) -> None:
        default_rates = [
            (0.012, 0.012),
            (0.015, 0.013),
            (0.018, 0.016),
            (0.016, 0.017),
            (0.020, 0.018),
        ]
        phase_boundaries = [0, 15, 35, 60, 75, 90]
        self.phases = []

        for i in range(5):
            start = phase_boundaries[i]
            end = phase_boundaries[i + 1]
            home_rate, away_rate = default_rates[i]
            Q = np.zeros((self.n_states, self.n_states))
            Q[0, 1] = home_rate * 0.6
            Q[0, 2] = away_rate * 0.4
            Q[1, 3] = home_rate * 1.2
            Q[2, 4] = away_rate * 1.2
            Q[3, 5] = home_rate * 0.8
            Q[4, 6] = away_rate * 0.8
            for j in range(self.n_states):
                Q[j, j] = -Q[j, :].sum()

            self.phases.append(MatchPhase(
                start_minute=start,
                end_minute=end,
                description=f'Phase {i+1} ({start:.0f}-{end:.0f} min)',
                goal_rates=(home_rate, away_rate),
                transition_matrix=Q,
            ))
        self._calibrated = True

    def predict_live(self, home_goals: int, away_goals: int, minute_elapsed: float) -> LivePrediction:
        if not self._calibrated:
            self._calibrate_default()

        if home_goals == away_goals:
            current_state = 'balanced'
        elif home_goals == away_goals + 1:
            current_state = 'home_leads_1'
        elif away_goals == home_goals + 1:
            current_state = 'away_leads_1'
        elif home_goals >= away_goals + 2:
            current_state = 'home_leads_2'
        elif away_goals >= home_goals + 2:
            current_state = 'away_leads_2'
        elif home_goals > away_goals:
            current_state = 'home_dominant'
        else:
            current_state = 'away_dominant'

        time_remaining = max(0, 90 - minute_elapsed)
        if time_remaining == 0:
            if home_goals > away_goals:
                return LivePrediction(current_state, minute_elapsed, 1.0, 0.0, 0.0, 0.0, 0.0)
            if home_goals == away_goals:
                return LivePrediction(current_state, minute_elapsed, 0.0, 1.0, 0.0, 0.0, 0.0)
            return LivePrediction(current_state, minute_elapsed, 0.0, 0.0, 1.0, 0.0, 0.0)

        remaining_phases = [p for p in self.phases if p.end_minute > minute_elapsed]
        if not remaining_phases:
            remaining_phases = [self.phases[-1]]

        expected_home_remaining = 0.0
        expected_away_remaining = 0.0
        for phase in remaining_phases:
            phase_start = max(phase.start_minute, minute_elapsed)
            phase_end = phase.end_minute
            duration = phase_end - phase_start
            if duration > 0:
                home_rate, away_rate = phase.goal_rates
                expected_home_remaining += home_rate * duration
                expected_away_remaining += away_rate * duration

        n_simulations = 1000
        rng = np.random.default_rng(42)
        home_wins = draws = away_wins = 0
        for _ in range(n_simulations):
            h_remaining = rng.poisson(expected_home_remaining)
            a_remaining = rng.poisson(expected_away_remaining)
            final_home = home_goals + h_remaining
            final_away = away_goals + a_remaining
            if final_home > final_away:
                home_wins += 1
            elif final_home == final_away:
                draws += 1
            else:
                away_wins += 1

        return LivePrediction(
            current_state=current_state,
            minutes_elapsed=minute_elapsed,
            p_home_win=home_wins / n_simulations,
            p_draw=draws / n_simulations,
            p_away_win=away_wins / n_simulations,
            expected_home_goals_remaining=expected_home_remaining,
            expected_away_goals_remaining=expected_away_remaining,
        )
