from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import NamedTuple


class MatchOutcome(NamedTuple):
    home: float
    draw: float
    away: float

    def as_array(self) -> NDArray[np.floating]:
        return np.array([self.home, self.draw, self.away], dtype=float)

    def __iter__(self):
        return iter((self.home, self.draw, self.away))


@dataclass(frozen=True, slots=True)
class QuantumAmplitudes:
    amp_home: complex
    amp_draw: complex
    amp_away: complex

    @classmethod
    def from_probabilities_and_phases(
        cls,
        probs: MatchOutcome,
        phases: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "QuantumAmplitudes":
        arr = probs.as_array()
        arr = np.clip(arr, 1e-12, None)
        arr /= arr.sum()

        r = np.sqrt(arr)
        phi_h, phi_d, phi_a = phases
        return cls(
            amp_home=r[0] * np.exp(1j * phi_h),
            amp_draw=r[1] * np.exp(1j * phi_d),
            amp_away=r[2] * np.exp(1j * phi_a),
        )

    def probabilities(self) -> MatchOutcome:
        p_h = abs(self.amp_home) ** 2
        p_d = abs(self.amp_draw) ** 2
        p_a = abs(self.amp_away) ** 2
        total = p_h + p_d + p_a
        return MatchOutcome(p_h / total, p_d / total, p_a / total)

    def coherence_measure(self) -> float:
        num = abs(
            self.amp_home.conjugate() * self.amp_draw
            + self.amp_draw.conjugate() * self.amp_away
            + self.amp_away.conjugate() * self.amp_home
        )
        den = abs(self.amp_home) ** 2 + abs(self.amp_draw) ** 2 + abs(self.amp_away) ** 2
        return float(num / den) if den > 1e-12 else 0.0


@dataclass(slots=True)
class QuantumMatchState:
    state: QuantumAmplitudes
    evidence_log: list[str] = field(default_factory=list)

    @staticmethod
    def _rotation_matrix(axis: int, angle: float) -> NDArray[np.complexfloating]:
        R = np.eye(3, dtype=complex)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            R[0, 0] = c
            R[0, 1] = -s
            R[1, 0] = s
            R[1, 1] = c
        elif axis == 1:
            R[1, 1] = c
            R[1, 2] = -s
            R[2, 1] = s
            R[2, 2] = c
        elif axis == 2:
            R[2, 2] = c
            R[2, 0] = -s
            R[0, 2] = s
            R[0, 0] = c
        return R

    def apply_evidence(self, axis: int, strength: float, description: str) -> "QuantumMatchState":
        clipped = np.clip(strength, -np.pi / 2, np.pi / 2)
        R = self._rotation_matrix(axis, clipped)
        psi = np.array([self.state.amp_home, self.state.amp_draw, self.state.amp_away], dtype=complex)
        psi_new = R @ psi
        new_state = QuantumAmplitudes(amp_home=complex(psi_new[0]), amp_draw=complex(psi_new[1]), amp_away=complex(psi_new[2]))
        return QuantumMatchState(state=new_state, evidence_log=[*self.evidence_log, f"{description} (θ={clipped:.3f})"])

    def apply_home_advantage(self, strength: float = 0.25) -> "QuantumMatchState":
        return self.apply_evidence(axis=2, strength=strength, description="Home advantage")

    def apply_injury_impact(self, team: str, severity: float) -> "QuantumMatchState":
        axis = 2
        direction = -severity if team == "home" else severity
        return self.apply_evidence(axis=axis, strength=direction * 0.5, description=f"Injury impact ({team})")

    def apply_draw_tendency(self, strength: float) -> "QuantumMatchState":
        s1 = self.apply_evidence(axis=0, strength=strength * 0.5, description="Draw tendency (H-D)")
        return s1.apply_evidence(axis=1, strength=strength * 0.5, description="Draw tendency (D-A)")

    def collapse(self) -> MatchOutcome:
        return self.state.probabilities()

    def diagnosis(self) -> dict:
        probs = self.collapse()
        coherence = self.state.coherence_measure()
        return {
            "probabilities": {
                "p_home": round(probs.home, 4),
                "p_draw": round(probs.draw, 4),
                "p_away": round(probs.away, 4),
            },
            "quantum_coherence": round(coherence, 4),
            "interpretation": self._interpret(coherence, probs),
            "evidence_count": len(self.evidence_log),
            "evidence_log": self.evidence_log,
        }

    @staticmethod
    def _interpret(coherence: float, probs: MatchOutcome) -> str:
        if coherence < 0.15:
            return "Estado DECOHERENTE: señales contradictorias (modelo incierto)."
        if coherence > 0.7:
            dominant = max(("home", probs.home), ("draw", probs.draw), ("away", probs.away), key=lambda x: x[1])
            return f"Estado COHERENTE: señal clara a favor de {dominant[0]}"
        return "Estado INTERMEDIO: coherencia parcial."
