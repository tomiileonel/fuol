from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DirichletProcessResult:
    n_clusters: int
    cluster_sizes: list[int]
    cluster_means: list[float]
    global_mean: float
    clusters: dict[int, dict[str, Any]]
    team_assignments: dict[str, int]


class DirichletProcessMixture:
    """Implementación mínima de DP mixture para agrupar tasas de goles por confederación."""

    def __init__(self, alpha: float = 1.0, n_iterations: int = 200, min_cluster_size: int = 1) -> None:
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.min_cluster_size = min_cluster_size

    def fit(self, values: Any, n_iterations: int | None = None, burn_in: int = 0) -> DirichletProcessResult:
        if isinstance(values, dict):
            team_names = list(values.keys())
            points = np.asarray([values[name] for name in team_names], dtype=float)
        else:
            team_names = [str(i) for i in range(len(values))]
            points = np.asarray(values, dtype=float)

        if points.ndim == 2:
            points = points[:, 0]

        x = np.asarray(points, dtype=float)
        if x.size == 0:
            return DirichletProcessResult(1, [0], [0.0], 0.0, {0: {'attack_mean': 0.0, 'defense_mean': 0.0, 'n_members': 0, 'member_teams': []}}, {})

        n_iter = self.n_iterations if n_iterations is None else n_iterations
        rng = np.random.default_rng(42)
        assignments = rng.integers(0, 2, size=len(x)).tolist()
        cluster_means = []
        for _ in range(n_iter):
            for i in range(len(x)):
                current = assignments[i]
                counts = {}
                for j, c in enumerate(assignments):
                    if j != i:
                        counts[c] = counts.get(c, 0) + 1
                if current in counts:
                    counts[current] -= 1
                if counts.get(current, 0) <= 0:
                    assignments[i] = max(assignments) + 1
                else:
                    if rng.random() < self.alpha / (self.alpha + len(x)):
                        assignments[i] = max(assignments) + 1
                    else:
                        cluster_ids = list(counts.keys())
                        probs = np.array([counts[c] for c in cluster_ids], dtype=float)
                        probs = probs / probs.sum()
                        assignments[i] = cluster_ids[int(rng.choice(len(cluster_ids), p=probs))]

            unique_clusters = sorted(set(assignments))
            cluster_means = [float(x[[j for j, c in enumerate(assignments) if c == cluster]].mean()) for cluster in unique_clusters]

        cluster_sizes = [sum(1 for c in assignments if c == cluster) for cluster in sorted(set(assignments))]
        clusters = {}
        team_assignments = {}
        for cluster_id, cluster_label in enumerate(sorted(set(assignments))):
            members = [team_names[j] for j, c in enumerate(assignments) if c == cluster_label]
            cluster_members = [team_names[j] for j, c in enumerate(assignments) if c == cluster_label]
            if len(cluster_members) < self.min_cluster_size:
                continue
            cluster_values = x[[j for j, c in enumerate(assignments) if c == cluster_label]]
            clusters[cluster_id] = {
                'attack_mean': float(cluster_values.mean()),
                'defense_mean': float(cluster_values.mean()),
                'n_members': len(cluster_members),
                'member_teams': cluster_members,
            }
            for member in cluster_members:
                team_assignments[member] = cluster_id

        if not clusters:
            clusters[0] = {'attack_mean': float(x.mean()), 'defense_mean': float(x.mean()), 'n_members': len(x), 'member_teams': team_names}
            for team_name in team_names:
                team_assignments[team_name] = 0

        return DirichletProcessResult(
            n_clusters=len(clusters),
            cluster_sizes=cluster_sizes,
            cluster_means=cluster_means,
            global_mean=float(x.mean()),
            clusters=clusters,
            team_assignments=team_assignments,
        )
