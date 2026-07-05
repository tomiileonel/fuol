from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NetworkMetrics:
    betweenness_mean: float
    betweenness_max: float
    clustering_mean: float
    clustering_std: float
    eigenvector_centrality_max: float
    assortativity: float
    algebraic_connectivity: float
    density: float
    avg_path_length: float

    def as_feature_vector(self) -> NDArray[np.floating]:
        return np.array([
            self.betweenness_mean,
            self.betweenness_max,
            self.clustering_mean,
            self.clustering_std,
            self.eigenvector_centrality_max,
            self.assortativity,
            self.algebraic_connectivity,
            self.density,
            self.avg_path_length,
        ], dtype=float)


class PassNetworkAnalyzer:
    def __init__(self, n_players: int = 11) -> None:
        self._n = n_players

    def build_adjacency(self, pass_matrix: NDArray[np.floating], symmetrize: bool = True) -> NDArray[np.floating]:
        A = np.asarray(pass_matrix, dtype=float)
        if symmetrize:
            A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        return A

    def betweenness_centrality(self, A: NDArray[np.floating]) -> NDArray[np.floating]:
        n = A.shape[0]
        binary = (A > 0).astype(int)
        cb = np.zeros(n, dtype=float)
        for s in range(n):
            stack: list[int] = []
            pred: list[list[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=float)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=int)
            dist[s] = 0
            queue = [s]
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in range(n):
                    if binary[v, w]:
                        if dist[w] < 0:
                            dist[w] = dist[v] + 1
                            queue.append(w)
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
            delta = np.zeros(n, dtype=float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        cb /= 2.0
        if n > 2:
            cb /= ((n - 1) * (n - 2) / 2.0)
        return cb

    def clustering_coefficients(self, A: NDArray[np.floating]) -> NDArray[np.floating]:
        n = A.shape[0]
        binary = (A > 0).astype(int)
        degrees = binary.sum(axis=1)
        cc = np.zeros(n, dtype=float)
        A2 = binary @ binary
        triangles = (A2 * binary).sum(axis=1) / 2.0
        for v in range(n):
            k = degrees[v]
            if k >= 2:
                cc[v] = (2.0 * triangles[v]) / (k * (k - 1))
        return cc

    def eigenvector_centrality(self, A: NDArray[np.floating], max_iter: int = 100, tol: float = 1e-8) -> NDArray[np.floating]:
        n = A.shape[0]
        x = np.ones(n, dtype=float) / np.sqrt(n)
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-15:
                return np.zeros(n)
            x_new /= norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x

    def algebraic_connectivity(self, A: NDArray[np.floating]) -> float:
        binary = (A > 0).astype(float)
        degrees = binary.sum(axis=1)
        L = np.diag(degrees) - binary
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()
        return float(eigvals[1]) if len(eigvals) > 1 else 0.0

    def assortativity(self, A: NDArray[np.floating]) -> float:
        binary = (A > 0).astype(int)
        degrees = binary.sum(axis=1)
        edges: list[tuple[int, int]] = []
        for i in range(len(degrees)):
            for j in range(i + 1, len(degrees)):
                if binary[i, j]:
                    edges.append((degrees[i], degrees[j]))
        if not edges:
            return 0.0
        jk = np.array([j * k for j, k in edges])
        j_plus_k = np.array([j + k for j, k in edges])
        j2_plus_k2 = np.array([j**2 + k**2 for j, k in edges])
        M = len(edges)
        mean_jk = jk.mean()
        mean_j_plus_k = j_plus_k.mean() / 2.0
        mean_j2_plus_k2 = j2_plus_k2.mean() / 2.0
        numerator = mean_jk - mean_j_plus_k ** 2
        denominator = mean_j2_plus_k2 - mean_j_plus_k ** 2
        return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0

    def analyze(self, pass_matrix: NDArray[np.floating]) -> NetworkMetrics:
        A = self.build_adjacency(pass_matrix, symmetrize=True)
        betweenness = self.betweenness_centrality(A)
        clustering = self.clustering_coefficients(A)
        eigenvector = self.eigenvector_centrality(A)
        fiedler = self.algebraic_connectivity(A)
        assort = self.assortativity(A)
        binary = (A > 0).astype(int)
        n = A.shape[0]
        density = binary.sum() / (n * (n - 1)) if n > 1 else 0.0
        avg_path = self._average_path_length(binary)
        return NetworkMetrics(
            betweenness_mean=float(betweenness.mean()),
            betweenness_max=float(betweenness.max()),
            clustering_mean=float(clustering.mean()),
            clustering_std=float(clustering.std()),
            eigenvector_centrality_max=float(eigenvector.max()),
            assortativity=assort,
            algebraic_connectivity=fiedler,
            density=float(density),
            avg_path_length=avg_path,
        )

    @staticmethod
    def _average_path_length(binary: NDArray[np.integer]) -> float:
        n = binary.shape[0]
        total_dist = 0
        count = 0
        for s in range(n):
            dist = np.full(n, -1, dtype=int)
            dist[s] = 0
            queue = [s]
            while queue:
                v = queue.pop(0)
                for w in range(n):
                    if binary[v, w] and dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
            finite = dist[(dist > 0)]
            total_dist += finite.sum()
            count += len(finite)
        return float(total_dist / count) if count > 0 else 0.0
