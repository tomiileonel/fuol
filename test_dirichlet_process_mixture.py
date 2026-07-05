from dirichlet_process_mixture import DirichletProcessMixture


def test_dp_mixture_accepts_team_stats_and_returns_clusters():
    model = DirichletProcessMixture(alpha=1.0, min_cluster_size=2)
    team_stats = {
        'Team A': (1.8, 1.2),
        'Team B': (1.6, 1.3),
        'Team C': (0.7, 1.9),
        'Team D': (0.8, 1.8),
    }

    result = model.fit(team_stats, n_iterations=20, burn_in=5)

    assert result.n_clusters >= 1
    assert len(result.team_assignments) == len(team_stats)
    assert hasattr(result, 'clusters')
    assert all(isinstance(cluster, dict) for cluster in result.clusters.values())
