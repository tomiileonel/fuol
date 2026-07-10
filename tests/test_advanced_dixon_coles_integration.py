import pytest
import numpy as np
from advanced_dixon_coles import AdvancedDixonColes

def test_rho_limits_no_negative_probabilities():
    """
    Verifica matemáticamente que score_matrix nunca arroje probabilidades negativas
    en las celdas, incluso si rho toca límites teóricos (-1, 1).
    La función np.clip o la validación estricta de dominios deben impedir masa negativa.
    """
    lam = 1.5
    mu = 1.0
    
    # Test bounds of rho
    rho_candidates = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for rho in rho_candidates:
        matrix = AdvancedDixonColes.score_matrix(lam, mu, rho, max_goals=5)
        
        # 1. No negative probabilities
        assert np.all(matrix >= 0), f"Negative probability found for rho={rho}"
        
        # 2. Sums to 1.0 (valid PMF)
        assert np.isclose(matrix.sum(), 1.0, atol=1e-5), f"Matrix sum is {matrix.sum()} (not 1.0) for rho={rho}"
        
def test_extract_1x2():
    lam = 1.5
    mu = 1.0
    rho = -0.1
    matrix = AdvancedDixonColes.score_matrix(lam, mu, rho, max_goals=10)
    
    p1, px, p2 = AdvancedDixonColes.extract_1x2(matrix)
    
    assert p1 >= 0 and px >= 0 and p2 >= 0
    assert np.isclose(p1 + px + p2, 1.0, atol=1e-5)
