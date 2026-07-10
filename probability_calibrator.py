import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Literal, Optional

class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities to empirical probabilities.
    Supports Platt Scaling (Logistic) and Isotonic Regression.
    """
    def __init__(self, method: Literal['platt', 'isotonic', 'none'] = 'platt'):
        self.method = method
        # We need separate calibrators for Home Win, Draw, Away Win
        self.calibrators = {
            'home': None,
            'draw': None,
            'away': None
        }
        
    def _create_calibrator(self):
        if self.method == 'platt':
            return LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        elif self.method == 'isotonic':
            return IsotonicRegression(out_of_bounds='clip')
        return None

    def fit(self, probs_matrix: np.ndarray, y_true: np.ndarray):
        """
        Fits calibrators.
        probs_matrix: shape (n_samples, 3) -> [P(Home), P(Draw), P(Away)]
        y_true: shape (n_samples,) -> 0 for Home, 1 for Draw, 2 for Away
        """
        if self.method == 'none':
            return
            
        # Home
        self.calibrators['home'] = self._create_calibrator()
        y_home = (y_true == 0).astype(int)
        if self.method == 'platt':
            self.calibrators['home'].fit(probs_matrix[:, 0].reshape(-1, 1), y_home)
        else:
            self.calibrators['home'].fit(probs_matrix[:, 0], y_home)
            
        # Draw
        self.calibrators['draw'] = self._create_calibrator()
        y_draw = (y_true == 1).astype(int)
        if self.method == 'platt':
            self.calibrators['draw'].fit(probs_matrix[:, 1].reshape(-1, 1), y_draw)
        else:
            self.calibrators['draw'].fit(probs_matrix[:, 1], y_draw)
            
        # Away
        self.calibrators['away'] = self._create_calibrator()
        y_away = (y_true == 2).astype(int)
        if self.method == 'platt':
            self.calibrators['away'].fit(probs_matrix[:, 2].reshape(-1, 1), y_away)
        else:
            self.calibrators['away'].fit(probs_matrix[:, 2], y_away)

    def predict_proba(self, probs_matrix: np.ndarray) -> np.ndarray:
        """
        Calibrates the input probabilities.
        Renormalizes to ensure they sum to 1.
        """
        if self.method == 'none':
            return probs_matrix
            
        calibrated = np.zeros_like(probs_matrix)
        
        # Home
        if self.method == 'platt':
            calibrated[:, 0] = self.calibrators['home'].predict_proba(probs_matrix[:, 0].reshape(-1, 1))[:, 1]
            calibrated[:, 1] = self.calibrators['draw'].predict_proba(probs_matrix[:, 1].reshape(-1, 1))[:, 1]
            calibrated[:, 2] = self.calibrators['away'].predict_proba(probs_matrix[:, 2].reshape(-1, 1))[:, 1]
        else:
            calibrated[:, 0] = self.calibrators['home'].predict(probs_matrix[:, 0])
            calibrated[:, 1] = self.calibrators['draw'].predict(probs_matrix[:, 1])
            calibrated[:, 2] = self.calibrators['away'].predict(probs_matrix[:, 2])
            
        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums
