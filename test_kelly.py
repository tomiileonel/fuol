import unittest

def calculate_fractional_kelly(market_prob, engine_prob, brier_score, fractional_scale=0.10):
    if market_prob <= 0 or engine_prob <= 0:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    p = engine_prob
    q = 1.0 - p
    kelly_f = ((b * p) - q) / b if b > 0 else 0
    
    confidence_multiplier = max(0.5, 1.0 - (brier_score * 2))
    fractional_kelly = max(0.0, kelly_f * fractional_scale * confidence_multiplier)
    return fractional_kelly

class TestKellyCriterion(unittest.TestCase):
    def test_positive_alpha(self):
        # Market Prob: 40% (odds 2.5, b = 1.5)
        # Engine Prob: 50%
        # b = 1.5, p = 0.5, q = 0.5
        # kelly_f = ((1.5 * 0.5) - 0.5) / 1.5 = (0.75 - 0.5) / 1.5 = 0.25 / 1.5 = 0.1666...
        # brier_score = 0.0 (multiplier = 1.0)
        kf = calculate_fractional_kelly(0.40, 0.50, 0.0, 1.0)
        self.assertAlmostEqual(kf, 0.166666, places=5)
        
    def test_fractional_scale_and_confidence(self):
        # Same as above but 10% scale and 0.1 Brier score (0.8 multiplier)
        # expected: 0.166666 * 0.10 * 0.8 = 0.01333...
        kf = calculate_fractional_kelly(0.40, 0.50, 0.1, 0.10)
        self.assertAlmostEqual(kf, 0.013333, places=5)

    def test_negative_alpha(self):
        # Market Prob 60%
        # Engine Prob 50%
        # Should return 0
        kf = calculate_fractional_kelly(0.60, 0.50, 0.0, 1.0)
        self.assertEqual(kf, 0.0)

if __name__ == "__main__":
    unittest.main()
