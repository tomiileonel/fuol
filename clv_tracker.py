import pandas as pd
import numpy as np
from scipy import stats

class CLVTracker:
    """
    Closing Line Value (CLV) Tracker.
    Measures model predictions against Sharp Bookies (Pinnacle/Asianodds) closing lines.
    This is the ultimate benchmark for a quantitative betting model.
    """
    def __init__(self, historical_odds_csv: str):
        # The CSV should contain columns like: date, home, away, pin_H, pin_D, pin_A
        try:
            self.odds_df = pd.read_csv(historical_odds_csv)
            self.odds_df['date'] = pd.to_datetime(self.odds_df['date'])
        except FileNotFoundError:
            # Fallback if no odds file provided
            self.odds_df = pd.DataFrame(columns=['date', 'home', 'away', 'pin_H', 'pin_D', 'pin_A'])
            
    def _remove_vig(self, odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
        """
        Removes the bookmaker's margin (vig) using proportional normalization.
        """
        if pd.isna(odds_h) or pd.isna(odds_d) or pd.isna(odds_a) or odds_h == 0 or odds_d == 0 or odds_a == 0:
            return (0.0, 0.0, 0.0)
            
        implied_h = 1.0 / odds_h
        implied_d = 1.0 / odds_d
        implied_a = 1.0 / odds_a
        
        overround = implied_h + implied_d + implied_a
        if overround <= 1.0:
            return implied_h, implied_d, implied_a
            
        return implied_h / overround, implied_d / overround, implied_a / overround

    def calculate_clv(self, date, home, away, model_p1, model_px, model_p2) -> dict:
        """
        Calculates the CLV (Expected Value based on closing lines) for a specific match.
        If we find an edge (model > closing implied), we measure the theoretical EV.
        """
        if self.odds_df.empty:
            return {'status': 'no_data'}
            
        # Match finding logic
        date_mask = (self.odds_df['date'].dt.date == pd.to_datetime(date).date())
        match_row = self.odds_df[date_mask & (self.odds_df['home'] == home) & (self.odds_df['away'] == away)]
        
        if match_row.empty:
            return {'status': 'not_found'}
            
        row = match_row.iloc[0]
        
        # We enforce using Sharp books closing lines (Pinnacle prefix 'pin_')
        pin_h = row.get('pin_H', np.nan)
        pin_d = row.get('pin_D', np.nan)
        pin_a = row.get('pin_A', np.nan)
        
        sharp_p1, sharp_px, sharp_p2 = self._remove_vig(pin_h, pin_d, pin_a)
        
        if sharp_p1 == 0.0:
             return {'status': 'missing_odds'}
             
        # Calculate Edge (Model Prob - Sharp Prob)
        edge_1 = model_p1 - sharp_p1
        edge_x = model_px - sharp_px
        edge_2 = model_p2 - sharp_p2
        
        # Calculate EV if betting at market odds based on our probabilities
        ev_1 = (model_p1 * pin_h) - 1.0 if edge_1 > 0 else 0.0
        ev_x = (model_px * pin_d) - 1.0 if edge_x > 0 else 0.0
        ev_2 = (model_p2 * pin_a) - 1.0 if edge_2 > 0 else 0.0
        
        return {
            'status': 'success',
            'sharp_p1': sharp_p1,
            'sharp_px': sharp_px,
            'sharp_p2': sharp_p2,
            'ev_1': ev_1,
            'ev_x': ev_x,
            'ev_2': ev_2,
            'max_clv': max(ev_1, ev_x, ev_2)
        }

    def market_efficiency_test(self, clv_results: list[dict]) -> dict:
        """
        Statistical test to see if the model systematically beats the closing line.
        Null hypothesis: Model max_clv <= 0 (No edge over sharp books).
        """
        valid_clvs = [r['max_clv'] for r in clv_results if r['status'] == 'success' and r['max_clv'] > 0]
        
        if len(valid_clvs) < 30:
            return {'significant': False, 'message': 'Not enough data points for T-test (<30)', 'mean_clv': np.mean(valid_clvs) if valid_clvs else 0}
            
        t_stat, p_val = stats.ttest_1samp(valid_clvs, 0.0, alternative='greater')
        
        return {
            'mean_clv': float(np.mean(valid_clvs)),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
            'n_bets': len(valid_clvs)
        }
