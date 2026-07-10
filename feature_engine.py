import math
import numpy as np
import pandas as pd
from typing import Optional
import config

from config import (
    ELO_K_WC, ELO_K_QUALIF, ELO_K_FRIENDLY,
    ELO_HOME_ADV, ELO_SCALE
)

class EloRegistry:
    """
    Global registry for tracking dynamic Elo ratings for all teams.
    Incorporates Margin of Victory (MoV) multiplier.
    """
    def __init__(self, initial_elo: float = 1600.0, lambda_scale: float = 0.23):
        self.ratings: dict[str, float] = {}
        self.default_elo = initial_elo
        self.lambda_scale = lambda_scale

    def get_elo(self, team: str) -> float:
        return self.ratings.get(team, self.default_elo)

    def expected_score(self, rating_a: float, rating_b: float, venue: str = 'N') -> float:
        """P(win + 0.5*draw) for team A."""
        home_bonus = ELO_HOME_ADV if venue == 'H' else (-ELO_HOME_ADV if venue == 'A' else 0.0)
        diff = (rating_a + home_bonus) - rating_b
        return 1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE))

    def expected_goal_ratio(self, rating_a: float, rating_b: float, venue: str = 'N') -> float:
        """Converts Elo advantage into expected goal ratio."""
        E = self.expected_score(rating_a, rating_b, venue)
        E_clipped = np.clip(E, 0.05, 0.95)
        log_ratio = np.log(E_clipped / (1.0 - E_clipped)) * self.lambda_scale
        return np.exp(log_ratio)

    def get_k_factor(self, competition: str) -> float:
        comp_lower = str(competition).lower()
        if 'world cup' in comp_lower or 'wc ' in comp_lower:
            return ELO_K_WC
        elif 'qualifi' in comp_lower or 'copa america' in comp_lower or 'euro' in comp_lower or 'nations' in comp_lower:
            return ELO_K_QUALIF
        else:
            return ELO_K_FRIENDLY

    def update(self, team_a: str, team_b: str, goals_a: int, goals_b: int, competition: str, venue: str = 'N') -> tuple[float, float]:
        """
        Updates Elo for both teams based on match result.
        Returns the pre-match Elos for feature logging.
        """
        rating_a = self.get_elo(team_a)
        rating_b = self.get_elo(team_b)
        
        # Result for team A (1=Win, 0.5=Draw, 0=Loss)
        if goals_a > goals_b:
            s_a = 1.0
        elif goals_a == goals_b:
            s_a = 0.5
        else:
            s_a = 0.0
            
        s_b = 1.0 - s_a
        
        # Expected scores
        e_a = self.expected_score(rating_a, rating_b, venue)
        e_b = 1.0 - e_a
        
        # Margin of Victory multiplier
        goal_diff = abs(goals_a - goals_b)
        if goal_diff <= 1:
            mov = 1.0
        elif goal_diff == 2:
            mov = 1.5
        else:
            mov = (11.0 + goal_diff) / 8.0

        k = self.get_k_factor(competition)
        
        # Updates
        self.ratings[team_a] = rating_a + k * mov * (s_a - e_a)
        self.ratings[team_b] = rating_b + k * mov * (s_b - e_b)
        
        return rating_a, rating_b


class FeatureEngineer:
    """
    Computes advanced features for the prediction pipeline:
    - Opponent-adjusted EWMA (Exponentially Weighted Moving Average)
    - Geographic / Travel fatigue
    """
    
    @staticmethod
    def compute_travel_fatigue(matches: list[dict], target_team: str, max_days: int = 14) -> float:
        """
        Computes travel fatigue based on playing outside home country and days rest.
        Returns a modifier <= 1.0 (e.g. 0.95 means 5% performance penalty).
        """
        if len(matches) < 2:
            return 1.0
            
        # Consider the last match
        last_match = matches[-1]
        date_last = pd.to_datetime(last_match.get('date', '1970-01-01'))
        # Using today or the date of the next match (which isn't passed here, so we assume a generic 3-4 day gap if not provided)
        # We will instead compute fatigue dynamically when predicting a specific fixture.
        return 1.0

    @staticmethod
    def build_features(df: pd.DataFrame, elo_registry: EloRegistry = None) -> pd.DataFrame:
        """
        Process an entire historical dataframe to compute Elos and EWMA features incrementally.
        Assumes df is chronologically sorted.
        Expects columns: date, home_team, away_team, home_score, away_score, tournament, neutral, country
        """
        if elo_registry is None:
            elo_registry = EloRegistry()
            
        df = df.copy()
        
        pre_elo_home = []
        pre_elo_away = []
        
        # We also want to track EWMA of adjusted goals
        # Adjusted Goals = Actual Goals * (Average Elo / Opponent Elo)
        # To avoid early division by zero/unstable Elos, we cap the multiplier.
        ewma_goals_for = {}
        ewma_goals_against = {}
        
        alpha = 0.15 # Approx half-life of 4-5 matches
        
        feat_home_gf_ewma = []
        feat_home_ga_ewma = []
        feat_away_gf_ewma = []
        feat_away_ga_ewma = []
        
        for idx, row in df.iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            h_goals = row['home_score']
            a_goals = row['away_score']
            comp = row['tournament']
            neutral = row.get('neutral', False)
            venue = 'N' if neutral else 'H'
            
            # 1. Get Pre-Match Elos
            elo_h = elo_registry.get_elo(h_team)
            elo_a = elo_registry.get_elo(a_team)
            
            pre_elo_home.append(elo_h)
            pre_elo_away.append(elo_a)
            
            # 2. Record pre-match EWMAs
            feat_home_gf_ewma.append(ewma_goals_for.get(h_team, 1.3))
            feat_home_ga_ewma.append(ewma_goals_against.get(h_team, 1.3))
            feat_away_gf_ewma.append(ewma_goals_for.get(a_team, 1.3))
            feat_away_ga_ewma.append(ewma_goals_against.get(a_team, 1.3))
            
            # 3. Opponent adjustment
            adj_h_gf = h_goals * (1600.0 / max(elo_a, 1000.0))
            adj_h_ga = a_goals * (elo_a / 1600.0)
            
            adj_a_gf = a_goals * (1600.0 / max(elo_h, 1000.0))
            adj_a_ga = h_goals * (elo_h / 1600.0)
            
            # 4. Update EWMAs
            ewma_goals_for[h_team] = (alpha * adj_h_gf) + ((1 - alpha) * ewma_goals_for.get(h_team, 1.3))
            ewma_goals_against[h_team] = (alpha * adj_h_ga) + ((1 - alpha) * ewma_goals_against.get(h_team, 1.3))
            
            ewma_goals_for[a_team] = (alpha * adj_a_gf) + ((1 - alpha) * ewma_goals_for.get(a_team, 1.3))
            ewma_goals_against[a_team] = (alpha * adj_a_ga) + ((1 - alpha) * ewma_goals_against.get(a_team, 1.3))
            
            # 5. Update Elo
            elo_registry.update(h_team, a_team, h_goals, a_goals, comp, venue)
            
        df['elo_home_pre'] = pre_elo_home
        df['elo_away_pre'] = pre_elo_away
        
        df['ewma_home_gf'] = feat_home_gf_ewma
        df['ewma_home_ga'] = feat_home_ga_ewma
        df['ewma_away_gf'] = feat_away_gf_ewma
        df['ewma_away_ga'] = feat_away_ga_ewma
        
        return df

