import pandas as pd
import numpy as np
from feature_engine import FeatureEngineer, EloRegistry

class DataPipeline:
    """
    Handles data ingestion, cleaning, and feature engineering for the historical dataset.
    Prepared for future StatsBomb/FBref xG integration.
    """
    def __init__(self, csv_path: str = "results.csv"):
        self.csv_path = csv_path
        
    def load_and_clean(self) -> pd.DataFrame:
        """
        Loads the historical dataset, cleans column names, handles missing values,
        and ensures chronological ordering.
        """
        df = pd.read_csv(self.csv_path)
        
        # Ensure chronological ordering
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Drop matches with missing scores
        df = df.dropna(subset=['home_score', 'away_score'])
        
        # Ensure scores are integers
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        
        # Add placeholder for real xG (to be filled by future StatsBomb/FBref scraper)
        df['home_xg'] = np.nan
        df['away_xg'] = np.nan
        
        return df
        
    def compute_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, EloRegistry]:
        """
        Runs the feature engine to compute Dynamic Elo and EWMA features.
        """
        elo_registry = EloRegistry()
        df_enriched = FeatureEngineer.build_features(df, elo_registry)
        return df_enriched, elo_registry
        
    def prepare_data(self) -> tuple[pd.DataFrame, EloRegistry]:
        """
        End-to-end data preparation.
        """
        df = self.load_and_clean()
        df, elo_registry = self.compute_features(df)
        return df, elo_registry
        
    def get_team_history(self, df: pd.DataFrame, team: str) -> list[dict]:
        """
        Extracts chronological match history for a specific team as a list of dicts.
        Format compatible with UnifiedEngine.
        """
        team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        
        history = []
        for _, row in team_df.iterrows():
            is_home = row['home_team'] == team
            opp = row['away_team'] if is_home else row['home_team']
            gf = row['home_score'] if is_home else row['away_score']
            gc = row['away_score'] if is_home else row['home_score']
            venue = 'N' if row.get('neutral', False) else ('H' if is_home else 'A')
            
            # Extract features computed by feature engine
            gf_ewma = row['ewma_home_gf'] if is_home else row['ewma_away_gf']
            ga_ewma = row['ewma_home_ga'] if is_home else row['ewma_away_ga']
            elo_pre = row['elo_home_pre'] if is_home else row['elo_away_pre']
            
            match_dict = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'opponent': opp,
                'venue': venue,
                'gf': gf,
                'gc': gc,
                'competition': row['tournament'],
                'elo_pre': elo_pre,
                'gf_ewma': gf_ewma,
                'ga_ewma': ga_ewma
            }
            history.append(match_dict)
            
        return history
