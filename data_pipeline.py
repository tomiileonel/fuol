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
        
    def get_team_history(self, df: pd.DataFrame, team: str) -> list:
        """
        Obtiene el historial de un equipo estandarizando las claves para el UnifiedEngine.
        El motor requiere 'gf' (goles a favor) y 'gc' (goles en contra) desde la 
        perspectiva del equipo, independientemente de si jugó de local o visitante.
        """
        # Filtrar partidos donde el equipo participó
        mask = (df['home_team'] == team) | (df['away_team'] == team)
        team_df = df[mask].copy()
        
        # Ordenar por fecha cronológica
        team_df = team_df.sort_values('date')
        
        history = []
        for _, row in team_df.iterrows():
            is_home = row['home_team'] == team
            
            # Extraer goles desde la perspectiva del equipo objetivo
            # (Acepta tanto home_score/away_score como gh/ga por si el CSV cambia)
            home_score = row.get('home_score', row.get('gh', 0))
            away_score = row.get('away_score', row.get('ga', 0))
            
            gf = int(home_score) if is_home else int(away_score)
            gc = int(away_score) if is_home else int(home_score)
            
            # Extract features computed by feature engine for compatibility
            gf_ewma = row.get('ewma_home_gf', 0) if is_home else row.get('ewma_away_gf', 0)
            ga_ewma = row.get('ewma_home_ga', 0) if is_home else row.get('ewma_away_ga', 0)
            elo_pre = row.get('elo_home_pre', 1600.0) if is_home else row.get('elo_away_pre', 1600.0)
            
            history.append({
                'date': str(row['date']),
                'home': row['home_team'],
                'away': row['away_team'],
                'opponent': row['away_team'] if is_home else row['home_team'],
                'venue': 'N' if row.get('neutral', False) else ('H' if is_home else 'A'),
                'gf': gf,  # Goals For (Estandarizado)
                'gc': gc,  # Goals Against (Estandarizado)
                'gh': int(home_score), # Mantener originales por compatibilidad
                'ga': int(away_score),
                'tournament': row.get('tournament', 'Unknown'),
                'elo_pre': elo_pre,
                'gf_ewma': gf_ewma,
                'ga_ewma': ga_ewma
            })
            
        return history
