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
        Obtiene el historial de un equipo usando el EloRegistry global ya
        precomputado sobre el dataset completo (con rival explícito en
        cada partido), en lugar de recalcular un Elo simplificado que
        asume "rival promedio" fijo.

        Esto garantiza coherencia entre el Elo que ve el motor unificado
        y el Elo que se calcula en `feature_engine.FeatureEngineer.build_features`.
        """
        if 'elo_home_pre' not in df.columns or 'elo_away_pre' not in df.columns:
            # Si el DF no fue enriquecido, lo enriquecemos in-place.
            df, _ = self.compute_features(df)

        mask = (df['home_team'] == team) | (df['away_team'] == team)
        team_df = df[mask].copy().sort_values('date')

        history: list[dict] = []
        for _, row in team_df.iterrows():
            is_home = row['home_team'] == team
            home_score = int(row.get('home_score', row.get('gh', 0)))
            away_score = int(row.get('away_score', row.get('ga', 0)))

            gf = home_score if is_home else away_score
            gc = away_score if is_home else home_score

            # Elo pre-partido del equipo (ya calculado con rival real).
            elo_pre = (
                row['elo_home_pre'] if is_home else row['elo_away_pre']
            )

            history.append({
                'date': str(row['date']),
                'home': row['home_team'],
                'away': row['away_team'],
                'gf': gf,
                'gc': gc,
                'gh': home_score,
                'ga': away_score,
                'tournament': row.get('tournament', 'Unknown'),
                'elo_pre': float(elo_pre),
            })

        return history
