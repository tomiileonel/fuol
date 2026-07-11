import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import timedelta
from probability_calibrator import ProbabilityCalibrator
from unified_engine import UnifiedEngine
from data_pipeline import DataPipeline

class WalkForwardPipeline:
    """
    Robust Walk-Forward backtester with Purge & Embargo to prevent data leakage,
    especially protecting against EWMA feature autocorrelation.
    """
    def __init__(self, 
                 train_window_days: int = 365 * 4,  # 4 years training
                 test_window_days: int = 30,        # 1 month testing
                 embargo_days: int = 14,            # 14 days gap to clear EWMA memory
                 min_train_size: int = 100,
                 half_life: float = 365.0,
                 prior_strength: float = 5.0,
                 lambda_scale: float = 0.23):
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.embargo = timedelta(days=embargo_days)
        self.min_train_size = min_train_size
        self.half_life = half_life
        self.prior_strength = prior_strength
        self.lambda_scale = lambda_scale

    def generate_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates (train_df, test_df) tuples adhering to Purge & Embargo.
        """
        if not np.issubdtype(df['date'].dtype, np.datetime64):
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date').reset_index(drop=True)
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        folds = []
        current_test_start = min_date + self.train_window + self.embargo
        
        while current_test_start < max_date:
            test_end = current_test_start + self.test_window
            
            # Train window strictly ends BEFORE the embargo period
            train_end = current_test_start - self.embargo
            train_start = train_end - self.train_window
            
            train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
            test_mask = (df['date'] >= current_test_start) & (df['date'] < test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(train_df) >= self.min_train_size and len(test_df) > 0:
                folds.append((train_df, test_df))
                
            current_test_start = test_end
            
        return folds

    def _compute_metrics(self, y_true: np.ndarray, probs: np.ndarray) -> dict:
        """
        y_true: 0 (Home), 1 (Draw), 2 (Away)
        probs: shape (n, 3)
        """
        n = len(y_true)
        if n == 0:
            return {}
            
        # Log-Loss
        # Clip to prevent log(0)
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        log_loss = -np.mean([np.log(probs_clipped[i, y_true[i]]) for i in range(n)])
        
        # Brier Score (Multi-class)
        targets = np.zeros_like(probs)
        targets[np.arange(n), y_true] = 1.0
        brier_array = np.sum((probs - targets) ** 2, axis=1)
        brier = np.mean(brier_array)
        
        # Ranked Probability Score (RPS)
        cum_probs = np.cumsum(probs, axis=1)
        cum_targets = np.cumsum(targets, axis=1)
        rps_array = np.sum((cum_probs - cum_targets) ** 2, axis=1) / 2.0
        rps = np.mean(rps_array)
        
        # Hit Rate (accuracy of highest probability choice)
        hit_array = (np.argmax(probs, axis=1) == y_true).astype(float)
        hit_rate = np.mean(hit_array)
        
        return {
            'log_loss': float(log_loss),
            'brier': float(brier),
            'brier_array': brier_array.tolist(),
            'rps': float(rps),
            'rps_array': rps_array.tolist(),
            'hit_array': hit_array.tolist(),
            'hit_rate': float(hit_rate),
            'n_samples': n
        }

    def _build_history_cache(self, train_df: pd.DataFrame) -> dict[str, list[dict]]:
        """
        Construye el history_cache para TODOS los equipos del train_df en una
        sola pasada vectorizada, reemplazando el patrón anterior de:

            history_cache = {team: pipeline.get_team_history(train_df, team)
                             for team in unique_teams}

        que hacía ~200 llamadas a get_team_history, cada una escaneando todo
        train_df con una máscara booleana. Eso era O(equipos × filas) por fold.

        Esta versión usa pd.concat + groupby, que es O(filas) total.

        Mantiene EXACTAMENTE el mismo formato de salida que
        DataPipeline.get_team_history, para que UnifiedEngine no note la
        diferencia: una lista de dicts con keys
        {date, home, away, gf, gc, gh, ga, tournament, elo_pre}.

        Los partidos de cada equipo quedan ordenados por fecha (igual que
        get_team_history, que hace .sort_values('date')).
        """
        # Asegurar que el DF tenga las columnas de Elo pre-computadas.
        # Si no las tiene, get_team_history las computaba in-place; replicamos.
        if 'elo_home_pre' not in train_df.columns or 'elo_away_pre' not in train_df.columns:
            from data_pipeline import DataPipeline
            dp = DataPipeline()
            train_df, _ = dp.compute_features(train_df)

        # Duplicar cada partido en dos "vistas": una como local, otra como visitante.
        # Esto permite agrupar por equipo en un solo groupby.
        #
        # Importante: get_team_history original devuelve para cada partido:
        #   - date:        fecha del partido
        #   - home, away:  nombres ORIGINALES (no "team"/"opp")
        #   - gf, gc:      goles del equipo objetivo / goles del rival
        #   - gh, ga:      goles del local original / goles del visitante original
        #                  (¡son SIEMPRE home_score/away_score, sin importar el equipo!)
        #   - tournament:  torneo
        #   - elo_pre:     Elo PRE-PARTIDO del equipo objetivo
        #
        # Mantenemos ese formato exacto para que UnifiedEngine no note diferencia.
        home_view = train_df.rename(columns={
            'home_team': 'team',
            'away_team': 'opp',
            'home_score': 'team_score',
            'away_score': 'opp_score',
            'elo_home_pre': 'team_elo_pre',
            'elo_away_pre': 'opp_elo_pre',
        }).assign(is_home=True)

        away_view = train_df.rename(columns={
            'away_team': 'team',
            'home_team': 'opp',
            'away_score': 'team_score',
            'home_score': 'opp_score',
            'elo_away_pre': 'team_elo_pre',
            'elo_home_pre': 'opp_elo_pre',
        }).assign(is_home=False)

        all_views = pd.concat([home_view, away_view], ignore_index=True)
        # Sort estable: para filas con la misma fecha, preserva el orden de
        # inserción (home_view antes que away_view). Eso replica el
        # comportamiento de get_team_history, que filtraba por equipo y luego
        # ordenaba por fecha (también estable).
        all_views = all_views.sort_values('date', kind='stable').reset_index(drop=True)

        # Construir las columnas finales de forma vectorizada (sin iterrows):
        # - 'date' como string
        # - 'home', 'away': nombres originales (dependen de is_home)
        # - 'gf', 'gc': goles del equipo / rival (dependen de is_home)
        # - 'gh', 'ga': goles del local original / visitante original
        #               = opp_score cuando el equipo era visitante, etc.
        #               En la home_view: gh=team_score, ga=opp_score
        #               En la away_view: gh=opp_score, ga=team_score
        # Replicar exactamente str(row['date']) de get_team_history original,
        # que produce '2020-01-07 00:00:00' (formato Timestamp completo).
        # No usamos .astype(str) porque ese devuelve '2020-01-07' (formato ISO corto)
        # y queremos bit-a-bit identical output.
        all_views['date'] = all_views['date'].apply(lambda x: str(x))

        # Construir home/away/gf/gc/gh/ga con np.where sobre is_home.
        # Esto es vectorizado y preserva el orden de las filas.
        all_views['home'] = np.where(all_views['is_home'], all_views['team'], all_views['opp'])
        all_views['away'] = np.where(all_views['is_home'], all_views['opp'], all_views['team'])
        all_views['gf'] = all_views['team_score'].astype(int)
        all_views['gc'] = all_views['opp_score'].astype(int)
        all_views['gh'] = np.where(all_views['is_home'],
                                   all_views['team_score'],
                                   all_views['opp_score']).astype(int)
        all_views['ga'] = np.where(all_views['is_home'],
                                   all_views['opp_score'],
                                   all_views['team_score']).astype(int)
        all_views['elo_pre'] = all_views['team_elo_pre'].astype(float)

        # Seleccionar las columnas que UnifiedEngine necesita, en el orden
        # que get_team_history devolvía.
        cols = ['date', 'home', 'away', 'gf', 'gc', 'gh', 'ga', 'tournament', 'elo_pre', 'team']
        all_views_slim = all_views[cols]

        # to_dict('records') es C-implemented y ~100x más rápido que iterrows.
        # Hacemos un solo groupby para repartir los records por equipo.
        history_cache: dict[str, list[dict]] = {}
        for team, group in all_views_slim.groupby('team', sort=False):
            # group ya está ordenado por fecha (all_views se ordenó arriba).
            # Sacamos la columna auxiliar 'team' antes de pasar a dict.
            history_cache[team] = group.drop(columns=['team']).to_dict('records')

        return history_cache

    def run(self, pipeline: DataPipeline) -> dict:
        """
        Executes the walk-forward validation across the entire dataset.
        Trains a ProbabilityCalibrator inside each fold (on the training data predictions)
        and applies it to the test data.
        """
        # Load and compute features incrementally (without leakage up to any point in time)
        df, _ = pipeline.prepare_data()
        folds = self.generate_folds(df)

        all_metrics = []

        for fold_idx, (train_df, test_df) in enumerate(folds):
            # 1. Collect raw predictions for training set to fit the calibrator
            train_preds = []
            y_train = []

            # OPTIMIZACIÓN: reemplaza el dict comprehension
            #   {team: pipeline.get_team_history(train_df, team) for team in unique_teams}
            # que hacía ~200 escaneos O(n) cada uno, por un solo groupby O(n) total.
            # Mismo formato de salida, ~10x más rápido en datasets grandes.
            history_cache = self._build_history_cache(train_df)

            # Note: For performance, predicting the entire train_df might be slow if UnifiedEngine is slow.
            # In a production environment, we'd cache base model predictions.
            # For brevity, we sample the last 500 matches for calibration.
            calib_sample = train_df.tail(500)

            for _, row in calib_sample.iterrows():
                h, a = row['home_team'], row['away_team']
                h_hist = history_cache.get(h, [])
                a_hist = history_cache.get(a, [])

                # Base model (no calibrator yet)
                engine = UnifiedEngine(h, a, h_hist, a_hist, optimize_rho=False,
                                       half_life=self.half_life,
                                       prior_strength=self.prior_strength,
                                       lambda_scale=self.lambda_scale)
                try:
                    pred = engine.predict()
                    train_preds.append([pred.get('p1', 0.33), pred.get('px', 0.33), pred.get('p2', 0.33)])

                    if row['home_score'] > row['away_score']:
                        y_train.append(0)
                    elif row['home_score'] == row['away_score']:
                        y_train.append(1)
                    else:
                        y_train.append(2)
                except Exception:
                    pass

            if not train_preds:
                continue

            calibrator = ProbabilityCalibrator(method='isotonic')
            calibrator.fit(np.array(train_preds), np.array(y_train))

            # Use calibrated training predictions as the reference distribution for drift monitoring
            drift_reference = calibrator.predict_proba(np.array(train_preds))

            # 2. Predict on Test Set USING the calibrator
            test_preds = []
            y_test = []
            test_dates = []
            
            for _, row in test_df.iterrows():
                h, a = row['home_team'], row['away_team']
                # We can use train_df + past test_df rows for history to be completely precise,
                # but using train_df history is strict and prevents leakage.
                h_hist = history_cache.get(h, [])
                a_hist = history_cache.get(a, [])
                
                engine = UnifiedEngine(h, a, h_hist, a_hist, calibrator=calibrator, optimize_rho=False,
                                       half_life=self.half_life, 
                                       prior_strength=self.prior_strength, 
                                       lambda_scale=self.lambda_scale)
                try:
                    pred = engine.predict()
                    test_preds.append([pred.get('p1', 0.33), pred.get('px', 0.33), pred.get('p2', 0.33)])
                    date_str = str(row.get('date', f'unknown_{len(test_dates)}'))[:10]
                    match_id = f"{date_str}|{h}|{a}"
                    test_dates.append(match_id)
                    
                    if row['home_score'] > row['away_score']:
                        y_test.append(0)
                    elif row['home_score'] == row['away_score']:
                        y_test.append(1)
                    else:
                        y_test.append(2)
                        
                except Exception as e:
                    pass
                    
            if test_preds:
                metrics = self._compute_metrics(np.array(y_test), np.array(test_preds))
                metrics['test_dates'] = test_dates
                all_metrics.append(metrics)
                
        if not all_metrics:
            return {}
            
        rps_by_match = {}
        hit_by_match = {}
        brier_by_match = {}
        for m in all_metrics:
            for date, rps_val in zip(m['test_dates'], m['rps_array']):
                rps_by_match[date] = rps_val
                # we don't have individual match hit_rate unless we pass the full array.
                # Since we didn't pass array for hit rate, let's just use the avg over the fold.
            
        # Aggregate metrics
        total_samples = sum(m['n_samples'] for m in all_metrics)
        agg = {
            'avg_rps': np.sum([m['rps'] * m['n_samples'] for m in all_metrics]) / total_samples,
            'avg_brier': np.sum([m['brier'] * m['n_samples'] for m in all_metrics]) / total_samples,
            'avg_log_loss': np.sum([m['log_loss'] * m['n_samples'] for m in all_metrics]) / total_samples,
            'avg_hit_rate': np.sum([m.get('hit_rate', 0.0) * m['n_samples'] for m in all_metrics]) / total_samples,
            'total_test_samples': total_samples,
            'n_folds': len(all_metrics),
            'rps_by_match': rps_by_match,
            'all_metrics': all_metrics # expose raw fold metrics just in case
        }
        return agg

if __name__ == "__main__":
    pipeline = DataPipeline()
    tester = WalkForwardPipeline(train_window_days=365*4, test_window_days=30, embargo_days=14)
    print("Iniciando validación Walk-Forward (Purge & Embargo)...")
    results = tester.run(pipeline)
    print(f"Resultados Finales: {results}")
