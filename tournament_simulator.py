import numpy as np
from unified_engine import UnifiedEngine, TimeWeighter, DixonColes

class MonteCarloTournament:
    def __init__(self, teams, team_histories, forms, n_simulations=10000, base_modifiers=None, black_swan_config=None):
        """
        teams: Lista de 16 selecciones en el orden estricto del Bracket.
        team_histories: Diccionario con la lista de partidos dinámicos base de cada equipo.
        forms: Diccionario de formaciones tácticas (ya no se usa en el motor unificado, pero se mantiene para compatibilidad).
        base_modifiers: Modificadores cualitativos iniciales (fatiga, localía).
        black_swan_config: Configuración estocástica de cisnes negros.
        """
        self.teams = teams 
        self.base_histories = team_histories 
        self.forms = forms
        self.n_simulations = n_simulations
        self.base_modifiers = base_modifiers or {}
        self.black_swan_config = black_swan_config or {}
        self.champions_distribution = {team: 0 for team in teams}
        
        # PRE-COMPUTE half_life and rho (Fix for Performance)
        print("Pre-computando half_life y rho para simulaciones Monte Carlo...")
        self.optimal_hl = {}
        all_matches = []
        for team in self.teams:
            hist = self.base_histories.get(team, [])
            self.optimal_hl[team] = TimeWeighter.optimize_half_life(hist)
            all_matches.extend(hist)
            
        lam_all = np.mean([m.get('gf', 0) for m in all_matches]) if all_matches else 1.3
        mu_all = np.mean([m.get('gc', 0) for m in all_matches]) if all_matches else 1.3
        
        if len(all_matches) >= 15:
            dc = DixonColes()
            self.precomputed_rho = dc.fit_rho(all_matches, lam_all, mu_all)
        else:
            self.precomputed_rho = -0.13
        print(f"Pre-computación lista. rho = {self.precomputed_rho:.4f}")
        
        # PRE-COMPUTE Elo Ratings (Fix for Performance)
        print("Pre-computando Elo base de cada equipo...")
        from unified_engine import EloRating
        self.base_elos = {}
        for team in self.teams:
            elo = EloRating(team)
            elo.elo_from_matches(self.base_histories.get(team, []))
            self.base_elos[team] = elo.rating

    # ==============================================================================
    # 1. COLAPSO ESTOCÁSTICO DE LA MATRIZ
    # ==============================================================================
    def _stochastic_collapse(self, matrix):
        """
        Tira los dados ponderados sobre la matriz de probabilidad de Dixon-Coles
        para extraer un marcador exacto con varianza controlada.
        """
        flat_matrix = matrix.flatten()
        flat_matrix /= flat_matrix.sum()  # Normalización estricta por precisión flotante
        
        # Muestreo estadístico puro
        sampled_idx = np.random.choice(len(flat_matrix), p=flat_matrix)
        goals_a, goals_b = np.unravel_index(sampled_idx, matrix.shape)
        
        # Resolución de empates (Knockout)
        if goals_a == goals_b:
            # Coin flip estricto para simular penales. En una versión expandida, 
            # este evento discreto se puede sesgar con el vector de PageRank.
            if np.random.rand() > 0.5:
                goals_a += 1
            else:
                goals_b += 1
                
        return goals_a, goals_b

    # ==============================================================================
    # 2. MOTOR DE UNIVERSOS (PROPAGACIÓN DE CONCEPT DRIFT)
    # ==============================================================================
    def _simulate_universe(self):
        """
        Simula una línea temporal completa desde Octavos hasta la Final.
        """
        # Aislamos la línea temporal: clonación profunda del historial base
        # para que la entropía de esta simulación no contamine a las demás.
        # Clonación profunda de historiales y modificadores
        current_knowledge = {
            team: [match.copy() for match in self.base_histories[team]]
            for team in self.teams
        }
        
        current_modifiers = {
            team: self.base_modifiers.get(team, {}).copy()
            for team in self.teams
        }
        
        # Copia de los base Elos que irán evolucionando en la simulación
        current_elos = self.base_elos.copy()
        
        # Inyección de Cisne Negro Estocástico
        if self.black_swan_config:
            bs_team = self.black_swan_config.get("team")
            if bs_team in current_modifiers and np.random.rand() < self.black_swan_config.get("probability", 0):
                current_modifiers[bs_team]["injury_modifier"] = current_modifiers[bs_team].get("injury_modifier", 1.0) * self.black_swan_config.get("injury_multiplier", 1.5)
        
        current_round = self.teams.copy()
        
        while len(current_round) > 1:
            next_round = []
            
            # Iterar sobre las llaves emparejadas del bracket
            for i in range(0, len(current_round), 2):
                t_a = current_round[i]
                t_b = current_round[i+1]
                
                # Instanciar el motor unificado con parámetros pre-computados
                hl = (self.optimal_hl.get(t_a, 365) + self.optimal_hl.get(t_b, 365)) / 2.0
                
                engine = UnifiedEngine(
                    t_a, t_b, 
                    current_knowledge[t_a], current_knowledge[t_b], 
                    venue='N', # Neutral venue for knockout stage
                    modifiers_a=current_modifiers[t_a],
                    modifiers_b=current_modifiers[t_b],
                    half_life=hl,
                    optimize_rho=False,
                    precomputed_rho=self.precomputed_rho,
                    base_elo_a=current_elos[t_a],
                    base_elo_b=current_elos[t_b]
                )
                
                # Ejecutar cálculo
                output = engine.predict()
                matrix = output["score_matrix"]
                
                # Colapsar resultado probabilístico
                gf_a, gf_b = self._stochastic_collapse(matrix)
                
                # PROPAGACIÓN DEL DRIFT: 
                # El algoritmo asimila y aprende del resultado simulado.
                current_knowledge[t_a].append({
                    "date": "2026-07-01", "opponent": t_b, 
                    "gf": gf_a, "gc": gf_b, "res": "W" if gf_a > gf_b else "L"
                })
                current_knowledge[t_b].append({
                    "date": "2026-07-01", "opponent": t_a, 
                    "gf": gf_b, "gc": gf_a, "res": "W" if gf_b > gf_a else "L"
                })
                
                # Update Incremental Elo inside the universe
                engine.elo_a.update(current_knowledge[t_a][-1], engine.elo_b.rating)
                engine.elo_b.update(current_knowledge[t_b][-1], engine.elo_a.rating)
                
                # Store the updated Elos to be used as base in the next round
                current_elos[t_a] = engine.elo_a.rating
                current_elos[t_b] = engine.elo_b.rating
                
                if gf_a > gf_b:
                    next_round.append(t_a)
                else:
                    next_round.append(t_b)
                
                # Propagación de fatiga de viaje (Acumulativa) - Deshabilitada hasta tener datos reales de distancia de viaje
                # Si se quiere mantener el concepto, condicionarlo a datos reales:
                # current_modifiers[t_a]["travel_fatigue"] = current_modifiers[t_a].get("travel_fatigue", 1.0) * travel_penalty(t_a, venue_ciudad)
            
            # Avanzar a Cuartos, Semifinal, Final
            current_round = next_round
            
        return current_round[0] # Retorna el Campeón del Universo

    # ==============================================================================
    # 3. ORQUESTADOR MASIVO
    # ==============================================================================
    def run(self):
        print(f"Iniciando colapso de {self.n_simulations} universos paralelos...")
        
        for _ in range(self.n_simulations):
            champion = self._simulate_universe()
            self.champions_distribution[champion] += 1
            
        # Transformación a vector de probabilidades absolutas
        for team in self.champions_distribution:
            self.champions_distribution[team] = (self.champions_distribution[team] / self.n_simulations) * 100
            
        # Retornar diccionario ordenado por probabilidad de campeonato
        return dict(sorted(self.champions_distribution.items(), key=lambda item: item[1], reverse=True))
