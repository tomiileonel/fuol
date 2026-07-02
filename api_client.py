import json
import random
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

class APIFootballClient:
    """
    Wrapper para extraer datos de API-Football (o simularlos para CI/CD y Paper Trading).
    Focalizado en la extracción robusta del JSON raw.
    """
    def __init__(self, api_key=None, use_mock=True):
        self.api_key = api_key
        self.use_mock = use_mock
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-rapidapi-key": self.api_key if self.api_key else "DUMMY_KEY",
            "x-rapidapi-host": "v3.football.api-sports.io"
        }

    def fetch_match_data(self, fixture_id):
        """Obtiene la telemetría del partido."""
        if self.use_mock:
            return self._generate_mock_data(fixture_id)
            
        try:
            response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params={"id": fixture_id})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[API ERROR] Fallo al extraer datos del fixture {fixture_id}: {e}")
            return None

    def fetch_odds_data(self, fixture_id):
        """Obtiene las cuotas de cierre del mercado (Bookmakers) para validación contra el mercado."""
        if self.use_mock:
            return self._generate_mock_odds(fixture_id)
            
        try:
            response = requests.get(f"{self.base_url}/odds", headers=self.headers, params={"fixture": fixture_id, "bookmaker": 8}) # 8 = Bet365
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[API ERROR] Fallo al extraer odds del fixture {fixture_id}: {e}")
            return None

    def _generate_mock_data(self, fixture_id):
        """Simula la estructura anidada y caótica de API-Football para la Liga Argentina."""
        return {
            "response": [{
                "fixture": {
                    "id": fixture_id,
                    "date": datetime.now().isoformat(),
                    "venue": {"name": "Monumental", "city": "Buenos Aires"}
                },
                "teams": {
                    "home": {"id": 100, "name": "River Plate"},
                    "away": {"id": 200, "name": "Boca Juniors"}
                },
                "goals": {
                    "home": None,
                    "away": None
                },
                "lineups": [
                    {
                        "team": {"id": 100, "name": "River Plate"},
                        "formation": "4-2-3-1"
                    },
                    {
                        "team": {"id": 200, "name": "Boca Juniors"},
                        "formation": random.choice(["4-3-3", "4-4-2", "3-4-2-1"])
                    }
                ],
                "statistics": [
                    {"team": {"id": 100}, "statistics": [{"type": "Ball Possession", "value": "60%"}, {"type": "expected_goals", "value": "1.8"}]},
                    {"team": {"id": 200}, "statistics": [{"type": "Ball Possession", "value": "40%"}, {"type": "expected_goals", "value": "0.9"}]}
                ]
            }]
        }
        
    def _generate_mock_odds(self, fixture_id):
        """Simula el JSON de cuotas de apuestas."""
        return {
            "response": [{
                "bookmakers": [{
                    "id": 8,
                    "name": "Bet365",
                    "bets": [{
                        "name": "Match Winner",
                        "values": [
                            {"value": "Home", "odd": "2.10"},
                            {"value": "Draw", "odd": "3.10"},
                            {"value": "Away", "odd": "3.60"}
                        ]
                    }]
                }]
            }]
        }
