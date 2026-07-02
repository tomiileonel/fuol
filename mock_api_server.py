from flask import Flask, jsonify
import time
import random

app = Flask(__name__)

@app.route('/v1/odds/<match_id>', methods=['GET'])
def get_mock_odds(match_id):
    # Simulación de latencia aleatoria (para probar nuestro Circuit Breaker de 2s)
    latencia = random.uniform(0.5, 3.0) 
    time.sleep(latencia)
    
    # Payload simulado
    odds = {"team_a": 2.10, "team_b": 3.40, "draw": 3.10}
    
    # Inyectar ocasionalmente un cisne negro (payload corrupto o micro-movimiento anómalo)
    if random.random() < 0.05:
        odds["team_a"] = -1.0
    elif random.random() < 0.05:
        odds["team_a"] = 1.01
        
    return jsonify({
        "fixture_id": match_id,
        "odds": odds,
        "latency_simulated": latencia
    })

if __name__ == '__main__':
    print("[Mock API] Servidor iniciado en puerto 5000...")
    app.run(port=5000)
