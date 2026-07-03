import asyncio
import websockets
import json
import random
import time

async def stream_odds(websocket):
    print(f"Client connected: {websocket.remote_address}")
    
    # Base probabilities for Australia vs Egypt (approx model probabilities)
    base_p_a = 0.38 # Australia
    base_p_d = 0.29 # Draw
    base_p_b = 0.33 # Egypt
    
    tick = 0
    try:
        while True:
            tick += 1
            
            # Alternar estado normal y Cisne Negro cada 10 ticks
            if (tick % 20) < 10:
                # Estado Normal
                base_p_a = 0.38
                base_p_d = 0.29
                base_p_b = 0.33
            else:
                # Estado de Anomalía / Cisne Negro
                if tick % 20 == 10:
                    print(">>> INJECTING BLACK SWAN (Market Crash - Australia Goal) <<<")
                base_p_a = 0.70
                base_p_d = 0.20
                base_p_b = 0.10
            
            # Add some white noise (market fluctuation)
            noise_a = random.uniform(-0.015, 0.015)
            noise_d = random.uniform(-0.015, 0.015)
            
            p_a = max(0.01, base_p_a + noise_a)
            p_d = max(0.01, base_p_d + noise_d)
            p_b = max(0.01, 1.0 - p_a - p_d)
            
            # Add overround (Bookmaker margin ~ 105%)
            overround = 1.05
            o_a = round(1.0 / (p_a / overround), 2)
            o_d = round(1.0 / (p_d / overround), 2)
            o_b = round(1.0 / (p_b / overround), 2)
            
            payload = {
                "match_id": "AUS-EGY",
                "timestamp": time.time(),
                "tick": tick,
                "odds": {
                    "team_a": o_a,
                    "draw": o_d,
                    "team_b": o_b
                },
                "status": "LIVE"
            }
            
            await websocket.send(json.dumps(payload))
            
            # Emit tick every 500ms
            await asyncio.sleep(0.5)
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    print("Starting Mock WebSocket Server on ws://localhost:5001")
    async with websockets.serve(stream_odds, "localhost", 5001):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
