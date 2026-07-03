import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def run_manual_settlement():
    print("="*50)
    print(" 🏦 FUOL QUANT: Módulo de Liquidación Manual")
    print("="*50)
    
    # Conexión asíncrona al Data Lake (Alineado con paper_trader.py)
    client = AsyncIOMotorClient("mongodb://localhost:27017/")
    db = client["fuol_lake"]
    ledger = db["paper_trading_ledger"]
    wallet_state = db["wallet_state"]

    # 1. Buscar todas las posiciones abiertas
    cursor = ledger.find({"status": "PENDING"})
    pending_trades = await cursor.to_list(length=100)

    if not pending_trades:
        print("\n[✓] No hay operaciones pendientes de liquidación.")
        print("[✓] Tu portafolio está 100% conciliado.")
        return

    print(f"\nSe encontraron {len(pending_trades)} posiciones pendientes.\n")

    # 2. Auditar cada posición
    for trade in pending_trades:
        match_id = trade["match_id"]
        team = trade["selection"]
        stake = trade["stake"]
        odds = trade["market_odds"]

        print(f"[-] OPERACIÓN: {match_id} | Selección: {team}")
        print(f"    Riesgo: ${stake:.2f} | Cuota Mercado: {odds}")
        
        # Como es una herramienta de terminal manual, esperamos el input del operador
        outcome = input("    ¿Resultado de la selección? (W=Ganada / L=Perdida / V=Anulada): ").strip().upper()

        if outcome == 'W':
            # Matemática Financiera: El stake ya se restó al abrir la orden en el cálculo dinámico.
            # Si ganamos, sumamos el retorno BRUTO (Stake * Cuota) al bankroll.
            gross_return = round(stake * odds, 2)
            net_profit = round(gross_return - stake, 2)
            
            # Actualización atómica en MongoDB
            await ledger.update_one({"_id": trade["_id"]}, {"$set": {"status": "WON", "pnl_usd": net_profit}})
            await wallet_state.update_one({"_id": "main_wallet"}, {"$inc": {"balance": gross_return}})
            print(f"    [+] LIQUIDADA (WON): +${net_profit:.2f} de ganancia neta. Retorno acreditado.\n")

        elif outcome == 'L':
            # Si perdemos, el Bankroll no cambia (el dinero ya se descontó antes).
            await ledger.update_one({"_id": trade["_id"]}, {"$set": {"status": "LOST", "pnl_usd": -stake}})
            print(f"    [-] LIQUIDADA (LOST): La pérdida de ${stake:.2f} fue absorbida.\n")

        elif outcome == 'V':
            # Si se suspende el partido, se devuelve el dinero intacto.
            await ledger.update_one({"_id": trade["_id"]}, {"$set": {"status": "VOID", "pnl_usd": 0.0}})
            await wallet_state.update_one({"_id": "main_wallet"}, {"$inc": {"balance": stake}})
            print(f"    [~] LIQUIDADA (VOID): Capital devuelto a la cuenta.\n")
            
        else:
            print("    [!] Comando no reconocido. Operación dejada en estado PENDING.\n")

    print("="*50)
    print(" 💼 LIQUIDACIÓN FINALIZADA. Revisa tu Dashboard para ver el nuevo Bankroll.")
    print("="*50)

if __name__ == "__main__":
    # Ejecutamos el Event Loop de asincronismo
    asyncio.run(run_manual_settlement())
