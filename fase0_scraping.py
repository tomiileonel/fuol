import csv
import asyncio
from live_scouting_provider import HistoryResolver
from search_backends import DDGSearchBackend

async def main():
    backend = DDGSearchBackend()
    resolver = HistoryResolver(backend=backend)
    teams = ["ARGENTINA", "FRANCIA"]
    
    with open('historial_partidos.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "team", "opponent", "goals_for", "goals_against", "competition", "venue"])
        
        for team in teams:
            print(f"Scraping matches for {team}...")
            matches = await resolver.resolve(team)
            print(f"Got {len(matches)} matches for {team}.")
            for m in matches:
                writer.writerow([m.date, team, m.opponent, m.gf, m.gc, m.competition, m.venue])
                
    print("historial_partidos.csv created.")

if __name__ == '__main__':
    asyncio.run(main())
