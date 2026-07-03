import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import stealth
import re

class TransfermarktScraper:
    """
    Scraper financiero utilizando Playwright y Stealth para evitar bloqueos (Ej. Cloudflare).
    Extrae el valor total de mercado de un equipo nacional.
    """
    def __init__(self):
        self.base_url = "https://www.transfermarkt.com"
        
    async def get_squad_value(self, team_name: str) -> float:
        """
        Busca el equipo y extrae su valor total de mercado en Euros.
        Si falla, retorna un valor por defecto para no romper el orquestador.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                
                # Apply stealth to bypass basic bot protections
                await stealth(page)
                
                # 1. Search for the team
                search_url = f"{self.base_url}/schnellsuche/ergebnis/schnellsuche?query={team_name}"
                await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
                
                # 2. Extract the market value from the first result (simplified logic)
                # Note: Transfermarkt HTML structure can change. This is a robust attempt.
                # Usually, national teams are in the results. We will look for elements containing '€' and 'm' or 'bn'
                
                # As a fallback for this Quant system, if the direct scrape fails, we try to grab any large monetary value
                # or we just return a default prior if the CSS selector fails.
                
                content = await page.content()
                
                # Basic regex to find something like "€1.05bn" or "€850.00m" in the page text
                # A more precise way is targeting the exact CSS, but for resilience we use text search as fallback
                val_match = re.search(r'€([0-9\.]+)(k|m|bn)', content, re.IGNORECASE)
                
                await browser.close()
                
                if val_match:
                    num = float(val_match.group(1))
                    multiplier = val_match.group(2).lower()
                    
                    if multiplier == 'bn':
                        return num * 1_000_000_000
                    elif multiplier == 'm':
                        return num * 1_000_000
                    elif multiplier == 'k':
                        return num * 1_000
                        
                # Default fallback if no regex matches
                print(f"[Scraper] No se pudo parsear el valor exacto para {team_name}. Usando valor por defecto.")
                return 150_000_000.0 # 150M EUR fallback
                
        except Exception as e:
            print(f"[Scraper ERROR] Fallo al extraer valor de {team_name}: {e}")
            return 150_000_000.0 # 150M EUR default fallback for national teams

    async def get_squad_values(self, team_a: str, team_b: str):
        """Extrae los valores de ambos equipos concurrentemente."""
        val_a, val_b = await asyncio.gather(
            self.get_squad_value(team_a),
            self.get_squad_value(team_b)
        )
        return {"team_a": val_a, "team_b": val_b}

if __name__ == "__main__":
    # Test rápido
    async def test():
        scraper = TransfermarktScraper()
        res = await scraper.get_squad_values("Argentina", "France")
        print(res)
    asyncio.run(test())
