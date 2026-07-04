import httpx
import urllib.parse
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any

class WebSearchBackend:
    """Interfaz abstracta para motores de búsqueda web."""
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        raise NotImplementedError

    async def fetch(self, url: str) -> str:
        raise NotImplementedError

class DDGSearchBackend(WebSearchBackend):
    """Implementación de búsqueda utilizando DuckDuckGo HTML (ligero y sin JS)."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True) as client:
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
            except Exception as e:
                print(f"[DDGSearchBackend] Error en búsqueda: {e}")
                return []
                
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.find_all('div', class_='result'):
            if len(results) >= max_results:
                break
                
            title_tag = result.find('a', class_='result__url')
            snippet_tag = result.find('a', class_='result__snippet')
            
            if title_tag and snippet_tag:
                raw_url = title_tag.get('href', '')
                if raw_url.startswith('//duckduckgo.com/l/?uddg='):
                    # Extract the actual URL from the uddg parameter
                    parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse('https:' + raw_url).query)
                    if 'uddg' in parsed_qs:
                        raw_url = parsed_qs['uddg'][0]
                
                results.append({
                    'title': title_tag.text.strip(),
                    'body': snippet_tag.text.strip(),
                    'url': raw_url
                })
                
        return results

    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True) as client:
            try:
                response = await client.get(url, timeout=15.0)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"[DDGSearchBackend] Error al extraer url {url}: {e}")
                return ""
