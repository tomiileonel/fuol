import asyncio
from search_backends import DDGSearchBackend

async def main():
    backend = DDGSearchBackend()
    res = await backend.search('ultimos partidos resultados Francia futbol')
    for r in res:
        print(r)

asyncio.run(main())
