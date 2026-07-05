Plantilla de endpoint público para datos de fútbol.

1. Publica un JSON como este en una URL accesible:
2. Define EXTERNAL_DATA_URL con esa URL.
3. Ejecuta: python sync_external_data.py --source-url <URL>

Formato esperado:
- JSON con un array `matches` o un objeto con `matches`.
- Cada match debe incluir: `date`, `home`, `away`, `gh`, `ga`, `competition`.
