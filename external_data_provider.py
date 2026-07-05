from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen


class ExternalDataProvider:
    def __init__(self, source_url: str | None = None, api_key: str | None = None) -> None:
        self.source_url = source_url or os.getenv('EXTERNAL_DATA_URL')
        self.api_key = api_key or os.getenv('EXTERNAL_DATA_API_KEY')

    def fetch_json(self, endpoint: str | None = None) -> dict[str, Any]:
        if not self.source_url:
            return {}
        url = f"{self.source_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else self.source_url
        req = Request(url, headers={'User-Agent': 'FUOL/1.0', 'Authorization': f'Bearer {self.api_key}' if self.api_key else ''})
        with urlopen(req, timeout=15) as response:
            payload = response.read().decode('utf-8')
            return json.loads(payload)

    def fetch_matches(self, endpoint: str | None = None) -> list[dict[str, Any]]:
        payload = self.fetch_json(endpoint)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ['matches', 'response', 'data']:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []
