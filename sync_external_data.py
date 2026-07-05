from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from external_data_provider import ExternalDataProvider


def sync_matches(output_path: str, source_url: str | None = None, api_key: str | None = None) -> Path:
    provider = ExternalDataProvider(source_url=source_url, api_key=api_key)
    matches = provider.fetch_matches()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['date', 'home', 'away', 'gh', 'ga', 'competition'])
        writer.writeheader()
        for match in matches:
            normalized = {
                'date': match.get('date') or match.get('fixture', {}).get('date', '1970-01-01')[:10],
                'home': match.get('home') or match.get('teams', {}).get('home', {}).get('name', ''),
                'away': match.get('away') or match.get('teams', {}).get('away', {}).get('name', ''),
                'gh': match.get('gh') or match.get('goals', {}).get('home', 0),
                'ga': match.get('ga') or match.get('goals', {}).get('away', 0),
                'competition': match.get('competition') or match.get('league', {}).get('name', 'external'),
            }
            writer.writerow(normalized)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize external match data into CSV')
    parser.add_argument('--output', default='external_matches.csv')
    parser.add_argument('--source-url', default=None)
    parser.add_argument('--api-key', default=None)
    args = parser.parse_args()
    path = sync_matches(args.output, source_url=args.source_url, api_key=args.api_key)
    print(f'External data synced to {path}')
