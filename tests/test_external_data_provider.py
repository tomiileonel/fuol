from external_data_provider import ExternalDataProvider


def test_external_data_provider_returns_empty_without_url(monkeypatch):
    provider = ExternalDataProvider(source_url=None)
    assert provider.fetch_matches() == []
