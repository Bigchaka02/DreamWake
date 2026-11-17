from src import client


def test_url_for_joins_base_and_path():
    assert client.url_for("/ping") == "http://localhost/ping"


def test_request_timeout_is_30_seconds():
    assert client.request_timeout_seconds() == 30.0
