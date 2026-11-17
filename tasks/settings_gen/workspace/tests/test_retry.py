import pytest

from src import retry


def test_succeeds_after_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("boom")
        return "ok"

    assert retry.call_with_retry(flaky) == "ok"


def test_gives_up_after_configured_attempts():
    with pytest.raises(ValueError):
        retry.call_with_retry(lambda: int("x"))
