"""Retry a call a configured number of times."""
from . import settings


def attempts():
    return settings.RETRIES


def backoff_seconds():
    return float(settings.BACKOFF_S)


def call_with_retry(fn):
    last_error = None
    for _ in range(attempts()):
        try:
            return fn()
        except Exception as error:
            last_error = error
    raise last_error
