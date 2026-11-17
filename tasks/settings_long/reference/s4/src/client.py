"""Minimal HTTP client helpers."""
from . import settings


def url_for(path):
    return settings.BASE_URL.rstrip("/") + "/" + path.lstrip("/")


def request_timeout_seconds():
    return settings.TIMEOUT_MS / 1000


def build_request(path):
    return {"url": url_for(path), "timeout": request_timeout_seconds()}


def is_secure():
    return settings.BASE_URL.startswith("https://")


def service_name():
    return settings.SERVICE_NAME
