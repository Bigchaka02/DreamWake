"""Command functions for tinycli."""


def cmd_ping():
    return {"ok": True, "out": "pong"}


def cmd_whoami():
    return {"ok": True, "out": "guest"}


def cmd_hello():
    return {"ok": True, "out": "Hello!"}


def version():
    return {"ok": True, "out": "1.4.0"}
