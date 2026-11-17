import pytest

from cli import core


def test_ping():
    assert core.render("ping") == "pong"


def test_whoami():
    assert core.render("whoami") == "guest"


def test_hello():
    assert core.render("hello") == "Hello!"


def test_unknown_command_is_rejected():
    with pytest.raises(LookupError):
        core.render("nope")
