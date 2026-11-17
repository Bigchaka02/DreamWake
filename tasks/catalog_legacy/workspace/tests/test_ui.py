from app import ui


def test_title_is_notes():
    assert ui.title() == "Notes"


def test_greeting_says_hello():
    assert ui.greeting() == "Hello!"
