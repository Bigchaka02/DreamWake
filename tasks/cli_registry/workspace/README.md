# tinycli

A tiny command-line tool.

Commands are plain functions in `cli/commands.py`.
Each command returns the text to print.
`cli.core.render(name)` gives the text for a command name.

Tests live in `tests/` and run with pytest.
