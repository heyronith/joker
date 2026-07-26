"""Allow ``python -m joker`` to invoke the CLI."""

from joker.cli.main import app

if __name__ == "__main__":
    app()
