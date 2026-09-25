"""Command-line entry point: ``bankcanary <command>``.

Commands are registered by the modules that implement them; this file only wires them up.
"""

from __future__ import annotations

import typer

from bankcanary import __version__

app = typer.Typer(
    name="bankcanary",
    help="Early-warning system for US bank failures (educational project, not advice).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """BankCanary pipeline commands. Run a command with --help for its options."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
