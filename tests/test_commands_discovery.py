import typer
from typer.testing import CliRunner

from bankcanary.commands import register_all


def test_register_all_imports_every_module_and_keeps_existing_commands():
    from bankcanary.cli import app

    names = register_all(typer.Typer())
    assert names == sorted(names)
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "build-panel", "build-labels", "build-features", "train", "evaluate"):
        assert cmd in result.output
