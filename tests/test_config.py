import datetime as dt

from bankcanary.config import PROJECT_ROOT, load_settings


def test_settings_load_and_anchor_paths():
    s = load_settings()
    assert s.availability_lag_days == 60
    assert s.horizons_quarters == [4, 8]
    assert s.start_quarter == dt.date(2001, 3, 31)
    assert s.data_dir.is_absolute() and s.data_dir == PROJECT_ROOT / "data"
    assert s.fixed_split.test_start > s.fixed_split.train_end


def test_cli_version_command():
    from typer.testing import CliRunner

    from bankcanary import __version__
    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
