"""Auto-discovered CLI command modules.

Each module in this package defines ``register(app: typer.Typer) -> None`` and adds its own
commands to the app. ``register_all`` imports every module in alphabetical order, so a new
pipeline stage only has to drop a file here; ``cli.py`` never needs editing.
"""

from __future__ import annotations

import importlib
import pkgutil

import typer


def register_all(app: typer.Typer) -> list[str]:
    """Import every command module in this package and let it register its commands."""
    registered: list[str] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{__name__}.{info.name}")
        register = getattr(module, "register", None)
        if register is None:
            continue
        register(app)
        registered.append(info.name)
    return registered
