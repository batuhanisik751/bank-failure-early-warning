"""``bankcanary build-features-v2``: the Prototype 2 feature table (``features_v2``)."""

from __future__ import annotations

import logging

import typer


def register(app: typer.Typer) -> None:
    @app.command("build-features-v2")
    def build_features_v2_cmd() -> None:
        """Build features_v2 = every features_v1 column + the Prototype 2 features.

        ``features_v1`` is left untouched (P1 reproducibility); the v2 table is written
        to Parquet and DuckDB under its own name, keyed by (cert, repdte).
        """
        from bankcanary.config import load_settings
        from bankcanary.features import registry
        from bankcanary.features.build import build_features_table

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        _, s = build_features_table(load_settings(), version="v2")
        p2 = [n for n in registry.feature_names(version="v2") if n not in registry.feature_names()]
        typer.echo(
            f"features_v2: {s['rows']} rows, {s['features']} features "
            f"({len(p2)} new in P2; duckdb {s['duckdb_rows']} rows)"
        )
        typer.echo("P2 columns: " + ", ".join(p2))
        if "contrast" in s:
            typer.echo("median feature values, label-complete rows, y_4q = 0 vs 1:")
            typer.echo(s["contrast"].round(4).to_string())
