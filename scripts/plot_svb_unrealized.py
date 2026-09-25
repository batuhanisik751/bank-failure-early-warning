"""Sanity plot for the rate and run-risk features: SVB, Signature and First Republic vs peers.

Two panels, 2020Q1-2023Q1: ``unrealized_loss_to_tier1`` and ``uninsured_share`` for the
three banks that failed in March 2023 against the median and 5th-95th percentile band
of banks with more than $10 billion in assets. A working feature set shows SVB's
unrealised loss approaching -1 x Tier 1 by 2022Q4 while its uninsured share sits far
above the peer band. Run ``uv run python scripts/plot_svb_unrealized.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bankcanary.config import load_settings  # noqa: E402
from bankcanary.features import run_risk, sensitivity  # noqa: E402
from bankcanary.storage.parquet import read_table  # noqa: E402

BANKS = {24735: "Silicon Valley Bank", 57053: "Signature Bank", 59017: "First Republic Bank"}
COLORS = {24735: "#2a78d6", 57053: "#eb6834", 59017: "#1baf7a"}
PEER_MIN_ASSETS = 10_000_000  # thousands of dollars = $10 billion
START, END = "2020-03-31", "2023-03-31"
PANELS = [
    ("unrealized_loss_to_tier1", "Unrealised securities loss / Tier 1 capital"),
    ("uninsured_share", "Uninsured deposits / total deposits"),
]
FIGURE = "svb_unrealized_losses.png"


def features_window(settings) -> pd.DataFrame:
    """P2 sensitivity and run-risk features for every panel row in the plot window."""
    panel = read_table("panel", settings=settings)
    panel = panel[(panel["repdte"] >= START) & (panel["repdte"] <= END)].reset_index(drop=True)
    feats = pd.concat([sensitivity.build(panel), run_risk.build(panel)], axis=1)
    return pd.concat([panel[["cert", "repdte", "asset"]], feats], axis=1)


def peer_bands(feats: pd.DataFrame, col: str) -> pd.DataFrame:
    """Median and 5th/95th percentile by quarter among banks above the asset floor."""
    peers = feats[feats["asset"] > PEER_MIN_ASSETS]
    return peers.groupby("repdte")[col].quantile([0.05, 0.5, 0.95]).unstack()


def draw(ax: plt.Axes, feats: pd.DataFrame, col: str, title: str) -> None:
    band = peer_bands(feats, col)
    ax.fill_between(band.index, band[0.05], band[0.95], color="#d9d8d3", label="Peers 5th-95th pct")
    ax.plot(band.index, band[0.5], color="#52514e", lw=2, label="Peer median")
    for cert, name in BANKS.items():
        s = feats[feats["cert"] == cert].set_index("repdte")[col].dropna()
        ax.plot(s.index, s.to_numpy(), color=COLORS[cert], lw=2, marker="o", ms=4, label=name)
        ax.annotate(
            name,
            (s.index[-1], s.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color="#0b0b0b",
            va="center",
        )
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.grid(axis="y", color="#e8e7e2", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.02)


def main() -> None:
    settings = load_settings()
    feats = features_window(settings)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for ax, (col, title) in zip(axes, PANELS):
        draw(ax, feats, col, title)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[0].axhline(0, color="#52514e", lw=0.8)
    fig.suptitle("Rate and run-risk features, banks above $10B, 2020Q1-2023Q1", x=0.01, ha="left")
    out = settings.reports_dir / "figures" / FIGURE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    svb = feats[(feats["cert"] == 24735) & (feats["repdte"] == "2022-12-31")].iloc[0]
    print(f"wrote {out}")
    print(f"SVB 2022Q4 unrealized_loss_to_tier1={svb['unrealized_loss_to_tier1']:.4f}")
    print(f"SVB 2022Q4 uninsured_share={svb['uninsured_share']:.4f}")


if __name__ == "__main__":
    main()
