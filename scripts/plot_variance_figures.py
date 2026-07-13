"""
Final figure: intrinsic vs. ERA5 variance by lead day, with the
intrinsic/ERA5 ratio on a secondary axis and the day-12 convergence
point annotated.

Reads variance_summary_all.csv produced by 01_compute_variance.py.
Saves 600 dpi PDF + PNG, no in-image title (per AMS/AGU figure conventions).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "variance_summary_all.csv"
OUT_BASENAME = "intrinsic_era5_ratio"
CONVERGENCE_DAY = 12

# Wong/Okabe-Ito colorblind-safe palette
COL = {
    "var_intrinsic": "#D55E00",
    "var_era5": "#009E73",
    "ratio": "#CC79A7",
    "reference": "#000000",
}

# ── Shared style settings (AMS/AGU-style) ───────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def plot_variance_and_ratio(df: pd.DataFrame, convergence_day: int = CONVERGENCE_DAY):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    l1, = ax.plot(
        df.index, df["var_intrinsic"], color=COL["var_intrinsic"],
        marker="s", markersize=4, linewidth=1.2, label="Intrinsic (reforecast)",
    )
    l2, = ax.plot(
        df.index, df["var_era5"], color=COL["var_era5"],
        marker="^", markersize=4, linewidth=1.2, linestyle="--", label="ERA5",
    )
    ax.set_ylabel(r"Variance ($^\circ$C$^2$)")
    ax.set_xlabel("Forecast lead time (days)")

    ax2 = ax.twinx()
    l3, = ax2.plot(
        df.index, df["ratio_intrinsic_era5"], color=COL["ratio"],
        marker="o", markersize=4, linewidth=1.2, label="Intrinsic / ERA5",
    )
    ax2.axhline(1.0, color=COL["reference"], linewidth=0.8, alpha=0.6, linestyle=":")
    ax2.set_ylabel("Ratio")

    y_meet = df.loc[convergence_day, "var_intrinsic"]
    ax.annotate(
        f"day {convergence_day}",
        xy=(convergence_day, y_meet), xycoords="data",
        xytext=(convergence_day + 5, y_meet - 5), textcoords="data",
        fontsize=9, color=COL["reference"],
        arrowprops=dict(arrowstyle="->", linewidth=1, color=COL["reference"]),
    )

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = [l1, l2, l3]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="lower right")
    ax.set_xticks(np.arange(0, df.index.max() + 1, 5))

    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv(IN_CSV, index_col="days")
    fig = plot_variance_and_ratio(df)
    fig.savefig(f"{OUT_BASENAME}.pdf")
    fig.savefig(f"{OUT_BASENAME}.png")
    print(f"Saved -> {OUT_BASENAME}.pdf, {OUT_BASENAME}.png")


if __name__ == "__main__":
    main()