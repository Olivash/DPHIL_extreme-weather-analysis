"""
Two variance-decomposition figures, both built from variance_summary_all.csv
(output of 01_compute_variance.py):

  1. plot_variance_ratio_twin()
     Single panel, twin y-axis. Total + Intrinsic variance on the left axis,
     Intrinsic/Total ratio on the right axis. Day-12 convergence line and
     a "full saturation" (ratio = 1.0) reference line.

  2. plot_variance_decomposition_two_panel()
     Two stacked panels sharing the x-axis. Top: Total + Intrinsic variance.
     Bottom: all three ratios (Total/ERA5, Intrinsic/ERA5, Intrinsic/Total).

Both save 600 dpi PDF + PNG, no in-image title, x-ticks every 5 days.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "variance_summary_all.csv"

# ── Question 1: has the reforecast's own ensemble spread saturated? ────────
# Internal to the reforecast (ratio_intrinsic_total) -- no ERA5 involved.
# There is no fixed target here; the ratio plateaus wherever it plateaus
# (for this dataset, ~0.90), so "saturated" means "reached its own
# asymptote," found by looking at where the curve goes flat.
SATURATION_DAY_OVERRIDE = None
SATURATION_RATIO_COL = "ratio_intrinsic_total"
SATURATION_TOLERANCE = 0.05      # fraction within the plateau value counts as "saturated"
SATURATION_SUSTAIN_DAYS = 3      # must stay within tolerance this many consecutive leads
SATURATION_PLATEAU_WINDOW = 10   # trailing lead days used to estimate the plateau value

# ── Question 2: has the reforecast's intrinsic variance caught up to reality? ──
# Compares intrinsic variance against ERA5's observed variance
# (ratio_intrinsic_era5). Ratio = 1 is the exact point where the two are
# equal, so this is a threshold crossing against a real target, not a
# plateau estimate. Day 12 in the JoC draft (ratio = 0.973) corresponds to
# "first day ratio >= 0.95" -- close enough to equality, even if still
# fractionally under 1.
ERA5_CONVERGENCE_RATIO_COL = "ratio_intrinsic_era5"
ERA5_CONVERGENCE_THRESHOLD = 0.95

# Wong/Okabe-Ito colorblind-safe palette
COL = {
    "total": "#0072B2",
    "intrinsic": "#D55E00",
    "ratio_it": "#CC79A7",   # Intrinsic / Total
    "ratio_ie": "#D55E00",   # Intrinsic / ERA5 (shares intrinsic's colour, different panel)
    "ratio_te": "#0072B2",   # Total / ERA5 (shares total's colour, different panel)
    "ratio_it_alt": "#009E73",  # Intrinsic / Total, green variant used in the two-panel figure
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


def find_internal_saturation_day(
    df: pd.DataFrame,
    ratio_col: str = SATURATION_RATIO_COL,
    tolerance: float = SATURATION_TOLERANCE,
    sustain_days: int = SATURATION_SUSTAIN_DAYS,
    plateau_window: int = SATURATION_PLATEAU_WINDOW,
) -> tuple[int, float]:
    """
    Question 1: has the reforecast's own ensemble spread saturated?

    Return the first lead day at which `ratio_col` settles near its own
    long-lead plateau, plus the plateau value itself. Purely internal to
    the reforecast -- no ERA5 reference. The plateau is defined as the
    mean of `ratio_col` over the last `plateau_window` lead days, NOT a
    fixed target like 1.0: intrinsic variance is always some fraction of
    total variance by construction, so this ratio has no reason to reach 1.

    A day counts as saturated once the ratio stays within `tolerance` of
    the plateau value for `sustain_days` consecutive lead days -- this
    avoids picking a day that only briefly touches the plateau band by
    chance, and avoids picking whichever single day has the noisiest peak.
    """
    ratio = df[ratio_col].sort_index()
    plateau_value = ratio.iloc[-plateau_window:].mean()

    lower, upper = plateau_value * (1 - tolerance), plateau_value * (1 + tolerance)
    within_plateau = ratio.between(lower, upper)

    for day in ratio.index:
        window = within_plateau.loc[day: day + sustain_days - 1]
        if len(window) == sustain_days and window.all():
            return int(day), float(plateau_value)

    # Never sustained within tolerance -- fall back to the day closest to
    # the plateau value (still relative to the plateau, not to 1.0).
    closest_day = (ratio - plateau_value).abs().idxmin()
    return int(closest_day), float(plateau_value)


def find_era5_convergence_day(
    df: pd.DataFrame,
    ratio_col: str = ERA5_CONVERGENCE_RATIO_COL,
    threshold: float = ERA5_CONVERGENCE_THRESHOLD,
) -> int | None:
    """
    Question 2: has intrinsic variance caught up to ERA5's observed variance?

    Return the first lead day at which `ratio_col` >= `threshold`. Unlike
    saturation, this is a threshold crossing against a real target: ratio
    = 1 is the exact point where intrinsic variance equals ERA5 variance,
    so `threshold` = 0.95 means "close enough to that equality," not an
    arbitrary cutoff. Returns None if the threshold is never reached.
    """
    ratio = df[ratio_col].sort_index()
    reached = ratio[ratio >= threshold]
    return int(reached.index[0]) if not reached.empty else None


def _set_lead_day_ticks(ax, df):
    ax.set_xticks(np.arange(0, df.index.max() + 1, 5))
    ax.set_xlabel("Lead day")


def plot_variance_ratio_twin(df: pd.DataFrame, saturation_day: int):
    """Single-panel twin-axis figure: variance (left) + Intrinsic/Total ratio (right).

    `saturation_day` marks internal ensemble-spread saturation (Question 1),
    not ERA5 convergence -- this figure doesn't reference ERA5 at all.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    l1, = ax.plot(
        df.index, df["var_total"], color=COL["total"],
        marker="o", markersize=4, linewidth=1.3, label="Total (reforecast)",
    )
    l2, = ax.plot(
        df.index, df["var_intrinsic"], color=COL["intrinsic"],
        marker="s", markersize=4, linewidth=1.3, label="Intrinsic (reforecast)",
    )
    ax.set_ylabel(r"Variance ($^\circ$C$^2$)")

    ax2 = ax.twinx()
    l3, = ax2.plot(
        df.index, df["ratio_intrinsic_total"], color=COL["ratio_it"],
        marker="^", markersize=4, linewidth=1.3, label="Ratio (Intrinsic / Total)",
    )
    ax2.axhline(1.0, color=COL["reference"], linewidth=0.8, linestyle=":")
    ax2.text(
        df.index.max(), 1.01, "full saturation",
        ha="right", va="bottom", fontsize=8, color=COL["reference"],
    )
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(top=1.05)

    ax.axvline(saturation_day, color=COL["reference"], linewidth=0.9)
    ax.text(
        saturation_day + 0.3, ax.get_ylim()[0] + 0.05 * np.ptp(ax.get_ylim()),
        f"day {saturation_day}", rotation=90, va="bottom", ha="left",
        fontsize=8, color=COL["reference"],
    )

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = [l1, l2, l3]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="lower right")
    _set_lead_day_ticks(ax, df)

    fig.tight_layout()
    return fig


def plot_variance_decomposition_two_panel(df: pd.DataFrame, era5_convergence_day: int | None = None):
    """Two-panel figure: (top) variance, (bottom) all three ratios.

    `era5_convergence_day`, if given, marks Question 2 (intrinsic variance
    catching up to ERA5) on the ratio panel -- the only panel that plots
    ratio_intrinsic_era5.
    """
    fig, (ax_var, ax_ratio) = plt.subplots(
        2, 1, figsize=(7.2, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    # ── Top: variance ──
    ax_var.plot(
        df.index, df["var_total"], color=COL["total"],
        marker="o", markersize=4, linewidth=1.3, label="Total (reforecast)",
    )
    ax_var.plot(
        df.index, df["var_intrinsic"], color=COL["intrinsic"],
        marker="s", markersize=4, linewidth=1.3, label="Intrinsic (reforecast)",
    )
    ax_var.set_ylabel(r"Variance ($^\circ$C$^2$)")
    ax_var.spines["top"].set_visible(False)
    ax_var.legend(frameon=False, loc="lower right")

    # ── Bottom: ratios ──
    ax_ratio.plot(
        df.index, df["ratio_total_era5"], color=COL["ratio_te"],
        marker="o", markersize=4, linewidth=1.3, label="Total / ERA5",
    )
    ax_ratio.plot(
        df.index, df["ratio_intrinsic_era5"], color=COL["ratio_ie"],
        marker="s", markersize=4, linewidth=1.3, label="Intrinsic / ERA5",
    )
    ax_ratio.plot(
        df.index, df["ratio_intrinsic_total"], color=COL["ratio_it_alt"],
        marker="^", markersize=4, linewidth=1.3, label="Intrinsic / Total",
    )
    ax_ratio.axhline(1.0, color=COL["reference"], linewidth=0.8)

    if era5_convergence_day is not None:
        ax_ratio.axvline(era5_convergence_day, color=COL["reference"], linewidth=0.9, linestyle="--")
        ax_ratio.text(
            era5_convergence_day + 0.3, ax_ratio.get_ylim()[0],
            f"day {era5_convergence_day} (Intrinsic/ERA5 $\\geq$ {ERA5_CONVERGENCE_THRESHOLD})",
            rotation=90, va="bottom", ha="left", fontsize=7, color=COL["reference"],
        )

    ax_ratio.set_ylabel("Ratio")
    ax_ratio.spines["top"].set_visible(False)
    ax_ratio.legend(frameon=False, loc="lower right")

    _set_lead_day_ticks(ax_ratio, df)

    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv(IN_CSV, index_col="days")

    # Question 1: internal ensemble-spread saturation (no ERA5)
    if SATURATION_DAY_OVERRIDE is not None:
        saturation_day = SATURATION_DAY_OVERRIDE
        plateau_value = df.loc[saturation_day, SATURATION_RATIO_COL]
    else:
        saturation_day, plateau_value = find_internal_saturation_day(df)

    print(
        f"[Q1] Internal saturation day: {saturation_day} "
        f"({SATURATION_RATIO_COL} = {df.loc[saturation_day, SATURATION_RATIO_COL]:.3f}, "
        f"plateau = {plateau_value:.3f} +/- {SATURATION_TOLERANCE:.0%}, "
        f"sustained {SATURATION_SUSTAIN_DAYS} days)"
    )

    # Question 2: intrinsic variance catching up to ERA5
    era5_day = find_era5_convergence_day(df)
    if era5_day is not None:
        print(
            f"[Q2] ERA5 convergence day: {era5_day} "
            f"({ERA5_CONVERGENCE_RATIO_COL} = {df.loc[era5_day, ERA5_CONVERGENCE_RATIO_COL]:.3f}, "
            f"threshold = {ERA5_CONVERGENCE_THRESHOLD})"
        )
    else:
        print(f"[Q2] ERA5 convergence day: never reached threshold = {ERA5_CONVERGENCE_THRESHOLD}")

    fig1 = plot_variance_ratio_twin(df, saturation_day)
    fig1.savefig("variance_and_ratio_twin.pdf")
    fig1.savefig("variance_and_ratio_twin.png")

    fig2 = plot_variance_decomposition_two_panel(df, era5_convergence_day=era5_day)

    fig2.savefig("variance_decomposition_two_panel.pdf")
    fig2.savefig("variance_decomposition_two_panel.png")

    print("Saved -> variance_and_ratio_twin.{pdf,png}, variance_decomposition_two_panel.{pdf,png}")


if __name__ == "__main__":
    main()