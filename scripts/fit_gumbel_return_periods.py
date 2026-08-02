"""
Gumbel POT return periods: ERA5 vs. reforecast (single lead day).

Fits a Gumbel distribution to the top 5% (values above the 95th
percentile) of ERA5 t2m and reforecast t2m at a fixed lead day, pooled
across the JJA heatwave valid-date window and all years/ensemble members,
then converts fitted and empirical exceedance probabilities into return
periods in years.

Reforecast ensemble members are treated as independent draws (as stated
for this analysis, in the spirit of the UNSEEN approach -- e.g. Thompson
et al. 2017, Kelder et al. 2020) to extend the effective record length
beyond the calendar years actually sampled. See RATE_MODE below: this is
the one modelling choice in this script that materially changes the
reforecast return-period axis, so it is kept as an explicit switch rather
than baked in.

Caveat: fitting a plain (unconditional) Gumbel by MLE to a top-percentile
subset is what was asked for here, but it is an approximation -- the
textbook peaks-over-threshold model fits a generalized Pareto to the
threshold exceedances (values - threshold) instead, precisely because a
Gumbel fit to a left-truncated sample is a biased estimator of the
unconditional distribution. If the two datasets' fitted curves look
inconsistent with their own empirical points, that's the first thing to
swap out (see fit_pot_gumbel).

Inputs
------
pnw_box_era5.nc                 ERA5 t2m over the PNW box
reforecast_0_t0_13.csv          medium-range reforecast (days 0-13)
                                 columns: inidate, hDate, number, t2m,
                                 days, day_month, forecast_date

Outputs
-------
gumbel_return_period_summary.csv   fitted parameters + rates, both datasets
gumbel_return_periods.pdf/.png     return period plot
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

# ── Paths (matches scripts/compute_variance.py) ────────────────────────────
ERA5_PATH = "/network/group/aopp/predict/AWH020_AYIM_EXTREME/ERA5/era5_t2m/pnw_box_era5.nc"
MEDIUM_RANGE_CSV = "reforecast_0_t0_13.csv"

OUT_CSV = "gumbel_return_period_summary.csv"
OUT_BASENAME = "gumbel_return_periods"

LEAD_DAY = 12  # ERA5 convergence day, see scripts/READMe.md

# Target valid dates. If you have a specific list of 11 mm-dd dates, put
# them here, e.g. TARGET_DATES = ["06-15", "06-19", ..., "07-21"] --
# whatever's in this list is what n_years / rate below get derived from.
# Left as None, every date in [VALID_START_MMDD, VALID_END_MMDD] is used.
TARGET_DATES = None
VALID_START_MMDD = "06-15"
VALID_END_MMDD = "07-21"

THRESHOLD_PERCENTILE = 95  # top 5%

# How to convert the reforecast's pooled member-years into events/year:
#   "unseen"     -> rate = n_exceedances / n_years. Each ensemble
#                   member-day counts as its own draw, extending the
#                   effective record to n_years * n_members. Matches the
#                   "treat all data points as independent" assumption
#                   this analysis is built on; reforecast return periods
#                   will come out much shorter than ERA5's for the same
#                   magnitude, which is expected under this framework.
#   "match_era5" -> rate = (n_exceedances / n_members) / n_years. Members
#                   only sharpen the threshold/tail estimate; the annual
#                   occurrence rate is kept equal to ERA5's so the two
#                   curves are directly comparable at face value.
RATE_MODE = "unseen"

RETURN_PERIODS_PLOT = np.logspace(0, 4, 400)  # 1-10,000 years

# Wong/Okabe-Ito colourblind-safe palette, matches plot_variance_figures.py
COL = {"era5": "#009E73", "reforecast": "#D55E00", "reference": "#000000"}

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


def _in_target_window(mmdd: pd.Series) -> pd.Series:
    if TARGET_DATES is not None:
        return mmdd.isin(TARGET_DATES)
    return (mmdd >= VALID_START_MMDD) & (mmdd <= VALID_END_MMDD)


def load_era5(path: str) -> pd.DataFrame:
    """Load ERA5 t2m, subset to the target dates, return long-format df."""
    era5 = xr.open_dataset(path)
    df = (
        era5.to_dataframe()
        .reset_index()
        .rename(columns={"time": "date", "t2m": "t2m_era5"})
    )
    df["date"] = pd.to_datetime(df["date"])
    df["valid_mmdd"] = df["date"].dt.strftime("%m-%d")
    df = df[_in_target_window(df["valid_mmdd"])].copy()
    df["year"] = df["date"].dt.year
    return df.dropna(subset=["t2m_era5"])


def load_reforecast_lead(path: str, lead_day: int) -> pd.DataFrame:
    """Load the reforecast csv, subset to one lead day and the target dates."""
    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df = df[df["days"] == lead_day].copy()
    df["valid_mmdd"] = df["forecast_date"].dt.strftime("%m-%d")
    df = df[_in_target_window(df["valid_mmdd"])].copy()
    df["year"] = df["forecast_date"].dt.year
    return df.dropna(subset=["t2m"])


def fit_pot_gumbel(
    values: np.ndarray, n_years: int, n_members: int = 1, rate_mode: str = "unseen"
) -> dict:
    """
    Fit a Gumbel distribution to the top (100 - THRESHOLD_PERCENTILE)% of
    values, and derive an annual occurrence rate for those exceedances
    under the given rate_mode. See RATE_MODE comment above.
    """
    values = np.asarray(values)
    threshold = np.percentile(values, THRESHOLD_PERCENTILE)
    exceed = np.sort(values[values >= threshold])
    m = len(exceed)

    if rate_mode == "unseen":
        rate = m / n_years
    elif rate_mode == "match_era5":
        rate = (m / n_members) / n_years
    else:
        raise ValueError(f"unknown rate_mode: {rate_mode!r}")

    loc, scale = gumbel_r.fit(exceed)
    return {
        "threshold": threshold,
        "exceedances": exceed,
        "m": m,
        "n_years": n_years,
        "n_members": n_members,
        "rate_mode": rate_mode,
        "rate": rate,  # exceedances of `threshold` per year
        "loc": loc,
        "scale": scale,
    }


def empirical_return_periods(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """Weibull plotting-position return periods for the raw exceedances."""
    m = fit["m"]
    ranks = np.arange(1, m + 1)
    survival = (m + 1 - ranks) / (m + 1)  # P(X > x_i | X > threshold)
    T = 1.0 / (fit["rate"] * survival)
    return T, fit["exceedances"]


def fitted_return_levels(fit: dict, return_periods: np.ndarray) -> np.ndarray:
    """Gumbel-fit return level x(T) for each return period T (years)."""
    q = 1.0 / (fit["rate"] * return_periods)  # P(X > x) implied by T
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return gumbel_r.ppf(1 - q, loc=fit["loc"], scale=fit["scale"])


def plot_return_periods(fit_era5: dict, fit_rf: dict):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    T_emp_era5, x_emp_era5 = empirical_return_periods(fit_era5)
    T_emp_rf, x_emp_rf = empirical_return_periods(fit_rf)

    ax.scatter(
        T_emp_era5, x_emp_era5, s=16, color=COL["era5"], marker="^",
        label="ERA5 (empirical)", zorder=3,
    )
    ax.scatter(
        T_emp_rf, x_emp_rf, s=10, color=COL["reforecast"], marker="o", alpha=0.5,
        label=f"Reforecast day {LEAD_DAY} (empirical)", zorder=2,
    )

    T_fit = RETURN_PERIODS_PLOT
    ax.plot(
        T_fit, fitted_return_levels(fit_era5, T_fit), color=COL["era5"],
        linewidth=1.4, label="ERA5 (Gumbel fit)",
    )
    ax.plot(
        T_fit, fitted_return_levels(fit_rf, T_fit), color=COL["reforecast"],
        linewidth=1.4, linestyle="--", label=f"Reforecast day {LEAD_DAY} (Gumbel fit)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(r"t2m ($^\circ$C)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def summarize(name: str, fit: dict) -> dict:
    return {
        "dataset": name,
        "n_years": fit["n_years"],
        "n_members": fit["n_members"],
        "rate_mode": fit["rate_mode"],
        "threshold_pct": THRESHOLD_PERCENTILE,
        "threshold": fit["threshold"],
        "n_exceedances": fit["m"],
        "rate_per_year": fit["rate"],
        "gumbel_loc": fit["loc"],
        "gumbel_scale": fit["scale"],
    }


def main():
    era5_df = load_era5(ERA5_PATH)
    rf_df = load_reforecast_lead(MEDIUM_RANGE_CSV, LEAD_DAY)

    n_years_era5 = era5_df["year"].nunique()
    n_years_rf = rf_df["year"].nunique()
    n_members_rf = rf_df["number"].nunique()

    fit_era5 = fit_pot_gumbel(era5_df["t2m_era5"].values, n_years_era5)
    fit_rf = fit_pot_gumbel(
        rf_df["t2m"].values, n_years_rf, n_members=n_members_rf, rate_mode=RATE_MODE
    )

    summary = pd.DataFrame(
        [summarize("era5", fit_era5), summarize(f"reforecast_day{LEAD_DAY}", fit_rf)]
    )
    summary.to_csv(OUT_CSV, index=False)
    print(summary.to_string(index=False))

    fig = plot_return_periods(fit_era5, fit_rf)
    fig.savefig(f"{OUT_BASENAME}.pdf")
    fig.savefig(f"{OUT_BASENAME}.png")
    print(f"\nSaved -> {OUT_CSV}, {OUT_BASENAME}.pdf, {OUT_BASENAME}.png")


if __name__ == "__main__":
    main()
