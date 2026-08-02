"""
Gumbel POT return periods: ERA5 vs. reforecast (single lead day).

Fits a Gumbel distribution to the top 5% (values above the 95th
percentile) of ERA5 t2m and reforecast t2m at a fixed lead day, pooled
across all years and (for the reforecast) all ensemble members, then
converts fitted and empirical exceedance probabilities into return
periods in years. 90% bootstrap confidence bands are shown around each
fitted curve (see bootstrap_return_level_ci).

The reforecast's valid dates at the chosen lead day (forecast_date =
inidate + lead_day) are taken as the definition of "the same dates" --
ERA5 is subset to exactly that set of calendar (month, day) values, not a
broader date window, so the two datasets are compared on the same 11
valid dates x 20 years.

Annual occurrence rate (RATE_MODE)
-----------------------------------
Ensemble members are alternate, equally-plausible realizations of the
*same* year, not extra calendar years -- so the correct normalization for
"the reforecast has n_years x n_members independent samples" (as stated
for this analysis, in the spirit of the UNSEEN approach, e.g. Thompson et
al. 2017, Kelder et al. 2020) is to divide exceedance counts by
n_years x n_members, not by n_years alone. Dividing by n_years alone would
claim extreme days become n_members times more frequent per real calendar
year, which members don't represent.

Concretely: with m exceedances pooled from n_years x n_members points,
    "unseen"            -> rate = m / (n_years * n_members)   [default]
    "per_calendar_year"  -> rate = m / n_years                [kept only
                             for comparison -- overstates annual frequency]
The two give the same reforecast curve shape (same Gumbel fit, since that
doesn't depend on rate_mode) but very different x-axis placement: e.g. the
single hottest pooled reforecast value sits at roughly (n_years*n_members)
years under "unseen" vs. only about n_years years under
"per_calendar_year" -- because "per_calendar_year" is, empirically, always
close to n_years for the max point regardless of how many exceedances
went into the fit (it's the n_members-fold rate inflation that pulls it
back down to a ~n_years return period despite the deeper pooled sample).
ERA5 has n_members = 1, so both modes agree for ERA5.

Distribution (--dist)
---------------------
"gumbel" (default) fits a plain, unconditional Gumbel by MLE to the top
5%, which is what was originally asked for here, but it is an
approximation -- a Gumbel fit to an already-thresholded sample is a
biased estimator of the unconditional distribution. "gpd" instead fits
the textbook peaks-over-threshold model: a generalized Pareto to the
threshold exceedances (values - threshold), which is the distribution
POT theory actually motivates. See fit_pot.

Reforecast bias correction (--recompute-lead-bias, compute_lead_bias)
-----------------------------------------------------------------------
The bias-corrected column originally shipped in the reforecast csv
(REFORECAST_VALUE_COL default, "adjusted_t2m") was verified to be
computed incorrectly for this analysis: it comes from a linear
regression of ERA5 against the reforecast's *lead day 0* ensemble mean
(intercept only, slope discarded), then applied uniformly to every lead
day including 12. Two problems, found by tracing the exact code that
produced it: (1) it's a lead-0 bias applied to lead-12 data, and lead-day
bias is not obviously stable -- the entire reason lead day 12 was chosen
elsewhere in this project is that the ensemble's statistics *change*
with lead time; (2) at lead day 12 specifically, the ERA5-vs-model
regression this bias comes from is very weak (checked: r as low as 0.12,
slope as low as 0.07), so a regression intercept isn't a meaningful bias
at all -- it's an artifact of extrapolating a near-flat line. On the
verified data this pushed the reforecast curve up by 1-2 degC across the
whole tail, changing the 39.5 degC crossing from ~84,000 to ~141,000
years once fixed.

compute_lead_bias() instead computes a plain mean-climatology difference
(model mean - ERA5 mean, both evaluated at the correct lead day and
calendar date) per inidate, robust regardless of correlation strength.
Pass --recompute-lead-bias to use it instead of REFORECAST_VALUE_COL.

Empirical tail-slope cross-check (empirical_tail_slope, --empirical-slope-plot)
---------------------------------------------------------------------------------
A second, fully distribution-free cross-check on the POT Gumbel/GPD fit:
linear regression of ln P(X > x | X > threshold) against magnitude x,
using the top 5% points directly (Weibull plotting position), no
Gumbel/GPD assumption at all. The slope translates into a multiplicative
factor: each +1 degree multiplies the within-tail exceedance probability
by exp(slope). MODEL_COLORS/plot_empirical_slopes are written to take an
arbitrary list of named datasets, so additional models (e.g. CMIP) can be
added without restructuring -- see plot_empirical_slopes' docstring.

Inputs
------
ERA5_PATH   NetCDF with a `time` coordinate and a `t2m` variable (daily,
            already box-averaged -- e.g. pnw_box_era5.nc)
REFORE_CSV  reforecast csv with columns: inidate, hDate, number, t2m,
            days (lead day), forecast_date, and optionally a
            bias-corrected column (see REFORECAST_VALUE_COL)

Outputs
-------
gumbel_return_period_summary.csv   fitted parameters + rates, both datasets
gumbel_return_periods.pdf/.png     return period plot (log10-scaled x-axis).
                                    Reference-value crossings, if requested,
                                    are reported in the on-plot summary box
                                    in both plain and scientific notation.

Usage
-----
python fit_gumbel_return_periods.py \
    --era5-path pnw_box_era5.nc --reforecast-csv reforecast_0_t0_13.csv
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, genpareto, genextreme

# ── Defaults (matches scripts/compute_variance.py; override via CLI) ───────
ERA5_PATH = "/network/group/aopp/predict/AWH020_AYIM_EXTREME/ERA5/era5_t2m/pnw_box_era5.nc"
MEDIUM_RANGE_CSV = "reforecast_0_t0_13.csv"

OUT_CSV = "gumbel_return_period_summary.csv"
OUT_BASENAME = "gumbel_return_periods"

LEAD_DAY = 12  # ERA5 convergence day, see scripts/READMe.md

# Reforecast column to analyse. Falls back to "t2m" if this column isn't
# present. Use "t2m" explicitly instead if you want the raw, un-bias-
# corrected reforecast.
REFORECAST_VALUE_COL = "adjusted_t2m"

THRESHOLD_PERCENTILE = 95  # top 5%

# See "Annual occurrence rate" in the module docstring.
RATE_MODE = "unseen"

RETURN_PERIODS_PLOT = np.logspace(0, 4, 400)  # 1-10,000 years

N_BOOTSTRAP = 1000
CI_LEVEL = 0.90  # 90% bootstrap band (5th-95th percentile)
BOOTSTRAP_SEED = 0

COL = {
    "era5": "#ff7f0e",
    "era5_ci": "#ffbb78",
    "reforecast": "#1f77b4",
    "reforecast_ci": "#aec7e8",
    "reference": "#000000",
}
GRID_COLOR = "#d3d3d3"

# tab10/tab20-style (dark, light) pairs for empirical_tail_slope datasets.
# era5/reforecast match COL above; four pre-assigned slots are ready for
# CMIP models -- add more keys here (same {"color", "ci", "marker"} shape)
# rather than hardcoding new colors at the call site.
MODEL_COLORS = {
    "era5": {"color": "#ff7f0e", "ci": "#ffbb78", "marker": "^"},
    "reforecast": {"color": "#1f77b4", "ci": "#aec7e8", "marker": "o"},
    "cmip_model_1": {"color": "#2ca02c", "ci": "#98df8a", "marker": "s"},  # green
    "cmip_model_2": {"color": "#d62728", "ci": "#ff9896", "marker": "D"},  # red
    "cmip_model_3": {"color": "#9467bd", "ci": "#c5b0d5", "marker": "P"},  # purple
    "cmip_model_4": {"color": "#e377c2", "ci": "#f7b6d2", "marker": "X"},  # pink
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


def load_reforecast_lead(path: str, lead_day: int, value_col: str) -> pd.DataFrame:
    """Load the reforecast csv and subset to one lead day."""
    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df = df[df["days"] == lead_day].copy()
    df["valid_mmdd"] = df["forecast_date"].dt.strftime("%m-%d")
    df["year"] = df["forecast_date"].dt.year

    col = value_col if value_col in df.columns else "t2m"
    df = df.rename(columns={col: "value"})
    return df.dropna(subset=["value"])


def load_era5(path: str, target_mmdd: set = None) -> pd.DataFrame:
    """
    Load ERA5 t2m, optionally subset to the given calendar (month, day)
    values. target_mmdd=None returns every calendar day across all years
    unfiltered -- needed by compute_lead_bias, which looks up ERA5
    climatology for whatever calendar date each inidate's lead day lands on.
    """
    era5 = xr.open_dataset(path)
    df = (
        era5.to_dataframe()
        .reset_index()
        .rename(columns={"time": "date", "t2m": "value"})
    )
    df["date"] = pd.to_datetime(df["date"])
    df["valid_mmdd"] = df["date"].dt.strftime("%m-%d")
    if target_mmdd is not None:
        df = df[df["valid_mmdd"].isin(target_mmdd)].copy()
    df["year"] = df["date"].dt.year
    return df.dropna(subset=["value"])


def compute_lead_bias(era5_daily: pd.DataFrame, reforecast_raw: pd.DataFrame, lead_day: int) -> pd.DataFrame:
    """
    Genuine climatological mean-bias correction for the reforecast at a
    specific lead day. See the module docstring's "Reforecast bias
    correction" section for why this replaces the csv's shipped
    "adjusted_t2m" (a lead-0-derived regression intercept, invalid at
    lead 12).

    For each inidate, the model's mean climatology at `lead_day` (mean
    over ALL ensemble members x hindcast years, valid on that calendar
    day) is compared against ERA5's mean climatology for that same
    calendar day across all its years. bias = model_clim - era5_clim is
    a plain mean difference, robust regardless of how strongly the model
    and ERA5 correlate across years (unlike a regression intercept).

    era5_daily: full daily ERA5 dataframe (load_era5(path, target_mmdd=None)
                or any df with a date-like column and a t2m/value column),
                covering every calendar day across all years.
    reforecast_raw: the *unfiltered* reforecast dataframe (all lead days),
                columns 'inidate', 'forecast_date', 'days', 't2m'.
    lead_day: the lead day to compute bias for (e.g. 12).

    Returns columns ['inidate', 'target_mmdd', 'era5_clim', 'model_clim', 'bias'].
    Correct the model with apply_bias(), i.e. t2m - bias.
    """
    era5 = era5_daily.rename(columns={"time": "date", "t2m": "value"}) if "date" not in era5_daily.columns else era5_daily.copy()
    value_col = "value" if "value" in era5.columns else "t2m"
    era5["date"] = pd.to_datetime(era5["date"])
    era5["month"], era5["day"] = era5["date"].dt.month, era5["date"].dt.day

    lead = reforecast_raw[reforecast_raw["days"] == lead_day].copy()
    lead["inidate"] = pd.to_datetime(lead["inidate"])
    lead["forecast_date"] = pd.to_datetime(lead["forecast_date"])

    rows = []
    for inidate, sub in lead.groupby("inidate"):
        month, day = sub["forecast_date"].dt.month.iloc[0], sub["forecast_date"].dt.day.iloc[0]
        era5_clim = era5.loc[(era5["month"] == month) & (era5["day"] == day), value_col].mean()
        model_clim = sub["t2m"].mean()
        rows.append({
            "inidate": inidate, "target_mmdd": f"{month:02d}-{day:02d}",
            "era5_clim": era5_clim, "model_clim": model_clim, "bias": model_clim - era5_clim,
        })
    return pd.DataFrame(rows)


def apply_bias(reforecast_df: pd.DataFrame, bias_df: pd.DataFrame, out_col: str = "adjusted_t2m") -> pd.DataFrame:
    """
    Merge a compute_lead_bias() result onto reforecast_df (matched on
    'inidate', which must be the same dtype on both sides -- callers
    should pd.to_datetime() it first if in doubt) and subtract it from
    't2m'. Drops any pre-existing 'bias' column first to avoid a merge
    suffix collision with the csv's original (lead-0-derived) bias column.
    """
    df = reforecast_df.drop(columns=["bias"], errors="ignore").merge(
        bias_df[["inidate", "bias"]], on="inidate", how="left"
    )
    df[out_col] = df["t2m"] - df["bias"]
    return df


def fit_pot(
    values: np.ndarray,
    n_years: int,
    n_members: int = 1,
    rate_mode: str = "unseen",
    dist: str = "gumbel",
) -> dict:
    """
    Fit a distribution to the top (100 - THRESHOLD_PERCENTILE)% of values,
    and derive an annual occurrence rate for those exceedances under the
    given rate_mode. See the module docstring.

    dist:
      "gumbel" -- plain (unconditional) Gumbel MLE fit to the raw
                  exceedance values. What was originally asked for here;
                  an approximation (see module docstring caveat).
      "gpd"    -- textbook peaks-over-threshold model: a generalized
                  Pareto fit (via MLE, location fixed at 0) to the
                  exceedances *above* the threshold (values - threshold).
                  This is the distribution POT theory actually motivates,
                  since GPD is the limiting distribution of threshold
                  exceedances as the threshold rises, unlike an
                  unconditional Gumbel fit to a left-truncated sample.
    """
    values = np.asarray(values)
    threshold = np.percentile(values, THRESHOLD_PERCENTILE)
    exceed = np.sort(values[values >= threshold])
    m = len(exceed)

    if rate_mode == "unseen":
        rate = m / (n_years * n_members)
    elif rate_mode == "per_calendar_year":
        rate = m / n_years
    else:
        raise ValueError(f"unknown rate_mode: {rate_mode!r}")

    if dist == "gumbel":
        loc, scale = gumbel_r.fit(exceed)
        params = {"loc": loc, "scale": scale}
    elif dist == "gpd":
        c, loc, scale = genpareto.fit(exceed - threshold, floc=0)
        params = {"c": c, "loc": loc, "scale": scale}
    else:
        raise ValueError(f"unknown dist: {dist!r}")

    return {
        "threshold": threshold,
        "exceedances": exceed,
        "m": m,
        "n_years": n_years,
        "n_members": n_members,
        "rate_mode": rate_mode,
        "rate": rate,  # exceedances of `threshold` per effective year
        "dist": dist,
        "params": params,
    }


def fit_block_maxima(values: np.ndarray, years: np.ndarray, dist: str = "gumbel") -> dict:
    """
    Block-maxima alternative to fit_pot: instead of pooling all dates/years
    and thresholding to the top 5%, take the single highest value *within
    each year* (one block maximum per year, n_years total) and fit the
    extreme-value distribution directly to that full set -- no further
    thresholding needed, since block maxima are already extremes by
    construction. This is the classical justification for the Gumbel/GEV
    family (Fisher-Tippett-Gnedenko theorem), and uses every year's worth
    of information rather than just the ~5% of dates that happen to be
    hottest, which is the natural thing to try if a POT fit (fit_pot) is
    unstable for want of exceedances (e.g. ERA5's m=11 with only 20 years
    of daily data).

    One block maximum per year by construction means "one event per year"
    is exact, not an estimated rate -- rate = 1.0 always.

    dist: "gumbel" (2-param) or "gev" (3-param generalized extreme value,
    scipy's genextreme -- the shape parameter lets the tail depart from
    the Gumbel's fixed exponential decay, at the cost of needing more data
    to estimate reliably, same tradeoff as gpd vs. gumbel in fit_pot).
    """
    values = np.asarray(values)
    years = np.asarray(years)
    uniq_years = np.unique(years)
    block_max = np.array([values[years == y].max() for y in uniq_years])
    n_years = len(uniq_years)

    if dist == "gumbel":
        loc, scale = gumbel_r.fit(block_max)
        params = {"loc": loc, "scale": scale}
    elif dist == "gev":
        c, loc, scale = genextreme.fit(block_max)
        params = {"c": c, "loc": loc, "scale": scale}
    else:
        raise ValueError(f"unknown dist: {dist!r}")

    return {
        "threshold": block_max.min(),
        "exceedances": np.sort(block_max),
        "m": n_years,
        "n_years": n_years,
        "n_members": 1,
        "rate_mode": "block_maxima",
        "rate": 1.0,  # exactly one block maximum per year
        "dist": dist,
        "params": params,
    }


def _sf(fit: dict, x) -> np.ndarray:
    """P(X > x) under the fitted distribution."""
    d, p = fit["dist"], fit["params"]
    if d == "gumbel":
        return gumbel_r.sf(x, loc=p["loc"], scale=p["scale"])
    if d == "gev":
        return genextreme.sf(x, c=p["c"], loc=p["loc"], scale=p["scale"])
    return genpareto.sf(np.asarray(x) - fit["threshold"], c=p["c"], loc=0, scale=p["scale"])


def _isf(fit: dict, q) -> np.ndarray:
    """x such that P(X > x) = q under the fitted distribution."""
    q = np.clip(q, 1e-300, None)
    d, p = fit["dist"], fit["params"]
    if d == "gumbel":
        return gumbel_r.isf(q, loc=p["loc"], scale=p["scale"])
    if d == "gev":
        return genextreme.isf(q, c=p["c"], loc=p["loc"], scale=p["scale"])
    return fit["threshold"] + genpareto.isf(q, c=p["c"], loc=0, scale=p["scale"])


def empirical_return_periods(fit: dict):
    """Weibull plotting-position return periods for the raw exceedances."""
    m = fit["m"]
    ranks = np.arange(1, m + 1)
    survival = (m + 1 - ranks) / (m + 1)  # P(X > x_i | X > threshold)
    T = 1.0 / (fit["rate"] * survival)
    return T, fit["exceedances"]


def fitted_return_levels(fit: dict, return_periods: np.ndarray) -> np.ndarray:
    """
    Fitted return level x(T) for each return period T (years).

    T below 1/rate implies an exceedance probability q = 1/(rate*T) > 1,
    which is undefined -- there's no return level shorter than the
    average spacing between threshold exceedances. Those points are NaN
    (dropped by the plot) rather than clipped, which would otherwise
    produce a spurious plunge toward the fitted distribution's tail
    (unbounded below, for a Gumbel fit).
    """
    q = 1.0 / (fit["rate"] * return_periods)  # P(X > x) implied by T
    x = _isf(fit, np.clip(q, None, 1 - 1e-12))
    return np.where(q < 1, x, np.nan)


def format_scientific(x: float) -> str:
    """'8.4e+04' -> mathtext '$8.4\\times10^{4}$', for the summary box."""
    if not np.isfinite(x):
        return r"$\infty$"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exp
    return rf"${mantissa:.1f}\times10^{{{exp}}}$"


def return_period_for_value(fit: dict, x: float) -> float:
    """Invert fitted_return_levels: the fitted return period implied by value x."""
    q = float(_sf(fit, x))  # P(X > x)
    if q <= 0:
        return np.inf
    return 1.0 / (fit["rate"] * q)


def bootstrap_return_level_ci(
    values: np.ndarray,
    n_years: int,
    n_members: int,
    rate_mode: str,
    return_periods: np.ndarray,
    dist: str = "gumbel",
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
):
    """
    Case-resampling bootstrap CI for the fitted return-level curve.

    Each replicate resamples the pooled values with replacement (same
    size as the original), then re-selects the top-5% threshold, refits
    the distribution, and recomputes the return-level curve -- so the
    band reflects both sampling uncertainty in the fitted parameters and
    in the threshold itself. Consistent with this analysis treating all
    pooled points as independent draws.
    """
    values = np.asarray(values)
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_levels = np.full((n_boot, len(return_periods)), np.nan)

    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        try:
            fit_b = fit_pot(sample, n_years, n_members, rate_mode, dist=dist)
            if fit_b["m"] < 2:
                continue
            boot_levels[b] = fitted_return_levels(fit_b, return_periods)
        except Exception:
            continue

    lo_pct, hi_pct = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    with np.errstate(invalid="ignore"):
        lower = np.nanpercentile(boot_levels, lo_pct, axis=0)
        upper = np.nanpercentile(boot_levels, hi_pct, axis=0)
    return lower, upper


def bootstrap_return_level_ci_block_maxima(
    block_max_values: np.ndarray,
    dist: str,
    return_periods: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
):
    """Case-resampling bootstrap CI for a fit_block_maxima curve (rate fixed at 1/yr)."""
    block_max_values = np.asarray(block_max_values)
    n = len(block_max_values)
    rng = np.random.default_rng(seed)
    boot_levels = np.full((n_boot, len(return_periods)), np.nan)

    for b in range(n_boot):
        sample = rng.choice(block_max_values, size=n, replace=True)
        try:
            if dist == "gumbel":
                loc, scale = gumbel_r.fit(sample)
                params = {"loc": loc, "scale": scale}
            else:
                c, loc, scale = genextreme.fit(sample)
                params = {"c": c, "loc": loc, "scale": scale}
            fit_b = {"dist": dist, "params": params, "rate": 1.0, "threshold": sample.min()}
            boot_levels[b] = fitted_return_levels(fit_b, return_periods)
        except Exception:
            continue

    lo_pct, hi_pct = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    with np.errstate(invalid="ignore"):
        lower = np.nanpercentile(boot_levels, lo_pct, axis=0)
        upper = np.nanpercentile(boot_levels, hi_pct, axis=0)
    return lower, upper


def plot_return_periods(
    fit_era5: dict,
    fit_rf: dict,
    era5_values: np.ndarray,
    rf_values: np.ndarray,
    lead_day: int,
    fit_era5_sensitivity: dict = None,
    era5_sensitivity_values: np.ndarray = None,
    sensitivity_label: str = None,
    fit_era5_block_maxima: dict = None,
    era5_block_maxima_values: np.ndarray = None,
    block_maxima_label: str = None,
    reference_value: float = None,
    reference_label: str = None,
    xscale: str = "log",
):
    """
    Return-period plot for both datasets: empirical (plotting-position)
    points, Gumbel-fit curves, and a shaded bootstrap CI band per curve.
    era5_values / rf_values are the full pooled arrays (not just the
    exceedances) since the bootstrap re-selects its own threshold on
    each resample.

    xscale: "log" (return period axis in powers of 10) or "linear"
    (actual year values).

    fit_era5_sensitivity (optional): a second ERA5 fit -- e.g. with its
    single most extreme point dropped -- drawn as a dotted overlay so the
    two ERA5 curves can be compared directly for sensitivity to that point.

    fit_era5_block_maxima (optional): a fit_block_maxima result -- drawn as
    a dash-dot overlay, for comparing the POT fit against fitting directly
    to one maximum per year (era5_block_maxima_values must be the block
    maxima array itself, for bootstrapping).

    reference_value (optional): draws a horizontal line (e.g. the 2021 PNW
    heatwave's observed peak) and marks/annotates the return period each
    fitted curve implies for it.
    """
    curves = [(fit_era5, era5_values, COL["era5"], COL["era5_ci"], "-", "ERA5")]
    if fit_era5_sensitivity is not None:
        curves.append(
            (fit_era5_sensitivity, era5_sensitivity_values, "#555555", "#cccccc", ":",
             sensitivity_label or "ERA5 (sensitivity)")
        )
    if fit_era5_block_maxima is not None:
        curves.append(
            (fit_era5_block_maxima, era5_block_maxima_values, "#8c564b", "#c49c94", "-.",
             block_maxima_label or "ERA5 (block maxima)")
        )
    curves.append(
        (fit_rf, rf_values, COL["reforecast"], COL["reforecast_ci"], "--",
         f"Reforecast day {lead_day}")
    )

    # Extend the return-period axis far enough to show every curve's
    # crossing of reference_value on-chart, up to a sanity cap (curves that
    # stay far more extreme than the cap -- e.g. ERA5's POT fit against an
    # unprecedented heatwave, in the hundreds of millions of years -- are
    # reported in the summary box below instead of stretching the axis by
    # 6+ orders of magnitude to reach them, which would crush everything
    # else into an unreadable sliver near the left edge).
    AXIS_EXTENSION_CAP = 1e6
    T_max_plot = RETURN_PERIODS_PLOT.max()
    if reference_value is not None:
        finite_T_ref = [
            return_period_for_value(fit, reference_value) for fit, *_ in curves
        ]
        in_range = [t for t in finite_T_ref if np.isfinite(t) and t <= AXIS_EXTENSION_CAP]
        if in_range:
            T_max_plot = max(T_max_plot, max(in_range) * 1.3)
    T_fit = np.logspace(0, np.log10(T_max_plot), 400)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for fit, values, _, ci_color, _, _ in curves:
        if fit["rate_mode"] == "block_maxima":
            lower, upper = bootstrap_return_level_ci_block_maxima(values, fit["dist"], T_fit)
        else:
            lower, upper = bootstrap_return_level_ci(
                values, fit["n_years"], fit["n_members"], fit["rate_mode"], T_fit, dist=fit["dist"]
            )
        ax.fill_between(T_fit, lower, upper, color=ci_color, alpha=0.6, linewidth=0, zorder=1)

    T_emp_era5, x_emp_era5 = empirical_return_periods(fit_era5)
    T_emp_rf, x_emp_rf = empirical_return_periods(fit_rf)

    ax.scatter(
        T_emp_era5, x_emp_era5, s=16, color=COL["era5"], marker="^",
        label="ERA5 (empirical)", zorder=3,
    )
    ax.scatter(
        T_emp_rf, x_emp_rf, s=10, color=COL["reforecast"], marker="o", alpha=0.5,
        label=f"Reforecast day {lead_day} (empirical)", zorder=2,
    )

    dist_label = {"gumbel": "Gumbel", "gpd": "GPD", "gev": "GEV"}
    for fit, _, color, _, linestyle, label in curves:
        ax.plot(
            T_fit, fitted_return_levels(fit, T_fit), color=color,
            linewidth=1.4, linestyle=linestyle,
            label=f"{label} ({dist_label[fit['dist']]} fit)", zorder=4,
        )
    ax.plot([], [], color=COL["reference"], alpha=0.25, linewidth=8,
            label=f"{int(CI_LEVEL * 100)}% bootstrap CI")

    if reference_value is not None:
        ax.axhline(reference_value, color=COL["reference"], linewidth=1, linestyle="-.", zorder=5)
        ax.text(
            T_fit.max() * 0.7, reference_value, reference_label or f"{reference_value:g}",
            fontsize=8, color=COL["reference"], va="bottom", ha="right",
        )

        ref_desc = reference_label or f"{reference_value:g}"
        lines = [f"Return period implied by {ref_desc}:"]
        for fit, _, color, _, _, label in curves:
            T_ref = return_period_for_value(fit, reference_value)
            if np.isfinite(T_ref) and T_ref <= T_fit.max():
                ax.scatter([T_ref], [reference_value], color=color, marker="x", s=45, zorder=6)
                lines.append(f"{label}: {T_ref:,.0f} yr ({format_scientific(T_ref)} yr)")
            elif np.isfinite(T_ref):
                lines.append(f"{label}: {T_ref:,.0f} yr ({format_scientific(T_ref)} yr, off-chart)")
            else:
                lines.append(f"{label}: never (beyond fit's upper support)")
        ax.text(
            0.98, 0.03, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, color=COL["reference"], ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
        )

    ax.set_xscale(xscale)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(r"t2m ($^\circ$C)")
    if reference_value is not None:
        ax.set_ylim(top=max(ax.get_ylim()[1], reference_value + 1))
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if reference_value is not None:
        # An in-axes legend can land right on top of the reference line/label
        # (e.g. a high reference_value pushes the horizontal line and its
        # text up into the legend's usual upper-left corner) -- move the
        # legend above the axes entirely so it can never collide with
        # anything plotted inside, regardless of where reference_value falls.
        ax.legend(
            frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
            loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=7,
        )
    else:
        ax.legend(
            frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
            loc="upper left", fontsize=7,
        )
    fig.tight_layout()
    return fig


def empirical_tail_slope(values: np.ndarray, threshold_percentile: float = THRESHOLD_PERCENTILE) -> dict:
    """
    Distribution-free cross-check on the POT fit: linear regression of
    ln P(X > x | X > threshold) against magnitude x, using the top
    (100 - threshold_percentile)% of values directly (Weibull plotting
    position) -- no Gumbel/GPD assumption. The slope translates into a
    multiplicative factor: each +1 degree multiplies the within-tail
    exceedance probability by exp(slope), i.e. "exp(slope)x as likely per
    +1 degree" or equivalently "1/exp(slope) x rarer per +1 degree".
    """
    values = np.asarray(values)
    threshold = np.percentile(values, threshold_percentile)
    exceed = np.sort(values[values >= threshold])
    m = len(exceed)
    ranks = np.arange(1, m + 1)
    survival = (m + 1 - ranks) / (m + 1)
    log_survival = np.log(survival)

    slope, intercept = np.polyfit(exceed, log_survival, 1)
    pred = slope * exceed + intercept
    ss_res = np.sum((log_survival - pred) ** 2)
    ss_tot = np.sum((log_survival - log_survival.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "threshold": threshold, "exceedances": exceed, "log_survival": log_survival,
        "m": m, "slope": slope, "intercept": intercept, "r2": r2,
        "factor_per_plus1degC": np.exp(slope), "factor_per_minus1degC": np.exp(-slope),
    }


def plot_empirical_slopes(
    datasets: list, xlabel: str = "P(X > x | X > threshold)", ylabel: str = r"t2m ($^\circ$C)"
):
    """
    empirical_tail_slope for an arbitrary list of named datasets, plotted
    together: x-axis = within-tail exceedance probability (log scale,
    inverted so rarer sits to the right, matching plot_return_periods'
    convention), y-axis = magnitude.

    datasets: list of dicts, each with keys:
        name (str), values (array), color (str), marker (str, optional),
        threshold_percentile (float, optional, defaults to THRESHOLD_PERCENTILE)
    To add a CMIP model, append another dict -- e.g.
        datasets.append({"name": "CMIP6 model A", **MODEL_COLORS["cmip_model_1"],
                          "values": cmip1_values})
    Mutates each dict in place with a "_fit" key (the empirical_tail_slope
    result) so the caller can build a summary table afterward.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in datasets:
        fit = empirical_tail_slope(ds["values"], ds.get("threshold_percentile", THRESHOLD_PERCENTILE))
        c, marker = ds["color"], ds.get("marker", "o")
        prob = np.exp(fit["log_survival"])
        ax.scatter(
            prob, fit["exceedances"], s=12, color=c, marker=marker, alpha=0.6,
            label=f"{ds['name']} (empirical)",
        )
        logP_line = np.linspace(fit["log_survival"].min(), fit["log_survival"].max(), 100)
        T_line = (logP_line - fit["intercept"]) / fit["slope"]
        ax.plot(
            np.exp(logP_line), T_line, color=c, linewidth=1.4, linestyle="--",
            label=f"{ds['name']}: {fit['factor_per_plus1degC']:.2f}x per +1$^\\circ$C (R$^2$={fit['r2']:.2f})",
        )
        ds["_fit"] = fit

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
        fontsize=7, loc="upper left",
    )
    fig.tight_layout()
    return fig, datasets


def summarize(name: str, fit: dict) -> dict:
    return {
        "dataset": name,
        "n_years": fit["n_years"],
        "n_members": fit["n_members"],
        "rate_mode": fit["rate_mode"],
        "threshold_pct": None if fit["rate_mode"] == "block_maxima" else THRESHOLD_PERCENTILE,
        "threshold": fit["threshold"],
        "n_exceedances": fit["m"],
        "rate_per_year": fit["rate"],
        "dist": fit["dist"],
        **{f"param_{k}": v for k, v in fit["params"].items()},
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5-path", default=ERA5_PATH)
    p.add_argument("--reforecast-csv", default=MEDIUM_RANGE_CSV)
    p.add_argument("--lead-day", type=int, default=LEAD_DAY)
    p.add_argument("--value-col", default=REFORECAST_VALUE_COL)
    p.add_argument(
        "--rate-mode", default=RATE_MODE, choices=["unseen", "per_calendar_year"]
    )
    p.add_argument(
        "--dist", default="gumbel", choices=["gumbel", "gpd"],
        help="gumbel: unconditional Gumbel MLE fit to the top 5%% (as originally asked). "
             "gpd: textbook POT model, generalized Pareto fit to the threshold exceedances.",
    )
    p.add_argument("--out-prefix", default=None, help="directory/prefix for outputs")
    p.add_argument(
        "--drop-era5-max", action="store_true",
        help="also fit ERA5 with its single highest value removed, overlaid for comparison",
    )
    p.add_argument(
        "--era5-block-maxima", action="store_true",
        help="also fit ERA5 using one maximum per year (20 points, no POT threshold) instead "
             "of the pooled top-5%% -- a check for whether limited POT exceedance counts "
             "(e.g. m=11) are driving instability, overlaid for comparison",
    )
    p.add_argument(
        "--block-maxima-dist", default="gumbel", choices=["gumbel", "gev"],
        help="distribution for --era5-block-maxima: gumbel (2-param) or gev (3-param, scipy "
             "genextreme)",
    )
    p.add_argument(
        "--reference-value", type=float, default=None,
        help="draw a horizontal line at this value and report the return period each "
             "fitted curve implies for it (e.g. 39.5 for the 2021 PNW heatwave peak)",
    )
    p.add_argument("--reference-label", default=None)
    p.add_argument(
        "--recompute-lead-bias", action="store_true",
        help="ignore --value-col / the csv's shipped bias column and instead compute a "
             "proper lead-day-specific mean-climatology bias correction (compute_lead_bias) "
             "-- see the module docstring; the shipped 'adjusted_t2m' was verified incorrect "
             "for lead-12 analysis",
    )
    p.add_argument(
        "--empirical-slope-plot", action="store_true",
        help="also produce a distribution-free tail-slope cross-check plot "
             "(empirical_tail_slope / plot_empirical_slopes), saved as "
             "{out_base}_empirical_slope.png/.pdf",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_csv = f"{args.out_prefix}{OUT_CSV}" if args.out_prefix else OUT_CSV
    out_base = f"{args.out_prefix}{OUT_BASENAME}" if args.out_prefix else OUT_BASENAME

    if args.recompute_lead_bias:
        # Bypasses --value-col / load_reforecast_lead's column selection
        # entirely: builds rf_df straight from the raw csv plus a freshly
        # computed lead-day-specific mean bias. See the module docstring's
        # "Reforecast bias correction" section for why the csv's shipped
        # "adjusted_t2m" isn't trustworthy for lead-12 analysis.
        era5_full = load_era5(args.era5_path, target_mmdd=None)
        reforecast_raw = pd.read_csv(args.reforecast_csv)
        bias_df = compute_lead_bias(era5_full, reforecast_raw, args.lead_day)

        reforecast_raw["forecast_date"] = pd.to_datetime(reforecast_raw["forecast_date"])
        reforecast_raw["inidate"] = pd.to_datetime(reforecast_raw["inidate"])
        rf_df = reforecast_raw[reforecast_raw["days"] == args.lead_day].copy()
        rf_df = apply_bias(rf_df, bias_df, out_col="value")
        rf_df["valid_mmdd"] = rf_df["forecast_date"].dt.strftime("%m-%d")
        rf_df["year"] = rf_df["forecast_date"].dt.year
        rf_df = rf_df.dropna(subset=["value"])

        print("Recomputed lead-day-specific mean bias (model_clim - era5_clim):")
        print(bias_df[["target_mmdd", "era5_clim", "model_clim", "bias"]].to_string(index=False))
    else:
        rf_df = load_reforecast_lead(args.reforecast_csv, args.lead_day, args.value_col)

    target_mmdd = set(rf_df["valid_mmdd"].unique())
    era5_df = load_era5(args.era5_path, target_mmdd)

    n_years_era5 = era5_df["year"].nunique()
    n_years_rf = rf_df["year"].nunique()
    n_members_rf = rf_df["number"].nunique()

    era5_values = era5_df["value"].values
    rf_values = rf_df["value"].values

    fit_era5 = fit_pot(era5_values, n_years_era5, dist=args.dist)
    fit_rf = fit_pot(
        rf_values, n_years_rf, n_members=n_members_rf, rate_mode=args.rate_mode, dist=args.dist
    )

    summaries = [summarize("era5", fit_era5), summarize(f"reforecast_day{args.lead_day}", fit_rf)]

    fit_era5_sens, era5_sens_values, sens_label = None, None, None
    if args.drop_era5_max:
        drop_idx = np.argmax(era5_values)
        dropped_value = era5_values[drop_idx]
        era5_sens_values = np.delete(era5_values, drop_idx)
        fit_era5_sens = fit_pot(era5_sens_values, n_years_era5, dist=args.dist)
        sens_label = "ERA5 (excl. max)"
        summaries.append(summarize("era5_excl_max", fit_era5_sens))
        print(f"Dropped ERA5 max: {dropped_value:.3f} (of {len(era5_values)} points)")

    fit_era5_bm, era5_bm_values, bm_label = None, None, None
    if args.era5_block_maxima:
        fit_era5_bm = fit_block_maxima(
            era5_values, era5_df["year"].values, dist=args.block_maxima_dist
        )
        era5_bm_values = fit_era5_bm["exceedances"]  # the block maxima themselves, sorted
        bm_label = "ERA5 (block maxima)"
        summaries.append(summarize("era5_block_maxima", fit_era5_bm))
        print(f"ERA5 block maxima: {fit_era5_bm['m']} years -> {sorted(era5_bm_values.round(2))}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_csv, index=False)
    print(f"ERA5: n={len(era5_df)} over {n_years_era5} yr, {len(target_mmdd)} dates/yr")
    print(f"Reforecast: n={len(rf_df)} over {n_years_rf} yr, {n_members_rf} members")
    print(summary.to_string(index=False))

    if args.reference_value is not None:
        curves = [("ERA5", fit_era5)]
        if fit_era5_sens is not None:
            curves.append((sens_label, fit_era5_sens))
        if fit_era5_bm is not None:
            curves.append((bm_label, fit_era5_bm))
        curves.append((f"Reforecast day {args.lead_day}", fit_rf))
        print(f"\nReturn period implied by {args.reference_value:g}:")
        for label, fit in curves:
            T_ref = return_period_for_value(fit, args.reference_value)
            if np.isfinite(T_ref):
                exp = int(np.floor(np.log10(T_ref))) if T_ref > 0 else 0
                print(f"  {label}: {T_ref:,.1f} yr ({T_ref / 10**exp:.1f}e{exp:+03d} yr)")
            else:
                print(f"  {label}: never (below fit support)")

    fig = plot_return_periods(
        fit_era5, fit_rf, era5_values, rf_values, args.lead_day,
        fit_era5_sensitivity=fit_era5_sens, era5_sensitivity_values=era5_sens_values,
        sensitivity_label=sens_label,
        fit_era5_block_maxima=fit_era5_bm, era5_block_maxima_values=era5_bm_values,
        block_maxima_label=bm_label,
        reference_value=args.reference_value, reference_label=args.reference_label,
    )
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png")
    print(f"\nSaved -> {out_csv}, {out_base}.pdf, {out_base}.png")

    if args.empirical_slope_plot:
        datasets = [
            {"name": "ERA5", **MODEL_COLORS["era5"], "values": era5_values},
            {"name": f"Reforecast day {args.lead_day}", **MODEL_COLORS["reforecast"], "values": rf_values},
        ]
        if fit_era5_sens is not None:
            datasets.append({"name": sens_label, "color": "#555555", "marker": "s", "values": era5_sens_values})
        fig_slope, datasets = plot_empirical_slopes(datasets)
        fig_slope.savefig(f"{out_base}_empirical_slope.pdf")
        fig_slope.savefig(f"{out_base}_empirical_slope.png")
        print(f"\nEmpirical tail-slope cross-check ({THRESHOLD_PERCENTILE}th percentile threshold):")
        for ds in datasets:
            f = ds["_fit"]
            print(f"  {ds['name']}: m={f['m']}  slope={f['slope']:.4f}/degC  R^2={f['r2']:.3f}  "
                  f"factor(+1degC)={f['factor_per_plus1degC']:.3f}  "
                  f"=> {f['factor_per_minus1degC']:.2f}x rarer per +1degC")
        print(f"Saved -> {out_base}_empirical_slope.pdf, {out_base}_empirical_slope.png")


if __name__ == "__main__":
    main()
