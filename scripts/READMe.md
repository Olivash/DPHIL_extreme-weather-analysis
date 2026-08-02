# Variance decomposition: reforecast intrinsic vs ERA5 variance

This directory contains the pipeline used to compute and visualise the variance
decomposition of ECMWF reforecast ensemble members against ERA5 observations,
covering the medium range (lead days 0 to 13) and the sub seasonal to seasonal
(S2S) extension (lead days 15 to 40), for the 2021 Pacific Northwest heatwave
attribution analysis.

## Contents

### `scripts/01_compute_variance.py`

Loads the ERA5 reference dataset and both reforecast datasets (medium range and
S2S), merges them by valid date, and computes total variance, intrinsic
variance, and ERA5 variance for each forecast lead day.

Intrinsic variance isolates the within ensemble (chaotic, unpredictable)
component of forecast spread by removing the ensemble mean signal and adding
back daily climatology.

Output: `variance_summary_all.csv`

### `scripts/02_plot_final_figure.py`

Produces the publication figure showing intrinsic and ERA5 variance by lead
day, with the intrinsic/ERA5 ratio on a secondary axis and the ERA5
convergence day annotated.

### `scripts/03_plot_figures.py`

Produces two further diagnostic figures from the same `variance_summary_all.csv`:

1. A twin axis figure showing total and intrinsic variance alongside the
   intrinsic/total ratio, with the internal saturation day annotated.
2. A two panel figure showing variance (top) and all three ratios (bottom):
   total/ERA5, intrinsic/ERA5, and intrinsic/total.

This script also determines two distinct lead day criteria automatically,
rather than relying on a fixed, manually chosen day. These answer two
different scientific questions and should not be conflated:

**Internal saturation day** (`find_internal_saturation_day`)
The first lead day at which the intrinsic/total ratio settles within 5
percent of its own long lead plateau (estimated from the mean of the last 10
lead days) and stays there for 3 consecutive lead days. This ratio has no
reason to converge to 1, since intrinsic variance is always some fraction of
total variance by construction. Convergence is therefore defined relative to
the series' own plateau, not to a fixed target.

**ERA5 convergence day** (`find_era5_convergence_day`)
The first lead day at which the intrinsic/ERA5 ratio reaches 0.95. Unlike the
saturation day, this is a genuine threshold crossing: a ratio of 1 is the
exact point where intrinsic variance equals ERA5's observed variance, so this
criterion answers whether the reforecast's internal spread has caught up to
real world variability.

The ERA5 convergence day is the one relevant to the lead time justification in
the JoC paper (day 12, ratio 0.973, on the current dataset).

### `scripts/fit_gumbel_return_periods.py`

Fits a Gumbel distribution to the top 5% of ERA5 t2m and of reforecast t2m at
a single lead day (day 12 by default), pooling across the JJA valid-date
window, all years, and (for the reforecast) all ensemble members, then plots
return periods in years for both datasets on one axis, with empirical
(plotting-position) points alongside the fitted curves.

Ensemble members are treated as independent draws to extend the reforecast's
effective record length beyond its calendar years, matching this repo's
sample-size assumption for the reforecast; `RATE_MODE` at the top of the
script switches between that framework (`"unseen"`) and a mode that keeps the
reforecast's annual occurrence rate equal to ERA5's (`"match_era5"`), so the
two curves are comparable at face value.

If you have a specific list of target dates rather than the full
`06-15`-`07-21` window, set `TARGET_DATES` at the top of the script.

Output: `gumbel_return_period_summary.csv`, `gumbel_return_periods.pdf/.png`

## Requirements

```
numpy
pandas
xarray
matplotlib
scipy
```

## Usage

Run in this order, from the same working directory:

```
python scripts/compute_variance.py
python scripts/plot_variance_figures.py
python scripts/plot_variance_figures_2.py
python scripts/fit_gumbel_return_periods.py
```

Both plotting scripts read `variance_summary_all.csv`, produced by the first
script. `fit_gumbel_return_periods.py` is independent of the other three --
it reads `pnw_box_era5.nc` and `reforecast_0_t0_13.csv` directly.

## Data paths

The ERA5 and reforecast file paths at the top of `01_compute_variance.py`
point to the AOPP HPC cluster (`/network/group/aopp/predict/...`). Update
these paths if running elsewhere.

## Figure conventions

All figures follow AMS/AGU style: serif font, no in-image titles, 600 dpi PDF
and PNG output, x-ticks every 5 lead days, Wong/Okabe-Ito colourblind safe
palette.
