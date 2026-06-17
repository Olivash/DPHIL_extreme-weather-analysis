# Variance Decomposition Analysis

This analysis decomposes reforecast temperature variance into total, intrinsic, and ERA5 components across forecast lead day, for the Pacific Northwest extreme-heat case study. It supports the lead-time justification (day 12) used in the *Journal of Climate* paper and the corresponding thesis chapter.

## Data

The working dataframe `df` is a long-format table with one row per (inidate, number, hDate, lead day) combination:

| column | meaning |
|---|---|
| `inidate` | forecast initialisation date |
| `number` | ensemble member index |
| `hDate` | hindcast year (2001–2020) |
| `t2m` | reforecast 2m temperature (°C) |
| `days` | lead day (0–13) |
| `forecast_date` | actual calendar date being verified, `base_date + days` |
| `t2m_era5` | ERA5 observed 2m temperature for `forecast_date`, merged in by date |

Currently spans 11 inidates × 11 members × 20 hDates × 14 lead days = 33,880 rows.

## Variance decomposition

For each lead day:

```python
df['ensemble_mean'] = df.groupby(['inidate', 'hDate', 'days'])['t2m'].transform('mean')
df['climatology']   = df.groupby(['days'])['t2m'].transform('mean')
df['intrinsic']     = df['t2m'] - df['ensemble_mean'] + df['climatology']

var_total_per_day     = df.groupby('days')['t2m'].var()
var_intrinsic_per_day  = df.groupby('days')['intrinsic'].var()
```

`ensemble_mean` strips out the within-(inidate, hDate) forced signal, leaving pure within-ensemble (chaotic) spread. `climatology` adds back a single reference value so the result sits on a physical temperature scale rather than as an anomaly, without altering the variance, since adding any constant to a set of values never changes its variance.

**Important: `climatology` must stay a single value per lead day (grouped only by `days`), not grouped by `inidate` or `number`.** Letting it vary by `inidate` reintroduces the seasonal cycle (June dates run cooler than July dates) into what's supposed to be pure chaos variance, inflating intrinsic variance for the wrong reason. Letting it vary by `number` assumes ensemble members carry a consistent identity across different inidates, which isn't generally true for independently-generated reforecast perturbations. Both were tried and reverted during development; keep the single-value version.

## ERA5 comparison

ERA5 has no ensemble dimension, so it must be deduplicated across `number` before computing variance, otherwise each value is counted 11 times:

```python
era5_unique = df.drop_duplicates(subset=['inidate', 'hDate', 'days'])
var_era5_per_day = era5_unique.groupby('days')['t2m_era5'].var()
```

Two ERA5 reference framings are used: the per-lead-day "varying" version above, and a single "fixed" reference (`var_era5_per_day.loc[12]`) anchored to the day-12 verification window specifically, used for the headline day-12 ratio claims since it isolates the actual target verification dates without conflating them with other lead days' calendar windows.

## Outputs

- `variance_decomposition_final.png` — two-panel figure: (a) total/intrinsic variance with ERA5 reference line, (b) intrinsic/total and intrinsic/ERA5 ratios, both vs lead day, with day-12 marked.
- `hist_day12.png` / `.pdf` — single-panel histogram comparing reforecast, intrinsic, and ERA5 (×11 weighted) distributions at lead day 12.
- `hist_all_leaddays.png` / `.pdf` — same histogram comparison gridded across all 14 lead days.

## Status

At lead day 12, intrinsic/total ≈ 0.5 and intrinsic/ERA5 ≈ 0.6–0.7, neither has reached 1. This differs from the original single-inidate JOC analysis, which found ratios closer to saturation by day 12. Before finalising the day-12 justification using this pooled, multi-inidate version, the original single-inidate case should be rerun with the same single-value climatology to check whether the discrepancy comes from pooling across inidates, or from the climatology formula used in the original analysis.
