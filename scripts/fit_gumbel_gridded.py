"""
Gridded (per-grid-cell) empirical tail-slope / rarity-factor mapping.

Same empirical tail-slope cross-check as fit_gumbel_simple.py's
empirical_tail_slope() -- top 5% POT, Weibull plotting position, temperature
anomaly (each cell centered on its own mean first) -- but computed
INDEPENDENTLY AT EVERY GRID CELL instead of box-averaged over the PNW region
first. ERA5 and the reforecast are global; CMIP is also global now (not
restricted to the PNW box the scalar scripts used).

Vectorized via xr.apply_ufunc: a fixed top-K exceedance count (not a
per-cell percentile threshold) keeps the Weibull plotting-position y-values
identical at every cell, so the whole grid's regression reduces to one
closed-form OLS pass -- no per-cell Python loop over ~1e5-1e6 cells. Verified
against the tracked scalar empirical_tail_slope() on tiled real ERA5 data
(slope/R^2/rarity_factor matched to 4+ decimal places) and against a
synthetic spatial-pattern grid.

No cartopy in this environment -- maps render as plain lat/lon pcolormesh,
no coastlines/projection. Swap in cartopy's GeoAxes / add_feature if wanted.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
ERA5_PATH = "/network/group/aopp/predict/AWH020_AYIM_EXTREME/ERA5/era5_t2m/nick_testing/all_years.zarr"
REFORECAST_PATH = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/full_refore.zarr"
CMIP_GLOB = "Network/historical_ssp/*2001_2020.nc"
CMIP_VAR = "tasmax"
LEAD_DAY = 12  # must match the reforecast zarr's fixed lead (checked at runtime)

THRESHOLD_PERCENTILE = 95  # top 5%, same convention as the scalar scripts
ANOMALY = True  # center each cell on its own time-mean first (matches empirical_tail_slope)

# Spatial chunk sizes for dask parallelism (tune to your cluster)
CHUNK_LAT = 90
CHUNK_LON = 180

# PNW box, for a sanity cross-check against the scalar (box-averaged) results
CMIP_LAT_BOUNDS = (45, 52)
CMIP_LON_BOUNDS = (-123, -119)

OUT_PREFIX = "gridded"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ── Vectorized per-cell empirical tail-slope ────────────────────────────────

def empirical_tail_slope_grid(data: xr.DataArray, sample_dim: str,
                               threshold_percentile: float = THRESHOLD_PERCENTILE,
                               anomaly: bool = ANOMALY) -> xr.Dataset:
    """
    Per-cell version of fit_gumbel_simple.empirical_tail_slope(). `data` must
    already be filtered to the sample set you want pooled at each cell (e.g.
    ERA5/CMIP: target-season days across all years; reforecast: the stacked
    hDate x inidate x number ensemble) -- every cell sees the same sample
    COUNT, which is what makes the fixed top-K vectorization valid.
    """
    n = data.sizes[sample_dim]
    K = int(round((100 - threshold_percentile) / 100 * n))
    if K < 2:
        raise ValueError(f"only {K} exceedances at {threshold_percentile}th pct of n={n} -- need >= 2")

    ranks = np.arange(1, K + 1)
    log_survival = np.log((K + 1 - ranks) / (K + 1))
    y_mean = log_survival.mean()
    yc = log_survival - y_mean
    ss_tot = float((yc ** 2).sum())

    def _fit(arr):
        if anomaly:
            arr = arr - arr.mean(axis=-1, keepdims=True)
        top_k = np.sort(arr, axis=-1)[..., -K:]
        x_mean = top_k.mean(axis=-1, keepdims=True)
        xc = top_k - x_mean
        denom = (xc ** 2).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (xc * yc).sum(axis=-1) / denom
        intercept = y_mean - slope * x_mean.squeeze(-1)
        pred = slope[..., None] * top_k + intercept[..., None]
        ss_res = ((log_survival - pred) ** 2).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1 - ss_res / ss_tot
        threshold = top_k[..., 0]
        rarity_factor = np.exp(-slope)
        return threshold, slope, intercept, r2, rarity_factor

    threshold, slope, intercept, r2, rarity_factor = xr.apply_ufunc(
        _fit, data,
        input_core_dims=[[sample_dim]],
        output_core_dims=[[], [], [], [], []],
        dask="parallelized",
        output_dtypes=[float, float, float, float, float],
    )
    out = xr.Dataset({
        "threshold": threshold, "slope": slope, "intercept": intercept,
        "r2": r2, "rarity_factor_per_plus1degC": rarity_factor,
    })
    out.attrs.update(m=K, threshold_percentile=threshold_percentile, anomaly=anomaly)
    return out


def box_mean(out: xr.Dataset, lat_bounds=CMIP_LAT_BOUNDS, lon_bounds=CMIP_LON_BOUNDS) -> dict:
    """PNW-box area-weighted mean, as a cheap cross-check against the scalar scripts' numbers."""
    lat_lo, lat_hi = lat_bounds
    lon_lo, lon_hi = lon_bounds
    lat = out["lat"]
    box = out.sel(lat=slice(min(lat_lo, lat_hi), max(lat_lo, lat_hi)), lon=slice(lon_lo, lon_hi))
    if box.sizes.get("lat", 0) == 0:
        box = out.sel(lat=slice(max(lat_lo, lat_hi), min(lat_lo, lat_hi)), lon=slice(lon_lo, lon_hi))
    weights = np.cos(np.deg2rad(box["lat"]))
    return {k: float(box[k].weighted(weights).mean().compute()) for k in
            ["slope", "r2", "rarity_factor_per_plus1degC"]}


def plot_tail_slope_map(out: xr.Dataset, field: str = "rarity_factor_per_plus1degC",
                         cmap: str = "viridis", title: str = None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    da = out[field].compute()
    mesh = ax.pcolormesh(out["lon"], out["lat"], da, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=field, shrink=0.85)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title or field)
    fig.tight_layout()
    return fig


# ── Data loading (global grid, no box selection) ────────────────────────────

def compute_target_mmdd(refore_ds: xr.Dataset, lead_day: int) -> set:
    """Calendar mm-dd of the reforecast's valid dates (inidate + lead), independent of hDate."""
    lead_days = refore_ds["time"].values / np.timedelta64(1, "D")
    lead_days = np.atleast_1d(lead_days)
    if not np.allclose(lead_days, lead_day):
        raise ValueError(f"reforecast zarr's 'time' (lead) is {lead_days} days, expected {lead_day} -- "
                          f"LEAD_DAY config doesn't match the file")
    valid_dates = pd.to_datetime(refore_ds["inidate"].values) + pd.Timedelta(days=lead_day)
    return set(valid_dates.strftime("%m-%d"))


def _maybe_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = da.attrs.get("units", "").lower()
    if units == "k":
        return da - 273.15
    if units in ("c", "degc", "degrees_c", ""):
        corner = da.isel({d: slice(0, min(3, da.sizes[d])) for d in da.dims})
        if float(corner.mean().compute()) > 100:
            return da - 273.15
    return da


def load_era5_grid(path: str, target_mmdd: set, var: str = "t2m") -> xr.DataArray:
    ds = xr.open_zarr(path)
    da = ds[var].rename({"latitude": "lat", "longitude": "lon"})
    mask = da["time"].dt.strftime("%m-%d").isin(sorted(target_mmdd))
    da = da.isel(time=mask.values)
    da = _maybe_kelvin_to_celsius(da)
    return da.chunk({"time": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})


def load_reforecast_grid(path: str, var: str = "t2m") -> xr.DataArray:
    ds = xr.open_zarr(path)
    da = ds[var].rename({"latitude": "lat", "longitude": "lon"})
    da = da.stack(sample=("hDate", "inidate", "number"))
    da = _maybe_kelvin_to_celsius(da)
    return da.chunk({"sample": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})


def load_cmip_grid(path: str, target_mmdd: set, var: str = CMIP_VAR) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks={})
    da = ds[var]
    if "member" in da.dims:
        da = da.squeeze("member", drop=True)
    mask = da["time"].dt.strftime("%m-%d").isin(sorted(target_mmdd))
    da = da.isel(time=mask.values)
    da = _maybe_kelvin_to_celsius(da)
    return da.chunk({"time": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})


def cmip_model_name(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ("_2001_2020.nc",):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    refore_ds = xr.open_zarr(REFORECAST_PATH)
    target_mmdd = compute_target_mmdd(refore_ds, LEAD_DAY)
    print(f"target_mmdd ({len(target_mmdd)} dates): {sorted(target_mmdd)}")

    # ERA5
    era5_da = load_era5_grid(ERA5_PATH, target_mmdd)
    print(f"ERA5 grid: {dict(era5_da.sizes)}")
    era5_out = empirical_tail_slope_grid(era5_da, "time")
    era5_out.to_netcdf(f"{OUT_PREFIX}_era5.nc")
    print("ERA5 PNW-box cross-check:", box_mean(era5_out))
    fig = plot_tail_slope_map(era5_out, title="ERA5: rarity factor per +1C (empirical tail slope)")
    fig.savefig(f"{OUT_PREFIX}_era5.png")

    # Reforecast
    refore_da = load_reforecast_grid(REFORECAST_PATH)
    print(f"Reforecast grid: {dict(refore_da.sizes)}")
    refore_out = empirical_tail_slope_grid(refore_da, "sample")
    refore_out.to_netcdf(f"{OUT_PREFIX}_reforecast.nc")
    print("Reforecast PNW-box cross-check:", box_mean(refore_out))
    fig = plot_tail_slope_map(refore_out, title=f"Reforecast day {LEAD_DAY}: rarity factor per +1C")
    fig.savefig(f"{OUT_PREFIX}_reforecast.png")

    # CMIP (global, no box restriction), one file per model
    cmip_files = sorted(glob.glob(CMIP_GLOB))
    print(f"CMIP files: {len(cmip_files)}")
    for path in cmip_files:
        name = cmip_model_name(path)
        cmip_da = load_cmip_grid(path, target_mmdd)
        print(f"  {name} grid: {dict(cmip_da.sizes)}")
        cmip_out = empirical_tail_slope_grid(cmip_da, "time")
        cmip_out.to_netcdf(f"{OUT_PREFIX}_cmip_{name}.nc")
        print(f"  {name} PNW-box cross-check:", box_mean(cmip_out))
        fig = plot_tail_slope_map(cmip_out, title=f"{name}: rarity factor per +1C")
        fig.savefig(f"{OUT_PREFIX}_cmip_{name}.png")

    print("Done.")


if __name__ == "__main__":
    main()
