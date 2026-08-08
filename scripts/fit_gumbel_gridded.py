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

Uses cartopy (Robinson projection, coastlines/borders, gridlines) when available, matching the
plotting style used elsewhere in this project -- falls back to plain lat/lon pcolormesh (no
projection/coastlines) if cartopy isn't installed, so this still runs somewhere without it.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ── Config ───────────────────────────────────────────────────────────────────
ERA5_PATH = "/network/group/aopp/predict/AWH020_AYIM_EXTREME/ERA5/era5_t2m/nick_testing/all_years.zarr"
REFORECAST_PATH = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/full_refore.zarr"
CMIP_GLOB = "Network/historical_ssp/*2001_2020.nc"
AMIP_GLOB = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/dtree_zarr_output_now/*.nc"
CMIP_VAR = "tasmax"
LEAD_DAY = 12  # must match the reforecast zarr's fixed lead (checked at runtime)

# Restrict to the first N inidates (chronologically) so the season window matches the original
# refore.csv-based averaged-series reforecast. N_INIDATES=11 -> 06-15..07-24 (confirmed against
# the CSV). Set to None to use all 25 available inidates instead (06-15..09-11 -- note 3 of those
# fall outside JJA, in early September) for a broader-season run; see fit_gumbel_gridded.py's
# docstring discussion for why pooling more dates needs day-of-year de-seasonalizing to be valid.
N_INIDATES = 11

THRESHOLD_PERCENTILE = 95  # top 5%, same convention as the scalar scripts
ANOMALY = True  # center each cell on its own time-mean first (matches empirical_tail_slope)

# Spatial chunk sizes for dask parallelism (tune to your cluster)
CHUNK_LAT = 90
CHUNK_LON = 180

# PNW box, for a sanity cross-check against the scalar (box-averaged) results
CMIP_LAT_BOUNDS = (45, 52)
CMIP_LON_BOUNDS = (-123, -119)

OUT_PREFIX = "gridded"

# Regions of interest for the area-vs-model comparison: name -> (lat_bounds, lon_bounds), both
# in -180..180 lon (box_mean converts to 0..360 automatically for grids that need it, e.g. ERA5).
# Deliberately climate-CONTRASTING, not just other heatwave-history regions -- the point is to
# stress-test whether the reforecast and the empirical tail-slope method are robust across
# fundamentally different thermal/variance regimes, not just reproduce PNW-like results elsewhere.
# All bounds are in (min, max) or (max, min) order -- box_mean handles either. Adjust freely.
REGIONS = {
    "PNW": ((45, 52), (-123, -119)),        # temperate maritime -- the original 2021 case, kept as anchor
    "Sahara": ((20, 28), (0, 15)),           # hyper-arid desert -- extreme heat, very different variance structure
    "Amazon Basin": ((-8, 3), (-70, -55)),   # tropical -- minimal seasonal cycle, different tail behavior
    "Siberia": ((60, 68), (90, 120)),        # subarctic/continental -- cold-dominated, heat is not the norm here
}

# ONS "Regions (December 2023) Boundaries EN BFC" shapefile directory -- the .shp plus its
# .shx/.dbf/.prj siblings must all be present alongside it. Point this at wherever it actually
# lives on your cluster.
UK_SHAPEFILE_PATH = (
    "/network/group/aopp/predict/AWH020_AYIM_EXTREME/shapefiles/"
    "Regions_December_2023_Boundaries_EN_BFC_117415587253983129/"
    "RGN_DEC_2023_EN_BFC.shp"
)

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
        # Degenerate cells (near-constant top-K values, e.g. some polar/masked points) give
        # denom ~ 0, which blows slope/rarity_factor up to nonsense (1e6+) instead of erroring --
        # that silently wrecks both plot color scales and box-mean averages. Mark them NaN
        # explicitly instead so downstream skipna/percentile handling can ignore them cleanly.
        degenerate = denom < 1e-8 * max(float(np.nanmax(np.abs(xc))), 1.0) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (xc * yc).sum(axis=-1) / denom
        intercept = y_mean - slope * x_mean.squeeze(-1)
        pred = slope[..., None] * top_k + intercept[..., None]
        ss_res = ((log_survival - pred) ** 2).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1 - ss_res / ss_tot
        threshold = top_k[..., 0]
        with np.errstate(over="ignore"):
            rarity_factor = np.exp(-slope)
        slope = np.where(degenerate, np.nan, slope)
        intercept = np.where(degenerate, np.nan, intercept)
        r2 = np.where(degenerate, np.nan, r2)
        rarity_factor = np.where(degenerate, np.nan, rarity_factor)
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
    out.attrs.update(m=K, threshold_percentile=threshold_percentile, anomaly=int(anomaly))
    return out


def select_box(out: xr.Dataset, lat_bounds=CMIP_LAT_BOUNDS, lon_bounds=CMIP_LON_BOUNDS) -> xr.Dataset:
    """
    Crop a gridded Dataset to a lat/lon box, handling both -180..180 and 0..360 longitude
    conventions (bounds are given in -180..180; converted automatically for a 0..360 grid like
    ERA5's -- otherwise -123..-119 silently selects zero cells there) and either latitude
    ordering (ascending like CMIP, or descending like ERA5/reforecast). Raises instead of
    silently returning an empty selection.

    Also handles boxes that straddle the prime meridian (e.g. the UK: -6..2) on a native 0..360
    grid: converting -6..2 to 0..360 gives 354..2, where a single ascending .sel(lon=slice(...))
    would silently select almost the entire globe instead of the small wedge intended (the
    complement of what was wanted), since 354 > 2 makes min/max pick the wrong pair of endpoints.
    Detected via lon_lo > lon_hi after the 0..360 conversion and handled by selecting the two
    wedges either side of the seam (354..360 and 0..2) and concatenating them.
    """
    lat_lo, lat_hi = lat_bounds
    lon_lo, lon_hi = lon_bounds
    wraps = False
    if float(out["lon"].min()) >= 0:
        lon_lo, lon_hi = lon_lo % 360, lon_hi % 360
        wraps = lon_lo > lon_hi

    def _lat_sel(ds):
        b = ds.sel(lat=slice(min(lat_lo, lat_hi), max(lat_lo, lat_hi)))
        if b.sizes.get("lat", 0) == 0:
            b = ds.sel(lat=slice(max(lat_lo, lat_hi), min(lat_lo, lat_hi)))
        return b

    if wraps:
        lat_sel = _lat_sel(out)
        box = xr.concat([lat_sel.sel(lon=slice(lon_lo, 360)), lat_sel.sel(lon=slice(0, lon_hi))], dim="lon")
    else:
        box = _lat_sel(out).sel(lon=slice(min(lon_lo, lon_hi), max(lon_lo, lon_hi)))

    if box.sizes.get("lat", 0) == 0 or box.sizes.get("lon", 0) == 0:
        raise ValueError(f"box selection is empty (lat={lat_bounds}, lon={lon_bounds} -> "
                          f"resolved to lon={lon_lo, lon_hi}) -- check the grid's lat/lon convention")
    return box


def box_mean(out: xr.Dataset, lat_bounds=CMIP_LAT_BOUNDS, lon_bounds=CMIP_LON_BOUNDS) -> dict:
    """Area-weighted mean over a lat/lon box, as a cheap cross-check against the scalar scripts' numbers."""
    box = select_box(out, lat_bounds, lon_bounds)
    weights = np.cos(np.deg2rad(box["lat"]))
    return {k: float(box[k].weighted(weights).mean(skipna=True).compute()) for k in
            ["slope", "r2", "rarity_factor_per_plus1degC"]}


def to_180_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Plotting-only longitude normalization to -180..180. ERA5 is natively 0..360; CMIP and the
    reforecast are natively -180..180 -- left as-is, this makes ERA5 maps read on a 0-360 axis
    while the others read -180..180, which is confusing to compare side by side even though the
    underlying data/analysis (select_box, box_mean, mask_ocean) are already convention-agnostic
    and don't need this. No-op if already -180..180. Re-sorts after the wrap so lon stays
    monotonic -- required for correct pcolormesh rendering, otherwise there's a jump at 180.
    """
    if float(da["lon"].min()) < 0:
        return da
    new_lon = ((da["lon"] + 180) % 360) - 180
    return da.assign_coords(lon=new_lon).sortby("lon")


def mask_ocean(da: xr.DataArray) -> xr.DataArray:
    """
    Mask out ocean cells using regionmask's Natural Earth 110m land polygon (pip install
    regionmask). .mask() handles both -180..180 and 0..360 longitude conventions natively --
    no manual conversion needed, unlike select_box's PNW-box logic.
    """
    try:
        import regionmask
    except ImportError as e:
        raise ImportError("mask_ocean() requires the 'regionmask' package (pip install "
                           "regionmask) -- or pass mask_ocean_cells=False to skip land masking.") from e
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask(da["lon"], da["lat"])
    return da.where(np.isfinite(land_mask))


def _discrete_log_levels(data, n_levels: int = 10, vmin=None, vmax=None) -> np.ndarray:
    """n_levels+1 log-spaced bin edges spanning the 2nd/98th percentile of data's finite, positive
    values (rarity_factor is always > 0; a log scale can't represent <= 0 values anyway). `data`
    can be a DataArray or a plain array -- just needs to support np.asarray()."""
    values = np.asarray(data)
    finite = values[np.isfinite(values) & (values > 0)]
    if vmin is None or vmax is None:
        if finite.size:
            pmin, pmax = np.nanpercentile(finite, [2, 98])
            vmin = pmin if vmin is None else vmin
            vmax = pmax if vmax is None else vmax
        else:
            vmin, vmax = 1.0, 2.0
    return np.geomspace(max(vmin, 1e-6), max(vmax, vmin * 1.01), n_levels + 1)


def plot_tail_slope_map(out: xr.Dataset, field: str = "rarity_factor_per_plus1degC",
                         cmap: str = "viridis", title: str = None, vmin=None, vmax=None,
                         n_levels: int = 10, mask_ocean_cells: bool = True):
    da = out[field].compute()
    if mask_ocean_cells:
        da = mask_ocean(da)
    da = to_180_lon(da)
    levels = _discrete_log_levels(da, n_levels=n_levels, vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap, n_levels + 2)  # +2 colors for extend="both" (below-min, above-max bins)
    norm = BoundaryNorm(levels, ncolors=cmap_obj.N, extend="both")
    tick_labels = [f"{x:.2g}" for x in levels]

    if HAS_CARTOPY:
        proj = ccrs.Robinson(central_longitude=0)
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": proj})
        mesh = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap_obj,
                                   add_colorbar=False, shading="auto", rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="none")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}
        cax = fig.add_axes([0.13, 0.02, 0.74, 0.045])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
        cbar.set_ticks(levels)
        cbar.ax.minorticks_off()
        cbar.set_ticklabels(tick_labels)
        cbar.set_label(field)
    else:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        mesh = ax.pcolormesh(da["lon"], da["lat"], da, cmap=cmap_obj, norm=norm, shading="auto")
        fig.colorbar(mesh, ax=ax, label=field, shrink=0.85, ticks=levels,
                     format=lambda x, _: f"{x:.2g}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    ax.set_title(title or field)
    return fig


# ── Regional comparison: area vs model ──────────────────────────────────────

def build_model_file_map(cmip_glob: str = CMIP_GLOB, amip_glob: str = AMIP_GLOB,
                          out_prefix: str = OUT_PREFIX, include_amip: bool = True) -> dict:
    """
    {model_name: saved .nc path}, for every dataset main() already wrote to disk. Derives the
    model lists from CMIP_GLOB/AMIP_GLOB (matching cmip_model_name()) rather than hardcoded
    lists, so it can't drift out of sync with what main() actually saved. CMIP and AMIP entries
    are labeled distinctly ("(hist-ssp)" / "(AMIP)") since some model names appear in both sets
    but represent different experiments -- exactly the "these might give different results"
    comparison this is for. Set include_amip=False to only get the CMIP historical_ssp set.
    """
    files = {"ERA5": f"{out_prefix}_era5.nc", "Reforecast": f"{out_prefix}_reforecast.nc"}
    for path in sorted(glob.glob(cmip_glob)):
        name = cmip_model_name(path)
        files[f"{name} (hist-ssp)"] = f"{out_prefix}_cmip_{name}.nc"
    if include_amip:
        for path in sorted(glob.glob(amip_glob)):
            name = cmip_model_name(path)
            files[f"{name} (AMIP)"] = f"{out_prefix}_amip_{name}.nc"
    return files


def region_model_table(regions: dict = REGIONS, model_files: dict = None,
                        field: str = "rarity_factor_per_plus1degC") -> pd.DataFrame:
    """Region x model table of box_mean(field), re-opening each already-saved .nc (no recompute)."""
    if model_files is None:
        model_files = build_model_file_map()
    rows = {}
    for region_name, (lat_bounds, lon_bounds) in regions.items():
        row = {}
        for model_name, path in model_files.items():
            ds = xr.open_dataset(path)
            row[model_name] = box_mean(ds, lat_bounds=lat_bounds, lon_bounds=lon_bounds)[field]
        rows[region_name] = row
    return pd.DataFrame(rows).T  # rows=regions, columns=models


def plot_region_model_heatmap(table: pd.DataFrame, field_label: str = "rarity factor per +1C",
                               cmap: str = "viridis", annotate: bool = True):
    """Single heatmap: regions (rows) x models (columns), color = table value."""
    data = table.values.astype(float)
    fig, ax = plt.subplots(figsize=(0.9 * len(table.columns) + 2.5, 0.6 * len(table.index) + 2))
    mesh = ax.pcolormesh(data, cmap=cmap, shading="flat", edgecolors="white", linewidth=1)
    ax.set_xticks(np.arange(len(table.columns)) + 0.5)
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(table.index)) + 0.5)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, label=field_label, shrink=0.85)
    if annotate:
        finite = data[np.isfinite(data)]
        mid = float(np.nanmean(finite)) if finite.size else 0.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if val > mid else "black")
                else:
                    ax.text(j + 0.5, i + 0.5, "NaN", ha="center", va="center", fontsize=8, color="0.5")
    ax.set_title(f"{field_label}: region x model")
    fig.tight_layout()
    return fig


def plot_region_maps(region_name: str, lat_bounds, lon_bounds, model_files: dict = None,
                      field: str = "rarity_factor_per_plus1degC", cmap: str = "viridis", ncols: int = 4,
                      n_levels: int = 10, mask_ocean_cells: bool = True):
    """
    Actual spatial maps (not an averaged number) for one region: one subplot per model, each a
    pcolormesh crop to that region's lat/lon box. All subplots share one set of discrete,
    log-spaced color levels (2nd/98th percentile across every model's crop, finite land values
    only) so the models are visually comparable to each other, not each auto-scaled to its own range.
    """
    if model_files is None:
        model_files = build_model_file_map()
    crops = {}
    for model_name, path in model_files.items():
        ds = xr.open_dataset(path)
        da = select_box(ds, lat_bounds, lon_bounds)[field].compute()
        if mask_ocean_cells:
            da = mask_ocean(da)
        crops[model_name] = to_180_lon(da)

    # Pool raw values (not xr.concat) -- models are on different native grids (e.g. ERA5 0.25deg
    # vs CMIP 1deg), so concat would outer-join mismatched lat/lon coords and pad with NaN instead
    # of just pooling the numbers.
    pooled_values = np.concatenate([da.values.ravel() for da in crops.values()])
    levels = _discrete_log_levels(pooled_values, n_levels=n_levels)

    n = len(crops)
    ncols_eff = min(ncols, n)
    nrows = int(np.ceil(n / ncols_eff))
    cmap_obj = plt.get_cmap(cmap, n_levels + 2)
    norm = BoundaryNorm(levels, ncolors=cmap_obj.N, extend="both")

    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(3.2 * ncols_eff, 2.8 * nrows),
                              squeeze=False, subplot_kw=subplot_kw)
    mesh = None
    for ax, (model_name, da) in zip(axes.flat, crops.items()):
        if HAS_CARTOPY:
            mesh = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap_obj,
                                       add_colorbar=False, shading="auto")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4)
            ax.set_title("")
        else:
            mesh = ax.pcolormesh(da["lon"], da["lat"], da, cmap=cmap_obj, norm=norm, shading="auto")
            ax.set_xlabel("lon", fontsize=8)
            ax.set_ylabel("lat", fontsize=8)
        ax.set_title(model_name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, label=field, shrink=0.8, ticks=levels)
        cbar.set_ticklabels([f"{x:.2g}" for x in levels])
    fig.suptitle(f"{region_name}: {field}", y=1.02)
    return fig


def plot_all_region_maps(regions: dict = REGIONS, model_files: dict = None,
                          field: str = "rarity_factor_per_plus1degC", out_prefix: str = OUT_PREFIX) -> dict:
    """One map-grid figure per region, saved as {out_prefix}_mapgrid_{region}.png. Returns {region: fig}."""
    if model_files is None:
        model_files = build_model_file_map()
    figs = {}
    for region_name, (lat_bounds, lon_bounds) in regions.items():
        fig = plot_region_maps(region_name, lat_bounds, lon_bounds, model_files, field=field)
        safe_name = region_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        fig.savefig(f"{out_prefix}_mapgrid_{safe_name}.png")
        figs[region_name] = fig
    return figs


# ── UK regions (shapefile clip) ─────────────────────────────────────────────

def load_uk_regions(shapefile_path: str = UK_SHAPEFILE_PATH, name_col: str = "RGN23NM"):
    """
    Reads the ONS regions shapefile with pyshp + shapely + pyproj (pip install pyshp) instead of
    geopandas -- geopandas pulls in a much larger, more fragile compiled dependency chain
    (GEOS/GDAL/fiona/pyogrio) that's a common source of shapely/geopandas ABI mismatches on
    cluster conda envs, whereas pyshp is pure Python and only needs shapely (already a
    scipy/xarray-stack dependency) and pyproj (already needed for any CRS-aware work here).
    Reprojects to EPSG:4326 using the shapefile's own .prj sidecar if it isn't already WGS84 --
    ONS boundary files are commonly delivered in EPSG:27700 (British National Grid), which is
    metres, not degrees, and would silently break every lon/lat comparison below. Returns a
    regionmask.Regions object (not a GeoDataFrame) -- used directly by uk_bounds/mask_to_regions/
    plot_uk_map below via its own .bounds_global/.mask()/.plot_regions() API. overlap=False since
    admin regions are disjoint by construction -- without it, .mask() can raise "Found
    overlapping regions" on some grids (regionmask's overlap autodetection is a per-gridpoint
    check done lazily inside .mask(), not a pure geometry check, so whether it triggers can
    depend on grid resolution even for genuinely non-overlapping polygons).
    """
    import shapefile  # pyshp
    from shapely.geometry import shape
    import regionmask

    sf = shapefile.Reader(shapefile_path)
    fields = [f[0] for f in sf.fields[1:]]  # sf.fields[0] is the DeletionFlag pseudo-field
    name_idx = fields.index(name_col) if name_col in fields else 0
    geoms = [shape(s.__geo_interface__) for s in sf.shapes()]
    names = [r[name_idx] for r in sf.records()]

    prj_path = os.path.splitext(shapefile_path)[0] + ".prj"
    if os.path.exists(prj_path):
        import pyproj
        from shapely.ops import transform
        crs = pyproj.CRS.from_wkt(open(prj_path).read())
        if crs.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            geoms = [transform(transformer.transform, g) for g in geoms]

    return regionmask.Regions(geoms, names=names, name="UK regions", overlap=False)


def uk_bounds(uk_regions, pad: float = 0.5):
    """(lat_bounds, lon_bounds) from a regionmask.Regions' global bounding box, padded by `pad`
    degrees so the select_box crop below doesn't clip cells sitting right at the boundary edge --
    same (lat_bounds, lon_bounds) format REGIONS/select_box/box_mean already use."""
    minx, miny, maxx, maxy = uk_regions.bounds_global
    return (miny - pad, maxy + pad), (minx - pad, maxx + pad)


def mask_to_regions(da: xr.DataArray, uk_regions) -> xr.DataArray:
    """
    Mask out every cell whose center falls outside every polygon in `uk_regions`, via
    regionmask.Regions.mask() -- the same cell-center membership test mask_ocean already uses
    against the Natural Earth land polygon, just region polygons here instead of a coastline.
    Handles both -180..180 and 0..360 longitude conventions natively (regionmask reads da['lon']
    directly), so it works whether or not to_180_lon has already been applied.
    """
    mask = uk_regions.mask(da["lon"], da["lat"])
    return da.where(np.isfinite(mask))


def plot_uk_map(out: xr.Dataset, uk_regions, field: str = "rarity_factor_per_plus1degC",
                 cmap: str = "viridis", title: str = None, vmin=None, vmax=None,
                 n_levels: int = 10, region_pad: float = 0.5):
    """
    Same rendering as plot_tail_slope_map, but clipped to the UK instead of the whole globe:
    cropped to the shapefile's bounding box (region_pad degrees of margin) and masked to NaN
    outside every region polygon (mask_to_regions), rather than ocean-masked land everywhere.
    Draws the region boundaries themselves on top via regionmask's own plot_regions() (works on
    both a plain matplotlib Axes and a cartopy GeoAxes) instead of generic coastline/borders,
    since the point here is the English admin regions, not the coastline.
    """
    lat_bounds, lon_bounds = uk_bounds(uk_regions, pad=region_pad)
    da = select_box(out, lat_bounds, lon_bounds)[field].compute()
    da = to_180_lon(da)
    da = mask_to_regions(da, uk_regions)
    levels = _discrete_log_levels(da, n_levels=n_levels, vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap, n_levels + 2)  # +2 colors for extend="both" (below-min, above-max bins)
    norm = BoundaryNorm(levels, ncolors=cmap_obj.N, extend="both")
    tick_labels = [f"{x:.2g}" for x in levels]
    minx, miny, maxx, maxy = uk_regions.bounds_global

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(6, 7), subplot_kw={"projection": proj})
        mesh = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap_obj,
                                   add_colorbar=False, shading="auto", rasterized=True)
        uk_regions.plot_regions(ax=ax, add_label=False, line_kws=dict(color="black", linewidth=0.6))
        ax.set_extent([minx - region_pad, maxx + region_pad, miny - region_pad, maxy + region_pad],
                       crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", extend="both", shrink=0.8)
        cbar.set_ticks(levels)
        cbar.ax.minorticks_off()
        cbar.set_ticklabels(tick_labels)
        cbar.set_label(field)
    else:
        fig, ax = plt.subplots(figsize=(6, 7))
        mesh = ax.pcolormesh(da["lon"], da["lat"], da, cmap=cmap_obj, norm=norm, shading="auto")
        uk_regions.plot_regions(ax=ax, add_label=False, line_kws=dict(color="black", linewidth=0.6))
        ax.set_xlim(minx - region_pad, maxx + region_pad)
        ax.set_ylim(miny - region_pad, maxy + region_pad)
        fig.colorbar(mesh, ax=ax, label=field, shrink=0.85, ticks=levels,
                     format=lambda x, _: f"{x:.2g}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    ax.set_title(title or field)
    return fig


def plot_all_uk_maps(uk_regions=None, model_files: dict = None, field: str = "rarity_factor_per_plus1degC",
                      out_prefix: str = OUT_PREFIX) -> dict:
    """
    One UK-clipped map per dataset (ERA5, Reforecast, each CMIP model, each AMIP model), saved as
    {out_prefix}_uk_{dataset}.png. Reopens the already-saved global .nc files (no recompute) --
    same reuse pattern as region_model_table/plot_all_region_maps.
    """
    if uk_regions is None:
        uk_regions = load_uk_regions()
    if model_files is None:
        model_files = build_model_file_map()
    figs = {}
    for name, path in model_files.items():
        ds = xr.open_dataset(path)
        fig = plot_uk_map(ds, uk_regions, field=field, title=f"{name}: {field}")
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        fig.savefig(f"{out_prefix}_uk_{safe_name}.png")
        figs[name] = fig
    return figs


def build_uk_datatree(das: list, names: list, field: str = "rarity_factor_per_plus1degC") -> xr.DataTree:
    """
    Wraps an in-memory list of single-field DataArrays (e.g. already-loaded
    rarity_factor_per_plus1degC grids for ERA5/Reforecast/each CMIP/AMIP model -- not reopened
    from saved .nc files) into an xr.DataTree, one node per dataset. `names` must be aligned with
    `das` by index (das[i]'s node is named names[i]). reset_coords(drop=True) strips any leftover
    scalar metadata coords some sources carry (e.g. reforecast's 'experiment', CMIP/AMIP's
    'height') -- same fix as the disagreement-hotspot MergeError elsewhere in this project -- so
    nodes don't drag around inconsistent per-source coordinate baggage.
    """
    if len(das) != len(names):
        raise ValueError(f"das ({len(das)}) and names ({len(names)}) must be the same length")
    return xr.DataTree.from_dict({
        name: xr.Dataset({field: da.reset_coords(drop=True)}) for name, da in zip(names, das)
    })


def plot_all_uk_maps_from_datatree(tree: xr.DataTree, uk_regions, field: str = "rarity_factor_per_plus1degC",
                                    out_prefix: str = OUT_PREFIX) -> dict:
    """
    Same as plot_all_uk_maps, but walks an in-memory DataTree (see build_uk_datatree) instead of
    reopening each dataset's saved .nc file by path -- for when the gridded output is already
    loaded (e.g. as a list of DataArrays) rather than written to disk.
    """
    figs = {}
    for name, node in tree.children.items():
        fig = plot_uk_map(node.dataset, uk_regions, field=field, title=f"{name}: {field}")
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        fig.savefig(f"{out_prefix}_uk_{safe_name}.png")
        figs[name] = fig
    return figs


# ── Data loading (global grid, no box selection) ────────────────────────────

def compute_target_mmdd(refore_ds: xr.Dataset, lead_day: int, n_inidates: int = None):
    """
    Calendar mm-dd of the reforecast's valid dates (inidate + lead), independent of hDate.
    n_inidates=None uses every inidate in the zarr (25, spanning 06-15..09-11 -- note 3 of
    those fall outside JJA). n_inidates=11 matches the original refore.csv-based averaged
    series: the first 11 inidates chronologically (06-15..07-24).
    Returns (target_mmdd: set, selected_inidates: array of the actual inidate coord values used).
    """
    lead_days = refore_ds["time"].values / np.timedelta64(1, "D")
    lead_days = np.atleast_1d(lead_days)
    if not np.allclose(lead_days, lead_day):
        raise ValueError(f"reforecast zarr's 'time' (lead) is {lead_days} days, expected {lead_day} -- "
                          f"LEAD_DAY config doesn't match the file")
    inidates = pd.to_datetime(refore_ds["inidate"].values)
    selected_inidates = np.sort(inidates.values)
    if n_inidates is not None:
        selected_inidates = selected_inidates[:n_inidates]
    valid_dates = pd.to_datetime(selected_inidates) + pd.Timedelta(days=lead_day)
    return set(valid_dates.strftime("%m-%d")), selected_inidates


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


def load_reforecast_grid(path: str, inidate_sel=None, var: str = "t2m") -> xr.DataArray:
    """inidate_sel restricts to a subset of inidate values (e.g. the 11 from compute_target_mmdd)
    so the reforecast's own sample count/season window matches whatever ERA5/CMIP were filtered
    to -- otherwise the reforecast would still pool all 25 inidates regardless of target_mmdd."""
    ds = xr.open_zarr(path)
    da = ds[var].rename({"latitude": "lat", "longitude": "lon"})
    if inidate_sel is not None:
        da = da.sel(inidate=inidate_sel)
    da = da.stack(sample=("hDate", "inidate", "number"))
    da = _maybe_kelvin_to_celsius(da)
    return da.chunk({"sample": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})


def load_cmip_grid(path: str, target_mmdd: set, var: str = CMIP_VAR) -> xr.DataArray:
    """
    Loads one CMIP/AMIP model, filtered to target_mmdd. Handles both "member" (the historical_ssp
    files) and "member_id" (the AMIP dtree_zarr_output_now files) as the ensemble-member
    dimension name -- otherwise identical structure, so one loader covers both. Not every file
    has exactly one ensemble member -- when member > 1, pool them into the sample dimension as
    additional independent draws (same UNSEEN-style treatment already used for the reforecast's
    ensemble) instead of discarding all but one. Always returns a "sample" dim (renamed from
    "time" when there's only one member) so callers don't need to know which case applied.
    """
    ds = xr.open_dataset(path, chunks={})
    da = ds[var]
    mask = da["time"].dt.strftime("%m-%d").isin(sorted(target_mmdd))
    da = da.isel(time=mask.values)
    member_dim = next((d for d in ("member", "member_id") if d in da.dims), None)
    if member_dim is not None and da.sizes[member_dim] > 1:
        da = da.stack(sample=("time", member_dim))
    else:
        if member_dim is not None:
            da = da.squeeze(member_dim, drop=True)
        da = da.rename({"time": "sample"})
    da = _maybe_kelvin_to_celsius(da)
    return da.chunk({"sample": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})


def cmip_model_name(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ("_2001_2020.nc",):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    refore_ds = xr.open_zarr(REFORECAST_PATH)
    target_mmdd, selected_inidates = compute_target_mmdd(refore_ds, LEAD_DAY, n_inidates=N_INIDATES)
    print(f"target_mmdd ({len(target_mmdd)} dates): {sorted(target_mmdd)}")

    # ERA5
    era5_da = load_era5_grid(ERA5_PATH, target_mmdd)
    print(f"ERA5 grid: {dict(era5_da.sizes)}")
    era5_out = empirical_tail_slope_grid(era5_da, "time")
    era5_out = era5_out.load()  # materialize once -- to_netcdf/box_mean/plot below would each
    # otherwise re-run the full lazy dask graph (re-read + refit the whole grid) from scratch
    era5_out.to_netcdf(f"{OUT_PREFIX}_era5.nc")
    print("ERA5 PNW-box cross-check:", box_mean(era5_out))
    fig = plot_tail_slope_map(era5_out, title="ERA5: rarity factor per +1C (empirical tail slope)")
    fig.savefig(f"{OUT_PREFIX}_era5.png")

    # Reforecast
    refore_da = load_reforecast_grid(REFORECAST_PATH, inidate_sel=selected_inidates)
    print(f"Reforecast grid: {dict(refore_da.sizes)}")
    refore_out = empirical_tail_slope_grid(refore_da, "sample")
    refore_out = refore_out.load()  # see ERA5 comment above -- avoid 3x recompute of a 20GB read
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
        cmip_out = empirical_tail_slope_grid(cmip_da, "sample")
        cmip_out = cmip_out.load()  # see ERA5 comment above
        cmip_out.to_netcdf(f"{OUT_PREFIX}_cmip_{name}.nc")
        print(f"  {name} PNW-box cross-check:", box_mean(cmip_out))
        fig = plot_tail_slope_map(cmip_out, title=f"{name}: rarity factor per +1C")
        fig.savefig(f"{OUT_PREFIX}_cmip_{name}.png")

    # AMIP (dtree_zarr_output_now) -- separate from the CMIP historical_ssp set above. Same
    # loader (load_cmip_grid handles both "member" and "member_id" dim names) and same
    # target_mmdd, but a different output prefix so results don't collide even for model names
    # that appear in both sets -- the point is to compare the two, not merge them.
    amip_files = sorted(glob.glob(AMIP_GLOB))
    print(f"AMIP files: {len(amip_files)}")
    for path in amip_files:
        name = cmip_model_name(path)
        amip_da = load_cmip_grid(path, target_mmdd)
        print(f"  {name} (AMIP) grid: {dict(amip_da.sizes)}")
        amip_out = empirical_tail_slope_grid(amip_da, "sample")
        amip_out = amip_out.load()  # see ERA5 comment above
        amip_out.to_netcdf(f"{OUT_PREFIX}_amip_{name}.nc")
        print(f"  {name} (AMIP) PNW-box cross-check:", box_mean(amip_out))
        fig = plot_tail_slope_map(amip_out, title=f"{name} (AMIP): rarity factor per +1C")
        fig.savefig(f"{OUT_PREFIX}_amip_{name}.png")

    # Regional comparison: area vs model (reopens the .nc files just saved, no recompute)
    table = region_model_table()
    print(table)
    fig = plot_region_model_heatmap(table)
    fig.savefig(f"{OUT_PREFIX}_region_vs_model.png")
    table.to_csv(f"{OUT_PREFIX}_region_vs_model.csv")

    # Actual spatial maps per region (not just the averaged table above) -- one figure per
    # region, one subplot per model, so spatial patterns are visible, not just a single number
    plot_all_region_maps()

    # UK regions (shapefile clip): one map per dataset, cropped and masked to the ONS English
    # regions shapefile instead of a rectangular box
    uk_regions = load_uk_regions()
    plot_all_uk_maps(uk_regions)

    print("Done.")


if __name__ == "__main__":
    main()
