############################################
### ERA5-Land hourly -> daily
###
### Input: hourly monthly files from the CDS download (soil.py), named
###   {var}_{YYYY}_{MM}.nc  for var in t2m, d2m, tp, u10, v10
###
### Output: one file per year, daily_{YYYY}.nc, with
###   tp        daily total precipitation        (mm/day)
###   t2m       daily mean 2 m temperature       (degC)
###   t2m_max   daily max 2 m temperature        (degC)
###   t2m_min   daily min 2 m temperature        (degC)
###   rh        daily mean relative humidity     (%)
###   ws        daily mean 10 m wind speed       (m/s)
###   ws_max    daily max hourly 10 m wind speed (m/s)
###
### Days are UTC days (00-23 UTC).
############################################
import os
import numpy as np
import pandas as pd
import xarray as xr


in_dir = "."        # folder with the hourly {var}_{YYYY}_{MM}.nc files
out_dir = "daily"   # where daily_{YYYY}.nc is written
years = [2023, 2024]

os.makedirs(out_dir, exist_ok=True)


def open_var(short, year, month):
    """Open one hourly monthly file and tidy the new-CDS netcdf layout."""
    path = os.path.join(in_dir, f"{short}_{year}_{month:02d}.nc")
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    # drop the scalar bookkeeping coords the new CDS adds
    ds = ds.drop_vars([c for c in ("number", "expver") if c in ds.coords or c in ds.data_vars])
    return ds[short]


def next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


##################################################
##### PRECIPITATION ######################
###################################################
# ERA5-Land tp is accumulated from 00 UTC, and the value stamped 00 UTC on
# day D+1 is the full 24 h total for day D. So the daily total for D is the
# 00 UTC value of the following day. The last day of each month therefore
# needs the first time step of the next month's file.
def daily_tp(year, month):
    tp = open_var("tp", year, month)
    ny, nm = next_month(year, month)
    tp_next = open_var("tp", ny, nm)

    at00 = tp.sel(time=tp.time.dt.hour == 0)
    if tp_next is not None:
        at00 = xr.concat([at00, tp_next.isel(time=[0])], dim="time")
    # the 00 UTC value belongs to the day before
    at00["time"] = at00.time - pd.Timedelta(days=1)

    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    daily = at00.reindex(time=days)

    if tp_next is None:
        # no next-month file (e.g. 31 Dec 2024): fall back to the 23 UTC value,
        # which misses the last hour of the day
        last = days[-1]
        print(f"  no tp file for {ny}-{nm:02d}: {last.date()} uses 23 UTC value (23 h total)")
        daily.loc[dict(time=last)] = tp.sel(time=last + pd.Timedelta(hours=23)).values

    return daily * 1000.0  # m -> mm


##################################################
##### RELATIVE HUMIDITY ######################
###################################################
# saturation vapour pressure over water, Tetens formula as used in the
# ECMWF IFS (IFS documentation Part IV, eq. 7.5); T in K
def esat(T):
    return 611.21 * np.exp(17.502 * (T - 273.16) / (T - 32.19))


##################################################
##### MAIN LOOP ######################
###################################################
for year in years:
    monthly = []
    for month in range(1, 13):
        print(f"Processing {year}-{month:02d} ...")

        t2m = open_var("t2m", year, month).load()
        d2m = open_var("d2m", year, month).load()
        u10 = open_var("u10", year, month).load()
        v10 = open_var("v10", year, month).load()

        rh = (100.0 * esat(d2m) / esat(t2m)).clip(max=100.0)
        ws = np.sqrt(u10**2 + v10**2)
        t2m_c = t2m - 273.15

        # 1D resample bins 00-23 UTC into the same calendar day
        daily = xr.Dataset({
            "tp":      daily_tp(year, month),
            "t2m":     t2m_c.resample(time="1D").mean(),
            "t2m_max": t2m_c.resample(time="1D").max(),
            "t2m_min": t2m_c.resample(time="1D").min(),
            "rh":      rh.resample(time="1D").mean(),
            "ws":      ws.resample(time="1D").mean(),
            "ws_max":  ws.resample(time="1D").max(),
        })
        monthly.append(daily)

        del t2m, d2m, u10, v10, rh, ws, t2m_c

    out = xr.concat(monthly, dim="time")
    out["tp"].attrs = {"long_name": "daily total precipitation", "units": "mm/day"}
    out["t2m"].attrs = {"long_name": "daily mean 2 m temperature", "units": "degC"}
    out["t2m_max"].attrs = {"long_name": "daily max 2 m temperature", "units": "degC"}
    out["t2m_min"].attrs = {"long_name": "daily min 2 m temperature", "units": "degC"}
    out["rh"].attrs = {"long_name": "daily mean relative humidity (from t2m, d2m)", "units": "%"}
    out["ws"].attrs = {"long_name": "daily mean 10 m wind speed", "units": "m/s"}
    out["ws_max"].attrs = {"long_name": "daily max hourly 10 m wind speed", "units": "m/s"}
    out.attrs["source"] = "ERA5-Land hourly, aggregated to UTC days"

    out_path = os.path.join(out_dir, f"daily_{year}.nc")
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=encoding)
    print(f"Saved {out_path}")

print("All done")
