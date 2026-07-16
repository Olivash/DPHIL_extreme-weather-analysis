#!/usr/bin/env python3
# build_reforecast_part.py

import argparse
import gc
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def collect_fpaths(base: str, var: str) -> list[str]:
    """Collect GRIB files for the requested variable."""
    if var == "t2m":
        months = ["June", "July", "August"]
    elif var == "tp":
        months = ["April", "May", "June", "July", "August"]
    else:
        raise ValueError(f"Unknown variable: {var}")

    fpaths = []
    for month in months:
        fpaths.extend(glob.glob(f"{base}/{month}/*.grib"))
    return sorted(fpaths)


def extract_inidates(fpaths: list[str]) -> list[str]:
    """Extract unique init dates from filenames like ref_cf_20210708.grib."""
    return sorted(
        set(Path(fp).name.split("_")[-1].split(".")[0] for fp in fpaths)
    )


def preproc_ds(ds: xr.Dataset) -> xr.Dataset:
    """Shared cleanup for both fields."""
    if "valid_time" in ds.coords:
        ds = ds.drop_vars("valid_time")

    if "number" not in ds.dims:
        ds = ds.expand_dims({"number": [0]})

    ds = ds.transpose("time", ...)
    return ds


def open_by_type(
    inidate: str,
    fpaths: list[str],
    filter_keys: dict | None = None,
) -> list[xr.Dataset]:
    """Open control and perturbed files for one init date."""
    files = [fp for fp in fpaths if inidate in fp]
    cf_files = [fp for fp in files if "_cf_" in fp]
    pf_files = [fp for fp in files if "_pf_" in fp]

    chunks_cf = {"time": 10, "step": 40, "latitude": 119, "longitude": 240}
    chunks_pf = {"number": 2, "time": 20, "step": 40, "latitude": 119, "longitude": 240}

    datasets = []

    if cf_files:
        if filter_keys:
            backend_kwargs["filter_by_keys"] = filter_keys

        cf_ds = xr.open_dataset(
            cf_files[0],
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
            decode_timedelta=True,
        ).chunk(chunks_cf)
        datasets.append(cf_ds)

    if pf_files:
        
        pf_ds = xr.open_dataset(
            pf_files[0],
            engine="cfgrib",

            decode_timedelta=True,
        ).chunk(chunks_pf)
        datasets.append(pf_ds)

    return datasets


def preproc_t2m(ds: xr.Dataset) -> xr.DataArray:
    """Process 2 m temperature."""
    ds = preproc_ds(ds)
    ds = ds.assign_coords(time=ds.time.dt.year)
    return ds["t2m"].coarsen(step=4, boundary="trim").max()


def preproc_tp(ds: xr.Dataset) -> xr.DataArray:
    """Process total precipitation."""
    ds = preproc_ds(ds)

    tp_6h = ds["tp"].diff("step") * 1000
    tp_6h.attrs["units"] = "mm"

    daily_tp = tp_6h.resample(step="1D").sum()
    daily_tp.attrs["units"] = "mm day$^{-1}$"
    return daily_tp


def process_inidate(var: str, inidate: str, fpaths: list[str]) -> xr.DataArray:
    """Process one init date for one variable."""

    if var == "t2m":
        filter_keys = {"shortName": "2t"}
    elif var == "tp":
        filter_keys = {"shortName": "tp"}
    else:
        raise ValueError(f"Unknown variable: {var}")

    datasets = open_by_type(inidate, fpaths, filter_keys=filter_keys)

    if var == "t2m":
        prepped = [preproc_t2m(ds) for ds in datasets]
    elif var == "tp":
        prepped = [preproc_tp(ds) for ds in datasets]
    else:
        raise ValueError(f"Unknown variable: {var}")

    return xr.concat(prepped, dim="number")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", required=True, choices=["t2m", "tp"])
    parser.add_argument("--base", default="/network/group/aopp/predict/MRA001_AYIM_REFORCST")
    parser.add_argument("--outdir", default="/network/group/aopp/predict/MRA001_AYIM_REFORCST/zarr_parts")
    parser.add_argument("--dates-per-task", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fpaths = collect_fpaths(args.base, args.var)
    inidates = extract_inidates(fpaths)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    start = task_id * args.dates_per_task
    task_inidates = inidates[start : start + args.dates_per_task]

    if not task_inidates:
        print(f"No dates assigned to task {task_id}. Exiting.")
        return

    part_zarr = os.path.join(args.outdir, f"reforecast_{args.var}_part_{task_id:04d}.zarr")
    first = True

    for inidate in task_inidates:  # Process only the first inidate for testing
        print(f"Processing inidate: {inidate}")
        print(f"[{args.var}] task {task_id} processing {inidate}")

        merged_ds = (
            process_inidate(args.var, inidate, fpaths)
            .expand_dims({"inidate": [pd.to_datetime(inidate)]})
            .to_dataset(name=args.var)
            .chunk(
                {
                    "inidate": 1,
                    "time": 10,
                    "number": 11,
                    "step": 13,
                    "latitude": 120,
                    "longitude": 240,
                }
            )
        )

        if first:
            merged_ds.to_zarr(part_zarr, mode="w")
            first = False
        else:
            merged_ds.to_zarr(part_zarr, mode="a", append_dim="inidate")

        del merged_ds
        gc.collect()

    print(f"Wrote {part_zarr}")


if __name__ == "__main__":
    main()
