'''
============================================================
Reforecast processing script


This script reads ECMWF reforecast GRIB files, separates them
into control and perturbed members, preprocesses the fields,
and writes two Zarr datasets:
1) reforecast_t2m.zarr -> daily max 2 m temperature
2) reforecast_tp.zarr -> daily precipitation totals
============================================================
'''



#############################################
### Dependencies

##############################################
import glob
import os
import gc

import numpy as np
import pandas as pd
import xarray as xr

#### DATA #####

# base directory containing the reforecast GRIB files, organised by month
base = "/network/group/aopp/predict/MRA001_AYIM_REFORCST"

# collect all GRIB file paths across June, July, and August, sorted for consistent ordering
fpaths = sorted(
    glob.glob(f"{base}/June/*.grib") +
    glob.glob(f"{base}/July/*.grib") +
    glob.glob(f"{base}/August/*.grib")
)

# extract unique initialisation dates (inidates) from filenames, e.g. 'ref_cf_20210708.grib' -> '20210708'
inidates = sorted(list(set(
    [x.split('/')[-1].split('_')[-1].split('_')[0].split('.')[0] for x in fpaths]
)))

# find any existing cfgrib .idx cache files across the same three months
idx_files = (
    glob.glob(f"{base}/June/*.idx") +
    glob.glob(f"{base}/July/*.idx") +
    glob.glob(f"{base}/August/*.idx")
)

# UNCOMMENT TO remove stale/corrupted index files so cfgrib rebuilds them cleanly on next open
for f in idx_files:
    os.remove(f)

print(f"removed {len(idx_files)} idx files")

######FUNCTIONS #########
def preproc_ds(ds):
    # Extract filename for metadata
    fname = ds.encoding['source'].split("/")[-1].split(".")[0].split("_")[-1]
    
    # Extract the initialization date from the time variable
    inidate = pd.to_datetime(ds.time[0].values)
    
    if 'valid_time' in ds.coords:
        ds = ds.drop_vars('valid_time')
    
    if 'number' not in ds.dims:
        ds = ds.expand_dims({'number': [0]})
    
    # Reorganize dimensions to place 'time' first
    ds = ds.transpose('time', ...)
    
    return ds


def preproc_t2m(ds):
    ds = preproc_ds(ds)
    ds=ds.assign_coords(time=ds.time.dt.year)
    ds=ds['t2m'].coarsen(step=4, boundary="trim").max()
    return ds.to_dataset(name="t2m")


def open_by_type(inidate, fpaths, filter_keys=None):
    files = [fp for fp in fpaths if inidate in fp]
    cf_files = [fp for fp in files if '_cf_' in fp]
    pf_files = [fp for fp in files if '_pf_' in fp]

    chunks_cf = {"time": 10, "step": 40, "latitude": 119, "longitude": 240}
    chunks_pf ={"number": 2, "time": 20, "step": 40, "latitude": 119, "longitude": 240}
    datasets = [] 
    if cf_files:
        cf_ds = xr.open_dataset(
            cf_files[0],
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": filter_keys} if filter_keys else {},
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


def process_inidate(inidate, fpaths, filter_keys=None):
    datasets = open_by_type(inidate, fpaths, filter_keys)
    prepped = [preproc_t2m(ds) for ds in datasets]
    merged = xr.concat(prepped, dim='number')
    #merged = merged.expand_dims({'inidate': [pd.to_datetime(inidate)]})
    return merged




####### MAIN LOOP #####


zarr_path = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/reforecast_t2m.zarr"
first = True
for inidate in sorted(inidates[:1]):  # Process only the first inidate for testing
    print(f"Processing inidate: {inidate}")
    merged_ds = (
    process_inidate(inidate, fpaths, filter_keys={"shortName": "2t"})
    .expand_dims({"inidate": [pd.to_datetime(inidate)]}).chunk({
        "inidate": 1,
        "time": 10,
        "number": 11,
        "step": 13,
        "latitude": 120,
        "longitude": 240,
    })
)
    print('done processing inidate:', inidate)
    if first:
        merged_ds.to_zarr(zarr_path, mode='w')
        print('done writing first inidate to zarr:', inidate)
        first = False
    else:
        merged_ds.to_zarr(zarr_path, mode='a', append_dim='inidate')
        print('done appending inidate to zarr:', inidate)
    
    # Clean up to free memory
    del merged_ds
    gc.collect()







##################################################
##### PRECIPITATION ######################
###################################################


### DATA #####
# collect all GRIB file paths across June, July, and August, sorted for consistent ordering
fpaths = sorted(glob.glob(f"{base}/April/*.grib") +
                glob.glob(f"{base}/May/*.grib") +
    glob.glob(f"{base}/June/*.grib") +
    glob.glob(f"{base}/July/*.grib") +
    glob.glob(f"{base}/August/*.grib")
)

# extract unique initialisation dates (inidates) from filenames, e.g. 'ref_cf_20210708.grib' -> '20210708'
inidates = sorted(list(set(
    [x.split('/')[-1].split('_')[-1].split('_')[0].split('.')[0] for x in fpaths]
)))

# find any existing cfgrib .idx cache files across the same three months
idx_files = (
    glob.glob(f"{base}/April/*.idx") +
    glob.glob(f"{base}/May/*.idx") +
    glob.glob(f"{base}/June/*.idx") +
    glob.glob(f"{base}/July/*.idx") +
    glob.glob(f"{base}/August/*.idx")
)

# UNCOMMENT TO remove stale/corrupted index files so cfgrib rebuilds them cleanly on next open
for f in idx_files:
    os.remove(f)

print(f"removed {len(idx_files)} idx files")


## FUNCTIONS ####

def preproc_tp(ds):
    ds = preproc_ds(ds)
    tp_6h = ds["tp"].diff("step") * 1000
    daily_tp = tp_6h.resample(step="1D").sum()

    daily_tp.attrs["units"] = "mm day$^{-1}$"

    return daily_tp.to_dataset(name="tp")

def process_inidate(inidate, fpaths, filter_keys=None):
    datasets = open_by_type(inidate, fpaths, filter_keys)
    prepped = [preproc_tp(ds) for ds in datasets]
    merged = xr.concat(prepped, dim='number')
    #merged = merged.expand_dims({'inidate': [pd.to_datetime(inidate)]})
    return merged


zarr_path = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/reforecast_tp.zarr"
first = True
for inidate in sorted(inidates[:1]):  # Process only the first inidate for testing
    print(f"Processing inidate: {inidate}")
    merged_ds = (
    process_inidate(inidate, fpaths, filter_keys={"shortName": "tp"})
    .expand_dims({"inidate": [pd.to_datetime(inidate)]}).chunk({
        "inidate": 1,
        "time": 10,
        "number": 11,
        "step": 13,
        "latitude": 120,
        "longitude": 240,
    })
)
    print('done processing inidate:', inidate)
    if first:
        merged_ds.to_zarr(zarr_path, mode='w')
        print('done writing first inidate to zarr:', inidate)
        first = False
    else:
        merged_ds.to_zarr(zarr_path, mode='a', append_dim='inidate')
        print('done appending inidate to zarr:', inidate)
    
    # Clean up to free memory
    del merged_ds
    gc.collect()