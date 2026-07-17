############################################
### Dependencies

##############################################
import glob
import os
import gc
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import xarray as xr





# Define spatial  bounds (replace with actual values)
latitude_upper = 52  
latitude_lower = 45  
longitude_upper =237   #-123
longitude_lower =241 # -119


# base directory containing the reforecast GRIB files, organised by month
base = "/network/group/aopp/predict/MRA001_AYIM_REFORCST"

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





######## FUNCTIONS ###############
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


def preproc_mclim(ds):
    ds = preproc_ds(ds)

    ds = ds.sel(
        latitude=slice(latitude_upper, latitude_lower),
        longitude=slice(longitude_upper, longitude_lower),
    )

    tp_6h = ds["tp"].diff("step") * 1000
    daily_tp = tp_6h.resample(step="1D").sum()

    daily_tp.attrs["units"] = "mm day$^{-1}$"
    daily_tp = daily_tp.mean(dim=("latitude", "longitude"))

    return daily_tp.to_dataset(name="tp")



def open_by_type(inidate, fpaths):
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
            backend_kwargs={"filter_by_keys": {"stepType": "instant","shortName": "tp"}},
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


def process_inidate(inidate, fpaths):
    datasets = open_by_type(inidate, fpaths)
    prepped = [preproc_mclim(ds) for ds in datasets]
    merged = xr.concat(prepped, dim='number')
    #merged = merged.expand_dims({'inidate': [pd.to_datetime(inidate)]})
    return merged

import gc


zarr_path = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/reforecast_tp_FINAL.zarr"
first = True
for inidate in inidates:
    with ProgressBar():
        print(f"Processing inidate: {inidate}")
        merged_ds = (
        process_inidate(inidate, fpaths)
        .expand_dims({"inidate": [pd.to_datetime(inidate)]}).chunk({
            "inidate": 1,
            "time": 10,
            "number": 11,
            "step": 13
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






