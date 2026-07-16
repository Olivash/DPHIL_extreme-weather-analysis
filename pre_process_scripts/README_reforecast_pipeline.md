# Reforecast Processing Pipeline

########### RUN ON XESMF-ENV#######################

This repository contains scripts for processing ECMWF reforecast GRIB files into Zarr stores for:
- 2 m temperature (`t2m`)
- total precipitation (`tp`)

## What the pipeline does

The workflow:
1. Finds GRIB files by month.
2. Groups files by initialization date.
3. Opens control (`cf`) and perturbed (`pf`) members with `cfgrib`.
4. Preprocesses the data:
   - `t2m`: selects `shortName="2t"` and computes 4-step maxima.
   - `tp`: selects `shortName="tp"`, converts accumulated precipitation to increments, then aggregates to daily totals.
5. Writes one Zarr store per SLURM array task.
6. Optionally merges part stores later into a final dataset.

## Files

- `build_reforecast_part.py`  
  Processes a slice of initialization dates and writes one part Zarr store per task.

- `merge_reforecast_parts.py`  
  Merges part Zarr stores into a final dataset, if you want a single store at the end.

- `build_reforecast_part.sh`  
  SLURM job array submission script.

- `merge_reforecast.slurm`  
  Example merge job submission script.

## Requirements

Recommended environment:
- Python 3.10+
- `xarray`
- `cfgrib`
- `dask`
- `zarr`
- `pandas`
- `numpy`
- `scipy`

You also need ECMWF GRIB support libraries available in the environment used by `cfgrib`.

## Notes on cfgrib filtering

For the temperature files, the data variable is saved as `t2m`, but the GRIB filter key is:

```python
{"shortName": "2t"}
```

For precipitation, the filter key is:

```python
{"shortName": "tp"}
```

The script uses `indexpath=""` to avoid creating persistent cfgrib index files on shared filesystems.

## Running a test locally

Example:

```bash
python3 build_reforecast_part.py --var t2m --outdir ./zarr_parts_test
python3 build_reforecast_part.py --var tp --outdir ./zarr_parts_test
```

## Running with SLURM

Example array job:

```bash
#SBATCH --array=0-99%20
python build_reforecast_part.py --var t2m --dates-per-task 4 --outdir /network/group/aopp/predict/MRA001_AYIM_REFORCST/zarr_parts_t2m
```

And similarly for precipitation:

```bash
python build_reforecast_part.py --var tp --dates-per-task 4 --outdir /network/group/aopp/predict/MRA001_AYIM_REFORCST/zarr_parts_tp
```

## Output structure

Example output:

```text
zarr_parts_t2m/
  reforecast_t2m_part_0000.zarr/
  reforecast_t2m_part_0001.zarr/
  ...

zarr_parts_tp/
  reforecast_tp_part_0000.zarr/
  reforecast_tp_part_0001.zarr/
  ...
```

## Merge step

After the array jobs finish, merge the part stores into a final dataset.

## Caution

Do not run multiple jobs appending to the same Zarr store at the same time.  
The safer pattern is:
- one part store per task, or
- region writes into a preallocated final store.
