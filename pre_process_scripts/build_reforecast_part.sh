#!/bin/bash
#SBATCH --job-name=reforecast_t2m
#SBATCH --array=0-99%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=priority-predict


#module load python

python build_reforecast_part.py \
  --var tp \
  --dates-per-task 4 \
  --outdir /network/group/aopp/predict/MRA001_AYIM_REFORCST/zarr_parts_tp
