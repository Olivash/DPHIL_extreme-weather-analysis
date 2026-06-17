import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import linregress, kendalltau
import seaborn as sns
from  matplotlib.ticker import FuncFormatter
from scipy.stats import genextreme as gev
import scipy.stats as stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime,timedelta
#import LMom as lm
from scipy.stats import genpareto as gpd
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import matplotlib.dates as mdates
import functions as f
import matplotlib.colors
import matplotlib.ticker as ticker

import matplotlib.dates as mdates
import matplotlib.lines as mlines


def open_and_process_one_inidate(inidate, fpaths):
    files = [fp for fp in fpaths if inidate in fp]

    ds = xr.open_mfdataset(
        files,
        preprocess=lambda x: preproc_mclim(x[["mx2t6"]]),
        combine="by_coords",
        parallel=False,
        chunks={'time': 1, 'step': -1, 'latitude': -1, 'longitude': -1},
    )

    return ds
# main preprocessing function
def preproc_ds(ds):
    ds = ds.copy().squeeze()  # Copy and remove singleton dimensions
    
    # Extract filename for metadata
    fname = ds.encoding['source'].split('/')[-1].split('.')[0]
    expver = fname.split('_')[0]
    ds = ds.expand_dims({'experiment': [expver]})  # Add 'experiment' as a dimension
    
    # Extract the initialization date from the time variable
    inidate = pd.to_datetime(ds.time[0].values)
    
    # Add HINDCAST dates and ensemble members dimension if necessary
    if 'hDate' not in ds:
        ds = ds.expand_dims({'inidate': [inidate]})
    if 'number' not in ds:
        ds = ds.expand_dims({'number': [0]})
    
    # Reorganize dimensions to place 'time' first
    ds = ds.transpose('time', ...)
    
    return ds

    
    
    
# need a couple more steps for preprocessing the m-climate

def preproc_mclim(ds):
    ds = ds.copy().squeeze()  # Copy and remove singleton dimensions
    ds = preproc_ds(ds)  # Apply the `preproc_ds` function
    
    # Create an index of hours since the first time point
    ds_hours = ((ds.time - ds.time.isel(time=0)) / 1e9 / 3600).astype(int)
    
    # Assign the calculated hours as a new coordinate and rename 'time' to 'hour'
    ds = ds.assign_coords(time=ds_hours).rename(dict(time='hour'))
    
    return ds


def draw_box_on_map(ax, lat1, lon1, lat2, lon2,color):
    """
    Function to draw a box on a map given the latitude and longitude dimensions.

    Parameters:
        ax (matplotlib.axes.Axes): Subplot axes object
        lat1 (float): Latitude of the first corner
        lon1 (float): Longitude of the first corner
        lat2 (float): Latitude of the second corner
        lon2 (float): Longitude of the second corner
    """

    # Plot the rectangle on the map
    rect = mpatches.Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, linewidth=2, edgecolor=color, facecolor='none',
                               transform=ccrs.PlateCarree())
    ax.add_patch(rect)

#read the txt file
rgb_data_in_the_txt_file = np.loadtxt("../new_project/personal_cmaps/misc_div.txt")

#create the colormap
my_colormap = mcolors.LinearSegmentedColormap.from_list('colormap', (rgb_data_in_the_txt_file/255))


# Define the preprocessing function for ERA5
def preprocess_era(ds):
    # Subset latitude, longitude, and time
    ds = ds.sel(
        latitude=slice(70,30),
        longitude=slice(-150+360,-100+360),
        
    )
    
    # Convert temperature to Celsius if 't2m' exists
    if 't2m' in ds:
        ds['t2m'] = ds['t2m'] - 273.15
        
        
    # Resample to daily maximum values if time dimension exists
    if 'time' in ds.dims:
        ds = ds.resample(time="1D").max()
    
    return ds



# Function to filter data based on month and day regardless of the year
def select_dates_across_years(data, lead):
    dates_array= ds.inidate.values + pd.Timedelta(days=lead)
    # Extract month-day pairs from the dates array
    month_day_pairs = [(pd.to_datetime(date).month, pd.to_datetime(date).day) for date in dates_array]

    selected_data = []
    for month, day in month_day_pairs:
        condition = (data['time.month'] == month) & (data['time.day'] == day)
        selected_data.append(data.sel(time=condition))
    return xr.concat(selected_data, dim='time')

def standard(data):
    # Step 1: Calculate the mean and standard deviation for the series
    mean_data = np.mean(data)
    std_data = np.std(data)

    # Step 2: Calculate the anomalies (difference from the mean)
    anomalies = data- mean_data

    # Step 3: Normalize the anomalies by dividing by the standard deviation
    # This expresses them in units of standard deviation
    normalized_anomalies = anomalies / std_data
    return normalized_anomalies