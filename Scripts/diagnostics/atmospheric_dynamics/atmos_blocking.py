#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
from scipy import interpolate

def cal_onset_date(x):
    """
    Calculate the onset, end, and duration of an event marked by 1s.

    Parameters:
    x (numpy.ndarray): Array of values indicating events (0 or 1).

    Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray: Arrays containing the onset,
    end, and duration of events.
    """
    # Calculate the difference of shifted arrays
    shift = np.array(x[1:]) - np.array(x[0:len(x)-1]).astype(int)
    onset = np.array(np.where(shift == 1)).astype(int).flatten()  # Transition from 0 to 1
    end = np.array(np.where(shift == -1)).astype(int).flatten()  # Transition from 1 to 0

    # Check the first and last event
    if len(end) == 0 or len(onset) == 0:
        duration = end
    else:
        if end[0] <= onset[0]:
            end = end[1:len(onset)+1]
        try:
            if onset[len(onset)-1] >= end[len(end)-1]:
                onset = onset[0:len(onset)-1]
        except:
            pass
        duration = np.array(end - onset)

    return onset, end, duration

def filter_min_duration(x, min_dur):
    """
    Filter events based on a minimum duration criterion.

    Parameters:
    x (numpy.ndarray): Array of values indicating events.
    min_dur (int): Minimum duration of events to be retained.

    Returns:
    numpy.ndarray: Filtered array of events based on the minimum duration criterion.
    """
    on, end, dur = cal_onset_date(x)
    x_new = np.copy(x)
    if len(on) >= 1:
        on_d = on[np.logical_not(np.isnan(xr.where(dur >= min_dur, np.nan, on)))].astype(int)
        end_d = end[np.logical_not(np.isnan(xr.where(dur >= min_dur, np.nan, end)))].astype(int)
        for k, onset in enumerate(on_d):
            x_new[onset:end_d[k]+1] = 0
    del(on, end, dur)
    return x_new


def near_box_points(index):
    i_smth_1 = index.pad(lat=1,constant_values=0).rolling(lat=3, center=True).mean().dropna('lat')
    i_smth_2 = i_smth_1.pad(lon=2, mode='wrap').rolling(lon=5, center=True).mean().dropna('lon')
    index_new = xr.where((i_smth_2>0.01),1,0) 
    del(i_smth_1,i_smth_2)
    return index_new

def calculate_blocking_index(z, dlat=15, min_lon_ext=6, min_dur=5, res=1.0):
    """
    Calculate the blocking index based on geopotential height data.

    Parameters:
    z (xarray.DataArray): Geopotential height data.
    dlat (float): Latitude interval for calculations (in degrees)
    min_lon_ext (int): Minimum longitudinal extent (in N. of gridpoints)
    min_dur (int): Minimum duration of blocking events (in days)

    Returns:
    xarray.Dataset: Computed blocking indices.
    """

    # TM index (Absolute geopotential height index)
    lat = z.lat
    res = res
    dj = int(dlat/res)

    ###### Gradient reversal index calculations ###########3

    #gradient South
    Z_GS = (z.sel(lat=slice(90-dlat,2*dlat)) - np.array(z.sel(lat=slice(90-2*dlat,dlat))))/dlat
    #gradient North
    Z_GN = (- z.sel(lat=slice(90-dlat,dlat)) + np.array(z.sel(lat=slice(90,dlat*2))))/dlat

    #conditions: reversal of the wind south
    Z_GS_reversal = xr.where(Z_GS>0, 1,0)
    #winds north are westerlies 
    Z_GN_condition = xr.where(Z_GN<-10, 1,0) #

    #alltogether 
    i_block_tm = xr.where((Z_GS_reversal == 1) & (Z_GN_condition == 1), 1, 0)


    ####### Anomaly index calculations ###########3

    #31 days running mean climatology
    z_clim = z.groupby('time.dayofyear').mean('time').load()
    z_clim_smth = z_clim.pad(dayofyear=15, mode='wrap').rolling(dayofyear=31, center=True).mean().dropna(dim="dayofyear")
    
    #anomaly respect to the smoothed climatology
    z_anom = (z.groupby('time.dayofyear') - z_clim_smth)
    
    #compute the treshold for each day, independent of lat
    
    #select the 50 to 80 weighted latitudinal band
    z_9020 = z_anom.sel(lat=slice(90,20))
    #calculate the 90th percentile for each day of year
    z_9020_p90 = z_9020.groupby('time.dayofyear').quantile(0.9, dim=('time','lat','lon')) 
    
    #smooth it 90 days
    z_p90_smth = z_9020_p90.pad(dayofyear=45, mode='wrap').rolling(dayofyear=91, center=True).mean().dropna(dim="dayofyear")
    i_block_anom = xr.where(z_anom.groupby('time.dayofyear')>z_p90_smth, 1,0).sel(lat=slice(90,20))


    # Combined method (fulfillment of the two criteria)
    i_block_mix = xr.where((i_block_tm == 1) & (i_block_anom == 1), 1, 0)

    # Further criteria (minimum longitudinal extent)
    i_block_mix_S = remove_single_points(i_block_mix)
    i_block_anom_S = remove_single_points(i_block_anom)
    i_block_tm_S = remove_single_points(i_block_tm)

    # Loop over lat, lon and modify the i_block_S on the fly
    # filering mimimum longitudinal extent...')
    for i, lat in enumerate(i_block_tm.lat):
        print("lat= "+str(lat.data))
        for t,time in enumerate(i_block_tm.time):
            i_block_tm_S[t,i,:] = filter_min_duration(i_block_tm[t,i,:].pad(lon=4,mode='wrap').values, min_lon_ext)[4:-4]
            i_block_mix_S[t,i,:] = filter_min_duration(i_block_mix[t,i,:].pad(lon=4,mode='wrap').values, min_lon_ext)[4:-4]
            i_block_anom_S[t,i,:] = filter_min_duration(i_block_anom[t,i,:].pad(lon=4,mode='wrap').values, min_lon_ext)[4:-4]


    ####### Minimum duration criteria #######################

    #A blocking event is finally defined if a LSB occurs within a box 5° latitude × 10° longitude centered on that grid point for at least 5 days. (3 lat points, and 5 lon points)\
    #put a one if some point the above mentioned box is blocked
    i_block_mix_S2 = near_box_points(i_block_mix_S)
    i_block_tm_S2 = near_box_points(i_block_tm_S)
    i_block_anom_S2 = near_box_points(i_block_anom_S)

    #define new set of vars with the same size
    i_block_tm_persist = i_block_tm.copy(data=np.zeros(i_block_tm.shape)) #to have the same coordinates
    i_block_anom_persist = i_block_anom.copy(data=np.zeros(i_block_anom.shape)) #to have the same coordinates
    i_block_mix_persist = i_block_mix.copy(data=np.zeros(i_block_mix.shape)) #to have the same coordinates

    # loop over lat, lon and modify the i_block_S on the fly removing blocks with a duration fo less than 5 days
    min_dur = 5 #days
    for i, lat in enumerate(i_block_tm.lat):
        print("lat= "+str(lat.data))
        for j,lon in enumerate(i_block_tm.lon):
            i_block_tm_persist[:,i,j] = filter_min_duration(i_block_tm_S2[:,i,j].values, min_dur)

    min_dur = 5 #days
    for i, lat in enumerate(i_block_anom.lat):
        print("lat= "+str(lat.data))
        for j,lon in enumerate(i_block_anom.lon):
            i_block_anom_persist[:,i,j] = filter_min_duration(i_block_anom_S2[:,i,j].values, min_dur)

    min_dur = 5 #days
    for i, lat in enumerate(i_block_mix.lat):
        print("lat= "+str(lat.data))
        for j,lon in enumerate(i_block_mix.lon):
            i_block_mix_persist[:,i,j] = filter_min_duration(i_block_mix_S2[:,i,j].values, min_dur)


    ds = xr.Dataset(coords=i_block_anom.coords)

    # Add the new data
    ds = ds.assign(i_block_tm_persist=i_block_tm_persist)
    ds = ds.assign(i_block_anom_persist=i_block_anom_persist)
    ds = ds.assign(i_block_mix_persist=i_block_mix_persist)

    return ds

# #Example usage
# input_dir = '/path/to/your/input/data/files/*.nc'
# z = xr.open_mfdataset(input_dir, combine='by_coords').load()
# block_ds = calculate_blocking_index(z=z,dlat=15,min_lon_ext=6)
