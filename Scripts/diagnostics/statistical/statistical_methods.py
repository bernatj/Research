import numpy as np
import xarray as xr
from scipy import signal
from eofs.xarray import Eof

def compute_eofs_pcs_regressed(field, num_pcs, season='year-round', region=None):
    """
    Compute EOFs, standardized PCs, and regressed patterns.

    Parameters
    ----------
    field : xarray.DataArray
        The input field.
    num_pcs : int
        The number of Principal Components (PCs) to compute.
    season : str, optional
        The season for anomaly calculation ('year-round', 'DJF', 'MAM', 'JJA', 'SON').
    region : tuple, optional
        The region to select (latitude_s, latitude_n, longitude_w, longitude_e).

    Returns
    -------
    eofs : xarray.DataArray
        EOF patterns.
    pcs : xarray.DataArray
        Standardized PCs.
    regressed_map : xarray.DataArray
        Regressed patterns.

    """
    # Select season
    if season == 'year-round':
        field_season = field
    else:
        field_season = field.where(field['time.season'] == season)

    # Drop missing values
    field_season = field_season.dropna(dim='time', how='any')

    # Seasonal anomalies
    field_anom = field_season - field_season.mean('time')

    # Detrend
    field_anom_dtrend = field_anom.copy(data=signal.detrend(field_anom, axis=0))

    # Select region
    if region is not None:
        lat_s, lat_n, lon_w, lon_e = region
        field_anom_dtrend = field_anom_dtrend.sel(latitude=slice(lat_n, lat_s), longitude=slice(lon_w, lon_e))

    # Compute weights
    weights = np.sqrt(np.cos(np.deg2rad(field_anom_dtrend.latitude))).squeeze()

    # Create solver class
    solver = Eof(field_anom_dtrend.transpose("time", "longitude", "latitude"), center=True, weights=weights)

    # Retrieve EOFs, PCs, and variance fraction
    eofs = solver.eofs(neofs=num_pcs).transpose("mode", "latitude", "longitude")
    pcs = solver.pcs(npcs=num_pcs, pcscaling=1)
    var_frac = solver.varianceFraction(neigs=num_pcs)

    # Standardize PCs
    pcs_std = (pcs - pcs.mean('time')) / pcs.std('time')

    var_frac = solver.varianceFraction(neigs=12) #explained varience

    # Regress the index onto the field
    lats, lons, modes = field_anom.latitude, field_anom.longitude, np.arange(num_pcs)
    regressed_map = xr.DataArray(np.empty([num_pcs, *field_anom[0, :, :].shape]),
                                 coords=(modes, lats, lons), dims=['mode', 'lat', 'lon'])

    for i, lat in enumerate(lats):
        for j, mode in enumerate(modes):
            regressed_map[j, i, :] = np.polyfit(pcs_std[:, j], field_anom[:, i, :], 1)[0]

    return eofs, pcs_std, var_frac, regressed_map


### time filtering fuctions ####
from scipy.signal import butter, lfilter, filtfilt, freqz

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = data.copy(data=y,deep=False) 
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = data.copy(data=y,deep=False) 
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = data.copy(data=y, deep=False) 
    return y
#################################

#cluster analysis

#make a function
