# Written by Bernat Jiménez on March 2024 
# Instituto de Geociencias, CSIC-UCM, Madrid

import xarray as xr
import numpy as np

def interpolate_to_daily_climatology(data, method='cubic'):
    """
    Interpolates monthly climatology data to daily climatology data.
    
    Parameters:
        data (xarray.Dataset or xarray.DataArray): Monthly climatology dataset or array.
        method (str, optional): Interpolation method. Default is 'cubic'.
        
    Returns:
        xarray.Dataset or xarray.DataArray: Daily climatology dataset or array.
    """
    # Number of days per month
    num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Calculate the center day of each month
    dayofyear = [(num_days[i] // 2) + 1 + sum(num_days[0:i]) for i in range(0, 12)]
    

    # Rename 'month' dimension to 'dayofyear'
    data = data.rename({'month': 'dayofyear'})
        
    # Assign new coordinates for 'dayofyear'
    data = data.assign_coords({'dayofyear': dayofyear})
        
    # Interpolate to daily resolution
    data_interpolated = data.interp(dayofyear=np.arange(1, 366), method=method)

    return data_interpolated

import os

def create_delta_pwg(levels, period_1, period_2, surface_vars, pressure_vars, output_dir):
    """
    Create delta climatology for pressure and surface weather variables.
    
    Parameters:
        levels (list): List of pressure levels to interpolate.
        period_1 (slice): Time period 1.
        period_2 (slice): Time period 2.
        surface_vars (list): List of surface weather variables.
        pressure_vars (list): List of pressure variables.
        output_dir (str): Directory to store the output data.
        
    Returns:
        xarray.Dataset: Daily climatology dataset.
    """
    # Make sure output directories exist
    pressure_dir = os.path.join(output_dir, 'pressure')
    surface_dir = os.path.join(output_dir, 'surface')
    os.makedirs(pressure_dir, exist_ok=True)
    os.makedirs(surface_dir, exist_ok=True)
    
    # Read monthly data
    dir = '/pool/usuarios/bernatj/Data/my_data/era5/monthly/'

    # Read surface variables
    ds_sl_vars = {}
    for var in surface_vars:
        ds_sl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*.nc')[var]
    
    ds_sl = xr.merge(list(ds_sl_vars.values()))

    # Read pressure variables
    ds_pl_vars = {}
    for var in pressure_vars:
        ds_pl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*.nc')[var]

    ds_pl = xr.merge(list(ds_pl_vars.values()))

    # Calculate delta for surface variables
    for var in surface_vars:
        var_clim_period1 = ds_sl_vars[var].sel(time=period_1).groupby('time.month').mean('time')
        var_clim_period2 = ds_sl_vars[var].sel(time=period_2).groupby('time.month').mean('time') 
        var_diff = var_clim_period2 - var_clim_period1

        ds_sl[var + '_delta'] = var_diff
    
    # Calculate delta for pressure variables
    for var in pressure_vars:
        var_clim_period1 = ds_pl_vars[var].sel(time=period_1).groupby('time.month').mean('time')
        var_clim_period2 = ds_pl_vars[var].sel(time=period_2).groupby('time.month').mean('time') 
        var_diff = var_clim_period2 - var_clim_period1

        ds_pl[var + '_delta'] = var_diff

    # Interpolate to daily climatology
    for var in surface_vars:
        ds_sl[var + '_delta_dayclim'] = interpolate_to_daily_climatology(ds_sl[var + '_delta'])
        ds_sl[var + '_delta_dayclim'].to_netcdf(os.path.join(surface_dir, f'{var}_delta_dayclim.nc'))
    
    for var in pressure_vars:
        ds_pl[var + '_delta_moncli'] = ds_pl[var + '_delta'].interp(level=levels)
        ds_pl[var + '_delta_dayclim'] = interpolate_to_daily_climatology(ds_pl[var + '_delta_moncli'])
        ds_pl[var + '_delta_dayclim'].to_netcdf(os.path.join(pressure_dir, f'{var}_delta_dayclim.nc'))

    return xr.merge([ds_sl, ds_pl])

# Example usage:
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
period_1 = slice('1940-01-01','1970-12-31')
period_2 = slice('1992-01-01','2022-12-31')
surface_vars = ['t2m', 'tcwv']
pressure_vars = ['t', 'r']
output_dir = '/path/to/output/'

result = create_delta_pwg(levels, period_1, period_2, surface_vars, pressure_vars, output_dir)
print(result)
