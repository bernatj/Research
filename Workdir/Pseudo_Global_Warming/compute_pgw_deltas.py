# Written by Bernat Jiménez on March 2024 
# Instituto de Geociencias, CSIC-UCM, Madrid

import xarray as xr
import numpy as np

def interpolate_to_daily_climatology(data, method='linear'):
    """
    Interpolates monthly climatology data to daily climatology data.
    
    Parameters:
        data (xarray.Dataset or xarray.DataArray): Monthly climatology dataset or array.
        method (str, optional): Interpolation method. Default is 'linear'.
        
    Returns:
        xarray.Dataset or xarray.DataArray: Daily climatology dataset or array.
    """
    # Number of days per month
    num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Calculate the center day of each month
    dayofyear = [(num_days[i] // 2) + 1 + sum(num_days[0:i]) for i in range(0, 12)]

    #add one december at the begining and january at the end to be able to interpolate all days
    data_ext = data.pad(month=1, mode='wrap') 
   
    dayofyear_ext = [dayofyear[0]-30] + dayofyear + [dayofyear[-1]+30]
    
    # Rename 'month' dimension to 'dayofyear'
    data_ext = data_ext.rename({'month': 'dayofyear'})
        
    # Assign new coordinates for 'dayofyear'
    data_ext = data_ext.assign_coords({'dayofyear': dayofyear_ext})

    # Interpolate to daily resolution
    data_interpolated = data_ext.interp(dayofyear=np.arange(1, 366), method=method)

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
        print(var)
        ds_sl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*-2.nc')[var]
    
    ds_sl = xr.merge(list(ds_sl_vars.values()))

    # Read pressure variables
    ds_pl_vars = {}
    for var in pressure_vars:
        if var == 'rh':
            ds_pl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*-2.nc')['r']
        else:
            ds_pl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*-2.nc')[var]

    ds_pl = xr.merge(list(ds_pl_vars.values()))

    # Calculate delta for surface variables
    for var in surface_vars:
        print(f'calculating climatologies for {var}...')
        var_clim_period1 = ds_sl_vars[var].sel(time=period_1).groupby('time.month').mean('time')
        var_clim_period2 = ds_sl_vars[var].sel(time=period_2).groupby('time.month').mean('time') 
        print(f'calculating the delta between the two periods...')
        var_diff = var_clim_period2 - var_clim_period1

        ds_sl[var + '_delta'] = var_diff
        ds_sl[var + '_delta'].to_netcdf(os.path.join(surface_dir, f'{var}_delta_monclim.nc'))
    
    # Calculate delta for pressure variables
    for var in pressure_vars:
        print(f'calculating climatologies for {var}...')
        var_clim_period1 = ds_pl_vars[var].sel(time=period_1).groupby('time.month').mean('time')
        var_clim_period2 = ds_pl_vars[var].sel(time=period_2).groupby('time.month').mean('time') 
        print(f'calculating the delta between the two periods...')
        var_diff = var_clim_period2 - var_clim_period1

        ds_pl[var + '_delta'] = var_diff
        ds_pl[var + '_delta'].to_netcdf(os.path.join(pressure_dir, f'{var}_delta_monclim.nc'))

    # Interpolate to daily climatology
    for var in surface_vars:
        print(f'calculating daily climatology from monthly climatology for {var}...')
        ds_sl[var + '_delta_dayclim'] = interpolate_to_daily_climatology(ds_sl[var + '_delta'])
        print(f'saving in file...')
        ds_sl[var + '_delta_dayclim'].to_netcdf(os.path.join(surface_dir, f'{var}_delta_dayclim.nc'))
    
    for var in pressure_vars:
        print(f'interpolating pressure levels for {var}...')
        ds_pl[var + '_delta_moncli'] = ds_pl[var + '_delta'].sel(level=levels)
        print(ds_pl[var + '_delta_moncli'])
        print(f'calculating daily climatology from monthly climatology for {var}...')
        ds_pl[var + '_delta_dayclim'] = interpolate_to_daily_climatology(ds_pl[var + '_delta_moncli'])
        print(ds_pl[var + '_delta_dayclim'].values)
        print(f'saving in file...')
        ds_pl[var + '_delta_dayclim'].to_netcdf(os.path.join(pressure_dir, f'{var}_delta_dayclim.nc'))

    return xr.merge([ds_sl, ds_pl])

# Example usage:
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
period_1 = slice('1940-01-01','1970-12-31')
period_2 = slice('1990-01-01','2022-12-31')
surface_vars = ['t2m', 'tcwv']
pressure_vars = ['t', 'rh']
output_dir = '/pool/usuarios/bernatj/Data/pgw-climatologies'

result = create_delta_pwg(levels, period_1, period_2, surface_vars, pressure_vars, output_dir)
print(result)
