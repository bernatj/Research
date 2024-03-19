# Written by Bernat Jiménez on March 2024 
# Instituto de Geociencias, CSIC-UCM, Madrid

# instantiate workers
from distributed import Client
import xarray as xr
import numpy as np
from scipy.stats import linregress
import dask

# Enable Dask for parallel processing
xr.set_options(enable_cftimeindex=True)

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



def reshape_to_year_month(original_array):
    """
    Reshape an array with time dimensions into an array with separate dimensions for years and months.

    Parameters:
        original_array (numpy.ndarray or xarray.DataArray): The original array with time dimensions. The first dimension represents time.

    Returns:
        numpy.ndarray or xarray.DataArray: The reshaped array with separate dimensions for years and months.
    """
    # If the input is a DataArray, convert it to a numpy array
    if hasattr(original_array, 'data'):
        original_array_data = original_array.data
        coords = original_array.coords

    # Get the shape of the original array
    original_shape = original_array_data.shape

    # Calculate the total number of months
    total_months = original_shape[0]

    # Calculate the number of years and remaining months
    num_years = total_months // 12
    remaining_months = total_months % 12

    # Detect the starting year from the "time" coordinate
    if coords and 'time' in coords:
        first_year = coords['time'].dt.year[0].item()
    else:
        raise ValueError("Could not detect the starting year.")

    # Reshape the array to have separate dimensions for years and months
    if remaining_months == 0:
        reshaped_array = np.reshape(original_array_data, (num_years, 12) + original_shape[1:])
    else:
        num_years += 1
        reshaped_array = np.zeros((num_years, 12) + original_shape[1:])
        reshaped_array[0, :remaining_months] = original_array[:remaining_months]
        for i in range(1, num_years):
            reshaped_array[i] = original_array_data[remaining_months + (i - 1) * 12:remaining_months + i * 12]

    # If the input was a DataArray, convert the reshaped array back to a DataArray
    if hasattr(original_array, 'data'):
        coords = {**{'year': np.arange(first_year, first_year + num_years, 1), 'month': np.arange(1, 13)},  **original_array.drop('time').coords}
        print(coords)
        dims = ['year', 'month'] + list(original_array.dims[1:])
        reshaped_array = xr.DataArray(reshaped_array, coords=coords, dims=dims)

    return reshaped_array

def polyfit_monthly(ds, max_workers=10):
    def polyfit_monthly_func(arr):
        x = np.arange(len(arr))
        slope, _, _, _, _ = linregress(x, arr)
        return slope
    
    # Apply linear regression along the time dimension
    with dask.config.set(scheduler='threads', num_workers=max_workers):
        slopes = xr.apply_ufunc(
            polyfit_monthly_func,
            ds,
            input_core_dims=[['year']],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',  # Enable parallelization for dask arrays
            output_dtypes=[float]
        )
    
    return slopes

def create_delta_pwg(levels,surface_vars, pressure_vars, output_dir):
    """
    Create delta climatology for pressure and surface weather variables.
    
    Parameters:

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
        if var == 'r':
            ds_pl_vars[var] = xr.open_mfdataset(f'{dir}rh/rh-*-2.nc')[var].sel(level=levels)
        else:
            ds_pl_vars[var] = xr.open_mfdataset(f'{dir}{var}/{var}-*-2.nc')[var].sel(level=levels)

    ds_pl = xr.merge(list(ds_pl_vars.values()))

    # Calculate trends for surface variables
    for var in surface_vars:
        print(f'calculating climatologies for {var}...')
        ds_sl[f'{var}_trend_mon'] = polyfit_monthly(reshape_to_year_month(ds_sl[var]).compute())

    # Calculate trends for pressure variables
    for var in pressure_vars:
        print(f'calculating climatologies for {var}...')
        ds_pl[f'{var}_trend_mon'] = polyfit_monthly(reshape_to_year_month(ds_pl[var]).compute())

    # Interpolate to daily 
    for var in surface_vars:
        print(f'calculating daily trends from monthly climatology for {var}...')
        ds_sl[var + '_trend_day'] = interpolate_to_daily_climatology(ds_sl[var + '_trend_mon'])
        print(f'saving in file...')
        ds_sl[var + '_trend_day'].to_netcdf(os.path.join(surface_dir, f'{var}_trend_daily.nc'))
    
    for var in pressure_vars:
        ds_pl[var + '_delta_dayclim'] = interpolate_to_daily_climatology(ds_pl[var + '_delta'])
        ds_pl[var + '_trend_day'] = interpolate_to_daily_climatology(ds_pl[var + '_trend_mon'])
        print(f'saving in file...')
        ds_pl[var + '_trend_day'].to_netcdf(os.path.join(pressure_dir, f'{var}_trend_daily.nc'))
    return xr.merge([ds_sl, ds_pl])

# Example usage:
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

surface_vars = ['t2m','tcwv']
pressure_vars = ['t', 'r']
output_dir = '/pool/usuarios/bernatj/Data/pgw-climatologies'


result = create_delta_pwg(levels, surface_vars, pressure_vars, output_dir)
print(result)
