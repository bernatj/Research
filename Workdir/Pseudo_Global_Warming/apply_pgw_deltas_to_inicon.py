import metpy 
from metpy.units import units
import numpy as np
import xarray as xr
import datetime as dt


def adjust_geopotential(z, t, r):
    """
    Adjust geopotential based on temperature and relative humidity.
    
    Parameters:
        z (xarray.DataArray): Geopotential data.
        t (xarray.DataArray): Temperature data.
        r (xarray.DataArray): Relative humidity data.
        
    Returns:
        xarray.DataArray: Adjusted geopotential.
    """
    R_d = 287.05    # gas constant for dry air [J K-1 kg-1]

    pressure = z.level.broadcast_like(z)
    lnp = np.log(pressure)

    # Calculate mixing ratio from relative humidity
    w = metpy.calc.mixing_ratio_from_relative_humidity(pressure * units.hPa , t * units.K , r/100)

    # Calculate virtual temperature
    tv = metpy.calc.virtual_temperature(t*units.K, w).values
   
    z_new = z.copy()

    # Integrate with height using the hydrostatic balance and ideal gas law
    for k in range(0, len(z.level)-1, 1):
        tv_k12 = (tv[k]*lnp[k] + t[k+1]*lnp[k+1])/(lnp[k] + lnp[k+1])
        z_new[k+1] = z_new[k] - (lnp[k+1] - lnp[k]) * R_d * tv_k12

    return z_new

def apply_delta_to_initial_condition(start_date, surface_delta_files, pressure_delta_files):
    """
    Apply delta files to an existing initial condition given a start date.
    
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        surface_delta_files (dict): Dictionary containing surface delta file paths for each variable.
        pressure_delta_files (dict): Dictionary containing pressure delta file paths for each variable.
        
    Returns:
        xarray.Dataset: Modified initial condition.
    """
    # Open initial condition dataset
    initial_condition = xr.open_dataset('/path/to/initial_condition.nc')

    # Convert start_date to datetime object
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')

    # Calculate day of the year for start_date
    day_of_year = start_date.timetuple().tm_yday

    # Apply surface delta
    for var, delta_file in surface_delta_files.items():
        delta_data = xr.open_dataset(delta_file)
        initial_condition[var] -= delta_data.sel(dayofyear=day_of_year)[var + '_delta_dayclim']

    # Apply pressure delta
    for var, delta_file in pressure_delta_files.items():
        delta_data = xr.open_dataset(delta_file)
        initial_condition[var] -= delta_data.sel(dayofyear=day_of_year)[var + '_delta_dayclim']

    # Adjust geopotential
    initial_condition['z'] = adjust_geopotential(initial_condition['z'], initial_condition['t'], initial_condition['r'])

    return initial_condition

# Example usage:
start_date = '2024-03-10'
surface_delta_files = {'t2m': 't2m_delta_era5_1940to1970_1992to2022.nc',
                       'tcwv': 'tcwv_delta_era5_1940to1970_1992to2022.nc'}
pressure_delta_files = {'t': 't_delta_era5_1940to1970_1992to2022.nc',
                        'r': 'r_delta_era5_1940to1970_1992to2022.nc'}

modified_initial_condition = apply_delta_to_initial_condition(start_date, surface_delta_files, pressure_delta_files)
print(modified_initial_condition)