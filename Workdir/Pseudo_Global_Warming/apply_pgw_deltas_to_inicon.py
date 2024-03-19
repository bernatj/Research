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

    try:
        pressure = z.level.broadcast_like(z)
    except:
        pressure = z.isobaricInhPa.broadcast_like(z)

    lnp = np.log(pressure)

    # Calculate mixing ratio from relative humidity
    w = metpy.calc.mixing_ratio_from_relative_humidity(pressure * units.hPa , t * units.K , r/100)

    # Calculate virtual temperature
    tv = metpy.calc.virtual_temperature(t*units.K, w).values
   
    z_new = z.copy()

    # Integrate with height using the hydrostatic balance and ideal gas law
    for k in range(0, len(z.isobaricInhPa)-1, 1):
        tv_k12 = (tv[k]*lnp[k] + t[k+1]*lnp[k+1])/(lnp[k] + lnp[k+1])
        z_new[k+1] = z_new[k] - (lnp[k+1] - lnp[k]) * R_d * tv_k12

    return z_new

def apply_delta_to_initial_condition(start_date, initial_conditions_files, surface_delta_files, pressure_delta_files):
    """
    Apply delta files to an existing initial condition given a start date.
    
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        initial_condition_files (dict): path containing the initial condition file
        surface_delta_files (dict): Dictionary containing surface delta file paths for each variable.
        pressure_delta_files (dict): Dictionary containing pressure delta file paths for each variable.
        
    Returns:
        xarray.Dataset: Modified initial condition.
    """

    # Convert start_date to datetime object
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')

    # Calculate day of the year for start_date
    day_of_year = start_date.timetuple().tm_yday

    # Apply surface delta
    # Open initial condition dataset
    initial_condition_S  = xr.open_dataset(initial_condition_files['surface'],engine='cfgrib')
    initial_condition_S = initial_condition_S.assign_coords(longitude=initial_condition_S["longitude"] % 360)
    initial_condition_S= initial_condition_S.sortby("longitude")
    print(initial_condition_S)

    for var, delta_file in surface_delta_files.items():

        delta_data = xr.open_dataset(delta_file)
        initial_condition_S[var] -= delta_data.sel(dayofyear=day_of_year)[var + '_delta_dayclim']

    # Apply pressure delta
    initial_condition_P  = xr.open_dataset(initial_condition_files['pressure'],engine='cfgrib')
    initial_condition_P = initial_condition_P.assign_coords(longitude=initial_condition_P["longitude"] % 360)
    initial_condition_P = initial_condition_P.sortby("longitude")
    levels = initial_condition_P.isobaricInhPa.values

    for var, delta_file in pressure_delta_files.items():
        delta_data = xr.open_dataset(delta_file).sel(level=levels)
        if var == 'r':
            initial_condition_P[var] -= delta_data.sel(dayofyear=day_of_year)['rh_delta_dayclim'].rename({'level' : 'isobaricInhPa'})
        else:
            initial_condition_P[var] -= delta_data.sel(dayofyear=day_of_year)[var + '_delta_dayclim'].rename({'level' : 'isobaricInhPa'})

    # Adjust geopotential
    initial_condition_P['z'] = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['r'])

    return initial_condition_S, initial_condition_P

# Example usage:
start_date = '2024-07-31'
path_delta_surf = '/home/bernatj/Data/pgw-climatologies/surface/'
surface_delta_files = {'t2m': path_delta_surf + 't2m_delta_dayclim.nc',
                       'tcwv': path_delta_surf + 'tcwv_delta_dayclim.nc'}
path_delta_press = '/home/bernatj/Data/pgw-climatologies/pressure/'
pressure_delta_files = {'t': path_delta_press + 't_delta_dayclim.nc',
                        'r': path_delta_press + 'rh_delta_dayclim.nc'}

path_ic = '/home/bernatj/Data/ai-forecasts/input/grib/'
initial_condition_files = {'surface' : path_ic + 'fcnv2sm_sl_20180731.grib',
                           'pressure' : path_ic + 'fcnv2sm_pl_20180731.grib'}

mod_initial_condition_S, mod_initial_condition_P = apply_delta_to_initial_condition(start_date, initial_condition_files, surface_delta_files, pressure_delta_files)

mod_initial_condition_S.to_netcdf('/home/bernatj/Data/ai-forecasts/input/netcdf/fcnv2sm_sl_PGW_era5_20180731.nc')
mod_initial_condition_P.to_netcdf('/home/bernatj/Data/ai-forecasts/input/netcdf/fcnv2sm_pl_PGW_era5_20180731.nc')