import metpy 
from metpy.units import units
import numpy as np
import xarray as xr
import datetime as dt


def adjust_geopotential(z, t, r, sp):
    """
    Adjust geopotential based on temperature and relative humidity.
    
    Parameters:
        z (xarray.DataArray): Geopotential data.
        t (xarray.DataArray): Temperature data.
        r (xarray.DataArray): Relative humidity data.
        sp(xarray.DataArray): Surface Pressure.
        
    Returns:
        xarray.DataArray: Adjusted geopotential.
    """
    R_d = 287.05    # gas constant for dry air [J K-1 kg-1]

    # Convert surface pressure to hPa if it's in Pascals
    if np.max(sp) > 2000:  # Assuming pressure in Pascals if max value is greater than 2000
        sp = sp / 100  # Convert from Pa to hPa

    print(sp.values)

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
        tv_k12 = (tv[k] + t[k+1])/2
        #z_new[k+1] = np.where(pressure[k+1] < sp, z_new[k] - (lnp[k+1] - lnp[k]) * R_d * tv_k12, z_new[k+1])
        z_new[k+1] = z_new[k] - (lnp[k+1] - lnp[k]) * R_d * tv_k12


    return z_new

def apply_delta_to_initial_condition_cmip(initial_condition_files, ds_cmip_deltas):
    """
    Apply delta files to an existing initial condition given a start date.
    
    Parameters:
        initial_condition_files (dict): path containing the initial condition file
        surface_delta_files (dict): Dictionary containing surface delta file paths for each variable.
        pressure_delta_files (dict): Dictionary containing pressure delta file paths for each variable.
        
    Returns:
        xarray.Dataset: Modified initial condition.
    """

    # Apply surface delta
    # Open initial condition dataset
    initial_condition_S  = xr.open_dataset(initial_condition_files['surface'],engine='cfgrib')
    initial_condition_S = initial_condition_S.assign_coords(longitude=initial_condition_S["longitude"] % 360)
    initial_condition_S= initial_condition_S.sortby("longitude")

    for var in ['t2m','tcwv']:
        initial_condition_S[var] -= ds_cmip_deltas[var]

    # Apply pressure delta
    initial_condition_P  = xr.open_dataset(initial_condition_files['pressure'],engine='cfgrib')
    initial_condition_P = initial_condition_P.assign_coords(longitude=initial_condition_P["longitude"] % 360)
    initial_condition_P = initial_condition_P.sortby("longitude")

    z_baro_before = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['r'], \
                                                   initial_condition_S['sp'])
    for var in ['t','r']:
        initial_condition_P[var] -= ds_cmip_deltas[var].rename({'level' : 'isobaricInhPa'})

    # Adjust geopotential
    z_baro_after = adjust_geopotential(initial_condition_P['z'], initial_condition_P['t'], initial_condition_P['r'], \
                                                   initial_condition_S['sp'])
    
    delta_z = z_baro_after - z_baro_before            
    initial_condition_P['z'] = initial_condition_P['z']  - delta_z

    return initial_condition_S, initial_condition_P


def fix_cmip6_data(delta_files, levels):

    dict_vars = {'t2m' : 'tas', 'tcwv' : 'prw', 't' : 'ta', 'r' : 'hur' }
    reversed_dict = {v: k for k, v in dict_vars.items()}

        # Read surface variables
    ds_vars = {}
    for var, delta_file in delta_files.items():
        var_cmip6 = dict_vars[var] 
        ds_vars[var] = xr.open_dataset(delta_file)[var_cmip6]
    
    ds_vars = xr.merge(list(ds_vars.values()))

    ds_vars = ds_vars.rename(reversed_dict)

    # Convert pressure levels from Pa to hPa
    ds_vars['plev'] = ds_vars['plev'] / 100

    # Optionally, update the units attribute
    ds_vars['plev'].attrs['units'] = 'hPa'

    # Interpolate each variable in the dataset to the new pressure levels
    interpolated_ds = xr.Dataset()
    for var_name in ds_vars.data_vars:
        if 'plev' in ds_vars[var_name].dims:  # Check if the variable depends on pressure
            interpolated_var = ds_vars[var_name].interp(plev=levels)
            interpolated_ds[var_name] = interpolated_var
        interpolated_ds[var_name] = ds_vars[var_name]

    # Optionally, update the pressure coordinate values
    interpolated_ds['plev'] = levels

    #rename variable to level
    interpolated_ds = interpolated_ds.rename({'plev' : 'level'})

    #horizontal interpolation to 0,25deg grid
    new_lons=np.arange(0,360,0.25)
    new_lats=np.arange(90,-90.1,-0.25)
    interpolated_grid_ds = interpolated_ds.interp(lon=new_lons, lat=new_lats, method='linear', kwargs={"fill_value": "extrapolate"})
    interpolated_grid_ds = interpolated_grid_ds.rename({'lat' : 'latitude', 'lon' : 'longitude'})

    return interpolated_grid_ds

def interpolate_to_dayofyear(data, day_of_year, method='linear'):
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

    # Rename 'time' dimension to 'dayofyear'
    data= data.rename({'time': 'dayofyear'})

    #add one december at the begining and january at the end to be able to interpolate all days
    data_ext = data.pad(dayofyear=1, mode='wrap') 
   
    dayofyear_ext = [dayofyear[0]-30] + dayofyear + [dayofyear[-1]+30]
           
    # Assign new coordinates for 'dayofyear'
    data_ext = data_ext.assign_coords({'dayofyear': dayofyear_ext})

    # Interpolate to the day we are interested
    data_interpolated = data_ext.interp(dayofyear=day_of_year, method=method)

    return data_interpolated

# Example usage:
start_date = '2024-07-31'
path_delta = '/home/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg-clim/'

dict_vars = {'t2m' : 'tas', 'tcwv' : 'prw', 't' : 'ta', 'r' : 'hur' }

delta_files = {'t2m': path_delta + 'tas/tas_historical_ec-earth3-veg_r1i1p1f1_interpolated_1980-2014_1850-1900_delta.nc',
               'tcwv': path_delta + 'prw/prw_historical_ec-earth3-veg_r1i1p1f1_interpolated_1980-2014_1850-1900_delta.nc',
               't' : path_delta + 'ta/ta_historical_ec-earth3-veg_r1i1p1f1_interpolated_1980-2014_1850-1900_delta.nc',
               'r' : path_delta + 'hur/hur_historical_ec-earth3-veg_r1i1p1f1_interpolated_1980-2014_1850-1900_delta.nc',}

path_ic = '/home/bernatj/Data/ai-forecasts/input/grib/'
initial_condition_files = {'surface' : path_ic + 'fcnv2sm_sl_20180731.grib',
                           'pressure' : path_ic + 'fcnv2sm_pl_20180731.grib'}

#first we need to create the daily clims and interpolate to the right grid
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

print('interpolate data to ERA5 resolution and levels for the AI model ...')
ds_cmip6_deltas = fix_cmip6_data(delta_files, levels)

#need to interpolate to daily climatology
# Convert start_date to datetime object
start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')

# Calculate day of the year for start_date
print('interpolate to day of the year ...')
day_of_year = start_date.timetuple().tm_yday
ds_cmip6_deltas_doy = interpolate_to_dayofyear(ds_cmip6_deltas, day_of_year)

print('applying teh deltas...')
mod_initial_condition_S, mod_initial_condition_P = apply_delta_to_initial_condition_cmip(initial_condition_files, ds_cmip6_deltas_doy)

mod_initial_condition_S.to_netcdf('/home/bernatj/Data/ai-forecasts/input/netcdf/fcnv2sm_sl_PGW_ec-earth3-veg_v4_20180731.nc')
mod_initial_condition_P.to_netcdf('/home/bernatj/Data/ai-forecasts/input/netcdf/fcnv2sm_pl_PGW_ec-earth3-veg_v4_20180731.nc')