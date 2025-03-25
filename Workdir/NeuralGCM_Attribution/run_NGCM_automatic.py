import gcsfs
import os
import jax
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # second gpu
import numpy as np
import pickle
import xarray as xr
import cftime
import metpy 
from metpy.units import units

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

import datetime 


###-----------------------CONFIGURATION--------------------------------------###
#dates
start_time = '2018-7-22-00'
end_time = '2018-7-22-00'
data_inner_steps = 6  # process every 24th hour

#path_delta_mm = '/home/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg-multimodel/'
path_delta_mm = '/home/bernatj/Data/postprocessed-cmip6/climatology-interpolated-2p5deg-deltas/'

# 'neural_gcm_dynamic_forcing_deterministic_0_7_deg.pkl',
# 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl', 
# 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl',
# 'neural_gcm_dynamic_forcing_stochastic_1_4_deg.pkl'] 
model_name = 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'

#read delta q for mm mean:  
flag_pwg=True
adjust_z =True
delta_vars = {'specific_humidity' : 'hus', 'temperature' : 'ta', 
              'sea_surface_temperature' : 'tos',
              'sea_ice_cover' : 'siconc'}

#delta_vars = {'specific_humidity' : 'hus', 'temperature' : 'ta'}
#delta_vars = {'sea_surface_temperature' : 'tos', 'sea_ice_cover' : 'siconc'}
#delta_vars = {'sea_surface_temperature' : 'tos'}


#simulation options.
#we want to run our model for different dates
t0_i = datetime.datetime(2018,7,22,0)
#t0_f = datetime.datetime(2022,4,1,18)
t0_f = datetime.datetime(2018,7,22,0)
delta_h = 6
inner_steps = 6  # in hours
outer_steps = 1 * 24 // inner_steps  # total of 10 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours
n_members=20 #used if stocastic model is chosen
seeds = np.random.randint(100, 2000, size=n_members) #seeds used to generate random initial perturbations

#data_storage:
save_path='/pool/usuarios/bernatj/Data/NeuralGCM_forecasts/'

#exper = 'stoc_1p4'
#exper = 'stoc_1p4_pgw_t_q_siconc_sst'
#exper = 'det_1p4'
exper = 'det_1p4_pgw_t_q_zadjust_siconc_sst_v1'
#exper = 'det_1p4_pgw_t_q_v1'
#exper = 'det_1p4_pgw_siconc_sst_v1'
#exper = 'det_1p4_pgw_sst_v1'
###--------------------------------------------------------------------------###

def adjust_geopotential(z, t, r, sp=False, plev_dim_name='isobaricInhPa'):
    """
    Adjust geopotential based on temperature and relative humidity.
    
    Parameters:
        z (xarray.DataArray): Geopotential data.
        t (xarray.DataArray): Temperature data.
        r (xarray.DataArray): Relative humidity data.
        sp (Bolean): If True relative humidity data is specific humidity instead
        
    Returns:
        xarray.DataArray: Adjusted geopotential.
    """
    R_d = 287.05    # gas constant for dry air [J K-1 kg-1]


    pressure = z[plev_dim_name].broadcast_like(z)
    lnp = np.log(pressure)

    print(pressure)

    # Calculate mixing ratio from relative humidity or specific humidity
    if not sp:
        w = metpy.calc.mixing_ratio_from_relative_humidity(pressure * units.hPa , t * units.K , r/100)
    elif sp:
        w = r / (1 -r)

    # Calculate virtual temperature
    tv = metpy.calc.virtual_temperature(t*units.K, w).values
   
    z_new = z.copy()

    #check the order of the pressure levels
    if z[plev_dim_name][0] > z[plev_dim_name][-1]:
        # Integrate with height using the hydrostatic balance and ideal gas law
        for k in range(0, len(z[plev_dim_name])-1, 1):
            tv_k12 = (tv[k] + tv[k+1])/2
            z_new[k+1] = z_new[k] - (lnp[k+1] - lnp[k]) * R_d * tv_k12
    else:
        # Integrate with height using the hydrostatic balance and ideal gas law
        for k in range(len(z[plev_dim_name])-1, 0, -1):
            print(k)
            tv_k12 = (tv[k] + tv[k-1])/2
            z_new[k-1] = z_new[k] - (lnp[k-1] - lnp[k]) * R_d * tv_k12

    print(z_new.mean('latitude').mean('longitude'))

    return z_new

def interpolate_to_dayofyear(data, day_of_year, method='linear'):
    """
    Interpolates monthly climatology data to daily climatology data.
    
    Parameters:
        data (xarray.Dataset or xarray.DataArray): Monthly climatology dataset or array.
        day_of_year (int): The day of the year to which the data should be interpolated.
        method (str, optional): Interpolation method. Default is 'linear'.
        
    Returns:
        xarray.Dataset or xarray.DataArray: Daily climatology dataset or array interpolated to the specified day of the year.
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

gcs = gcsfs.GCSFileSystem(token='anon')

with gcs.open(f'gs://gresearch/neuralgcm/04_30_2024/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

#we first load the ERA5 data that we are interested in:
print('Opening ERA5 data...')
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xr.open_zarr(gcs.get_mapper(era5_path), chunks=None)

sliced_era5 = (
    full_era5
    [model.input_variables + model.forcing_variables]
    .pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift='24 hours',
    )
    .sel(time=slice(start_time, end_time, data_inner_steps))
    .compute()
)

if flag_pwg:
    print('PGW simulation. Start interpolating ACC signals to ERA5 grid...')

    #sliced era modified
    sliced_era5_pgw = sliced_era5.copy(deep=False)

    ds_vars = {}
    for var in delta_vars.keys():
        var_cmip6 = delta_vars[var] 
        ds_vars[var] = xr.open_dataset(path_delta_mm+f'{var_cmip6}/{var_cmip6}_multimodel_mean_10models_v1.nc')[var_cmip6]
        #ds_vars[var] = xr.open_dataset(path_delta_mm+f'{var_cmip6}/{var_cmip6}_multimodel_mean.nc')[var_cmip6]

        # Check if time is in `cftime.DatetimeNoLeap` format and convert to `datetime64[ns]` if needed
        if isinstance(ds_vars[var]['time'].values[0], cftime.DatetimeNoLeap):
            ds_vars[var]['time'] = ds_vars[var].indexes['time'].to_datetimeindex()
            print(f"Time calendar for {var_cmip6}: {ds_vars[var]['time']}")

    ds_vars = xr.merge(list(ds_vars.values()))

    # Create a reversed dictionary that only contains variables that are in the dataset
    reversed_dict = {v: k for k, v in delta_vars.items() if v in ds_vars.data_vars}
    ds_vars = ds_vars.rename(reversed_dict)

    print(ds_vars)

    #horizontal interpolation to 0,25deg grid
    new_lons=np.arange(0,360,0.25)
    new_lats=np.arange(90,-90.1,-0.25)

    if 'plev' in ds_vars.coords:
        # Convert pressure levels from Pa to hPa
        ds_vars['plev'] = ds_vars['plev'] / 100
        # Optionally, update the units attribute
        ds_vars['plev'].attrs['units'] = 'hPa'
        #rename variable to level
        ds_vars = ds_vars.rename({'plev' : 'level'})

        #levels as in the 37 levels used by NeuralGCM
        levels = sliced_era5_pgw.level.values

        # Interpolate each variable in the dataset to the new pressure levels
        interpolated_ds = xr.Dataset()
        for var_name in ds_vars.data_vars:
            if 'level' in ds_vars[var_name].dims:  # Check if the variable depends on pressure
                interpolated_var = ds_vars[var_name].interp(level=levels,method='linear', kwargs={"fill_value": "extrapolate"})
                interpolated_ds[var_name] = interpolated_var.interpolate_na(dim='level', method='linear')
            interpolated_ds[var_name] = ds_vars[var_name]

        interpolated_ds = interpolated_ds.interpolate_na(dim='level', method='linear')


        interpolated_grid_ds = interpolated_ds.interp(lon=new_lons, lat=new_lats, method='linear', kwargs={"fill_value": "extrapolate"})
        interpolated_grid_ds = interpolated_grid_ds.rename({'lat' : 'latitude', 'lon' : 'longitude'})
    else:
        interpolated_grid_ds = ds_vars.interp(lon=new_lons, lat=new_lats, method='linear', kwargs={"fill_value": "extrapolate"})
        interpolated_grid_ds = interpolated_grid_ds.rename({'lat' : 'latitude', 'lon' : 'longitude'})

    #sliced era modified
    sliced_era5_pgw = sliced_era5.copy(deep=True)

    #let's start applying the delta for each of the vars
    for i,time in enumerate(sliced_era5_pgw.time.values):
        #intepolate to the specific data
        day_of_year = time.astype('datetime64[s]').astype(datetime.datetime).timetuple().tm_yday
        for var in delta_vars.keys():
            if var == 'sea_ice_cover':
                sliced_era5_pgw[var][i] = sliced_era5_pgw[var][i] - 0.01 *  interpolate_to_dayofyear(interpolated_grid_ds[var], day_of_year)
            else:
                sliced_era5_pgw[var][i] = sliced_era5_pgw[var][i] - interpolate_to_dayofyear(interpolated_grid_ds[var], day_of_year)
        
        if adjust_z:
            print('adjsuting geopotential height...')
            z_baro_before = adjust_geopotential(sliced_era5['geopotential'][i], sliced_era5['temperature'][i], sliced_era5['specific_humidity'][i], sp=True, plev_dim_name='level')
            z_baro_after = adjust_geopotential(sliced_era5_pgw['geopotential'][i], sliced_era5_pgw['temperature'][i], sliced_era5_pgw['specific_humidity'][i], sp=True, plev_dim_name='level')
            delta_z = z_baro_before - z_baro_after            
            print(delta_z)
            sliced_era5_pgw['geopotential'][i] =  sliced_era5_pgw['geopotential'][i] - delta_z

    #make sure sea ice cover does not get below 0
    sliced_era5_pgw['sea_ice_cover'] = xr.where(sliced_era5_pgw['sea_ice_cover']<0, 0., sliced_era5_pgw['sea_ice_cover'])


###---------------------- Regriding ------------------------####

print('Regridding ERA5 data to Neural GCM model grid...')

#definition of the grid. Means we can later try usng different input grids
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)

#here we define the regrider
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

#here is regredding the data
if flag_pwg:
    reggrided_era5 = xarray_utils.regrid(sliced_era5_pgw, regridder)
else: 
    reggrided_era5 = xarray_utils.regrid(sliced_era5, regridder)

reggrided_era5 = xarray_utils.fill_nan_with_nearest(reggrided_era5) #filling missing values over land


###---------------------- Simulation ------------------------####

init_times = []
current_time = t0_i
while current_time <= t0_f:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_h)


#dictornary between short and long var names
variable_short_names = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "specific_humidity": "q",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc",
    "sea_surface_temperature": "sst",
    "sea_ice_cover": "siconc"
}

print('Simulation starts...')
for init_time in init_times:
    print(f'init time: {init_time}')
    
    # initialize model state
    inputs = model.inputs_from_xarray(reggrided_era5.sel(time=init_time))
    input_forcings = model.forcings_from_xarray(reggrided_era5.sel(time=init_time))

    # use persistence for forcing variables (SST and sea ice cover)
    all_forcings = model.forcings_from_xarray(reggrided_era5.sel(time=init_time).expand_dims({"time": 1}))

    if 'stochastic' in model_name:
        predictions_member = []
        for member in range(0,n_members):

            ng_key = jax.random.key(seeds[member])  # optional for deterministic models
            initial_state = model.encode(inputs, input_forcings, ng_key)

            # make forecast
            final_state, predictions = model.unroll(
                initial_state,
                all_forcings,
                steps=outer_steps,
                timedelta=timedelta,
                start_with_input=True,
            )
            predictions_member.append(model.data_to_xarray(predictions, times=times))

        predictions_ds = xr.concat(predictions_member, dim='member')

    else:
        
        initial_state = model.encode(inputs, input_forcings)

        # make forecast
        final_state, predictions = model.unroll(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta,
            start_with_input=True,
        )
        predictions_ds = model.data_to_xarray(predictions, times=times)

    # Save the output in a NetCDF file
    print(f'simulation finished... saving data into netcdf')
    yyyymmddhh = init_time.strftime('%Y%m%d%H')
    os.makedirs(save_path+'/'+yyyymmddhh, exist_ok=True)

    for var in predictions_ds:
        if var != 'sim_time':
            short_name = variable_short_names[var]
            if 'stochastic' in model_name:
                ds = predictions_ds[var].to_dataset(name=short_name).transpose('member','time','level','latitude','longitude')
            else:
                ds = predictions_ds[var].to_dataset(name=short_name).transpose('time','level','latitude','longitude')
            ds = ds.assign_coords(time = model.sim_time_to_datetime64(predictions['sim_time']).astype('datetime64[ns]'))
            ds.to_netcdf(save_path+f'{yyyymmddhh}/{short_name}_neuralgcm_{exper}_{yyyymmddhh}.nc')




