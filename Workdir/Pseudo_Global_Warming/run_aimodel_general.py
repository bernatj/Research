import sys
sys.path.append('/home/bernatj/ddf')

import numpy as np
from pathlib import Path
import xarray as xr
import os
import datetime
from earth2mip import inference_ensemble, registry
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds
from earth2mip.inference_ensemble import run_basic_inference
from ddf._src.data.local.xrda import LocalDataSourceXArray

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


##############################################################################
#                               config                                       #
##############################################################################
model_name =  "pangu" # 'graphcast' # "fcnv2_sm" # "pangu_6" # 
model = f"e2mip://{model_name}"
device = "cuda:0"

file_format = 'netcdf'
outputdir="/home/bernatj/Data/ai-forecasts/fcst/"
ai_model='pangu' #'fcnv2' # 'pangu' #
experiment= '_PGW_multimodel' #  '' # '_PGW_multimodel' # 

#we want to run our model for different dates
init_times = []

first_init_time = datetime.datetime(2021, 6, 18, 0)
end_init_time = datetime.datetime(2021, 7, 3, 18)
delta_hours = 6

current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)

# number of forecast steps
num_steps = 4 * 10 # 6h intervals
vars_to_save = ['t2m','t850','z500','u850','v850','msl','q850','q1000']
#vars_to_save = ['u300', 'v300', 't300']

######################################################################################

print(f'loading AI-model: {model}')
time_loop  = get_model(
    model=model,
    device=device,
)
channel_names = time_loop.in_channel_names
print(f'model loaded')


def do_forecast(channel_names, file_paths,  pressure_name='isobaricInhPa', engine='netcdf4'):

    print('read input data...')    
    #get the init data
    data_source_xr = LocalDataSourceXArray(
    channel_names=channel_names,
    file_paths=file_paths,
    pressure_name=pressure_name, 
    name_convention="short_name",
    engine=engine, 
    )

    print(data_source_xr)

    print('run the forecast...') 
    #run the model
    forecast = run_basic_inference(
    time_loop, 
    n=num_steps, 
    data_source=data_source_xr, 
    time=t0
    )

    return forecast

if file_format == 'grib':
    ending='grib'
    engine='cfgrib'
elif file_format == 'netcdf':
    ending='nc'
    engine='netcdf4'

#integration
print(f'starting model runs...')
for t0 in init_times:
    yyyymmddhh = t0.strftime('%Y%m%d%H')
    print(f'date {yyyymmddhh} ...')
    
    file_paths = [  
    f"/home/bernatj/Data/ai-forecasts/input/{file_format}/{yyyymmddhh}/{ai_model}_sl{experiment}_{yyyymmddhh}.{ending}",
    f"/home/bernatj/Data/ai-forecasts/input/{file_format}/{yyyymmddhh}/{ai_model}_pl{experiment}_{yyyymmddhh}.{ending}"
    #f"/home/bernatj/Data/ai-forecasts/input/ifs-ana/{file_format}/{yyyymmddhh}/pangu_fcnv2_sl{experiment}_{yyyymmddhh}.{ending}",
    #f"/home/bernatj/Data/ai-forecasts/input/ifs-ana/{file_format}/{yyyymmddhh}/pangu_fcnv2_pl{experiment}_{yyyymmddhh}.{ending}"
    ]

    #run one forecast
    forecast = do_forecast(channel_names, file_paths, pressure_name='isobaricInhPa', engine=engine)

    #store the data
    print(f'saving data into file ...')
    
    os.makedirs(outputdir+'/'+yyyymmddhh, exist_ok=True)
    for var in vars_to_save:
        forecast.sel(channel=var).squeeze().drop_vars('channel').to_dataset(name=var).to_netcdf(
            outputdir + f'{yyyymmddhh}/{var}_{ai_model}{experiment}_{yyyymmddhh}.nc',
            mode='w'  # Explicitly specify overwriting mode
        )

    print(f'finished forecast for init {t0}')

