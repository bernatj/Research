
#!/usr/bin/env python
from ecmwfapi import *
import numpy as np
import os
import datetime

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

server = ECMWFService("mars")

#Configuration of the download

modelnames = ['fcnv2','pangu'] 
#modelname = 'pangu'

# Initialize dictionaries to store model specific variables
model_variables = {
    'fcnv2': {
        'plevels': ['50', '100', '150', '200', '250', '300', '400','500', '600', '700', '850', '925', '1000'],
        'variables_pl': ['129', '130', '131', '132', '157'], #z, t, u, v, r
        'variables_sf': ['134', '137', '151', '165', '166', '167', '228246', '228247'] 
    },
    'pangu': {
        'plevels': ['50', '100', '150', '200', '250', '300', '400','500', '600', '700', '850', '925', '1000'],
        'variables_pl': ['129', '130', '131', '132', '133'], #z, t, u, v, q
        'variables_sf': ['151', '165', '166', '167'] #msl, u10m, v10m, t2m
    }
}

# Initialize empty sets for unique plevels, variables_pl, and variables_sf
plevels, variables_pl, variables_sf = set(), set(), set()

# Add model specific variables to the sets
for model in modelnames:
    plevels.update(model_variables[model]['plevels'])
    variables_pl.update(model_variables[model]['variables_pl'])
    variables_sf.update(model_variables[model]['variables_sf'])

# Convert sets back to lists
plevels, variables_pl, variables_sf = list(plevels), list(variables_pl), list(variables_sf)

resolution = ['0.25', '0.25'] # cannot be changed for monthly means as far as I know

first_init_time = datetime.datetime(2018,9,5,00)
end_init_time = datetime.datetime(2018,9,15,18)
delta_hours = 6

init_times = []
current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)

#add a second period
first_init_time = datetime.datetime(2023,10,25,0)
end_init_time = datetime.datetime(2023,11,5,18)
current_time = first_init_time
while current_time <= end_init_time:
    init_times.append(current_time)
    current_time += datetime.timedelta(hours=delta_hours)


def download_init(yyyymmddhh, filename, leveltype, variables, plevels, savedir):
    try:
        if os.path.exists(os.path.join(savedir, filename)):
            print(f'{filename} already exists. Skipping download.')
            return
        print(f'Downloading init {yyyymmddhh} for {leveltype}')

        year = yyyymmddhh[0:4]
        month = yyyymmddhh[4:6]
        day = yyyymmddhh[6:8]
        hour = yyyymmddhh[8:10]
        server.execute(
                        {
                        "class" : "od",
                        "date" : f"{year}-{month}-{day}",
                        "expver" : "1",
                        "levelist" : plevels,
                        "levtype": leveltype,
                        "param" :  variables,
                        "stream" : "oper",
                        "time" : f"{hour}:00:00",
                        "grid": "0.25/0.25",
                        "type" : "an",
                        "format" : "netcdf"
                        },
                        os.path.join(savedir, filename))

        print(f'Download complete for .')
    except Exception as e:
        print(f"Error during download: {e}")

# Use ThreadPoolExecutor for parallel downloads
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for init in init_times:
        print(init)

        yyyymmddhh = init.strftime('%Y%m%d%H')
        savedir = f'/pool/usuarios/bernatj/Data/ai-forecasts/input/ifs-ana/{yyyymmddhh}'
        # Create the directory (and its parent directories if missing)
        os.makedirs(savedir, exist_ok=True)

        filename = f'pangu_fcnv2_sf_{yyyymmddhh}.nc'
        futures.append(executor.submit(download_init,yyyymmddhh, filename, 'sfc', variables_sf, plevels, savedir))
        filename = f'pangu_fcnv2_pl_{yyyymmddhh}.nc'
        futures.append(executor.submit(download_init,yyyymmddhh, filename, 'pl', variables_pl, plevels, savedir))

    # Wait for all downloads to finish
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error during download: {e}")
