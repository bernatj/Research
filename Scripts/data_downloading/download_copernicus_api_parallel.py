import cdsapi
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor

c = cdsapi.Client()

#Configuration of the download
dataset = 'reanalysis-era5-pressure-levels'
#dataset = 'reanalysis-era5-pressure-levels-monthly-means'
dataset_filename = 'era5'
#plevels = '1/2/3/5/7/10/20/30/50/70/100/25/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000'
plevels = ['1','5','10','50','100','150','200','250','300','400','500','600','700','800','850','900','925','950','1000']
#variables = ['geopotential','temperature','u_component_of_wind','v_component_of_wind','relative_humidity','2m_temperature', 'mean_sea_level_pressure', 'total_precipitation', 'surface_pressure','total_column_water_vapour']
#var_filenames = ['z','t','u','v','rh','t2m','mslp','tp','sp','tcwv']
variables = ['temperature','geopotential','u_component_of_wind','v_component_of_wind','relative_humidity','2m_temperature', 'mean_sea_level_pressure', 'total_precipitation', 'surface_pressure','total_column_water_vapour']
var_filenames = ['t','z','u','v','rh','t2m','mslp','tp','sp','tcwv']
years = [str(year) for year in np.arange(2002,2024)]
#freq = 'hourly'
#freq = '6hourly'
freq = 'monthly'
resolution = ['0.25', '0.25'] # cannot be changed for monthly means as far as I know

#list of variables that are in only given at the surface
surface_vars = ['2m_temperature', 'mean_sea_level_pressure', 'total_precipitation', 'surface_pressure', 'total_column_water_vapour']

#this is a dictornary to assign a parameter id to each variable:
#dict_param = {'geopotential' : '129.128', 'z' : '129.128', 'temperature' : '130.128', 't' : '130.128', 'u-wind' : '131.128', 'u' : '131,128',\
#              'v-wind' : '132.128', 'v' : '132.128', 't2m' : '167.128', 't2': '167.128', 'mslp' : '151.128', 'slp' : '151.128'}

def download_data_monthly(year, variable, dataset, plevels, savedir, var_filename, freq):
    try:
        filename = f'{var_filename}-{dataset_filename}-{freq}-{year}-2.nc'
        if os.path.exists(os.path.join(savedir, filename)):
            print(f'{filename} already exists. Skipping download.')
            return
        print(f'Downloading {dataset_filename} {variable} for year {year}...')
        c.retrieve(
            dataset,
            {
                'product_type': 'monthly_averaged_reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'pressure_level': plevels,
                'year': [year],
                'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                'time': ['00:00'],
            },
            os.path.join(savedir, filename)
        )
        print(f'Download complete for {dataset_filename} {variable} for year {year}.')
    except Exception as e:
        print(f"Error during download: {e}")

def download_data_6hourly(year, variable, dataset, plevels, savedir, var_filename, freq):
    try:
        print(f'Downloading {dataset_filename} {variable} for year {year}...')
        c.retrieve(
            dataset,
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'pressure_level': plevels,
                'year': year,
                'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                'day': ['01', '02', '03','04', '05', '06','07', '08', '09',
                    '10', '11', '12','13', '14', '15','16', '17', '18',
                    '19', '20', '21','22', '23', '24','25', '26', '27',
                    '28', '29', '30','31'],
                'time': ['00:00', '06:00', '12:00','18:00'],
                'grid': resolution
            },
            savedir + f'{var_filename}-{dataset_filename}-{freq}-{year}.nc'
        )
        print(f'Download complete for {dataset_filename} {variable} for year {year}.')
    except Exception as e:
        print(f"Error during download: {e}")

def download_data_hourly(year, variable, dataset, plevels, savedir, var_filename, freq):
    try:
        print(f'Downloading {dataset_filename} {variable} for year {year}...')
        for month in range(1,13):
            c.retrieve(
                dataset,
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variable,
                    'pressure_level': plevels,
                    'year': year,
                    'month': [month],
                    'day': ['01', '02', '03','04', '05', '06','07', '08', '09',
                        '10', '11', '12','13', '14', '15','16', '17', '18',
                        '19', '20', '21','22', '23', '24','25', '26', '27',
                        '28', '29', '30','31'],
                    'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                             '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                             '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                             '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                    'grid': resolution
                },
                savedir + f'{var_filename}-{dataset_filename}-{freq}-{year}{str(month).zfill(2)}.nc'
            )
            print(f'Download complete for {dataset_filename} {variable} for year {year}.')
    except Exception as e:
        print(f"Error during download: {e}")

# Use ThreadPoolExecutor for parallel downloads
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []

    for year in years:
        for i, variable in enumerate(variables):
            savedir = f'/pool/usuarios/bernatj/Data/my_data/{dataset_filename}/{freq}/{var_filenames[i]}/'
            
            # Create the directory (and its parent directories if missing)
            os.makedirs(savedir, exist_ok=True)

            if freq == 'monthly':
                if variable in surface_vars:
                    dataset = 'reanalysis-era5-single-levels-monthly-means'
                else:
                    dataset = 'reanalysis-era5-pressure-levels-monthly-means'
                futures.append(executor.submit(
                    download_data_monthly, year, variable, dataset, plevels, savedir, var_filenames[i], freq))
            if freq == '6-hourly':
                if variable in surface_vars:
                    dataset = 'reanalysis-era5-single-levels'
                else:
                    dataset = 'reanalysis-era5-pressure-levels'
                futures.append(executor.submit(
                    download_data_6hourly, year, variable, dataset, plevels, savedir, var_filenames[i], freq))
            if freq == 'hourly':
                if variable in surface_vars:
                    dataset = 'reanalysis-era5-single-levels'
                else:
                    dataset = 'reanalysis-era5-pressure-levels'
                futures.append(executor.submit(
                    download_data_hourly, year, variable, dataset, plevels, savedir, var_filenames[i], freq))

    # Wait for all downloads to finish
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error during download: {e}")
