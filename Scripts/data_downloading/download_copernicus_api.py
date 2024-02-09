import cdsapi
import numpy as np
import os

c = cdsapi.Client()

#Configuration of the download
dataset = 'reanalysis-era5-pressure-levels'
#dataset = 'reanalysis-era5-pressure-levels-monthly-means'
dataset_filename = 'era5'
#plevels = '1/2/3/5/7/10/20/30/50/70/100/25/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000'
plevels = ['1','5','10','50','100','150','200','250','300','400','500','600','700','800','850','900','950','1000']
#variables = ['geopotential','temperature','u_component_of_wind','v_component_of_wind','2m_temperature', 'mean_sea_level_pressure', 'total_precipitation']
#var_filenames = ['z','t','u','v','t2m','mslp','tp']
variables = ['temperature','specific_humidity','u_component_of_wind','v_component_of_wind','2m_temperature','mean_sea_level_pressure']
var_filenames = ['t','q','u','v','t2m','mslp']
years = [str(year) for year in np.arange(1940,2023)]
#freq = '6hourly'
freq = 'monthly'
resolution = ['0.25', '0.25']

#list of variables that are in only given at the surface
surface_vars = ['2m_temperature', 'mean_sea_level_pressure', 'total_precipitation']

#this is a dictornary to assign a parameter id to each variable:
#dict_param = {'geopotential' : '129.128', 'z' : '129.128', 'temperature' : '130.128', 't' : '130.128', 'u-wind' : '131.128', 'u' : '131,128',\
#              'v-wind' : '132.128', 'v' : '132.128', 't2m' : '167.128', 't2': '167.128', 'mslp' : '151.128', 'slp' : '151.128'}

if freq == '6hourly':

    for i,variable in enumerate(variables):

        if variable in surface_vars:
            dataset = 'reanalysis-era5-single-levels'
        else:
            dataset = 'reanalysis-era5-pressure-levels'

        savedir=f'/home/bernatj/Data/my_data/{dataset_filename}/{freq}/{var_filenames[i]}/'
        print(savedir)
    
        # Create the directory (and its parent directories if missing)
        os.makedirs(savedir, exist_ok=True)

        for year in years:
            print(f'downloading {dataset_filename} {variable} for year {year}...')

            c.retrieve(
                dataset,
                {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'pressure_level': plevels,
                'year': year,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'grid' : resolution
            },
            savedir+f'{var_filenames[i]}-{dataset_filename}-6h-{year}.nc')

elif freq == 'monthly':
        
    for i,variable in enumerate(variables):

        if variable in surface_vars:
            dataset = 'reanalysis-era5-single-levels-monthly-means'
        else:
            dataset = 'reanalysis-era5-pressure-levels-monthly-means'

        savedir=f'/home/bernatj/Data/my_data/{dataset_filename}/{freq}/{var_filenames[i]}/'
        print(savedir)
    

        savedir=f'/home/bernatj/Data/my_data/{dataset_filename}/{freq}/{var_filenames[i]}/'
        print(savedir)
    
        # Create the directory (and its parent directories if missing)
        os.makedirs(savedir, exist_ok=True)

        for year in years:

            print(f'downloading {dataset_filename} {variable} for year {year}...')

            c.retrieve(
                dataset,
                {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'pressure_level': plevels,
                'year': year,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01',
                ],
                'time': [
                    '00:00',
                ],
                'grid' : resolution
            },
            savedir+f'{var_filenames[i]}-{dataset_filename}-monmean-{year}.nc')