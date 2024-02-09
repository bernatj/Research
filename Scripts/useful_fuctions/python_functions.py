import numpy as np
import xarray as xr

def interpolate_and_concat_files_while_reading(file_list, var_name, plev=50000, lat_S=25, lat_N=72, dx=1):
    '''
    Opens one file at a time, interpolates and copies the input variable in order to avoid loading too many files.
    
    Parameters:
    -----------
    file_list : list
        List of files to be read in and concatenated.
    var_name : str
        Name of the variable to be read in and interpolated.
    plev : int, optional
        Pressure level in pascals. Defaults to 50000.
    lat_S : int, optional
        Southern latitude boundary. Defaults to 25.
    lat_N : int, optional
        Northern latitude boundary. Defaults to 72.
    dx : int, optional
        Spacing of grid points in degrees. Defaults to 1.
        
    Returns:
    --------
    var : xarray DataArray
        Interpolated and concatenated variable data over time.
    '''
    # Define the grid to interpolate to:
    lon = np.arange(-180, 180+dx, dx)
    lat = np.arange(lat_S, lat_N, dx)
    
    # Loop through each file and interpolate the variable
    for i, file in enumerate(file_list):
        print(f'Reading in {file}')
        
        # Read in the variable and select the specified pressure level and latitude range
        var_new = xr.open_dataset(file)[var_name].sel(latitude=slice(lat_N, lat_S))

        
        # Interpolate the variable onto the specified grid
        var_new = var_new.interp(latitude=lat, longitude=lon)
        
        # Concatenate the interpolated variable with the existing data
        if i == 0:
            var = var_new
        else:
            var = xr.concat([var, var_new], dim='time')

    return var

def plot_dataarray(da, significance=False, p_values=None, cmap='coolwarm', title=None, figsize=(10, 6), vmin=None, vmax=None, cbar_kwargs=None, 
                    add_land=True, filled_land=True, add_gridlines=True, region=None, projection =''):
    """
    Plots an xarray DataArray with optional coastlines, gridlines, and a colorbar.

    Parameters
    ----------
    da : xarray.DataArray
        The data to plot.
    significance : bool, optional
        If True, indicates significance levels on the plot based on p_values (default is False).
    p_values : array-like, optional
        Array of p-values corresponding to the significance levels (default is None).
    cmap : str, optional
        The colormap to use (default is 'coolwarm').
    title : str, optional
        The title of the plot.
    figsize : tuple, optional
        The size of the figure in inches (default is (10, 6)).
    vmin : float, optional
        The minimum value of the color scale. If None, the minimum value of the data is used.
    vmax : float, optional
        The maximum value of the color scale. If None, the maximum value of the data is used.
    cbar_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the colorbar constructor.
    add_land : bool, optional
        Whether to add land to the plot (default is True).
    filled_land : bool, optional
        Whether to fill the land with a color (default is True).
    add_gridlines : bool, optional
        Whether to add gridlines to the plot (default is True).
    region : tuple, optional
        A tuple (lon1, lon2, lat1, lat2) defining the region to plot.
    projection : str, optional
        The projection to use. Options are 'northpole' or 'southpole' for polar projections. Default is PlateCarree.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        The plot axis.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Change latitude and longitude labels if not present
    if 'lat' not in da.dims and 'lon' not in da.dims:
        # Rename 'latitude' and 'longitude' dimensions to 'lat' and 'lon'
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Check if the latitude dimension is sorted
    if not da.coords['lat'].to_index().is_monotonic_increasing:
        # If not sorted, sort the latitude dimension
        da = da.sortby('lat')

    # Select region if specified
    if region is not None:
        lon1, lon2, lat1, lat2 = region
        da = da.sel(lon=slice(lon1, lon2), lat=slice(lat1, lat2))

    # Create the figure and axis with the specified projection
    if projection == 'northpole':
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0)})
        # Set the extent of the map
        ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
    elif projection == 'southpole':
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0.0)})
        # Set the extent of the map
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    elif projection == 'Robinson':
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
        ax.set_extent([-180, 180, -87.5, 87.5], ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot significance levels if specified
    if significance:
        significant = p_values > 0.05
        # Create a boolean array with True where the p-value is less than 0.05
        ax.contourf(da.lon, da.lat, significant, levels=[0, 0.5, 1], hatches=['', '////'], alpha=0.0, colors=['0.5'],
                transform=ccrs.PlateCarree())

    # Plot the data
    im = ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

    # Add coastlines
    if add_land:
        ax.coastlines()
        # Add a filled continent using NaturalEarthFeature
        if filled_land:
            ax.add_feature(cfeature.LAND, facecolor='#AAAAAA')

    # Add gridlines
    if add_gridlines:
        ax.gridlines(draw_labels=True, linewidth=1, color='gray', linestyle='--')

    # Set the face color for missing values
    ax.set_facecolor('lightgray')

    # Add a colorbar
    cbar_kwargs = cbar_kwargs or {}
    cbar = plt.colorbar(im, ax=ax, **cbar_kwargs)

    # Set the title
    ax.set_title(title)

    # Set the x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set the label bar size
    if 'fraction' not in cbar_kwargs and 'shrink' not in cbar_kwargs:
        cbar.ax.set_ylabel(da.units)

    return fig, ax


def fix_data_format(da, region=None, flip_lon = True):
    """
    A function to standardize the format of an xarray DataArray by ensuring that latitude and longitude dimensions are
    appropriately named ('lat' and 'lon'), sorted in ascending order, and optionally selecting a specified region.

    Parameters:
    - da (xarray.DataArray): The input DataArray containing geospatial data.
    - region (tuple or None, optional): A tuple specifying the bounding box region to select, in the format (lon1, lon2, lat1, lat2).
      If None, no region selection is performed. Default is None.

    Returns:
    - da (xarray.DataArray): The processed DataArray with standardized format and, if specified, a selected region.
    """
    # Rename 'latitude' and 'longitude' dimensions to 'lat' and 'lon' if not already present
    if 'lat' not in da.dims and 'lon' not in da.dims:
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Check if the 'lat' dimension is sorted
    if not da.coords['lat'].to_index().is_monotonic_increasing:
        # If not sorted, sort the 'lat' dimension
        da = da.sortby('lat')

    # Select a specified region if provided
    if region is not None:
        lon1, lon2, lat1, lat2 = region
        da = da.sel(lon=slice(lon1, lon2), lat=slice(lat1, lat2))

    #flip longitude dimensions from [0,360] to [-180,180]
    if flip_lon:
        lon = da.lon
        da = da.assign_coords({"lon": lon.where(lon <= 180, lon - 360)})
        da = da.sortby(da.lon)
    return da


def flip_lon_360_2_180(var_360,lon):
    """
    This function shifts the longitude dimension form [0,360]
    to [-180,180]
    """
    try:
        var_180 = var_360.assign_coords({"lon": lon.where(lon <= 180, lon - 360)})
        var_180 = var_180.sortby(var_180.lon)
    except:
        var_180 = var_360.assign_coords({"longitude": lon.where(lon <= 180, lon - 360)})
        var_180 = var_180.sortby(var_180.longitude)

    return var_180 

def regrid_to_regular_grid(data_array, new_lon, new_lat):
    """
    Regrid xarray data to a new regular grid defined by longitude and latitude arrays.
    
    Parameters:
        data_array (xarray.DataArray): The input data array with longitude and latitude dimensions.
        new_lon (numpy.ndarray): 1-D array of new longitudes for the regular grid.
        new_lat (numpy.ndarray): 1-D array of new latitudes for the regular grid.
    
    Returns:
        xarray.DataArray: Regridded data array on the new regular grid.
    """
    regridded_data = data_array.interp(lon=new_lon, lat=new_lat, method='linear')
    return regridded_data



def interpolate_to_regular_grid(data_array, new_lon, new_lat, method='linear'):
    """
    Interpolate xarray data to a new regular grid defined by longitude and latitude arrays.
    
    Parameters:
        data_array (xarray.DataArray): The input data array with longitude and latitude dimensions.
        new_lon (numpy.ndarray): 1-D array of new longitudes for the regular grid.
        new_lat (numpy.ndarray): 1-D array of new latitudes for the regular grid.
        method (str, optional): Interpolation method. Can be 'linear', 'nearest', or 'cubic'.
                                Defaults to 'linear'.
    
    Returns:
        xarray.DataArray: Interpolated data array on the new regular grid.
    """
    # Create a meshgrid of new_lon and new_lat
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)
    
    # Reshape the new lon and lat grids to 1-D arrays
    new_lon_flat = new_lon_grid.flatten()
    new_lat_flat = new_lat_grid.flatten()
    
    # Flatten the input data array's lon and lat values
    lon_flat = data_array.coords[data_array.dims[-1]].values.flatten()
    lat_flat = data_array.coords[data_array.dims[-2]].values.flatten()
    
    # Flatten the data values
    data_flat = data_array.values.flatten()
    
    # Perform the interpolation
    interpolated_data_flat = griddata((lon_flat, lat_flat), data_flat, (new_lon_flat, new_lat_flat), method=method)
    
    # Reshape the interpolated data back to a grid
    interpolated_data = interpolated_data_flat.reshape(new_lon_grid.shape)
    
    # Create a new DataArray with the interpolated data and coordinates
    interpolated_data_array = xr.DataArray(interpolated_data, coords={'lon': new_lon, 'lat': new_lat}, dims=('lat', 'lon'))
    
    return interpolated_data_array
