def daily_anom(daily_var,clim=False):
    #daily climatology 
    var_clim = daily_var.groupby('time.dayofyear').mean('time').load()
    #daymean anomalies
    var_anom = daily_var.groupby('time.dayofyear') - var_clim
    if(clim==True):
        return var_anom, var_clim
    else:
        return var_anom
    
def monthly_anom(mon_var,clim=False):
    #monthly climatology 
    var_clim = mon_var.groupby('time.month').mean('time').load()
    #monthly anomalies
    var_anom = mon_var.groupby('time.month') - var_clim
    if(clim==True):
        return var_anom, var_clim
    else:
        return var_anom

def month_to_season(var_mon,sea='DJF'):
    # mask other months with nan
    a =var_mon.where(var_mon['time.season'] == sea)
    # rolling mean -> only Jan is not nan
    # however, we loose Jan/ Feb in the first year and Dec in the last
    b = a.rolling(min_periods=1, center=True, time=3).mean()
    # make annual mean
    var_sea = b.groupby('time.year').mean('time')
    return var_sea
    
def flip_lon_360_2_180(var_360,lon):
    """
    This function shifts the longitude dimension form [0,360]
    to [-180,180]
    """
    var_180 = var_360.assign_coords({"lon": lon.where(lon <= 180, lon - 360)})
    var_180 = var_180.sortby(var_180.lon)
    return var_180 

def flip_longitude_360_2_180(var_360,lon):
    """
    This function shifts the longitude dimension form [0,360]
    to [-180,180]
    """
    var_180 = var_360.assign_coords({"longitude": lon.where(lon <= 180, lon - 360)})
    var_180 = var_180.sortby(var_180.longitude)
    return var_180 

def roll_longitude(var,c_lon):
    """
    This function shifts the longitude dimension form [0,360]
    to any new central longitude
    """
    try:
        #sifft the longitude to a center longitude
        shift= int(len(var.lon)/2) - int(np.array(np.where(var.lon==c_lon)))
        return var.roll(roll_coords=True,lon=shift)
    except:
        shift= int(len(var.longitude)/2) - int(np.array(np.where(var.longitude==c_lon)))
        return var.roll(roll_coords=True,longitude=shift)

def cal_onset_date_1d(x):
    """
    Calculates the onset, end and duration of a binary series of 1s and 0s
    """    
    #difference of shifted arrays
    shift = np.array(x[1:]) - np.array(x[0:len(x)-1]).astype(int)
    onset = np.array(np.where(shift == 1)).astype(int).flatten() #transition form 0 to 1
    end = np.array(np.where(shift == -1)).astype(int).flatten() #transition form 1 to 0
    
    duration = np.array(end - onset)
    
    return onset, end, duration

def wn_filter_3d(y, kmin, kmax):
    """
    ############################################################################################################################
    - Wavenumber restriction: Take a 3-D function and return it restricted to the (kmin,kmax) wavenumber range. 
    ############################################################################################################################
    - INPUT:
    * y: variable with (time,lat,lon) dimensions
    * kmin, kmax: wavenumber range
    ############################################################################################################################
    """
    ffty = fft.fft(y.values,axis=-1)
    mask = np.zeros((ffty.shape))
    mask[:,:,kmin:kmax+1] = 1 # Keep certain freqs only. Values outside the selected range remain zero. 
    mafft = ffty*mask
    fedit = fft.ifft(mafft)
    fedit = 2*fedit.real # Since the ignored negative frequencies would contribute the same as the positive ones.
    #  the DC component of the signal is unique and should not be multiplied by 2:
    if kmin == 0:
        fedit = fedit - ffty.real[0]/(ffty.shape[-1]) # Subtract the pre-edited function's mean. We don't want a double contribution from it in fedit (as was demanded by fedit=2*fedit.real).
    elif kmin > 0:
        fedit = fedit + ffty.real[0]/(ffty.shape[-1])  # Add the pre-edited function's mean. The zero frequency in FFT should never be left out when editing a function. 
    fedit = y.copy(data=fedit,deep=False) 
    return fedit