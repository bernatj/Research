#######TIME FILTERING FUNCTIONS#################
from scipy.signal import butter, lfilter, filtfilt, freqz

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = data.copy(data=y,deep=False) 
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = data.copy(data=y,deep=False) 
    return y
########################

def Maximum_eady_growth_rate(U,T,Z):
    """
    Maximum Eady Growth Rate (EGR) diagnostic 
    Bernat Jimenez-Esteve, ETH Zurich, June 2021
    
    This computes the maximum Eady Growth rate (EGR, Eady (1949, Tellus)), a measure of baroclinic instability and potential growth of baroclinic eddies. For reference equation 1 in Jimenez-Esteve and Domeisen (2018, JClim).
    
    eady_growth_rate = 0.3098*g*abs(f)*abs(du/dz)/brunt_vaisala_atm   
    
    Input variables:
    - U (zonal wind in m/s)
    - Z (geopotential height in m)
    - T (temperature in K)
    
    Output:
    - Eady Growth Rate (EGR)
    
    """
    
    #---Constants & quantities needed-------
    a     = 6.37122e06       # radius of the earth (m)
    PI    = 3.14159          # 3.14159265358979
    omega = 7.2921e-5        # Earth angular velocity (s-1)
    g     =  9.806           # Earth gravity (m/s2)
    
    #Some useful parameters in the equation
    phi   = T.latitude*PI/180.0   # latitude in radians
    fcor = 2*omega*np.sin(phi)    #coriolis parameter
    
    #compute potencial temperature THETA (degK)
    from metpy.calc import potential_temperature
    from metpy.calc import brunt_vaisala_frequency
    from metpy.units import units

    #calculate potential temperature
    theta = potential_temperature(T.level, T).transpose('time','level','latitude','longitude')

    #The buoyancy frequency
    brunt_vaisala_freq = brunt_vaisala_frequency(Z, theta, vertical_dim=1).metpy.dequantify()
    
    #---vertical wind shear
    from metpy.calc import first_derivative
    dUdz = theta.copy(data=first_derivative(U, x=Z, axis=1).magnitude, deep=True)
                      
    #--calculate the Eady-growth rate
    EGR = 0.3098*g*np.abs(fcor)*np.abs(dUdz)/(brunt_vaisala_freq)
    seconds_to_days = 3600*24 
    EGR = EGR*seconds_to_days
    EGR = EGR.assign_attrs({"var_name" :"Eady growth rate", "units" : "1/day"})
                      
    return EGR



def Transient_E_vector(U,V,T):
    """
    Transient 3D-WAF  diagnostic (E-vector)
    Bernat Jimenez-Esteve, ETH Zurich, June 2021 
    For reference see equation 18 in Trenberth (1986).
    This is a simpliciation of the Plumb (1986) transient wave activity flux used in Jimenez-Esteve and Domeisen (2018, Jclim), but a good anough aproximation when the climatology of PV is almost zonal. The same application has been used in Schem et al. (2018, JAS). The vector is similar to the Hoskins E-vector (Hoskins, 1983), which seems to be good enough for most of the applications.
 
    - We assume a zonal mean background flow. 
    - This version implemented here works only for daily mean data (a time filter has to be applied). 6-hourly data is possible
    with some adaptations.
    - It uses xarray to keep track of coordinates. Please adjust you coordinate names accordingly.
    - lowpass (T>30 days) and highpass filter (T<8 days) is applied to compute overbars and primes in the equation.
    - Beaware that because of the timefiltering values at the start and end of your window are not reliable
 
    Input variables:
    - U (zonal wind in m/s)
    - V (meridional wind in m/s)
    - T (temperature in K)
 
    Output:
    - $E_x(x,y,z)$ (meridional component of the WAF flux)
    - $E_y(x,y,z)$ (meridional component of the WAF flux)
    - $E_z(x,y,z)$ (vertical component of the WAF flux)
    - 3D-divF (divergence of the flux, denotes acceleration/deceleration of the zonal wind)
    """
        
    #---Constants & quantities needed-------
    a     = 6.37122e06       # radius of the earth (m)
    PI    = 3.14159          # 3.14159265358979
    omega = 7.2921e-5        # Earth angular velocity (s-1)
    H     = 7000             # vertical lenght scale (m)  

    #Some useful parameters in the equation
    phi   = T.latitude*PI/180.0     # latitude in radians
    lamb  = T.longitude*PI/180.0            # longuitude in radians
    cphi  = np.cos(phi)
    fcor = 2*omega*np.sin(phi)    #coriolis parameter

    #compute potencial temperature THETA (degK)
    from metpy.calc import potential_temperature
    from metpy.units import units
    theta = potential_temperature(T.level, T).transpose('time','level','latitude','longitude').metpy.dequantify()

    #---low pass filter the data

    # Filter requirements.
    order = 5
    fs = 1.0      # sample rate, 1/d
    
    # desired cutoff frequency of the filter, 1/d
    lowcut = 1/30
    highcut = 1/8

    #--Filter the data, and plot both the original and filtered signals.
    v_prime = butter_highpass_filter(V, highcut, fs, order)
    u_prime = butter_highpass_filter(U, highcut, fs, order)
    theta_prime = butter_highpass_filter(theta, highcut, fs, order)

    #--compute variances and overbars
    v2 = butter_lowpass_filter(v_prime*v_prime, lowcut, fs, order)
    u2 = butter_lowpass_filter(u_prime*u_prime, lowcut, fs, order)

    #meriodonal momentum flux (proportinal to the poleard componet)
    uv = butter_lowpass_filter(u_prime*v_prime, lowcut, fs, order)

    #poleward heat flux (propotional to the upward component)
    vtheta =  butter_lowpass_filter(v_prime*theta_prime, lowcut, fs, order)

    #--vertical stability

    #log-pressure coordinate (for the evrtical componet)
    if (max(T.level)<2000):               # must be hPa
        plev = T.level*100
    PS      = 100000                            # "Pa"
    Z       = -H*np.log(plev/PS)             # m
    
    #---d(theta)/dz 
    theta_lowpass = butter_lowpass_filter(theta, lowcut, fs, order)
    theta_lowpass = theta_lowpass.assign_coords({"z" : Z})
    dthetaz = theta_lowpass.differentiate(coord="z", edge_order=2)

    #components of the flux in spherical coordinates
    Ex = cphi*0.5*(v2 - u2)
    Ey = cphi*(-1.0)*uv
    Ez = cphi*fcor*vtheta/(dthetaz)

    Ex = Ex.assign_attrs({"var_name" :"zonal component of the transient WAF", "units" : "m2/s2"})
    Ey = Ey.assign_attrs({"var_name" :"meridional component of the transient WAF", "units" : "m2/s2"})
    Ez = Ez.assign_attrs({"var_name" :"vertical component of the transient WAF", "units" : "m2/s2"})

    #---3D-Flux Divergence (in spherical coordinates)
    Ex = Ex.assign_coords({"lamb" : lamb})
    Exx = (1/(a*cphi))*Ex.differentiate(coord="lamb",edge_order=2) 
    Ey = Ey.assign_coords({"phi" : phi})
    Eyy = (1/(a*cphi))*(Ey*cphi).differentiate(coord="phi",edge_order=2) 
    Ezz = Ez.differentiate(coord="z",edge_order=2) 

    divE  = Exx + Eyy + Ezz

    Exx = Ex.assign_attrs({"var_name" :"zonal divergence of the transient WAF", "units" : "m/s2"})
    Eyy = Ey.assign_attrs({"var_name" :"meridional divergence of the transient WAF", "units" : "m/s2"})
    Ezz = Ez.assign_attrs({"var_name" :"vertical divergence of the transient WAF", "units" : "m/s2"})
    divE  = divE.assign_attrs({"var_name" :"transient activity flux divergence", "units" : "m/s2"})
    
    return Ex,Ey,Ez,divE
    

def Stationary_Plumb_WAF(PHI,T):
    """
    Stationary Plumb's 3D-flux flux diagnostic
    Bernat Jimenez-Esteve, ETH Zurich, June 2021
    For reference see Plumb (1985) (equation 5.1)
    
    - This version implemented here works for daily and monthly data (for daily ideally you should lowpass filter anomalies first)
    - It uses xarray to keep track of coordinates. Please adjust you coordinate names accordingly.
    
    Input variables:
    - PHI (geopotential m2/s2)
    - T (temperature in K)
    
    Output:
    - F_x(x,y,z) (meridional component of the WAF flux)
    - F_y(x,y,z)(meridional component of the WAF flux)
    - F_z(x,y,z)$ (vertical component of the WAF flux)
    - 3D-divF (divergence of the flux, denotes acceleration/deceleration of the zonal wind)
    """
    
    #---Constants & quantities needed-------
    a     = 6.37122e06       # radius of the earth (m)
    PI    = 3.14159          # 3.14159265358979
    omega = 7.2921e-5        # Earth angular velocity (s-1)
    g     =  9.806           # Earth gravity (m/s2)
    H     = 7000             # vertical lenght scale (m)  
    k     =  0.286           # R/cp
    Ra    = 287              # dry air gas constant

    #Some useful parameters in the equation
    phi   = T.latitude*PI/180.0     # latitude in radians
    lamb  = T.longitude*PI/180.0            # longuitude in radians
    cphi  = np.cos(phi)
    fcor = 2*omega*np.sin(phi)    #coriolis parameter

    #log-pressure coordinate
    if (max(T.level)<2000):               # must be hPa
        plev = T.level*100

    PS      = 100000                            # "Pa"
    Z       = -7000*np.log(plev/PS)             # m
    Z = Z.assign_attrs({"units" : "m"})
    rho0    = PS/(g*H)*np.exp(-Z/H)             #density (kg/m3)

    #compute potencial temperature THETA (degK)
    from metpy.calc import potential_temperature
    from metpy.calc import brunt_vaisala_frequency
    from metpy.units import units

    #calculate potential temperature
    theta = potential_temperature(T.level, T).transpose('time','level','latitude','longitude')

    #The buoyancy frequency squared (N^2)
    N2= brunt_vaisala_frequency_squared(Z, theta, vertical_dim=1)

    #---compute QG stream-function
    PSI = PHI/fcor

    #---Deviation from the zonal mean
    PSI_za = PSI - PSI.mean('longitude')           # (time,lev,lat,lon)
    PSI_za.assign_attrs({"var_name":"QG stream function zonal anomaly"})  

    #---Derivatives of PSI
    PSI_za = PSI_za.assign_coords({"z": Z, "phi" : phi, "lamb" : lamb})  #latitude and longitude in radians

    PSIx = PSI_za.differentiate(coord="lamb",edge_order=2)
    PSIy = PSI_za.differentiate(coord="phi",edge_order=2)
    PSIz = PSI_za.differentiate(coord="z",edge_order=2)

    PSIxx = PSIx.differentiate(coord="lamb",edge_order=2)
    PSIxy = PSIx.differentiate(coord="phi",edge_order=2)
    PSIxz = PSIx.differentiate(coord="z",edge_order=2)

    #---comon factor
    factor = (plev/PS)

    #---Plumb flux compoments  
    Fx = factor/(2*(a**2)*cphi)*(PSIx*PSIx - PSI_za*PSIxx)
    Fy = factor/(2*a**2)*(PSIx*PSIy - PSI_za*PSIxy)
    Fz = factor*((fcor**2)/(2*a*N2.metpy.dequantify()))*(PSIx*PSIz - PSI_za*PSIxz)


    Fx = Fx.assign_attrs({"var_name" :"zonal component of the Plumb's WAF", "units" : "m2/s2"})
    Fy = Fy.assign_attrs({"var_name" :"meridional component of the Plumb's WAF", "units" : "m2/s2"})
    Fz = Fz.assign_attrs({"var_name" :"vertical component of the Plumb's WAF", "units" : "m2/s2"})

    #---3D-Flux Divergence
    Fxx = (1/(a*cphi))*Fx.differentiate(coord="lamb",edge_order=2) 
    Fyy = (1/(a*cphi))*(Fy*cphi).differentiate(coord="phi",edge_order=2) 
    Fzz = Fz.differentiate(coord="z",edge_order=2) 

    divF  = Fxx + Fyy + Fzz

    Fxx = Fx.assign_attrs({"var_name" :"zonal divergence of the Plumb's WAF", "units" : "m/s2"})
    Fyy = Fy.assign_attrs({"var_name" :"meridional divergence of the Plumb's WAF", "units" : "m/s2"})
    Fzz = Fz.assign_attrs({"var_name" :"vertical divergence of the Plumb's WAF", "units" : "m/s2"})
    divF  = divF.assign_attrs({"var_name" :"Plumb's activity flux divergence", "units" : "m/s2"})
    
    return Fx,Fy,Fz,divF



def EP_flux_QGaprox_2D(U,V,T):
    """
    Eliassen-Palm flux diagnostic (QG aproximation)
    Bernat Jimenez-Esteve, ETH Zurich, June 2021
    
    - This is a zonal mean diagnostic    
    - This version implemented here works for daily and monthly data.
    - It uses xarray to keep track of coordinates. Please adjust you coordinate names accordingly
    
    Input variables:
    - U (zonal wind in m/s)
    - V (meridional wind in m/s)
    - T (temperature in K)
    
    Output:
    - Fphi (meridional component of the EP-flux)
    - Fz (vertical component of the EP-flux)
    - divF (divergence of the flux, denotes acceleration.deceleration of the zonal wind)
    """
    
    #---Constants & quantities needed-------
    a     = 6.37122e06       # radius of the earth (m)
    PI    = 3.14159          # 3.14159265358979
    omega = 7.2921e-5        # Earth angular velocity (s-1)
    g     =  9.806           # Earth gravity (m/s2)
    H     = 7000             # vertical lenght scale (m)
    
    #log-pressure coordinate
    if (max(T.level)<2000):               # must be hPa
        plev = T.level*100
    PS      = 100000                            # "Pa"
    Z       = -7000*np.log(plev/PS)             # m
    rho0    = PS/(g*H)*np.exp(-Z/H)             #density (kg/m3)
    
    #compute potencial temperature THETA (degK)
    from metpy.calc import potential_temperature
    from metpy.units import units
    theta = potential_temperature(T.level, T)
    
    #zonal means
    theta_zm = theta.mean('longitude')
    u_zm = U.mean('longitude')
    v_zm = V.mean('longitude')

    #zonal anomalies
    theta_za = theta - theta_zm
    u_za = U - u_zm
    v_za = V - v_zm
    
    #---d(theta_zm)/dz 
    theta_zm = theta_zm.assign_coords({"z" : Z})
    dthetaz = theta_zm.metpy.dequantify().differentiate(coord="z", edge_order=2)
    
    #---Anomaly products
    UV  = u_za*v_za                       
    VTHETA  = v_za*theta_zm 

    #zonal means of the products
    UVzm    = UV.mean('longitude') 
    VTHETAzm   = VTHETA.mean('longitude') 
    
    phi   = U.latitude*PI/180.0     # latitude in radians
    f     = 2*omega*np.sin(phi)        # coriolis parameter
    cphi  = np.cos(phi)
    acphi = a*cphi

    #---EP flux meridional and vertical compoments 
    Fphi = -rho0*acphi*UVzm.metpy.dequantify()
    Fz = f*acphi*rho0*(VTHETAzm/dthetaz).metpy.dequantify()

    Fphi = Fphi.assign_attrs({"var_name" :"meridional component of EP flux", "units" : "kg/s2"})
    Fz = Fz.assign_attrs({"var_name" :"vertical component of EP flux", "units" : "kg/s2"})
    
    #---EP Flux Divergence
    Fphi = Fphi.assign_coords({"phi" : phi})
    EPdiv_phi = (Fphi*cphi).metpy.dequantify().differentiate(coord="phi",edge_order=2)  # Meridional divergence
    EPdiv_phi  = EPdiv_phi/acphi
    EPdiv_z  = Fz.metpy.dequantify().differentiate(coord="z",edge_order=2)     #Vertical divergence
    EPdiv  = (EPdiv_phi + EPdiv_z)
    EPdiv = EPdiv/rho0/acphi*3600*24 # to m/s/day  
    EPdiv = EPdiv.assign_attrs({"var_name" :"EP flux divergence", "units" : "m/s/day"})

    EPdiv_phi  = EPdiv_phi/rho0/acphi*3600*24
    EPdiv_z  = EPdiv_z/rho0/acphi*3600*24    # Vertical divergence 
    
    return Fphi, Fz, EPdiv
    

    
    

    