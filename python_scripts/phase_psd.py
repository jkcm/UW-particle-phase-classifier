#############################
# SOCRATES PHASE PSD BUILDER

#This notebook creates 2DS PSDs partitioned by phase using three methodologies:
#   1. Random forest machine learning approach (*_ml)
#   2. Holroyd classification scheme (*_holroyd)
#   3. Area ratio threshold (*_ar)
# Code developed by Joe Finlon, University of Washington 
# Initial commit by Joe Finlon on 07/24/2020

import xarray as xr
import numpy as np
import pandas as pd

def nan_helper(array_values):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(array_values)
        >>> array_values[nans]= np.interp(x(nans), x(~nans), array_values[~nans])
    """
    return np.isnan(array_values), lambda z: z.nonzero()[0]

def sample_area(): # TODO: finish this routine

def load_nav(navfilename):
    '''
    Load the *.PNI.nc navigation data to get the flight time and true air speed (TAS).
    
    Input:
        navfilename: full path to the navigation data file [str]
    Outputs:
        time: flight time [numpy.datetime64]
        tas: flight true airspeed [m/s]
    '''
    ds = xr.open_dataset(navfilename)
    
    time = ds['Time'].values
    tas = ds['TASX'].values
    
    # Interpolate TAS where NaNs exist
    nans, x= nan_helper(tas)
    tas[nans]= np.interp(x(nans), x(~nans), tas[~nans])
    
    return time, tas

def load_pbp(pbpfilename, Dmin=0.04, Dmax=3.2, iatThresh=1.e-6):
    '''
    Load the particle-by-particle data, only keeping the particles that will be accepted in the PSDs.
    
    Inputs:
        pbpfilename: full path to the PbP data file [str]
        phasefilename: full path to the phase ID data file [str]
        Dmin: smallest maximum dimenison to consider in the PDSs (speeds up processing) [mm]
        Dmax: largest maximum dimenison to consider in the PDSs (speeds up processing) [mm]
        iatThresh: smallest interarrival time to consider in the PDSs (speeds up processing) [s]
    Outputs:
        
    '''
    # First load the particle data
    print('Loading the particle data. This may take a minute...')
    ds = xr.open_dataset(pbpfilename)
    
    date_array = ds['Date'].values
    time_hhmmss = ds['Time'].values # HHMMSS from flight start date
    time_str = [str(date_array[i])[0:4]+'-'+str(date_array[i])[4:6]+'-'+str(date_array[i])[6:]+'T'+str(time_hhmmss[i]).zfill(6)[0:2]+':'+str(time_hhmmss[i]).zfill(6)[2:4]+':'+
        str(time_hhmmss[i]).zfill(6)[4:] for i in range(len(time_hhmmss))]
    time = np.array(time_str, dtype='datetime64[ns]')
    diam_minR = ds['image_diam_minR'].values # diameter of minimum enclosing circle (mm)
    diam_areaR = ds['image_diam_AreaR'].values # area equivalent diameter (mm)
    area = ds['image_area'].values
    ar = area / (np.pi / 4 * diam**2)
    rej = ds['image_auto_reject'].values.astype(int)
    centerin = ds['image_center_in'].values
    tempTime = ds['Time_in_seconds'].values # time in TAS clock cycles
    intArr = np.zeros(len(tempTime))
    intArr[1:] = np.diff(tempTime) # time difference between particles
    # TODO: load the overload flag
    
    # Load the partilce phase data
    print('Loading the phase ID data.')
    ds2 = xr.open_dataset(phasefilename)
    
    phase_ml = ds2['model_pred'].values # TODO: figure out what values indicate [1:liq, 0:ice]
    prob_ml = ds2['model_prob'].values # phase ID confidence [0-1, float]
    phase_holroyd = ds2['UIOOPS'].values # TODO: figure out what values indicate
    phase_ar = ds2['EOL'].values # TODO: figure out what values indicate
    
    # Determine the probability of a particle being liquid/ice based on the ML classification
    prob_ml = ds2['model_prob'].values # probability/confidence in phase classification
    prob_liq_ml = np.zeros(len(prob_ml))
    prob_ice_ml = np.zeros(len(prob_ml))
    prob_liq_ml[phase_ml==1] = prob_ml[phase_ml==1]
    prob_ice_ml[phase_ml==1] = 1. - prob_ml[phase_ml==1]
    prob_ice_ml[phase_ml==0] = prob_ml[phase_ml==0]
    prob_liq_ml[phase_ml==0] = 1. - prob_ml[phase_ml==0]
    
    
    
    print('Removing the rejected particles.')
    good_inds = (diam>=Dmin) & (diam<=Dmax) & (centerin==1) & (ar>=0.2) & (intArr>=iatThresh) & ((rej==48) | (rej==104) | (rej==72)) # TODO: Adopt flags in phase data file
    time = time[good_inds]
    diam_minR = diam[good_inds]
    diam_areaR = diam_areaR[good_inds]
    phase_ml = phase_ml[good_inds]
    prob_ml = prob_ml[good_inds] # TODO: see if this should be diagnostically incorporated at 1 Hz level
    phase_holroyd = phase_holroyd[good_inds]
    phase_ar = phase_ar[good_inds]
    
    return time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar, prob_ice_ml, prob_liq_ml

def make_psd(flight_time, tas, particle_time, diameter_minR, diameter_areaR, phase_ml, phase_holroyd, phase_ar,
             binEdges=None, tres=1):
    '''
    # This subroutine needs to be tested still.
    # TODO: Add any of the phase probability data? e.g., for uncertainty estimates
    '''
    if binEdges is None: # assign default bin edges
        binEdges = [40., 60., 80., 100., 125., 150., 200., 250., 300., 350., 400., 475., 550., 625., 700., 800., 900., 1000.,
                    1200., 1400., 1600., 1800., 2000., 2200., 2400., 2600., 2800., 3000., 3200.] / 1000. # [mm]
    binWidth = np.diff(binEdges) / 10. # [cm]
    
    # Prepare the time loop and allocate PSD arrays
    dur = (flight_time[-1] - flight_time[0]) / np.timedelta64(1, 's') # flight duration [s]
    num_times = int(dur / tres)
    
    count_dmax_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ml = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ar = np.zeros((num_times, len(binEdges)-1))
    
    # TODO: compute the sample area array
    
    time = np.array([], dtype='datetime64[s]')
    for time_ind in range(num_times): # loop through each N-second interval
        curr_time = flight_time[0] + np.timedelta64(int(time_ind*tres), 's')
        time = np.append(time, curr_time)
        
        # TODO: compute sample volume here, accounting for total distance traveled in period and dead time
        
        # Find the particles within the current time interval for each phase classification
        inds_liq_ml = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_ml==1)
        inds_ice_ml = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_ml==0)
        inds_liq_holroyd = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_holroyd==1)
        inds_ice_holroyd = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_holroyd==0)
        inds_liq_ar = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_ar==1)
        inds_ice_ar = (particle_time>=curr_time) & (particle_time<particle_time+np.timedelta64(tres, 's')) & (phase_ar==0)

        if len(inds_liq_ml)==1:
            count_dmax_liq_ml[time_ind, :] = np.histogram(diameter_minR[inds_liq_ml], bins=binEdges)[0]
            count_darea_liq_ml[time_ind, :] = np.histogram(diameter_areaR[inds_liq_ml], bins=binEdges)[0]
        # TODO: Add if statements for the other 5 indices
            
    return (time, binEdges, binWidth, sample_vol,
            count_dmax_liq_ml, count_dmax_ice_ml,
            count_dmax_liq_holroyd, count_dmax_ice_holroyd,
            count_dmax_liq_ar, count_dmax_ice_ar,
            count_darea_liq_ml, count_darea_ice_ml,
            count_darea_liq_holroyd, count_darea_ice_holroyd,
            count_darea_liq_ar, count_darea_ice_ar)