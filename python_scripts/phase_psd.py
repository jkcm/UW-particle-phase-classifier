#############################
# SOCRATES PHASE PSD BUILDER
#############################
# This script creates 2DS PSDs, partitioned by phase, using three methodologies:
#   1. Random forest machine learning approach (*_ml)
#   2. Holroyd classification scheme (*_holroyd)
#   3. Area ratio threshold (*_ar)
# Code developed by Joe Finlon, University of Washington 
# Initial commit by Joe Finlon on 08/13/2020
# 
# Example for generating the PSDs:
#   flt_string = 'rf01'
#   navfilename = glob.glob('/home/disk/eos9/jfinlon/socrates/' + flt_string + '/*.PNI.nc')[0]
#   pbpfilename = '/home/disk/eos9/jfinlon/socrates/' + flt_string + '/pbp.' + flt_string + '.2DS.H.nc'
#   phasefilename = '/home/disk/eos9/jkcm/Data/particle/classified/' + 'UW_particle_classifications.' + flt_string + '.nc'
#   [flt_time, flt_tas] = load_nav(navfilename)
#   [time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar,
#    prob_ice_ml, prob_liq_ml, time_all, intArr, ovrld_flag] = load_pbp(
#       pbpfilename, phasefilename, Dmin=0.05, Dmax=3.2, iatThresh=1.e-6)
#   psd = make_psd(flt_time, flt_tas, time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar,
#       time_all, intArr, ovrld_flag, binEdges=None, tres=1)

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

def sample_area(binmid): # TODO: finish this routine
    '''
    Compute the sample area array for SPEC probes using the center-in method.
    
    Input:
        binmid: array of bin midpoints [mm]
    Output:
        sa: array of sample area [mm^2]
    '''
    # Define SPEC defaults
    res = 0.01 # mm
    armdst = 63. # mm
    ndiodes = 128

    eaw = ndiodes * res # effective array width [mm]
    dof = 20.52 * 1000. * (binmid / 2.)**2 # depth of field [mm]
    dof[dof>armdst] = armdst # correct dof when greater than arm distance
    sa = dof * eaw # sample area [mm^2]

    return sa

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

def load_pbp(pbpfilename, phasefilename, Dmin=0.05, Dmax=3.2, iatThresh=1.e-6):
    '''
    Load the particle-by-particle data, only keeping the particles that will be accepted in the PSDs.
    
    Inputs:
        pbpfilename: full path to the PbP data file [str]
        phasefilename: full path to the phase ID data file [str]
        Dmin: smallest maximum dimenison to consider in the PDSs (speeds up processing) [mm]
        Dmax: largest maximum dimenison to consider in the PDSs (speeds up processing) [mm]
        iatThresh: smallest interarrival time to consider in the PDSs (speeds up processing) [s]
    Outputs:
        TODO: [Need to finish this part]
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
    #area = ds['image_area'].values
    #ar = area / (np.pi / 4 * diam_minR**2)
    #rej = ds['image_auto_reject'].values.astype(int)
    #centerin = ds['image_center_in'].values
    tempTime = ds['Time_in_seconds'].values # time in TAS clock cycles
    intArr = np.zeros(len(tempTime))
    intArr[1:] = np.diff(tempTime) # time difference between particles
    ovrld_flag = ds['DMT_DOF_SPEC_OVERLOAD'].values # SPEC overload flag (0 = no overload)
    
    # Load the partilce phase data
    print('Loading the phase ID data.')
    ds2 = xr.open_dataset(phasefilename)
    
    phase_ml = ds2['model_pred'].values # TODO: figure out what values indicate [1:liq, 0:ice]
    prob_ml = ds2['model_prob'].values # phase ID confidence [0-1, float]
    flag_ml = ds2['UW_flag'].values # 0: good, classifiable partile; else: rejected, not classified
    phase_holroyd = ds2['Holroyd_phase'].values # [1:liq, 0:ice]
    phase_ar = ds2['AR_threshold_phase'].values # [1:liq, 0:ice]

    # Determine the probability of a particle being liquid/ice based on the ML classification
    prob_liq_ml = np.zeros(len(prob_ml))
    prob_ice_ml = np.zeros(len(prob_ml))
    prob_liq_ml[phase_ml==1] = prob_ml[phase_ml==1.]
    prob_ice_ml[phase_ml==1] = 1. - prob_ml[phase_ml==1.]
    prob_ice_ml[phase_ml==0] = prob_ml[phase_ml==0.]
    prob_liq_ml[phase_ml==0] = 1. - prob_ml[phase_ml==0.]
    
    print('Removing the rejected particles.')
    time_all = np.copy(time) # first copy the time of all particles (for dead time calc in make_psd)
    good_inds = (flag_ml==0) # index the good particles based on the ML model flag
    #good_inds = (centerin==1) & (ar>=0.2) & (intArr>=iatThresh) & ((rej==48) | (rej==104) | (rej==72)) # flag based on standard UIOOPS criteria
    time = time[good_inds]
    diam_minR = diam_minR[good_inds]
    diam_areaR = diam_areaR[good_inds]
    phase_ml = phase_ml[good_inds]
    prob_ml = prob_ml[good_inds] # TODO: see if this should be diagnostically incorporated at 1 Hz level
    phase_holroyd = phase_holroyd[good_inds]
    phase_ar = phase_ar[good_inds]
    
    return (time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar, prob_ice_ml, prob_liq_ml,
            time_all, intArr, ovrld_flag)

def make_psd(flight_time, tas, particle_time, diameter_minR, diameter_areaR, phase_ml, phase_holroyd, phase_ar,
             particle_time_all, intArr_all, ovrld_flag_all, binEdges=None, tres=1):
    '''
    Inputs:
        tres: Temporal averaging resolution [s]
        # TODO add the other inputs
    # This subroutine needs to be tested still.
    # TODO: Add any of the phase probability data? e.g., for uncertainty estimates
    '''
    psd = {}
    
    if binEdges is None: # assign default bin edges
        binEdges = [50., 75., 100., 125., 150., 200., 250., 300., 350., 400., 475., 550., 625., 700., 800., 900., 1000.,
                    1200., 1400., 1600., 1800., 2000., 2200., 2400., 2600., 2800., 3000., 3200.] / 1000. # [mm]
    binWidth = np.diff(binEdges) / 10. # [cm]
    
    # Prepare the time loop and allocate PSD arrays
    dur = (flight_time[-1] - flight_time[0]) / np.timedelta64(1, 's') # flight duration [s]
    num_times = int(dur / tres)
    
    count_dmax_all = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ml = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_all = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ar = np.zeros((num_times, len(binEdges)-1))
    sv = np.zeros((num_times, len(binEdges)-1))
    
    sa = sample_area(binEdges[:-1] + np.diff(binEdges) / 2.) # mm^2
    
    time = np.array([], dtype='datetime64[s]')
    deadtime_flag = np.zeros(num_times).astype(int)
    for time_ind in range(num_times): # loop through each N-second interval
        if np.mod(time_ind, 1000.)==0:
            print(' Processing time {}/{}'.format(str(time_ind), str(num_times)))
        curr_time = flight_time[0] + np.timedelta64(int(time_ind*tres), 's')
        time = np.append(time, curr_time)

        # Compute the amount of dead time
        time_inds = (particle_time_all>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's'))
        intArr_subset = intArr_all[time_inds] # inter arrival time of particles for current flight time iteration
        intArr_subset[intArr_subset<-10.] = intArr_subset[intArr_subset<-10.] + (2**32-1) * (1.e-5 / 170.)
        intArr_subset[intArr_subset<0.] = 0.
        ovrld_flag_subset = ovrld_flag_all[time_inds] # overload flag of particles for current flight time iteration
        dead_time = np.sum(intArr_subset[ovrld_flag_subset!=0.]) # add up inter arrival times of overloaded particles
        dead_time[dead_time>np.float(tres)] = np.float(tres) # ensure probe dead time doesn't exceed the averaging interval
        if dead_time>0.8*np.float(tres):
            print(' {}: Dead time exceeds 80% of time interval. Flagging this period.'.format(np.datetime_as_string(curr_time)))
            deadtime_flag[time_ind] = 1
        
        # Compute the sample volume
        tas_mean = np.mean(tas[(flight_time>=curr_time) & (flight_time<curr_time+np.timedelta64(tres, 's'))])
        sv[time_ind] = (sa / 100.) * (tas_mean * 100.) * (np.float(tres)-dead_time) # cm^3
        
        # Find the particles within the current time interval for each phase classification
        inds_all = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) # all particles irrespective of phase
        inds_liq_ml = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_ml==1)
        inds_ice_ml = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_ml==0)
        inds_liq_holroyd = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_holroyd==1)
        inds_ice_holroyd = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_holroyd==0)
        inds_liq_ar = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_ar==1)
        inds_ice_ar = (particle_time>=curr_time) & (particle_time<curr_time+np.timedelta64(tres, 's')) & (phase_ar==0)

        if sum(inds_all)==1:
            count_dmax_all[time_ind, :] = np.histogram(diameter_minR[inds_all], bins=binEdges)[0]
            count_darea_all[time_ind, :] = np.histogram(diameter_areaR[inds_all], bins=binEdges)[0]
        if sum(inds_liq_ml)==1:
            count_dmax_liq_ml[time_ind, :] = np.histogram(diameter_minR[inds_liq_ml], bins=binEdges)[0]
            count_darea_liq_ml[time_ind, :] = np.histogram(diameter_areaR[inds_liq_ml], bins=binEdges)[0]
        if sum(inds_ice_ml)==1:
            count_dmax_ice_ml[time_ind, :] = np.histogram(diameter_minR[inds_ice_ml], bins=binEdges)[0]
            count_darea_ice_ml[time_ind, :] = np.histogram(diameter_areaR[inds_ice_ml], bins=binEdges)[0]
        if sum(inds_liq_holroyd)==1:
            count_dmax_liq_holroyd[time_ind, :] = np.histogram(diameter_minR[inds_liq_holroyd], bins=binEdges)[0]
            count_darea_liq_holroyd[time_ind, :] = np.histogram(diameter_areaR[inds_liq_holroyd], bins=binEdges)[0]
        if sum(inds_ice_holroyd)==1:
            count_dmax_ice_holroyd[time_ind, :] = np.histogram(diameter_minR[inds_ice_holroyd], bins=binEdges)[0]
            count_darea_ice_holroyd[time_ind, :] = np.histogram(diameter_areaR[inds_ice_holroyd], bins=binEdges)[0]
        if sum(inds_liq_ar)==1:
            count_dmax_liq_ar[time_ind, :] = np.histogram(diameter_minR[inds_liq_ar], bins=binEdges)[0]
            count_darea_liq_ar[time_ind, :] = np.histogram(diameter_areaR[inds_liq_ar], bins=binEdges)[0]
        if sum(inds_ice_ar)==1:
            count_dmax_ice_ar[time_ind, :] = np.histogram(diameter_minR[inds_ice_ar], bins=binEdges)[0]
            count_darea_ice_ar[time_ind, :] = np.histogram(diameter_areaR[inds_ice_ar], bins=binEdges)[0]

    # Save variables to object
    psd['time'] = time
    psd['bin_edges'] = binEdges # mm
    psd['bin_width'] = binWidth # cm
    psd['deadtime_flag'] = deadtime_flag # 0: < 80% deadtime for period; 1: Recommend skipping period due to high dead time
    psd['count_dmax_all'] = count_dmax_all
    psd['count_dmax_liq_ml'] = count_dmax_liq_ml
    psd['count_dmax_ice_ml'] = count_dmax_ice_ml
    psd['count_dmax_liq_holroyd'] = count_dmax_liq_holroyd
    psd['count_dmax_ice_holroyd'] = count_dmax_ice_holroyd
    psd['count_dmax_liq_ar'] = count_dmax_liq_ar
    psd['count_dmax_ice_ar'] = count_dmax_ice_ar
    psd['count_darea_all'] = count_darea_all
    psd['count_darea_liq_ml'] = count_darea_liq_ml
    psd['count_darea_ice_ml'] = count_darea_ice_ml
    psd['count_darea_liq_holroyd'] = count_darea_liq_holroyd
    psd['count_darea_ice_holroyd'] = count_darea_ice_holroyd
    psd['count_darea_liq_ar'] = count_darea_liq_ar
    psd['count_darea_ice_ar'] = count_darea_ice_ar
    psd['sample_volume'] = sv # cm**3

    return psd