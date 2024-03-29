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
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from datetime import datetime

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

def sample_area(binmid):
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

def bootstrap_ml_classifications(cats, certs, n_iters=1):
    """generate multiple realizations of an array of binary classifications based on the classification certainty.
    
    Inputs:
        cats: nx1 array of binary classifications (0 or 1) [numpy.array]
        certs: nx1 array of classification probability or model certainty (range 0-1) [numpy.array]
        n_iters: number of desired realizations [int]
    Outputs:
        realizations: nxn_iters array of different possible realizations [numpy.array]
    """
    flipped_certs = certs.copy()
    flipped_certs[~cats.astype(bool)] = 1 - flipped_certs[~cats.astype(bool)]
    flipped_certs = np.tile(flipped_certs, (n_iters,1)) # jkcm: I had a .squeeze() in here, for some idiot reason.
    
    realizations = (np.random.rand(*flipped_certs.shape) < flipped_certs).astype(int)
    return realizations

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
        time: time of (accepted) particles classified by the random forest model [numpy.datetime64]
        diam_minR: diameter of a minimum enclosing circle for accepted particles [mm]
        diam_areaR: area equivalent diameter for accepted particles [mm]
        phase_ml: particle phase determined by the random forest model [0: ice; 1: liquid]
        phase_holroyd: particle phase determined by the Holroyd (1987) scheme [0: ice; 1: liquid]
        phase_ar: particle phase determined by an area ratio = 0.4 threshold [0: ice; 1: liquid]
        prob_ml: confidence in the random forest model that phase classification is correct [0-1]
        time_all: time of all particles [numpy.datetime64]
        intArr: particle interarrival time [s]
        ovrld_flag: particle overload flag [0: no dead time associated with partle; else: dead time]
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
    tempTime = ds['Time_in_seconds'].values # time in TAS clock cycles
    intArr = np.zeros(len(tempTime))
    intArr[1:] = np.diff(tempTime) # time difference between particles
    ovrld_flag = ds['DMT_DOF_SPEC_OVERLOAD'].values # SPEC overload flag (0 = no overload)
    
    # Load the partilce phase data
    print('Loading the phase ID data.')
    ds2 = xr.open_dataset(phasefilename)
    
    phase_ml = ds2['UW_phase'].values # [1:liq, 0:ice]
    prob_ml = ds2['UW_certainty'].values # phase ID confidence [0-1, float]
    flag_ml = ds2['UW_flag'].values # 0: good, classifiable partile; else: rejected, not classified
    phase_holroyd = ds2['Holroyd_phase'].values # [1:liq, 0:ice]
    phase_ar = ds2['AR_threshold_phase'].values # [1:liq, 0:ice]
    
    print('Removing the rejected particles.')
    time_all = np.copy(time) # first copy the time of all particles (for dead time calc in make_psd)
    good_inds = (flag_ml==0) & (intArr>=iatThresh) # index the good particles based on the ML model flag and inter arrival threshold
    #good_inds = (centerin==1) & (ar>=0.2) & (intArr>=iatThresh) & ((rej==48) | (rej==104) | (rej==72)) # flag based on standard UIOOPS criteria
    time = time[good_inds]
    diam_minR = diam_minR[good_inds]
    diam_areaR = diam_areaR[good_inds]
    phase_ml = phase_ml[good_inds]
    prob_ml = prob_ml[good_inds] # TODO: see if this should be diagnostically incorporated at 1 Hz level
    phase_holroyd = phase_holroyd[good_inds]
    phase_ar = phase_ar[good_inds]
    
    return (time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar, prob_ml, time_all, intArr, ovrld_flag)

def make_psd(flight_time, tas, particle_time, diameter_minR, diameter_areaR, phase_ml, prob_ml, phase_holroyd, phase_ar,
             particle_time_all, intArr_all, ovrld_flag_all, bootstrap=False, binEdges=None, tres=1, outfile=None):
    '''
    Inputs:
        flight_time: flight time [numpy.datetime64]
        tas: flight true airspeed [m/s]
        particle_time: time of (accepted) particles classified by the random forest model [numpy.datetime64]
        diameter_minR: diameter of a minimum enclosing circle for accepted particles [mm]
        diameter_areaR: area equivalent diameter for accepted particles [mm]
        phase_ml: particle phase determined by the machine learning model [0: ice; 1: liquid]
        prob_ml: model confidence that phase classification is correct [0-1; used if bootstrap==True]
        phase_holroyd: particle phase determined by the Holroyd (1987) scheme [0: ice; 1: liquid]
        phase_ar: particle phase determined by an area ratio = 0.4 threshold [0: ice; 1: liquid]
        particle_time_all: time of all particles [numpy.datetime64]
        intArr_all: particle interarrival time [s]
        ovrld_flag_all: particle overload flag [0: no dead time associated with partle; else: dead time]
        bootstrap: boolean [True/False; False is default] to bootstrap phase PSDs and generate uncertainty metrics
        binEdges: array of size bin endpoints to make PSDs [None: use default array; mm]
        tres: temporal resolution for PSDs [s; default = 1 s]
        outfile: full file path to save the PSD data [None: skip saving; str]
    '''
    psd = {}
    n_bootstrap=30
    
    if binEdges is None: # assign default bin edges
#         binEdges = np.array([50., 75., 100., 125., 150., 200., 250., 300., 350., 400., 475., 550., 625., 700., 800., 900., 1000.,
#         1200., 1400., 1600., 1800., 2000., 2200., 2400., 2600., 2800., 3000., 3200.]) / 1000. # [mm]
        binEdges=np.logspace(np.log10(56.4),np.log10(3200))/1000

    binWidth = np.diff(binEdges) / 10. # [cm]
    binMid = binEdges[:-1] / 10. + binWidth / 2. # bin midpoint for LWC calculation [cm]

    # Compute the sample area (used in sample volume calculation)
    sa = sample_area(binEdges[:-1] + np.diff(binEdges) / 2.) # [mm^2]
    
    # Prepare the 1 Hz time loop and allocate PSD arrays
    dur = (flight_time[-1] - flight_time[0]) / np.timedelta64(1, 's') # flight duration [s]
    num_times = int(dur) + 1 # include the last second of flight
    
    count_dmax_all = np.zeros((num_times, len(binEdges)-1))

    count_dmax_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_dmax_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_all = np.zeros((num_times, len(binEdges)-1))

    count_darea_liq_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_holroyd = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ar = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ar = np.zeros((num_times, len(binEdges)-1))
    
#     if not bootstrap: # jkcm: original code runs if no bootstrap
    count_dmax_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_dmax_ice_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_liq_ml = np.zeros((num_times, len(binEdges)-1))
    count_darea_ice_ml = np.zeros((num_times, len(binEdges)-1))
    if bootstrap: # jkcm: adding a dimension for bootstrapped counts
        print(f'bootstrapping with {n_bootstrap} samples...')
        count_dmax_liq_ml_bs = np.zeros((num_times, len(binEdges)-1, n_bootstrap))
        count_dmax_ice_ml_bs = np.zeros((num_times, len(binEdges)-1, n_bootstrap))
        count_darea_liq_ml_bs = np.zeros((num_times, len(binEdges)-1, n_bootstrap))
        count_darea_ice_ml_bs = np.zeros((num_times, len(binEdges)-1, n_bootstrap))
        
        
    sv = np.zeros((num_times, len(binEdges)-1))
    dead_time = np.zeros(num_times)
    if tres==1:
        lwc_ml = np.zeros(num_times)
        lwc_holroyd = np.zeros(num_times)
        lwc_ar = np.zeros(num_times)
        deadtime_flag = np.zeros(num_times).astype(int)
        if bootstrap: # jkcm: have to also include a hook in here
            lwc_ml_bs = np.zeros((num_times, n_bootstrap))
    
    # Gather the unique particle times and the indices corresponding to each
    unique_ptime, unique_ptime_inverse = np.unique(particle_time, return_inverse=True) # accepted particles
    unique_ptime_all, unique_ptime_all_inverse = np.unique(particle_time_all, return_inverse=True) # all particles

    # Find 1 Hz flight times when there are particles
    #flttime_inds_ptime = np.where(np.isin(flt_time, time)==True)[0] # accepted particles
    flttime_inds_ptime_all = np.where(np.isin(flight_time, particle_time_all)==True)[0] # all particles

    # Analyze the 1 Hz flight times containing any (accepted or rejected) particles
    tstart = np.datetime64(datetime.now()) # start the wall clock
    for ind in range(len(flttime_inds_ptime_all)):
        time_ind = flttime_inds_ptime_all[ind] # index from 1 Hz flight file
        curr_time = flight_time[time_ind] # time from 1 Hz flight file
        if np.mod(ind+1, 1000)==0.:
            print('Processing {} of {} ({})'.format(str(ind+1), str(len(flttime_inds_ptime_all)),
                                                    np.datetime_as_string(curr_time, unit='s')))

        # Compute the dead time of all (accepted + rejected) particles
        curr_ind_ptime_all = np.where(unique_ptime_all==curr_time)[0][0] # index from unique array of all particle times
        pinds_all = np.where(unique_ptime_all_inverse==curr_ind_ptime_all)[0] # particle indices for current time
        intArr_subset = intArr_all[pinds_all] # inter arrival time of particles for current time
        intArr_subset[intArr_subset<-10.] = intArr_subset[intArr_subset<-10.] + (2**32-1) * (1.e-5 / 170.)
        intArr_subset[intArr_subset<0.] = 0.
        ovrld_flag_subset = ovrld_flag_all[pinds_all] # overload flag of particles for current flight time iteration
        dead_time[time_ind] = np.sum(intArr_subset[ovrld_flag_subset!=0.]) # sum interarrival times of overloaded particles
        if dead_time[time_ind]>1.:
            dead_time[time_ind] = 1. # ensure probe dead time doesn't exceed the averaging interval
                
        # Compute the sample volume
        sv[time_ind, :] = (sa / 100.) * (tas[time_ind] * 100.) * (1.-dead_time[time_ind]) # cm**3
        
        # Find the accepted particles for the current time and for each phase classification
        curr_ind_ptime = np.where(unique_ptime==curr_time)[0]
        if len(curr_ind_ptime)==1: # at least one particle found...continue
            pinds = np.where(unique_ptime_inverse==curr_ind_ptime[0])[0] # accepted particle indices for current time
            diameter_minR_subset = diameter_minR[pinds]
            diameter_areaR_subset = diameter_areaR[pinds]
            phase_ml_subset = phase_ml[pinds]
            prob_ml_subset = prob_ml[pinds]
            phase_holroyd_subset = phase_holroyd[pinds]
            phase_ar_subset = phase_ar[pinds]
            
            count_dmax_all[time_ind, :] = np.histogram(diameter_minR_subset, bins=binEdges)[0]
            count_darea_all[time_ind, :] = np.histogram(diameter_areaR_subset, bins=binEdges)[0]
#             if not bootstrap: #jkcm: if bootstrap==False, continue as before, treating phase_ml_subset as deterministic
            inds_liq_ml = np.where(phase_ml_subset==1)[0]
            if len(inds_liq_ml)>0:
                count_dmax_liq_ml[time_ind, :] = np.histogram(diameter_minR_subset[inds_liq_ml], bins=binEdges)[0]
                count_darea_liq_ml[time_ind, :] = np.histogram(diameter_areaR_subset[inds_liq_ml], bins=binEdges)[0]
                #lwc_ml[time_ind] = np.nansum(count_dmax_liq_ml[time_ind, :] /
                                             #sv[time_ind, :] * np.pi / 6. * binMid**3.) * 1.e6 # g m**-3
            inds_ice_ml = np.where(phase_ml_subset==0)[0]
            if len(inds_ice_ml)>0:
                count_dmax_ice_ml[time_ind, :] = np.histogram(diameter_minR_subset[inds_ice_ml], bins=binEdges)[0]
                count_darea_ice_ml[time_ind, :] = np.histogram(diameter_areaR_subset[inds_ice_ml], bins=binEdges)[0]
            if bootstrap:
                phase_ml_subset_bs = bootstrap_ml_classifications(phase_ml_subset, prob_ml_subset, n_iters=n_bootstrap)
                # jkcm: this is now len(subset) x n_bootstrap
                for i in range(n_bootstrap):
                    phase_ml_subset = phase_ml_subset_bs[i,:] # this is now just one realization
                    try:
                        assert phase_ml_subset.shape == prob_ml_subset.shape # jkcm: just a test, can delete
                    except AssertionError as e:
                        print(phase_ml_subset.shape)
                        print(prob_ml_subset.shape)
                        raise e
                    inds_liq_ml = np.where(phase_ml_subset==1)[0]
                    if len(inds_liq_ml)>0:
                        count_dmax_liq_ml_bs[time_ind, :, i] = np.histogram(diameter_minR_subset[inds_liq_ml], bins=binEdges)[0]
                        count_darea_liq_ml_bs[time_ind, :, i] = np.histogram(diameter_areaR_subset[inds_liq_ml], bins=binEdges)[0]
                        #lwc_ml[time_ind, i] = np.nansum(count_dmax_liq_ml[time_ind, :] /
                                                     #sv[time_ind, :] * np.pi / 6. * binMid**3.) * 1.e6 # g m**-3
                    inds_ice_ml = np.where(phase_ml_subset==0)[0]
                    if len(inds_ice_ml)>0:
                        count_dmax_ice_ml_bs[time_ind, :, i] = np.histogram(diameter_minR_subset[inds_ice_ml], bins=binEdges)[0]
                        count_darea_ice_ml_bs[time_ind, :, i] = np.histogram(diameter_areaR_subset[inds_ice_ml], bins=binEdges)[0]
            
            inds_liq_holroyd = np.where(phase_holroyd_subset==1)[0]
            if len(inds_liq_holroyd)>0:
                count_dmax_liq_holroyd[time_ind, :] = np.histogram(diameter_minR_subset[inds_liq_holroyd], bins=binEdges)[0]
                count_darea_liq_holroyd[time_ind, :] = np.histogram(diameter_areaR_subset[inds_liq_holroyd], bins=binEdges)[0]
                #lwc_holroyd[time_ind] = np.nansum(count_dmax_liq_holroyd[time_ind, :] /
                                                  #sv[time_ind, :] * np.pi / 6. * binMid**3.) * 1.e6 # g m**-3
            inds_ice_holroyd = np.where(phase_holroyd_subset==0)[0]
            if len(inds_ice_holroyd)>0:
                count_dmax_ice_holroyd[time_ind, :] = np.histogram(diameter_minR_subset[inds_ice_holroyd], bins=binEdges)[0]
                count_darea_ice_holroyd[time_ind, :] = np.histogram(diameter_areaR_subset[inds_ice_holroyd], bins=binEdges)[0]
            inds_liq_ar = np.where(phase_ar_subset==1)[0]
            if len(inds_liq_ar)>0:
                count_dmax_liq_ar[time_ind, :] = np.histogram(diameter_minR_subset[inds_liq_ar], bins=binEdges)[0]
                count_darea_liq_ar[time_ind, :] = np.histogram(diameter_areaR_subset[inds_liq_ar], bins=binEdges)[0]
                #lwc_ar[time_ind] = np.nansum(count_dmax_liq_ar[time_ind, :] /
                                             #sv[time_ind, :] * np.pi / 6. * binMid**3.) * 1.e6 # g m**-3
            inds_ice_ar = np.where(phase_ar_subset==0)[0]
            if len(inds_ice_ar)>0:
                count_dmax_ice_ar[time_ind, :] = np.histogram(diameter_minR_subset[inds_ice_ar], bins=binEdges)[0]
                count_darea_ice_ar[time_ind, :] = np.histogram(diameter_areaR_subset[inds_ice_ar], bins=binEdges)[0]
                
    if tres==1: # This block takes care of some bulk quantities if tres == 1 s
        # Dead time flag
        deadtime_flag[dead_time>0.8] = 1
    else: # This block takes care of averaging if tres > 1 s
        num_times = int(dur / tres)
        flight_time = flight_time[::tres]
        deadtime_flag = np.zeros(len(num_times)).astype(int)

        # Reshape arrays to be num_times x tres x num_bins, then compute the sum
        count_dmax_all = np.nansum(count_dmax_all.reshape(num_times, tres, len(binEdges)-1), axis=1)
        
        count_dmax_liq_holroyd = np.nansum(count_dmax_liq_holroyd.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_dmax_ice_holroyd = np.nansum(count_dmax_ice_holroyd.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_dmax_liq_ar = np.nansum(count_dmax_liq_ar.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_dmax_ice_ar = np.nansum(count_dmax_ice_ar.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_all = np.nansum(count_darea_all.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_liq_holroyd = np.nansum(count_darea_liq_holroyd.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_ice_holroyd = np.nansum(count_darea_ice_holroyd.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_liq_ar = np.nansum(count_darea_liq_ar.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_ice_ar = np.nansum(count_darea_ice_ar.reshape(num_times, tres, len(binEdges)-1), axis=1)
        sv = np.nansum(sv.reshape(num_times, tres, len(binEdges)-1), axis=1)
        dead_time = np.nansum(dead_time.reshape(num_times, tres), axis=1)
#         if not bootstrap: # jkcm: moved these down here for tidiness
        count_dmax_liq_ml = np.nansum(count_dmax_liq_ml.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_dmax_ice_ml = np.nansum(count_dmax_ice_ml.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_liq_ml = np.nansum(count_darea_liq_ml.reshape(num_times, tres, len(binEdges)-1), axis=1)
        count_darea_ice_ml = np.nansum(count_darea_ice_ml.reshape(num_times, tres, len(binEdges)-1), axis=1)
        if bootstrap: # jkcm: OK now it's getting complicated
            count_dmax_liq_ml_bs = np.nansum(count_dmax_liq_ml_bs.reshape(num_times, tres, len(binEdges)-1, n_bootstrap), axis=1)
            count_dmax_ice_ml_bs = np.nansum(count_dmax_ice_ml_bs.reshape(num_times, tres, len(binEdges)-1, n_bootstrap), axis=1)
            count_darea_liq_ml_bs = np.nansum(count_darea_liq_ml_bs.reshape(num_times, tres, len(binEdges)-1, n_bootstrap), axis=1)
            count_darea_ice_ml_bs = np.nansum(count_darea_ice_ml_bs.reshape(num_times, tres, len(binEdges)-1, n_bootstrap), axis=1)
        # Dead time flag
        deadtime_flag[dead_time>0.8] = 1
        
    if bootstrap:  # this is where we get all our pretty statistics
                    
        print('bootstrapping summary stats...')
        count_dmax_liq_ml_mean = np.nanmean(count_dmax_liq_ml_bs, axis=-1)
        count_dmax_ice_ml_mean = np.nanmean(count_dmax_ice_ml_bs, axis=-1)
        count_darea_liq_ml_mean = np.nanmean(count_darea_liq_ml_bs, axis=-1)
        count_darea_ice_ml_mean = np.nanmean(count_darea_ice_ml_bs, axis=-1)
        count_dmax_liq_ml_median = np.nanmedian(count_dmax_liq_ml_bs, axis=-1)
        count_dmax_ice_ml_median = np.nanmedian(count_dmax_ice_ml_bs, axis=-1)
        count_darea_liq_ml_median = np.nanmedian(count_darea_liq_ml_bs, axis=-1)
        count_darea_ice_ml_median = np.nanmedian(count_darea_ice_ml_bs, axis=-1)
        count_dmax_liq_ml_min = np.nanmin(count_dmax_liq_ml_bs, axis=-1)
        count_dmax_ice_ml_min = np.nanmin(count_dmax_ice_ml_bs, axis=-1)
        count_darea_liq_ml_min = np.nanmin(count_darea_liq_ml_bs, axis=-1)
        count_darea_ice_ml_min = np.nanmin(count_darea_ice_ml_bs, axis=-1)
        count_dmax_liq_ml_max = np.nanmax(count_dmax_liq_ml_bs, axis=-1)
        count_dmax_ice_ml_max = np.nanmax(count_dmax_ice_ml_bs, axis=-1)
        count_darea_liq_ml_max = np.nanmax(count_darea_liq_ml_bs, axis=-1)
        count_darea_ice_ml_max = np.nanmax(count_darea_ice_ml_bs, axis=-1)
        count_dmax_liq_ml_stddev = np.nanstd(count_dmax_liq_ml_bs, axis=-1)
        count_dmax_ice_ml_stddev = np.nanstd(count_dmax_ice_ml_bs, axis=-1)
        count_darea_liq_ml_stddev = np.nanstd(count_darea_liq_ml_bs, axis=-1)
        count_darea_ice_ml_stddev = np.nanstd(count_darea_ice_ml_bs, axis=-1)            
        count_dmax_liq_ml_25pct = np.nanpercentile(count_dmax_liq_ml_bs, q=25, axis=-1)
        count_dmax_ice_ml_25pct = np.nanpercentile(count_dmax_ice_ml_bs, q=25, axis=-1)
        count_darea_liq_ml_25pct = np.nanpercentile(count_darea_liq_ml_bs, q=25, axis=-1)
        count_darea_ice_ml_25pct = np.nanpercentile(count_darea_ice_ml_bs, q=25, axis=-1)        
        count_dmax_liq_ml_75pct = np.nanpercentile(count_dmax_liq_ml_bs, q=75, axis=-1)
        count_dmax_ice_ml_75pct = np.nanpercentile(count_dmax_ice_ml_bs, q=75, axis=-1)
        count_darea_liq_ml_75pct = np.nanpercentile(count_darea_liq_ml_bs, q=75, axis=-1)
        count_darea_ice_ml_75pct = np.nanpercentile(count_darea_ice_ml_bs, q=75, axis=-1)      
        
        

    
    # Compute LWC
#     print(f':count_dmax_liq_ml shape: {count_dmax_liq_ml.shape}')
#     if not bootstrap:
    lwc_ml = np.nansum(count_darea_liq_ml /sv * np.pi / 6. * binMid**3., axis=1) * 1.e6 # g m**-3  # changed from dmax to darea
    if bootstrap:
        lwc_ml_all = np.nansum(count_darea_liq_ml_bs /sv[:,:,None] * np.pi / 6. * (binMid**3)[None,:,None], axis=1) * 1.e6 # g m**-3
        lwc_ml_median = np.nanmedian(lwc_ml_all, axis=-1)
        lwc_ml_mean = np.nanmean(lwc_ml_all, axis=-1)
        lwc_ml_min = np.nanmin(lwc_ml_all, axis=-1)
        lwc_ml_max = np.nanmax(lwc_ml_all, axis=-1)
        lwc_ml_stddev = np.nanstd(lwc_ml_all, axis=-1)
        lwc_ml_25pct = np.nanpercentile(lwc_ml_all, q=25, axis=-1)        
        lwc_ml_75pct = np.nanpercentile(lwc_ml_all, q=75, axis=-1)
        
#         count_dmax_liq_ml = count_dmax_liq_ml_mean # just for some naming consistency
#         count_dmax_ice_ml = count_dmax_ice_ml_mean
#         count_darea_liq_ml = count_darea_liq_ml_mean
#         count_darea_ice_ml = count_darea_ice_ml_mean
    
    lwc_holroyd = np.nansum(count_dmax_liq_holroyd /sv * np.pi / 6. * binMid**3., axis=1) * 1.e6 # g m**-3
    lwc_ar = np.nansum(count_dmax_liq_ar /sv * np.pi / 6. * binMid**3., axis=1) * 1.e6 # g m**-3


    
    print('\nElapsed time: {} seconds'.format((np.datetime64(datetime.now())-tstart)/np.timedelta64(1, 's')))
    
    # Save variables to object
    psd['time'] = flight_time
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
    psd['count_dmax_all'] = count_dmax_all
    psd['count_darea_all'] = count_darea_all
    psd['count_darea_liq_ml'] = count_darea_liq_ml
    psd['count_darea_ice_ml'] = count_darea_ice_ml
    psd['count_darea_liq_holroyd'] = count_darea_liq_holroyd
    psd['count_darea_ice_holroyd'] = count_darea_ice_holroyd
    psd['count_darea_liq_ar'] = count_darea_liq_ar
    psd['count_darea_ice_ar'] = count_darea_ice_ar
    psd['sample_volume'] = sv # cm**3
    psd['lwc_ml'] = lwc_ml # g m**-3
    psd['lwc_holroyd'] = lwc_holroyd # g m**-3
    psd['lwc_ar'] = lwc_ar # g m**-3
    
    if bootstrap:  # jkcm: gotta add them to outputs
        
        psd['count_dmax_liq_ml_mean'] = count_dmax_liq_ml_mean
        psd['count_dmax_ice_ml_mean'] = count_dmax_ice_ml_mean
        psd['count_darea_liq_ml_mean'] = count_darea_liq_ml_mean
        psd['count_darea_ice_ml_mean'] = count_darea_ice_ml_mean
        psd['count_dmax_liq_ml_median'] = count_dmax_liq_ml_median
        psd['count_dmax_ice_ml_median'] = count_dmax_ice_ml_median
        psd['count_darea_liq_ml_median'] = count_darea_liq_ml_median
        psd['count_darea_ice_ml_median'] = count_darea_ice_ml_median
        psd['count_dmax_liq_ml_min'] = count_dmax_liq_ml_min
        psd['count_dmax_ice_ml_min'] = count_dmax_ice_ml_min
        psd['count_darea_liq_ml_min'] = count_darea_liq_ml_min
        psd['count_darea_ice_ml_min'] = count_darea_ice_ml_min
        psd['count_dmax_liq_ml_max'] = count_dmax_liq_ml_max
        psd['count_dmax_ice_ml_max'] = count_dmax_ice_ml_max
        psd['count_darea_liq_ml_max'] = count_darea_liq_ml_max
        psd['count_darea_ice_ml_max'] = count_darea_ice_ml_max
        psd['count_dmax_liq_ml_stddev'] = count_dmax_liq_ml_stddev
        psd['count_dmax_ice_ml_stddev'] = count_dmax_ice_ml_stddev
        psd['count_darea_liq_ml_stddev'] = count_darea_liq_ml_stddev
        psd['count_darea_ice_ml_stddev'] = count_darea_ice_ml_stddev
        
        psd['lwc_ml_all'] = lwc_ml_all
        psd['lwc_ml_mean'] = lwc_ml_mean
        psd['lwc_ml_median'] = lwc_ml_median
        psd['lwc_ml_min'] = lwc_ml_min
        psd['lwc_ml_max'] = lwc_ml_max
        psd['lwc_ml_stddev'] = lwc_ml_stddev
        psd['lwc_ml_25pct'] = lwc_ml_25pct
        psd['lwc_ml_75pct'] = lwc_ml_75pct
    
        psd['count_dmax_liq_ml_25pct'] = count_dmax_liq_ml_25pct
        psd['count_dmax_ice_ml_25pct'] = count_dmax_ice_ml_25pct
        psd['count_darea_liq_ml_25pct'] = count_darea_liq_ml_25pct
        psd['count_darea_ice_ml_25pct'] = count_darea_ice_ml_25pct
        psd['count_dmax_liq_ml_75pct'] = count_dmax_liq_ml_75pct
        psd['count_dmax_ice_ml_75pct'] = count_dmax_ice_ml_75pct
        psd['count_darea_liq_ml_75pct'] = count_darea_liq_ml_75pct
        psd['count_darea_ice_ml_75pct'] = count_darea_ice_ml_75pct    

    # Save the PSDs to file if specified
#     if outfile is not None:
    ds = xr.Dataset()

    ds.coords['time'] = ('time', flight_time)
    ds.coords['bin_edges'] = ('bin_edges', binEdges)
    ds.coords['bin_edges'].attrs['units'] = 'mm'
    ds.coords['bin_width'] = ('size_bin', binWidth)
    ds.coords['bin_width'].attrs['units'] = 'cm'

    #if bootstrap: # joefinlon: do we need this block?
        #count_dmax_liq_ml


    ds['count_dmax_all'] = (['time', 'size_bin'], count_dmax_all)
    ds['count_dmax_liq_ml'] = (['time', 'size_bin'], count_dmax_liq_ml)
    ds['count_dmax_liq_holroyd'] = (['time', 'size_bin'], count_dmax_liq_holroyd)
    ds['count_dmax_liq_ar'] = (['time', 'size_bin'], count_dmax_liq_ar)
    ds['count_dmax_ice_ml'] = (['time', 'size_bin'], count_dmax_ice_ml)
    ds['count_dmax_ice_holroyd'] = (['time', 'size_bin'], count_dmax_ice_holroyd)
    ds['count_dmax_ice_ar'] = (['time', 'size_bin'], count_dmax_ice_ar)
    ds['count_darea_all'] = (['time', 'size_bin'], count_darea_all)
    ds['count_darea_liq_ml'] = (['time', 'size_bin'], count_darea_liq_ml)
    ds['count_darea_liq_holroyd'] = (['time', 'size_bin'], count_darea_liq_holroyd)
    ds['count_darea_liq_ar'] = (['time', 'size_bin'], count_darea_liq_ar)
    ds['count_darea_ice_ml'] = (['time', 'size_bin'], count_darea_ice_ml)
    ds['count_darea_ice_holroyd'] = (['time', 'size_bin'], count_darea_ice_holroyd)
    ds['count_darea_ice_ar'] = (['time', 'size_bin'], count_darea_ice_ar)
    ds['sample_volume'] = (['time', 'size_bin'], sv)
    ds['sample_volume'].attrs['units'] = 'cm**3'
    ds['lwc_ml'] = ('time', lwc_ml)
    ds['lwc_ml'].attrs['units'] = 'g m**-3'
    ds['lwc_holroyd'] = ('time', lwc_holroyd)
    ds['lwc_holroyd'].attrs['units'] = 'g m**-3'
    ds['lwc_ar'] = ('time', lwc_ar)
    ds['lwc_ar'].attrs['units'] = 'g m**-3'
    ds['deadtime_flag'] = ('time', deadtime_flag)
    ds['deadtime_flag'].attrs['description'] = '0: < 80% deadtime for period; 1: Recommend skipping period due to high dead time'

    if bootstrap: #jkcm: and save it too!
        ds['count_dmax_liq_ml_mean']  = (['time', 'size_bin'], count_dmax_liq_ml_mean)
        ds['count_dmax_ice_ml_mean']  = (['time', 'size_bin'], count_dmax_ice_ml_mean)
        ds['count_darea_liq_ml_mean'] = (['time', 'size_bin'], count_darea_liq_ml_mean)
        ds['count_darea_ice_ml_mean'] = (['time', 'size_bin'], count_darea_ice_ml_mean)
        ds['count_dmax_liq_ml_median']  = (['time', 'size_bin'], count_dmax_liq_ml_median)
        ds['count_dmax_ice_ml_median']  = (['time', 'size_bin'], count_dmax_ice_ml_median)
        ds['count_darea_liq_ml_median'] = (['time', 'size_bin'], count_darea_liq_ml_median)
        ds['count_darea_ice_ml_median'] = (['time', 'size_bin'], count_darea_ice_ml_median)
        ds['count_dmax_liq_ml_min']     = (['time', 'size_bin'], count_dmax_liq_ml_min)
        ds['count_dmax_ice_ml_min']     = (['time', 'size_bin'], count_dmax_ice_ml_min)
        ds['count_darea_liq_ml_min']    = (['time', 'size_bin'], count_darea_liq_ml_min)
        ds['count_darea_ice_ml_min']    = (['time', 'size_bin'], count_darea_ice_ml_min)
        ds['count_dmax_liq_ml_max']     = (['time', 'size_bin'], count_dmax_liq_ml_max)
        ds['count_dmax_ice_ml_max']     = (['time', 'size_bin'], count_dmax_ice_ml_max)
        ds['count_darea_liq_ml_max']    = (['time', 'size_bin'], count_darea_liq_ml_max)
        ds['count_darea_ice_ml_max']    = (['time', 'size_bin'], count_darea_ice_ml_max)
        ds['count_dmax_liq_ml_stddev']  = (['time', 'size_bin'], count_dmax_liq_ml_stddev)
        ds['count_dmax_ice_ml_stddev']  = (['time', 'size_bin'], count_dmax_ice_ml_stddev)
        ds['count_darea_liq_ml_stddev'] = (['time', 'size_bin'], count_darea_liq_ml_stddev)
        ds['count_darea_ice_ml_stddev'] = (['time', 'size_bin'], count_darea_ice_ml_stddev)
        ds['count_dmax_liq_ml_25pct']   = (['time', 'size_bin'], count_dmax_liq_ml_25pct)
        ds['count_dmax_ice_ml_25pct']   = (['time', 'size_bin'], count_dmax_ice_ml_25pct)
        ds['count_darea_liq_ml_25pct']  = (['time', 'size_bin'], count_darea_liq_ml_25pct)
        ds['count_darea_ice_ml_25pct']  = (['time', 'size_bin'], count_darea_ice_ml_25pct)
        ds['count_dmax_liq_ml_75pct']   = (['time', 'size_bin'], count_dmax_liq_ml_75pct)
        ds['count_dmax_ice_ml_75pct']   = (['time', 'size_bin'], count_dmax_ice_ml_75pct)
        ds['count_darea_liq_ml_75pct']  = (['time', 'size_bin'], count_darea_liq_ml_75pct)
        ds['count_darea_ice_ml_75pct']  = (['time', 'size_bin'], count_darea_ice_ml_75pct)


        ds['lwc_ml_mean']  = ('time', lwc_ml_mean)
        ds['lwc_ml_median']  = ('time', lwc_ml_median)
        ds['lwc_ml_min']  = ('time', lwc_ml_min)
        ds['lwc_ml_max']  = ('time', lwc_ml_max)
        ds['lwc_ml_stddev']  = ('time', lwc_ml_stddev)
        ds['lwc_ml_25pct']  = ('time', lwc_ml_25pct)
        ds['lwc_ml_75pct']  = ('time', lwc_ml_75pct)

        if outfile:
    #         ds.to_netcdf(outfile)

            comp = dict(zlib=True, complevel=2)
            ds.to_netcdf(outfile, engine='h5netcdf', encoding={var: comp for var in ds.data_vars})
    return (psd, ds)