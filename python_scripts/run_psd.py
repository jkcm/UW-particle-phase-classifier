#Author: Hans Mohrmann

import xarray as xr
import numpy as np
import pandas as pd
import pickle
import os
from joblib import dump, load
import datetime
import matplotlib.pyplot as plt
import glob
from phase_psd import *


def add_datetime_to_processed_data(proc_data):
    """take the 2DS processed dataset and reindex the time properly"""
    secs = (proc_data.Time.values%100) + proc_data.msec.values/1000
    mins = (proc_data.Time.values//100)%100
    hours = proc_data.Time.values//10000
    
    day = proc_data.Date.values%100
    months = (proc_data.Date.values//100)%100
    year = proc_data.Date.values//10000
    datetime = np.array(
        [np.datetime64(f'{int(year[i])}-{int(months[i]):02}-{int(day[i]):02}T{int(hours[i]):02}:{int(mins[i]):02}:{int(secs[i]):06.3f}') for i in range(len(year))])
    proc_data['datetime'] = (['time'], datetime)
    return proc_data
def bootstrap_ml_classifications(cats, certs, n_iters=1):
    """generate multiple realizations of an array of binary classifications based on the classification certainty
    
    Inputs:
        cats: nx1 array of binary classifications (0 or 1) [numpy.array]
        certs: nx1 array of classification probability or model certainty (range 0-1) [numpy.array]
        n_iters: number of desired realizations [int]
    Outputs:
        realizations: nxn_iters array of different possible realizations [numpy.array]
    """
    flipped_certs = certs.copy()
    flipped_certs[~cats.astype(bool)] = 1 - flipped_certs[~cats.astype(bool)]
    flipped_certs = np.tile(flipped_certs, (n_iters,1)).squeeze()
    
    realizations = (np.random.rand(*flipped_certs.shape) < flipped_certs).astype(int)
    return realizations
    
    
def make_heterogeneity(phase_data, nav_data): # oof this is an ugly function. I am sorry Python

    data_indices = np.argwhere(~np.isnan(phase_data.UW_certainty.values))
    prob_ml = phase_data.UW_certainty.values
    phase_ml = phase_data.UW_phase.values
    prob_ml, phase_ml = prob_ml[data_indices], phase_ml[data_indices]
    ice_probs = prob_ml.copy()
    ice_probs[phase_ml==1] = 1-ice_probs[phase_ml==1]  # probability that particle is ice
    liq_probs = prob_ml.copy()
    liq_probs[phase_ml==0] = 1-liq_probs[phase_ml==0]  # probability that particle is liquid
    all_ice_prob = ice_probs[1:]*ice_probs[:-1]   # probability that a particle and its next neighbor are both ice
    all_liq_prob = liq_probs[1:]*liq_probs[:-1]   # ditto but for liquid
    all_ice_or_liq = all_ice_prob+all_liq_prob   #probability that a phase flip did not occur
    phase_flip_prob = np.insert((1-all_ice_or_liq), 0, 0)
    temp_arr = np.full_like(phase_data.UW_certainty.values, fill_value=np.nan)
    temp_arr[data_indices] = phase_flip_prob[:, None]
    print('here1')
    phase_data['phase_flip_prob'] = (('time'), temp_arr)
    phase_data = phase_data.rename({'datetime': 'time'}).set_coords('time')
    phase_data_resample = phase_data['phase_flip_prob'].resample(time='1s')
    print('after resample')
    phase_flip_counts_1hz = phase_data_resample.sum()
    particle_counts_1hz = phase_data_resample.count()
    print('done with 1hz proc')
    all_particles_1hz = phase_data.AR_threshold_phase.resample(time='1s').count() # all particle count
    print('done with all part count')
    #align with nav data
    start, end = max(nav_data.Time[0], phase_flip_counts_1hz.time[0]), min(nav_data.Time[-1], phase_flip_counts_1hz.time[-1])
    nav_data = nav_data.sel(Time=slice(start, end))
    phase_flip_counts_1hz = phase_flip_counts_1hz.sel(time=slice(start, end))
    particle_counts_1hz = particle_counts_1hz.sel(time=slice(start, end))
    all_particles_1hz = all_particles_1hz.sel(time=slice(start, end))
    nav_data['phase_flip_counts'] = (('Time'), phase_flip_counts_1hz)
    nav_data['particle_counts'] = (('Time'), particle_counts_1hz)
    nav_data['all_particles_counts'] = (('Time'), all_particles_1hz)

    return nav_data

    
if __name__ == "__main__":

    all_flights = [f'rf{i:02}' for i in np.arange(1,16)]
    for flt_string in all_flights:
        try:

            
            outfile = f'/home/disk/eos9/jkcm/Data/particle/psd/test/{flt_string}_psd.nc'
            navfilename = glob.glob('/home/disk/eos9/jfinlon/socrates/' + flt_string + '/*.PNI.nc')[0]
            pbpfilename = '/home/disk/eos9/jfinlon/socrates/' + flt_string + '/pbp.' + flt_string + '.2DS.H.nc'
            phasefilename = '/home/disk/eos9/jkcm/Data/particle/classified/' + 'UW_particle_classifications.' + flt_string + '.nc'
            [flt_time, flt_tas] = load_nav(navfilename)
            [time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar, prob_ml, time_all, intArr, ovrld_flag] = load_pbp(
                pbpfilename, phasefilename, Dmin=0.05, Dmax=3.2, iatThresh=1.e-6)
 
            (_, ds) = make_psd(flt_time, flt_tas, time, diam_minR, diam_areaR, phase_ml, prob_ml, phase_holroyd, phase_ar,
            time_all, intArr, ovrld_flag, bootstrap=True, binEdges=None, tres=1, outfile=None)


            nav_data = xr.open_dataset(navfilename)
            phase_data = xr.open_dataset(phasefilename)
            assert(np.all((ds.time.values == nav_data.Time.values)))            
            het = make_heterogeneity(phase_data, nav_data)
            ds['particle_counts'] = (('time'), het['particle_counts'].values)
            ds['phase_flip_counts'] = (('time'), het['phase_flip_counts'].values)
            ds['particle_counts'] = (('time'), het['particle_counts'].values)
            ds['all_particle_counts'] = (('time'), het['all_particles_counts'].values)
            ds.to_netcdf(outfile)
#             het_outfile = f'/home/disk/eos9/jkcm/Data/particle/psd/test/{flt_string}_heterogeneity.nc'
                
                
                
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'error on flight {flt_string}:')
            print(e)
            continue
            