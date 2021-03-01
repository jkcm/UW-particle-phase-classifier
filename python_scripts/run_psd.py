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

            psd = make_psd(flt_time, flt_tas, time, diam_minR, diam_areaR, phase_ml, prob_ml, phase_holroyd, phase_ar,
            time_all, intArr, ovrld_flag, bootstrap=True, binEdges=None, tres=1, outfile=outfile)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'error on flight {flt_string}:')
            print(e)
            continue
            