import xarray as xr
import numpy as np
import pandas as pd
import pickle
import os
from joblib import dump, load
import datetime
import matplotlib.pyplot as plt
import sys

import matplotlib as mpl

import glob
# from phase_psd import *
from run_psd import *

print('loaded')


if __name__=="__main__":
    
    
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import PBSCluster
    cluster = LocalCluster(n_workers=4, dashboard_address=37188)
    client = Client(cluster)



    proc_num = int(sys.argv[1])
    end_num = int(sys.argv[2])

    all_flights = [f'rf{i:02}' for i in np.arange(1,16)]
    all_data = {}
    print(f'working on flights {str(all_flights[proc_num:end_num])}')
    
    for flt_string in all_flights[proc_num:end_num]:
        
        try:
            outfile = f'/home/disk/eos9/jkcm/Data/particle/psd/test/{flt_string}_heterogeneity.nc'
            navfilename = glob.glob('/home/disk/eos9/jfinlon/socrates/' + flt_string + '/*.PNI.nc')[0]
            pbpfilename = '/home/disk/eos9/jfinlon/socrates/' + flt_string + '/pbp.' + flt_string + '.2DS.H.nc'
            phasefilename = '/home/disk/eos9/jkcm/Data/particle/classified/' + 'UW_particle_classifications.' + flt_string + '.nc'
            [flight_time, flight_tas] = load_nav(navfilename)
            # [time, diam_minR, diam_areaR, phase_ml, phase_holroyd, phase_ar, prob_ml, time_all, intArr, ovrld_flag] = load_pbp(
            #     pbpfilename, phasefilename, Dmin=0.05, Dmax=3.2, iatThresh=1.e-6)
            phase_data = xr.open_dataset(phasefilename, chunks={})
            nav_data = xr.open_dataset(navfilename, chunks={})
            print(f'loaded pbp data, {flt_string}')
            new_nav_data = make_heterogeneity(phase_data, nav_data)
            all_data[flt_string] = new_nav_data
            new_nav_data.to_netcdf(outfile)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'error on flight {flt_string}:')
            raise e
            continue
            