#!/usr/bin/env conda run -n classified-cset python
# -*- coding: utf-8 -*-
"""Create the filtered 2DS particle dataset for liquid particle.
    Relies on the 2DS processed dataset (Joe Finlon), as well as the standard low-res aircraft data.
    Created by Johannes Mohrmann, March 5 2020"""
    
    
import numpy as np
import xarray as xr
import random
import matplotlib.pyplot as plt
import datetime as dt
import glob
import pickle
from functools import reduce
import math
import os

def get_file_lists():
    """Return the lists of the processed 2DS files, 2DS summary probe data files, and flight summary files. 
        Only returns files when all three exist for a flight"""
    processed_files = sorted(glob.glob(r'/home/disk/eos9/jfinlon/socrates/*/proc2DS_H.*.nc'))
    processed_flights = [i[-7:-3].lower() for i in processed_files]
    probe_files = sorted(glob.glob(r'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/processed_probe_data/product.*_rf*.2DS.nc'))
    probe_flights = [i[-11:-7].lower() for i in probe_files]
    flight_files = sorted(glob.glob(r'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/low_rate/RF*.PNI.nc'))
    flight_flights = [os.path.basename(i)[:4].lower() for i in flight_files]
    complete_flights = sorted(list(set(processed_flights).intersection(probe_flights, flight_flights)))
    processed_files = [f for i, f in zip(processed_flights, processed_files) if i in complete_flights]
    probe_files = [f for i, f in zip(probe_flights, probe_files) if i in complete_flights]
    flight_files = [f for i, f in zip(flight_flights, flight_files) if i in complete_flights]

    return{'probe_files': probe_files, 'flight_files': flight_files, 'processed_files': processed_files, 'flights': complete_flights}


def add_datetime_to_processed_data(proc_data):
    """take the 2DS processed dataset and add a datetime variable
        This can definitely be sped up (create a datetime array fast, add secs/mins/hours etc to it). Not worth it for now."""
    secs = (proc_data.Time.values%100) + proc_data.msec.values/1000
    mins = (proc_data.Time.values//100)%100
    hours = proc_data.Time.values//10000
    
    day = proc_data.Date.values%100
    months = (proc_data.Date.values//100)%100
    year = proc_data.Date.values//10000
    datetime = np.array(
        [np.datetime64(f'{year[i]}-{months[i]:02}-{day[i]}T{hours[i]:02}:{mins[i]:02}:{secs[i]:06.3f}') for i in range(len(year))])
    proc_data['datetime'] = (['time'], datetime)
    return proc_data


def find_nearest(array,value):
    """find nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
           

def get_warm_periods(flight_data, thresh=5):
    """return the start/end times for a warm period, defined by thresh"""
    warm = flight_data.ATX>thresh
    if warm[0]:
        ones = [0]+list(np.nonzero(np.diff(warm.astype(int))==1)[0])
    else: 
        ones = list(np.nonzero(np.diff(warm.astype(int))==1)[0])
    if warm[-1]:
        mins = list(np.nonzero(np.diff(warm.astype(int))==-1)[0])+[len(warm)-1]
    else:
        mins = list(np.nonzero(np.diff(warm.astype(int))==-1)[0])
    assert len(ones) == len(mins)
    starts = warm.Time.values[ones]
    ends = warm.Time.values[mins]
    return (starts, ends)


def get_bool_from_periods(array, starts, ends):
    """generic: given an array and a bunch of periods within it, retubrn the boolean index of where the array is in those periods.
        Usually meant for times, but can be any increasing array I suppose"""
   
    res = np.full_like(array, False, dtype=bool)
    for s, e in zip(starts, ends):
        res = np.logical_or(res, np.logical_and(array>=s, array<=e))
    return res
        
           
if __name__ == "__main__":
    data_files = get_file_lists()
    size_lims = [0.0025, 0.01, 0.07]
    
    temp_threshs = [5]
    summary_dict = dict() # just for keeping track of a few things like # of particles...
    for i, flight in enumerate(data_files['flights']):
        if not flight=='rf05':
            continue
        print(f'working on flight {flight}')
        flight_info_dict = dict()
        
        #open datasets for this flight
        probe_data = xr.open_dataset(data_files['probe_files'][i])
        flight_data = xr.open_dataset(data_files['flight_files'][i])
        proc_data = add_datetime_to_processed_data(xr.open_dataset(data_files['processed_files'][i]))
        
        #calculation of all boolean filters
        auto_reject_index = np.isin(proc_data.image_auto_reject.values, (48, 72, 104)) # removed 82
        image_center_in_index = proc_data.image_center_in==1
        proc_data["area_ratio"] = proc_data.image_area / (np.pi/4 * proc_data.image_diam_minR ** 2)
        area_ratio_index = proc_data.area_ratio >= 0.2
        iat = np.insert(np.diff(proc_data['Time_in_seconds'].values), 0, 0.0)
        good_iat_index = iat>=0
        proc_data['log10_iat'] = (('time'), np.log10(iat))

        temp_thresh_indices = dict()
        for temp in temp_threshs: # we do this outsize the loop over sizes to avoid needlessly doing it twice
            # the approach here is to use the flight data ATX variable to get the periods where it's warm...
            starts, ends = get_warm_periods(flight_data=flight_data, thresh=temp)
            # and then check which indices from the proc data times fall within one of those periods.
            temp_thresh_indices[temp] = get_bool_from_periods(proc_data.datetime.values, starts, ends)
        size_lim_indices = dict()
        
        #now we loop over both size and temp lists
        for size_lim in size_lims:
            print(f'working on size {size_lim}...')
            size_lim_index = proc_data.image_area>=size_lim  #size filter
            size_info_dict = dict()
            for temp, temp_thresh_index in temp_thresh_indices.items():
                    
                #this big expression ANDS together all filters
                all_index = reduce(np.logical_and, 
                                   [auto_reject_index, image_center_in_index, size_lim_index, temp_thresh_index, area_ratio_index, good_iat_index])
                index_da = proc_data.image_auto_reject.copy(data=all_index)
                count = np.sum(all_index)
                size_info_dict[temp] = count
                subset = proc_data.where(index_da, drop=True)
                savename = f'/home/disk/eos9/jkcm/Data/particle/liquid/liquid_training_data_{flight}.pixcount_{int(size_lim*1e4)}.tempthresh_{temp}C.nc'
                print(f'saving {savename}...')
#                 print('nahhhh')
                subset.to_netcdf(savename)
            flight_info_dict[size_lim] = size_info_dict
        summary_dict[flight] = flight_info_dict
        
    with open('/home/disk/eos9/jkcm/Data/particle/liquid/summary.pickle', 'wb') as f:
        pickle.dump(summary_dict, f, protocol=pickle.HIGHEST_PROTOCOL)