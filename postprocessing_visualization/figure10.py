#!/usr/bin/env conda run -n classified-cset python
# -*- coding: utf-8 -*-
"""plot particle PSD
    Created by Johannes Mohrmann"""


import xarray as xr
import numpy as np
# import pandas as pd
# import pickle
# import os
# from joblib import dump, load
# import datetime
import matplotlib.pyplot as plt
import glob

psd_files = sorted([i for i in glob.glob('/home/disk/eos9/jkcm/Data/particle/psd/rf*_psd.nc') if not 'rf15' in i])
flight_files = sorted([i for i in glob.glob('/home/disk/eos9/jfinlon/socrates/*/*.PNI.nc') if not 'RF15' in i])
psds = xr.open_mfdataset(psd_files, combine='by_coords')

ds = []
for f in sorted(flight_files):
    flights = xr.open_dataset(f)
    flights = flights.drop('PSTFC', errors='ignore')
    ds.append(flights)
flights = xr.concat(ds, dim='Time', data_vars=['ATX'], compat="no_conflicts")
flights = flights.rename({'Time': 'time'})

fig, axg = plt.subplots(figsize=(15, 10), nrows=2, ncols=3, sharex=True, sharey=True)
axl = axg.flatten()

all_cloud = psds.count_darea_all.sum(dim='size_bin')>10

temp_ranges = [-40, -20, -5, 0, 5, 40, 100]
for i in range(len(temp_ranges)-1):
    tmin = temp_ranges[i]
    tmax = temp_ranges[i+1]
    if tmax==100:
        tmin = -100
    ax = axl[i]
    idx = np.logical_and(flights.ATX>tmin, flights.ATX<tmax)
    ds = psds.isel(time=idx)
    
    some_particles = ds.count_darea_all.sum(dim='size_bin')>10
    ds = ds.isel(time=some_particles)
    
    bins = np.exp((np.log(ds.bin_edges.values)[1:]+np.log(ds.bin_edges.values)[:-1])/2)
    log_width = np.log(ds.bin_edges.values)[1:]-np.log(ds.bin_edges.values)[:-1]
    
    
    sv = ds.sample_volume.mean(dim='time').values
#     sv = 1
    ice_ml_hist = ds.count_darea_ice_ml.mean(dim='time').values/sv
    liq_ml_hist = ds.count_darea_liq_ml.mean(dim='time').values/sv
#     ice_ar_hist = ds.count_darea_ice_ar.mean(dim='time').values
#     liq_ar_hist = ds.count_darea_liq_ar.mean(dim='time').values
#     ice_ho_hist = ds.count_darea_ice_holroyd.mean(dim='time').values
#     liq_ho_hist = ds.count_darea_liq_holroyd.mean(dim='time').values
    ice_ml_hist_ml = ds.count_darea_ice_ml_median.mean(dim='time').values/sv
    liq_ml_hist_ml = ds.count_darea_liq_ml_median.mean(dim='time').values/sv
    ice_ml_hist_25 = ds.count_darea_ice_ml_25pct.mean(dim='time').values/sv
    liq_ml_hist_25 = ds.count_darea_liq_ml_25pct.mean(dim='time').values/sv
    ice_ml_hist_75 = ds.count_darea_ice_ml_75pct.mean(dim='time').values/sv
    liq_ml_hist_75 = ds.count_darea_liq_ml_75pct.mean(dim='time').values/sv

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid('True')
    if i in [0, 3, 6]:
#     if True:
        ax.set_ylabel('dN/dlog(D) (cm$^{-3}$)')
    if i in [3, 4, 5]:
#     if True:
        ax.set_xlabel('area-equivalent diameter (mm)')
    ax.plot(bins, ice_ml_hist/log_width, label='ice', c='tab:orange')
    ax.plot(bins, ice_ml_hist_ml/log_width, label='ice (bootstrap)', c='tab:orange', ls='--')
    ax.fill_between(bins, ice_ml_hist_25/log_width, ice_ml_hist_75/log_width, color='tab:orange', alpha=0.2)
#     ax.set_ylim((1e-10, 1e-4))
    ax.set_ylim((1e-7, 1e0))
    ax.plot(bins, liq_ml_hist/log_width, label='liquid', c='tab:blue')
    ax.plot(bins, liq_ml_hist_ml/log_width, label='liquid (bootstrap)', c='tab:blue', ls='--')
    ax.fill_between(bins, liq_ml_hist_25/log_width, liq_ml_hist_75/log_width, color='tab:blue', alpha=0.2)
    if i ==0:
        ax.legend()
    pct = (np.sum(some_particles)/np.sum(all_cloud)).values.item()
    title=f'{tmin}$^\circ$C  to  {tmax}$^\circ$C ({pct:0.1%} of data)'
    print(f'crossover for {tmin}C-{tmax}C: {bins[np.where(ice_ml_hist>liq_ml_hist)[0][0]]:0.2f} mm')
    if tmax==100:
        title = 'All data'
    ax.set_title(title)
for ax in axl:
    ticks = [0.056, 0.1, 0.3, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    
    for v in [0.1, 0.3]:
        ax.axvline(v, ls='--', c='r')
    
fig.savefig(r'/home/disk/p/jkcm/plots/particle/size_dist_by_temp.png', bbox_inches='tight')