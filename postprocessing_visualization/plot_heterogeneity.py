#!/usr/bin/env conda run -n classified-cset python
# -*- coding: utf-8 -*-
"""plot particle heterogeneity 2d histogram
    Created by Johannes Mohrmann, March 6 2021"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#loading data
temps, flips, counts, full_counts = np.array([]), np.array([]), np.array([]), np.array([])
all_flights = [f'rf{i:02}' for i in np.arange(1,15)]
for f in all_flights:
    with xr.open_dataset(f'/home/disk/eos9/jkcm/Data/particle/heterogeneity/{f}_heterogeneity.nc') as v:
        temps = np.append(temps, v.ATX.values)
        flips = np.append(flips, v.phase_flip_counts.values)
        counts = np.append(counts, v.particle_counts.values)
        full_counts = np.append(full_counts, v.all_particles_counts.values)
good_idx = np.all((~np.isnan(temps), ~np.isnan(flips), ~(counts==0), ~np.isnan(full_counts)), axis=0)
temps, flips, counts, full_counts = np.array(temps)[good_idx], np.array(flips)[good_idx], np.array(counts)[good_idx], np.array(full_counts)[good_idx]
norm_flips = flips/counts
    
    
#plotting data    
fig, ax = plt.subplots()
xbins = 10**np.linspace(0, 5, 20)
ybins = 10**np.linspace(0, 5, 50)

_,_,_,sc = ax.hist2d(full_counts, flips, bins=(xbins, ybins),)
ax.set_xlabel('all particle counts (#/sec)')
ax.set_ylabel('phase flips (#/sec)')
ax.plot([1,1e5],[0.5*1,0.5*1e5], 'w', label='0.5')
ax.plot([1,1e5],[0.02*1,0.02*1e5], '--w', label='0.02')
ax.plot([1,1e5],[0.002*1,0.002*1e5], ':w', label='0.002')
ax.plot([1,1e5],[0.00005*1,0.00005*1e5], '-.w', label='0.00005')

plt.colorbar(sc, ax=ax, label='# of 1Hz samples')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend(title = 'flips per particle')
ax.set_xlim((1, 6e4))
ax.set_ylim((1, 2e3))

fig.savefig('/home/disk/eos12/ratlas/SOCRATES/ML/plots/heterogeneity_histogram.png',dpi=600)
#fig.savefig('/home/disk/p/jkcm/plots/particle/flips per particle plot.png', bbox_inches='tight', dpi=300)
