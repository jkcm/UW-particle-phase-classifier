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

fig,ax=plt.subplots(2,gridspec_kw={'height_ratios':[.8,.2]},\
                    sharex=True,figsize=(6.4,6.4))
#fig, ax = plt.subplots()
xbins = 10**np.linspace(0, 5, 20)
ybins = 10**np.linspace(0, 5, 50)

_,_,_,sc = ax[0].hist2d(full_counts, flips, bins=(xbins, ybins),)
ax[0].set_ylabel('Phase Flips '+r"(s$^{-1}$)",fontsize=14)
ax[0].plot([1,1e5],[0.5*1,0.5*1e5], 'w', label='0.5')
ax[0].plot([1,1e5],[0.02*1,0.02*1e5], '--w', label='0.02')
ax[0].plot([1,1e5],[0.002*1,0.002*1e5], ':w', label='0.002')
ax[0].plot([1,1e5],[0.00005*1,0.00005*1e5], '-.w', label='0.00005')

print(np.histogram(full_counts,bins=xbins))

cb=plt.colorbar(sc, ax=ax, label='Number of 1 Hz samples')
cb.ax.tick_params(labelsize=14)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend(title = 'Flips per Particle')
ax[0].set_xlim((1, 5.4e4))
ax[0].set_ylim((1, 2e3))
ax[1].hist(full_counts,bins=xbins,histtype='step',color='b')
ax[1].set_xlabel('2DS Imaged Particles '+r"(s$^{-1}$)",fontsize=14)
ax[1].set_ylabel('Freq',fontsize=14)
fig.savefig('/home/disk/eos12/ratlas/SOCRATES/ML/plots/heterogeneity_histogram.png',dpi=600)
#fig.savefig('/home/disk/p/jkcm/plots/particle/flips per particle plot.png', bbox_inches='tight', dpi=300)
