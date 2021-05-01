"""plot particle heterogeneity 2d histogram
    Created by Johannes Mohrmann, March 6 2021"""

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr


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

gs = GridSpec(5,4, height_ratios=[1,1,1,1,0.2])
fig = plt.figure(figsize=(6.4,6.4))
ax = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3], sharex=ax)
ax_marg_y = fig.add_subplot(gs[1:4,3], sharey=ax)
ax_cb = fig.add_subplot(gs[4, 0:4])


xbins = 10**np.linspace(0, 5, 20)
ybins = 10**np.linspace(0, 3, 50)

_,_,_,sc = ax.hist2d(full_counts, flips, bins=(xbins, ybins),)
ax.set_xlabel('2DS Imaged Particle Count '+r"(s$^{-1}$)",fontsize=14)
ax.set_ylabel('Phase Flips '+r"(s$^{-1}$)",fontsize=14)
for val, ls in zip((0.5, 0.02, 0.002, 0.00005), ('w', '--w', ':w', '-.w')):
    ax.plot([1,1e5],[val,1e5*val], ls, label=str(val))



cb = plt.colorbar(sc, cax=ax_cb, orientation='horizontal')
ax_cb.set_xlabel('Number of 1 Hz samples', fontsize=14)
cb.ax.tick_params(labelsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((1, 5.4e4))
ax.set_ylim((1, 1e3))
ax.legend(title = 'Flips per Particle')
# ax.set_xlim((10, 1e3))

# ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
# ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
ax.xaxis.tick_top()
ax.yaxis.tick_right()


ax_marg_x.hist(full_counts, histtype='step', bins=xbins)
# ax_marg_x.set_yscale('log')
ax_marg_x.set_yticks([0, 2e3, 4e3, 6e3])
ax_marg_x.grid(True, ls='--')
ax_marg_x.set_xlabel('2DS Imaged Particle Count '+r"(s$^{-1}$)",fontsize=14)
ax_marg_x.set_ylabel('Freq',fontsize=14)
ax_marg_x.xaxis.set_label_position('top')

# ax_marg_x.set_yticklabels([1e3, 2e3])



ax_marg_y.hist(flips, histtype='step', orientation="horizontal", bins=ybins)
# ax_marg_y.set_xscale('log')
ax_marg_y.set_xticks([0, 5e2, 1e3, 1.5e3])
ax_marg_y.set_xticklabels([int(i) for i in ax_marg_y.get_xticks()], rotation=45)
ax_marg_y.grid(True, ls='--')
ax_marg_y.set_ylabel('Phase Flips '+r"(s$^{-1}$)",fontsize=14)
ax_marg_y.set_xlabel('Freq',fontsize=14)
ax_marg_y.yaxis.set_label_position('right')
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

plt.tight_layout(h_pad=0, w_pad=0)

fig.savefig(r'/home/disk/p/jkcm/plots/particle/figure11.png', bbox_inches='tight')