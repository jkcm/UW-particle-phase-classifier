"""2D histograms of temperature/RH and size
    Created by Rachel Atlas"""

#standard
import pickle

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np



combo_path = \
'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/high_rate/class/'

temp_bins=np.arange(-45,26,1)
area_bins=np.array([56.49, 75., 100., 125., 150., 200., 250., \
                    300., 350., 400., 475., 550., 625., 700., \
                    800., 900., 1000., 1200., 1400., 1600., 1800., \
                    2000., 2200., 2400., 2600., 2800., 3000., \
                    3200.]) / 1000.
area_bins=np.logspace(np.log10(56.49),np.log10(3200))/1000
rh_bins=np.arange(0,200,2)

with open('/home/disk/eos12/ratlas/SOCRATES/ML/save_files/'+\
          'temp_area_hist_sans_rf15','rb') as f:
  Lib=pickle.load(f)
  total_hist=Lib['total_hist']
  liquid_UW_hist=Lib['liquid_UW_hist']
  liquid_EOL_hist=Lib['liquid_EOL_hist']
  liquid_UIOOPS_hist=Lib['liquid_UIOOPS_hist']

with open('/home/disk/eos12/ratlas/SOCRATES/ML/save_files/'+\
          'confidence_temp_area_hist_sans_rf15','rb') as f:
  Lib=pickle.load(f)
  confidence_UW_hist=Lib['liquid_UW_hist']

liquid_UW_hist[np.where(total_hist<100)]=np.nan
liquid_EOL_hist[np.where(total_hist<100)]=np.nan
liquid_UIOOPS_hist[np.where(total_hist<100)]=np.nan
total_hist[np.where(total_hist<100)]=np.nan
confidence_UW_hist[np.where(total_hist<100)]=np.nan

plt.clf()
fig,axes=plt.subplots(5,2,gridspec_kw={'width_ratios':[.452,.548]},sharex=True,figsize=(8,10))
im=axes[0,0].pcolormesh(area_bins[:45],temp_bins[7:-12,],total_hist[7:-12,:45],\
           norm=LogNorm(100,20000),\
           cmap='viridis')
#axes[0,0].plot([.1,.1],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
#axes[0,0].plot([.3,.3],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
axes[0,0].plot([.06,.2,.2,.06,.06],[-10,-10,0,0,-10],'r--')
axes[0,0].plot([.2,.5,.5,.2],[-10,-10,0,0],'r--')
axes[0,0].plot([.5,.5,.1,.1],[0,10,10,0],'r--')
axes[0,0].plot([.2,.5,.5,.2,.2],[-30,-30,-20,-20,-30],'r--')
axes[0,0].plot([.06,.1,.1,.06,.06],[-36,-36,-33,-33,-36],'r--')
axes[0,0].text(.55,3,s='A',color='r',fontsize=16)
axes[0,0].text(.15,-8,s='B',color='r',fontsize=16)
axes[0,0].text(.55,-8,s='C',color='r',fontsize=16)
axes[0,0].text(.55,-28,s='D',color='r',fontsize=16)
axes[0,0].text(.11,-38,s='E',color='r',fontsize=16)
axes[0,0].set_title('Total Particles',fontsize=10)
axes[0,0].set_facecolor('navajowhite')
axes[0,0].set_yticks(ticks=[-30,-20,-10,0,10])
im=axes[1,0].pcolormesh(area_bins[:45],temp_bins[7:-12,],\
           confidence_UW_hist[7:-12,:45]/total_hist[7:-12,:45],\
           vmin=.5,vmax=1.0,cmap='viridis_r')
#axes[1,0].plot([.1,.1],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
#axes[1,0].plot([.3,.3],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
axes[1,0].plot([.06,.2,.2,.06,.06],[-10,-10,0,0,-10],'r--')
axes[1,0].plot([.2,.5,.5,.2],[-10,-10,0,0],'r--')
axes[1,0].plot([.5,.5,.1,.1],[0,10,10,0],'r--')
axes[1,0].plot([.2,.5,.5,.2,.2],[-30,-30,-20,-20,-30],'r--')
axes[1,0].plot([.06,.1,.1,.06,.06],[-36,-36,-33,-33,-36],'r--')
axes[1,0].set_title('Confidence (UWILD)',fontsize=10)
axes[1,0].set_facecolor('navajowhite')
axes[1,0].set_yticks(ticks=[-30,-20,-10,0,10])
im=axes[2,0].pcolormesh(area_bins[:45],temp_bins[7:-12],\
           liquid_UW_hist[7:-12,:45]/total_hist[7:-12,:45],cmap='seismic',vmin=0.0,vmax=1.0)
#axes[2,0].plot([.1,.1],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
#axes[2,0].plot([.3,.3],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
axes[2,0].set_facecolor('navajowhite')
axes[2,0].plot([.06,.2,.2,.06,.06],[-10,-10,0,0,-10],'c--')
axes[2,0].plot([.2,.5,.5,.2],[-10,-10,0,0],'c--')
axes[2,0].plot([.5,.5,.1,.1],[0,10,10,0],'c--')
axes[2,0].plot([.2,.5,.5,.2,.2],[-30,-30,-20,-20,-30],'c--')
axes[2,0].plot([.06,.1,.1,.06,.06],[-36,-36,-33,-33,-36],'c--')
axes[2,0].set_title('Liquid Fraction (UWILD)',fontsize=10)
axes[2,0].set_yticks(ticks=[-30,-20,-10,0,10])
im=axes[3,0].pcolormesh(area_bins[:45],temp_bins[7:-12],\
           liquid_EOL_hist[7:-12,:45]/total_hist[7:-12,:45],cmap='seismic',vmin=0.0,vmax=1.0)
#axes[3,0].plot([.1,.1],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
#axes[3,0].plot([.3,.3],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
axes[3,0].plot([.06,.2,.2,.06,.06],[-10,-10,0,0,-10],'c--')
axes[3,0].plot([.2,.5,.5,.2],[-10,-10,0,0],'c--')
axes[3,0].plot([.5,.5,.1,.1],[0,10,10,0],'c--')
axes[3,0].plot([.2,.5,.5,.2,.2],[-30,-30,-20,-20,-30],'c--')
axes[3,0].plot([.06,.1,.1,.06,.06],[-36,-36,-33,-33,-36],'c--')
axes[3,0].set_title('Liquid Fraction (Area Ratio)',fontsize=10)
axes[3,0].set_facecolor('navajowhite')
axes[3,0].set_yticks(ticks=[-30,-20,-10,0,10])
im=axes[4,0].pcolormesh(area_bins[:45],temp_bins[7:-12],\
           liquid_UIOOPS_hist[7:-12,:45]/total_hist[7:-12,:45],cmap='seismic',vmin=0.0,vmax=1.0)
#axes[4,0].plot([.1,.1],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
#axes[4,0].plot([.3,.3],[temp_bins[7],temp_bins[-13]],'orange',linestyle='-')
axes[4,0].set_facecolor('navajowhite')
axes[4,0].plot([.06,.2,.2,.06,.06],[-10,-10,0,0,-10],'c--')
axes[4,0].plot([.2,.5,.5,.2],[-10,-10,0,0],'c--')
axes[4,0].plot([.5,.5,.1,.1],[0,10,10,0],'c--')
axes[4,0].plot([.2,.5,.5,.2,.2],[-30,-30,-20,-20,-30],'c--')
axes[4,0].plot([.06,.1,.1,.06,.06],[-36,-36,-33,-33,-36],'c--')
axes[4,0].set_title('Liquid Fraction (Holroyd)',fontsize=10)
axes[0,0].set_ylabel('Temperature (C)')
axes[4,0].set_xlabel('Area-equivalent diameter (mm)')
axes[4,0].set_yticks(ticks=[-30,-20,-10,0,10])
axes[0,0].set_xscale('log')

with open('/home/disk/eos12/ratlas/SOCRATES/ML/save_files/'+\
          'rh_diao_area_hist_sans_rf15','rb') as f:
  Lib=pickle.load(f)
  total_hist=Lib['total_hist']
  liquid_UW_hist=Lib['liquid_UW_hist']
  liquid_EOL_hist=Lib['liquid_EOL_hist']
  liquid_UIOOPS_hist=Lib['liquid_UIOOPS_hist']

with open('/home/disk/eos12/ratlas/SOCRATES/ML/save_files/'+\
          'confidence_RH_area_hist_sans_rf15','rb') as f:
  Lib=pickle.load(f)
  confidence_UW_hist=Lib['liquid_UW_hist']

liquid_UW_hist[np.where(total_hist<100)]=np.nan
liquid_EOL_hist[np.where(total_hist<100)]=np.nan
liquid_UIOOPS_hist[np.where(total_hist<100)]=np.nan
total_hist[np.where(total_hist<100)]=np.nan
confidence_UW_hist[np.where(total_hist<100)]=np.nan

im=axes[0,1].pcolormesh(area_bins[:45],rh_bins[10:-30],total_hist[10:-30,:45],\
           norm=LogNorm(100,20000),\
           cmap='viridis')
axes[0,1].set_title('Total Particles',fontsize=10)
plt.colorbar(im,ax=axes[0,1])
axes[0,1].set_facecolor('navajowhite')
#axes[0,1].plot([.1,.1],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
#axes[0,1].plot([.3,.3],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
im=axes[1,1].pcolormesh(area_bins[:45],rh_bins[10:-30,],\
           confidence_UW_hist[10:-30,:45]/total_hist[10:-30,:45],\
           vmin=.5,vmax=1.0,cmap='viridis_r')
axes[1,1].set_title('Confidence (UWILD)',fontsize=10)
plt.colorbar(im,ax=axes[1,1])
axes[1,1].set_facecolor('navajowhite')
#axes[1,1].plot([.1,.1],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
#axes[1,1].plot([.3,.3],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
im=axes[2,1].pcolormesh(area_bins[:45],rh_bins[10:-30],\
           liquid_UW_hist[10:-30,:45]/total_hist[10:-30,:45],cmap='seismic',vmin=0.0,vmax=1.0)
axes[2,1].set_facecolor('navajowhite')
#axes[2,1].plot([.1,.1],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
#axes[2,1].plot([.3,.3],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
plt.colorbar(im,ax=axes[2,1])
axes[2,1].set_title('Liquid Fraction (UWILD)',fontsize=10)
im=axes[3,1].pcolormesh(area_bins[:45],rh_bins[10:-30],\
           liquid_EOL_hist[10:-30,:45]/total_hist[10:-30,:45],cmap='seismic',vmin=0.0,vmax=1.0)
plt.colorbar(im,ax=axes[3,1])
axes[3,1].set_title('Liquid Fraction (Area Ratio)',fontsize=10)
axes[3,1].set_facecolor('navajowhite')
#axes[3,1].plot([.1,.1],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
#axes[3,1].plot([.3,.3],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
im=axes[4,1].pcolormesh(area_bins[:45],rh_bins[10:-30],\
           liquid_UIOOPS_hist[10:-30,:45]/total_hist[10:-30,:45],cmap='seismic',vmin=0.0,vmax=1.0)
axes[4,1].set_facecolor('navajowhite')
plt.colorbar(im,ax=axes[4,1])
axes[4,1].set_title('Liquid Fraction (Holroyd)',fontsize=10)
axes[0,1].set_ylabel('RH (%)')
axes[4,1].set_xlabel('Area-equivalent diameter (mm)')
#axes[4,1].plot([.1,.1],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
#axes[4,1].plot([.3,.3],[rh_bins[10],rh_bins[-31]],'orange',linestyle='-')
axes[0,1].set_xscale('log')

axes[0,0].text(0.98, 0.95, 'a)', transform=axes[0,0].transAxes,
      fontsize=14, va='top', ha='right')
axes[1,0].text(0.98, 0.95, 'b)', transform=axes[1,0].transAxes,
      fontsize=14, va='top', ha='right')
axes[2,0].text(0.98, 0.95, 'c)', transform=axes[2,0].transAxes,
      fontsize=14, va='top', ha='right')
axes[3,0].text(0.98, 0.95, 'd)', transform=axes[3,0].transAxes,
      fontsize=14, va='top', ha='right')
axes[4,0].text(0.98, 0.95, 'e)', transform=axes[4,0].transAxes,
      fontsize=14, va='top', ha='right')
axes[0,1].text(0.98, 0.95, 'f)', transform=axes[0,1].transAxes,
      fontsize=14, va='top', ha='right')
axes[1,1].text(0.98, 0.95, 'g)', transform=axes[1,1].transAxes,
      fontsize=14, va='top', ha='right')
axes[2,1].text(0.98, 0.95, 'h)', transform=axes[2,1].transAxes,
      fontsize=14, va='top', ha='right')
axes[3,1].text(0.98, 0.95, 'i)', transform=axes[3,1].transAxes,
      fontsize=14, va='top', ha='right')
axes[4,1].text(0.98, 0.95, 'j)', transform=axes[4,1].transAxes,
      fontsize=14, va='top', ha='right')

plt.tight_layout()
plt.savefig('figure8.png',dpi=600)
