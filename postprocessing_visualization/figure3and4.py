import numpy as np
from netCDF4 import Dataset, num2date
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subset_flight_table as sft
from scipy import interpolate, stats
from scipy.special import erfc
import pandas
from datetime import date, datetime, timedelta
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.signal import find_peaks
from subprocess import call
import sat_vap_pres

colors=['darkred','midnightblue','midnightblue',\
        'darkorchid','darkorchid']
lns=['-','-','--', '-','--']

Mair=28.97
Mh2o=18.01528
R=8.314

p_bins=np.arange(600,1010,5)
p_bins_mid=p_bins[0:-1]+np.diff(p_bins)/2

flight_nos=[1,4,5,12,3,11,6,8,7]

area_bins=np.logspace(np.log10(56.49),np.log10(3200),20)/1000

sts=[4.+25/60.,22.+15/60.,0.+43./60.,4+50./60.,3+29/60.,\
     5.65,5.+57./60.,4.78,18./60]
ets=[4.+28/60.,22.+30/60.,0.+53./60.,4.+55./60,3+38/60.,\
     5.8,6.0+10./60.,4.95,28./60]

flight_nos=[1,4,5,12,3]

sts=[4.+25/60.,0.+47./60.,4+50./60.,3+36.5/60.,21.5/60]
ets=[4.+26.8/60.,0.+53./60.,4.+55./60.,3+38/60.,27.5/60]

fig1,axes1 = plt.subplots(4,5,figsize=(8,6.5),sharex='col',sharey='row')

flight_nos=[5,1,4,12,3]
sts=[4.+50./60.,4.+25/60.,0.+47./60.,3+36.5/60.,21.5/60]
ets=[4.+55./60.,4.+26.8/60.,0.+53./60.,3+38/60.,27.5/60]

for j in range(5):

  flight_no=flight_nos[j]

  st=sts[j]
  et=ets[j]

  matplotlib.rcParams.update({'font.size': 10})

  if flight_no<10:
    flight='rf0'+str(flight_no)
  else:
    flight='rf'+str(flight_no)

  low_rate_path = \
  '/home/disk/eos12/ratlas/SOCRATES/data/in-situ/low_rate_final/'

  diao_path = \
  '/home/disk/eos12/ratlas/SOCRATES/data/in-situ/diao_wv/'

  particle_path= \
  '/home/disk/eos9/jfinlon/socrates/'

  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*phase.nc')[0]
  low_rate=Dataset(low_rate_filename)

  alt=low_rate.variables['GGALT'][:]/1e3
  T=low_rate.variables['ATX'][:]
  pres=low_rate.variables['PSXC'][:]
  T=low_rate.variables['ATX'][:]
  rice=low_rate.variables['RICE'][:]
  plwcc=low_rate.variables['PLWCC'][:]
  time=low_rate.variables['Time']
  rh=low_rate.variables['RHUM']
  mr=low_rate.variables['MR']
  rhi=low_rate.variables['RHUM']*sat_vap_pres.liquid(T+273.15)/\
      sat_vap_pres.ice(T+273.15)
  ice=low_rate.variables['UW_ICE_PARTICLES'][:]
  liquid=low_rate.variables['UW_LIQUID_PARTICLES'][:]
  total=ice+liquid
  fraction=liquid/total

  time_convert=num2date(time[:],time.units)
  hour=np.zeros(len(time_convert))
  minute=np.zeros(len(time_convert))
  second=np.zeros(len(time_convert))

  for index in range(len(time_convert)):
    hour[index]=time_convert[index].hour
    minute[index]=time_convert[index].minute
    second[index]=time_convert[index].second

  hourminsec=hour+minute/60+second/3600
  in_situ_sub=np.squeeze(np.where((hourminsec>st)&\
                        (hourminsec<et)))

  rf_phase=Dataset(glob.glob('/home/disk/eos9/jkcm/Data/particle/'+\
           'classified/UW_particle_classifications.*'+flight+'.nc')[0])
  flag=rf_phase.variables['UW_flag'][:]
  good_values=np.where(flag==0)
  time_rf=rf_phase.variables['datetime']
  time_convert=num2date(time_rf[good_values],time_rf.units)
  rf_phase.close()

  pp_file=Dataset('/home/disk/eos9/jfinlon/socrates/'+flight+\
   '/pbp.'+flight+'.2DS.H.nc')
  diam_areaR=pp_file.variables['image_diam_AreaR'][good_values]
  pp_file.close()

  hour_pp=np.zeros(len(time_convert))
  minute_pp=np.zeros(len(time_convert))
  second_pp=np.zeros(len(time_convert))

  for index in range(len(time_convert)):
    hour_pp[index]=time_convert[index].hour
    minute_pp[index]=time_convert[index].minute
    second_pp[index]=time_convert[index].second

  hourminsec_pp=hour_pp+minute_pp/60+second_pp/3600
  in_situ_sub_pp=np.squeeze(np.where((hourminsec_pp>st)&\
                        (hourminsec_pp<et)))

  diam_areaR_sub=diam_areaR[in_situ_sub_pp]

  if j==0:
    liq_deq=diam_areaR[in_situ_sub_pp]

  if j==1:
    ice_deq=diam_areaR[in_situ_sub_pp]

  if j==2:
    ice_deq=np.concatenate((ice_deq,diam_areaR[in_situ_sub_pp]))

  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*.nc')[0]

  wv_search=glob.glob(diao_path+'scrf'+flight[2:4]+'*.txt')

  diao=np.genfromtxt(wv_search[0],delimiter='\t')

  wvp_diao=diao[1:,2]

  esl=sat_vap_pres.liquid(T+273.15)
  esi=sat_vap_pres.ice(T+273.15)

  mr_diao=Mh2o/Mair*wvp_diao
  rh_diao=mr_diao/mr*rh/1000.0
  rhi_diao=mr_diao/mr*rh*(esl/esi)/1000.0

  axes1[0,j].plot(hourminsec[in_situ_sub],rice[in_situ_sub],color=colors[j])
  if j==0:
    axes1[0,j].fill([hourminsec[in_situ_sub[0]],\
                     hourminsec[in_situ_sub[0]],\
                     hourminsec[in_situ_sub[58]],
                     hourminsec[in_situ_sub[58]]],\
                     [0,5.5,5.5,0],color='silver')
    axes1[0,j].fill([hourminsec[in_situ_sub[202]],\
                     hourminsec[in_situ_sub[202]],\
                     hourminsec[in_situ_sub[298]],
                     hourminsec[in_situ_sub[298]]],\
                     [0,5.5,5.5,0],color='silver')
    axes1[1,j].fill([hourminsec[in_situ_sub[0]],\
                     hourminsec[in_situ_sub[0]],\
                     hourminsec[in_situ_sub[239]],
                     hourminsec[in_situ_sub[239]]],\
                     [75,140,140,75],color='silver')
  axes1[0,j].set_ylim(0,7)
  axes1[0,j].text(hourminsec[in_situ_sub][0],6,s='  '+\
             str(int(np.floor(np.min(T[in_situ_sub]))))+r"$\degree$C < T <"+\
             str(int(np.ceil(np.max(T[in_situ_sub]))))+r"$\degree$C",\
             fontsize=8)
  axes1[1,j].plot(hourminsec[in_situ_sub],rh_diao[in_situ_sub],color=colors[j])
  axes1[1,j].plot(hourminsec[in_situ_sub],rhi_diao[in_situ_sub],color=colors[j],\
                  linestyle='dotted')
  axes1[1,j].plot(hourminsec[in_situ_sub],rhi_diao[in_situ_sub]*0.0+100.0,\
                  'k--',alpha=.5)
  axes1[1,j].set_ylim(75,140)
  axes1[2,j].plot(hourminsec[in_situ_sub],total[in_situ_sub],color=colors[j])
  axes1[2,j].set_ylim(0,1500)
  axes1[3,j].plot(hourminsec[in_situ_sub],fraction[in_situ_sub],color=colors[j])
  liq=np.where(fraction[in_situ_sub]>.2)
  axes1[3,j].set_ylim(0,1)
axes1[0,1].set_title('RF01 (Ice)')
axes1[0,2].set_title('RF04 (Ice)')
axes1[0,0].set_title('RF05 (Liquid)')
axes1[0,3].set_title('RF12 (Mixed)')
axes1[0,4].set_title('RF03 (Mixed)')

axes1[0,0].set_ylabel('RICE (v)')
axes1[1,0].set_ylabel('RH (-)')
axes1[2,0].set_ylabel('# of Particles (-)')
axes1[3,0].set_ylabel('Liquid/Total (-)')

axes1[1,0].legend(['RH','RHi'],fontsize=8,\
           frameon=False,loc='upper left')
axes1[3,1].set_xlabel('Hour on 16/01/2018',fontsize=8)
axes1[3,2].set_xlabel('Hour on 24/01/2018',fontsize=8)
axes1[3,0].set_xlabel('Hour on 26/01/2018',fontsize=8)
axes1[3,3].set_xlabel('Hour on 18/02/2018',fontsize=8)
axes1[3,4].set_xlabel('Hour on 23/01/2018',fontsize=8)

fig1.tight_layout()
fig1.savefig('figure3.png',dpi=600)

fig2,axes2 = plt.subplots(1,figsize=(3.5,2.5))
hist=np.histogram(ice_deq,bins=area_bins)[0]/len(ice_deq)
axes2.step(area_bins,np.concatenate(([0],hist)),color='midnightblue',\
           where='pre')
hist=np.histogram(liq_deq,bins=area_bins)[0]/len(liq_deq)
axes2.step(area_bins,np.concatenate(([0],hist)),color='darkred',\
           where='pre')
axes2.set_xscale('log')
axes2.set_ylim(0.002,0.6)
axes2.set_xlim(0.0,1.0)
axes2.set_xlabel('area-equivalent diameter (mm)')
axes2.set_ylabel('Normalized Frequency')
axes2.legend(['Ice-Dominated','Liquid-Only'],frameon=False)
fig2.tight_layout()
fig2.savefig('figure4.png',dpi=600)

