import numpy as np
from netCDF4 import Dataset, num2date
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from scipy.special import erfc
import pandas
from datetime import date, datetime, timedelta
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.signal import find_peaks
from subprocess import call

#colors=['midnightblue','midnightblue','midnightblue',\
#        'darkred','indigo','midnightblue','darkred']

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

#4.78,4.95

sts=[4.+25/60.,22.+15/60.,0.+43./60.,4+50./60.,3+29/60.,\
     5.65,5.+57./60.,4.78,18./60]#,7.+3/60.]
ets=[4.+28/60.,22.+30/60.,0.+53./60.,4.+55./60,3+38/60.,\
     5.8,6.0+10./60.,4.95,28./60]#,7.+8/60.]

flight_nos=[1,4,5,12,3]

sts=[4.+25/60.,0.+47./60.,4+50./60.,3+36.5/60.,21.5/60]
ets=[4.+26.8/60.,0.+53./60.,4.+55./60.,3+38/60.,27.5/60]

fig1,axes1 = plt.subplots(4,5,figsize=(8,6.5),sharex='col',sharey='row')
#fig1,axes1 = plt.subplots(6,5,figsize=(8,9),sharex='col',sharey='row')

flight_nos=[5,1,4,12,3]
sts=[4.+50./60.,4.+25/60.,0.+47./60.,3+36.5/60.,21.5/60]
ets=[4.+55./60.,4.+26.8/60.,0.+53./60.,3+38/60.,27.5/60]

for j in range(5):
#for j in [2,0,1,3,4]:


  print(j)

  flight_no=flight_nos[j]

  st=sts[j]
  et=ets[j]

  matplotlib.rcParams.update({'font.size': 10})

  if flight_no<10:
    flight='rf0'+str(flight_no)
  else:
    flight='rf'+str(flight_no)
  print(flight)
  print(st,et)

  low_rate_path = \
  '/home/disk/eos12/ratlas/SOCRATES/data/in-situ/low_rate_final/'

  diao_path = \
  '/home/disk/eos12/ratlas/SOCRATES/data/in-situ/diao_wv/'

  particle_path= \
  '/home/disk/eos9/jfinlon/socrates/'

  phips_path = \
  '/home/disk/eos12/ratlas/SOCRATES/data/in-situ/phips/readable/'

  hetero_path= \
  '/home/disk/eos9/jkcm/Data/particle/heterogeneity/'

  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*phase.nc')[0]
  phips_search=glob.glob(phips_path+'PhipsData_rf'+\
               flight[2:4]+'.csv')
  low_rate=Dataset(low_rate_filename)

  alt=low_rate.variables['GGALT'][:]/1e3
  T=low_rate.variables['ATX'][:]
  pres=low_rate.variables['PSXC'][:]
  T=low_rate.variables['ATX'][:]
  rice=low_rate.variables['RICE'][:]
  plwcc=low_rate.variables['PLWCC'][:]
  time=low_rate.variables['Time']
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
  #print(np.min(hourminsec),np.max(hourminsec))
  #print(in_situ_sub)

  hetero=glob.glob(hetero_path+'*'+flight+'*')[0]
  het_data=Dataset(hetero)
  flip=het_data.variables['phase_flip_counts'][:]
  count=het_data.variables['all_particles_counts'][:]
  particle=het_data.variables['particle_counts'][:]

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

  print('number of particles')
  print(np.shape(in_situ_sub_pp))

  if j==0:
    liq_deq=diam_areaR[in_situ_sub_pp]

  if j==1:
    ice_deq=diam_areaR[in_situ_sub_pp]

  if j==2:
    ice_deq=np.concatenate((ice_deq,diam_areaR[in_situ_sub_pp]))

  wv_search=diao_path+'socrates_RF'+flight[2:4]+'_H2O_25hz_v18_Diao.txt'
  diao=np.genfromtxt(wv_search,delimiter='\t')
  rh_diao=diao[1:,2]
  rhi_diao=diao[1:,3]
  time_diao=diao[1:,0]

  rh_diao[np.where(rh_diao>150)]=np.nan
  rhi_diao[np.where(rhi_diao>150)]=np.nan

  rh_diao=np.nanmean(rh_diao.reshape(-1, 25), axis=1)
  rhi_diao=np.nanmean(rhi_diao.reshape(-1, 25), axis=1)
  time_diao=np.nanmean(time_diao.reshape(-1, 25), axis=1)

  time_convert=num2date(time_diao,time.units)
  hour=np.zeros(len(time_convert))
  minute=np.zeros(len(time_convert))
  second=np.zeros(len(time_convert))

  for index in range(len(time_convert)):
    hour[index]=time_convert[index].hour
    minute[index]=time_convert[index].minute
    second[index]=time_convert[index].second

  diao_hourminsec=hour+minute/60+second/3600
  diao_in_situ_sub=np.squeeze(np.where((diao_hourminsec>st)&\
                        (diao_hourminsec<et)))

  try:
    [droplet_flag,droplet,diameter_C1,dmax_C1,diameter_C2,dmax_C2]=\
    np.transpose(np.loadtxt(phips_search[0],\
    skiprows=1,delimiter=';',usecols=(7,23,29,30,38,39)))

    phips_t=np.genfromtxt(phips_search[0],\
             delimiter=';',usecols=1,dtype='str')[1:]
    T_phips=np.ones((len(phips_t)))
    time_phips=np.zeros((len(phips_t)))

    n=0
    for i in range(len(phips_t)):
      #if phips_t[i][8:10]==phips_t[0][8:10]:
      pt=int(phips_t[i][11:13])+\
                     int(phips_t[i][14:16])/60+\
                     int(phips_t[i][17:19])/3600
      #else:
      #  pt=int(int(phips_t[i][11:13])*3600+\
      #               int(phips_t[i][14:16])*60+\
      #               int(phips_t[i][17:19])+86400)
      if (((pt>=hourminsec[0])&(pt<=24))or\
         ((pt>=0)&(pt<=hourminsec[-1]))):
        #print(pt,np.min(hourminsec),np.max(hourminsec))
        match=np.argwhere(hourminsec==pt)[0][0]
        T_phips[n]=T[match]
        time_phips[n]=pt
        n=n+1
    #print(np.min(pt),np.max(pt))
    #n=0
    #print(time_phips)
    #if j==3:
#    print('j is 3')
    #  subset=np.squeeze(np.where((time_phips>st)&\
    #                             (time_phips<et)&\
    #                             (T_phips>-60.0)))
#    print(phips_t[subset])
    #else:
    subset=np.squeeze(np.where((time_phips>st)&\
                               (time_phips<et)))
    ice_index=np.where(droplet_flag[subset]==0)
    liq_index=np.where(droplet_flag[subset]==1)
    iceman_index=np.where(droplet[subset]==0)
    liqman_index=np.where(droplet[subset]==1)
    ice_flag=np.zeros(len(droplet_flag))
    ice_flag[np.where(droplet_flag==0)]=1

    time_subset=time_phips[subset]
    liq_subset=droplet_flag[subset]
    ice_subset=ice_flag[subset]
 
    print(diameter_C2[ice_index])
    print(diameter_C2[liq_index])
    print('total = ',np.size(subset))
    print('auto liquid = ',np.shape(liq_index))
    print('auto ice = ',np.shape(ice_index))
    print('manual liquid = ',np.shape(liqman_index))
    print('manual ice = ',np.shape(iceman_index))
 
    if j==0:
      rf05_ice=np.size(ice_flag[subset])
      rf05_liq=np.size(droplet_flag[subset])
      rf01_ice=np.size(ice_flag[subset])
      rf01_liq=np.size(droplet_flag[subset])
      rf04_ice=np.size(ice_flag[subset])
      rf04_liq=np.size(droplet_flag[subset])

    avg_n=int((time_subset[-1]-time_subset[0])*3600//60)

    time_avg=np.zeros(avg_n)
    liq_avg=np.zeros(avg_n)
    ice_avg=np.zeros(avg_n)

    for i in range(avg_n):
      time_st=time_subset[0]*3600+60*i
      index=np.squeeze(np.where((time_subset*3600>=time_st)&\
                      (time_subset*3600<time_st+60)))
      try:
        time_avg[i]=(time_subset[0]*3600+60*i+30)/3600
        liq_avg[i]=np.sum(liq_subset[index])
        ice_avg[i]=np.sum(ice_subset[index])
      except:
        time_avg[i]=np.nan
        liq_avg[i]=np.nan
        ice_avg[i]=np.nan
      #print(i,time_avg[i],liq_avg[i],ice_avg[i])
  except:
    ice_index=[]
    liq_index=[]
    print('no phips')

  #if j==3:
#    print('j is 3')
  #  subset=np.squeeze(np.where((time_phips>st*3600)&\
  #                             (time_phips<et*3600)&\
  #                             (T_phips>-60.0)))
#    print(phips_t[subset])
  #else:
  #  subset=np.squeeze(np.where((time_phips>st*3600)&\
  #                             (time_phips<et*3600)))
#
#  print(np.shape(subset))


  #print(liq_index)
 
#
#  if j==3:
#    print('ice index')
#    print(phips_t[ice_index])
#  else:
#    print('liquid index')
#    print(phips_t[liq_index])

  try:
    bins=np.logspace(1,3)
  
    fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
  
    axes[0].plot([60,60],[0,100],'k')
    axes[0].hist(diameter_C1[liq_index],bins=bins,histtype='step')
    axes[0].hist(diameter_C2[liq_index],bins=bins,histtype='step')
    axes[0].hist(dmax_C1[liq_index],bins=bins,histtype='step')
    axes[0].hist(dmax_C2[liq_index],bins=bins,histtype='step')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(['Detection limit','Diameter (cam 1)','Diameter (cam 2)',\
                    'Dmax (cam 1)','Dmax (cam 2)'])
    axes[0].set_title('Liquid Particles')
    axes[0].set_ylim(0,100)
    axes[0].set_xscale('log')
    axes[1].plot([20,20],[0,2100],'k')
    axes[1].hist(diameter_C1[ice_index],bins=bins,histtype='step')
    axes[1].hist(diameter_C2[ice_index],bins=bins,histtype='step')
    axes[1].hist(dmax_C1[ice_index],bins=bins,histtype='step')
    axes[1].hist(dmax_C2[ice_index],bins=bins,histtype='step')
    axes[1].set_xlabel('Particle Size')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Ice Particles')
    plt.tight_layout()
    plt.savefig('/home/disk/eos7/ratlas/SOCRATES/plots/'+\
                'phips_phase_histograms_'+flight+'.png')
  except:
    print('no phips')

  #print(rice[in_situ_sub])
  #axes1[0,j].plot(hourminsec[in_situ_sub],T[in_situ_sub],color=colors[j])
#  axes2 = axes1[0,j].twinx()
#  axes2.plot(hourminsec[in_situ_sub],alt[in_situ_sub],color=colors[j],linestyle='dotted')
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
  #if j==5:
  #  print(np.where(np.ma.getmask(rh_diao[in_situ_sub]==True)))
    #rice_mask=np.ma.getmask(rice[in_situ_sub[0]])
    #print(np.where(rice_mask==True))
    #rh_mask=np.ma.getmask(rice[in_situ_sub[0]])
    #print(np.where(rh_mask==True))
  #  axes1[2,j].fill([hourminsec[in_situ_sub[34]],\
  #                   hourminsec[in_situ_sub[34]],\
  #                   hourminsec[in_situ_sub[139]],
  #                   hourminsec[in_situ_sub[139]]],\
  #                   [50,150,150,50],color='silver')
  axes1[0,j].set_ylim(0,7)
  axes1[0,j].text(hourminsec[in_situ_sub][0],6,s='  '+\
             str(int(np.floor(np.min(T[in_situ_sub]))))+r"$\degree$C < T <"+\
             str(int(np.ceil(np.max(T[in_situ_sub]))))+r"$\degree$C",\
             fontsize=8)
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rh_diao[diao_in_situ_sub],color=colors[j])
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rhi_diao[diao_in_situ_sub],color=colors[j],\
                  linestyle='dotted')
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rhi_diao[diao_in_situ_sub]*0.0+100.0,\
                  'k--',alpha=.5)
  axes1[1,j].set_ylim(75,140)

#  axes1[2,j].set_ylim(0,6)
  axes1[2,j].plot(hourminsec[in_situ_sub],total[in_situ_sub],color=colors[j])
  axes1[2,j].set_ylim(0,1500)
  axes1[3,j].plot(hourminsec[in_situ_sub],fraction[in_situ_sub],color=colors[j])
  liq=np.where(fraction[in_situ_sub]>.2)
  axes1[3,j].set_ylim(0,1)
#  axes1[2,j].set_ylim(50.,150.)

  #axes1[4,j].plot(hourminsec[in_situ_sub],flip[in_situ_sub]/count[in_situ_sub],color=colors[j])

  #axes1[5,j].plot(hourminsec[in_situ_sub],flip[in_situ_sub],color=colors[j])

  print(flight)
  print(flip[in_situ_sub])
  print(particle[in_situ_sub])
  print(count[in_situ_sub])

  hist=np.histogram(diam_areaR_sub,bins=area_bins)[0]/len(diam_areaR_sub)
#
  #axes2.step(area_bins,np.concatenate(([0],hist)),color=colors[j],\
  #           linestyle=lns[j],where='pre')
  #axes2.hist(diam_areaR_sub,bins=area_bins,color=colors[j],\
  #           linestyle=lns[j],histtype='step')
  #try:
  #  liq_avg[np.where(liq_avg==0.0)]=np.nan
  #  ice_avg[np.where(ice_avg==0.0)]=np.nan
  #  axes1[j,2].plot(time_avg,liq_avg,'k*')
  #  axes1[j,2].plot(time_avg,ice_avg,'r*')
  #try:
  #try:
  #  axes1[3,j].hist(diameter_C1[liq_index]+\
  #                  diameter_C2[liq_index],\
  #                  bins=bins,histtype='step')
  #  axes1[3,j].hist(diameter_C1[ice_index]+\
  #                  diameter_C2[ice_index],\
  #                  bins=bins,histtype='step')
  #except:
  #  print('no phips')
axes1[0,1].set_title('RF01 (Ice)')
#axes1[0,1].set_title('RF03 (Ice)')
axes1[0,2].set_title('RF04 (Ice)')
#axes1[0,3].set_title('RF11 (Ice)')
axes1[0,0].set_title('RF05 (Liquid)')
#axes1[0,5].set_title('RF06 (Liquid)')
axes1[0,3].set_title('RF12 (Mixed)')
axes1[0,4].set_title('RF03 (Mixed)')

#axes1[0,0].set_ylabel('Temp. (C)')
#axes2.set_ylabel('Height (km)')

axes1[0,0].set_ylabel('RICE (V)')
axes1[1,0].set_ylabel('RH (%)')
axes1[2,0].set_ylabel('# of Particles (-)')
axes1[3,0].set_ylabel('Liquid/Total (-)')
#axes1[4,0].set_ylabel('flips/particle')
#axes1[5,0].set_ylabel('flips/second')
#axes1[0,2].set_title('Particles per minute from the PHIPS (-)')

axes1[1,0].legend(['RH','RHi'],fontsize=8,\
           frameon=False,loc='upper left')
axes1[3,1].set_xlabel('Hour on 16/01/2018',fontsize=8)
axes1[3,2].set_xlabel('Hour on 24/01/2018',fontsize=8)
axes1[3,0].set_xlabel('Hour on 26/01/2018',fontsize=8)
axes1[3,3].set_xlabel('Hour on 18/02/2018',fontsize=8)
axes1[3,4].set_xlabel('Hour on 23/01/2018',fontsize=8)

fig1.tight_layout()

axes1[0,0].text(0.0, 1.19, 'a)', transform=axes1[0,0].transAxes,
      fontsize=14, va='top', ha='right')
axes1[0,1].text(0.0, 1.19, 'b)', transform=axes1[0,1].transAxes,
      fontsize=14, va='top', ha='right')
axes1[0,2].text(0.0, 1.19, 'c)', transform=axes1[0,2].transAxes,
      fontsize=14, va='top', ha='right')
axes1[0,3].text(0.0, 1.19, 'd)', transform=axes1[0,3].transAxes,
      fontsize=14, va='top', ha='right')
axes1[0,4].text(0.0, 1.19, 'e)', transform=axes1[0,4].transAxes,
      fontsize=14, va='top', ha='right')

#print(np.shape(ice_deq))
#print(np.shape(np.where(ice_deq<.2)))

#axes1[j,2].legend(['Liquid','Ice'])
fig1.savefig('/home/disk/eos7/ratlas/SOCRATES/plots/'+\
             'socrates_ml_train-test-val_set_proposed.png',\
             dpi=600)

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
fig2.savefig('figure3.png',dpi=600)

fig2,axes2 = plt.subplots(2,figsize=(3.5,2.5))
hist=np.histogram(ice_deq,bins=area_bins)[0]/len(ice_deq)
axes[0].
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
axes2.legend(['Ice-Dominated','Liquid-Only'])
fig2.tight_layout()
fig2.savefig('figure4.png',dpi=600)

