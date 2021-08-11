"""plot test data sample periods and PSD
    Created by Rachel Atlas"""

#standard
import glob

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

  low_rate_filename=glob.glob('RF'+flight[2:4]+'*phase.nc')[0]
  phips_search=glob.glob('PhipsData_rf'+flight[2:4]+'.csv')
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

  rf_phase=Dataset(glob.glob(+\
           'UW_particle_classifications.*'+flight+'.nc')[0])
  flag=rf_phase.variables['UW_flag'][:]
  good_values=np.where(flag==0)
  time_rf=rf_phase.variables['datetime']
  time_convert=num2date(time_rf[good_values],time_rf.units)
  rf_phase.close()

  pp_file=Dataset(flight+'/pbp.'+flight+'.2DS.H.nc')
  diam_areaR=pp_file.variables['image_diam_AreaR'][good_values]
  log_iat=np.log10(pp_file.variables['inter_arrival'][good_values])
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
    liq_iat=log_iat[in_situ_sub_pp]

  if j==1:
    ice_deq=diam_areaR[in_situ_sub_pp]
    ice_iat=log_iat[in_situ_sub_pp]

  if j==2:
    ice_deq=np.concatenate((ice_deq,diam_areaR[in_situ_sub_pp]))
    ice_iat=np.concatenate((ice_iat,log_iat[in_situ_sub_pp]))

  wv_search='socrf'+flight[2:4]+'_H2O_25hz_v2_Diao.txt'
  diao=np.genfromtxt(wv_search,delimiter='\t')
  rh_diao=diao[1:,3]
  rhi_diao=diao[1:,4]
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
      pt=int(phips_t[i][11:13])+\
                     int(phips_t[i][14:16])/60+\
                     int(phips_t[i][17:19])/3600
      if (((pt>=hourminsec[0])&(pt<=24))or\
         ((pt>=0)&(pt<=hourminsec[-1]))):
        match=np.argwhere(hourminsec==pt)[0][0]
        T_phips[n]=T[match]
        time_phips[n]=pt
        n=n+1
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
  except:
    ice_index=[]
    liq_index=[]

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
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rh_diao[diao_in_situ_sub],color=colors[j])
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rhi_diao[diao_in_situ_sub],color=colors[j],\
                  linestyle='dotted')
  axes1[1,j].plot(diao_hourminsec[diao_in_situ_sub],rhi_diao[diao_in_situ_sub]*0.0+100.0,\
                  'k--',alpha=.5)
  axes1[1,j].set_ylim(75,140)

  axes1[2,j].plot(hourminsec[in_situ_sub],total[in_situ_sub],color=colors[j])
  axes1[2,j].set_ylim(0,1500)
  axes1[3,j].plot(hourminsec[in_situ_sub],fraction[in_situ_sub],color=colors[j])
  liq=np.where(fraction[in_situ_sub]>.2)
  axes1[3,j].set_ylim(0,1)

  hist=np.histogram(diam_areaR_sub,bins=area_bins)[0]/len(diam_areaR_sub)

axes1[0,1].set_title('RF01 (Ice)')
axes1[0,2].set_title('RF04 (Ice)')
axes1[0,0].set_title('RF05 (Liquid)')
axes1[0,3].set_title('RF12 (Mixed)')
axes1[0,4].set_title('RF03 (Mixed)')

axes1[0,0].set_ylabel('RICE (V)')
axes1[1,0].set_ylabel('RH (%)')
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
fig1.savefig('fig04.png',dpi=600)

fig2,axes2 = plt.subplots(1,2,figsize=(8,3.5))
hist=np.histogram(ice_deq,bins=area_bins)[0]/len(ice_deq)
axes2[1].step(area_bins,np.concatenate(([0],hist)),color='midnightblue',\
           where='pre')
hist=np.histogram(liq_deq,bins=area_bins)[0]/len(liq_deq)
axes2[1].step(area_bins,np.concatenate(([0],hist)),color='darkred',\
           where='pre')
axes2[1].set_xscale('log')
axes2[1].set_ylim(0,0.4)
axes2[1].set_xlim(0.0,1.0)
axes2[1].set_xlabel('area-equivalent diameter (mm)')
axes2[1].set_ylabel('Normalized Frequency')
axes2[1].legend(['Ice-Dominated','Liquid-Only'],frameon=False)
axes2[1].text(.1, .97, 'a)', transform=axes2[1].transAxes,
      fontsize=14, va='top', ha='right')

iat_bins=np.arange(-8.25,0,.25)
iat_bins_mid=iat_bins[:-1]+np.diff(iat_bins)/2.
hist=np.histogram(ice_iat,bins=iat_bins)[0]/len(ice_iat)
axes2[0].step(iat_bins,np.concatenate(([0],hist)),color='midnightblue',\
           where='pre')
hist=np.histogram(liq_iat,bins=iat_bins)[0]/len(liq_iat)
axes2[0].step(iat_bins,np.concatenate(([0],hist)),color='darkred',\
           where='pre')
axes2[0].set_xlim(-7,-1)
axes2[0].set_ylabel('Normalized Frequency')
axes2[0].set_xlabel('log'+r"$_{10}$"+'_iat (log'+r"$_{10}$"+'[s])')
axes2[0].text(.1, .97, 'b)', transform=axes2[0].transAxes,
      fontsize=14, va='top', ha='right')
axes2[0].set_ylim(0,.15)
fig2.tight_layout()
fig2.savefig('fig03.png',dpi=600)

