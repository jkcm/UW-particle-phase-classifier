"""plot map and temperature distribution
    Created by Rachel Atlas"""

#standard
import glob

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, num2date
import numpy as np




fig,axes=plt.subplots(1,2,figsize=(12,5))
m = Basemap(projection='nsper',lat_0=-52,\
    lon_0=147.5,satellite_height=6e5,ax=axes[0])
m.bluemarble()
m.drawparallels(np.arange(-90,-30,5),color='c',linestyle='--')
m.drawmeridians(np.arange(120,180,5),color='c',linestyle='--')
temp_bins=np.arange(-40,25,5)

temp_UWILD=[]
temp_CDP=[]

low_rate_path = \
'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/low_rate_final/'

flight_nos=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]

for flight_no in flight_nos:
  if flight_no<10:
    flight='rf0'+str(flight_no)
  else:
    flight='rf'+str(flight_no)

  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*phase.nc')[0]
  low_rate=Dataset(low_rate_filename)
  lat=low_rate.variables['GGLAT'][:]
  lon=low_rate.variables['GGLON'][:]
  ice=low_rate.variables['UW_ICE_PARTICLES'][:]
  liquid=low_rate.variables['UW_LIQUID_PARTICLES'][:]
  total=ice+liquid
  uwild=np.where(total>0.0)
  temp=low_rate.variables['ATX'][:]
  temp[np.where(temp<-100.)]=np.nan
  temp_UWILD=np.concatenate((temp_UWILD,low_rate.variables['ATX'][uwild]))
  x,y=m(lon,lat)
  axes[0].plot(x,y,color='goldenrod')
  low_rate.close()

cold_sts=[4.+25/60.,47/60.]
cold_ets=[4.+26.8/60.,53/60.]
warm_sts=[4.+50./60.]
warm_ets=[4.+55./60.]
cold_flight_nos=[1,4]
warm_flight_nos=[5]

cold_all=[]
warm_all=[]

for j in range(2):
  flight_no=cold_flight_nos[j]
  flight='rf0'+str(flight_no)
  st=cold_sts[j]
  et=cold_ets[j]
  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*phase.nc')[0]
  low_rate=Dataset(low_rate_filename)
  time=low_rate.variables['Time']
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
  T=low_rate.variables['ATX'][in_situ_sub]
  ice=low_rate.variables['UW_ICE_PARTICLES'][in_situ_sub]
  liquid=low_rate.variables['UW_LIQUID_PARTICLES'][in_situ_sub]
  total=ice+liquid
  uwild=np.where(total>0.0)
  T=T[uwild]
  cold_all=np.concatenate((cold_all,T))

for j in range(1):
  flight_no=warm_flight_nos[j]
  flight='rf0'+str(flight_no)
  st=warm_sts[j]
  et=warm_ets[j]
  low_rate_filename=glob.glob(low_rate_path+'RF'+flight[2:4]+'*phase.nc')[0]
  low_rate=Dataset(low_rate_filename)
  time=low_rate.variables['Time']
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
  T=low_rate.variables['ATX'][in_situ_sub]
  ice=low_rate.variables['UW_ICE_PARTICLES'][in_situ_sub]
  liquid=low_rate.variables['UW_LIQUID_PARTICLES'][in_situ_sub]
  total=ice+liquid
  uwild=np.where(total>0.0)
  T=T[uwild]
  warm_all=np.concatenate((warm_all,T))

axes[1].hist(temp_UWILD,bins=temp_bins,histtype='step',\
             color='k',lw=2,hatch='o')
axes2 = axes[1].twinx()
axes2.hist(cold_all,bins=temp_bins,histtype='stepfilled',\
           color='midnightblue',alpha=.5)
axes2.hist(warm_all,bins=temp_bins,histtype='stepfilled',\
           color='darkred',alpha=.5)
axes[1].hist(temp_UWILD,bins=temp_bins,histtype='step',color='k')
axes[1].set_xlabel('Temperature '+r"($\degree$C)",fontsize=14)
axes[1].set_ylabel('Number of 1Hz data points (All SOCRATES)',fontsize=14)
axes2.set_ylabel('Number of 1Hz data points (TTV set)',fontsize=14)
axes[0].set_title('A. SOCRATES Flight Tracks',fontsize=20)
axes[1].set_title('B. In-cloud temperature distribution',fontsize=20)
axes[1].legend(['All SOCRATES'],loc='upper left',frameon=False)
axes2.legend(['Ice-Dominated','Liquid-Only'],frameon=False)
axes[1].set_ylim(0,17000)
plt.tight_layout()
plt.savefig('figure2.png',dpi=600)



