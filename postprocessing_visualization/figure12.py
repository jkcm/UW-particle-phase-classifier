import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path='/home/disk/eos9/jkcm/Data/particle/heterogeneity/'

files=glob.glob(path+'*nc')

all_flips=[]
all_counts=[]

for fn in files[:-1]:

  data=Dataset(fn)

  flip=data.variables['phase_flip_counts'][:]
  count=data.variables['particle_counts'][:]
  total_count=data.variables['all_particles_counts'][:]
  count_2ds=data.variables['CONC2DSA_2H'][:]
  time=data.variables['Time'][:]
  tas=data.variables['TASX'][:]
  alt=data.variables['GGALT'][:]

  plt.clf()
  fig,ax=plt.subplots(2,sharex=True)
  ax[0].set_title(fn[-21:-17])
  ax[0].plot(time,count,'k')
  ax[0].set_ylabel('classified particles')
  ax[1].plot(time,flip/count,'k')
  ax[1].set_ylabel('flips/classified particles')
  ax[1].set_xlabel('Time')
  ax[1].legend(['2DS EOL','2DS UWILD'])

  index=np.where((flip/count>.2)&(count>100))

  plt.savefig(fn[-21:-17]+'_hetero_ts.png')

  if fn[-21:-17]=='rf03':
    fig,ax=plt.subplots(2,sharex=True)
    index=np.where((time>3600*22+1100)&(time<3600*22+1110))
    ax[0].plot(time[index],count[index],'k')
    ax[0].plot(time[index],total_count[index],'r')
    ax[0].plot(time[index],tas[index],'b')
    ax[1].plot(time[index],flip[index]/total_count[index],'k')
    plt.savefig(fn[-21:-17]+'_hetero_ts_closeup.png')

  if fn[-21:-17]=='rf07':
    fig,ax=plt.subplots(2,sharex=True)
    index=np.where((time>3600*5-700)&(time<3600*5+800))
    ax[0].plot(time[index],count[index],'k')
    ax[0].plot(time[index],tas[index],'b')
    ax[1].plot(time[index],flip[index]/count[index],'k')
    plt.savefig(fn[-21:-17]+'_hetero_ts_closeup.png')

