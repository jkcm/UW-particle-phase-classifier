#Author: Jeremy Lu, Ian Hsiao

import xarray as xr
import numpy as np
import pandas as pd
import pickle
from joblib import dump, load
import datetime

#takes input files

file = "/home/disk/eos15/ijjhsiao/Particle_Research/data/procData/proc2DS_H.rf01.nc"
data = xr.open_dataset(file)

def add_datetime_to_processed_data(proc_data):
    """take the 2DS processed dataset and reindex the time properly"""
    secs = (proc_data.Time.values%100) + proc_data.msec.values/1000
    mins = (proc_data.Time.values//100)%100
    hours = proc_data.Time.values//10000
    
    day = proc_data.Date.values%100
    months = (proc_data.Date.values//100)%100
    year = proc_data.Date.values//10000
    datetime = np.array(
        [np.datetime64(f'{year[i]}-{months[i]:02}-{day[i]}T{hours[i]:02}:{mins[i]:02}:{secs[i]:06.3f}') for i in range(len(year))])
    proc_data['datetime'] = (['time'], datetime)
    return proc_data

def add_epoch_time_to_processed_data(proc_data):
    """take the 2DS processed dataset and reindex the time properly"""
    msec = proc_data.msec.values
    secs = proc_data.Time.values%100
    mins = (proc_data.Time.values//100)%100
    hours = proc_data.Time.values//10000
    
    day = proc_data.Date.values%100
    months = (proc_data.Date.values//100)%100
    year = proc_data.Date.values//10000
    epoch = np.array([datetime.datetime(year[i],months[i],day[i],hours[i],mins[i],secs[i],msec[i]*1000).timestamp() for i in range(len(year))])
    proc_data['epoch'] = (['time'], epoch)
    return proc_data

#formats data and splits it into good and bad (for the model)

data = add_datetime_to_processed_data(data)
data = add_epoch_time_to_processed_data(data)
data["area_ratio"] = data.image_area / (np.pi/4 * data.image_diam_minR ** 2)
iat = np.insert(np.diff(data['Time_in_seconds'].values), 0, .001)
data['log10_iat'] = (('time',), np.log10(iat))
# flag particles - 1 is good, 0 is too small, -1 is bad
data["flag"] = (('time',), np.ones(len(data.Time)))
data['flag'][data.image_area <= 0.0025] = 0
t_48 = data.image_auto_reject != 48
t_72 = data.image_auto_reject != 72
t_104 = data.image_auto_reject != 104
bad_particles = np.logical_and(np.logical_and(t_48, t_72), t_104)
data['flag'][bad_particles] = -1
data['flag'][iat<=0] = -2
good_data = data.where(data.flag == 1, drop = True)
bad_data = data.where(data.flag != 1, drop = True)


time_g = np.array(good_data.Time)
time_b = np.array(bad_data.Time)

#Perform UIOOPS and EOL classifications

hollow=good_data.variables['image_hollow'][:]
dmax=good_data.variables['image_diam_minR'][:]*1e3 #Convert from mm to micron
holroyd=good_data.variables['holroyd_habit'][:]
area_ratio=good_data.variables['area_ratio'][:]
 
UIOOPS_g=np.zeros(len(hollow))
UIOOPS_g[np.where((hollow==1)&(dmax<300.))]=1
UIOOPS_g[np.where(holroyd>115)]=1
 

EOL_g=np.zeros(len(hollow))
EOL_g[np.where(area_ratio>=.5)]=1

hollow=bad_data.variables['image_hollow'][:]
dmax=bad_data.variables['image_diam_minR'][:]*1e3 #Convert from mm to micron
holroyd=bad_data.variables['holroyd_habit'][:]
area_ratio=bad_data.variables['area_ratio'][:]
 
UIOOPS_b=np.zeros(len(hollow))
UIOOPS_b[np.where((hollow==1)&(dmax<300.))]=1
UIOOPS_b[np.where(holroyd>115)]=1
 
EOL_b=np.zeros(len(hollow))
EOL_b[np.where(area_ratio>=.5)]=1

# perform model classifications and probabilities

good_data = good_data.drop_dims(['bin_count', 'pos_count'])
good_data = good_data.to_dataframe()
good_data=good_data.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime','area_hole_ratio', 'flag'],axis=1)

model = pickle.load(open('/home/disk/eos9/jlu43/random_forests/model.0.8751679637015776',"rb"))
predictions_g = model.predict(good_data)
probability = model.predict_proba(good_data)
i=0
probs_g=np.full(len(predictions_g),None)
for prediction in predictions_g:
    probs_g[i]=probability[i][prediction]
    i=i+1

predictions_b = np.empty(len(bad_data)) * np.nan
probs_b = predictions_b

#Create output netcdf

time= np.concatenate((time_g, time_b),axis=None)
UIOOPS= np.concatenate((UIOOPS_g, UIOOPS_b),axis=None)
EOL= np.concatenate((EOL_g,EOL_b),axis=None)
predictions= np.concatenate((predictions_g,predictions_b),axis=None)
probs= np.concatenate((probs_g,probs_b),axis=None)

output=xr.Dataset(
    {
        "Time" : time,
        "UIOOPS" : UIOOPS,
        "EOL" : EOL,
        "model_pred" : predictions,
        "model_prob" : probs
    })

output=output.sortby(output.Time)

output.to_netcdf(path="/home/disk/eos9/jlu43/class_cdfs/output.nc")
