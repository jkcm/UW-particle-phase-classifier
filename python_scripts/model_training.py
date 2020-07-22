#Author: Jeremy Lu
#Using processed training and validation data, trains a random forest model on paritcle phase claissifcaiotn

import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import datetime
import joblib
from sklearn.inspection import permutation_importance

#load data
ice_tfile="/home/disk/eos15/ijjhsiao/Particle_Research/data/training/ice/ice_data.train.nc"
ice_vfile="/home/disk/eos15/ijjhsiao/Particle_Research/data/training/ice/ice_data.validate.nc"

liquid_tfile="/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.train.nc"
liquid_vfile="/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.validate.nc"

ice_tdata = xr.open_dataset(ice_tfile)
ice_vdata = xr.open_dataset(ice_vfile)

liquid_tdata = xr.open_dataset(liquid_tfile)
liquid_vdata = xr.open_dataset(liquid_vfile)
RSEED = 50

# Create y_train and y_validate
#0 is ice, 1 is liquid
ice_tresults = [0] * len(ice_tdata.image_length)
ice_vresults = [0] * len(ice_vdata.image_length)

liquid_tresults = [1] * len(liquid_tdata.image_length)
liquid_vresults = [1] * len(liquid_vdata.image_length)

y_train = np.concatenate((np.array(ice_tresults), np.array(liquid_tresults)))
y_validate = np.concatenate((np.array(ice_vresults), np.array(liquid_vresults)))

#Create x_train and x_validate
ice_tdata = ice_tdata.drop_dims(['bin_count', 'pos_count'])
liquid_tdata = liquid_tdata.drop_dims(['bin_count', 'pos_count'])

ice_vdata = ice_vdata.drop_dims(['bin_count', 'pos_count'])
liquid_vdata = liquid_vdata.drop_dims(['bin_count', 'pos_count'])

x_train = ice_tdata.to_dataframe().append(liquid_tdata.to_dataframe(),sort=False)
x_validate = ice_vdata.to_dataframe().append(liquid_vdata.to_dataframe(), sort=False)

x_train=x_train.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime','area_hole_ratio'],axis=1)
x_validate=x_validate.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime', 'area_hole_ratio'],axis=1)

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(x_train, y_train)