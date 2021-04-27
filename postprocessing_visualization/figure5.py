#Primary Author: Ian Hsiao

import pickle
import xarray as xr
import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from joblib import dump, load
from scipy import stats
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


directory = "/home/disk/eos15/ijjhsiao/New_Particles/data"
# load model
model = pickle.load(open('/home/disk/eos9/jlu43/random_forests/model.0.9500382144075897',"rb"))

#load data
ice_vfile=f"/{directory}/training/test.nc"
liquid_vfile="/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.test.nc"

ice_vdata = xr.open_dataset(ice_vfile)
liquid_vdata = xr.open_dataset(liquid_vfile)

# Create y_test
#0 is ice, 1 is liquid
ice_vresults = [0] * len(ice_vdata.image_length)
liquid_vresults = [1] * len(liquid_vdata.image_length)

y_test = np.concatenate((np.array(ice_vresults), np.array(liquid_vresults)))

#Create x_test
ice_vdata = ice_vdata.drop_dims(['bin_count', 'pos_count'])
liquid_vdata = liquid_vdata.drop_dims(['bin_count', 'pos_count'])

x_test = ice_vdata.to_dataframe().append(liquid_vdata.to_dataframe())

x_test=x_test.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime', 'area_hole_ratio'],axis=1)


subset = x_test
subset_t_labels = y_test

prob = model.predict_proba(subset)[:,1]
acc, conf = calibration_curve(subset_t_labels, prob, n_bins=20)
prob[np.where(prob<.5)]=1-prob[np.where(prob<.5)]

fig,ax=plt.subplots(2,gridspec_kw={'height_ratios':[.8,.2]},\
                    sharex=True,figsize=(6.4,6.4))

# make 1-1 line
x = np.linspace(0,1,100)
ax[0].plot(x,x,"--",lw=2,label = "1:1 line",color='b')
# plot confidence curve
ax[0].plot(conf, acc,lw=2,label = "Calibration Curve",color='k')
# plot settings
ax[0].set_title("Calibration Curve",fontsize=14)
ax[0].set_ylabel("Model Accuracy",fontsize=14)
ax[0].set_xlim(0.5,1)
ax[0].set_ylim(0.5,1)
ax[0].legend(loc="best",frameon=False,fontsize=14)
ax[1].plot([.5,1.0],[1e3,1e3],'k--',alpha=.3)
ax[1].plot([.5,1.0],[1e4,1e4],'k--',alpha=.3)
ax[1].hist(prob,bins=20,histtype='step',color='b')
ax[1].set_ylim(100,20000)
ax[1].set_yscale('log')
ax[1].set_xlabel("Model Confidence",fontsize=14)
ax[1].set_ylabel('Freq',fontsize=14)
plt.tight_layout()
plt.savefig("/home/disk/eos12/ratlas/SOCRATES/ML/plots/figure5.png", dpi=600)

