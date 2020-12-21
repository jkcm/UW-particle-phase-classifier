#Author: Jeremy Lu
#Plots feature importances of model

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

#import model
model = pickle.load(open('/home/disk/eos9/jlu43/random_forests/model.0.9500382144075897',"rb")) 


#TESTING DATA 

ice_test="/home/disk/eos15/ijjhsiao/New_Particles/data/training/test.nc"
liquid_test="/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.test.nc"

ice_testdt = xr.open_dataset(ice_test)
liquid_testdt = xr.open_dataset(liquid_test)

ice_testdt = ice_testdt.drop_dims(['bin_count', 'pos_count'])
liquid_testdt = liquid_testdt.drop_dims(['bin_count', 'pos_count'])

ice_testr = [0] * len(ice_testdt.image_length)
liquid_testr = [1] * len(liquid_testdt.image_length)
y_test = np.concatenate((np.array(ice_testr), np.array(liquid_testr)))

x_test = ice_testdt.to_dataframe().append(liquid_testdt.to_dataframe(),sort=False)

x_test=x_test.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime','area_hole_ratio'],axis=1)

x_test['phase']=y_test
small_test = x_test.query("image_area >= 0.0025 and image_area < 0.01")
medium_test = x_test.query("image_area >= 0.01 and image_area < 0.07")
big_test = x_test.query("image_area >= 0.07")


#calculate and plot importances
y_titles = np.array(['log10_iat','area_ratio', 'fine_detail_ratio', 'max_hole_diameter',
            'edge_at_max_hole','percent_shadow_area','perimeter','eq_diameter',
            'max_dimension','touching_edge','max_bottom_edge_touching',
            'max_top_edge_touching','area','width','length'][::-1])

fi = model.feature_importances_
sorted_idx = np.argsort(fi)

importances = []
for i in sorted_idx:
    importances.append([tree.feature_importances_[i] for tree in model.estimators_])
    
fig, ax = plt.subplots()
ax.set_title("Feature importances")

bp=ax.boxplot(importances, vert=False, 
              labels = y_titles[sorted_idx], showbox=False,
              showcaps=False, showfliers=False, showmeans=True,
              medianprops=dict(color="white"),
              whiskerprops=dict(color="white"),
              meanprops=dict(color="black"));

sresult = permutation_importance(model, small_test.drop('phase', axis=1), small_test.phase, n_repeats=10,
                                random_state=42)

mresult = permutation_importance(model, medium_test.drop('phase', axis=1), medium_test.phase, n_repeats=10,
                                random_state=42)

bresult = permutation_importance(model, big_test.drop('phase', axis=1), big_test.phase, n_repeats=10,
                                random_state=42)


c = "red"
bp1=ax.boxplot(sresult.importances[sorted_idx].T,
           vert=False, labels=['']*15,
           boxprops=dict(color=c),
           capprops=dict(color=c),
           whiskerprops=dict(color=c),
           flierprops=dict(color=c, markeredgecolor=c),
           medianprops=dict(color=c))
c="blue"
bp2=ax.boxplot(mresult.importances[sorted_idx].T,
           vert=False, labels=['']*15,
           boxprops=dict(color=c),
           capprops=dict(color=c),
           whiskerprops=dict(color=c),
           flierprops=dict(color=c, markeredgecolor=c),
           medianprops=dict(color=c))
c="green"
bp3=ax.boxplot(bresult.importances[sorted_idx].T,
           vert=False, labels=['']*15,
           boxprops=dict(color=c),
           capprops=dict(color=c),
           whiskerprops=dict(color=c),
           flierprops=dict(color=c, markeredgecolor=c),
           medianprops=dict(color=c))


ax.legend([bp["means"][0],bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0]], 
          ['overall','small', 'medium','large'], title="Particle Size", loc="best")
fig.tight_layout()
plt.savefig("all_feature_importances.png", dpi=400)
plt.show()