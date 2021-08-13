#Primary Author: Jeremy Lu
#Plots feature importances of model

#standard
import pickle

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
import xarray as xr


#import model
model = pickle.load(open('/home/disk/eos9/jlu43/random_forests/model.0.9500382144075897',"rb")) 



#load test data
ice_test="/home/disk/eos15/ijjhsiao/New_Particles/data/training/test.nc"
liquid_test="/home/disk/eos9/jkcm/Data/particle/training/liquid/liquid_data.test.nc"

ice_testdt = xr.open_dataset(ice_test)
liquid_testdt = xr.open_dataset(liquid_test)

ice_testdt = ice_testdt.drop_dims(['bin_count', 'pos_count'])
liquid_testdt = liquid_testdt.drop_dims(['bin_count', 'pos_count'])

ice_testr = [0] * len(ice_testdt.image_length)
liquid_testr = [1] * len(liquid_testdt.image_length)
y_test = np.concatenate((np.array(ice_testr), np.array(liquid_testr)))

x_test = ice_testdt.to_dataframe().append(liquid_testdt.to_dataframe())

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
fi=permutation_importance(model, x_test.drop('phase', axis=1), \
                          x_test.phase, n_repeats=10,random_state=42)
sorted_idx = np.argsort(fi.importances_mean)
print(sorted_idx)

fig, ax = plt.subplots(figsize=(6,5.5))
ax.set_xlabel("Permutation Feature importance",fontsize=14)
sorted_idx=sorted_idx[5:]
oresult = permutation_importance(model, x_test.drop('phase', axis=1), x_test.phase, n_repeats=10,
                                random_state=42)

sresult = permutation_importance(model, small_test.drop('phase', axis=1), small_test.phase, n_repeats=10,
                                random_state=42)

mresult = permutation_importance(model, medium_test.drop('phase', axis=1), medium_test.phase, n_repeats=10,
                                random_state=42)

bresult = permutation_importance(model, big_test.drop('phase', axis=1), big_test.phase, n_repeats=10,
                                random_state=42)

ax.plot(np.mean(oresult.importances[sorted_idx].T,axis=0),\
        np.arange(10),'k',marker='s',linestyle='')
ax.plot(np.mean(sresult.importances[sorted_idx].T,axis=0),\
        np.arange(10),'royalblue',marker='o',linestyle='')
ax.plot(np.mean(mresult.importances[sorted_idx].T,axis=0),\
        np.arange(10),'darkgoldenrod',marker='D',linestyle='')
ax.plot(np.mean(bresult.importances[sorted_idx].T,axis=0),\
        np.arange(10),'purple',marker='*',linestyle='',markersize=10)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(y_titles[sorted_idx])
plt.yticks(fontsize=12)
ax.legend(['All Particles','Small Particles',\
           'Medium Particles','Large Particles'],\
           loc="lower right",fontsize=12,frameon=False)
fig.tight_layout()
plt.savefig("figure6.png", dpi=600)
