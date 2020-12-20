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


#calculate and plot importances
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
s5 = np.percentile([tree.feature_importances_ for tree in model.estimators_], 75, axis=0)
t5 = np.percentile([tree.feature_importances_ for tree in model.estimators_], 25, axis=0)
iqr = s5-t5

feature_names = np.array(['log10_iat','area_ratio', 'fine_detail_ratio', 'max_hole_diameter',
                       'edge_at_max_hole','percent_shadow_area','perimeter','eq_diameter'
                       ,'max_dimension','touching_edge','max_bottom_edge_touching',
                       'max_top_edge_touching','area','width','length'][::-1])

sorted_idx = importances.argsort()


fig, ax = plt.subplots()
# Plot the impurity-based feature importances of the forest
#axes[0].figure()
ax.set_title("Feature importances")
ax.barh(range(15), importances[sorted_idx],
        color="r", xerr=iqr[sorted_idx], align="center")
ax.set_yticklabels(feature_names[sorted_idx])

ax.set_yticks(range(0,15))
plt.show()