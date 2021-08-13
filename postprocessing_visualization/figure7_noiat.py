"""plot F1 scores
    Created by Jeremy Lu, edited by Rachel Atlas"""

#standard
import pickle

#nonstandard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.metrics import f1_score


# model = pickle.load(open('/home/disk/eos9/jlu43/random_forests/model.0.9500382144075897',"rb"))
model = pickle.load(open('/home/disk/p/jkcm/Code/UW-particle-phase-classifier/model_data/model_no_iat',"rb"))

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

oc = x_test
oc['phase']=y_test
small_test_oc = oc.query("image_area >= 0.0025 and image_area < 0.01")
medium_test_oc = oc.query("image_area >= 0.01 and image_area < 0.07")
big_test_oc = oc.query("image_area >= 0.07")
                       
x_test=x_test.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
             'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
             'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
             'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
             'part_z', 'size_factor', 'holroyd_habit','datetime','area_hole_ratio', 'log10_iat'],axis=1)

x_test['phase']=y_test
small_test = x_test.query("image_area >= 0.0025 and image_area < 0.01")
medium_test = x_test.query("image_area >= 0.01 and image_area < 0.07")
big_test = x_test.query("image_area >= 0.07")

small_predictions = model.predict(small_test.drop('phase', axis=1))
medium_predictions = model.predict(medium_test.drop('phase', axis=1))
big_predictions = model.predict(big_test.drop('phase', axis=1))

UWILD_small_ice=np.round(f1_score(small_predictions,small_test_oc.phase,pos_label=0),3)
UWILD_small_liq=np.round(f1_score(small_predictions,small_test_oc.phase,pos_label=1),3)
UWILD_medium_ice=np.round(f1_score(medium_predictions,medium_test_oc.phase,pos_label=0),3)
UWILD_medium_liq=np.round(f1_score(medium_predictions,medium_test_oc.phase,pos_label=1),3)
UWILD_big_ice=np.round(f1_score(big_predictions,big_test_oc.phase,pos_label=0),3)
UWILD_big_liq=np.round(f1_score(big_predictions,big_test_oc.phase,pos_label=1),3)

hollow=small_test_oc['image_hollow'][:]
dmax=small_test_oc['image_diam_minR'][:]*1e3 #Convert from mm to micron
holroyd=small_test_oc['holroyd_habit'][:]
area_ratio=small_test_oc['area_ratio'][:]
 
small_UIOOPS=np.zeros(len(hollow))
small_UIOOPS[np.where((hollow==1)&(dmax<300.))]=1
small_UIOOPS[np.where(holroyd>115)]=1
 
small_EOL=np.zeros(len(hollow))
small_EOL[np.where(area_ratio>=.5)]=1

hollow=medium_test_oc['image_hollow'][:]
dmax=medium_test_oc['image_diam_minR'][:]*1e3 #Convert from mm to micron
holroyd=medium_test_oc['holroyd_habit'][:]
area_ratio=medium_test_oc['area_ratio'][:]

medium_UIOOPS=np.zeros(len(hollow))
medium_UIOOPS[np.where((hollow==1)&(dmax<300.))]=1
medium_UIOOPS[np.where(holroyd>115)]=1

medium_EOL=np.zeros(len(hollow))
medium_EOL[np.where(area_ratio>=.5)]=1

hollow=big_test_oc['image_hollow'][:]
dmax=big_test_oc['image_diam_minR'][:]*1e3 #Convert from mm to micron
holroyd=big_test_oc['holroyd_habit'][:]
area_ratio=big_test_oc['area_ratio'][:]

big_UIOOPS=np.zeros(len(hollow))
big_UIOOPS[np.where((hollow==1)&(dmax<300.))]=1
big_UIOOPS[np.where(holroyd>115)]=1

big_EOL=np.zeros(len(hollow))
big_EOL[np.where(area_ratio>=.5)]=1

UIOOPS_small_ice=np.round(f1_score(small_UIOOPS,small_test_oc.phase,pos_label=0),3)
UIOOPS_small_liq=np.round(f1_score(small_UIOOPS,small_test_oc.phase,pos_label=1),3)
UIOOPS_medium_ice=np.round(f1_score(medium_UIOOPS,medium_test_oc.phase,pos_label=0),3)
UIOOPS_medium_liq=np.round(f1_score(medium_UIOOPS,medium_test_oc.phase,pos_label=1),3)
UIOOPS_big_ice=np.round(f1_score(big_UIOOPS,big_test_oc.phase,pos_label=0),3)
UIOOPS_big_liq=np.round(f1_score(big_UIOOPS,big_test_oc.phase,pos_label=1),3)

EOL_small_ice=np.round(f1_score(small_EOL,small_test_oc.phase,pos_label=0),3)
EOL_small_liq=np.round(f1_score(small_EOL,small_test_oc.phase,pos_label=1),3)
EOL_medium_ice=np.round(f1_score(medium_EOL,medium_test_oc.phase,pos_label=0),3)
EOL_medium_liq=np.round(f1_score(medium_EOL,medium_test_oc.phase,pos_label=1),3)
EOL_big_ice=np.round(f1_score(big_EOL,big_test_oc.phase,pos_label=0),3)
EOL_big_liq=np.round(f1_score(big_EOL,big_test_oc.phase,pos_label=1),3)

labels = ['','','Small','','Medium','','Large']

x = np.arange(3)  # the label locations
width = 0.30  # the width of the bars

fig, axes = plt.subplots(1,2,figsize=(7, 3.5),sharey=True)
rects1 = axes[0].bar(x - width, [UWILD_small_ice,UWILD_medium_ice,UWILD_big_ice], width, label='Model',color='g',hatch='//')
rects2 = axes[0].bar(x, [UIOOPS_small_ice,UIOOPS_medium_ice,UIOOPS_big_ice], width, label='UIOOPS',color='r',hatch='///')
rects3 = axes[0].bar(x + width, [EOL_small_ice,EOL_medium_ice,EOL_big_ice], width, label='EOL',color='b')
rects4 = axes[1].bar(x - width,[UWILD_small_liq,UWILD_medium_liq,UWILD_big_liq], width, label='Model',color='g',hatch='//')
rects5 = axes[1].bar(x,[UIOOPS_small_liq,UIOOPS_medium_liq,UIOOPS_big_liq], width, label='UIOOPS',color='r',hatch='///')
rects6 = axes[1].bar(x + width, [EOL_small_liq,EOL_medium_liq,EOL_big_liq], width, label='EOL',color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
axes[0].set_ylabel('F1 Scores')
axes[0].set_xticklabels(labels)
axes[1].set_xticklabels(labels)
axes[0].set_title('Ice')
axes[1].set_title('Liquid')
axes[0].legend(['UWILD','Holroyd','Area Ratio'],fontsize=7.5,\
               loc='upper left',frameon=False)


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=6.5)


autolabel(rects1,axes[0])
autolabel(rects2,axes[0])
autolabel(rects3,axes[0])
autolabel(rects4,axes[1])
autolabel(rects5,axes[1])
autolabel(rects6,axes[1])

fig.tight_layout()

# plt.savefig('figure7.png', dpi=600)
plt.savefig('figure7_noiat_2.png', dpi=600)

