import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import datetime as dt
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset

# functions
def im2dec(arr):
    """takes in an n-by-8 array of 16-bit compressed representation of 2ds pixels and returns an n-by-128 array 
    of the expanded visual field
    """
    def dec2binarr(arr):
        return np.array(list(''.join([bin(int(i))[2:].zfill(16) for i in arr]))).astype(int)
    
    assert arr.shape[1] == 8
    return np.array([dec2binarr(i) for i in arr])


def plot_random_from_flight(random_seed,image_area,image_diam_AreaR,UW,UW_certainty,AR_threshold_pase,Holroyd_phase,T,time,parent_rec_num,position,flight,size_min,size_max,T_min,T_max,RH_min,RH_max,seglenx,segleny,numrows,numcols,phase_no=0,apply_phase_selection=False,confidence=0.0,apply_conf_selection=False):
    size_match=np.where((image_diam_AreaR>size_min)&(image_diam_AreaR<size_max))   
    T_match=np.where((T>T_min)&(T<T_max)) 
    RH_match=np.where((T>T_min)&(T<T_max)) 

    possibilities=np.intersect1d(RH_match,np.intersect1d(size_match,T_match))
    if apply_phase_selection:
        phase_match=np.where(UW==phase_no) 
        possibilities=np.intersect1d(possibilities,phase_match)
        phase_match=np.where(AR_threshold_phase==phase_no) 
        possibilities=np.intersect1d(possibilities,phase_match)
        phase_match=np.where(Holroyd_phase==phase_no) 
        possibilities=np.intersect1d(possibilities,phase_match)

    if apply_conf_selection:
      conf_match=np.where(UW_certainty>confidence)
      possibilities=np.intersect1d(conf_match,possibilities)

    np.random.seed(random_seed)  
 
    random_selection = np.random.choice(possibilities, \
                       numrows*numcols, replace=False)

    random_selection = random_selection[np.flip(np.argsort(\
                       UW_certainty[random_selection]))]

    fig = plt.figure(constrained_layout=True,figsize=(6.5*seglenx/100,\
                     4*segleny/100))

    spec = fig.add_gridspec(nrows=numrows, ncols=numcols,\
           width_ratios=np.ones(numcols))

    for i in np.arange(numrows*numcols):

        next_particle=random_selection[i]

        flight_selection=flight[next_particle]

        color='Greys_r'

        probe_path = \
        '/home/disk/eos9/jfinlon/socrates/'
        particle_frame = int(parent_rec_num[next_particle])
        [slice_start, slice_end] = position[next_particle]
        probe_data=Dataset(probe_path+flight_selection+'/img.'+flight_selection+'.2DS.H.nc')
        image_data=probe_data.variables['data'][int(particle_frame-1),int(slice_start-1):int(slice_end),:]
        plot_data=~im2dec(image_data)
        aspect_ratio=probe_data.variables['tasRatio'][int(particle_frame-1)]
        pixels=np.where(plot_data==-2)
        bottom=np.min(pixels[0])
        left=np.min(pixels[1])
        width=np.max(pixels[1])-np.min(pixels[1])
        axi=fig.add_subplot(spec[i//numcols,i%numcols])
        axi.imshow(plot_data[int(bottom):,int(left):int(left+\
                   width+1)], cmap=color)
        axi.set_aspect(1/aspect_ratio)
        plt.plot([0,seglenx],[0,0],'k',alpha=.2)
        plt.plot([0,0],[0,segleny*(aspect_ratio)],'k',alpha=.2)
        plt.title(str(UW_certainty[next_particle]),fontsize=8)
        if AR_threshold_phase[next_particle]==0.0:
          plt.text(seglenx*.65,segleny*(aspect_ratio)*.95,s='I',\
                   color='b',fontsize=8)
        else:
          plt.text(seglenx*.65,segleny*(aspect_ratio)*.95,s='L',\
                   color='b',fontsize=8)
        if Holroyd_phase[next_particle]==0.0:
          plt.text(seglenx*.9,segleny*(aspect_ratio)*.95,s='I',\
                   color='r',fontsize=8)
        else:
          plt.text(seglenx*.9,segleny*(aspect_ratio)*.95,s='L',\
                   color='r',fontsize=8)
        if UW[next_particle]==0.0:
          plt.text(seglenx*.4,segleny*(aspect_ratio)*.95,s='I',\
                   color='g',fontsize=8)
        else:
          plt.text(seglenx*.4,segleny*(aspect_ratio)*.95,s='L',\
                   color='g',fontsize=8)

        axi.axis('off')


        i=i+1

    return 

combo_path = \
'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/high_rate/class/'

combo_filename=combo_path+'all_classifications_and_particle_properties.nc'
combo=Dataset(combo_filename)
image_area=combo.variables['image_area'][:]
image_diam_AreaR=combo.variables['image_diam_AreaR'][:]
UW=combo.variables['UW_phase'][:]
UW_certainty=combo.variables['UW_certainty'][:]
AR_threshold_phase=combo.variables['AR_threshold_phase'][:]
Holroyd_phase=combo.variables['Holroyd_phase'][:]
T=combo.variables['Temp_c'][:]
time=combo.variables['Time'][:]
parent_rec_num=combo.variables['parent_rec_num'][:]
position=combo.variables['position'][:]
flight=combo.variables['flight'][:]

confidences=[]
indices=[]
UW_phases=[]
AR_phases=[]
Holroyd_phases=[]

fig = plt.figure(constrained_layout=True,figsize=(7.5*seglenx/100,\
                 6*segleny/100))
plot_random_from_flight(1207479957,image_area,image_diam_AreaR,UW,UW_certainty,\
AR_threshold_phase,Holroyd_phase,T,time,\
parent_rec_num,position,flight,size_min=.1,size_max=.5,T_min=0,T_max=10,\
RH_min=0,RH_max=300,seglenx=50,segleny=50,numrows=5,numcols=10,phase_no=0,\
apply_phase_selection=False)
plt.savefig('campaign_size_p1_p5_temperature_0_10.png',transparent=True,dpi=600)

plot_random_from_flight(1207479957,image_area,image_diam_AreaR,UW,UW_certainty,\
AR_threshold_phase,Holroyd_phase,T,time,\
parent_rec_num,position,flight,size_min=.06,size_max=.2,T_min=-10,T_max=0,\
RH_min=0,RH_max=300,seglenx=50,segleny=50,numrows=5,numcols=10,phase_no=0,\
apply_phase_selection=False)
plt.savefig('campaign_size_p06_p2_temperature_neg10_neg0.png',transparent=True,dpi=600)

plot_random_from_flight(1207479957,image_area,image_diam_AreaR,UW,UW_certainty,\
AR_threshold_phase,Holroyd_phase,T,time,\
parent_rec_num,position,flight,size_min=.2,size_max=.5,T_min=-10,T_max=0,\
RH_min=0,RH_max=300,seglenx=60,segleny=100,numrows=5,numcols=9,phase_no=0,\
apply_phase_selection=False)
plt.savefig('campaign_size_p2_p5_temperature_neg10_neg0.png',transparent=True,dpi=600)

plot_random_from_flight(1207479957,image_area,image_diam_AreaR,UW,UW_certainty,\
AR_threshold_phase,Holroyd_phase,T,time,\
parent_rec_num,position,flight,size_min=.2,size_max=.5,T_min=-30,T_max=-20,\
RH_min=0,RH_max=300,seglenx=60,segleny=100,numrows=5,numcols=9,phase_no=0,\
apply_phase_selection=False)
plt.savefig('campaign_size_p2_p5_temperature_neg30_neg20.png',transparent=True,dpi=600)

plot_random_from_flight(3181286496,image_area,image_diam_AreaR,UW,UW_certainty,\
AR_threshold_phase,Holroyd_phase,T,time,\
parent_rec_num,position,flight,size_min=.06,size_max=.1,T_min=-36,T_max=-33,\
RH_min=0,RH_max=300,seglenx=50,segleny=50,numrows=5,numcols=10,phase_no=1,\
apply_phase_selection=False,confidence=.9,apply_conf_selection=True)

plt.savefig('campaign_size_p06_p1_temperature_neg36_neg33.png',transparent=True,dpi=600)


