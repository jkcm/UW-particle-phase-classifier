import numpy as np
from netCDF4 import Dataset, num2date
import glob
from scipy import interpolate, stats
import pickle
import xarray as xr
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

length_TTV=[]
width_TTV=[]
area_TTV=[]
perimeter_TTV=[]
diam_minR_TTV=[]
diam_areaR_TTV=[]
area_ratio_TTV=[]
percent_shadow_area_TTV=[]
max_bottom_edge_touching_TTV=[]
max_top_edge_touching_TTV=[]
edge_at_max_hole_TTV=[]
max_hole_diameter_TTV=[]
fine_detail_ratio_TTV=[]
inter_arrival_TTV=[]

for combo_filename in \
    ['/home/disk/eos15/ijjhsiao/New_Particles/data/training/test.nc',\
     '/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.test.nc',\
     '/home/disk/eos15/ijjhsiao/New_Particles/data/training/train.nc',\
     '/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.train.nc',\
     '/home/disk/eos15/ijjhsiao/New_Particles/data/training/validate.nc',\
     '/home/disk/eos9/jkcm/Data/particle/liquid/liquid_data.validate.nc']:

  pp_file=Dataset(combo_filename)
  length_TTV=out_of_sample=np.concatenate((length_TTV,\
               pp_file.variables['image_length'][:]))
  width_TTV=out_of_sample=np.concatenate((width_TTV,\
              pp_file.variables['image_width'][:]))
  area_TTV=out_of_sample=np.concatenate((area_TTV,\
             pp_file.variables['image_area'][:]))
  perimeter_TTV=out_of_sample=np.concatenate((perimeter_TTV,\
                  pp_file.variables['image_perimeter'][:]))
  diam_minR_TTV=out_of_sample=np.concatenate((diam_minR_TTV,\
                  pp_file.variables['image_diam_minR'][:]))
  diam_areaR_TTV=out_of_sample=np.concatenate((diam_areaR_TTV,\
             pp_file.variables['image_diam_AreaR'][:]))
  area_ratio_TTV=out_of_sample=np.concatenate((area_ratio_TTV,\
             pp_file.variables['area_ratio'][:]))
  print(np.histogram(area_ratio_TTV))
  percent_shadow_area_TTV=out_of_sample=np.concatenate((percent_shadow_area_TTV,\
             pp_file.variables['percent_shadow_area'][:]))
  max_bottom_edge_touching_TTV=out_of_sample=np.concatenate((\
             max_bottom_edge_touching_TTV,\
             pp_file.variables['image_max_bottom_edge_touching'][:]))
  max_top_edge_touching_TTV=out_of_sample=np.concatenate((\
             max_top_edge_touching_TTV,\
             pp_file.variables['image_max_top_edge_touching'][:]))
  edge_at_max_hole_TTV=out_of_sample=np.concatenate((\
             edge_at_max_hole_TTV,\
             pp_file.variables['edge_at_max_hole'][:]))
  max_hole_diameter_TTV=out_of_sample=np.concatenate((max_hole_diameter_TTV,\
             pp_file.variables['max_hole_diameter'][:]))
  fine_detail_ratio_TTV=out_of_sample=np.concatenate((fine_detail_ratio_TTV,\
             pp_file.variables['fine_detail_ratio'][:]))
  inter_arrival_TTV=out_of_sample=np.concatenate((inter_arrival_TTV,\
             pp_file.variables['log10_iat'][:]))

  pp_file.close()

print('loaded the TTV')

combo_path = \
'/home/disk/eos12/ratlas/SOCRATES/data/in-situ/high_rate/class/'

length=[]
width=[]
area=[]
perimeter=[]
diam_minR=[]
diam_areaR=[]
area_ratio=[]
percent_shadow_area=[]
max_bottom_edge_touching=[]
max_top_edge_touching=[]
edge_at_max_hole=[]
max_hole_diameter=[]
fine_detail_ratio=[]
inter_arrival=[]

flight_nos=[1,2,3,4,5,6,7,8,9,1,11,12,13,14]

for flight_no in flight_nos:
  if flight_no<10:
    flight='rf0'+str(flight_no)
  else:
    flight='rf'+str(flight_no)

  print(flight)

  pp_file=Dataset('/home/disk/eos12/ratlas/SOCRATES/data/in-situ/'+\
                  'high_rate/class/'+flight+\
                  '_classifications_and_all_particle_properties.nc')
  length=out_of_sample=np.concatenate((length,\
               pp_file.variables['image_length'][:]))
  width=out_of_sample=np.concatenate((width,\
              pp_file.variables['image_width'][:]))
  area=out_of_sample=np.concatenate((area,\
             pp_file.variables['image_area'][:]))
  perimeter=out_of_sample=np.concatenate((perimeter,\
                  pp_file.variables['image_perimeter'][:]))
  diam_minR=out_of_sample=np.concatenate((diam_minR,\
                  pp_file.variables['image_diam_areaR'][:]))
  diam_areaR=out_of_sample=np.concatenate((diam_areaR,\
             pp_file.variables['image_diam_minR'][:]))
  area_ratio=out_of_sample=np.concatenate((area_ratio,\
             area/(np.pi*(diam_minR/2.)**2.)))
  percent_shadow_area=out_of_sample=np.concatenate((percent_shadow_area,\
             pp_file.variables['percent_shadow_area'][:]))
  max_bottom_edge_touching=out_of_sample=np.concatenate((\
             max_bottom_edge_touching,\
             pp_file.variables['max_bottom_edge_touching']\
             [:]))
  max_top_edge_touching=out_of_sample=np.concatenate((\
             max_top_edge_touching,\
             pp_file.variables['max_top_edge_touching']\
             [:]))
  edge_at_max_hole=out_of_sample=np.concatenate((\
             edge_at_max_hole,\
             pp_file.variables['edge_at_max_hole']\
             [:]))
  max_hole_diameter=out_of_sample=np.concatenate((max_hole_diameter,\
             pp_file.variables['max_hole_diameter'][:]))
  fine_detail_ratio=out_of_sample=np.concatenate((fine_detail_ratio,\
             pp_file.variables['fine_detail_ratio'][:]))
  inter_arrival=out_of_sample=np.concatenate((inter_arrival,\
             np.log10(pp_file.variables['inter_arrival'][:])))

  pp_file.close()

fig,ax=plt.subplots(7,2,figsize=(8,10),sharey=True)

out_of_sample=[]

under=np.squeeze(np.where(length<np.min(length_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(length>np.max(length_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(width<np.min(width_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(width>np.max(width_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(area<np.min(area_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(area>np.max(area_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(perimeter<np.min(perimeter_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(perimeter>np.max(perimeter_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(diam_minR<np.min(diam_minR_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(diam_minR>np.max(diam_minR_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(diam_areaR<np.min(diam_areaR_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(diam_areaR>np.max(diam_areaR_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(area_ratio<np.min(area_ratio_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(area_ratio>np.max(area_ratio_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(percent_shadow_area<np.min(percent_shadow_area_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(percent_shadow_area>np.max(percent_shadow_area_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(max_bottom_edge_touching<np.min(max_bottom_edge_touching_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(max_bottom_edge_touching>np.max(max_bottom_edge_touching_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(max_top_edge_touching<np.min(max_top_edge_touching_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(max_top_edge_touching>np.max(max_top_edge_touching_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(edge_at_max_hole<np.min(edge_at_max_hole_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(edge_at_max_hole>np.max(edge_at_max_hole_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(max_hole_diameter<np.min(max_hole_diameter_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(max_hole_diameter>np.max(max_hole_diameter_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(fine_detail_ratio<np.min(fine_detail_ratio_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(fine_detail_ratio>np.max(fine_detail_ratio_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

under=np.squeeze(np.where(inter_arrival<np.min(inter_arrival_TTV)))
out_of_sample=np.concatenate((out_of_sample,under))
over=np.squeeze(np.where(inter_arrival>np.max(inter_arrival_TTV)))
out_of_sample=np.concatenate((out_of_sample,over))

ax[0,0].set_title('length (pixels)')
ax[0,0].hist(length,color='k',bins=np.arange(0,550,50),histtype='step')
ax[0,0].hist(length,color='b',bins=np.arange(0,550,50),histtype='step')
ax1=ax[0,0].twinx()
ax1.hist(length_TTV,color='k',bins=np.arange(0,550,50),histtype='step')
ax1.set_yscale('log')
ax[0,0].set_ylim(1,2e7)
ax1.set_ylim(1,1.3e5)
ax[0,0].legend(['TTV set','All SOCRATES'],loc='upper right',\
               frameon=False,fontsize=8)
ax[0,0].tick_params(axis='y', colors='b')

ax[1,0].set_title('width (pixels)')
ax[1,0].hist(width,color='b',bins=np.arange(0,165,15),histtype='step')
ax[1,0].tick_params(axis='y', colors='b')

ax1=ax[1,0].twinx()
ax1.hist(width_TTV,color='k',bins=np.arange(0,165,15),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[2,0].set_title('area (mm'+r"$^{2}$)")
ax[2,0].hist(area,color='b',bins=np.arange(0,5.5,.5),histtype='step')
ax[2,0].tick_params(axis='y', colors='b')

ax1=ax[2,0].twinx()
ax1.hist(area_TTV,color='k',bins=np.arange(0,5.5,.5),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[3,0].set_title('perimeter (mm)')
ax[3,0].hist(perimeter,color='b',bins=np.arange(0,27.5,2.5),histtype='step')
ax[3,0].tick_params(axis='y', colors='b')

ax1=ax[3,0].twinx()
ax1.hist(perimeter_TTV,color='k',bins=np.arange(0,27.5,2.5),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[4,0].set_title('max_dimension (mm)')
ax[4,0].hist(diam_minR,color='b',bins=np.arange(0,3.3,.3),histtype='step')
ax[4,0].tick_params(axis='y', colors='b')

ax1=ax[4,0].twinx()
ax1.hist(diam_minR_TTV,color='k',bins=np.arange(0,3.3,.3),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[5,0].set_title('eq_diameter (mm)')
ax[5,0].hist(diam_areaR,color='b',bins=np.arange(0,3.3,.3),histtype='step')
ax[5,0].tick_params(axis='y', colors='b')

ax1=ax[5,0].twinx()
ax1.hist(diam_areaR_TTV,color='k',bins=np.arange(0,3.3,.3),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[6,0].set_title('area_ratio (-)')
ax[6,0].hist(area_ratio,color='b',bins=np.arange(0,3.3,.3),histtype='step')
ax[6,0].tick_params(axis='y', colors='b')

ax1=ax[6,0].twinx()
ax1.hist(area_ratio_TTV,color='k',bins=np.arange(0,3.3,.3),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[0,1].set_title('percent_shadow_area (-)')
ax[0,1].hist(percent_shadow_area,color='b',bins=np.arange(0.0,1.65,.15),histtype='step')
ax[0,1].tick_params(axis='y', colors='b')

ax1=ax[0,1].twinx()
ax1.hist(percent_shadow_area_TTV,color='k',bins=np.arange(0.0,1.65,.15),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[1,1].set_title('max_bottom_edge_touching (pixels)')
ax[1,1].hist(max_bottom_edge_touching,color='b',bins=np.arange(0,385,35),histtype='step')
ax[1,1].tick_params(axis='y', colors='b')

ax1=ax[1,1].twinx()
ax1.hist(max_bottom_edge_touching_TTV,color='k',bins=np.arange(0,385,35),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[2,1].set_title('max_top_edge_touching (pixels)')
ax[2,1].hist(max_top_edge_touching,color='b',bins=np.arange(0,385,35),histtype='step')
ax[2,1].tick_params(axis='y', colors='b')

ax1=ax[2,1].twinx()
ax1.hist(max_top_edge_touching_TTV,color='k',bins=np.arange(0,385,35),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[3,1].set_title('edge_at_max_hole (pixels)')
ax[3,1].hist(edge_at_max_hole,color='b',bins=np.arange(0,165,15),histtype='step')
ax[3,1].tick_params(axis='y', colors='b')

ax1=ax[3,1].twinx()
ax1.hist(edge_at_max_hole_TTV,color='k',bins=np.arange(0,165,15),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[4,1].set_title('max_hole_diameter (pixels)')
ax[4,1].hist(max_hole_diameter,color='b',bins=np.arange(0,165,15),histtype='step')
ax[4,1].tick_params(axis='y', colors='b')

ax1=ax[4,1].twinx()
ax1.hist(max_hole_diameter_TTV,color='k',bins=np.arange(0,165,15),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[5,1].set_title('fine_detail_ratio (pixels)')
ax[5,1].hist(fine_detail_ratio,color='b',bins=np.arange(0,121,11),histtype='step')
ax[5,1].tick_params(axis='y', colors='b')

ax1=ax[5,1].twinx()
ax1.hist(fine_detail_ratio_TTV,color='k',bins=np.arange(0,121,11),histtype='step')
ax1.set_yscale('log')
ax1.set_ylim(1,1.3e5)

ax[6,1].set_title('log'+r"$_{10}$"+'_iat (log'+r"$_{10}$"+'[s])')
ax[6,1].hist(inter_arrival,color='b',bins=np.arange(-7,1.5,.5),histtype='step')
ax[6,1].tick_params(axis='y', colors='b')

ax1=ax[6,1].twinx()
ax1.hist(inter_arrival_TTV,color='k',bins=np.arange(-7,1.5,.5),histtype='step')
ax1.set_yscale('log')

ax[0,0].set_yscale('log')
ax1.set_ylim(1,1.3e5)

fig.tight_layout()
fig.savefig('figurea1.png')

