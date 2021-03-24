#Author: Jeremy Lu, Ian Hsiao, Hans Mohrmann

import xarray as xr
import numpy as np
import pandas as pd
import pickle
import os
import datetime
import glob

def add_datetime_to_processed_data(proc_data):
    """take the 2DS processed dataset and reindex the time properly"""
    secs = (proc_data.Time.values%100) + proc_data.msec.values/1000
    mins = (proc_data.Time.values//100)%100
    hours = proc_data.Time.values//10000
    
    day = proc_data.Date.values%100
    months = (proc_data.Date.values//100)%100
    year = proc_data.Date.values//10000
    datetime = np.array(
        [np.datetime64(f'{year[i]}-{months[i]:02}-{day[i]:02}T{hours[i]:02}:{mins[i]:02}:{secs[i]:06.3f}') for i in range(len(year))])
    proc_data['datetime'] = (['time'], datetime)
    return proc_data

# def add_epoch_time_to_processed_data(proc_data):
#     """take the 2DS processed dataset and reindex the time properly"""
#     msec = proc_data.msec.values
#     secs = proc_data.Time.values%100
#     mins = (proc_data.Time.values//100)%100
#     hours = proc_data.Time.values//10000
    
#     day = proc_data.Date.values%100
#     months = (proc_data.Date.values//100)%100
#     year = proc_data.Date.values//10000
#     epoch = np.array([datetime.datetime(year[i],months[i],day[i],hours[i],mins[i],secs[i],msec[i]*1000).timestamp() for i in range(len(year))])
#     proc_data['epoch'] = (['time'], epoch)
#     return proc_data

# Moving processing code to separate function to make it easier to process multiple flights
def make_dataset_from_processed_UIOPS_data(input_file, output_file, model_file): 
    data = xr.open_dataset(input_file)  # We're going to use this file as the Dataset object to add/remove from. That way we've got the right dimensions.
    
    data = add_datetime_to_processed_data(data)
    # adding in a few extra variables...
    data["area_ratio"] = data.image_area / (np.pi/4 * data.image_diam_minR ** 2)
    iat = np.insert(np.diff(data['Time_in_seconds'].values), 0, .001)
    data['log10_iat'] = (('time',), np.log10(iat))

        
    # construct dataframe for classification
    data = data.drop_dims(['bin_count', 'pos_count'])
    data_df = data.to_dataframe()
#     data_df=data_df.drop(['Date','Time','msec','Time_in_seconds','SliceCount', 'DMT_DOF_SPEC_OVERLOAD',
#                  'Particle_number_all', 'particle_time', 'particle_millisec', 'inter_arrival',
#                  'particle_microsec', 'parent_rec_num', 'particle_num', 'image_longest_y',
#                  'image_auto_reject', 'image_hollow', 'image_center_in', 'image_axis_ratio',
#                  'part_z', 'size_factor', 'holroyd_habit', 'area_hole_ratio', 'datetime'],axis=1)
    
    data_df = data_df[['image_length', 'image_width', 'image_area',
       'image_max_top_edge_touching', 'image_max_bottom_edge_touching',
       'image_touching_edge', 'image_diam_minR', 'image_diam_AreaR',
       'image_perimeter', 'percent_shadow_area', 'edge_at_max_hole',
       'max_hole_diameter', 'fine_detail_ratio', 'area_ratio', 'log10_iat']]
    
    
    # working on filters and flags...
    # filtering for image_auto_reject code, image_center_in, area_ratio_index, good_iat_index, size_index
    #for each filter, 1=fail, 0=pass so for the flag, need ALL to be pass, so 0=OK.
    size_index = data.image_area <= 0.0025
    auto_reject_index = ~np.isin(data.image_auto_reject.values, (48, 72, 104)) # removed 82
    image_center_in_index = data.image_center_in==0
    area_ratio_index = data.area_ratio <= 0.2
    iat_index = iat<=0
    nan_in_df_index = (data_df.apply(np.isnan).any(axis=1)).values
    bad_fine_detail_ratio_index = data.fine_detail_ratio<0
    
    # flag: 0th bit is iat_index, 1st bit=area_ratio_index, 2nd bit=image_center_in_index, 3rd bit=auto_reject_index 4th bit=size_index, 5th bit=nan in dataframe
    # if flag==0, data is good. recover individual flags with flag>>X)&1 where X is the reverse shift
    # what I'm doing is bit-shifting and adding, since we've got bunch of binary flags, then setting the datatype to be an 8-bit int (smallest container)
    flag = (
            (iat_index<<0)+
            (area_ratio_index.values<<1)+
            (image_center_in_index.values<<2)+
            (auto_reject_index<<3)+
            (size_index.values<<4)+
            (nan_in_df_index<<5)+
            (bad_fine_detail_ratio_index.values<<6)
           ).astype("uint8")

    # generating UIOPS and EOL classifications for comparisons
    hollow=data.variables['image_hollow'][:]
    dmax=data.variables['image_diam_minR'][:]*1e3 #Convert from mm to micron
    holroyd=data.variables['holroyd_habit'][:]
    area_ratio=data.variables['area_ratio'][:]

    UIOPS=np.zeros(len(hollow))
    UIOPS[np.where((hollow==1)&(dmax<300.))]=1
    UIOPS[np.where(holroyd>115)]=1

    EOL=np.zeros(len(hollow))
    EOL[np.where(area_ratio>=.5)]=1
    
    #applying our model, but only to rows where the flag is 0
    model = pickle.load(open(model_file,"rb"))
    predictions_g = model.predict(data_df[~flag.astype(bool)])
    probability_g = model.predict_proba(data_df[~flag.astype(bool)]).max(axis=1) # size prediction is whichever probability is max, this is quicker
    
    #then setting those rows in a dummy array to the right values. This way, everything we make is the same length, and we can add them to one tidy Dataset
    predictions = np.full_like(EOL, fill_value=np.nan)
    predictions[~flag.astype(bool)] = predictions_g
    probability = np.full_like(EOL, fill_value=np.nan)
    probability[~flag.astype(bool)] = probability_g
    
    # construct a dataset to save, initialize new dataset with dataArray from old one, so that the time gets copied over. 
    # then add the remaining data variables and save it
    save_data_attrs = {'Contact': 'Johannes Mohrmann (jkcm@uw.edu)',
                       'Institution': 'University of Washington, Dept. of Atmospheric Sciences',
                       'Creation Time': str(datetime.datetime.utcnow()),
                       'Project': 'UW random forest particle phase identification',
                        'website': 'https://github.com/jkcm/UW-particle-phase-classifier',
                       'References': 'random forest relies on output from the UIOPS 2DS processing suite, DOI:10.5281/zenodo.3667054',
                       'authors': 'Rachel Atlas, Joe Finlon, Ian Hsiao, Jeremy Lu, Johannes Mohrmann'}
    
    save_data = xr.Dataset(data_vars={'datetime': data['datetime']}, attrs=save_data_attrs) 
    save_data['UW_flag'] = (('time'), flag, {'Units': '--', 'Description': 'UW data quality flag: 0 is good values, '+
                                                       '0th bit is interarrival time=negative, 1st bit is area ratio index< 0.2, 2nd bit is image center not in particle, '+
                                                       '3rd bit is UIOPS auto_reject index is not in (48, 72, 104), 4th bit is size is above 25 pixels, 5th bit is NaN in '+
                                                       'input data, 6th bit is negative fine detail ratios'})
    save_data['UW_phase'] = (('time'), predictions, {'Units': '0:ice_1:liquid', 'Description': 'UW random forest phase classification, 1=liquid, 0=ice'})
    save_data['UW_certainty'] = (('time'), probability, {'Units': '0-1', 'Description': 'UW phase prediction certainty'})
    save_data['AR_threshold_phase'] = (('time'), EOL, {'Units': '0:ice_1:liquid', 'Description': 'area ratio threshold classification, AR>0.5=liquid'})
    save_data['Holroyd_phase'] = (('time'), UIOPS, {'Units': '0:ice_1:liquid', 'Description': 'Holroyd habit phase classification. Habit=round=liquid, or small with poisson spot=liquid'})

    
    # saving, note the use of compression as well. This makes the difference between a 100 MB file and a 2 MB file!
    comp = dict(zlib=True, complevel=2) 
    save_data.to_netcdf(output_file, engine='h5netcdf', encoding={var: comp for var in save_data.data_vars})
    
    
if __name__ == "__main__":
        #we're going to use glob to get a list of files to work on
#     input_files = glob.glob("/home/disk/eos15/ijjhsiao/Particle_Research/data/procData/proc2DS_H.*.nc")
    
    flights_to_classify = ['rf01', 'rf02', 'rf03', 'rf04', 'rf05', 'rf06', 'rf07', 'rf08', 'rf09', 'rf10', 'rf11', 'rf12', 'rf13', 'rf14', 'rf15']
                           
                            
    
#     model_file = '/home/disk/eos9/jlu43/random_forests/model.0.8751679637015776'
    model_file = '/home/disk/eos9/jlu43/random_forests/model.0.9500382144075897'
        
    for i in flights_to_classify:
        print(i)
        input_file = f"/home/disk/eos9/jfinlon/socrates/{i}/pbp.{i}.2DS.H.nc"
        output_file = f"/home/disk/eos9/jkcm/Data/particle/classified/new/UW_particle_classifications.{i}.nc"
        make_dataset_from_processed_UIOPS_data(input_file, output_file, model_file)
        