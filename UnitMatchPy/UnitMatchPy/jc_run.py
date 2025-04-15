# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:59:52 2025

@author: labadmin
"""

import os
from pathlib import Path

import UnitMatchPy.extract_raw_data as erd
import numpy as np 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd

def call_UM_extract(meta_path, phy_dir, extract_good_units_only):
#Extract the units
    spike_width = 82
    sample_amount = 1000
    KS4_data = True
    samples_before=40
    samples_after = spike_width - samples_before
    half_width = np.floor(spike_width/2).astype(int)
    max_width = np.floor(spike_width/2).astype(int) #Size of area at start and end of recording to ignore to get only full spikes
    

    ks_path_list = list()
    ks_path_list.append(phy_dir)
    spike_ids_arr, spike_times_arr, dummy = erd.extract_KS_data(ks_path_list, False)
    qm_str = 'bc_call'
    call_path, good_units = create_call_tsv(phy_dir, qm_str)
    print(good_units)
    spike_ids = spike_ids_arr[0]
    spike_times = spike_times_arr[0]

    if extract_good_units_only:
    
        #load metadata
        meta_data = erd.read_meta(Path(meta_path))
        n_elements = int(meta_data['fileSizeBytes']) / 2
        n_channels_tot = int(meta_data['nSavedChans'])
        aplfsy_str = meta_data['snsApLfSy']
        n_channels = int(aplfsy_str.split(sep=',')[0])
        

        #create memmap to raw data, for that session
        data_path = meta_path.with_suffix('.bin')
        data = np.memmap(data_path, dtype = 'int16', shape =(int(n_elements / n_channels_tot), n_channels_tot))

        # Remove spike which won't have a full waveform recorded
        spike_ids_tmp = np.delete(spike_ids, np.logical_or( (spike_times < max_width), ( spike_times > (data.shape[0] - max_width))))
        spike_times_tmp = np.delete(spike_times, np.logical_or( (spike_times < max_width), ( spike_times > (data.shape[0] - max_width))))

        #might be slow extracting sample for good units only?
        sample_idx = erd.get_sample_idx(spike_times_tmp, spike_ids_tmp, sample_amount, units = good_units)

        if KS4_data:
            avg_waveforms = Parallel(n_jobs = -1, verbose = 10, mmap_mode='r', max_nbytes=None )\
            (delayed(erd.extract_a_unit_KS4)\
            (sample_idx[uid], data, samples_before, samples_after, spike_width, n_channels, sample_amount)\
            for uid in range(good_units.shape[0]))
                
            avg_waveforms = np.asarray(avg_waveforms)           
        else:
            avg_waveforms = Parallel(n_jobs = -1, verbose = 10, mmap_mode='r', max_nbytes=None )\
            (delayed(erd.extract_a_unit)\
            (sample_idx[uid], data, half_width, spike_width, n_channels, sample_amount)\
            for uid in range(good_units.shape[0]))
                
            avg_waveforms = np.asarray(avg_waveforms)

        #Save in file named 'RawWaveforms' in the KS Directory
        erd.save_avg_waveforms(avg_waveforms, phy_dir, good_units = good_units, extract_good_units_only = extract_good_units_only)

    else:
    
        #Extracting ALL the Units
        n_units = len(np.unique(spike_ids))
        #load metadata
        meta_data = erd.read_meta(Path(meta_path))
        n_elements = int(meta_data['fileSizeBytes']) / 2
        n_channels_tot = int(meta_data['nSavedChans'])
        aplfsy_str = meta_data['snsApLfSy']
        n_channels = int(aplfsy_str.split(sep=',')[0])
        
        #create memmap to raw data, for that session
        data_path = meta_path.with_suffix('.bin')
        data = np.memmap(data_path, dtype = 'int16', shape =(int(n_elements / n_channels_tot), n_channels_tot))
        
        # Remove spike which won't have a full wavefunction recorded        
        spike_ids_tmp = np.delete(spike_ids, np.logical_or( (spike_times < max_width), ( spike_times > (data.shape[0] - max_width))))
        spike_times_tmp = np.delete(spike_times, np.logical_or( (spike_times < max_width), ( spike_times > (data.shape[0] - max_width))))


        sample_idx = erd.get_sample_idx(spike_times_tmp, spike_ids_tmp, sample_amount, units= np.unique(spike_ids))
        
        if KS4_data:
            avg_waveforms = Parallel(n_jobs = -1, verbose = 10, mmap_mode='r', max_nbytes=None )\
            (delayed(erd.extract_a_unit_KS4)\
            (sample_idx[uid], data, samples_before, samples_after, spike_width, n_channels, sample_amount)\
            for uid in range(n_units))
                
            avg_waveforms = np.asarray(avg_waveforms)           
        else:
            avg_waveforms = Parallel(n_jobs = -1, verbose = 10, mmap_mode='r', max_nbytes=None )\
            (delayed(erd.extract_a_unit)\
            (sample_idx[uid], data, half_width, spike_width, n_channels, sample_amount)\
            for uid in range(n_units))
                
            avg_waveforms = np.asarray(avg_waveforms)

        #Save in file named 'RawWaveforms' in the KS Directory
        erd.save_avg_waveforms(avg_waveforms, phy_dir, good_units = good_units, extract_good_units_only = extract_good_units_only)
    del data

def create_call_tsv(ks_path, qm_str):
    # create a call_tsv file with a given metric and calls_sum.csv file
    # to be read by UnitMatch
    call_df = pd.read_csv(os.path.join(ks_path,'calls_sum.csv'))
    n_unit = call_df.shape[0]
    curr_call_str = np.asarray(['mua']*n_unit,dtype='<U4')
    good_ind = np.squeeze(np.where(call_df[qm_str].values == 1))
    curr_call_str[good_ind] = 'good'
    cluster_id = call_df['cluster_id'].values
    new_df = pd.DataFrame(data=cluster_id, index=None, columns=['cluster_id'])
    new_df.insert(loc=1, column='unit_label', value=curr_call_str)
    out_path = os.path.join(ks_path,f'{qm_str}_labels.tsv')
    new_df.to_csv(out_path, sep='\t', index=False)
    return out_path, good_ind
    


def main():
    
    phy_parentDir = r'Z:\AL032_cons_output\KS4_results'    
    raw_parentDir = r'Z:\AL_data\AL032_out\results'
    out_parent = r'Z:\AL032_cons_output\results\UnitMatch_output'
    
    
    np1_scale = 2.34375
    np2_scale = 0.762939
    
    # due to inconsistent run naming, we need a run name for the phy_dir and the raw data
    # first is for the phy_dir, which matches the original data name. 2nd is for the 
    # raw data directory, some of which got renamed.
    runList = [
        ['AL032_2019-11-21_stripe192-natIm_sh_g0_imec0', 'AL032_2019-11-21_stripe192_NatIm_sh_g0_imec0', np2_scale, 'al','cortex'],
    ]
    
    kilosort_version = 4  # for bombcell, 2 or 4, needs to be an integer
    #sort_tag = 'ks25_rs'
    sort_tag = 'ks4'
    phy_dir_list = list()
    raw_data_list = list()
    
    sort_ind = [0]
    
    for i in range(len(runList)):
        for j in range(len(sort_ind)):
            phy_cn = runList[i][0]
            prb_str = phy_cn[len(phy_cn)-1]
            #phy_dir_name = f'imec{prb_str}_{sort_tag}{sort_ind[j]}'
            phy_dir_name = f'imec{prb_str}_{sort_tag}'
            phy_dir = os.path.join(phy_parentDir,phy_cn,phy_dir_name)
            impos = phy_cn.find('imec')
            run_gate_phy = phy_cn[0:impos-1]
            raw_cn = runList[i][1]
            impos = raw_cn.find('imec')
            run_gate_dir = raw_cn[0:impos-1]
            catgt_name = f'catgt_{run_gate_dir}'
            prb_folder = f'{run_gate_dir}_imec{prb_str}'            
            raw_meta = f'{run_gate_phy}_tcat.imec{prb_str}.ap.meta'
            meta_path = os.path.join(raw_parentDir,catgt_name,prb_folder,raw_meta)            
            call_UM_extract(Path(meta_path), Path(phy_dir), True)


                            
if __name__ == "__main__":
        main()     