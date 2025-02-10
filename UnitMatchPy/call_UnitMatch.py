# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:20:41 2024

@author: labadmin
"""


import sys
import os
import pandas as pd

from pathlib import Path
import UnitMatchPy.param_functions as pf
import UnitMatchPy.metric_functions as mf
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import numpy as np
import matplotlib.pyplot as plt
import UnitMatchPy.GUI as gui
import UnitMatchPy.save_utils as su
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

def runUMPair( ks_dir_1, ks_dir_2, qm_str, save_dir ):

    # run UnitMatch for the input directories, assuming bombcell has already
    # been run to generate the pairs of average waveforms and calls
    # expecting paths as strings (rather than Path objects)
    
    # get default parameters, can add your own before or after!
    param = default_params.get_default_param()
    
    
    # Give the paths to the KS directories for each session (with a file 'RawWaveforms' )
    KS_dirs = [ks_dir_1, ks_dir_2]
    
    param['KS_dirs'] = KS_dirs
    # channel_pos = [n_session,n_channels,3], [distance from probe (set to 1), x, z]
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)
    param = util.get_probe_geometry(channel_pos[0], param)  # fills in shank distance 
    
    # default unit label file set by paths_from_KS = cluster_bc_unitType.tsv 
    # to run UM using other metric for 'good', create new tsv file and set 
    # unit_label_path1 = [unit_label_path1, unit_label_path2] here.
    # Label for units to use is 'good'
    print('loading custom unit labels')
    unit_label_path1 = create_call_tsv(ks_dir_1, qm_str)
    unit_label_path2 = create_call_tsv(ks_dir_2, qm_str)
    unit_label_paths = [unit_label_path1, unit_label_path2]
    
    # read in data and select the good units and exact metadata

    good_units = util.get_good_units(unit_label_paths, good = True ) # good = False to load in ALL units  
    waveform, session_id, session_switch, within_session, param = util.load_good_units(good_units, wave_paths, param)
    
    #waveform, session_id, session_switch, within_session, param = util.load_good_waveforms(wave_paths, unit_label_paths, param) # 1-step version of above
    
    # create clus_info, contains all unit id/session related info
    clus_info = {'good_units' : good_units, 'session_switch' : session_switch, 'session_id' : session_id, 
                'original_ids' : np.concatenate(good_units) }
    
    
    # Run the Unit Match process
    
    #  Extract parameters from the waveforms e.g Amplitudes, weighted average waveforms and Spatial Decay lengths
    #   
    #  Calculate metrics/scores for matching e.g Amplitude Score and Waveform similarity
    #  Using putative matches find a estimate of drift correction between session (can be done per shank for 2.0 probes)
    #  Re-Calculate metrics/scores with the drift corrected metrics
    #  Use a naive Bayes classifier to get suggested 'matches' and 'non'matches'
    
    #Get parameters from the wavefunction
    print('waveform shape: ' + repr(waveform.shape))
    print('number of nan: ' + repr(np.sum(np.isnan(waveform))))
    waveform = pf.detrend_waveform(waveform)
    
    max_site, good_idx, good_pos, max_site_mean = pf.get_max_sites(waveform, channel_pos, clus_info, param)
    
    spatial_decay_fit , spatial_decay,  d_10, avg_centroid, avg_waveform, peak_time = pf.decay_and_average_waveform(waveform, channel_pos, good_idx, max_site, max_site_mean, clus_info, param)
    
    amplitude, waveform, avg_waveform = pf.get_amplitude_shift_waveform(waveform, avg_waveform, peak_time, param)
    
    waveform_duration, avg_waveform_per_tp, good_wave_idxs = pf.get_avg_waveform_per_tp(waveform, channel_pos, d_10, max_site_mean, amplitude, avg_waveform, clus_info, param)

    #get Metrics/Scores from the extracted parameters
    amp_score = mf.get_simple_metric(amplitude)
    spatial_decay_score = mf.get_simple_metric(spatial_decay)
    spatial_decay_fit_score = mf.get_simple_metric(spatial_decay_fit, outlier = True)
    wave_corr_score = mf.get_wave_corr(avg_waveform, param)
    wave_mse_score = mf.get_waveforms_mse(avg_waveform, param)
    
    avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param)
    euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param)
    
    centroid_dist, centroid_var = mf.centroid_metrics(euclid_dist, param)
    
    euclid_dist_rc = mf.get_recentered_euclidean_dist(avg_waveform_per_tp_flip, avg_centroid, param)
    
    centroid_dist_recentered = mf.recentered_metrics(euclid_dist_rc)
    traj_angle_score, traj_dist_score = mf.dist_angle(avg_waveform_per_tp_flip, param)
    
    #Collate the metrics and find the putative matches
    # Average Euc Dist
    euclid_dist = np.nanmin(euclid_dist[:,param['peak_loc'] - param['waveidx'] == 0, :,:].squeeze(), axis = 1 )
    
    # TotalScore
    include_these_pairs = np.argwhere( euclid_dist < param['max_dist']) #array indices of pairs to include
    
    # Make a dictionary of score to include
    centroid_overlord_score = (centroid_dist_recentered + centroid_var) / 2
    waveform_score = (wave_corr_score + wave_mse_score) / 2
    trajectory_score = (traj_angle_score + traj_dist_score) / 2
    
    scores_to_include = {'amp_score' : amp_score, 'spatial_decay_score' : spatial_decay_score, 'centroid_overlord_score' : centroid_overlord_score,
                    'centroid_dist' : centroid_dist, 'waveform_score' : waveform_score, 'trajectory_score': trajectory_score }
    
    total_score, predictors = mf.get_total_score(scores_to_include, param)
    
    #Initial thresholding
    
    thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass = True)
    
    param['nExpectedMatches'] = np.sum( (total_score > thrs_opt).astype(int))
    prior_match = 1 - ( param['nExpectedMatches'] / len(include_these_pairs))
    candidate_pairs = total_score > thrs_opt
    
    #drift
    drifts, avg_centroid, avg_waveform_per_tp = mf.drift_n_sessions(candidate_pairs, session_switch, avg_centroid, avg_waveform_per_tp, total_score, param)
    
    # re-do metric extraction with the drift corrected arrays
    
    avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param)
    euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip,param)
    
    centroid_dist, centroid_var = mf.centroid_metrics(euclid_dist, param)
    
    euclid_dist_rc = mf.get_recentered_euclidean_dist(avg_waveform_per_tp_flip, avg_centroid, param)
    
    centroid_dist_recentered = mf.recentered_metrics(euclid_dist_rc)
    traj_angle_score, traj_dist_score = mf.dist_angle(avg_waveform_per_tp_flip, param)
    
    # Average Euc Dist
    euclid_dist = np.nanmin(euclid_dist[:,param['peak_loc'] - param['waveidx'] == 0, :,:].squeeze(), axis = 1 )
    
    # TotalScore
    include_these_pairs = np.argwhere( euclid_dist < param['max_dist']) #array indices of pairs to include, in ML its IncludeThesePairs[:,1]
    include_these_pairs_idx = np.zeros_like(total_score)
    include_these_pairs_idx[euclid_dist < param['max_dist']] = 1 
    
    # Make a dictionary of score to include
    centroid_overlord_score = (centroid_dist_recentered + centroid_var) / 2
    waveform_score = (wave_corr_score + wave_mse_score) / 2
    trajectory_score = (traj_angle_score + traj_dist_score) / 2
    
    scores_to_include = {'amp_score' : amp_score, 'spatial_decay_score' : spatial_decay_score, 'centroid_overlord_score' : centroid_overlord_score,
                    'centroid_dist' : centroid_dist, 'waveform_score' : waveform_score, 'trajectory_score': trajectory_score }
    
    total_score, predictors = mf.get_total_score(scores_to_include, param)
    thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass = False)
    
    
    param['n_expected_matches'] = np.sum( (total_score > thrs_opt).astype(int))
    prior_match = 1 - ( param['n_expected_matches'] / len(include_these_pairs))
    param['drift_corrected_prior_match'] = prior_match  

    # Set-up Bayes analysis
    thrs_opt = np.quantile(total_score[include_these_pairs_idx.astype(bool)], prior_match)
    candidate_pairs = total_score > thrs_opt
    
    prior_match = 1 - (param['n_expected_matches'] / param['n_units']**2 ) #Can change value of priors
    
    Priors = np.array((prior_match, 1-prior_match))
    
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param['score_vector']
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)
    
    # Run bayes analysis
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one = 0)
    # save parameter kernels numpy array: (len(score vector), len(scores_to_include), 2)
    # also save score vector used to make these histograms
    np.save(os.path.join(save_dir,'score_vector.npy'),param['score_vector'])
    np.save(os.path.join(save_dir,'parameter_kernels.npy'), parameter_kernels)
    
    probability = bf.apply_naive_bayes(parameter_kernels, Priors, predictors, param, cond)
    
    output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])
    
    # Optional function to summarise the output
    match_threshold = param['match_threshold']
    #match_threshold = try different values here!
    
    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1
    
    # plt.imshow(output_threshold, cmap = 'Greys')
    # plt.colorbar()
    
    #all idx pairs where the probability is above the threshold
    matches_within_session = np.argwhere(output_threshold == 1) #include within session matches
    matches = np.argwhere( ((output_threshold * within_session)) == True) #exclude within session matches

    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    #NOTE - change to matches to matches_curated if done manual curation with the GUI
    su.save_to_output(save_dir, scores_to_include, matches # matches_curated
                      , output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
                       total_score, output_threshold, drifts, clus_info, param, UIDs = UIDs, matches_curated = None, save_match_table = True)
    #save separate CV output, option to save data so cross verification pairs are split up
    #su.save_to_output_seperate_CV(save_dir, scores_to_include, matches, output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
    #                   total_score, output_threshold, clus_info, param, UIDs = UIDs, matches_curated = None, save_match_table = True)

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
    return out_path
    
    
    
def main():
    
    ks_parent = r'Z:\AL032_cons_output\results'
    out_parent = r'Z:\AL032_cons_output\results\UM_new'
    
    rec_list = ['AL032_2019-11-21_stripe192-natIm_sh_g0',
                'AL032_2019-11-22_stripe192-natIm_sh_g0',
                #'AL032_2019-12-03_stripe192_natIm_sh_g0',
                #'AL032_2019-12-13_stripe192_NatIm_sh_g0',
                #'AL032_2020-01-07_stripe192_NatIm_sh_g0'
                ]
    
            
    # loop over tables of pairs generated from different sorts
    # run UnitMatch on each pair
    #qm_str_set=['ks_call','bc_call','ibl_call','cons_call']
    sh_list = [0,1,2,3]
    rs_list = [1,3,5,7,9,11,13,15,17,19]
    #sort_ind = [0,1,2,3,4,5,6,7,8,9] #for getting the correct overlap table
    day_list = [1,2,3,4,5]   
    qm_str_set = ['ks_call','bc_call','ibl_call','cons_call']
    sh_list = [0]
    rs_list=[1] 
    day_list = [1,2]
    qm_str_set = ['cons_call']
    
    for qm_str in qm_str_set:        
        for sh_ind in sh_list:            
            for ind1 in range(len(day_list)):
                for ind2 in range(ind1+1,len(day_list)):                    
                    for rs1 in rs_list:
                        id1 = day_list[ind1]
                        id2 = day_list[ind2]
                        # paths to kilosort results
                        ks_name = f'imec{sh_ind}_ks25_rs{rs1}'
                        ks_dir_1 = os.path.join(ks_parent,f'{rec_list[id1-1]}_imec{sh_ind}',ks_name)
                        ks_dir_2 = os.path.join(ks_parent,f'{rec_list[id2-1]}_imec{sh_ind}',ks_name)
                        print(f'ks_dir_1: {ks_dir_1}')
                        print(f'ks_dir_2: {ks_dir_2}')
                        # check for and create directory for output
                        dir_name = f'{qm_str}_rs{rs1}'
                        metric_dir = os.path.join(out_parent,dir_name)
                        if not os.path.exists(metric_dir):
                            os.mkdir(metric_dir)
                        sh_dir = os.path.join(metric_dir, f'sh{sh_ind}')
                        if not os.path.exists(sh_dir):
                            os.mkdir(sh_dir)
                        save_dir = os.path.join(sh_dir, f'result_{id1}_{id2}')
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        print(f'save dir: {save_dir}') 
                        runUMPair( ks_dir_1, ks_dir_2, qm_str, save_dir )

                            
if __name__ == "__main__":
        main()                        
    