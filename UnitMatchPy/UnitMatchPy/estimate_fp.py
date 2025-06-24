# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:59:35 2025

@author: colonellj
"""
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns # (only used for testing)

from scipy.stats.sampling import DiscreteAliasUrn


import UnitMatchPy.bayes_functions as bf
from UnitMatchPy.cross_sess_sum import cross_sess_pairs, remove_transpose


def create_sample_scores( npts, kernel, score_vector, urng ):
    # given a kernel estimate of a distribution (probability vector)
    # get npts sampled from that distribution
    
    # use the kernel as a discrete probablility vector to get counts in 
    # the 100 bins of the kernel estimate. To get analog values within the bins,
    # approximate the within bin probability distribution with uniform distribution
    # plus a linear distribution with slope (dm) >=-2 and <=2. 
    # set dm = 0 for bin 0 (use a uniform distribution)
    # all others, if p(i)>0, set dm = (p(i)-p(i-1))/p(i) (to get fold change in p(i))
    # f(y) = dm * x + (1-dm/2),   x >=0 and <=1, spanning the bin
    # cdf = (dm/2) * x^2 + (1-dm/2)*x
    # invert for a specific cdf value by getting the positive root of
    # 0 = (dm/2) * x^2 + (1-dm/2)*x - U    (u is the uniform deviate)
    # for the quadratic, a = dm/2, b = (1-dm/2), c = -1
    
    # difference is small betwee uniform and this linear tweak.
    # dif in random sampling is larger
    

    
    b_use_unif = False
    bin_width = score_vector[1]-score_vector[0]
    k_norm = kernel/np.sum(kernel)
    k_rng = DiscreteAliasUrn(k_norm, random_state=urng)
    bin_ind = k_rng.rvs(npts)    #bin indicies
    n_bin = kernel.shape[0]
    
    scores = np.zeros((npts,), dtype='float64')
    # loop over bin values; for each, get samples from the linear estimate
    # of the local distribution to set analog values
    for i in range(n_bin):
        ind_to_fill = bin_ind==i
        if np.sum(ind_to_fill) > 0:
            unif_diffs = urng.uniform(size=(np.sum(ind_to_fill),))
            if i == 0:
                dm = 0
            else:
                if k_norm[i-1] == 0:
                    dm = 0
                else:
                    dm = (k_norm[i]-k_norm[i-1])/k_norm[i-1]
                    if dm > 2:
                        dm = 2
                    elif dm < -2:
                        dm = -2
            if dm == 0 or b_use_unif:
                dm_diffs = unif_diffs
            else:                
                b = 1 - dm/2  # intercept for the distribution
                dm_diffs = (-b + np.sqrt( b*b + 2*dm*unif_diffs))/dm

                    
            score_vals = (dm_diffs-0.5)*bin_width + score_vector[i]
            scores[ind_to_fill] = score_vals
        
    return scores


def create_all_score_matrices(num_unit, kernels, score_vector, cross_match_vals=[0.2,0.4,0.6,0.8]):
    # make score matrices for range of mismatch fractions.
    # For each metric, fetch a sufficient number of matche and mismatch
    # scores to cover matched fractions given in cross_match_vales
    # num_unit is (n_unit_sess1, n_unit_sess2)
    cross_match_vals = np.sort(np.asarray(cross_match_vals))  # must be sorted to 're-use' the score matrix  
    num_cmf = len(cross_match_vals)
    
    num_bin, num_metric, num_cond = kernels.shape
    
    urng = np.random.default_rng() # initializing once should be OK
    total_unit = int(np.sum(num_unit))
    predictor_arr = np.zeros((total_unit, total_unit,num_metric, num_cmf))
    label_all_arr = np.zeros((total_unit, total_unit, num_cmf))
    
    # how many mismatch scores do we need? Make enough to fill top half of matrix,
    # then replace cross session matches with samples from the match distribution
    # so there are sufficient mismatches for all fractions of cross-session matches
    
    nmm_unique = int((total_unit)*(total_unit-1)/2)         # if the simularity metrics commute
    max_cross_match = int(np.min(num_unit) * np.max(cross_match_vals))
    
    
    # print( f'total_unit: {total_unit}, num_match: {n_match}, n_cross_match: {n_cross_match}')
    
    for j in range(num_metric): 
        score_matrix = np.zeros((total_unit,total_unit)) 
        label_arr = np.zeros((total_unit,total_unit)) 
                
        match_scores = create_sample_scores(total_unit+max_cross_match, kernels[:,j,1],score_vector,urng)
        
        mismatch_scores = create_sample_scores(nmm_unique, kernels[:,j,0],score_vector,urng)
        

        for k, cmf in enumerate(cross_match_vals):
            # the score matrix is filled in so that the diagonal (matching within sessions)
            # are all positives. The cross session pairs are filled in so that they are the
            # first n pairs in off diagonal matches.
            # Start with the smallest number of cross session matches;
            # for larger values, just replace more mismatches with match scores
            if k == 0:
                n_cross_match = int(cross_match_vals[k] * np.min(num_unit))
                n_start = 0     
                for i in range(1,total_unit):
                    score_matrix[i-1,i:total_unit] = mismatch_scores[n_start:(n_start+total_unit-i)] 
                    n_start = n_start + total_unit-i
                           
                # cross sesion matches start at column n_s0, go to n_cross_match
                match_count = 0
                for i in range(n_cross_match):
                    #print(f'{i},{int(num_unit[0])+i}')
                    score_matrix[i,int(num_unit[0])+i] = match_scores[match_count]            
                    label_arr[i,int(num_unit[0])+i] = 1
                    label_arr[int(num_unit[0])+i,i] = 1   # don't update with sum because it would happen multiple times
                    match_count += 1
                                                
                # fill bottom half before filling diagonal
                score_matrix = score_matrix + score_matrix.T  
          
                
                # fill diagonal -- these are the within session matches     
                for i in range(total_unit):
                    score_matrix[i,i] = match_scores[match_count]
                    label_arr[i,i] = 1
                    match_count += 1
           
            else:
                old_cross_match = int(cross_match_vals[k-1] * np.min(num_unit))
                new_cross_match = int(cross_match_vals[k] * np.min(num_unit)) - old_cross_match
                # these new matches
                for i in range(old_cross_match, old_cross_match + new_cross_match):                    
                    score_matrix[i,int(num_unit[0])+i] = match_scores[match_count] 
                    score_matrix[int(num_unit[0])+i,i] = match_scores[match_count] # here set the transpose element manually
                    label_arr[i,int(num_unit[0])+i] = 1
                    label_arr[int(num_unit[0])+i,i] = 1   # don't update with sum because it would happen multiple times
                    match_count += 1
                    
            # concatente onto predictors array for apply bayes
            predictor_arr[:,:,j,k] = score_matrix
            label_all_arr[:,:,k] = label_arr
            
    return predictor_arr, label_all_arr

def create_score_matrix(n_match, n_unmatch, num_unit, kernels, score_vector):
    num_bin, num_metric, num_cond = kernels.shape
    
    urng = np.random.default_rng() # initializing once should be OK
    total_unit = int(np.sum(num_unit))
    predictor_arr = np.zeros((total_unit,total_unit,num_metric))
    label_arr = np.zeros((total_unit,total_unit))
    
    # how many mismatch scores do we need? Make enough to fill top half of matrix,
    # then replace cross session matches with samples from the match distribution
    
    nmm_unique = int((total_unit)*(total_unit-1)/2)         # if the simularity metrics commute
    n_cross_match = n_match-total_unit
    
    print( f'total_unit: {total_unit}, num_match: {n_match}, n_cross_match: {n_cross_match}')
    
    for j in range(num_metric): 
        score_matrix = np.zeros((total_unit,total_unit)) 
                
        match_scores = create_sample_scores(n_match, kernels[:,j,1],score_vector,urng)
        
        mismatch_scores = create_sample_scores(nmm_unique, kernels[:,j,0],score_vector,urng)
        
        # plt.hist(match_scores)
        # plt.show()
        # plt.close()
        # plt.hist(mismatch_scores)
        # plt.show()
        # plt.close()
        
        # the score matrix is filled in so that the diagonal (matching within sessions)
        # are all positives. The cross session pairs are filled in so that they are the
        # first n pairs in 
        n_start = 0     
        for i in range(1,total_unit):
            score_matrix[i-1,i:total_unit] = mismatch_scores[n_start:(n_start+total_unit-i)] 
            n_start = n_start + total_unit-i
                   
        # cross sesion matches start at column n_s0, go to n_cross_match
        match_count = 0
        for i in range(n_cross_match):
            #print(f'{i},{int(num_unit[0])+i}')
            score_matrix[i,int(num_unit[0])+i] = match_scores[match_count]            
            label_arr[i,int(num_unit[0])+i] = 1
            label_arr[int(num_unit[0])+i,i] = 1   # don't update with sum because it would happen multiple times
            match_count += 1
                                        
        # fill bottom half before filling diagonal
        score_matrix = score_matrix + score_matrix.T  
  
        
        # fill diagonal -- these are the within session matches     
        for i in range(total_unit):
            score_matrix[i,i] = match_scores[match_count]
            label_arr[i,i] = 1
            match_count += 1
        
        # concatente onto predictors array for apply bayes
        
        predictor_arr[:,:,j] = score_matrix
    
    
    return predictor_arr, label_arr
    
def remove_dup(vp_df, id_col_name):
    # find any multiples in column with id_col_name
    # for each set, keep the one with the maximum UM Probabilities value
    id_orig = vp_df[id_col_name] #returns dataframe with original indicies + column
    id_dup = id_orig.duplicated()
    id_dup_ind = np.unique(id_orig[id_dup].values)
    for i in range(id_dup_ind.size):
        curr_ind = id_dup_ind[i]
        prob_df = vp_df[vp_df[id_col_name]==curr_ind]['match_prob']
        max_ind = np.argmax(prob_df.values)
        if max_ind.size > 1:
            max_ind = max_ind[0]
        prob_df = prob_df.drop(prob_df.index[max_ind], axis=0)
        ind_to_drop = np.asarray(prob_df.index)
        vp_df = vp_df.drop(ind_to_drop, axis='index')     
    return vp_df

def build_pair_table(prob_df, match_th):   
    vp_df = prob_df[prob_df['match_prob'] > match_th]
    vp_df = remove_dup(vp_df, 'label_0')
    vp_df = remove_dup(vp_df, 'label_1')
    
    true_pos = np.sum(vp_df['true_match'].values)
    false_pos = np.sum(vp_df['true_match'] == 0)
    
    return vp_df,true_pos,false_pos

def plot_tp_fp(roc):
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('threshold')
    ax1.set_ylabel('true positive', color=color)
    ax1.set_ylim([0.7,1.0])
    ax1.plot(roc[:,0], roc[:,1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    
    color = 'tab:red'
    ax2.set_ylabel('false positive', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim([0,0.2])
    ax2.plot(roc[:,0], roc[:,2], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    
def x_formatter(x):
    return '{:1.2}'.format(x)

def box_tp_fp(roc_df):
    
    th_vals = roc_df['threshold'].values
    threshold_labels = ['{:1.2}'.format(x) for x in th_vals]
    
    cmf_vals = np.unique(roc_df['cross_match_frac'].values)
    cmf_labels = ['{:1.2}'.format(x) for x in cmf_vals]
    
    for i, cmf in enumerate(cmf_vals):
       
        # set the figure size
        fig = plt.figure(figsize=(6,4))
        sns.boxplot(x='threshold',y='true_pos_frac',
                    data=roc_df[roc_df['cross_match_frac']==cmf],formatter=x_formatter )
        plt.title(f'Match fraction: {cmf_labels[i]}')
        plt.ylim([0.5,1.0])
        plt.show()
        plt.close()
        
        fig = plt.figure(figsize=(6,4))
        sns.boxplot(x='threshold',y='false_pos_frac',
                    data=roc_df[roc_df['cross_match_frac']==cmf],formatter=x_formatter)
        plt.title(f'Match fraction: {cmf_labels[i]}')
        plt.ylim([0,0.2])
        plt.show()
        plt.close()
    
def estFP(result_path, threshold=None, est_cross_match_frac = 0.6, n_trial=20):
    # Given a folder of UM output, generate a set of known matches and 
    # mismatches distributed according the saved distribution kernels:
    # create a set of known mismatch and match score vectors drawn from these 
    # distributions. Calculate TP/FP called for this set from the naive
    # bayes + threshold + elimination of duplicate pairs.
    
    # hard coded params
    cond = [0,1] # 0 = match, 1 = no match
    min_th = 0.3 # for calling postential matches. Adjust if real threshold is lower   
    b_write_prob = True # to write out the big dataframe of all n_trials. Only for debugging         
    
    UMparam = np.load(os.path.join(result_path, 'UMparam.pickle'), allow_pickle=True)
    with np.load(os.path.join(result_path,'UM Scores.npz'), mmap_mode='r') as sc_file:
        scores_to_include = sc_file.files
    if threshold is None:
        threshold = UMparam['match_threshold']
        
    if threshold < min_th:
        min_th = threshold
        
    th_tp_fp = np.zeros((n_trial,2))    # for results at requested threshold    
        
    # because all of the calculation time is creating the pairs, use them to 
    # create a rough roc curve in addition to returning the estimate for 
    # the caller specified threshold
    roc_step = 0.1
    roc_nstep = (1 + np.floor((0.99 - min_th)/roc_step)).astype(int)    
    roc = np.zeros((roc_nstep*n_trial,3))
    roc_col_names = ['threshold',
                     'true_pos_frac',
                     'false_pos_frac']    
        
           
    cond = [0,1] # 0 = match, 1 = no match
    
    kernels = np.load(os.path.join(result_path,'paraemter_kernels.npy'))
    clus_info = np.load(os.path.join(result_path,"ClusInfo.pickle"),allow_pickle=True)
    num_bin, num_metric, num_cond = kernels.shape
    
    num_unit = np.zeros((2,))

    num_unit[0] = clus_info['good_units'][0].size
    num_unit[1] = clus_info['good_units'][1].size
    
    # estimate expected matches bewteen sessions 
    # for each n_trial, estimate the fraction FP using number of units
    # and estiamted cross matches for the real pair of files
    total_units = int(np.sum(num_unit))
    cross_match = int(est_cross_match_frac*np.min(num_unit))
    num_match = int(np.sum(num_unit) + cross_match)  # same session units are matched + cross match
    num_unmatch = int(np.sum(num_unit) - num_match) # can't match more units than the smaller set
    prior_match = 1 - (num_match / total_units**2 ) #see line 172 of CallUnitMatch, check current notebooks
    Priors = np.array((prior_match, 1-prior_match))
    
    
    # columns for dataframe of all trials
    col_names = ['trial',
                 'label_0',
                 'label_1',
                 'sess_label_0',
                 'sess_label_1',
                 'match_prob',
                 'no_match_prob',
                 'true_match'
                 ]
    
    for k in range(num_metric):
        col_names.append(scores_to_include[k])
        
    for nt in range(n_trial):
        # build simulated score vectors
        predictor_arr, label_arr = create_score_matrix(num_match, num_unmatch, num_unit, kernels, UMparam['score_vector'])
                              
        prob = bf.apply_naive_bayes(kernels, Priors, predictor_arr, UMparam, cond)
        # reshape label_arr to match prob
        label_arr = label_arr.reshape((total_units*total_units,))
        
        n_pair = total_units*total_units    
        df_pair_met = np.zeros((n_pair,num_metric+8))
        for i in range(total_units):
            for j in range(total_units):
                curr_pair = i*total_units + j
                df_pair_met[curr_pair,0:3] = [nt,i,j] #nit labels
                # session labels
                if i >= num_unit[0]:
                    df_pair_met[curr_pair,3] = 1
                if j >= num_unit[0]:
                    df_pair_met[curr_pair,4] = 1
                # match prob
                df_pair_met[curr_pair,5:8] = \
                    [prob[curr_pair,1],prob[curr_pair,0],label_arr[curr_pair]]

                for k in range(num_metric):
                    df_pair_met[curr_pair,k+8] = predictor_arr[i,j,k]
                                
        curr_df = pd.DataFrame(df_pair_met,columns=col_names)
       
        # cross session and within session matches are subtly different--
        # because for within session matches there is always a 'genuine' match --
        # and FPs can be removed by selecting the maximum. So, restrict
        # TP/FP measurements to cross session matches, with a 'minimum' threshold
           
        curr_cross = curr_df[(curr_df['sess_label_0'] == 0) & (curr_df['sess_label_1'] == 1)\
                        & (curr_df['match_prob'] > min_th) ]
          
        vp_df,true_pos,false_pos =  build_pair_table(curr_cross, threshold)   
        th_tp_fp[nt,:] = [true_pos/cross_match, false_pos/(true_pos + false_pos)]
        
        for i in range(roc_nstep):
           vp_df,true_pos,false_pos =  build_pair_table(curr_cross, min_th + i*roc_step)           
           roc[nt*roc_nstep + i, 0] = min_th + i*roc_step
           roc[nt*roc_nstep + i, 1] = true_pos/cross_match
           roc[nt*roc_nstep + i, 2] = false_pos/(true_pos + false_pos)
    
        
        if nt == 0:   
            prob_df = pd.DataFrame(df_pair_met,columns=col_names)           
        else:
            prob_df = pd.concat([prob_df,curr_df],axis=0)
            
    roc_df = pd.DataFrame(data=roc,columns=roc_col_names)        
    if b_write_prob:
        new_name = f'sim_prob_cross_match_{cross_match}.csv'
        prob_df.to_csv(os.path.join(result_path,new_name),index=None)
    
    tp_mean, fp_mean = np.mean(th_tp_fp,axis=0)
    tp_min = np.min(th_tp_fp[:,0])
    fp_max = np.max(th_tp_fp[:,1])                
    return tp_mean, fp_mean, tp_min, fp_max, roc_df, threshold

def estFP_vs_matchfrac(result_path, threshold=None, n_trial=20):
    # Given a folder of UM output, generate a set of known matches and 
    # mismatches distributed according the saved distribution kernels.
    # With these known match/mismatch score vectors and a set of 
    # fractions of cross matches, calculate TP/FP called for this set from the naive
    # bayes + threshold + elimination of duplicate pairs.
    # Repeat for n_trials to get estimated error.
    
    # hard coded params
    cond = [0,1] # 0 = match, 1 = no match
    min_th = 0.3 # for calling postential matches. Adjust if real threshold is lower 
    cross_match_vals = [0.2,0.4,0.6,0.8]    
    num_cmf = len(cross_match_vals)
    b_write_prob = False # to write out the big dataframe of all n_trials. Only for debugging         
    
    UMparam = np.load(os.path.join(result_path, 'UMparam.pickle'), allow_pickle=True)
    with np.load(os.path.join(result_path,'UM Scores.npz'), mmap_mode='r') as sc_file:
        scores_to_include = sc_file.files
    if threshold is None:
        threshold = UMparam['match_threshold']
        
    if threshold < min_th:
        min_th = threshold
        
    th_tp_fp = np.zeros((n_trial,2,num_cmf))    # for results at requested threshold    
        
    # because all of the calculation time is creating the pairs, use them to 
    # create a rough roc curve in addition to returning the estimate for 
    # the caller specified threshold
    roc_step = 0.1
    roc_nstep = (1 + np.floor((0.99 - min_th)/roc_step)).astype(int)    
    roc = np.zeros((roc_nstep*n_trial*num_cmf,4))
    roc_col_names = ['cross_match_frac',
                     'threshold',
                     'true_pos_frac',
                     'false_pos_frac']    
        
           
    cond = [0,1] # 0 = match, 1 = no match
    
    kernels = np.load(os.path.join(result_path,'parameter_kernels.npy'))
    clus_info = np.load(os.path.join(result_path,"ClusInfo.pickle"),allow_pickle=True)
    num_bin, num_metric, num_cond = kernels.shape
    
    num_unit = np.zeros((2,))

    num_unit[0] = clus_info['good_units'][0].size
    num_unit[1] = clus_info['good_units'][1].size
    
    # columns for dataframe of all cross_match_vals, all trials
    col_names = ['cross_match_frac',
                 'trial',
                 'label_0',
                 'label_1',
                 'sess_label_0',
                 'sess_label_1',
                 'match_prob',
                 'no_match_prob',
                 'true_match'
                 ]
    
    for k in range(num_metric):
        col_names.append(scores_to_include[k])
        
    
    # estimate expected matches bewteen sessions 
    # for each n_trial, estimate the fraction FP using number of units
    # and estiamted cross matches for the real pair of files
    total_units = int(np.sum(num_unit))
    
        
    for nt in range(n_trial):
        # build simulated score vectors
        predictor_arr, label_arr = create_all_score_matrices(num_unit, kernels, 
                                    UMparam['score_vector'], cross_match_vals)
        
        for nf, cross_match_frac in enumerate(cross_match_vals):    
            cross_match = int(cross_match_frac*np.min(num_unit))
            num_match = int(np.sum(num_unit) + cross_match)  # same session units are matched + cross match           
            prior_match = 1 - (num_match / total_units**2 ) #see line 172 of CallUnitMatch, check current notebooks
            Priors = np.array((prior_match, 1-prior_match))             
            prob = bf.apply_naive_bayes(kernels, Priors, predictor_arr[:,:,:,nf], UMparam, cond)
            # reshape label_arr to match prob
            curr_label_arr = label_arr[:,:,nf].reshape((total_units*total_units,))
            
            n_pair = total_units*total_units    
            df_pair_met = np.zeros((n_pair,num_metric+9))
            for i in range(total_units):
                for j in range(total_units):
                    curr_pair = i*total_units + j
                    df_pair_met[curr_pair,0:4] = [cross_match_frac,nt,i,j] #unit labels
                    # session labels
                    if i >= num_unit[0]:
                        df_pair_met[curr_pair,4] = 1
                    if j >= num_unit[0]:
                        df_pair_met[curr_pair,5] = 1
                    # match prob
                    df_pair_met[curr_pair,6:9] = \
                        [prob[curr_pair,1],prob[curr_pair,0],curr_label_arr[curr_pair]]
    
                    for k in range(num_metric):
                        df_pair_met[curr_pair,k+9] = predictor_arr[i,j,k,nf]
                                    
            curr_df = pd.DataFrame(df_pair_met,columns=col_names)
           
            # cross session and within session matches are subtly different--
            # because for within session matches there is always a 'genuine' match --
            # and FPs can be removed by selecting the maximum. So, restrict
            # TP/FP measurements to cross session matches, with a 'minimum' threshold
               
            curr_cross = curr_df[(curr_df['sess_label_0'] == 0) & (curr_df['sess_label_1'] == 1)\
                            & (curr_df['match_prob'] > min_th) ]
              
            vp_df,true_pos,false_pos =  build_pair_table(curr_cross, threshold)   
            th_tp_fp[nt,:,nf] = [true_pos/cross_match, false_pos/(true_pos + false_pos)]
            
            for i in range(roc_nstep):
               vp_df,true_pos,false_pos =  build_pair_table(curr_cross, min_th + i*roc_step)
               roc[nt*num_cmf*roc_nstep + nf*roc_nstep + i, 0] = cross_match_frac
               roc[nt*num_cmf*roc_nstep + nf*roc_nstep + i, 1] = min_th + i*roc_step
               roc[nt*num_cmf*roc_nstep + nf*roc_nstep + i, 2] = true_pos/cross_match
               roc[nt*num_cmf*roc_nstep + nf*roc_nstep + i, 3] = false_pos/(true_pos + false_pos)
        
            
            if nt == 0 and nf == 0:   
                prob_df = pd.DataFrame(df_pair_met,columns=col_names)           
            else:
                prob_df = pd.concat([prob_df,curr_df],axis=0)
    
    
    roc_df = pd.DataFrame(data=roc,columns=roc_col_names)        
    if b_write_prob:
        new_name = f'sim_prob_cross_match_{cross_match}.csv'
        prob_df.to_csv(os.path.join(result_path,new_name),index=None)
    
    # tp_mean, fp_mean = np.mean(th_tp_fp,axis=0)
    # tp_min = np.min(th_tp_fp[:,0])
    # fp_max = np.max(th_tp_fp[:,1])                
    return roc_df

def thresh_for_FP(roc_df, estfrac_df, fp_max, min_thresh):
    # From the estimated FP vs. a vector of match thresholds
    cm_frac_vals = np.unique(roc_df['cross_match_frac'].values)
    thresh_vals = np.unique(roc_df['threshold'].values)
    match_frac = estfrac_df['est_matchfrac'].values
    # calc matrix of mean fp for cm_frac and threshold
    # could replace with some other metric (+75th percentile?)
    fp_matrix = np.zeros((len(cm_frac_vals),len(thresh_vals)))
    for i, cmf in enumerate(cm_frac_vals):
        for j, th in enumerate(thresh_vals):
            qs = f'threshold=={th} and cross_match_frac=={cmf}'
            fp_values = roc_df.query(qs)['false_pos_frac'].values
            fp_matrix[i,j] = np.percentile(fp_values,75)
    
    print(fp_matrix)
    out_thresh = np.zeros((len(match_frac),))
    for i, mf in enumerate(match_frac):
        cm_ind = np.argmin(np.abs(cm_frac_vals - mf))    
        fp_test = fp_matrix[cm_ind,:] < fp_max
        if sum(fp_test) > 0:
            out_thresh[i] = thresh_vals[np.min(np.argwhere(fp_test))]
        else:
            out_thresh[i] = np.max(thresh_vals)
        if out_thresh[i] < min_thresh:
            out_thresh[i] = min_thresh
    
    estfrac_df.insert(loc=3, column='adj_thresh', value=out_thresh)
    print(estfrac_df)
    return estfrac_df
      
def est_matchfrac(result_path, threshold):
    # read in the match prob table, calculate pair table with threshold
    prob_df = pd.read_csv(os.path.join(result_path, 'MatchTable.csv'))
    clus_info = np.load(os.path.join(result_path,'ClusInfo.pickle'), allow_pickle = True)
    dup_rem_df = cross_sess_pairs(prob_df)
    tn_rem_df = remove_transpose(dup_rem_df)
    qs = f'`UM Probabilities` > {threshold}'
    tn_rem_df = tn_rem_df.query(qs)
    sess_switch = clus_info['session_switch']
    n_unit = np.diff(sess_switch)
    n_sess = len(n_unit)
    # session labels are 1 to n_sess
    # pair table includes unique pairings for sess id 2 > sess 1
    # create df of cross match frac = est matches/min(n_unit for the two sessions)
    col_names = ['RecSes 1',
                 'RecSes 2',
                 'est_matchfrac'
                 ]
    n_pairs = int((n_sess * (n_sess - 1))/2)
    em_data = np.zeros((n_pairs,3))
    k = 0
    for i in range(n_sess):
        for j in range(i+1, n_sess):
            max_match = min(n_unit[i],n_unit[j])
            qs = f' `RecSes 1` == {i+1} and `RecSes 2` == {j+1}'  # threshold applied in cross_sess_pairs
            [est_match, nc] = tn_rem_df.query(qs).shape
            est_frac = est_match/max_match
            em_data[k,:] = [i+1,j+1,est_frac]
            k = k + 1
    estfrac_df = pd.DataFrame(em_data, columns = col_names)
    return estfrac_df
            
            
    
    
    
def main():


    # repeat n trials to get error estimate on the FPs. To mimic the data, the
    # number of included units is low, so serveral (20) trials are required to
    # get a good estimate of teh mean.
    b_recalc = True
    result_path = r'Z:\AL032_cons_output\results\UM_new\cons_call_rs1\sh0\result_d1_d2_d3_d4_d5'
    output_path = result_path 
    min_thresh = 0.5
    fp_max = 0.02
    
    roc_name = 'sim_roc.csv'
    if b_recalc:
 
        #tp_mean, fp_mean, tp_min, fp_max, roc_df, threshold = estFP(result_path, est_cross_match_frac = 0.9)
        roc_df = estFP_vs_matchfrac(result_path, threshold=None, n_trial=20)
        roc_df.to_csv(os.path.join(output_path,roc_name),index=None)
        

        
    else:
        # load roc_df t0 plot
        roc_df = pd.read_csv(os.path.join(output_path,roc_name))
    box_tp_fp(roc_df)
    
    # estimate the match fraction in the real data
    estfrac_df = est_matchfrac(result_path, min_thresh)
    adj_thresh_df = thresh_for_FP(roc_df, estfrac_df, fp_max, min_thresh)
    
        
if __name__ == "__main__":
    main()