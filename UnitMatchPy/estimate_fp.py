# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:59:35 2025

@author: colonellj
"""
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.sampling import DiscreteAliasUrn


import UnitMatchPy.bayes_functions as bf

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


def create_score_matrix(n_match, n_unmatch, num_unit, kernels, score_vector):
    num_bin, num_metric, num_cond = kernels.shape
    
    urng = np.random.default_rng() # initializing once should be OK
    total_unit = int(np.sum(num_unit))
    predictor_arr = np.zeros((total_unit,total_unit,num_metric))
    label_arr = np.zeros((total_unit,total_unit))
    
    # how many mismatch scores do we need? Make enough to fill top half of matrix,
    # then replace cross session matches with samples from the match distribution
    # samples from match distribution
    
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
    
    # set plot style: grey grid in the background:
   
    # set the figure size
    fig, ax = plt.subplots()
    
    ax = sns.boxplot(x='threshold',y='true_pos_frac',data=roc_df,formatter=x_formatter )
     
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(8,6))
    sns.boxplot(x='threshold',y='false_pos_frac',data=roc_df,formatter=x_formatter)
    plt.show()
    plt.close()
    
def estFP(result_path, threshold=None, n_trial=20):
    # Given a folder of UM output, generate a set of known matches and 
    # mismatches distributed according the saved distribution kernels:
    # create a set of known mismatch and match score vectors drawn from these 
    # distributions. Calculate TP/FP called for this set from the naive
    # bayes + threshold + elimination of duplicate pairs.
    
    # hard coded params
    cond = [0,1] # 0 = match, 1 = no match
    min_th = 0.3 # for calling postential matches. Adjust if real threshold is lower
    est_cross_match_frac = 0.6 # set on the low side for a conservative FP estimate
    b_write_prob = False # to write out the big dataframe of all n_trials. Only for debugging         
    
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
    return tp_mean, fp_mean, tp_min, fp_max, roc_df

def main():


    # repeat n trials to get error estimate on the FPs. To mimic the data, the
    # number of included units is low, so serveral (20) trials are required to
    # get a good estimate of teh mean.
    b_recalc = True
    result_path = r'Z:\AL032_cons_output\results\UnitMatch_output\ks_call_rs1\sh0\result_1_2'
    output_path = result_path      
    
    roc_name = 'sim_roc.csv'
    if b_recalc:
 
        tp_mean, fp_mean, tp_min, fp_max, roc_df = estFP(result_path)
        roc_df.to_csv(os.path.join(output_path,roc_name),index=None)
        
        print(f'mean %fp: {100*fp_mean:.1f}, max %fp: {100*fp_max:.1f}')
    else:
        # load roc_df t0 plot
        roc_df = pd.read_csv(os.path.join(output_path,roc_name))
    box_tp_fp(roc_df)
        
if __name__ == "__main__":
    main()