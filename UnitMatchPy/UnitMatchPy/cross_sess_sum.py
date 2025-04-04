# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:10:55 2025

@author: labadmin
"""
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def plot_cross_match(prob_df):
    # this is for simulated match tables used for FP estimates
    cross_qs = ' sess_label_0 == 0 & sess_label_1 == 1 & trial == 0'
    cross_df = prob_df.query(cross_qs)
    
    sess0_labels = np.unique(cross_df['label_0'].values)
    sess1_labels = np.unique(cross_df['label_1'].values)
    n_unit_0 = (np.max(sess0_labels) + 1).astype(int)
    n_unit_1 = (np.max(sess1_labels) + 1 - n_unit_0).astype(int)
    mp_vals = cross_df['match_prob'].values
    mp_mat = np.reshape(mp_vals,(n_unit_0,n_unit_1))
    
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots(figsize=(4,4))     
    im = ax.imshow(mp_mat)
    plt.colorbar(im)
    ax.set_xlabel('session 0 label')
    ax.set_ylabel('session 1 label')
    plt.show()

def remove_dup(vp_df, id_col_name):
    # find any multiples in column with id_col_name
    # for each set, keep the one with the maximum UM Probabilities value
    id_orig = vp_df[id_col_name] #returns dataframe with original indicies + column
    id_dup = id_orig.duplicated()
    id_dup_ind = np.unique(id_orig[id_dup].values)
    for i in range(id_dup_ind.size):
        curr_ind = id_dup_ind[i]
        prob_df = vp_df[vp_df[id_col_name]==curr_ind]['UM Probabilities']
        max_ind = np.argmax(prob_df.values)
        if max_ind.size > 1:
            max_ind = max_ind[0]
        prob_df = prob_df.drop(prob_df.index[max_ind], axis=0)
        ind_to_drop = np.asarray(prob_df.index)
        vp_df = vp_df.drop(ind_to_drop, axis='index')     
    return vp_df

def remove_transpose(input_df):
    # create a df with just the rows for which recSes 1 < RecSes 2 
    # (that's an arbitrary choice)
    qs = "`RecSes 1` < `RecSes 2`"
    uh_df = input_df.query(qs)  # uh for 'upper half" of the matrix
    nr,nc = uh_df.shape
    uh_df.insert(loc = nc, column='um_prob_2', value=0)
    
    # loop over the rows. get ID1, ID2, RecSes 1, RecSes 2, and get the row
    # that is the 'transpose for ID1', e.g. *, ID1, RecSes 2, RecSes 1.
    # if it doesn't exist, remove this row.
    # if it does exist, check that its pair matches. Set UM_prob_2 to the second pairing
    # and remove the transpose.
    
    vp_df = pd.DataFrame()
    n_vp = 0
    
    for i in range(nr):
        curr_row = uh_df.iloc[i]
        id1,id2,r1,r2,m,um_prob = curr_row[0:6]
        #print(f'id1 {id1} id2 {id2}')
        qs = f"ID2=={id1} and `RecSes 1`=={r2} and `RecSes 2`=={r1}"
        t_row = input_df.query(qs) # returns a dataframe
        #print(t_row)
        if not t_row.empty:
            # did transpose match the same units?            
            t_id2, t_id1 = t_row.iloc[0,0:2].values  # id2 and id1 swapped to match RecSes
            if ((id1 == t_id1) and (id2 == t_id2)):
                #curr_row['um_prob_2']  = t_row['UM Probabilities'].values
                vp_df = pd.concat((vp_df,curr_row), axis=1)                             
                vp_df.iloc[[nc],[n_vp]] = t_row['UM Probabilities'].values
                n_vp = n_vp + 1
        # else, just skip, don't add curr_row to the validated pair set
    vp_df = vp_df.transpose()
    return vp_df
    
def cross_sess_pairs(prob_df):
    
    #cross_qs = f' `RecSes 1` != `RecSes 2` and `UM Probabilities` > {threshold}'
    cross_qs = ' `RecSes 1` != `RecSes 2`'
    cross_df = prob_df.query(cross_qs)
    sess_labels = np.unique(np.concatenate( (cross_df['RecSes 1'].values, cross_df['RecSes 2'].values)))
    n_label = len(sess_labels) 
    # UM calculates probabilities for all possible combinations, i -> j and j -> i, working with 
    # the first and 2nd halves of the recordins. To create a set of unique 'valid' pairs, 
    # e.g. i->j, for each unit in i, pick the pair (row) with maximum UM prob as match
 
    vp_df = pd.DataFrame()
    for i in range(n_label):
        for j in range(n_label):
            id1 = sess_labels[i]
            id2 = sess_labels[j]
            qs = f" `RecSes 1` == {id1} and `RecSes 2` == {id2} "
            curr_df = cross_df.query(qs)
            nr, nc = curr_df.shape            
            curr_df = remove_dup(curr_df, 'ID1')
            curr_df = remove_dup(curr_df,'ID2')
            vp_df = pd.concat((vp_df,curr_df))
    
    return vp_df

def apply_threshold(input_df, threshold):
    if isinstance(threshold, float):
        qs = f'`UM Probabilities` > {threshold}'
        vp_df = input_df.query(qs)
    else:
        # assume threshold is a df of thresholds for each pair of labels with RecSes 1 < RecSes 2
        sess_labels = np.unique(np.concatenate( (input_df['RecSes 1'].values, input_df['RecSes 2'].values)))
        n_label = len(sess_labels)
        # loop over all sets of labels, apply threshold, add those rows to vp_df
        vp_df = pd.DataFrame()
        for i in range(n_label):
            for j in range(i+1,n_label):
                id1 = sess_labels[i]
                id2 = sess_labels[j]
                qs = f"`RecSes 1` == {id1} and `RecSes 2` == {id2} "
                curr_th = threshold.query(qs)['adj_thresh'].values[0]
                qs2 = f"{qs} and `UM Probabilities` > {curr_th} and um_prob_2 > {curr_th}"
                print(qs2)
                if vp_df.shape[0] == 0:
                    vp_df = input_df.query(qs2)
                else:
                    vp_df = pd.concat((vp_df, input_df.query(qs2)))
                    
        
    return vp_df

def get_nearest(curr_sess, curr_pairs, n_sess):
    # curr_sess = all recording sessions that include uid
    # for all curr_sess, get rows in curr_pairs that 
    # include it, use the one with the lowest difference
    # in session label. 'Nearest' for that session is 
    # the 2nd session in that pair.
    
    ncs = len(curr_sess)
    nearest = -1*np.ones((ncs,)).astype(int)
    delta = curr_pairs[:,1] - curr_pairs[:,0]
    for i, rs in enumerate(curr_sess):
        sel_ind = np.concatenate((np.where(curr_pairs[:,0]==rs)[0], np.where(curr_pairs[:,1]==rs)[0]))
        min_ind = np.argmin(delta[sel_ind])
        sel_pair = curr_pairs[sel_ind[min_ind]]
        if sel_pair[0] == rs:
            nearest[i] = sel_pair[1]
        else:
            nearest[i] = sel_pair[0]
       
    return nearest
        
def get_prob(curr_rows, i, j):
    # for a given pair of session labels, get the match probability
    # from the verified pair table.
    # in this table, rec_sess 1 is always < rec_sess 2
    unit_i_id = -1
    unit_j_id = -1
    prob = -1
    if i < j:
        qs = f'`RecSes 1`=={i} and `RecSes 2`=={j}' 
        sel_row = curr_rows.query(qs)
        sr,sc = sel_row.shape
        if sr==1:
            unit_i_id = sel_row['ID1'].values
            unit_j_id = sel_row['ID2'].values
            prob = min( sel_row['UM Probabilities'].values, sel_row['um_prob_2'].values)
        
    else:
        qs = f'`RecSes 1`=={j} and `RecSes 2`=={i}' 
        sel_row = curr_rows.query(qs)
        sr,sc = sel_row.shape
        if sr==1:
            unit_i_id = sel_row['ID2'].values
            unit_j_id = sel_row['ID1'].values
            prob = min( sel_row['UM Probabilities'].values, sel_row['um_prob_2'].values)
    
    return unit_i_id, unit_j_id, prob
    
def build_cross_sum(input_df, uid_name):
    # build column names
    col1 = f'UID {uid_name} 1'
    col2 = f'UID {uid_name} 2'
    
    # restrict assignments to cases where the UID for this 'conservative' level
    # is the same for the two sessions
    qs = f'`{col1}`== `{col2}`'
    vp_df = input_df.query(qs)

    # get all UIDs
    
    uid_all = np.unique(np.concatenate((vp_df[col1].values,vp_df[col2].values))).astype(int)
   
    sess_labels = np.unique(np.concatenate( (vp_df['RecSes 1'].values, vp_df['RecSes 2'].values))).astype(int)
    n_label = len(sess_labels) 
    
    #cross session sum column names
    col_names = list() 
    col_names.append('UM UID')
    col_names.append('sess_count')  # how many recording sessions for this unit
    ncol_pre = 2  # columns before the label, prob data
    for i in range(n_label):
        col_names.append(f'ResSes{sess_labels[i]}_id')
    for i in range(n_label):
        col_names.append(f'RS{sess_labels[i]}_nearest')     
    
    cross_sum_df = pd.DataFrame(columns=col_names)
    ncol = len(col_names)
    
    for nj, j in enumerate(uid_all):
        qs = f'`{col1}`=={j} or `{col2}`=={j}'
        curr_rows = vp_df.query(qs)   #always returns a dataframe, should never be empty
        # get pairs of sessions for this uid
        nr, nc = curr_rows.shape
        cp = np.zeros((nr,2))
        cp[:,0] = curr_rows['RecSes 1'].values
        cp[:,1] = curr_rows['RecSes 2'].values           
        cs = np.unique(cp).astype(int)
        nearest = get_nearest(cs, cp, n_label)
        curr_vals = -1 * np.ones((ncol,))
        curr_vals[0] = j
        for i, curr_sess in enumerate(cs):
            unit_i_id, unit_j_id, nearest_prob = get_prob(curr_rows, curr_sess, nearest[i])
            curr_vals[ncol_pre+curr_sess-1] = (unit_i_id)  # label of this unit in this session
            curr_vals[ncol_pre+n_label+curr_sess-1]=nearest_prob
       
        curr_vals[1] = np.sum(curr_vals[ncol_pre:ncol_pre+n_label] > 0)
        
        if nj == 0:
            cross_sum_df = pd.DataFrame(data=curr_vals.reshape(1,ncol), columns=col_names)
        else:            
            cross_sum_df = pd.concat((cross_sum_df, pd.DataFrame(data=curr_vals.reshape(1,ncol), columns=col_names)))
       
    
    return cross_sum_df
    
    
def main():
    #match_table_path = r'Z:\AL032_cons_output\results\UnitMatch_output\ks_call_rs1\sh0\result_1_2\sim_prob_cross_match_78.csv'
    
    match_table_path = r'Z:\AL032_cons_output\results\UM_new\cons_call_rs1\sh0\result_d1_d2_d3_d4_d5\MatchTable.csv'
    out_path = Path(match_table_path).parent
    
    prob_df = pd.read_csv(match_table_path)   
    #plot_cross_match(prob_df)
    
    dup_rem_df = cross_sess_pairs(prob_df)    
    dup_rem_df.to_csv(os.path.join(out_path,'dup_rem_df.csv'),index=None)
    
    tn_rem_df = remove_transpose(dup_rem_df)
    tn_rem_df.to_csv(os.path.join(out_path,'tn_rem_df.csv'),index=None)
    
    cross_sum_df = build_cross_sum(tn_rem_df, 'int')
    cross_sum_df.to_csv(os.path.join(out_path,'cross_sum_df.csv'),index=None)
    
if __name__ == "__main__":
    main()