import os

import npx_utils as npx
import numpy as np
import pandas as pd
from tqdm import tqdm

import UnitMatchPy.default_params as default_params
from UnitMatchPy.cross_sess_sum import (
    apply_threshold,
    build_cross_sum,
    cross_sess_pairs,
    remove_transpose,
)
from UnitMatchPy.estimate_fp import est_matchfrac, estFP_vs_matchfrac, thresh_for_FP
from UnitMatchPy.run import unit_match
from UnitMatchPy.utils import get_probe_geometry, get_session_data, get_within_session


def get_tracking_paths(probe_folders, n_splits):
    mean_wf_paths = []
    metrics_paths = []
    channel_pos = []

    for ks_folder in probe_folders:
        # check if it is in KS directory
        # if os.path.exists(os.path.join(ks_folder, "RawWaveforms")):
        #     mean_wf_paths.append(os.path.join(ks_folder, "RawWaveforms"))
        # else:
        mean_wf_paths.append(
            os.path.join(ks_folder, f"mean_waveforms_{n_splits}split.npy")
        )

        if os.path.exists(os.path.join(ks_folder, "cilantro_metrics.tsv")):
            metrics_path = os.path.join(ks_folder, "cilantro_metrics.tsv")
        else:
            metrics_path = os.path.join(ks_folder, "cluster_group.tsv")
        metrics_paths.append(metrics_path)
        channel_pos.append(np.load(os.path.join(ks_folder, "channel_positions.npy")))

    return mean_wf_paths, metrics_paths, channel_pos


def get_good_units(metrics_paths, good_lbls=["good"]):
    good_units = []
    for metrics_path in metrics_paths:
        metrics = pd.read_csv(metrics_path, sep="\t", index_col="cluster_id")
        good_units.append(metrics[metrics["label"].isin(good_lbls)].index.values)
    return good_units


def load_good_units(good_units, mean_wf_paths, params):
    if len(mean_wf_paths) == len(good_units):
        n_sessions = len(mean_wf_paths)
    else:
        raise ("Warning: gave different number of paths for waveforms and labels!")

    # load mean_wf individually, then concatenate
    waveforms = []
    n_units_per_session = np.zeros(n_sessions, dtype=int)
    for i, mean_wf_path in enumerate(mean_wf_paths):
        if "RawWaveforms" in mean_wf_path:
            p_file = os.path.join(
                mean_wf_path, f"Unit{int(good_units[i][0].squeeze())}_RawSpikes.npy"
            )
            tmp = np.load(p_file)
            mean_wf_good = np.zeros(
                (len(good_units[i]), tmp.shape[0], tmp.shape[1], tmp.shape[2])
            )
            for j in range(len(good_units[i])):
                # loads in all GoodUnits for that session
                p_file_good = os.path.join(
                    mean_wf_path, f"Unit{int(good_units[i][j].squeeze())}_RawSpikes.npy"
                )
                mean_wf_good[j] = np.load(p_file_good)
        else:
            mean_wf = np.load(mean_wf_path)
            # only keep good units
            mean_wf_good = mean_wf[good_units[i]]

            # update mean_wf to match (n_units, spike_width, n_channels)
            if params["spike_width"] != mean_wf_good.shape[1]:
                mean_wf_good = np.transpose(mean_wf_good, (0, 2, 1, 3))
            # remove sync channel
            if mean_wf_good.shape[2] == 385:
                mean_wf_good = mean_wf_good[:, :, :-1]

        waveforms.append(mean_wf_good)
        n_units_per_session[i] = mean_wf_good.shape[0]

    # concatenate waveforms
    waveform = np.concatenate(waveforms, axis=0)
    params["n_units"], session_id, session_switch, params["n_sessions"] = (
        get_session_data(n_units_per_session)
    )
    within_session = get_within_session(session_id, params)

    params["n_channels"] = waveform.shape[2]
    return waveform, session_id, session_switch, within_session, params


def calc_mean_wf_wrapper(
    ks_folder,
    extract_good_units_only=False,
    good_lbls=["good"],
    processing_drive="D:",
    overwrite=False,
    n_splits=2,
):
    mean_wf_path = os.path.join(ks_folder, f"mean_waveforms_{n_splits}split.npy")
    if os.path.exists(mean_wf_path) and not overwrite:
        print(f"Mean waveforms already exist at {mean_wf_path}. Skipping...")
        return

    # copy folder to processing drive, if needed
    probe_folder = os.path.dirname(ks_folder)
    cur_drive = probe_folder.split(os.sep)[0]
    new_probe_folder = probe_folder.replace(cur_drive, processing_drive)
    npx.copy_folder_with_progress(probe_folder, new_probe_folder)
    ks_folder = ks_folder.replace(cur_drive, processing_drive)

    # collect necessary information
    params = npx.load_params(ks_folder)

    spike_clusters = np.load(os.path.join(ks_folder, "spike_clusters.npy"))
    cluster_ids = np.arange(np.max(spike_clusters) + 1)
    n_clusters = len(cluster_ids)

    if extract_good_units_only:
        label_path = os.path.join(ks_folder, "cluster_group.tsv")
        labels = pd.read_csv(label_path, sep="\t", index_col="cluster_id")
        cluster_ids = labels[labels["label"] in good_lbls].index.values
    else:
        cluster_ids = np.arange(n_clusters)

    spike_times = np.load(os.path.join(ks_folder, "spike_times.npy"))
    data = npx.get_data_memmap(ks_folder)

    npx.calc_mean_wf_split(params, n_clusters, cluster_ids, spike_times, data, n_splits)


if __name__ == "__main__":
    # parameters
    root_folder = "Z:\Psilocybin\Cohort_0"
    ks_version = "4"
    match_threshold = 0.5
    subjects = ["T02"]
    processing_drive = "D:"
    save_matches = True
    tracking_params = default_params.get_default_param()
    good_lbls = ["good"]
    fp_max = 0.02
    n_splits = 2

    # subjects below root folder
    if subjects is None:
        subjects = [
            f
            for f in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, f))
        ]

    # get ks_folder for each probe
    pbar = tqdm(subjects, "Processing subjects...")
    for subject in pbar:
        pbar.set_description(f"Processing subject {subject}")
        subject_folder = os.path.join(root_folder, subject)
        if not os.path.exists(subject_folder):
            print(f"Subject folder {subject_folder} does not exist. Skipping...")
            continue
        ks_folders = npx.get_ks_folders(subject_folder, ks_version)

        # group folders by probes
        all_probe_folders = npx.get_probe_folders(ks_folders, catgt_only=True)

        pbar2 = tqdm(all_probe_folders, "Processing probes...", leave=False)
        for probe_num in pbar2:
            pbar2.set_description(f"Processing probe {probe_num}")

            # get kilosort folders with same channel positions
            probe_folders = npx.get_same_channel_positions(all_probe_folders[probe_num])
            # sort assuming file name starts with date YYYYMMDD
            probe_folders.sort(key=lambda x: x.split(os.sep)[-2].split("_")[0])

            if save_matches:
                save_dir = os.path.join(
                    subject_folder, f"UnitMatch_ks{ks_version}_imec{probe_num}"
                )
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = None

            tracking_params["KS_dirs"] = probe_folders

            # create average waveforms if needed
            for probe_folder in probe_folders:
                calc_mean_wf_wrapper(
                    probe_folder,
                    extract_good_units_only=False,
                    good_lbls=good_lbls,
                    processing_drive=processing_drive,
                    n_splits=n_splits,
                )

            # setup to run tracking
            mean_wf_paths, metrics_path, channel_pos = get_tracking_paths(
                probe_folders, n_splits
            )

            # update parameters based on probe geometry
            probe_params = get_probe_geometry(channel_pos[0], tracking_params)
            good_units = get_good_units(metrics_path, good_lbls=good_lbls)
            mean_wf, session_id, session_switch, within_session, probe_params = (
                load_good_units(good_units, mean_wf_paths, probe_params)
            )

            # create clus_info to contain all unit id/session related info
            clus_info = {
                "good_units": good_units,
                "session_switch": session_switch,
                "session_id": session_id,
                "original_ids": np.concatenate(good_units),
            }

            # make channel positions 3D
            if channel_pos[0].ndim == 2:
                # add y being all ones
                for i in range(len(channel_pos)):
                    channel_pos[i] = np.insert(
                        channel_pos[i], 0, np.ones(channel_pos[i].shape[0]), axis=1
                    )

            # run UnitMatch
            matches = unit_match(
                mean_wf,
                channel_pos,
                clus_info,
                session_switch,
                within_session,
                match_threshold,
                probe_params,
                n_splits,
                save_dir=save_dir,
            )
            prob_df = pd.read_csv(os.path.join(save_dir, "MatchTable.csv"))
            dup_rem_df = cross_sess_pairs(prob_df)
            tn_rem_df = remove_transpose(dup_rem_df)

            roc_df = estFP_vs_matchfrac(save_dir, threshold=None, n_trial=50)
            roc_name = "est_roc_df.csv"
            roc_df.to_csv(os.path.join(save_dir, roc_name), index=None)

            estfrac_df = est_matchfrac(save_dir, match_threshold)
            adj_thresh_df = thresh_for_FP(roc_df, estfrac_df, fp_max, match_threshold)

            adj_th_name = "adj_th_df.csv"
            adj_thresh_df.to_csv(os.path.join(save_dir, adj_th_name), index=None)

            low_thresh_name = "low_th_df.csv"
            tn_rem_df.to_csv(os.path.join(save_dir, low_thresh_name), index=None)

            th_applied_df = apply_threshold(tn_rem_df, adj_thresh_df)

            cross_sum_df = build_cross_sum(th_applied_df, "int")
            cross_sum_df.to_csv(os.path.join(save_dir, "cross_sum_df.csv"), index=None)
