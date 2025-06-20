import os
import shutil

import npx_utils as npx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import UnitMatchPy.default_params as default_params
import UnitMatchPy.extract_raw_data as erd
from UnitMatchPy.cross_sess_sum import (
    apply_threshold,
    build_cross_sum,
    cross_sess_pairs,
    remove_transpose,
)
from UnitMatchPy.estimate_fp import est_matchfrac, estFP_vs_matchfrac, thresh_for_FP
from UnitMatchPy.run import unit_match
from UnitMatchPy.utils import (
    get_probe_geometry,
    get_session_data,
    get_within_session,
    load_good_waveforms,
)


def get_tracking_paths(probe_folders, n_splits):
    mean_wf_paths = []
    metrics_paths = []
    channel_pos = []

    for ks_folder in probe_folders:
        # mean_wf_paths.append(os.path.join(ks_folder, "RawWaveforms"))
        mean_wf_paths.append(
            os.path.join(ks_folder, f"mean_waveforms_{n_splits}split.npy")
        )
        metrics_path = os.path.join(ks_folder, "cluster_group.tsv")
        metrics_paths.append(metrics_path)
        channel_pos.append(np.load(os.path.join(ks_folder, "channel_positions.npy")))

    return mean_wf_paths, metrics_paths, channel_pos


def get_good_units(metrics_paths, good_lbls=["good"]):
    good_units = []
    for metrics_path in metrics_paths:
        metrics = pd.read_csv(metrics_path, sep="\t", index_col="cluster_id")
        if "label" in metrics.columns:
            good_units.append(metrics[metrics["label"].isin(good_lbls)].index.values)
        else:
            good_units.append(metrics[metrics["KSLabel"].isin(good_lbls)].index.values)
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

            # update mean_wf to match (n_units, spike_width, n_channels, 2)
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
    pre_samples=20,
    post_samples=62,
    max_spikes=1000,
):
    mean_wf_path = os.path.join(ks_folder, f"mean_waveforms_{n_splits}split.npy")
    if os.path.exists(mean_wf_path) and not overwrite:
        # double check mean_waveforms is same shape as the split in case I didnt recalc
        mean_wf_split = np.load(mean_wf_path)
        mean_wf = np.load(os.path.join(ks_folder, "mean_waveforms.npy"))
        if mean_wf_split.shape[0] == mean_wf.shape[0]:
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
    times_multi = npx.find_times_multi(
        spike_times, spike_clusters, cluster_ids, data, pre_samples, post_samples
    )

    params["KS_folder"] = ks_folder
    params["n_chan"] = params.pop("n_channels_dat")
    params["pre_samples"] = pre_samples
    params["post_samples"] = post_samples
    params["max_spikes"] = max_spikes
    params["meta_path"] = npx.get_meta_path(ks_folder)
    npx.calc_mean_wf_split(params, n_clusters, cluster_ids, times_multi, data, n_splits)


def calc_avg_wf_wrapper(
    ks_folder,
    samples_before=20,
    samples_after=62,
    sample_amount=1000,
    extract_good_units_only=False,
):
    # if os.path.exists(os.path.join(ks_folder, "RawWaveforms")):
    #     print(
    #         f"Raw waveforms already exist at {os.path.join(ks_folder, 'RawWaveforms')}. Skipping..."
    #     )
    #     return
    # delete raw waveforms folder if it exists
    if os.path.exists(os.path.join(ks_folder, "RawWaveforms")):
        shutil.rmtree(os.path.join(ks_folder, "RawWaveforms"))

    spike_width = samples_before + samples_after
    max_width = np.floor(spike_width / 2).astype(int)
    n_channels = 384

    spike_ids = np.load(os.path.join(ks_folder, "spike_clusters.npy"))
    spike_times = np.load(os.path.join(ks_folder, "spike_times.npy"))
    good_units = get_good_units([os.path.join(ks_folder, "cluster_group.tsv")])[0]

    params = npx.load_params(ks_folder)
    meta_path = npx.get_meta_path(ks_folder)
    meta = npx.read_meta(meta_path)

    n_elements = int(meta["fileSizeBytes"]) / 2
    n_channels_tot = int(meta["nSavedChans"])
    n_units = np.max(spike_ids) + 1

    data = np.memmap(
        params["dat_path"],
        dtype="int16",
        shape=(int(n_elements / n_channels_tot), n_channels_tot),
        mode="r",
    )
    spike_ids_tmp = np.delete(
        spike_ids,
        np.logical_or(
            (spike_times < max_width), (spike_times > (data.shape[0] - max_width))
        ),
    )
    spike_times_tmp = np.delete(
        spike_times,
        np.logical_or(
            (spike_times < max_width), (spike_times > (data.shape[0] - max_width))
        ),
    )
    sample_idx = erd.get_sample_idx(
        spike_times_tmp, spike_ids_tmp, sample_amount, units=np.unique(spike_ids)
    )
    avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None)(
        delayed(erd.extract_a_unit_KS4)(
            sample_idx[uid],
            data,
            samples_before,
            samples_after,
            spike_width,
            n_channels,
            sample_amount,
        )
        for uid in range(n_units)
    )
    avg_waveforms = np.asarray(avg_waveforms)

    erd.save_avg_waveforms(
        avg_waveforms,
        ks_folder,
        good_units=good_units,
        extract_good_units_only=extract_good_units_only,
    )
    del data


if __name__ == "__main__":
    # parameters
    # folders = [
    #     r"D:\Psilocybin\Cohort_2b\T16\20250522_T16_baseline1\supercat_20250522_T16_baseline1_g0",
    #     r"D:\Psilocybin\Cohort_2b\T16\20250523_T16_baseline2\supercat_20250523_T16_baseline2_g0",
    #     r"D:\Psilocybin\Cohort_2b\T16\20250524_T16_acute\supercat_20250524_T16_acute_g0",
    #     r"D:\Psilocybin\Cohort_2b\T16\20250525_T16_day1\catgt_20250525_T16_day1_g0",
    #     r"D:\Psilocybin\Cohort_2b\T16\20250603_T16_day10\catgt_20250603_T16_day10_g0",
    # ]
    subject_folder = r"D:\Psilocybin\Cohort_2b\Madison"
    displacement_path = os.path.join(subject_folder, "drift")
    save_dir = r"D:\Psilocybin\Cohort_2b\T16\UnitMatch_AvgWf"
    ks_version = "4"
    match_threshold = 0.5  # TODO 0.75?
    processing_drive = "D:"
    tracking_params = default_params.get_default_param()
    good_lbls = ["good"]
    fp_max = 0.02
    n_splits = 2
    probe_ids = None

    day_folders = [
        os.path.join(subject_folder, folder)
        for folder in os.listdir(subject_folder)
        if os.path.isdir(os.path.join(subject_folder, folder))
        and ("SvyPrb" not in folder)
        and ("old" not in folder)
    ]
    run_folders = []
    for day_folder in day_folders:
        possible_run_folders = [
            os.path.join(day_folder, folder)
            for folder in os.listdir(day_folder)
            if npx.is_run_folder(folder)
        ]
        # find one with supercat
        supercat_folders = [
            folder for folder in possible_run_folders if "supercat" in folder
        ]
        if len(supercat_folders) > 1:
            raise ValueError(
                f"Multiple supercat folders found in {day_folder}: {supercat_folders}"
            )
        if len(supercat_folders) == 1:
            run_folders.append(supercat_folders[0])
        else:
            # find catgt folders
            catgt_folders = [
                folder for folder in possible_run_folders if "catgt" in folder
            ]
            if len(catgt_folders) > 1:
                raise ValueError(
                    f"Multiple catgt folders found in {day_folder}: {catgt_folders}"
                )
            if len(catgt_folders) == 1:
                run_folders.append(catgt_folders[0])
            else:
                if len(possible_run_folders) == 1:
                    run_folders.append(possible_run_folders[0])
                elif len(possible_run_folders) == 0:
                    continue
                else:
                    raise ValueError(
                        f"Multiple run folders found in {day_folder}: {possible_run_folders}"
                    )
    run_folders.sort()

    os.makedirs(save_dir, exist_ok=True)

    ks_folders = []
    for folder in run_folders:
        ks_folders.extend(npx.get_ks_folders(folder, ks_version))

    # group folders by probes
    all_probe_folders = npx.get_probe_folders(ks_folders)
    if probe_ids is not None:
        all_probe_folders = {
            probe_id: all_probe_folders[probe_id]
            for probe_id in probe_ids
            if probe_id in all_probe_folders
        }

    pbar2 = tqdm(all_probe_folders, "Processing probes...", leave=False)
    for probe_num in pbar2:
        pbar2.set_description(f"Processing probe {probe_num}")

        probe_save_dir = os.path.join(save_dir, f"imec{probe_num}")
        os.makedirs(probe_save_dir, exist_ok=True)

        probe_folders = all_probe_folders[probe_num]
        # sort assuming file name starts with date YYYYMMDD
        probe_folders.sort(key=lambda x: x.split(os.sep)[-2].split("_")[0])

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
            # calc_avg_wf_wrapper(probe_folder)

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

        # load file
        # Replace z-drift estimates with values from manual inspection, Dredge, or other
        # First value is drift between 2nd recording and first, second value is drift from 3rd to 2nd, etc.
        displacement_file = os.path.join(
            displacement_path, f"imec{probe_num}", f"all_displacements.npy"
        )
        displacements = np.load(displacement_file)
        drift_est_all = np.zeros((len(probe_folders) - 1, displacements.shape[1], 3))
        drift_est_all[:, :, 2] = displacements
        drift_channels_npz = np.load(
            os.path.join(displacement_path, f"imec{probe_num}", "channels.npz")
        )
        drift_channels = list(drift_channels_npz.values())
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
            save_dir=probe_save_dir,
            est_drift=drift_est_all,
            drift_channels=drift_channels,
        )
        prob_df = pd.read_csv(os.path.join(probe_save_dir, "MatchTable.csv"))
        dup_rem_df = cross_sess_pairs(prob_df)
        tn_rem_df = remove_transpose(dup_rem_df)

        roc_df = estFP_vs_matchfrac(probe_save_dir, threshold=None, n_trial=50)
        roc_name = "est_roc_df.csv"
        roc_df.to_csv(os.path.join(probe_save_dir, roc_name), index=None)

        estfrac_df = est_matchfrac(probe_save_dir, match_threshold)
        adj_thresh_df = thresh_for_FP(roc_df, estfrac_df, fp_max, match_threshold)

        adj_th_name = "adj_th_df.csv"
        adj_thresh_df.to_csv(os.path.join(probe_save_dir, adj_th_name), index=None)

        low_thresh_name = "low_th_df.csv"
        tn_rem_df.to_csv(os.path.join(probe_save_dir, low_thresh_name), index=None)

        th_applied_df = apply_threshold(tn_rem_df, adj_thresh_df)

        cross_sum_df = build_cross_sum(th_applied_df, "int")
        cross_sum_df.to_csv(
            os.path.join(probe_save_dir, "cross_sum_df.csv"), index=None
        )
