import os
import shutil
from pathlib import Path

import cupy as cp
import npx_utils as npx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mode
from tqdm import tqdm

import UnitMatchPy.default_params as default_params
import UnitMatchPy.extract_raw_data as erd
from UnitMatchPy.run import unit_match
from UnitMatchPy.utils import get_probe_geometry, get_session_data, get_within_session


def get_tracking_paths(probe_folders):
    mean_wf_paths = []
    metrics_paths = []
    channel_pos = []

    for ks_folder in probe_folders:
        # check if it is in KS directory
        # if os.path.exists(os.path.join(ks_folder, "RawWaveforms")):
        #     mean_wf_paths.append(os.path.join(ks_folder, "RawWaveforms"))
        # else:
        mean_wf_paths.append(os.path.join(ks_folder, "mean_waveforms.npy"))

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
                mean_wf_good = np.transpose(mean_wf_good, (0, 2, 1))
            # remove sync channel
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


def calc_average_waveforms(
    ks_folders,
    ks_version,
    samples_before=20,
    samples_after=62,
    sample_amount=1000,
    extract_good_units_only=False,
    processing_drive="D:",
    overwrite=False,
):
    spike_width = samples_before + samples_after
    half_width = np.floor(spike_width / 2).astype(int)
    max_width = np.floor(spike_width / 2).astype(int)

    spike_ids, spike_times, _ = erd.extract_KS_data(
        ks_folders, extract_good_units_only=extract_good_units_only
    )
    for sid, ks_folder in enumerate(ks_folders):
        # check if need to calculate
        # if not overwrite and os.path.exists(os.path.join(ks_folder, "RawWaveforms")):
        #     continue

        probe_folder = os.path.dirname(ks_folder)
        cur_drive = probe_folder.split(os.sep)[0]
        new_probe_folder = probe_folder.replace(cur_drive, processing_drive)
        npx.copy_folder_with_progress(probe_folder, new_probe_folder)
        ks_folder = ks_folder.replace(cur_drive, processing_drive)

        n_units = len(np.unique(spike_ids[sid]))
        params = npx.load_params(ks_folder)
        data_path = npx.get_binary_path(ks_folder, params)
        meta_path = npx.get_meta_path(ks_folder, params)

        # load metadata
        meta_data = erd.read_meta(Path(meta_path))
        n_elements = int(meta_data["fileSizeBytes"]) / 2
        n_channels_tot = int(meta_data["nSavedChans"])

        # create memmap to raw data, for that session
        data = np.memmap(
            data_path,
            dtype="int16",
            shape=(int(n_elements / n_channels_tot), n_channels_tot),
        )

        # Remove spike which won't have a full wavefunction recorded
        spike_ids_tmp = np.delete(
            spike_ids[sid],
            np.logical_or(
                (spike_times[sid] < max_width),
                (spike_times[sid] > (data.shape[0] - max_width)),
            ),
        )
        spike_times_tmp = np.delete(
            spike_times[sid],
            np.logical_or(
                (spike_times[sid] < max_width),
                (spike_times[sid] > (data.shape[0] - max_width)),
            ),
        )

        sample_idx = erd.get_sample_idx(
            spike_times_tmp,
            spike_ids_tmp,
            sample_amount,
            units=np.unique(spike_ids[sid]),
        )

        if ks_version == 4:
            n_channels = n_channels_tot - 1  # TODO read from meta?
            avg_waveforms = Parallel(
                n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None
            )(
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
            # save all waveforms in a single file
        else:
            avg_waveforms = Parallel(
                n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None
            )(
                delayed(erd.extract_a_unit)(
                    sample_idx[uid],
                    data,
                    half_width,
                    spike_width,
                    n_channels,
                    sample_amount,
                )
                for uid in range(n_units)
            )
            avg_waveforms = np.asarray(avg_waveforms)

        # Save in file named 'RawWaveforms' in the KS Directory
        erd.save_avg_waveforms(
            avg_waveforms,
            ks_folder.replace(processing_drive, cur_drive),
            good_units[sid],
            extract_good_units_only=extract_good_units_only,
        )

        del data

        # if cur_drive != processing_drive:
        #     shutil.rmtree(new_probe_folder)


def calc_mean_waveforms2(
    ks_folder,
    ks_version,
    samples_before=20,
    samples_after=62,
    sample_amount=1000,
    extract_good_units_only=False,
    processing_drive="D:",
    overwrite=False,
):
    probe_folder = os.path.dirname(ks_folder)
    cur_drive = probe_folder.split(os.sep)[0]
    new_probe_folder = probe_folder.replace(cur_drive, processing_drive)
    npx.copy_folder_with_progress(probe_folder, new_probe_folder)
    ks_folder = ks_folder.replace(cur_drive, processing_drive)

    meta_path = npx.get_meta_path(ks_folder)
    meta = npx.read_meta(meta_path)
    spike_times = np.load(os.path.join(ks_folder, "spike_times.npy"))
    spike_clusters = np.load(os.path.join(ks_folder, "spike_clusters.npy"))
    cluster_ids = np.arange(np.max(spike_clusters) + 1)
    n_clusters = len(cluster_ids)
    n_channels = npx.get_ap_data_channel_count(meta) + 1 # TODO
    data = npx.get_data_memmap(ks_folder)
    times_multi = npx.find_times_multi(
        spike_times, spike_clusters, cluster_ids, data, samples_before, samples_after
    )

    bits_to_uV = npx.get_bits_to_uV(meta)
    bits_to_uV = cp.float32(bits_to_uV)  # convert to cupy float32
    mean_wf = cp.zeros((n_clusters, n_channels, samples_before + samples_after, 2))

    for i in tqdm(cluster_ids, desc="Calculating mean waveforms"):
        spikes = npx.extract_spikes(
            data,
            times_multi,
            i,
            samples_before,
            samples_after,
            sample_amount,
        )

        if len(spikes) > 0:  # edge case
            spikes_cp = cp.array(spikes, dtype=cp.float32)
            # calc mean waveform of first and second half of spikes
            mean_wf[i, :, :, 0] = cp.mean(spikes_cp[: len(spikes_cp) // 2], axis=0)
            mean_wf[i, :, :, 1] = cp.mean(spikes_cp[len(spikes_cp) // 2 :], axis=0)

    # convert mean_wf and std_wf to uV
    mean_wf *= bits_to_uV

    tqdm.write("Saving mean and std waveforms...")
    cp.save(
        os.path.join(
            ks_folder.replace(processing_drive, cur_drive),
            "RawWaveforms",
            "AllUnits_Mean_RawSpikes.npy",
        ),
        mean_wf,
    )


if __name__ == "__main__":
    # Parameters
    root_folder = "Z:\Psilocybin\Cohort_1"
    ks_version = 4
    match_threshold = 0.75
    subjects = "T11"
    processing_drive = "D:"
    save_matches = True
    tracking_params = default_params.get_default_param()

    # Subjects below root folder
    if subjects is None:
        subjects = [
            f
            for f in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, f))
        ]
    else:
        subject_folder = os.path.join(root_folder, subjects)
        if not os.path.exists(subject_folder):
            raise FileNotFoundError(f"Subject folder {subject_folder} not found")
        subjects = [subject_folder]

    # get ks_folder for each probe
    pbar = tqdm(subjects, "Processing subjects...")
    for subject in pbar:
        pbar.set_description(f"Processing subject {subject}")
        subject_folder = os.path.join(root_folder, subject)
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
                output_path = os.path.join(subject_folder, f"UnitMatch_imec{probe_num}")
                os.makedirs(output_path, exist_ok=True)
            else:
                output_path = None

            tracking_params["KS_dirs"] = probe_folders

            # create average waveforms if needed
            for probe_folder in probe_folders:
                calc_mean_waveforms2(
                    probe_folder,
                    ks_version,
                    extract_good_units_only=False,
                    processing_drive=processing_drive,
                )

            # setup to run tracking
            mean_wf_paths, metrics_path, channel_pos = get_tracking_paths(probe_folders)

            # update parameters based on probe geometry
            probe_params = get_probe_geometry(channel_pos[0], tracking_params)
            good_units = get_good_units(metrics_path, good_lbls=["good"])
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

            # update waveform and channel_pos for unit match as needed
            if mean_wf.ndim == 3:  # make cv0 == cv1
                mean_wf = mean_wf[:, :, :, np.newaxis]
                mean_wf = np.concatenate((mean_wf, mean_wf), axis=3)

            if channel_pos[0].ndim == 2:  # add y being all ones
                for i in range(len(channel_pos)):
                    # This makes position 3-D by inserting an axis of all one's in the first axis, to allow easy extension to 3-D coords
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
                save_dir=output_path,
            )
