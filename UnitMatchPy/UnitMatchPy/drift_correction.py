import os

import npx_utils as npx
import numpy as np
from cilantropy.curation import Curator
from dredge import dredge_ap, motion_util
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_shank_channel_idxs(channel_pos, shank_idx=None):
    #     channel_pos = np.load(os.path.join(ks_folders[0], "channel_positions.npy"))
    # channel_idxs = get_shank_channel_idxs(channel_pos, shank_choice)

    #  conc_data = []
    # conc_z_score = []
    # for ks_folder in tqdm(ks_folders):
    #     data = npx.get_data_memmap(ks_folder)
    #     shank_data = data[:, channel_idxs, :]
    #     # slice central snippet of data
    #     shank_data = shank_data[
    #         :,
    #         :,
    #         int(data.shape[2] / 2 - snip_length / 2) : int(
    #             data.shape[2] / 2 + snip_length / 2
    #         ),
    #     ]
    #     z_score = (shank_data - np.mean(shank_data, axis=2)) / np.std(
    #         shank_data, axis=2
    #     )
    #     conc_data.append(shank_data)
    #     conc_z_score.append(z_score)
    #     conc_data = np.concatenate(conc_data, axis=0)
    # conc_z_score = np.concatenate(conc_z_score, axis=0)
    # # save to disk
    # os.makedirs(save_dir, exist_ok=True)
    # np.save(
    #     os.path.join(save_dir, "chronic_data.npy"),
    #     conc_data,
    #     allow_pickle=False,
    # )
    # np.save(
    #     os.path.join(save_dir, "chronic_z_score.npy"),
    #     conc_z_score,
    #     allow_pickle=False,
    # )
    shank_cutoffs = [0, 168, 418, 668, 918]
    if shank_idx is not None:
        channel_idxs = np.where(
            (channel_pos[:, 0] > shank_cutoffs[shank_idx])
            & (channel_pos[:, 0] < shank_cutoffs[shank_idx + 1])
        )[0]
    else:
        max_count = -1
        best_channel_idxs = None
        for i in range(4):
            idxs = np.where(
                (channel_pos[:, 0] > shank_cutoffs[i])
                & (channel_pos[:, 0] < shank_cutoffs[i + 1])
            )[0]
            if len(idxs) > max_count:
                max_count = len(idxs)
                best_channel_idxs = idxs
        channel_idxs = best_channel_idxs
    if channel_idxs is None or len(channel_idxs) == 0:
        raise ValueError(f"No channels on shank {shank_idx}")
    return channel_idxs


def chronic_drift_correction(ks_folders, shank_choice=None, snip_length=600):
    all_amps = []
    all_depths = []
    all_times = []
    start_times = []
    bin_size = snip_length / 30000  # 30 kHz
    for i, ks_folder in tqdm(enumerate(ks_folders), "Extracting drift data..."):
        curator = Curator(ks_folder)
        n_samples = curator.raw_data.data.shape[0]
        start_sample = n_samples // 2 - snip_length // 2
        end_sample = n_samples // 2 + snip_length // 2
        probe_amps = []
        probe_depths = []
        probe_times = []
        start_time = i * bin_size
        good_ids = (
            curator.cluster_metrics["label"].isin(["good", "mua"]).index.to_list()
        )

        for unit_id in good_ids:
            spike_times = curator.times_multi[unit_id]
            mask = (spike_times >= start_sample) & (spike_times <= end_sample)
            filtered_times = spike_times[mask] / curator.params["sample_rate"]
            filtered_times -= start_sample / curator.params["sample_rate"]
            filtered_times += start_time
            if len(filtered_times) == 0:
                continue
            filtered_idxs = np.where(mask)[0]
            # TODO extract spikes
            spikes = curator.spikes[unit_id][filtered_idxs, :, :]
            amps = np.ptp(spikes, axis=2)
            peaks = np.argmax(amps, axis=1)
            depths = curator.channel_pos[peaks, 1]

            probe_amps.append(amps)
            probe_depths.append(depths)
            probe_times.append(filtered_times)

        all_amps.append(probe_amps)
        all_depths.append(probe_depths)
        all_times.append(probe_times)
        start_times.append(start_time)

    all_amps = np.concatenate(all_amps, axis=0)
    all_depths = np.concatenate(all_depths, axis=0)
    all_times = np.concatenate(all_times, axis=0)

    motion_est, extra = dredge_ap.register(
        all_amps, all_depths, all_times, rigid=True, bin_s=bin_size, device="cuda"
    )
    depths = motion_est.correct_s(all_times, all_depths)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        all_times.flatten(), all_depths.flatten(), s=1, c="k", alpha=0.5, label="raw"
    )
    plt.colorbar(scatter, ax=ax, shrink=0.25, label="depth (um)")
    lines = motion_util.plot_me_traces(motion_est, ax=ax, color="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Drift correction")
    ax.legend([lines[0]], ["motion estimate"], loc="lower right")


def lfp_drift_visualization(ks_folder, shank_choice, t_start, t_end):
    channel_pos = os.path.join(ks_folder, "channel_positions.npy")
    lfp = npx.get_lfp_memmap(ks_folder)


# lfp_drift_visualization(ks_folder,)
ks_folders = [
    r"D:\tracking_test\catgt_20241021_T09_OF_Hab_g0\20241021_T09_OF_Hab_g0_imec0\imec0_ks4",
    r"D:\tracking_test\catgt_20241022_T09_OF_Test1_g0\20241022_T09_OF_Test1_g0_imec0\imec0_ks4",
    r"D:\tracking_test\catgt_20241022_T09_OF_Test1_g0\20241022_T09_OF_Test1_g0_imec0\imec0_ks4",
]
chronic_drift_correction(ks_folders)
