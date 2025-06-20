import datetime
import json
import os
import shutil

import npx_utils as npx
import numpy as np
import spikeinterface.full as si
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def _get_combined_recording(folder1, folder2, stream, duration_s):
    rec1 = si.read_spikeglx(folder_path=folder1, stream_id=stream)
    rec2 = si.read_spikeglx(folder_path=folder2, stream_id=stream)

    # check channel positions are same
    if not np.array_equal(rec1.get_channel_locations(), rec2.get_channel_locations()):
        raise ValueError("Channel locations do not match between recordings")
    rec1_end = rec1.time_slice(
        start_time=rec1.get_total_duration() - duration_s,
        end_time=rec1.get_total_duration(),
    )
    rec2_start = rec2.time_slice(
        start_time=rec2.get_start_time(),
        end_time=rec2.get_start_time() + duration_s,
    )

    rec = si.concatenate_recordings([rec1_end, rec2_start])
    rec = rec.astype("float32")
    return rec


def _calc_mean_displacement(motion, drift_folder):
    middle = motion.displacement[0].shape[0] // 2
    before = motion.displacement[0][:middle].mean(axis=0)
    after = motion.displacement[0][middle:].mean(axis=0)
    displacement = after - before
    if drift_folder is not None:
        np.save(
            os.path.join(drift_folder, "motion", "mean_displacement.npy"), displacement
        )
    return displacement


def _calc_channels(motion, motion_info, rec, drift_folder):
    win_size = motion_info["parameters"]["estimate_motion_kwargs"]["win_step_um"]
    z_loc = rec.get_channel_locations()[:, 1]
    starts = motion.spatial_bins_um - win_size / 2
    ends = motion.spatial_bins_um + win_size / 2
    # ensure cover the full range
    starts[0] = z_loc.min()
    ends[-1] = z_loc.max()
    bounds = np.stack([starts, ends], axis=1)
    channels = []
    for start, end in bounds:
        mask = (z_loc >= start) & (z_loc <= end)
        channels.append(np.where(mask)[0])
    if drift_folder is not None:
        np.save(os.path.join(drift_folder, "motion", "spatial_bounds_um.npy"), bounds)
        np.savez(os.path.join(drift_folder, "motion", "channels.npz"), *channels)
    return channels


def estimate_drift(
    run_folders,
    probe_ids=None,
    duration_s=5 * 60,
    save_path=None,
    overwrite=False,
    plot=False,
    rigid=False,
):
    """
    Estimate drift between pairs of recordings and save the results.
    Parameters:
        run_folders (list): List of folders containing the recordings.
        probe_ids (list or None): List of probe IDs to process. If None, all probes are processed.
        duration_s (int): Duration in seconds to consider for drift estimation.
        save_path (str or None): Path to save the results. If None, results are not saved.
        overwrite (bool): Whether to overwrite existing results.
        plot (bool): Whether to plot the results.
        rigid (bool): Whether to use rigid motion estimation.
    Returns:
        all_displacements (np.ndarray): Array of displacements for each pair of recordings per probe. (# probes, # run_folders - 1, # segments). # segments is 1 if rigid, 7 otherwise.
        channels (list): Channel ids associated with each displacement per probe, if rigid then will be all channels.
    """
    # get probe_ids if not provided
    if probe_ids is None:
        # get folders in run_folders
        probe_ids = [
            int(folder.split("imec")[-1])
            for folder in os.listdir(run_folders[0])
            if ("imec" in folder)
            and os.path.isdir(os.path.join(run_folders[0], folder))
        ]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    job_kwargs = dict(chunk_duration="1s", n_jobs=10, progress_bar=True)

    all_displacements = [np.array([], dtype=np.float32)] * len(probe_ids)
    channels = [[]] * len(probe_ids)

    # loop through probe_ids
    for i, probe_id in tqdm(
        enumerate(probe_ids), desc="Processing probes", unit="probe"
    ):
        stream = f"imec{probe_id}.ap"

        if (
            not overwrite
            and (save_path is not None)
            and os.path.exists(
                os.path.join(save_path, f"imec{probe_id}", "params.json")
            )
        ):
            with open(
                os.path.join(save_path, f"imec{probe_id}", "params.json"), "r"
            ) as f:
                params = json.load(f)
            if (
                params["folders"] == run_folders
                and params["duration_s"] == duration_s
                and params["drift_type"] == "dredge"
                and params["rigid"] == rigid
            ):
                tqdm.write(f"Probe {probe_id} already processed, loading data.")
                # load existing displacements and channels
                all_displacements[i] = np.load(
                    os.path.join(save_path, f"imec{probe_id}", "all_displacements.npy")
                )
                probe_channels_npz = np.load(
                    os.path.join(save_path, f"imec{probe_id}", "channels.npz")
                )
                probe_channels = list(probe_channels_npz.values())
                channels[i] = probe_channels
                continue

        if save_path is not None:
            probe_path = os.path.join(save_path, f"imec{probe_id}")
            os.makedirs(probe_path, exist_ok=True)
        # loop through pairs of folders
        for j in tqdm(
            range(len(run_folders) - 1), desc="Estimating drift", unit="pair"
        ):
            # get recordings
            folder1 = run_folders[j]
            folder2 = run_folders[j + 1]
            rec = _get_combined_recording(folder1, folder2, stream, duration_s)

            drift_folder = (
                os.path.join(probe_path, f"pair_{j}_{j + 1}")
                if save_path is not None
                else None
            )
            if drift_folder is not None and os.path.exists(drift_folder):
                shutil.rmtree(drift_folder)

            estimate_motion_kwargs = dict(
                method="dredge_ap",
                direction="y",
                rigid=rigid,
                win_shape="gaussian",
                win_step_um=400.0,
                win_scale_um=400.0,
                win_margin_um=None,
            )

            motion, motion_info = si.compute_motion(
                rec,
                preset="dredge",
                output_motion_info=True,
                folder=drift_folder,
                overwrite=overwrite,
                estimate_motion_kwargs=estimate_motion_kwargs,
                **job_kwargs,
            )
            displacement = _calc_mean_displacement(motion, drift_folder)
            probe_channels = _calc_channels(motion, motion_info, rec, drift_folder)

            if all_displacements[i].size == 0:
                all_displacements[i] = np.zeros(
                    (len(run_folders) - 1, displacement.shape[0]), dtype=np.float32
                )
            all_displacements[i][j, :] = displacement
            channels[i] = probe_channels
            if plot:
                fig = plt.figure(figsize=(15, 10))
                si.plot_motion_info(motion_info, rec, figure=fig, color_amplitude=True)

        # save displacements from this probe
        np.save(
            os.path.join(probe_path, "all_displacements.npy"),
            all_displacements[i],
        )
        np.savez(os.path.join(probe_path, "channels.npz"), *channels[i])
        params = {
            "folders": run_folders,
            "duration_s": duration_s,
            "drift_type": "dredge",
            "rigid": False,
            "time": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        }
        with open(os.path.join(probe_path, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
    return all_displacements, channels


if __name__ == "__main__":
    subject_folder = r"D:\Psilocybin\Cohort_2b\T17"
    overwrite = False
    probe_ids = None
    duration_s = 5 * 60
    plot = False
    rigid = False

    save_path = os.path.join(subject_folder, "drift")
    if rigid:
        save_path += "_rigid"
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
    all_displacements, channels = estimate_drift(
        run_folders,
        probe_ids=probe_ids,
        duration_s=duration_s,
        save_path=save_path,
        overwrite=overwrite,
        plot=plot,
        rigid=rigid,
    )
