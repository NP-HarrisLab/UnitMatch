import os
import pickle
from collections import defaultdict

import npx_utils as npx
import numpy as np
import pandas as pd

from UnitMatchPy.save_utils import load_output


def load_units(save_dir, load_match_table=True):
    UM_scores, clus_info, param, match_prob, waveform_info, match_table = load_output(
        save_dir, load_match_table
    )
    # find matches
    # rows where match_table["Matches"] == 1
    matches = match_table[match_table["Matches"] == 1]
    # only want matches between recording sessions
    matches = matches[matches["RecSes 1"] != matches["RecSes 2"]]
    # read ks_folders.txt into list
    ks_folders_file = os.path.join(save_dir, "ks_folders.txt")
    if not os.path.exists(ks_folders_file):
        raise FileNotFoundError(f"File not found: {ks_folders_file}")
    with open(ks_folders_file, "r") as f:
        ks_folders = [line.strip() for line in f if line.strip()]
    return matches, ks_folders


def find_tracked_units(matches, sessions_to_track):
    graph = defaultdict(set)
    for _, row in matches.iterrows():
        u1 = (row["RecSes 1"], row["ID1"])
        u2 = (row["RecSes 2"], row["ID2"])
        graph[u1].add(u2)
        graph[u2].add(u1)

    tracked_chains = []

    def find_path(current_path):
        """
        A recursive function to find a valid path that follows the sessions_to_track list.
        """
        # Base case: If the path has a unit for every required session, we are done.
        if len(current_path) == len(sessions_to_track):
            tracked_chains.append([uid for _, uid in current_path])
            return

        # Determine the next session we need to find in the path
        last_unit_session, _ = current_path[-1]
        current_session_index = sessions_to_track.index(last_unit_session)
        next_session_to_find = sessions_to_track[current_session_index + 1]

        # Explore neighbors of the last unit in the path
        for neighbor_session, neighbor_uid in graph.get(current_path[-1], []):
            # Check if the neighbor is in the correct subsequent session
            if neighbor_session == next_session_to_find:
                # Continue the search from this valid neighbor
                find_path(current_path + [(neighbor_session, neighbor_uid)])

    start_session = sessions_to_track[0]

    # Find all UIDs present in the starting session
    start_uids = pd.concat(
        [
            matches.loc[matches["RecSes 1"] == start_session, "ID1"],
            matches.loc[matches["RecSes 2"] == start_session, "ID2"],
        ]
    ).unique()

    # Start a path-finding search from each unit in the starting session
    for start_uid in start_uids:
        start_node = (start_session, start_uid)
        find_path([start_node])

    return tracked_chains


if __name__ == "__main__":
    # Example usage
    save_dir = r"D:\Psilocybin\Cohort_2b\Madison\UnitMatch\imec0"
    matches, ks_folders = load_units(save_dir)
    sessions_to_track = [1, 5]
    tracked_units = find_tracked_units(matches, sessions_to_track)
    tracked_count = len(tracked_units)
    print("# Tracked Units:", tracked_count)
    good_counts = np.zeros(len(sessions_to_track), dtype=int)
    for i, session in enumerate(sessions_to_track):
        ks_folder = ks_folders[session - 1]
        labels = pd.read_csv(
            os.path.join(ks_folder, "cluster_group.tsv"),
            sep="\t",
            index_col="cluster_id",
        )
        n_units = (labels["label"] == "good").sum()
        print(f"# Units Session {session}: {n_units}")
        good_counts[i] = n_units
    # % units tracked based on least number of units in a session
    min_counts = min(good_counts)
    percentage_tracked = (tracked_count / min_counts) * 100
    print(f"Percentage of units tracked: {percentage_tracked:.2f}%")
