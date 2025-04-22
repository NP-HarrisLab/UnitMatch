import os

import numpy as np

import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.metric_functions as mf
import UnitMatchPy.overlord as ov
import UnitMatchPy.save_utils as su
import UnitMatchPy.utils as utils


def unit_match(
    mean_wf,
    channel_pos,
    clus_info,
    session_switch,
    within_session,
    match_threshold,
    params,
    n_splits,
    save_dir=None,
    est_drift=np.asarray([]),
):
    # extract waveforms properties
    extracted_wave_properties = ov.extract_parameters(
        mean_wf, channel_pos, clus_info, params
    )

    # if est_drift is not all zeros or empty, apply drift correction to all session, then re-calculate metrics
    # each session needs to be corrected to match the first session -- so sum over the drift estimates
    # calling apply drift with sesseion id i corrects session i+1, so call only for 0:nsess-1
    n_sess = len(params["KS_dirs"])
    if np.any(est_drift):
        for i in range(n_sess - 1):
            curr_drift = np.sum(est_drift[0 : i + 1, :], axis=0)
            print(curr_drift)
            drifts, avg_waveform_per_tp, avg_centroid = mf.apply_drift_correction_basic(
                None, i, session_switch, avg_centroid, avg_waveform_per_tp, curr_drift
            )

    # calculate metric scores from waveform properties. Estimate drift and correct for this.
    total_score, candidate_pairs, scores_to_include, predictors, drifts = (
        ov.extract_metric_scores(
            extracted_wave_properties, session_switch, within_session, params, n_splits=n_splits
        )
    )

    # use Bayes classifier to determine the probability of each pair being a match
    prior_match = 1 - (
        params["n_expected_matches"] / params["n_units"] ** 2
    )  # freedom of choose in prior prob
    priors = np.array((prior_match, 1 - prior_match))

    # construct distributions (kernels) for Naive Bayes Classifier
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = params["score_vector"]
    parameter_kernels = np.full(
        (len(score_vector), len(scores_to_include), len(cond)), np.nan
    )
    np.save(os.path.join(save_dir, "score_vector.npy"), params["score_vector"])
    np.save(os.path.join(save_dir, "parameter_kernels.npy"), parameter_kernels)

    parameter_kernels = bf.get_parameter_kernels(
        scores_to_include, labels, cond, params, add_one=1
    )

    # get probability of each pair of being a match
    probability = bf.apply_naive_bayes(
        parameter_kernels, priors, predictors, params, cond
    )

    output_prob_matrix = probability[:, 1].reshape(params["n_units"], params["n_units"])

    utils.evaluate_output(
        output_prob_matrix,
        params,
        within_session,
        session_switch,
        match_threshold,
    )

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    matches = np.argwhere(((output_threshold * within_session)) == True)

    # plt.imshow(output_threshold, cmap="Greys")
    # plt.show()

    if save_dir is not None:
        UIDs = aid.assign_unique_id(output_prob_matrix, params, clus_info)

        su.save_to_output(
            save_dir,
            scores_to_include,
            matches,  # matches_curated
            output_prob_matrix,
            extracted_wave_properties["avg_centroid"],
            extracted_wave_properties["avg_waveform"],
            extracted_wave_properties["avg_waveform_per_tp"],
            extracted_wave_properties["max_site"],
            total_score,
            output_threshold,
            drifts,
            clus_info,
            params,
            UIDs=UIDs,
            matches_curated=None,
            save_match_table=True,
        )

    return matches
