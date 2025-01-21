import numpy as np
from matplotlib import pyplot as plt

import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.overlord as ov
import UnitMatchPy.save_utils as su
from UnitMatchPy.utils import evaluate_output


def unit_match(
    mean_wf,
    channel_pos,
    clus_info,
    session_switch,
    within_session,
    match_threshold,
    params,
    save_dir=None,
):
    # Extract waveforms properties
    extracted_wave_properties = ov.extract_parameters(
        mean_wf, channel_pos, clus_info, params
    )

    # Calculate metric scores from waveform properties. Estimate drift and correct for this.
    total_score, candidate_pairs, scores_to_include, predictors = (
        ov.extract_metric_scores(
            extracted_wave_properties, session_switch, within_session, params, niter=2
        )
    )

    # Use Bayes classifier to determine the probability of each pair being a match
    prior_match = 1 - (
        params["n_expected_matches"] / params["n_units"] ** 2
    )  # freedom of choose in prior prob
    priors = np.array((prior_match, 1 - prior_match))

    # Construct distributions (kernels) for Naive Bayes Classifier
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = params["score_vector"]
    parameter_kernels = np.full(
        (len(score_vector), len(scores_to_include), len(cond)), np.nan
    )

    parameter_kernels = bf.get_parameter_kernels(
        scores_to_include, labels, cond, params, add_one=1
    )

    # Get probability of each pair of being a match
    probability = bf.apply_naive_bayes(
        parameter_kernels, priors, predictors, params, cond
    )

    output_prob_matrix = probability[:, 1].reshape(params["n_units"], params["n_units"])

    evaluate_output(
        output_prob_matrix,
        params,
        within_session,
        session_switch,
        match_threshold,
    )

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    matches = np.argwhere(output_threshold == 1)

    # plt.imshow(output_threshold, cmap="Greys")
    # plt.show()

    if save_dir is not None:
        UIDs = aid.assign_unique_id(output_prob_matrix, params, clus_info)

        su.save_to_output(
            save_dir,
            scores_to_include,
            matches,
            output_prob_matrix,
            extracted_wave_properties["avg_centroid"],
            extracted_wave_properties["avg_waveform"],
            extracted_wave_properties["avg_waveform_per_tp"],
            extracted_wave_properties["max_site"],
            total_score,
            output_threshold,
            clus_info,
            params,
            UIDs=UIDs,
            matches_curated=None,
            save_match_table=True,
        )

    return matches
