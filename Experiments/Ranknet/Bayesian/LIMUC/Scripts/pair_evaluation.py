"""Evaluation helpers for unseen-patient ranking predictions."""

import random

import numpy as np
import pandas as pd
from pair_acquisition import stable_sigmoid

DIRECTIONAL_MODES = ["all", "neighbor", "0_1", "1_2", "2_3"]


def make_balanced_reference(scores, labels, image_ids, seed=20191206):
    """Reproduce the existing class-balanced test reference construction."""
    sample_count = len(labels) // 4
    remainder = len(labels) - sample_count * 4
    rng = random.Random(seed)
    output_scores = []
    output_labels = []
    output_ids = []
    for grade in range(4):
        indexes = np.flatnonzero(labels == grade).tolist()
        if not indexes:
            raise RuntimeError(f"Mayo grade {grade} has no evaluation images.")
        target = sample_count + int(remainder >= grade + 1)
        if len(indexes) < target:
            indexes.extend(rng.choices(indexes, k=target - len(indexes)))
        elif len(indexes) > target:
            indexes = rng.sample(indexes, target)
        output_scores.append(scores[indexes])
        output_labels.append(labels[indexes])
        output_ids.append(image_ids[indexes])
    return (
        np.concatenate(output_scores),
        np.concatenate(output_labels),
        np.concatenate(output_ids),
    )


def make_evaluation_pairs(scores, labels, image_ids, seed=20191125):
    """Reproduce the existing grade-balanced random partner construction."""
    reference_scores, reference_labels, reference_ids = make_balanced_reference(
        scores, labels, image_ids
    )
    rng = random.Random(seed)
    used_pairs = set()
    partner_indexes = []
    for reference_id in reference_ids:
        while True:
            grade = rng.choice([0, 1, 2, 3])
            candidates = np.flatnonzero(labels == grade).tolist()
            candidate_index = rng.choice(candidates)
            candidate_id = image_ids[candidate_index]
            pair_key = tuple(sorted((str(reference_id), str(candidate_id))))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                partner_indexes.append(candidate_index)
                break
    partner_indexes = np.asarray(partner_indexes, dtype=int)
    return (
        reference_scores,
        reference_labels,
        reference_ids,
        scores[partner_indexes],
        labels[partner_indexes],
        image_ids[partner_indexes],
    )


def evaluate_predictions(prediction_data: pd.DataFrame) -> dict:
    """Evaluate directional ranking and report ties separately.

    Equal-grade pairs are excluded from every directional accuracy, matching the
    existing evaluation. They are reported with ``tie_mean_abs_probability_error``:
    the mean distance between ``sigmoid(mean_score_i - mean_score_j)`` and 0.5.

    Args:
        prediction_data (pd.DataFrame): Test image IDs, Mayo labels, and mean scores.

    Returns:
        dict: Counts and accuracies for the existing directional modes plus a
        separate same-grade ambiguity diagnostic.
    """
    required = {"filename", "label", "mean_score"}
    missing = required - set(prediction_data.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
    prediction_data = prediction_data.drop_duplicates("filename").reset_index(drop=True)
    scores = prediction_data["mean_score"].to_numpy(dtype=float)
    labels = prediction_data["label"].to_numpy(dtype=float)
    image_ids = prediction_data["filename"].astype(str).to_numpy()
    if not np.all(np.isfinite(scores)):
        raise ValueError("Evaluation scores contain NaN or infinity.")

    (
        reference_scores,
        reference_labels,
        _,
        partner_scores,
        partner_labels,
        _,
    ) = make_evaluation_pairs(scores, labels, image_ids)
    differences = reference_labels - partner_labels
    predictions = reference_scores > partner_scores
    expected = differences > 0
    masks = {
        "all": differences != 0,
        "neighbor": np.abs(differences) == 1,
    }
    masks["0_1"] = ((reference_labels == 0) & (partner_labels == 1)) | (
        (reference_labels == 1) & (partner_labels == 0)
    )
    masks["1_2"] = ((reference_labels == 1) & (partner_labels == 2)) | (
        (reference_labels == 2) & (partner_labels == 1)
    )
    masks["2_3"] = ((reference_labels == 2) & (partner_labels == 3)) | (
        (reference_labels == 3) & (partner_labels == 2)
    )

    result = {}
    for mode in DIRECTIONAL_MODES:
        mask = masks[mode]
        count = int(mask.sum())
        result[f"{mode}_pair_count"] = count
        result[f"{mode}_accuracy"] = (
            float(np.mean(predictions[mask] == expected[mask])) if count else np.nan
        )

    tie_mask = differences == 0
    tie_probabilities = stable_sigmoid(
        reference_scores[tie_mask] - partner_scores[tie_mask]
    )
    result["tie_pair_count"] = int(tie_mask.sum())
    result["tie_mean_abs_probability_error"] = (
        float(np.mean(np.abs(tie_probabilities - 0.5)))
        if len(tie_probabilities)
        else np.nan
    )
    result[
        "tie_evaluation_policy"
    ] = "excluded_from_directional_accuracy; reported_as_mean_abs_distance_from_p_0.5"
    return result
