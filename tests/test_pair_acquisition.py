"""Tests for label-free pair acquisition and RankNet oracle generation."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts/pair_acquisition.py"
)
SCRIPT_DIR = MODULE_PATH.parent


def load_module():
    """Load the acquisition module without importing the TensorFlow pipeline."""
    spec = importlib.util.spec_from_file_location("pair_acquisition", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bald_is_zero_when_all_model_samples_agree() -> None:
    """Identical comparison probabilities should contain no model information."""
    module = load_module()
    anchor_scores = np.array([2.0, 2.0, 2.0])
    candidate_scores = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    metrics = module.calculate_pair_metrics(anchor_scores, candidate_scores)

    np.testing.assert_allclose(metrics["bald"], 0.0, atol=1e-15)


def test_bald_is_positive_when_model_samples_disagree() -> None:
    """Opposing confident comparisons should have positive mutual information."""
    module = load_module()
    metrics = module.calculate_pair_metrics(
        np.array([10.0, -10.0]),
        np.array([[0.0], [0.0]]),
    )

    assert metrics["bald"][0] > 0.69


def test_pair_direction_reversal_preserves_entropy_and_bald() -> None:
    """Swapping pair direction should complement pbar without changing scores."""
    module = load_module()
    left = np.array([-2.0, 0.2, 3.0, 1.0])
    right = np.array([1.0, -0.5, 1.5, 1.2])

    forward = module.calculate_pair_metrics(left, right[:, np.newaxis])
    reverse = module.calculate_pair_metrics(right, left[:, np.newaxis])

    np.testing.assert_allclose(forward["pbar"], 1.0 - reverse["pbar"])
    np.testing.assert_allclose(
        forward["predictive_entropy"], reverse["predictive_entropy"]
    )
    np.testing.assert_allclose(forward["bald"], reverse["bald"])


def test_candidate_filter_excludes_self_reverse_duplicate_and_evaluation_images() -> (
    None
):
    """Only the supplied training pool may contribute eligible new pairs."""
    module = load_module()
    existing = pd.DataFrame({"x1_image": ["train_b"], "x2_image": ["train_a"]})
    rankings = module.build_candidate_rankings(
        anchors=["train_a"],
        candidates=["train_a", "train_b", "train_c"],
        method="random",
        acquired_pairs=module.acquired_pair_keys(existing),
        seed=7,
    )

    assert [record.candidate for record in rankings["train_a"]] == ["train_c"]
    assert "valid_image" not in {record.candidate for record in rankings["train_a"]}
    assert "test_image" not in {record.candidate for record in rankings["train_a"]}


def test_selection_api_does_not_receive_mayo_labels() -> None:
    """Pair selection should finish before labels are passed to the oracle."""
    module = load_module()
    rankings = module.build_candidate_rankings(
        anchors=["a", "b"],
        candidates=["a", "b", "c"],
        method="random",
        acquired_pairs=set(),
        seed=11,
    )
    selected = module.select_ranked_pairs(rankings, 2, set())
    unlabeled = module.selected_pairs_frame(selected)

    assert "relative_label" not in unlabeled
    labeled = module.apply_mayo_oracle(unlabeled, {"a": 0, "b": 1, "c": 1})
    assert set(labeled["relative_label"]) <= {0.0, 0.5, 1.0}


@pytest.mark.parametrize(
    "method", ["original", "random", "predictive_entropy", "pair_bald"]
)
def test_candidate_chunk_size_does_not_change_selection(method: str) -> None:
    """Candidate chunking should not affect random or information-based choices."""
    module = load_module()
    scores = {
        "a": np.array([0.0, 1.0, 0.5]),
        "b": np.array([0.1, 0.9, 0.2]),
        "c": np.array([1.0, -1.0, 1.0]),
        "d": np.array([-0.5, 1.2, -0.2]),
    }
    arguments = {
        "anchors": ["a", "b"],
        "candidates": ["a", "b", "c", "d"],
        "method": method,
        "acquired_pairs": set(),
        "score_by_image": scores,
        "seed": 13,
    }

    small_chunks = module.build_candidate_rankings(chunk_size=1, **arguments)
    large_chunks = module.build_candidate_rankings(chunk_size=10, **arguments)
    selected_small = module.select_ranked_pairs(small_chunks, 2, set())
    selected_large = module.select_ranked_pairs(large_chunks, 2, set())

    assert [(record.anchor, record.candidate) for record in selected_small] == [
        (record.anchor, record.candidate) for record in selected_large
    ]


def test_budget_smaller_than_anchor_coverage_stops() -> None:
    """The selector must reject a budget that cannot cover every anchor."""
    module = load_module()
    rankings = module.build_candidate_rankings(
        anchors=["a", "b"],
        candidates=["a", "b", "c"],
        method="random",
        acquired_pairs=set(),
    )

    with pytest.raises(RuntimeError, match="each anchor"):
        module.select_ranked_pairs(rankings, 1, set())


def test_resume_excludes_every_previously_acquired_orientation() -> None:
    """A resumed round must not acquire old pairs in either direction."""
    module = load_module()
    previous = pd.DataFrame(
        {
            "x1_image": ["a", "c"],
            "x2_image": ["b", "d"],
        }
    )
    acquired = module.acquired_pair_keys(previous)
    rankings = module.build_candidate_rankings(
        anchors=["b"],
        candidates=["a", "b", "c", "d"],
        method="random",
        acquired_pairs=acquired,
    )

    assert "a" not in [record.candidate for record in rankings["b"]]


def test_materially_negative_bald_is_not_silently_clipped(monkeypatch) -> None:
    """Only roundoff-sized negative mutual information may be set to zero."""
    module = load_module()
    original_entropy = module.binary_entropy
    calls = 0

    def biased_entropy(probabilities):
        nonlocal calls
        calls += 1
        values = original_entropy(probabilities)
        if calls == 1:
            return values - 1e-5
        return values

    monkeypatch.setattr(module, "binary_entropy", biased_entropy)
    with pytest.raises(FloatingPointError, match="materially negative"):
        module.calculate_pair_metrics(
            np.array([0.0, 0.0]),
            np.array([[0.0], [0.0]]),
        )


def test_original_method_runs_and_resumes_without_duplicate_budget(tmp_path) -> None:
    """The original-range baseline should persist one exact, idempotent round."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "select_comparison_pairs", SCRIPT_DIR / "7_select_comparison_pairs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    train_csv = tmp_path / "train.csv"
    previous_pair_csv = tmp_path / "initial_pairs.csv"
    history_csv = tmp_path / "initial_anchors.csv"
    anchor_csv = tmp_path / "anchors.csv"
    pd.DataFrame(
        {
            "filename": ["a", "b", "c", "d"],
            "MayoLabel": [0, 1, 1, 3],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        {
            "x1_image": ["a"],
            "x2_image": ["b"],
            "x1_label": [0],
            "x2_label": [1],
            "relative_label": [0.0],
        }
    ).to_csv(previous_pair_csv, index=False)
    pd.DataFrame({"filename": ["a"]}).to_csv(history_csv, index=False)
    pd.DataFrame({"filename": ["b", "c", "d"]}).to_csv(anchor_csv, index=False)

    args = SimpleNamespace(
        experiment_root=tmp_path / "experiments",
        experiment_id="dummy",
        diagnostic=False,
        method="original",
        seed=5,
        round_number=1,
        fold=1,
        resume=False,
        train_csv=train_csv,
        previous_pair_csv=previous_pair_csv,
        anchor_history_csv=history_csv,
        anchor_csv=anchor_csv,
        uncertainty_csv=None,
        anchor_count=None,
        anchor_rate=0.05,
        candidate_scope="cumulative_selected",
        pair_budget=3,
        allow_cpu_smoke_test=False,
        cpu_smoke_max_images=16,
        cuda_visible_devices="0",
        mc_cache=None,
        checkpoint=None,
        image_dir=tmp_path,
        mc_samples=30,
        image_batch_size=1,
        candidate_chunk_size=2,
        negative_bald_tolerance=1e-12,
        partner_cap=0,
    )
    script.run_selection(args)
    output_dir = script.output_directory(args)
    cumulative = pd.read_csv(output_dir / "cumulative_pairs.csv")
    assert len(cumulative) == 4
    module = load_module()
    assert len(module.acquired_pair_keys(cumulative)) == 4

    args.resume = True
    script.run_selection(args)
    resumed = pd.read_csv(output_dir / "cumulative_pairs.csv")
    pd.testing.assert_frame_equal(cumulative, resumed)
