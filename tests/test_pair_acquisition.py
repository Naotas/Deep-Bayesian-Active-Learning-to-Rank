"""Tests for label-free pair acquisition and RankNet oracle generation."""

import importlib.util
import json
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


@pytest.fixture
def selection_script():
    """Load the selection entry point without importing TensorFlow."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "select_comparison_pairs", SCRIPT_DIR / "7_select_comparison_pairs.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def final_round_args(tmp_path, selection_script, monkeypatch):
    """Create two remaining dummy anchors with a usual three-comparison budget."""
    image_ids = [f"dummy_{index:02}" for index in range(10)]
    train_csv = tmp_path / "train.csv"
    previous_csv = tmp_path / "previous.csv"
    history_csv = tmp_path / "history.csv"
    uncertainty_csv = tmp_path / "uncertainty.csv"
    pd.DataFrame(
        {
            "filename": image_ids,
            "MayoLabel": [index % 4 for index in range(10)],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        {
            "x1_image": [image_ids[0]],
            "x2_image": [image_ids[1]],
            "x1_label": [0],
            "x2_label": [1],
            "relative_label": [0],
        }
    ).to_csv(previous_csv, index=False)
    pd.DataFrame({"filename": image_ids[:8]}).to_csv(history_csv, index=False)
    pd.DataFrame(
        {
            "filename": image_ids,
            "var_score": list(range(10)),
        }
    ).to_csv(uncertainty_csv, index=False)
    scores = {
        image_id: np.array([index / 10, -index / 5, index + 1], dtype=float)
        for index, image_id in enumerate(image_ids)
    }
    monkeypatch.setattr(
        selection_script, "load_score_mapping", lambda *args: (scores, 0.0, False)
    )
    return SimpleNamespace(
        experiment_root=tmp_path / "experiments",
        experiment_id="dummy_final_round",
        diagnostic=False,
        method="random",
        seed=5,
        round_number=16,
        fold=3,
        resume=False,
        train_csv=train_csv,
        previous_pair_csv=previous_csv,
        anchor_history_csv=history_csv,
        anchor_csv=None,
        uncertainty_csv=uncertainty_csv,
        anchor_count=None,
        anchor_rate=0.3,
        candidate_scope=None,
        pair_budget=None,
        allow_cpu_smoke_test=False,
        cpu_smoke_max_images=16,
        cuda_visible_devices="0",
        mc_cache=None,
        checkpoint=None,
        image_dir=tmp_path,
        mc_samples=3,
        image_batch_size=1,
        candidate_chunk_size=2,
        negative_bald_tolerance=1e-12,
        partner_cap=0,
    )


@pytest.mark.parametrize(
    "method", ["original", "random", "predictive_entropy", "pair_bald"]
)
@pytest.mark.parametrize("pair_budget", [None, 5])
def test_final_round_keeps_pair_budget_and_resumes_without_duplicates(
    selection_script, final_round_args, method, pair_budget, capsys
):
    """A smaller final anchor set must not reduce any method's annotation budget."""
    args = final_round_args
    args.method = method
    args.pair_budget = pair_budget
    expected_budget = 3 if pair_budget is None else pair_budget
    selection_script.run_selection(args)
    directory = selection_script.output_directory(args)
    anchors = pd.read_csv(directory / "new_anchors.csv")
    assert anchors.filename.tolist() == ["dummy_09", "dummy_08"]
    pairs = pd.read_csv(directory / "new_pairs.csv")
    assert len(pairs) == expected_budget
    assert set(pairs.x1_image) == set(anchors.filename)
    assert (pairs.x1_image != pairs.x2_image).all()
    cumulative = pd.read_csv(directory / "cumulative_pairs.csv")
    assert len(cumulative) == expected_budget + 1
    assert len(selection_script.acquired_pair_keys(cumulative)) == len(cumulative)
    history = pd.read_csv(directory / "cumulative_anchors.csv")
    assert history.filename.nunique() == len(history) == 10
    configuration = json.loads((directory / "experiment_config.json").read_text())
    assert configuration["requested_anchor_count"] == 3
    assert configuration["anchor_count"] == 2
    assert configuration["remaining_anchor_count_before_selection"] == 2
    assert configuration["anchor_count_adjusted"] is True
    assert configuration["pair_budget"] == expected_budget
    assert f"new-pair budget remains {expected_budget}" in capsys.readouterr().out

    before = (directory / "cumulative_pairs.csv").read_bytes()
    args.resume = True
    selection_script.run_selection(args)
    assert (directory / "cumulative_pairs.csv").read_bytes() == before


def test_explicit_anchor_shortage_is_not_silently_adjusted(
    selection_script, final_round_args
):
    """An explicit request for three anchors must fail when only two remain."""
    final_round_args.anchor_count = 3
    with pytest.raises(
        RuntimeError, match="Requested 3 new anchors, but only 2 remain"
    ):
        selection_script.run_selection(final_round_args)
    assert not selection_script.output_directory(final_round_args).exists()


def test_no_remaining_anchors_stops_without_creating_output(
    selection_script, final_round_args
):
    """Never recycle old anchors or begin MC inference after exhausting the split."""
    args = final_round_args
    train = pd.read_csv(args.train_csv)
    train[["filename"]].to_csv(args.anchor_history_csv, index=False)
    with pytest.raises(RuntimeError, match="No new anchors remain"):
        selection_script.run_selection(args)
    assert not selection_script.output_directory(args).exists()


def test_final_round_preserves_budget_coverage_constraints(
    selection_script, final_round_args
):
    """Anchor truncation must not permit a budget too small to cover the remainder."""
    final_round_args.pair_budget = 1
    with pytest.raises(RuntimeError, match="each anchor"):
        selection_script.run_selection(final_round_args)
    assert not selection_script.output_directory(final_round_args).exists()


def test_final_round_stops_when_unique_pairs_cannot_fill_budget(
    selection_script, final_round_args
):
    """Do not shrink budgets or repeat pairs to satisfy a small candidate scope."""
    final_round_args.candidate_scope = "selected"
    with pytest.raises(RuntimeError, match="Cannot cover every anchor"):
        selection_script.run_selection(final_round_args)
    assert not selection_script.output_directory(final_round_args).exists()


def test_explicit_anchor_csv_keeps_its_own_default_budget(
    selection_script, final_round_args, tmp_path
):
    """Diagnostic anchors must not inherit the unrelated rate-based budget."""
    path = tmp_path / "anchors.csv"
    pd.DataFrame({"filename": ["dummy_09"]}).to_csv(path, index=False)
    final_round_args.anchor_csv = path
    selection_script.run_selection(final_round_args)
    directory = selection_script.output_directory(final_round_args)
    assert len(pd.read_csv(directory / "new_pairs.csv")) == 1
    configuration = json.loads((directory / "experiment_config.json").read_text())
    assert configuration["requested_anchor_count"] == 1
    assert configuration["anchor_count_adjusted"] is False


def test_ordinary_rate_selection_and_uncertainty_order_are_unchanged(
    selection_script, final_round_args
):
    """Keep the usual count, stable ties, and history exclusions before exhaustion."""
    args = final_round_args
    train = pd.read_csv(args.train_csv)
    uncertainty = pd.read_csv(args.uncertainty_csv)
    uncertainty["var_score"] = 1
    history = train[["filename"]].iloc[:2]
    anchors = selection_script.select_anchor_images(
        train.filename.tolist(), uncertainty, history, None, args.anchor_rate
    )
    assert anchors.filename.tolist() == ["dummy_02", "dummy_03", "dummy_04"]


def test_rate_selection_handles_339_requested_and_333_remaining(selection_script):
    """Reproduce the reported boundary using only synthetic IDs and uncertainty."""
    image_ids = [f"dummy_{index:05}" for index in range(6780)]
    uncertainty = pd.DataFrame(
        {
            "filename": image_ids,
            "var_score": np.arange(len(image_ids), dtype=float),
        }
    )
    history = pd.DataFrame({"filename": image_ids[:-333]})
    assert round(len(image_ids) * 0.05) == 339
    anchors = selection_script.select_anchor_images(
        image_ids, uncertainty, history, None, 0.05
    )
    assert len(anchors) == 333
    assert anchors.filename.tolist() == image_ids[-333:][::-1]


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
