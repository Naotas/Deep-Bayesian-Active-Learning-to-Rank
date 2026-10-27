"""Check publication-style plotting using only generated dummy CSVs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
)


@pytest.fixture
def plotter():
    """Import the plotting script without loading a training model."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "pair_experiment_plots", SCRIPT_DIR / "12_plot_pair_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved_results(tmp_path, plotter):
    """Build complete five-fold results for eight rounds and a shared initial model."""
    experiment = tmp_path / "dummy_experiment"
    experiment.mkdir()
    pair_root = tmp_path / "initial_pairs"
    initial_rows = []
    result_rows = []
    for fold in plotter.FOLDS:
        pair_dir = pair_root / f"fold_{fold}/RBS"
        pair_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "x1_image": ["dummy_a", "dummy_b", "dummy_c", "dummy_d"],
                "x2_image": ["dummy_b", "dummy_c", "dummy_d", "dummy_a"],
                "x1_label": [0, 1, 2, 3],
                "x2_label": [1, 2, 3, 0],
                "relative_label": [0, 0, 0, 1],
            }
        ).to_csv(pair_dir / f"{fold}_train_pair_RBS.csv", index=False)
        shared = {"experiment_id": experiment.name, "seed": 240119, "fold": fold}
        initial_rows.append(
            {
                **shared,
                "method": "initial",
                "round": 0,
                "cumulative_pair_count": 4,
                "all_accuracy": 0.62 + fold * 0.01,
                "neighbor_accuracy": 0.52 + fold * 0.01,
            }
        )
        for method_index, method in enumerate(plotter.METHOD_COLORS):
            for round_number in range(1, 9):
                result_rows.append(
                    {
                        **shared,
                        "method": method,
                        "round": round_number,
                        "cumulative_pair_count": 4 + round_number * 20,
                        "all_accuracy": 0.70
                        + fold * 0.01
                        + round_number * 0.002
                        + method_index * 0.001,
                        "neighbor_accuracy": 0.60
                        + fold * 0.01
                        + round_number * 0.002
                        + method_index * 0.001,
                    }
                )
                if round_number not in (3, 6, 8):
                    continue
                directory = (
                    experiment
                    / "rounds"
                    / method
                    / "seed_240119"
                    / f"round_{round_number}"
                    / f"fold_{fold}"
                )
                directory.mkdir(parents=True)
                (directory / "selection_manifest.json").write_text(
                    json.dumps({"status": "complete"}), encoding="utf-8"
                )
                pd.DataFrame(
                    {
                        "MayoLabel": [3, 1, 0, 2],
                        "unique_image_count": [4, 20 + 2 * fold, 10 + fold, 0],
                    }
                ).to_csv(directory / "mayo_grade_distribution.csv", index=False)
    pd.DataFrame(result_rows).to_csv(experiment / "performance.csv", index=False)
    pd.DataFrame(initial_rows).to_csv(
        experiment / "initial_performance.csv", index=False
    )
    args = plotter.parse_args(
        [
            "--experiment-dir",
            str(experiment),
            "--initial-pairs-root",
            str(pair_root),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )
    return args


def test_fold_means_sample_sd_nominal_axis_and_shared_initial(plotter, saved_results):
    """Use fold-wise mean ± sample SD and a common iteration-zero baseline."""
    performance, _ = plotter.load_report_data(saved_results)
    summary = plotter.summarize_performance(performance)
    row = summary.loc[(summary.method == "original") & (summary["round"] == 3)].iloc[0]
    assert row.nominal_labeling_percent == 35
    assert row.overall_accuracy_mean_percent == pytest.approx(73.6)
    assert row.overall_accuracy_sd_percent == pytest.approx(
        np.std([1, 2, 3, 4, 5], ddof=1)
    )
    assert row.neighboring_accuracy_mean_percent == pytest.approx(63.6)
    assert (summary.fold_count == 5).all()
    initial = summary.loc[summary["round"] == 0]
    assert len(initial) == 4
    assert initial.overall_accuracy_mean_percent.tolist() == pytest.approx([65] * 4)
    assert set(summary.nominal_labeling_percent) == set(range(20, 61, 5))
    assert len(performance) == 4 * 5 * 9


def test_grade_means_saved_counts_zero_grades_and_initial_deduplication(
    plotter, saved_results
):
    """Average saved counts equally and count repeated initial endpoints only once."""
    _, grades = plotter.load_report_data(saved_results)
    summary = plotter.summarize_grades(grades)
    initial = summary.loc[summary["round"] == 0]
    assert (initial.unique_image_count_mean == 1).all()
    values = summary.loc[(summary.method == "pair_bald") & (summary["round"] == 8)]
    assert values.unique_image_count_mean.tolist() == pytest.approx([13, 26, 0, 4])
    assert (values.fold_count == 5).all()
    expected_fraction = np.mean([(10 + fold) / (34 + 3 * fold) for fold in range(1, 6)])
    assert values.iloc[0].unique_image_fraction_mean == pytest.approx(expected_fraction)


def test_plots_have_requested_colors_bands_and_sixteen_bars(plotter, saved_results):
    """Check plotted data and band boundaries, not merely file creation."""
    performance, grades = plotter.load_report_data(saved_results)
    summary = plotter.summarize_performance(performance)
    figure = plotter.plot_accuracy(summary, "overall")
    axis = figure.axes[0]
    assert len(axis.lines) == len(axis.collections) == 4
    for line, band, (method, color) in zip(
        axis.lines, axis.collections, plotter.METHOD_COLORS.items()
    ):
        assert line.get_label() == method
        assert colors.to_hex(line.get_color()) == color
        assert colors.to_hex(band.get_facecolor()[0]) == color
        assert band.get_alpha() == pytest.approx(0.16)
        assert line.get_xdata().tolist() == list(range(20, 61, 5))
        values = summary.loc[summary.method == method].sort_values("round")
        vertices = band.get_paths()[0].vertices[:, 1]
        assert vertices.min() == pytest.approx(
            (
                values.overall_accuracy_mean_percent
                - values.overall_accuracy_sd_percent
            ).min()
        )
        assert vertices.max() == pytest.approx(
            (
                values.overall_accuracy_mean_percent
                + values.overall_accuracy_sd_percent
            ).max()
        )
    plotter.plt.close(figure)
    figure = plotter.plot_grades(plotter.summarize_grades(grades), 8)
    assert len(figure.axes[0].patches) == 16
    for index, color in enumerate(plotter.METHOD_COLORS.values()):
        assert all(
            colors.to_hex(bar.get_facecolor()) == color
            for bar in figure.axes[0].patches[index * 4 : (index + 1) * 4]
        )
    plotter.plt.close(figure)


@pytest.mark.parametrize("round_number", [0, 3, 6, 8])
def test_grade_bars_show_mean_within_fold_percentages(
    plotter, saved_results, round_number
):
    """Normalize each fold first, preserve zero grades, and sum to 100% per method."""
    _, grades = plotter.load_report_data(saved_results)
    summary = plotter.summarize_grades(grades)
    figure = plotter.plot_grades(summary, round_number)
    axis = figure.axes[0]
    assert axis.get_ylim() == (0, 100)
    assert axis.get_ylabel() == "Mean proportion of unique images (%)"
    for index, method in enumerate(plotter.METHOD_COLORS):
        heights = [
            bar.get_height() for bar in axis.patches[index * 4 : (index + 1) * 4]
        ]
        expected = (
            grades.loc[(grades.method == method) & (grades["round"] == round_number)]
            .groupby("MayoLabel")["unique_image_fraction"]
            .mean()
            * 100
        )
        np.testing.assert_allclose(heights, expected.to_numpy(), rtol=0, atol=1e-12)
        assert sum(heights) == pytest.approx(100)
        if round_number == 0:
            assert heights == pytest.approx([25] * 4)
        else:
            assert heights[2] == 0
            pooled_percentage = 13 / (13 + 26 + 4) * 100
            assert abs(heights[0] - pooled_percentage) > 1e-6
    plotter.plt.close(figure)


@pytest.mark.parametrize("problem", ["missing_fold", "duplicate", "nan", "budget"])
def test_invalid_performance_stops_before_creating_output(
    plotter, saved_results, problem
):
    """Never quietly average an incomplete fold set or change the comparison budget."""
    path = saved_results.performance_csv
    data = pd.read_csv(path)
    if problem == "missing_fold":
        data = data.loc[~((data["fold"] == 5) & (data["round"] == 8))]
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif problem == "nan":
        data.loc[0, "all_accuracy"] = np.nan
    else:
        data.loc[0, "cumulative_pair_count"] += 1
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        plotter.save_report(saved_results)
    assert not saved_results.output_dir.exists()


def test_other_seeds_and_experiments_are_not_pooled(plotter, saved_results):
    """Explicitly select one seed and experiment even in a combined input CSV."""
    path = saved_results.performance_csv
    data = pd.read_csv(path)
    other_seed = data.assign(seed=999, all_accuracy=0, neighbor_accuracy=0)
    other_experiment = data.assign(experiment_id="another_experiment", all_accuracy=0)
    pd.concat([data, other_seed, other_experiment]).to_csv(path, index=False)
    performance, _ = plotter.load_report_data(saved_results)
    assert len(performance) == 180
    assert set(performance.seed) == {240119}
    assert performance.all_accuracy.min() > 0.6


@pytest.mark.parametrize(
    "problem", ["incomplete", "missing_csv", "duplicate_grade", "negative"]
)
def test_invalid_grade_inputs_do_not_create_output(plotter, saved_results, problem):
    """Reject partial selections and malformed class-count tables."""
    directory = (
        saved_results.experiment_dir / "rounds/pair_bald/seed_240119/round_8/fold_5"
    )
    path = directory / "mayo_grade_distribution.csv"
    if problem == "incomplete":
        (directory / "selection_manifest.json").write_text('{"status":"running"}')
    elif problem == "missing_csv":
        path.rename(path.with_suffix(".saved"))
    else:
        data = pd.read_csv(path)
        if problem == "duplicate_grade":
            data.loc[0, "MayoLabel"] = 0
        else:
            data.loc[0, "unique_image_count"] = -1
        data.to_csv(path, index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        plotter.save_report(saved_results)
    assert not saved_results.output_dir.exists()


def test_conflicting_initial_labels_are_rejected(plotter, saved_results):
    """Iteration zero must not assign two oracle grades to one image."""
    path = saved_results.initial_pairs_root / "fold_1/RBS/1_train_pair_RBS.csv"
    data = pd.read_csv(path)
    data.loc[0, "x1_label"] = 2
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting Mayo"):
        plotter.load_report_data(saved_results)


def test_missing_initial_performance_is_not_invented(plotter, saved_results):
    """Never substitute round-one performance or a training-only prediction."""
    saved_results.initial_performance_csv = saved_results.experiment_dir / "absent.csv"
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        plotter.save_report(saved_results)
    assert not saved_results.output_dir.exists()


def test_existing_report_is_preserved(plotter, saved_results):
    """Refuse to overwrite any existing report directory, even if it is empty."""
    saved_results.output_dir.mkdir()
    sentinel = saved_results.output_dir / "keep.txt"
    sentinel.write_text("dummy existing output")
    with pytest.raises(FileExistsError, match="already exists"):
        plotter.save_report(saved_results)
    assert sentinel.read_text() == "dummy existing output"


def test_cli_produces_tables_pngs_pdfs_and_metadata_without_tensorflow(saved_results):
    """Exercise the terminal entry point with only dummy results and CPU plotting."""
    original_csv = saved_results.performance_csv.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "12_plot_pair_experiments.py"),
            "--experiment-dir",
            str(saved_results.experiment_dir),
            "--initial-pairs-root",
            str(saved_results.initial_pairs_root),
            "--output-dir",
            str(saved_results.output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved Figure 4/6" in result.stdout
    output = saved_results.output_dir
    assert len(list(output.glob("*.csv"))) == 4
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 6
    for path in output.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")
    for path in output.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    assert saved_results.performance_csv.read_bytes() == original_csv
    metadata = json.loads((output / "plot_config.json").read_text())
    assert metadata["folds"] == [1, 2, 3, 4, 5]
    assert "ddof=1" in metadata["accuracy_band"]
    assert "cumulative training pairs" in metadata["grade_population"]
    assert "within-fold grade fractions" in metadata["grade_axis"]
    assert metadata["grade_axis_limits_percent"] == [0, 100]


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-round", "6"],
        ["--bar-rounds", "0,3,3,8"],
        ["--bar-rounds", "0,3,x"],
        ["--round-labeling-percent", "nan"],
        ["--initial-labeling-percent", "90"],
    ],
)
def test_invalid_plot_arguments_are_rejected(plotter, tmp_path, extra):
    """Do not silently omit a requested snapshot or use invalid nominal ratios."""
    with pytest.raises(SystemExit):
        plotter.parse_args(
            [
                "--experiment-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                *extra,
            ]
        )
