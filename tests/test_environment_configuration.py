"""Tests for the reproducible legacy runtime configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_manifest_uses_compatible_packages() -> None:
    """The manifest should pin the compatible packages chosen for TensorFlow 2.4."""
    manifest = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    expected_dependencies = {
        '"tensorflow==2.4.0"',
        '"h5py==2.10.0"',
        '"numpy==1.19.5"',
        '"scipy==1.5.4"',
        '"opencv-python-headless==4.5.1.48"',
    }
    manifest_lines = {line.strip().rstrip(",") for line in manifest.splitlines()}
    assert expected_dependencies <= manifest_lines
    assert "scikit-learn" not in manifest
    assert 'requires-python = ">=3.8,<3.9"' in manifest


def test_docker_uses_standard_tensorflow_lockfile() -> None:
    """Docker should install the locked PyPI environment instead of an NGC image."""
    dockerfile = (PROJECT_ROOT / "Docker/Dockerfile").read_text(encoding="utf-8")

    assert "nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "pip install" not in dockerfile
    assert "nvcr.io/nvidia/tensorflow" not in dockerfile


def test_opencv_gui_api_is_not_used() -> None:
    """Headless OpenCV is sufficient while the project avoids GUI entry points."""
    gui_calls = ("cv2.imshow", "cv2.waitKey", "cv2.namedWindow", "cv2.selectROI")
    source_files = [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if ".venv" not in path.parts and path.parent.name != "tests"
    ]

    for source_file in source_files:
        source = source_file.read_text(encoding="utf-8")
        assert not any(call in source for call in gui_calls), source_file
