"""Shared data-location helpers for LIMUC preprocessing scripts."""

import argparse
import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/data/umeiro0/patient_based_classified_images")
GENERATED_DIRECTORY = "generated"


def default_data_root() -> Path:
    """Return the configured LIMUC data root.

    Returns:
        Path: Data root from ``LIMUC_DATA_ROOT`` or the server default.
    """
    return Path(os.environ.get("LIMUC_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common data-root option to a command-line parser.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.
    """
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help=(
            "Directory containing the patient folders. Defaults to "
            "$LIMUC_DATA_ROOT or "
            f"{DEFAULT_DATA_ROOT}."
        ),
    )


def generated_path(data_root: Path, *parts: str) -> Path:
    """Build a path below the generated-data directory.

    Args:
        data_root (Path): Directory containing the source patient folders.
        *parts (str): Additional path components.

    Returns:
        Path: Path below ``<data-root>/generated``.
    """
    return data_root.expanduser() / GENERATED_DIRECTORY / Path(*parts)
