"""Download BNCI2014001 into this project's public-data cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mne

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_DIR = PROJECT_ROOT / "data" / "public" / "BNCI2014001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help="subject ids to download, or 'all' for the complete dataset",
    )
    return parser.parse_args()


def configure_data_cache() -> None:
    """Keep MOABB/MNE downloads inside the project data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(DATA_DIR)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(DATA_DIR)
    mne.set_config("MNE_DATA", str(DATA_DIR), set_env=True)
    mne.set_config("MNE_DATASETS_BNCI_PATH", str(DATA_DIR), set_env=True)


def resolve_subjects(dataset, subjects: list[str]) -> list[int]:
    if len(subjects) == 1 and subjects[0].lower() == "all":
        return list(dataset.subject_list)
    return [int(subject) for subject in subjects]


def main() -> None:
    args = parse_args()
    from moabb.datasets import BNCI2014_001

    configure_data_cache()

    dataset = BNCI2014_001()
    subjects = resolve_subjects(dataset, args.subjects)
    data = dataset.get_data(subjects=subjects)

    for subject in subjects:
        print(f"Downloaded/verified subject {subject}: sessions={list(data[subject])}")
    print(f"Cache: {DATA_DIR}")


if __name__ == "__main__":
    main()
