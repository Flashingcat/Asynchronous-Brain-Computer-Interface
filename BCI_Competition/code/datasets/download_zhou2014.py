"""Download the MOABB Zhou2016 dataset into this project's Zhou2014 cache.

MOABB exposes this dataset as ``Zhou2016``.  The project keeps the directory
name ``Zhou2014`` to match the experiment naming requested here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mne

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_DIR = PROJECT_ROOT / "data" / "public" / "Zhou2014"


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
    """Keep MNE/MOABB downloads inside this project's public-data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(DATA_DIR)
    os.environ["MNE_DATASETS_ZHOU2016_PATH"] = str(DATA_DIR)
    mne.set_config("MNE_DATA", str(DATA_DIR), set_env=True)
    mne.set_config("MNE_DATASETS_ZHOU2016_PATH", str(DATA_DIR), set_env=True)


def resolve_subjects(dataset, subjects: list[str]) -> list[int]:
    if len(subjects) == 1 and subjects[0].lower() == "all":
        return list(dataset.subject_list)
    return [int(subject) for subject in subjects]


def main() -> None:
    args = parse_args()
    from moabb.datasets import Zhou2016

    configure_data_cache()

    dataset = Zhou2016()
    subjects = resolve_subjects(dataset, args.subjects)
    data = dataset.get_data(subjects=subjects)

    for subject in subjects:
        print(f"Downloaded/verified subject {subject}: sessions={list(data[subject])}")
    print(f"Dataset class: Zhou2016")
    print(f"Project cache: {DATA_DIR}")


if __name__ == "__main__":
    main()
