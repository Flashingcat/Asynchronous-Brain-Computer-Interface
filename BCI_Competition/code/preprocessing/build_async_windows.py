"""Build leakage-free asynchronous windows for BNCI2014001 subject 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np
from moabb.datasets import BNCI2014_001

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_ROOT = PROJECT_ROOT / "data" / "public" / "BNCI2014001"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_subject01_async.npz"
METADATA_FILE = OUTPUT_FILE.with_suffix(".json")
SAMPLING_RATE = 128
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
CUE_CODES = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
CLASS_NAMES = ["idle", "left_hand", "right_hand", "feet", "tongue"]
TASK_SECONDS = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-run-index",
        type=int,
        default=-1,
        help="run index from the train session used as validation; supports negative indexing",
    )
    return parser.parse_args()


def configure_data_cache(data_root: Path) -> None:
    """Configure MNE and MOABB to use the project's own public-data cache."""
    os.environ["MNE_DATA"] = str(data_root)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_root)
    mne.set_config("MNE_DATA", str(data_root), set_env=True)
    mne.set_config("MNE_DATASETS_BNCI_PATH", str(data_root), set_env=True)


def overlaps_any(start: int, stop: int, intervals: list[tuple[int, int]]) -> bool:
    """Return whether a window overlaps any task interval."""
    return any(start < interval_stop and stop > interval_start for interval_start, interval_stop in intervals)


def build_run_windows(raw: mne.io.BaseRaw) -> tuple[list[np.ndarray], list[int]]:
    """Extract task windows and label all non-task windows as idle."""
    filtered = raw.copy().pick("eeg").filter(8.0, 30.0, verbose=False).resample(SAMPLING_RATE, verbose=False)
    signal = filtered.get_data().astype(np.float32)
    events, _ = mne.events_from_annotations(filtered, event_id=CUE_CODES, verbose=False)
    window_size = int(WINDOW_SECONDS * SAMPLING_RATE)
    step_size = int(STRIDE_SECONDS * SAMPLING_RATE)
    task_size = int(TASK_SECONDS * SAMPLING_RATE)
    samples: list[np.ndarray] = []
    labels: list[int] = []
    task_intervals: list[tuple[int, int]] = []

    for onset, _, label in events:
        task_start = int(onset)
        task_stop = min(task_start + task_size, signal.shape[1])
        task_intervals.append((task_start, task_stop))

        for start in range(task_start, task_stop - window_size + 1, step_size):
            stop = start + window_size
            if stop <= signal.shape[1]:
                samples.append(signal[:, start:stop])
                labels.append(int(label))

    for start in range(0, signal.shape[1] - window_size + 1, step_size):
        stop = start + window_size
        if not overlaps_any(start, stop, task_intervals):
            samples.append(signal[:, start:stop])
            labels.append(0)

    return samples, labels


def choose_validation_run(run_names: list[str], val_run_index: int) -> str:
    if not run_names:
        raise RuntimeError("Cannot choose a validation run from an empty train session.")
    try:
        return run_names[val_run_index]
    except IndexError as exc:
        raise ValueError(f"Validation run index {val_run_index} is out of range for {len(run_names)} runs.") from exc


def build_dataset(val_run_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load BNCI subject 1 and split train session into train/validation runs."""
    subject_data = BNCI2014_001().get_data(subjects=[1])[1]
    features: list[np.ndarray] = []
    labels: list[int] = []
    split_labels: list[int] = []
    split_records: list[dict] = []

    for session_name, runs in subject_data.items():
        run_names = list(runs)
        validation_run = choose_validation_run(run_names, val_run_index) if "train" in session_name.lower() else None

        for run_name, raw in runs.items():
            run_features, run_labels = build_run_windows(raw)
            if "train" in session_name.lower():
                split = 1 if run_name == validation_run else 0
            else:
                split = 2

            features.extend(run_features)
            labels.extend(run_labels)
            split_labels.extend([split] * len(run_labels))
            split_records.append(
                {
                    "session": session_name,
                    "run": run_name,
                    "split": {0: "train", 1: "val", 2: "test"}[split],
                    "windows": len(run_labels),
                }
            )
            print(f"{session_name}/{run_name}: windows={len(run_labels)} split={split}")

    return np.stack(features), np.asarray(labels, dtype=np.int64), np.asarray(split_labels, dtype=np.int64), split_records


def main() -> None:
    args = parse_args()
    configure_data_cache(DATA_ROOT)
    features, labels, split, split_records = build_dataset(args.val_run_index)
    if not np.any(split == 0) or not np.any(split == 1) or not np.any(split == 2):
        raise RuntimeError("Expected train (0), validation (1), and test (2) windows.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=features, y=labels, split=split)
    metadata = {
        "dataset": "BNCI2014001",
        "subject": 1,
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "task_seconds": TASK_SECONDS,
        "task_definition": "MOABB cue-onset to cue-onset+4s, corresponding to raw MAT trial 2-6s motor imagery",
        "idle_definition": "all 2s windows that do not overlap any task interval",
        "classes": CLASS_NAMES,
        "split": "train-session run-level split: one run validation, remaining train-session runs train; official evaluation session test",
        "val_run_index": args.val_run_index,
        "split_records": split_records,
        "n_train": int((split == 0).sum()),
        "n_val": int((split == 1).sum()),
        "n_test": int((split == 2).sum()),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset: {OUTPUT_FILE}; X={features.shape}")


if __name__ == "__main__":
    main()
