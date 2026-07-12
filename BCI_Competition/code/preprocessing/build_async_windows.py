"""Build leakage-free asynchronous windows for BNCI2014001 subject 1."""

from __future__ import annotations

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


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load BNCI subject 1 and keep the two recording sessions separate."""
    subject_data = BNCI2014_001().get_data(subjects=[1])[1]
    features: list[np.ndarray] = []
    labels: list[int] = []
    split_labels: list[int] = []

    for session_name, runs in subject_data.items():
        session_features: list[np.ndarray] = []
        session_labels: list[int] = []
        for raw in runs.values():
            run_features, run_labels = build_run_windows(raw)
            session_features.extend(run_features)
            session_labels.extend(run_labels)

        split = 0 if "train" in session_name.lower() else 1
        features.extend(session_features)
        labels.extend(session_labels)
        split_labels.extend([split] * len(session_labels))
        print(f"{session_name}: windows={len(session_labels)} split={split}")

    return np.stack(features), np.asarray(labels, dtype=np.int64), np.asarray(split_labels, dtype=np.int64)


def main() -> None:
    configure_data_cache(DATA_ROOT)
    features, labels, split = build_dataset()
    if not np.any(split == 0) or not np.any(split == 1):
        raise RuntimeError("Expected separate train and test sessions; refusing a potentially leaky split.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=features, y=labels, split=split)
    metadata = {
        "dataset": "BNCI2014001",
        "subject": 1,
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "task_seconds": TASK_SECONDS,
        "task_definition": "BNCI2014001 cue-onset to cue-onset+4s, corresponding to trial 3-7s motor imagery",
        "idle_definition": "all 2s windows that do not overlap any task interval",
        "classes": CLASS_NAMES,
        "n_train": int((split == 0).sum()),
        "n_test": int((split == 1).sum()),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset: {OUTPUT_FILE}; X={features.shape}")


if __name__ == "__main__":
    main()
