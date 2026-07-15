"""Build Zhou2014/Zhou2016 windows with one validation run per subject."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_ROOT = PROJECT_ROOT / "data" / "public" / "Zhou2014"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "zhou2014_async.npz"
METADATA_FILE = OUTPUT_FILE.with_suffix(".json")
SAMPLING_RATE = 128
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
TASK_SECONDS = 5.0
CUE_CODES = {"left_hand": 1, "right_hand": 2, "feet": 3}
CLASS_NAMES = ["idle", "left_hand", "right_hand", "feet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help="subject ids to preprocess, or 'all' for all Zhou2016 subjects",
    )
    parser.add_argument(
        "--val-run-index",
        type=int,
        default=-1,
        help="run index used as validation for every subject; supports negative indexing",
    )
    return parser.parse_args()


def configure_data_cache(data_root: Path) -> None:
    import mne

    os.environ["MNE_DATA"] = str(data_root)
    os.environ["MNE_DATASETS_ZHOU2016_PATH"] = str(data_root)
    mne.set_config("MNE_DATA", str(data_root), set_env=True)
    mne.set_config("MNE_DATASETS_ZHOU2016_PATH", str(data_root), set_env=True)


def resolve_subjects(dataset, subjects: list[str]) -> list[int]:
    if len(subjects) == 1 and subjects[0].lower() == "all":
        return list(dataset.subject_list)
    return [int(subject) for subject in subjects]


def overlaps_any(start: int, stop: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < interval_stop and stop > interval_start for interval_start, interval_stop in intervals)


def build_run_windows(raw) -> tuple[list[np.ndarray], list[int]]:
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


def choose_validation_run(run_keys: list[tuple[str, str]], val_run_index: int) -> tuple[str, str]:
    if not run_keys:
        raise RuntimeError("Cannot choose a validation run from an empty run list.")
    try:
        return run_keys[val_run_index]
    except IndexError as exc:
        raise ValueError(f"Validation run index {val_run_index} is out of range for {len(run_keys)} runs.") from exc


def build_dataset(subjects: list[int], val_run_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    from moabb.datasets import Zhou2016

    subject_data = Zhou2016().get_data(subjects=subjects)
    features: list[np.ndarray] = []
    labels: list[int] = []
    split_labels: list[int] = []
    subject_labels: list[int] = []
    split_records: list[dict] = []

    for subject in subjects:
        sessions = subject_data[subject]
        run_keys = [(session_name, run_name) for session_name, runs in sessions.items() for run_name in runs]
        val_key = choose_validation_run(run_keys, val_run_index)

        for session_name, runs in sessions.items():
            for run_name, raw in runs.items():
                run_features, run_labels = build_run_windows(raw)
                split = 1 if (session_name, run_name) == val_key else 0
                features.extend(run_features)
                labels.extend(run_labels)
                split_labels.extend([split] * len(run_labels))
                subject_labels.extend([subject] * len(run_labels))
                split_records.append(
                    {
                        "subject": subject,
                        "session": session_name,
                        "run": run_name,
                        "split": "val" if split == 1 else "train",
                        "windows": len(run_labels),
                    }
                )
                print(
                    f"subject={subject} session={session_name} run={run_name} "
                    f"windows={len(run_labels)} split={split}"
                )

    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(split_labels, dtype=np.int64),
        np.asarray(subject_labels, dtype=np.int64),
        split_records,
    )


def main() -> None:
    args = parse_args()
    from moabb.datasets import Zhou2016

    configure_data_cache(DATA_ROOT)
    dataset = Zhou2016()
    subjects = resolve_subjects(dataset, args.subjects)
    features, labels, split, subject, split_records = build_dataset(subjects, args.val_run_index)
    if not np.any(split == 0) or not np.any(split == 1):
        raise RuntimeError("Expected both train (0) and validation (1) windows.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=features, y=labels, split=split, subject=subject)
    metadata = {
        "dataset": "Zhou2014",
        "moabb_class": "Zhou2016",
        "subjects": subjects,
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "task_seconds": TASK_SECONDS,
        "classes": CLASS_NAMES,
        "split": "one validation run per subject, all remaining runs are train",
        "val_run_index": args.val_run_index,
        "split_records": split_records,
        "n_train": int((split == 0).sum()),
        "n_val": int((split == 1).sum()),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset: {OUTPUT_FILE}; X={features.shape}")


if __name__ == "__main__":
    main()
