# =============================================================================
# Implementation of: Simplified BNCI2014001 OOF Preprocessing
#
# Reference:
#   Project-specific preprocessing for asynchronous BNCI2014001 motor imagery.
#   The pipeline builds clean continuous EEG segments, applies per-segment
#   causal filtering, cuts fixed windows, labels idle/task/MI, and assigns
#   leave-one-run-out OOF folds.
#
# Source: No external code copied.
# =============================================================================
"""Build a simple OOF-ready BNCI2014001 window dataset.

Output:
  BCI_Competition/data/processed/bnci2014001_oof_windows.npz

Arrays:
  X        float32, shape (n_windows, 22, 500)
  y        int64, 0 idle / 1 left_hand / 2 right_hand / 3 feet / 4 tongue
  subject  int64
  session  int64, 0 train / 1 test
  run      int64, run index within session
  fold     int64, train-session validation fold id; -1 for test session
  split    int64, 0 train-session windows / 2 official test-session windows
  segment  independent clean segment id within a run
  decision_sample  original sample where the window decision is made
  is_pure  bool, whether the window is fully idle or fully inside one MI event

The window table keeps the complete causal stream for both sessions. Training
must select ``is_pure`` windows explicitly; validation and test evaluation use
the complete stream, including event-boundary windows.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_ROOT = PROJECT_ROOT / "data" / "public" / "BNCI2014001"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"

SAMPLING_RATE = 250
WINDOW_SAMPLES = 500
STRIDE_SAMPLES = 125
LOW_HZ = 8.0
HIGH_HZ = 30.0
FILTER_ORDER = 4

CLASS_TO_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
CLASS_NAMES = ["idle", "left_hand", "right_hand", "feet", "tongue"]


@dataclass(frozen=True)
class TaskEvent:
    start: int
    stop: int
    label: int
    name: str


@dataclass(frozen=True)
class CleanSegment:
    start: int
    stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["all"], help="subject ids or 'all'")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def configure_data_cache(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_root)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_root)
    mne.set_config("MNE_DATA", str(data_root), set_env=True)
    mne.set_config("MNE_DATASETS_BNCI_PATH", str(data_root), set_env=True)


def resolve_subjects(dataset, requested: list[str]) -> list[int]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(dataset.subject_list)
    return [int(item) for item in requested]


def eeg_data(raw: mne.io.BaseRaw) -> np.ndarray:
    picked = raw.copy().pick("eeg")
    if int(round(picked.info["sfreq"])) != SAMPLING_RATE:
        raise RuntimeError(f"Expected native {SAMPLING_RATE} Hz, got {picked.info['sfreq']}")
    if len(picked.ch_names) != 22:
        raise RuntimeError(f"Expected 22 EEG channels, got {len(picked.ch_names)}: {picked.ch_names}")
    return picked.get_data().astype(np.float32)


def task_events(raw: mne.io.BaseRaw) -> list[TaskEvent]:
    events: list[TaskEvent] = []
    for onset_s, duration_s, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        name = str(desc)
        if name not in CLASS_TO_ID:
            continue
        start = int(round(float(onset_s) * SAMPLING_RATE))
        stop = int(round((float(onset_s) + float(duration_s)) * SAMPLING_RATE))
        events.append(TaskEvent(
            start=start,
            stop=stop,
            label=CLASS_TO_ID[name],
            name=name,
        ))
    return events


def artifact_intervals(raw: mne.io.BaseRaw) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for onset_s, duration_s, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        name = str(desc).lower()
        if "bad" not in name and "artifact" not in name and "reject" not in name:
            continue
        start = int(round(float(onset_s) * SAMPLING_RATE))
        stop = int(round((float(onset_s) + float(duration_s)) * SAMPLING_RATE))
        if stop > start:
            intervals.append((start, stop))
    return sorted(intervals)


# MOABB 只保留任务注释，没有把源 MAT 的试次级坏段标记写入 Raw；
# 这里按原始连续信号坐标恢复完整坏试次，不能把删段后的数据重新拼接成一条 run。
def trial_artifact_intervals(run) -> list[tuple[int, int]]:
    """Convert BNCI trial-level artifact flags into full-trial sample intervals."""
    trials = np.atleast_1d(run.trial).astype(np.int64) - 1
    flags = np.atleast_1d(run.artifacts).astype(bool)
    if trials.shape != flags.shape:
        raise RuntimeError("BNCI trial and artifact arrays do not align")
    stops = np.r_[trials[1:], len(run.X)]
    return [(int(start), int(stop)) for start, stop, bad in zip(trials, stops, flags) if bad]


def load_subject_trial_artifacts(dataset, subject: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Load artifact flags omitted by MOABB's Raw conversion from the source MAT files."""
    output: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for path in dataset.data_path(subject):
        suffix = Path(path).stem[-1].upper()
        if suffix not in {"T", "E"}:
            raise RuntimeError(f"unexpected BNCI session file: {path}")
        session_id = 0 if suffix == "T" else 1
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        labelled_runs = [run for run in np.atleast_1d(mat["data"]) if np.atleast_1d(run.trial).size]
        for run_index, run in enumerate(labelled_runs):
            output[(session_id, run_index)] = trial_artifact_intervals(run)
    return output


def clean_segments(n_samples: int, artifacts: list[tuple[int, int]]) -> list[CleanSegment]:
    segments: list[CleanSegment] = []
    cursor = 0
    for start, stop in artifacts:
        start = max(0, min(start, n_samples))
        stop = max(0, min(stop, n_samples))
        if start - cursor >= WINDOW_SAMPLES:
            segments.append(CleanSegment(cursor, start))
        cursor = max(cursor, stop)
    if n_samples - cursor >= WINDOW_SAMPLES:
        segments.append(CleanSegment(cursor, n_samples))
    return segments


def causal_filter_segment(segment: np.ndarray) -> np.ndarray:
    sos = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SAMPLING_RATE, output="sos")
    return sosfilt(sos, segment, axis=1).astype(np.float32)


def overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    return a_start < b_stop and a_stop > b_start


def label_window(start: int, stop: int, events: list[TaskEvent]) -> int | None:
    """Keep only idle windows or windows fully contained in one MI event."""
    overlapping = [event for event in events if overlap(start, stop, event.start, event.stop)]
    if not overlapping:
        return 0
    contained = [event for event in overlapping if start >= event.start and stop <= event.stop]
    return contained[0].label if len(contained) == 1 and len(overlapping) == 1 else None


def event_at(sample: int, events: list[TaskEvent]) -> TaskEvent | None:
    active = [event for event in events if event.start <= sample < event.stop]
    if len(active) > 1:
        raise RuntimeError(f"overlapping task annotations at sample {sample}")
    return active[0] if active else None


def build_run_windows(
    raw: mne.io.BaseRaw,
    trial_artifacts: list[tuple[int, int]] | None = None,
    pure_only: bool = True,
) -> tuple[dict[str, np.ndarray], list[TaskEvent], dict]:
    signal = eeg_data(raw)
    all_events = task_events(raw)
    artifacts = sorted(set([*artifact_intervals(raw), *(trial_artifacts or [])]))
    events = [
        event for event in all_events
        if not any(overlap(event.start, event.stop, start, stop) for start, stop in artifacts)
    ]
    segments = clean_segments(signal.shape[1], artifacts)

    # 保留原始 sample 位置和 segment 身份，避免删除坏片段后压缩时间轴。
    values: dict[str, list] = {
        "X": [],
        "y": [],
        "segment": [],
        "decision_sample": [],
        "is_pure": [],
    }
    dropped_boundary = 0
    for segment_id, segment in enumerate(segments):
        filtered = causal_filter_segment(signal[:, segment.start : segment.stop])
        for local_start in range(0, filtered.shape[1] - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            global_start = segment.start + local_start
            global_stop = global_start + WINDOW_SAMPLES
            pure_label = label_window(global_start, global_stop, events)

            # 训练删除混合窗口；测试保留完整流并按决策时刻标注。
            if pure_only:
                if pure_label is None:
                    dropped_boundary += 1
                    continue
                label = pure_label
            else:
                event = event_at(global_stop - 1, events)
                label = 0 if event is None else event.label

            values["X"].append(filtered[:, local_start : local_start + WINDOW_SAMPLES])
            values["y"].append(label)
            values["segment"].append(segment_id)
            values["decision_sample"].append(global_stop - 1)
            values["is_pure"].append(pure_label is not None)

    count = len(values["y"])
    arrays = {
        "X": np.stack(values["X"]).astype(np.float32) if count else np.empty((0, 22, WINDOW_SAMPLES), dtype=np.float32),
        **{
            key: np.asarray(values[key], dtype=np.int64)
            for key in ("y", "segment", "decision_sample")
        },
        "is_pure": np.asarray(values["is_pure"], dtype=bool),
    }
    info = {
        "segments": len(segments),
        "task_events": len(events),
        "artifacts": len(artifacts),
        "windows": count,
        "dropped_boundary": dropped_boundary,
    }
    return arrays, events, info


def build_dataset(subjects: list[int]) -> tuple[dict[str, np.ndarray], list[dict]]:
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    subject_data = dataset.get_data(subjects=subjects)

    arrays: dict[str, list[np.ndarray]] = {
        "X": [],
        "y": [],
        "subject": [],
        "session": [],
        "run": [],
        "fold": [],
        "split": [],
        "segment": [],
        "decision_sample": [],
        "is_pure": [],
    }
    event_arrays: dict[str, list[int]] = {
        "event_subject": [],
        "event_session": [],
        "event_run": [],
        "event_label": [],
        "event_start": [],
        "event_stop": [],
    }
    records: list[dict] = []

    for subject in subjects:
        trial_artifacts = load_subject_trial_artifacts(dataset, subject)
        for session_name, runs in subject_data[subject].items():
            is_train_session = "train" in session_name.lower()
            session_id = 0 if is_train_session else 1
            split_id = 0 if is_train_session else 2
            for run_index, (run_name, raw) in enumerate(runs.items()):
                artifact_key = (session_id, run_index)
                if artifact_key not in trial_artifacts:
                    raise RuntimeError(f"missing source artifact flags for subject {subject}, run {artifact_key}")
                # 训练和测试均保存完整连续流；模型训练阶段再用 is_pure 选择无歧义窗口。
                run_arrays, events, info = build_run_windows(
                    raw, trial_artifacts[artifact_key], pure_only=False
                )
                n = len(run_arrays["y"])
                for key, value in run_arrays.items():
                    arrays[key].append(value)
                arrays["subject"].append(np.full(n, subject, dtype=np.int64))
                arrays["session"].append(np.full(n, session_id, dtype=np.int64))
                arrays["run"].append(np.full(n, run_index, dtype=np.int64))
                arrays["fold"].append(np.full(n, run_index if is_train_session else -1, dtype=np.int64))
                arrays["split"].append(np.full(n, split_id, dtype=np.int64))
                # 两个 session 的事件都进入同一张带身份的事件表，供 OOF 与测试共用评估器。
                for event in events:
                    event_arrays["event_subject"].append(subject)
                    event_arrays["event_session"].append(session_id)
                    event_arrays["event_run"].append(run_index)
                    event_arrays["event_label"].append(event.label)
                    event_arrays["event_start"].append(event.start)
                    event_arrays["event_stop"].append(event.stop)
                record = {
                    "subject": subject,
                    "session": session_name,
                    "run": run_name,
                    "run_index": run_index,
                    "split": "train_session" if is_train_session else "test_session",
                    **info,
                    "label_counts": np.bincount(run_arrays["y"], minlength=len(CLASS_NAMES)).astype(int).tolist(),
                }
                records.append(record)
                print(f"s{subject:02d} {session_name}/{run_name}: windows={n} labels={record['label_counts']}")

    output = {
        key: np.concatenate(value, axis=0) if key != "X" else np.concatenate(value, axis=0).astype(np.float32)
        for key, value in arrays.items()
    }
    output.update({key: np.asarray(value, dtype=np.int64) for key, value in event_arrays.items()})
    output["sampling_rate"] = np.asarray(SAMPLING_RATE, dtype=np.int64)
    return output, records


def write_outputs(output_file: Path, arrays: dict[str, np.ndarray]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)
    print(f"Saved: {output_file}")
    print(f"X shape: {arrays['X'].shape}")


def main() -> None:
    args = parse_args()
    configure_data_cache(args.data_root)
    from moabb.datasets import BNCI2014_001

    subjects = resolve_subjects(BNCI2014_001(), args.subjects)
    arrays, _ = build_dataset(subjects)
    write_outputs(args.output_file, arrays)


if __name__ == "__main__":
    main()
