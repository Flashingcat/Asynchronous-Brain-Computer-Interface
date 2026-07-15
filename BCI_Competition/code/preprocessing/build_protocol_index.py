"""为 BNCI2014001 构建不含 EEG 数值的评估时间索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ((0, "0train", "T"), (1, "1test", "E"))
FS, WINDOW, STEP = 250, 500, 125
ARTIFACT_POLICY = "official_trial_exclusion"
EVENT_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("trial", "u1"),
    ("class_id", "u1"), ("artifact", "?"), ("trial_start", "<i8"),
    ("mi_start", "<i8"), ("mi_stop", "<i8"),
])
SEGMENT_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("segment", "u1"),
    ("start", "<i8"), ("stop", "<i8"), ("reset_reason", "U16"),
])
WINDOW_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("segment", "u1"),
    ("window", "<u4"), ("start", "<i8"), ("stop", "<i8"), ("decision_time", "<f8"),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="包含 A01T.mat 等文件的目录")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1], help="被试编号，可一次指定多个")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


def vector(value) -> np.ndarray:
    return np.atleast_1d(value).reshape(-1)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task_records(path: Path) -> list[tuple[int, object]]:
    """按 trial 数识别任务 run，并保留原始 MAT record 下标。"""
    records = vector(loadmat(path, squeeze_me=True, struct_as_record=False)["data"]).tolist()
    sizes = [vector(record.trial).size for record in records]
    if any(size not in (0, 48) for size in sizes):
        raise RuntimeError(f"{path.name}: 出现非 0/48 的 trial 数量 {sizes}")
    task_records = [(index, record) for index, (record, size) in enumerate(zip(records, sizes)) if size == 48]
    if len(task_records) != 6:
        raise RuntimeError(f"{path.name}: 预期 6 个任务 run，实际 {len(task_records)}")
    return task_records


def clean_segments(n_samples: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """返回伪迹区间补集，重叠排除区间自动合并。"""
    result: list[tuple[int, int, str]] = []
    cursor, reason = 0, "run_start"
    for start, stop in sorted(excluded):
        start, stop = min(n_samples, max(0, start)), min(n_samples, stop)
        if cursor < start:
            result.append((cursor, start, reason))
        cursor, reason = max(cursor, stop), "after_artifact"
    if cursor < n_samples:
        result.append((cursor, n_samples, reason))
    return result


def parse_run(record, path: Path, run_id: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """校验源字段后再转换类型，拒绝静默截断异常值。"""
    shape = np.asarray(record.X).shape
    raw_sampling_rate = vector(record.fs)
    raw_trial, raw_labels, raw_artifacts = vector(record.trial), vector(record.y), vector(record.artifacts)
    if (len(shape) != 2 or shape[1] != 25 or raw_sampling_rate.size != 1 or
            not np.isfinite(raw_sampling_rate[0]) or raw_sampling_rate[0] != FS):
        raise RuntimeError(f"{path.name} run {run_id}: shape={shape}, fs={record.fs}")
    if not (len(raw_trial) == len(raw_labels) == len(raw_artifacts) == 48):
        raise RuntimeError(f"{path.name} run {run_id}: trial 字段长度不一致")
    if not np.all(raw_trial == raw_trial.astype(np.int64)):
        raise RuntimeError(f"{path.name} run {run_id}: trial 坐标不是整数")
    trial, labels = raw_trial.astype(np.int64) - 1, raw_labels.astype(np.int64)
    if not np.all(raw_labels == labels) or Counter(labels.tolist()) != Counter({1: 12, 2: 12, 3: 12, 4: 12}):
        raise RuntimeError(f"{path.name} run {run_id}: 类别并非 1/2/3/4 各 12 个")
    if not set(np.unique(raw_artifacts)).issubset({0, 1}):
        raise RuntimeError(f"{path.name} run {run_id}: artifacts 只能为 0/1")
    if np.any(trial < 0) or np.any(np.diff(trial) <= 0) or np.any(trial + 6 * FS > shape[0]):
        raise RuntimeError(f"{path.name} run {run_id}: trial 坐标越界或未严格递增")
    return shape[0], trial, labels, raw_artifacts.astype(bool)


def build_subject(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if subject not in range(1, 10):
        raise ValueError("BNCI2014001 subject 必须为 1 至 9")
    events, segments, windows = [], [], []
    summaries, sources = {}, {}

    # 两个 session 使用同一构建逻辑，测试侧不会获得额外处理。
    for session_id, session_name, suffix in SESSIONS:
        path = data_root.resolve() / f"A{subject:02d}{suffix}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        records = load_task_records(path)
        sources[session_name] = {"filename": path.name, "sha256": file_hash(path)}
        artifacts_per_class, run_summaries = Counter(), []
        session_event_start = len(events)

        for run_id, (source_record, record) in enumerate(records):
            n_samples, trial, labels, artifacts = parse_run(record, path, run_id)
            excluded = [(int(start), int(start + 6 * FS)) for start, bad in zip(trial, artifacts) if bad]
            run_segments = clean_segments(n_samples, excluded)
            artifacts_per_class.update(labels[artifacts].tolist())
            segment_start, window_start = len(segments), len(windows)

            # 真值事件和模型可消费窗口分表保存。
            for trial_id, (start, class_id, bad) in enumerate(zip(trial, labels, artifacts)):
                events.append((subject, session_id, run_id, trial_id, class_id, bad,
                               start, start + 2 * FS, start + 6 * FS))
            for segment_id, (start, stop, reason) in enumerate(run_segments):
                segments.append((subject, session_id, run_id, segment_id, start, stop, reason))
                for window_id, left in enumerate(range(start, stop - WINDOW + 1, STEP)):
                    right = left + WINDOW
                    windows.append((subject, session_id, run_id, segment_id, window_id,
                                    left, right, right / FS))

            lengths = [stop - start for start, stop, _ in run_segments]
            excluded_samples = n_samples - sum(lengths)
            unwindowed_samples = sum(length if length < WINDOW else (length - WINDOW) % STEP for length in lengths)
            run_summaries.append({
                "run": run_id, "source_record_index": source_record, "original_samples": n_samples,
                "artifact_events": int(artifacts.sum()), "clean_events": 48 - int(artifacts.sum()),
                "artifacts_per_class": [int(((labels == class_id) & artifacts).sum()) for class_id in range(1, 5)],
                "artifact_excluded_samples": excluded_samples, "clean_samples": sum(lengths),
                "segments": len(segments) - segment_start,
                "short_segments": sum(length < WINDOW for length in lengths),
                "unwindowed_samples": unwindowed_samples, "windows": len(windows) - window_start,
            })
            if excluded_samples + sum(lengths) != n_samples:
                raise RuntimeError(f"{path.name} run {run_id}: 时间未守恒")

        session_events = len(events) - session_event_start
        artifact_count = sum(run["artifact_events"] for run in run_summaries)
        summaries[session_name] = {
            "events": session_events, "artifact_events": artifact_count,
            "clean_events": session_events - artifact_count, "runs": run_summaries,
            "artifacts_per_class": [artifacts_per_class[class_id] for class_id in range(1, 5)],
            "segments": sum(run["segments"] for run in run_summaries),
            "windows": sum(run["windows"] for run in run_summaries),
        }

    # 结构化字段避免下游依赖易错的列号。
    arrays = (np.asarray(events, dtype=EVENT_DTYPE), np.asarray(segments, dtype=SEGMENT_DTYPE),
              np.asarray(windows, dtype=WINDOW_DTYPE))
    if len(arrays[0]) != 576:
        raise RuntimeError(f"Subject {subject}: 预期 576 个事件，实际 {len(arrays[0])}")
    return *arrays, {"subject": subject, "source_files": sources, "summaries": summaries}


def _preflight_frozen_write(pairs: list[tuple[Path, Path]]) -> None:
    """已存在文件只有字节完全一致时才允许复用。"""
    for temporary, target in pairs:
        if target.exists() and file_hash(temporary) != file_hash(target):
            raise FileExistsError(f"冻结文件已存在但内容不同，请升级 protocol_id: {target}")


def _install_frozen_files(pairs: list[tuple[Path, Path]]) -> None:
    for temporary, target in pairs:
        if target.exists():
            temporary.unlink()
        else:
            os.replace(temporary, target)


def save_subject(output_dir: Path, subject: int, built: tuple) -> dict:
    events, segments, windows, metadata = built
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_id = f"bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
    index_file = output_dir / f"{protocol_id}.npz"
    manifest_file = output_dir / f"{protocol_id}_manifest.json"
    temporary_files: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as stream:
            temporary_index = Path(stream.name)
        temporary_files.append(temporary_index)
        np.savez_compressed(temporary_index, events=events, segments=segments, windows=windows)
        manifest = {
            "protocol_id": protocol_id, "artifact_policy": ARTIFACT_POLICY, **metadata,
            "sampling_rate": FS, "window_samples": WINDOW, "step_samples": STEP,
            "session_codes": {"0": "0train", "1": "1test"},
            "index_file": index_file.name, "index_sha256": file_hash(temporary_index),
        }
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".json", delete=False,
                                         mode="w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            temporary_manifest = Path(stream.name)
        temporary_files.append(temporary_manifest)
        pairs = [(temporary_index, index_file), (temporary_manifest, manifest_file)]
        _preflight_frozen_write(pairs)
        _install_frozen_files(pairs)
        return manifest
    finally:
        for path in temporary_files:
            if path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest = save_subject(args.output_dir, subject, build_subject(args.data_root, subject))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
