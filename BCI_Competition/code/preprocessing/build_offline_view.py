"""从冻结母索引派生原仓库风格的离线窗口视图。"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from build_protocol_index import (
    ARTIFACT_POLICY,
    EVENT_DTYPE,
    FS,
    SEGMENT_DTYPE,
    STEP,
    WINDOW,
    WINDOW_DTYPE,
    _install_frozen_files,
    _preflight_frozen_write,
    clean_segments,
    file_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_ID = "bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
OFFLINE_ID = "bnci2014001_s{subject:02d}_offline_original_grid_native250_v3"
TASK_DURATION = 4 * FS
TASK_OFFSETS = tuple(range(0, TASK_DURATION - WINDOW + 1, STEP))
OFFLINE_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("segment", "u1"),
    ("window", "<u4"), ("start", "<i8"), ("stop", "<i8"), ("trial", "<i2"),
    ("final_label", "u1"), ("stage1_label", "u1"), ("stage2_label", "i1"),
    ("is_task", "?"),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


def load_base(index_dir: Path, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, Path, Path]:
    """读取并验证指定被试的冻结母索引，拒绝文件名与内容身份不一致。"""
    base_id = BASE_ID.format(subject=subject)
    index_file = index_dir / f"{base_id}.npz"
    manifest_file = index_dir / f"{base_id}_manifest.json"
    if not index_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"请先构建冻结母索引: {index_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("index_file") != index_file.name:
        raise RuntimeError(f"Subject {subject}: 母索引清单记录的文件名不匹配")
    if manifest.get("index_sha256") != file_hash(index_file):
        raise RuntimeError(f"Subject {subject}: 母索引哈希校验失败")
    with np.load(index_file, allow_pickle=False) as data:
        if set(data.files) != {"events", "segments", "windows"}:
            raise RuntimeError(f"Subject {subject}: 母索引必须且只能包含 events/segments/windows")
        events = data["events"].copy()
        segments = data["segments"].copy()
        online_windows = data["windows"].copy()
    validate_base(events, segments, online_windows, manifest, subject)
    return events, segments, online_windows, manifest, index_file, manifest_file


def validate_base(events: np.ndarray, segments: np.ndarray, online_windows: np.ndarray,
                  manifest: dict, subject: int) -> None:
    """逐表重建母索引并核对清单，防止错误输入被首次冻结。"""
    if subject not in range(1, 10):
        raise ValueError("BNCI2014001 subject 必须为 1 至 9")
    expected_manifest = {
        "protocol_id": BASE_ID.format(subject=subject),
        "subject": subject,
        "artifact_policy": ARTIFACT_POLICY,
        "sampling_rate": FS,
        "window_samples": WINDOW,
        "step_samples": STEP,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Subject {subject}: 母索引清单配置不匹配 {mismatches}")

    expected_dtypes = (("events", events.dtype, EVENT_DTYPE),
                       ("segments", segments.dtype, SEGMENT_DTYPE),
                       ("windows", online_windows.dtype, WINDOW_DTYPE))
    wrong_dtypes = {name: (str(actual), str(expected))
                    for name, actual, expected in expected_dtypes if actual != expected}
    if wrong_dtypes:
        raise RuntimeError(f"Subject {subject}: 母索引 structured dtype 不匹配 {wrong_dtypes}")
    arrays = (("events", events), ("segments", segments), ("windows", online_windows))
    if any(np.unique(array["subject"]).tolist() != [subject] for _, array in arrays):
        raise RuntimeError(f"Subject {subject}: 母索引三张表中的被试身份不一致")

    expected_keys = {(session, run) for session in (0, 1) for run in range(6)}
    event_counts = Counter((int(row["session"]), int(row["run"])) for row in events)
    segment_keys = {(int(row["session"]), int(row["run"])) for row in segments}
    window_keys = {(int(row["session"]), int(row["run"])) for row in online_windows}
    if set(event_counts) != expected_keys or any(event_counts[key] != 48 for key in expected_keys):
        raise RuntimeError(f"Subject {subject}: events 不是每个 session/run 各 48 个 trial")
    if segment_keys != expected_keys:
        raise RuntimeError(f"Subject {subject}: segments 未覆盖全部 12 个任务 run")
    if window_keys != expected_keys:
        raise RuntimeError(f"Subject {subject}: windows 包含缺失或越界的 session/run")
    validate_event_timing(events, subject)

    # summary中的原始长度决定IDLE网格，必须能和事件、segment及在线窗口精确互相重建。
    summaries = manifest.get("summaries")
    if not isinstance(summaries, dict) or set(summaries) != {"0train", "1test"}:
        raise RuntimeError(f"Subject {subject}: 母索引 summaries 结构不完整")
    for session, session_name in ((0, "0train"), (1, "1test")):
        run_summaries = summaries[session_name].get("runs")
        if (not isinstance(run_summaries, list) or len(run_summaries) != 6 or
                [item.get("run") for item in run_summaries] != list(range(6))):
            raise RuntimeError(f"Subject {subject}: {session_name} 的 run summary 顺序或数量错误")
        session_artifacts_per_class = Counter()
        session_artifact_events = session_segments = session_windows = 0
        for run, run_summary in enumerate(run_summaries):
            run_events = events[(events["session"] == session) & (events["run"] == run)]
            if run_events["trial"].tolist() != list(range(48)):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: trial编号不完整")
            if Counter(run_events["class_id"].tolist()) != Counter({1: 12, 2: 12, 3: 12, 4: 12}):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: 类别必须为1至4各12个")
            n_samples = run_summary.get("original_samples")
            if not isinstance(n_samples, int) or n_samples <= 0:
                raise RuntimeError(f"Subject {subject} session {session} run {run}: 原始长度无效")
            trial_starts = run_events["trial_start"]
            if (np.any(trial_starts < 0) or np.any(np.diff(trial_starts) <= 0) or
                    np.any(trial_starts + 6 * FS > n_samples)):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: trial时间越界或未严格递增")
            excluded = [(int(row["trial_start"]), int(row["trial_start"] + 6 * FS))
                        for row in run_events if row["artifact"]]
            expected_segments = clean_segments(n_samples, excluded)
            selected = segments[(segments["session"] == session) & (segments["run"] == run)]
            selected = np.sort(selected, order="segment")
            actual_segments = [(int(row["start"]), int(row["stop"]), str(row["reset_reason"]))
                               for row in selected]
            if selected["segment"].tolist() != list(range(len(selected))) or actual_segments != expected_segments:
                raise RuntimeError(f"Subject {subject} session {session} run {run}: segment与清单长度不一致")

            # 在线母窗口由干净segment唯一决定；逐行重建可发现长度与segment协同篡改。
            expected_window_rows = []
            for segment_row in selected:
                for window_id, start in enumerate(range(int(segment_row["start"]),
                                                        int(segment_row["stop"]) - WINDOW + 1, STEP)):
                    stop = start + WINDOW
                    expected_window_rows.append((subject, session, run, int(segment_row["segment"]),
                                                 window_id, start, stop, stop / FS))
            expected_online = np.asarray(expected_window_rows, dtype=WINDOW_DTYPE)
            actual_online = online_windows[(online_windows["session"] == session) &
                                           (online_windows["run"] == run)]
            if not np.array_equal(actual_online, expected_online):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: 在线母窗口无法由segment重建")

            artifact_count = int(run_events["artifact"].sum())
            artifacts_per_class = [int(((run_events["class_id"] == class_id) &
                                        run_events["artifact"]).sum()) for class_id in range(1, 5)]
            lengths = [stop - start for start, stop, _ in expected_segments]
            clean_samples = sum(lengths)
            expected_run_summary = {
                "artifact_events": artifact_count,
                "clean_events": 48 - artifact_count,
                "artifacts_per_class": artifacts_per_class,
                "artifact_excluded_samples": n_samples - clean_samples,
                "clean_samples": clean_samples,
                "segments": len(expected_segments),
                "short_segments": sum(length < WINDOW for length in lengths),
                "unwindowed_samples": sum(length if length < WINDOW else (length - WINDOW) % STEP
                                            for length in lengths),
                "windows": len(expected_online),
            }
            wrong_summary = {key: (run_summary.get(key), expected)
                             for key, expected in expected_run_summary.items()
                             if run_summary.get(key) != expected}
            if wrong_summary:
                raise RuntimeError(
                    f"Subject {subject} session {session} run {run}: run summary不匹配 {wrong_summary}"
                )
            session_artifacts_per_class.update(
                {class_id: artifacts_per_class[class_id - 1] for class_id in range(1, 5)}
            )
            session_artifact_events += artifact_count
            session_segments += len(expected_segments)
            session_windows += len(expected_online)

        expected_session_summary = {
            "events": 288,
            "artifact_events": session_artifact_events,
            "clean_events": 288 - session_artifact_events,
            "artifacts_per_class": [session_artifacts_per_class[class_id] for class_id in range(1, 5)],
            "segments": session_segments,
            "windows": session_windows,
        }
        wrong_session_summary = {key: (summaries[session_name].get(key), expected)
                                 for key, expected in expected_session_summary.items()
                                 if summaries[session_name].get(key) != expected}
        if wrong_session_summary:
            raise RuntimeError(f"Subject {subject}: {session_name} summary不匹配 {wrong_session_summary}")


def validate_event_timing(events: np.ndarray, subject: int) -> None:
    """确认母索引中的 MI 始终是 trial 后 2 至 6 秒的完整四秒区间。"""
    valid = ((events["mi_start"] == events["trial_start"] + 2 * FS) &
             (events["mi_stop"] == events["trial_start"] + 6 * FS) &
             (events["mi_stop"] - events["mi_start"] == TASK_DURATION))
    if not np.all(valid):
        bad = events[int(np.flatnonzero(~valid)[0])]
        raise RuntimeError(
            f"Subject {subject} session {int(bad['session'])} run {int(bad['run'])} "
            f"trial {int(bad['trial'])}: MI 区间不是 trial 后 2 至 6 秒"
        )


def locate_segment(segments: np.ndarray, session: int, run: int, start: int, stop: int) -> int | None:
    """返回完整包含窗口的唯一干净 segment。"""
    mask = ((segments["session"] == session) & (segments["run"] == run) &
            (segments["start"] <= start) & (stop <= segments["stop"]))
    found = segments[mask]
    if len(found) > 1:
        raise RuntimeError("同一窗口被多个 segment 包含")
    return int(found[0]["segment"]) if len(found) == 1 else None


def overlaps_any(start: int, stop: int, intervals: np.ndarray) -> bool:
    """按左闭右开区间判断窗口是否与任一 MI 区间相交。"""
    return bool(np.any((start < intervals[:, 1]) & (stop > intervals[:, 0])))


def contained_in_any(start: int, stop: int, intervals: np.ndarray) -> bool:
    """判断窗口是否完整位于任一 MI 区间内。"""
    return bool(np.any((intervals[:, 0] <= start) & (stop <= intervals[:, 1])))


def build_offline_view(events: np.ndarray, segments: np.ndarray, online_windows: np.ndarray,
                       base_manifest: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """从已验证母索引构建 Stage 1 与 Stage 2 离线窗口表。"""
    subject = int(base_manifest["subject"])
    validate_base(events, segments, online_windows, base_manifest, subject)
    rows: list[tuple] = []
    run_summaries: dict[str, list[dict]] = {"0train": [], "1test": []}

    for session, session_name in ((0, "0train"), (1, "1test")):
        for run in range(6):
            run_events = events[(events["session"] == session) & (events["run"] == run)]
            clean_events = run_events[np.logical_not(run_events["artifact"])]
            run_segments = segments[(segments["session"] == session) & (segments["run"] == run)]
            n_samples = int(base_manifest["summaries"][session_name]["runs"][run]["original_samples"])
            mi_intervals = np.column_stack((run_events["mi_start"], run_events["mi_stop"]))
            run_rows: list[tuple] = []

            # Task窗口以MI起点为锚；每个完整4秒MI产生5个2秒窗口。
            for event in clean_events:
                for offset in TASK_OFFSETS:
                    start, stop = int(event["mi_start"] + offset), int(event["mi_start"] + offset + WINDOW)
                    if not (int(event["mi_start"]) <= start < stop <= int(event["mi_stop"])):
                        raise RuntimeError(f"Subject {subject} session {session} run {run}: Task窗口越出MI区间")
                    segment = locate_segment(run_segments, session, run, start, stop)
                    if segment is None:
                        raise RuntimeError(f"Subject {subject} session {session} run {run}: 干净MI不在有效segment")
                    class_id = int(event["class_id"])
                    run_rows.append((subject, session, run, segment, 0, start, stop,
                                     int(event["trial"]), class_id, 1, class_id - 1, True))

            # IDLE窗口沿用原仓库的run起点网格，同时不得跨伪迹缺口或MI边界。
            grid_total = pure_mi_count = boundary_count = gap_count = 0
            for start in range(0, n_samples - WINDOW + 1, STEP):
                stop = start + WINDOW
                grid_total += 1
                segment = locate_segment(run_segments, session, run, start, stop)
                if segment is None:
                    gap_count += 1
                elif overlaps_any(start, stop, mi_intervals):
                    if contained_in_any(start, stop, mi_intervals):
                        pure_mi_count += 1
                    else:
                        boundary_count += 1
                else:
                    run_rows.append((subject, session, run, segment, 0, start, stop,
                                     -1, 0, 0, -1, False))

            run_rows.sort(key=lambda row: (row[5], row[11]))
            run_rows = [row[:4] + (window_id,) + row[5:] for window_id, row in enumerate(run_rows)]
            rows.extend(run_rows)
            task_count = sum(row[11] for row in run_rows)
            run_summaries[session_name].append({
                "run": run, "clean_events": len(clean_events), "task_windows": task_count,
                "idle_windows": len(run_rows) - task_count, "offline_windows": len(run_rows),
                "run_grid_candidates": grid_total, "pure_mi_grid_windows": pure_mi_count,
                "boundary_grid_windows": boundary_count,
                "artifact_or_gap_grid_windows": gap_count,
            })

    stage1 = np.asarray(rows, dtype=OFFLINE_DTYPE)
    stage2 = stage1[stage1["is_task"]].copy()
    summary = {}
    for session, name in ((0, "0train"), (1, "1test")):
        selected = stage1[stage1["session"] == session]
        task = selected[selected["is_task"]]
        summary[name] = {
            "stage1_windows": len(selected), "idle_windows": int((~selected["is_task"]).sum()),
            "task_windows": len(task), "task_windows_per_class": [
                int((task["final_label"] == class_id).sum()) for class_id in range(1, 5)
            ], "runs": run_summaries[name],
        }
    return stage1, stage2, summary


def save_offline_view(output_dir: Path, subject: int, built: tuple, base_files: tuple[Path, Path]) -> dict:
    """重新核对母索引和派生结果，再以不可静默覆盖的方式冻结产物。"""
    stage1, stage2, summary = built
    base_index, base_manifest = base_files
    expected_base_id = BASE_ID.format(subject=subject)
    expected_names = (f"{expected_base_id}.npz", f"{expected_base_id}_manifest.json")
    if ((base_index.name, base_manifest.name) != expected_names or
            base_index.parent.resolve() != base_manifest.parent.resolve()):
        raise RuntimeError(f"Subject {subject}: 母索引文件身份或路径不匹配")

    # 冻结前从母文件重新派生一次，确保 subject、数组、summary 和 Stage 2 子集一致。
    events, segments, online_windows, verified_manifest, verified_index, verified_manifest_file = load_base(
        base_index.parent, subject
    )
    if verified_index.resolve() != base_index.resolve() or verified_manifest_file.resolve() != base_manifest.resolve():
        raise RuntimeError(f"Subject {subject}: 实际加载的母索引文件与请求不一致")
    expected_stage1, expected_stage2, expected_summary = build_offline_view(
        events, segments, online_windows, verified_manifest
    )
    if (stage1.dtype != expected_stage1.dtype or not np.array_equal(stage1, expected_stage1) or
            stage2.dtype != expected_stage2.dtype or not np.array_equal(stage2, expected_stage2) or
            summary != expected_summary):
        raise RuntimeError(f"Subject {subject}: 待冻结离线视图不是母索引的完整一致派生结果")

    protocol_id = OFFLINE_ID.format(subject=subject)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / f"{protocol_id}.npz"
    manifest_file = output_dir / f"{protocol_id}_manifest.json"
    temporary_files: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as stream:
            temporary_index = Path(stream.name)
        temporary_files.append(temporary_index)
        np.savez_compressed(temporary_index, stage1_windows=stage1, stage2_windows=stage2)
        manifest = {
            "protocol_id": protocol_id, "subject": subject, "artifact_policy": ARTIFACT_POLICY,
            "base_index_file": base_index.name, "base_index_sha256": file_hash(base_index),
            "base_manifest_file": base_manifest.name, "base_manifest_sha256": file_hash(base_manifest),
            "sampling_rate": FS, "window_samples": WINDOW, "step_samples": STEP,
            "interval_convention": "[start, stop)", "task_duration_samples": TASK_DURATION,
            "task_offsets_samples": list(TASK_OFFSETS),
            "task_grid": "mi_onset_anchored", "idle_grid": "run_start_anchored",
            "idle_policy": "complete_in_clean_segment_and_no_mi_overlap",
            "boundary_policy": "exclude_from_offline_view",
            "window_id_scope": "stage1_per_run",
            "stage2_window_id_policy": "preserve_stage1_window_id",
            "summaries": summary,
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
        events, segments, online_windows, base_manifest, base_index, base_manifest_file = load_base(
            args.index_dir, subject
        )
        built = build_offline_view(events, segments, online_windows, base_manifest)
        manifest = save_offline_view(args.output_dir, subject, built, (base_index, base_manifest_file))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
