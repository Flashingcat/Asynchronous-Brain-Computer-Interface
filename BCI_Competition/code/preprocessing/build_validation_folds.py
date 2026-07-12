"""为 BNCI2014001 生成训练 session 内固定的留一 run 验证清单。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from build_offline_view import OFFLINE_DTYPE, OFFLINE_ID, load_base
from build_protocol_index import file_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = tuple(range(6))
FOLD_ID = "bnci2014001_s{subject:02d}_train6fold_leave_one_run_out_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


def load_offline(index_dir: Path, subject: int, base_files: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray, dict, Path, Path]:
    """读取离线视图，并确认它与当前被试的冻结母索引对应。"""
    protocol_id = OFFLINE_ID.format(subject=subject)
    index_file = index_dir / f"{protocol_id}.npz"
    manifest_file = index_dir / f"{protocol_id}_manifest.json"
    if not index_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"请先构建离线窗口视图: {index_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base_index, base_manifest = base_files
    expected = {
        "protocol_id": protocol_id,
        "subject": subject,
        "index_file": index_file.name,
        "index_sha256": file_hash(index_file),
        "base_index_file": base_index.name,
        "base_index_sha256": file_hash(base_index),
        "base_manifest_file": base_manifest.name,
        "base_manifest_sha256": file_hash(base_manifest),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Subject {subject}: 离线视图与母索引不匹配")

    with np.load(index_file, allow_pickle=False) as data:
        if set(data.files) != {"stage1_windows", "stage2_windows"}:
            raise RuntimeError(f"Subject {subject}: 离线视图表不完整")
        stage1 = data["stage1_windows"].copy()
        stage2 = data["stage2_windows"].copy()
    if (stage1.dtype != OFFLINE_DTYPE or stage2.dtype != OFFLINE_DTYPE or
            not np.array_equal(stage2, stage1[stage1["is_task"]])):
        raise RuntimeError(f"Subject {subject}: Stage 1/Stage 2 离线视图不一致")
    return stage1, stage2, manifest, index_file, manifest_file


def count_partition(arrays: tuple[np.ndarray, ...], session: int, runs: list[int]) -> dict:
    """按同一组完整 run 统计母索引和离线视图中的数据量。"""
    events, segments, online_windows, stage1, stage2 = arrays

    def select(array: np.ndarray) -> np.ndarray:
        return array[(array["session"] == session) & np.isin(array["run"], runs)]

    chosen_events = select(events)
    chosen_stage1 = select(stage1)
    return {
        "events": len(chosen_events),
        "clean_events": int(np.logical_not(chosen_events["artifact"]).sum()),
        "artifact_events": int(chosen_events["artifact"].sum()),
        "segments": len(select(segments)),
        "online_windows": len(select(online_windows)),
        "offline_stage1_windows": len(chosen_stage1),
        "offline_idle_windows": int(np.logical_not(chosen_stage1["is_task"]).sum()),
        "offline_task_windows": int(chosen_stage1["is_task"].sum()),
        "offline_stage2_windows": len(select(stage2)),
    }


def build_fold_manifest(index_dir: Path, subject: int) -> dict:
    """生成六折清单；fold 编号固定等于被留出的验证 run 编号。"""
    events, segments, online_windows, _, base_index, base_manifest = load_base(index_dir, subject)
    stage1, stage2, _, offline_index, offline_manifest = load_offline(
        index_dir, subject, (base_index, base_manifest)
    )
    arrays = (events, segments, online_windows, stage1, stage2)
    folds = []
    for validation_run in RUNS:
        train_runs = [run for run in RUNS if run != validation_run]
        folds.append({
            "fold": validation_run,
            "train_runs": train_runs,
            "validation_runs": [validation_run],
            "train_counts": count_partition(arrays, session=0, runs=train_runs),
            "validation_counts": count_partition(arrays, session=0, runs=[validation_run]),
        })

    protocol_id = FOLD_ID.format(subject=subject)
    return {
        "protocol_id": protocol_id,
        "subject": subject,
        "strategy": "six_fold_leave_one_complete_run_out",
        "split_unit": "complete_run",
        "split_seed": None,
        "train_session": 0,
        "final_test_session": 1,
        "all_train_runs": list(RUNS),
        "final_fit_runs": list(RUNS),
        "final_test_runs": list(RUNS),
        "selection_policy": {
            "validation_predictions": "out_of_fold_only",
            "aggregation": "aggregate_after_independent_run_replay",
            "chronology": "preserve_within_each_run",
            "preprocessing_fit": "current_fold_train_runs_only",
            "test_usage": "once_after_all_parameters_are_frozen",
        },
        "sources": {
            "base_index_file": base_index.name,
            "base_index_sha256": file_hash(base_index),
            "base_manifest_file": base_manifest.name,
            "base_manifest_sha256": file_hash(base_manifest),
            "offline_index_file": offline_index.name,
            "offline_index_sha256": file_hash(offline_index),
            "offline_manifest_file": offline_manifest.name,
            "offline_manifest_sha256": file_hash(offline_manifest),
        },
        "full_train_counts": count_partition(arrays, session=0, runs=list(RUNS)),
        "final_test_counts": count_partition(arrays, session=1, runs=list(RUNS)),
        "folds": folds,
    }


def save_fold_manifest(output_dir: Path, manifest: dict) -> Path:
    """同一协议只允许复用相同字节，内容变化时要求升级 protocol_id。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{manifest['protocol_id']}.json"
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"冻结验证清单内容不同，请升级 protocol_id: {target}")
        return target

    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest = build_fold_manifest(args.index_dir, subject)
        path = save_fold_manifest(args.output_dir, manifest)
        print(f"{manifest['protocol_id']}: {path}")


if __name__ == "__main__":
    main()
