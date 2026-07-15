"""从已冻结上游构建仅含训练 session 的自包含 OOF bundle。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from build_fold_normalization import (
    NORMALIZATION_ID,
    build_normalization_manifest,
    load_sources,
    shared_training_pool,
)
from build_signal_store import write_frozen_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "code" / "train"
import sys
sys.path.insert(0, str(TRAIN_DIR))

from oof_training_bundle import (  # noqa: E402
    ARTIFACT_POLICY,
    BUNDLE_ID,
    DOMAINS,
    SEGMENT_POLICY,
    file_hash,
    load_bundle,
    window_identity_hash,
)


def parse_args() -> argparse.Namespace:
    processed = PROJECT_ROOT / "data" / "processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=processed)
    parser.add_argument("--signal-dir", type=Path, default=processed)
    parser.add_argument("--normalization-manifest", type=Path, default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path, default=processed)
    return parser.parse_args()


# ---------- 冻结写入：bundle 复制 session0 文件，不依赖原联合 store 才能训练 ----------
def copy_frozen_file(source: Path, target: Path, expected_sha256: str) -> None:
    if not source.is_file() or file_hash(source) != expected_sha256:
        raise RuntimeError(f"bundle 上游文件缺失或哈希错误: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != expected_sha256:
            raise FileExistsError(f"冻结 bundle 文件内容不同: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != expected_sha256:
            raise RuntimeError("bundle 文件复制后哈希改变")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_frozen_npz(path: Path, stage1: np.ndarray,
                     stage2: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, stage1_windows=stage1,
                                stage2_windows=stage2)
        digest = file_hash(temporary)
        if path.exists():
            if file_hash(path) != digest:
                raise FileExistsError(f"冻结 bundle 索引内容不同: {path}")
        else:
            os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


# ---------- 构建边界：此命令可读取联合上游，产物明确只保留 session0 ----------
def build_bundle(index_dir: Path, signal_dir: Path, normalization_path: Path,
                 output_dir: Path, subject: int) -> tuple[dict, Path]:
    if subject not in range(1, 10):
        raise ValueError("subject 必须为 1 至 9")
    loaded_normalization = json.loads(
        normalization_path.read_text(encoding="utf-8")
    )
    rebuilt_normalization = build_normalization_manifest(
        index_dir, signal_dir, subject
    )
    if loaded_normalization != rebuilt_normalization:
        raise RuntimeError("标准化清单无法由当前冻结上游重建")

    (stage1, _, _, _, _, _, causal_store, zero_store) = load_sources(
        index_dir, signal_dir, subject
    )
    stores = {"causal": causal_store, "zero_phase": zero_store}
    pool = shared_training_pool(stage1, causal_store, zero_store)
    if np.any(pool["session"] != 0):
        raise RuntimeError("共同训练池包含测试 session")
    stage2 = pool[pool["is_task"]].copy()

    bundle_id = BUNDLE_ID.format(subject=subject)
    root = output_dir / bundle_id
    index_path = root / "windows.npz"
    index_sha256 = write_frozen_npz(index_path, pool, stage2)

    domain_entries: dict[str, dict] = {}
    for domain, store in stores.items():
        records = []
        for source_record in store.manifest["segments"]:
            if int(source_record["session"]) != 0:
                continue
            source = store.manifest_path.parent / source_record["file"]
            relative = Path(domain) / source_record["file"]
            copy_frozen_file(source, root / relative,
                             source_record["sha256"])
            formal_stop = int(source_record.get(
                "formal_stop_native", source_record["stop_native"]
            ))
            records.append({
                "session": 0,
                "run": int(source_record["run"]),
                "segment": int(source_record["segment"]),
                "start_native": int(source_record["start_native"]),
                "stop_native": int(source_record["stop_native"]),
                "n_samples": int(source_record["n_samples"]),
                "shape": list(source_record["shape"]),
                "formal_start_native": int(source_record["formal_start_native"]),
                "formal_stop_native": formal_stop,
                "file": relative.as_posix(),
                "sha256": source_record["sha256"],
            })
        identity_fields = {
            key: store.manifest[key]
            for key in ("filter", "warmup_policy", "edge_policy", "causality")
            if key in store.manifest
        }
        domain_entries[domain] = {
            "preprocessing": store.manifest["preprocessing"],
            "source_protocol_id": store.manifest["protocol_id"],
            "source_manifest_sha256": file_hash(store.manifest_path),
            "filter_identity": identity_fields,
            "segment_count": len(records),
            "segments": records,
        }

    fold_entries = []
    for normal_entry in loaded_normalization["folds"]:
        fold = int(normal_entry["fold"])
        train_stage1 = pool[pool["run"] != fold].copy()
        validation_stage1 = pool[pool["run"] == fold].copy()
        train_stage2 = train_stage1[train_stage1["is_task"]].copy()
        validation_stage2 = validation_stage1[validation_stage1["is_task"]].copy()
        if (normal_entry["train_runs"] !=
                [run for run in range(6) if run != fold]):
            raise RuntimeError("标准化 fold 的训练 run 非法")
        fold_entries.append({
            "fold": fold,
            "train_runs": normal_entry["train_runs"],
            "validation_runs": [fold],
            "train_stage1": {
                "window_count": len(train_stage1),
                "window_sha256": window_identity_hash(train_stage1),
            },
            "train_stage2": {
                "window_count": len(train_stage2),
                "window_sha256": window_identity_hash(train_stage2),
            },
            "validation_stage1": {
                "window_count": len(validation_stage1),
                "window_sha256": window_identity_hash(validation_stage1),
            },
            "validation_stage2": {
                "window_count": len(validation_stage2),
                "window_sha256": window_identity_hash(validation_stage2),
            },
            "statistics": {
                domain: normal_entry["statistics"][domain]
                for domain in DOMAINS
            },
        })

    source_provenance = {
        "normalization_manifest_sha256": file_hash(normalization_path),
        **{
            key: value for key, value in loaded_normalization["sources"].items()
            if key.endswith("_sha256")
        },
    }
    manifest = {
        "protocol_id": bundle_id,
        "dataset": "BNCI2014001",
        "subject": subject,
        "purpose": "session0_only_oof_training_and_validation",
        "included_session": 0,
        "sampling_rate": 250,
        "channels": loaded_normalization["channels"],
        "window_shape": [len(loaded_normalization["channels"]), 500],
        "artifact_policy": ARTIFACT_POLICY,
        "segment_policy": SEGMENT_POLICY,
        "test_session_content_in_bundle": False,
        "index_file": index_path.name,
        "index_sha256": index_sha256,
        "builder_source_sha256": file_hash(Path(__file__).resolve()),
        "normalization_protocol_id": loaded_normalization["protocol_id"],
        "shared_pool": {
            "stage1_window_count": len(pool),
            "stage1_window_sha256": window_identity_hash(pool),
            "stage2_window_count": len(stage2),
            "stage2_window_sha256": window_identity_hash(stage2),
        },
        "folds": fold_entries,
        "domains": domain_entries,
        "source_provenance": source_provenance,
    }
    manifest_path = root / "manifest.json"
    write_frozen_json(manifest_path, manifest)
    loaded = load_bundle(manifest_path, verify_hashes=True)
    if loaded.manifest != manifest or not np.array_equal(loaded.rows, pool):
        raise RuntimeError("OOF bundle 写入后无法逐值重载")
    return manifest, manifest_path


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        normalization = args.normalization_manifest
        if normalization is None:
            normalization = args.index_dir / (
                NORMALIZATION_ID.format(subject=subject) + ".json"
            )
        manifest, path = build_bundle(
            args.index_dir.resolve(), args.signal_dir.resolve(),
            normalization.resolve(), args.output_dir.resolve(), subject,
        )
        print(
            f"{manifest['protocol_id']}: "
            f"windows={manifest['shared_pool']['stage1_window_count']} "
            f"path={path}"
        )


if __name__ == "__main__":
    main()
