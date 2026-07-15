"""从冻结母索引构建 BNCI2014001 原生 250 Hz 干净 EEG segment 存储。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from build_offline_view import load_base
from build_protocol_index import ARTIFACT_POLICY, FS, file_hash, vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_ID = "bnci2014001_s{subject:02d}_eeg22_clean_segments_native250_v1"
SEGMENT_POLICY = "separate_clean_segments_no_time_compression"
SESSION_NAMES = {0: "0train", 1: "1test"}
EEG_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


def load_run_signals(data_root: Path, base_manifest: dict) -> dict[tuple[int, int], np.ndarray]:
    """按母清单中的原始 record 下标读取 22 通道 EEG，并统一转为 float32 伏特。"""
    subject = int(base_manifest["subject"])
    result: dict[tuple[int, int], np.ndarray] = {}
    for session, session_name in SESSION_NAMES.items():
        source = base_manifest["source_files"][session_name]
        path = data_root.resolve() / source["filename"]
        if not path.is_file() or file_hash(path) != source["sha256"]:
            raise RuntimeError(f"Subject {subject}: 原始文件缺失或哈希不匹配 {path}")
        records = vector(loadmat(path, squeeze_me=True, struct_as_record=False)["data"]).tolist()

        for run_summary in base_manifest["summaries"][session_name]["runs"]:
            run = int(run_summary["run"])
            source_record = int(run_summary["source_record_index"])
            if source_record not in range(len(records)):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: record下标越界")
            record = records[source_record]
            values = np.asarray(record.X)
            if (values.shape != (int(run_summary["original_samples"]), 25) or
                    np.asarray(record.fs).size != 1 or float(record.fs) != FS):
                raise RuntimeError(f"Subject {subject} session {session} run {run}: 原始信号结构不匹配")

            # MAT 中数值单位为微伏；先以 float64 换算为伏特，再固定为模型常用 float32。
            eeg = np.ascontiguousarray(values[:, :22].T * 1e-6, dtype=np.float32)
            result[(session, run)] = eeg
    return result


def write_frozen_array(target: Path, array: np.ndarray) -> str:
    """写入确定性的 NPY；已有文件只有内容哈希相同时才复用。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.save(temporary, array, allow_pickle=False)
        digest = file_hash(temporary)
        if target.exists():
            if file_hash(target) != digest:
                raise FileExistsError(f"冻结segment内容不同，请升级 protocol_id: {target}")
            temporary.unlink()
        else:
            os.replace(temporary, target)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def write_frozen_json(target: Path, value: dict) -> None:
    """以相同规则写入清单，避免后续实验静默覆盖旧数据。"""
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"冻结清单内容不同，请升级 protocol_id: {target}")
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_signal_store(data_root: Path, index_dir: Path, output_dir: Path, subject: int) -> tuple[dict, Path]:
    """保存每个干净segment；原始run坐标保留在清单中，不拼接时间轴。"""
    _, segments, _, base_manifest, base_index, base_manifest_file = load_base(index_dir, subject)
    if base_manifest.get("artifact_policy") != ARTIFACT_POLICY:
        raise RuntimeError("母索引没有绑定官方 trial 级伪迹排除策略")
    run_signals = load_run_signals(data_root, base_manifest)
    protocol_id = SIGNAL_ID.format(subject=subject)
    store_dir = output_dir / protocol_id
    store_dir.mkdir(parents=True, exist_ok=True)
    segment_records = []

    for row in segments:
        session, run, segment = int(row["session"]), int(row["run"]), int(row["segment"])
        start, stop = int(row["start"]), int(row["stop"])
        signal = run_signals[(session, run)]
        if not (0 <= start < stop <= signal.shape[1]):
            raise RuntimeError(f"Subject {subject} session {session} run {run}: segment坐标越界")
        clean_signal = np.ascontiguousarray(signal[:, start:stop])
        filename = f"session{session}_run{run:02d}_segment{segment:02d}.npy"
        digest = write_frozen_array(store_dir / filename, clean_signal)
        run_summary = base_manifest["summaries"][SESSION_NAMES[session]]["runs"][run]
        segment_records.append({
            "session": session,
            "run": run,
            "segment": segment,
            "source_record_index": int(run_summary["source_record_index"]),
            "start_native": start,
            "stop_native": stop,
            "n_samples": stop - start,
            "shape": [22, stop - start],
            "file": filename,
            "sha256": digest,
        })

    summaries = {}
    for session, session_name in SESSION_NAMES.items():
        chosen = [item for item in segment_records if item["session"] == session]
        summaries[session_name] = {
            "segments": len(chosen),
            "samples": sum(item["n_samples"] for item in chosen),
            "bytes": sum(item["n_samples"] * 22 * np.dtype(np.float32).itemsize for item in chosen),
        }
    manifest = {
        "protocol_id": protocol_id,
        "subject": subject,
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "channel_type": "eeg",
        "source_unit": "microvolts",
        "stored_unit": "volts",
        "dtype": "float32",
        "layout": "channels_first_c_contiguous",
        "artifact_policy": ARTIFACT_POLICY,
        "segment_policy": SEGMENT_POLICY,
        "preprocessing": "unit_conversion_only_no_filter_no_standardization",
        "base_index_file": base_index.name,
        "base_index_sha256": file_hash(base_index),
        "base_manifest_file": base_manifest_file.name,
        "base_manifest_sha256": file_hash(base_manifest_file),
        "source_files": base_manifest["source_files"],
        "summaries": summaries,
        "segments": segment_records,
    }
    write_frozen_json(store_dir / "manifest.json", manifest)
    return manifest, store_dir / "manifest.json"


class SignalStore:
    """通过 run 原始坐标读取 segment 或任意合法窗口。"""

    def __init__(self, manifest_path: Path, verify_hashes: bool = True):
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.is_dir():
            self.manifest_path = self.manifest_path / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.subject = int(self.manifest["subject"])
        self._records = {
            (int(item["session"]), int(item["run"]), int(item["segment"])): item
            for item in self.manifest["segments"]
        }
        if len(self._records) != len(self.manifest["segments"]):
            raise RuntimeError("segment复合键重复")
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}
        if verify_hashes:
            for item in self.manifest["segments"]:
                path = self.manifest_path.parent / item["file"]
                if not path.is_file() or file_hash(path) != item["sha256"]:
                    raise RuntimeError(f"segment文件缺失或哈希错误: {path}")

    def load_segment(self, session: int, run: int, segment: int) -> np.ndarray:
        key = (session, run, segment)
        if key not in self._records:
            raise KeyError(f"未知segment: {key}")
        if key not in self._cache:
            item = self._records[key]
            path = self.manifest_path.parent / item["file"]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != item["shape"]:
                raise RuntimeError(f"segment数组结构错误: {path}")
            self._cache[key] = array
        return self._cache[key]

    def read_window(self, row: np.void) -> np.ndarray:
        """根据母索引或离线视图的一行元数据读取对应 EEG，并返回独立连续数组。"""
        required = {"subject", "session", "run", "segment", "start", "stop"}
        if not required.issubset(row.dtype.names or ()):
            raise TypeError("窗口元数据字段不完整")
        if int(row["subject"]) != self.subject:
            raise ValueError("窗口subject与信号存储不一致")
        key = (int(row["session"]), int(row["run"]), int(row["segment"]))
        item = self._records.get(key)
        if item is None:
            raise KeyError(f"窗口引用未知segment: {key}")
        start, stop = int(row["start"]), int(row["stop"])
        if not (item["start_native"] <= start < stop <= item["stop_native"]):
            raise ValueError("窗口越出干净segment")
        local_start, local_stop = start - item["start_native"], stop - item["start_native"]
        return np.ascontiguousarray(self.load_segment(*key)[:, local_start:local_stop])


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest, path = build_signal_store(
            args.data_root, args.index_dir, args.output_dir, subject
        )
        print(f"{manifest['protocol_id']}: segments={len(manifest['segments'])} manifest={path}")


if __name__ == "__main__":
    main()
