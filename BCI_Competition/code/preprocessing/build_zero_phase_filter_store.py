"""将原生干净 EEG segment 构建为正式 250 Hz 零相位 FIR 信号存储。"""

from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
from mne.filter import create_filter, filter_data

from build_causal_filter_store import is_sha256, validate_source_manifest
from build_protocol_index import FS, file_hash
from build_signal_store import (
    EEG_CHANNELS,
    SIGNAL_ID,
    SignalStore,
    write_frozen_array,
    write_frozen_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILTER_ID = "bnci2014001_s{subject:02d}_eeg22_zero_phase8_30hz_native250_v1"
LOW_HZ, HIGH_HZ = 8.0, 30.0
FILTER_LENGTH = 413
HALF_SUPPORT = (FILTER_LENGTH - 1) // 2
LOW_TRANSITION_HZ, HIGH_TRANSITION_HZ = 2.0, 7.5
REFERENCE_MNE_VERSION = "1.11.0"
FIR_SHA256 = "e5320a1f9cb72f0384501dd51625b664c3723729eb73dc5c52f425f3f1fa9103"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


def coefficient_hash(coefficients: np.ndarray) -> str:
    """以固定小端float64字节计算FIR身份，避免平台本地字节序改变哈希。"""
    values = np.ascontiguousarray(coefficients, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


# ---------- 固定滤波器设计：把MNE auto得到的参数显式冻结 ----------
@lru_cache(maxsize=1)
def fir_coefficients() -> np.ndarray:
    """重建设计并核对413个系数；版本变化不能静默沿用同一协议ID。"""
    if mne.__version__ != REFERENCE_MNE_VERSION:
        raise RuntimeError(
            f"正式零相位协议要求MNE {REFERENCE_MNE_VERSION}，当前为{mne.__version__}"
        )
    values = create_filter(
        None, sfreq=FS, l_freq=LOW_HZ, h_freq=HIGH_HZ,
        filter_length=FILTER_LENGTH,
        l_trans_bandwidth=LOW_TRANSITION_HZ,
        h_trans_bandwidth=HIGH_TRANSITION_HZ,
        method="fir", phase="zero", fir_window="hamming",
        fir_design="firwin", verbose=False,
    )
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.shape != (FILTER_LENGTH,) or coefficient_hash(values) != FIR_SHA256:
        raise RuntimeError(
            f"当前MNE {mne.__version__}生成的FIR与冻结协议不一致，请使用兼容环境或升级protocol_id"
        )
    values.setflags(write=False)
    return values


def filter_segment(signal: np.ndarray) -> np.ndarray:
    """在完整clean segment上执行MNE零相位FIR，不按2秒窗口重复滤波。"""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signal必须为有限值的[channels, samples]二维非空数组")
    fir_coefficients()  # 先确认运行环境仍能生成冻结系数。
    filtered = filter_data(
        values, sfreq=FS, l_freq=LOW_HZ, h_freq=HIGH_HZ,
        filter_length=FILTER_LENGTH,
        l_trans_bandwidth=LOW_TRANSITION_HZ,
        h_trans_bandwidth=HIGH_TRANSITION_HZ,
        n_jobs=1, method="fir", copy=True, phase="zero",
        fir_window="hamming", fir_design="firwin",
        pad="reflect_limited", verbose=False,
    )
    if not np.isfinite(filtered).all():
        raise RuntimeError("零相位滤波产生非有限值")
    return np.ascontiguousarray(filtered, dtype=np.float32)


def formal_bounds(start: int, stop: int) -> tuple[int, int]:
    """排除两侧半个FIR支撑区；短segment返回空的正式区间。"""
    if start < 0 or stop <= start:
        raise ValueError("segment坐标非法")
    formal_start = min(stop, start + HALF_SUPPORT)
    formal_stop = max(formal_start, stop - HALF_SUPPORT)
    return formal_start, formal_stop


def frozen_filter_spec(coefficients: np.ndarray) -> dict:
    return {
        "family": "MNE_Hamming_FIR",
        "implementation": "mne.filter.filter_data",
        "reference_mne_version": REFERENCE_MNE_VERSION,
        "sampling_rate": FS,
        "passband_hz": [LOW_HZ, HIGH_HZ],
        "lower_transition_hz": LOW_TRANSITION_HZ,
        "upper_transition_hz": HIGH_TRANSITION_HZ,
        "filter_length_samples": FILTER_LENGTH,
        "phase": "zero_noncausal",
        "fir_window": "hamming",
        "fir_design": "firwin",
        "padding": "reflect_limited",
        "coefficients_dtype": "float64",
        "coefficients_sha256": FIR_SHA256,
        "coefficients": coefficients.tolist(),
    }


def frozen_edge_policy() -> dict:
    return {
        "half_support_samples": HALF_SUPPORT,
        "half_support_seconds": HALF_SUPPORT / FS,
        "excluded_sides": "segment_start_and_segment_end",
        "window_rule": "reject_if_window_intersects_either_edge_prefix_or_suffix",
        "stored_samples": "edge_outputs_are_saved_for_audit_but_not_formal_use",
    }


def summarize_records(records: list[dict]) -> dict:
    summaries = {}
    for session, name in ((0, "0train"), (1, "1test")):
        chosen = [item for item in records if item["session"] == session]
        summaries[name] = {
            "segments": len(chosen),
            "samples": sum(item["n_samples"] for item in chosen),
            "edge_excluded_samples": sum(
                item["n_samples"] - item["formal_samples"] for item in chosen
            ),
            "formal_samples": sum(item["formal_samples"] for item in chosen),
        }
    return summaries


def validate_filter_manifest(manifest: dict) -> None:
    """读取时核对FIR身份、双侧边缘坐标、来源和文件记录。"""
    subject = int(manifest.get("subject", -1))
    coefficients = np.asarray(
        manifest.get("filter", {}).get("coefficients", []), dtype=np.float64
    )
    coefficient_error = (
        coefficients.shape != (FILTER_LENGTH,) or
        not np.isfinite(coefficients).all() or
        coefficient_hash(coefficients) != FIR_SHA256
    )
    expected = {
        "protocol_id": FILTER_ID.format(subject=subject),
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "stored_unit": "volts",
        "dtype": "float32",
        "layout": "channels_first_c_contiguous",
        "preprocessing": "zero_phase_fir_only_no_standardization_no_resampling",
        "causality": "noncausal_uses_past_and_future_within_fir_half_support",
        "edge_policy": frozen_edge_policy(),
        "source_signal_protocol_id": SIGNAL_ID.format(subject=subject),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if not coefficient_error:
        expected_filter = frozen_filter_spec(coefficients)
        if manifest.get("filter") != expected_filter:
            mismatches["filter"] = (manifest.get("filter"), expected_filter)

    records = manifest.get("segments", [])
    record_error = not isinstance(records, list) or not records
    keys, files = set(), set()
    for item in records if isinstance(records, list) else []:
        key = (item.get("session"), item.get("run"), item.get("segment"))
        start, stop = int(item.get("start_native", -1)), int(item.get("stop_native", -1))
        count = int(item.get("n_samples", -1))
        try:
            formal_start, formal_stop = formal_bounds(start, stop)
        except ValueError:
            formal_start = formal_stop = -1
            record_error = True
        expected_record = {
            "shape": [len(EEG_CHANNELS), count],
            "left_edge_stop_native": min(stop, start + HALF_SUPPORT),
            "right_edge_start_native": max(start, stop - HALF_SUPPORT),
            "formal_start_native": formal_start,
            "formal_stop_native": formal_stop,
            "formal_samples": max(0, formal_stop - formal_start),
        }
        path = Path(str(item.get("file", "")))
        if (key in keys or key[0] not in (0, 1) or count != stop - start or
                not path.name or path.is_absolute() or ".." in path.parts or
                str(path) in files or not is_sha256(item.get("sha256")) or
                any(item.get(name) != value for name, value in expected_record.items())):
            record_error = True
        keys.add(key)
        files.add(str(path))
    summaries_error = (not record_error and
                       manifest.get("summaries") != summarize_records(records))
    source_error = not is_sha256(manifest.get("source_signal_manifest_sha256"))
    if (subject not in range(1, 10) or coefficient_error or mismatches or
            record_error or summaries_error or source_error):
        raise RuntimeError(f"零相位滤波存储配置不匹配 {mismatches}")


# ---------- 冻结构建：保存全部输出，正式区只保留不依赖反射填充的内部样本 ----------
def build_zero_phase_filter_store(signal_dir: Path, output_dir: Path,
                                  subject: int) -> tuple[dict, Path]:
    source_manifest_path = signal_dir / SIGNAL_ID.format(subject=subject) / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"请先构建原生信号存储: {source_manifest_path}")
    source = SignalStore(source_manifest_path, verify_hashes=True)
    validate_source_manifest(source.manifest, subject)
    coefficients = fir_coefficients()

    protocol_id = FILTER_ID.format(subject=subject)
    store_dir = output_dir / protocol_id
    store_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in source.manifest["segments"]:
        key = (int(item["session"]), int(item["run"]), int(item["segment"]))
        filtered = filter_segment(source.load_segment(*key))
        digest = write_frozen_array(store_dir / item["file"], filtered)
        start, stop = int(item["start_native"]), int(item["stop_native"])
        formal_start, formal_stop = formal_bounds(start, stop)
        records.append({
            **{name: item[name] for name in (
                "session", "run", "segment", "source_record_index",
                "start_native", "stop_native", "n_samples", "shape"
            )},
            "left_edge_stop_native": min(stop, start + HALF_SUPPORT),
            "right_edge_start_native": max(start, stop - HALF_SUPPORT),
            "formal_start_native": formal_start,
            "formal_stop_native": formal_stop,
            "formal_samples": formal_stop - formal_start,
            "file": item["file"],
            "sha256": digest,
        })

    manifest = {
        "protocol_id": protocol_id,
        "subject": subject,
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "stored_unit": "volts",
        "dtype": "float32",
        "layout": "channels_first_c_contiguous",
        "preprocessing": "zero_phase_fir_only_no_standardization_no_resampling",
        "causality": "noncausal_uses_past_and_future_within_fir_half_support",
        "filter": frozen_filter_spec(coefficients),
        "edge_policy": frozen_edge_policy(),
        "source_signal_protocol_id": source.manifest["protocol_id"],
        "source_signal_manifest_sha256": file_hash(source_manifest_path),
        "summaries": summarize_records(records),
        "segments": records,
    }
    manifest_path = store_dir / "manifest.json"
    write_frozen_json(manifest_path, manifest)
    return manifest, manifest_path


# ---------- 正式读取：默认同时阻断左侧和右侧反射填充影响区 ----------
class ZeroPhaseFilterStore:
    """读取零相位信号；allow_edges只能用于诊断，不得进入正式指标。"""

    def __init__(self, manifest_path: Path, verify_hashes: bool = True):
        self._signal_store = SignalStore(manifest_path, verify_hashes=verify_hashes)
        self.manifest_path = self._signal_store.manifest_path
        self.manifest = self._signal_store.manifest
        validate_filter_manifest(self.manifest)
        self.subject = self._signal_store.subject
        self._records = {
            (int(item["session"]), int(item["run"]), int(item["segment"])): item
            for item in self.manifest["segments"]
        }

    def window_is_formal(self, row: np.void) -> bool:
        required = {"subject", "session", "run", "segment", "start", "stop"}
        if not required.issubset(row.dtype.names or ()):
            raise TypeError("窗口元数据字段不完整")
        if int(row["subject"]) != self.subject:
            raise ValueError("窗口subject与零相位存储不一致")
        key = (int(row["session"]), int(row["run"]), int(row["segment"]))
        item = self._records.get(key)
        if item is None:
            raise KeyError(f"窗口引用未知segment: {key}")
        start, stop = int(row["start"]), int(row["stop"])
        if not (item["start_native"] <= start < stop <= item["stop_native"]):
            raise ValueError("窗口越出干净segment")
        return (start >= item["formal_start_native"] and
                stop <= item["formal_stop_native"])

    def read_window(self, row: np.void, allow_edges: bool = False) -> np.ndarray:
        is_formal = self.window_is_formal(row)
        if not allow_edges and not is_formal:
            raise ValueError("窗口与零相位滤波双侧边缘排除区间相交")
        return self._signal_store.read_window(row)

    def load_segment(self, session: int, run: int, segment: int) -> np.ndarray:
        return self._signal_store.load_segment(session, run, segment)


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest, path = build_zero_phase_filter_store(
            args.signal_dir, args.output_dir, subject
        )
        total = sum(value["formal_samples"] for value in manifest["summaries"].values())
        print(f"{manifest['protocol_id']}: formal_samples={total} manifest={path}")


if __name__ == "__main__":
    main()
