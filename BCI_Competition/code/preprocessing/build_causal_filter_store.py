"""将原生干净 EEG segment 构建为正式 250 Hz 因果带通信号存储。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt

from build_protocol_index import ARTIFACT_POLICY, FS, file_hash
from build_signal_store import (
    EEG_CHANNELS,
    SEGMENT_POLICY,
    SIGNAL_ID,
    SignalStore,
    write_frozen_array,
    write_frozen_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILTER_ID = "bnci2014001_s{subject:02d}_eeg22_causal8_30hz_native250_v1"
LOW_HZ, HIGH_HZ = 8.0, 30.0
BUTTERWORTH_ORDER = 4
WARMUP_SAMPLES = FS  # 固定排除每个segment开头1秒，覆盖滤波器启动瞬态。
# 冻结由SciPy Butterworth设计得到的系数，避免不同SciPy版本末位变化阻断跨电脑读取。
FILTER_SOS = np.asarray([
    [0.003111903604586018, 0.0062238072091720361, 0.003111903604586018,
     1.0, -1.2672931915575452, 0.49378918414371398],
    [1.0, 2.0, 1.0, 1.0, -1.6370396227024675, 0.70196295360783312],
    [1.0, -2.0, 1.0, 1.0, -1.2863770965150896, 0.7317912749111386],
    [1.0, -2.0, 1.0, 1.0, -1.8683294067657878, 0.90915640650382246],
], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


# ---------- 因果滤波核心：显式传递状态，供整段和未来流式回放共用 ----------
def zero_filter_state(channel_count: int) -> np.ndarray:
    """生成每个segment独立使用的全零SOS状态，不继承伪迹前历史。"""
    if channel_count <= 0:
        raise ValueError("channel_count必须为正数")
    return np.zeros((FILTER_SOS.shape[0], channel_count, 2), dtype=np.float64)


def filter_chunk(signal: np.ndarray, state: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """仅用当前及历史样本前向滤波；返回状态可直接传给下一数据块。"""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signal必须为有限值的[channels, samples]二维非空数组")
    if state is None:
        state = zero_filter_state(values.shape[0])
    state = np.asarray(state, dtype=np.float64)
    expected = (FILTER_SOS.shape[0], values.shape[0], 2)
    if state.shape != expected or not np.isfinite(state).all():
        raise ValueError(f"滤波状态形状应为{expected}，实际为{state.shape}")
    filtered, final_state = sosfilt(FILTER_SOS, values, axis=1, zi=state)
    return filtered, final_state


def filter_segment(signal: np.ndarray) -> np.ndarray:
    """从全零状态独立过滤一个完整干净segment，并固定输出float32。"""
    filtered, _ = filter_chunk(signal)
    if not np.isfinite(filtered).all():
        raise RuntimeError("因果滤波产生非有限值")
    return np.ascontiguousarray(filtered, dtype=np.float32)


def validate_source_manifest(manifest: dict, subject: int) -> None:
    """拒绝将错误单位、通道或采样率的上游存储误当作正式输入。"""
    expected = {
        "protocol_id": SIGNAL_ID.format(subject=subject),
        "subject": subject,
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "stored_unit": "volts",
        "dtype": "float32",
        "artifact_policy": ARTIFACT_POLICY,
        "segment_policy": SEGMENT_POLICY,
        "preprocessing": "unit_conversion_only_no_filter_no_standardization",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Subject {subject}: 原生信号存储配置不匹配 {mismatches}")


def frozen_filter_spec() -> dict:
    """集中生成正式滤波规格，避免构建器和读取器各写一套后发生漂移。"""
    return {
        "family": "Butterworth",
        "implementation": "scipy.signal.sosfilt",
        "representation": "second_order_sections",
        "passband_hz": [LOW_HZ, HIGH_HZ],
        "butterworth_N": BUTTERWORTH_ORDER,
        "realized_bandpass_order": 2 * BUTTERWORTH_ORDER,
        "direction": "forward_only_causal",
        "state_initialization": "zeros_at_each_segment",
        "sos_coefficients_float64": FILTER_SOS.tolist(),
    }


def frozen_warmup_policy() -> dict:
    return {
        "configured_samples": WARMUP_SAMPLES,
        "configured_seconds": WARMUP_SAMPLES / FS,
        "window_rule": "reject_if_window_intersects_warmup_prefix",
        "stored_samples": "warmup_is_saved_for_audit_but_not_formal_scoring",
    }


def summarize_records(records: list[dict]) -> dict:
    """从segment记录重算两套session统计，清单不得只声明未经核对的总数。"""
    summaries = {}
    for session, name in ((0, "0train"), (1, "1test")):
        chosen = [item for item in records if item["session"] == session]
        summaries[name] = {
            "segments": len(chosen),
            "samples": sum(item["n_samples"] for item in chosen),
            "warmup_excluded_samples": sum(
                item["warmup_stop_native"] - item["warmup_start_native"]
                for item in chosen
            ),
            "formal_samples": sum(item["formal_samples"] for item in chosen),
        }
    return summaries


def is_sha256(value: object) -> bool:
    """只验证清单哈希的规范格式；内容一致性由读取器逐文件重新计算。"""
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def validate_filter_manifest(manifest: dict) -> None:
    """读取时再次核对正式身份，防止误把其他滤波产物送入评估。"""
    subject = int(manifest.get("subject", -1))
    expected = {
        "protocol_id": FILTER_ID.format(subject=subject),
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "stored_unit": "volts",
        "dtype": "float32",
        "artifact_policy": ARTIFACT_POLICY,
        "segment_policy": SEGMENT_POLICY,
        "preprocessing": "causal_bandpass_only_no_standardization_no_resampling",
        "filter": frozen_filter_spec(),
        "warmup_policy": frozen_warmup_policy(),
        "source_signal_protocol_id": SIGNAL_ID.format(subject=subject),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    records = manifest.get("segments", [])
    record_error = not isinstance(records, list) or not records
    keys = set()
    files = set()
    for item in records if isinstance(records, list) else []:
        key = (item.get("session"), item.get("run"), item.get("segment"))
        start, stop = int(item.get("start_native", -1)), int(item.get("stop_native", -1))
        count = int(item.get("n_samples", -1))
        warmup = min(WARMUP_SAMPLES, max(0, count))
        expected_record = {
            "shape": [len(EEG_CHANNELS), count],
            "warmup_start_native": start,
            "warmup_stop_native": start + warmup,
            "formal_start_native": start + warmup,
            "formal_samples": count - warmup,
        }
        path = Path(str(item.get("file", "")))
        if (key in keys or key[0] not in (0, 1) or start < 0 or stop <= start or
                count != stop - start or not path.name or path.is_absolute() or
                ".." in path.parts or str(path) in files or
                not is_sha256(item.get("sha256")) or
                any(item.get(name) != value for name, value in expected_record.items())):
            record_error = True
        keys.add(key)
        files.add(str(path))
    summaries_error = (not record_error and
                       manifest.get("summaries") != summarize_records(records))
    source_error = not is_sha256(manifest.get("source_signal_manifest_sha256"))
    if (subject not in range(1, 10) or mismatches or record_error or
            summaries_error or source_error):
        raise RuntimeError(f"因果滤波存储配置不匹配 {mismatches}")


# ---------- 冻结构建：保留全部坐标，只用formal_start_native标记正式有效区 ----------
def build_causal_filter_store(signal_dir: Path, output_dir: Path,
                              subject: int) -> tuple[dict, Path]:
    """逐segment因果滤波并写入可校验、不可静默覆盖的独立存储。"""
    source_manifest_path = (
        signal_dir / SIGNAL_ID.format(subject=subject) / "manifest.json"
    )
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"请先构建原生信号存储: {source_manifest_path}")
    source = SignalStore(source_manifest_path, verify_hashes=True)
    validate_source_manifest(source.manifest, subject)

    protocol_id = FILTER_ID.format(subject=subject)
    store_dir = output_dir / protocol_id
    store_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in source.manifest["segments"]:
        key = (int(item["session"]), int(item["run"]), int(item["segment"]))
        filtered = filter_segment(source.load_segment(*key))
        filename = item["file"]
        digest = write_frozen_array(store_dir / filename, filtered)
        warmup = min(WARMUP_SAMPLES, int(item["n_samples"]))
        records.append({
            **{name: item[name] for name in (
                "session", "run", "segment", "source_record_index",
                "start_native", "stop_native", "n_samples", "shape"
            )},
            "warmup_start_native": int(item["start_native"]),
            "warmup_stop_native": int(item["start_native"]) + warmup,
            "formal_start_native": int(item["start_native"]) + warmup,
            "formal_samples": int(item["n_samples"]) - warmup,
            "file": filename,
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
        "artifact_policy": ARTIFACT_POLICY,
        "segment_policy": SEGMENT_POLICY,
        "preprocessing": "causal_bandpass_only_no_standardization_no_resampling",
        "filter": frozen_filter_spec(),
        "warmup_policy": frozen_warmup_policy(),
        "source_signal_protocol_id": source.manifest["protocol_id"],
        "source_signal_manifest_sha256": file_hash(source_manifest_path),
        "summaries": summarize_records(records),
        "segments": records,
    }
    manifest_path = store_dir / "manifest.json"
    write_frozen_json(manifest_path, manifest)
    return manifest, manifest_path


# ---------- 正式读取：默认在数据入口阻断启动瞬态窗口 ----------
class CausalFilterStore:
    """读取因果滤波信号，并强制执行清单中的启动排除规则。"""

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
        """验证窗口坐标并判断其是否完整位于滤波启动段之后。"""
        required = {"subject", "session", "run", "segment", "start", "stop"}
        if not required.issubset(row.dtype.names or ()):
            raise TypeError("窗口元数据字段不完整")
        if int(row["subject"]) != self.subject:
            raise ValueError("窗口subject与滤波存储不一致")
        key = (int(row["session"]), int(row["run"]), int(row["segment"]))
        item = self._records.get(key)
        if item is None:
            raise KeyError(f"窗口引用未知segment: {key}")
        start, stop = int(row["start"]), int(row["stop"])
        if not (item["start_native"] <= start < stop <= item["stop_native"]):
            raise ValueError("窗口越出干净segment")
        return start >= item["formal_start_native"]

    def read_window(self, row: np.void, allow_warmup: bool = False) -> np.ndarray:
        """正式模式拒绝启动段窗口；显式allow_warmup只供诊断审计。"""
        is_formal = self.window_is_formal(row)
        if not allow_warmup and not is_formal:
            raise ValueError("窗口与因果滤波启动排除区间相交")
        return self._signal_store.read_window(row)

    def load_segment(self, session: int, run: int, segment: int) -> np.ndarray:
        return self._signal_store.load_segment(session, run, segment)


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest, path = build_causal_filter_store(
            args.signal_dir, args.output_dir, subject
        )
        total = sum(value["formal_samples"] for value in manifest["summaries"].values())
        print(f"{manifest['protocol_id']}: formal_samples={total} manifest={path}")


if __name__ == "__main__":
    main()
