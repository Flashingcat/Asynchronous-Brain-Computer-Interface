"""为共享训练窗口构建fold专属标准化参数，并提供统一窗口数据层。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_causal_filter_store import (
    FILTER_ID as CAUSAL_ID,
    CausalFilterStore,
    is_sha256,
)
from build_offline_view import load_base
from build_protocol_index import FS, file_hash
from build_signal_store import EEG_CHANNELS, write_frozen_json
from build_validation_folds import FOLD_ID, build_fold_manifest, load_offline
from build_zero_phase_filter_store import (
    FILTER_ID as ZERO_PHASE_ID,
    ZeroPhaseFilterStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZATION_ID = "bnci2014001_s{subject:02d}_shared_stage1_window_zscore_native250_v1"
DOMAINS = ("causal", "zero_phase")
DOMAIN_PREPROCESSING = {
    "causal": "causal_bandpass_only_no_standardization_no_resampling",
    "zero_phase": "zero_phase_fir_only_no_standardization_no_resampling",
}
IDENTITY_FIELDS = (
    "subject", "session", "run", "segment", "window", "start", "stop",
    "trial", "final_label", "stage1_label", "stage2_label", "is_task",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--signal-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


# ---------- 窗口身份与选择：训练分支必须逐行使用同一批正式窗口 ----------
def window_identity_hash(rows: np.ndarray) -> str:
    """以全部窗口元数据生成与平台字节序无关的稳定身份哈希。"""
    digest = hashlib.sha256()
    for row in rows:
        values = [int(row[name]) for name in IDENTITY_FIELDS]
        digest.update(("|".join(map(str, values)) + "\n").encode("ascii"))
    return digest.hexdigest()


def formal_mask(store, rows: np.ndarray) -> np.ndarray:
    return np.asarray([store.window_is_formal(row) for row in rows], dtype=np.bool_)


def shared_training_pool(stage1: np.ndarray, causal_store: CausalFilterStore,
                         zero_store: ZeroPhaseFilterStore) -> np.ndarray:
    """只保留训练session中两种滤波都合法的Stage 1窗口。"""
    mask = ((stage1["session"] == 0) & formal_mask(causal_store, stage1) &
            formal_mask(zero_store, stage1))
    rows = stage1[mask].copy()
    if len(rows) == 0:
        raise RuntimeError("共同训练窗口为空")
    return rows


def select_runs(rows: np.ndarray, runs: list[int]) -> np.ndarray:
    chosen = rows[np.isin(rows["run"], runs)].copy()
    if len(chosen) == 0 or np.any(chosen["session"] != 0):
        raise RuntimeError("标准化只能选择训练session中的非空run集合")
    return chosen


# ---------- 统计量：按模型实际看到的重叠训练窗口逐通道累计 ----------
def compute_window_statistics(store, rows: np.ndarray) -> dict:
    """以float64累计窗口维和时间维，返回不裁剪的逐通道均值与标准差。"""
    if len(rows) == 0:
        raise ValueError("不能从空窗口集合估计标准化参数")
    total = np.zeros(len(EEG_CHANNELS), dtype=np.float64)
    square = np.zeros(len(EEG_CHANNELS), dtype=np.float64)
    samples_per_window = None
    value_count = 0
    for row in rows:
        signal = np.asarray(store.read_window(row), dtype=np.float64)
        if signal.shape[0] != len(EEG_CHANNELS):
            raise RuntimeError("标准化窗口通道数错误")
        if samples_per_window is None:
            samples_per_window = signal.shape[1]
        elif signal.shape[1] != samples_per_window:
            raise RuntimeError("同一标准化集合出现不同窗口长度")
        total += signal.sum(axis=1)
        square += np.square(signal).sum(axis=1)
        value_count += signal.shape[1]

    mean = total / value_count
    variance = square / value_count - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    if (not np.isfinite(mean).all() or not np.isfinite(std).all() or
            np.any(std <= 0.0)):
        raise RuntimeError("训练窗口产生非有限或非正标准差，禁止静默clip")
    return {
        "window_count": len(rows),
        "samples_per_window": int(samples_per_window),
        "values_per_channel": int(value_count),
        "mean_volts": mean.tolist(),
        "std_volts": std.tolist(),
    }


def label_counts(rows: np.ndarray) -> dict:
    return {
        "stage1_idle": int((rows["stage1_label"] == 0).sum()),
        "stage1_task": int((rows["stage1_label"] == 1).sum()),
        "stage2_by_class": [int((rows["stage2_label"] == value).sum())
                            for value in range(4)],
    }


def build_entry(name: str, runs: list[int], pool: np.ndarray,
                stores: dict[str, object], fold: int | None = None) -> dict:
    rows = select_runs(pool, runs)
    stage2 = rows[rows["is_task"]]
    entry = {
        "name": name,
        "fold": fold,
        "train_runs": runs,
        "stage1_window_count": len(rows),
        "stage1_window_sha256": window_identity_hash(rows),
        "stage2_window_count": len(stage2),
        "stage2_window_sha256": window_identity_hash(stage2),
        "label_counts": label_counts(rows),
        "statistics": {
            domain: compute_window_statistics(store, rows)
            for domain, store in stores.items()
        },
    }
    if any(value["window_count"] != len(rows)
           for value in entry["statistics"].values()):
        raise RuntimeError("两种输入域没有使用同一批训练窗口")
    return entry


# ---------- 冻结构建：绑定离线视图、六折清单和两份滤波存储 ----------
def load_sources(index_dir: Path, signal_dir: Path, subject: int):
    _, _, _, _, base_index, base_manifest = load_base(index_dir, subject)
    stage1, stage2, _, offline_index, offline_manifest = load_offline(
        index_dir, subject, (base_index, base_manifest)
    )
    fold_path = index_dir / f"{FOLD_ID.format(subject=subject)}.json"
    if not fold_path.is_file():
        raise FileNotFoundError(f"请先构建六折验证清单: {fold_path}")
    fold_manifest = json.loads(fold_path.read_text(encoding="utf-8"))
    if fold_manifest != build_fold_manifest(index_dir, subject):
        raise RuntimeError("六折验证清单与当前冻结索引不一致")

    causal_path = signal_dir / CAUSAL_ID.format(subject=subject) / "manifest.json"
    zero_path = signal_dir / ZERO_PHASE_ID.format(subject=subject) / "manifest.json"
    causal_store = CausalFilterStore(causal_path, verify_hashes=True)
    zero_store = ZeroPhaseFilterStore(zero_path, verify_hashes=True)
    if causal_store.subject != subject or zero_store.subject != subject:
        raise RuntimeError("滤波存储subject不匹配")
    return (stage1, stage2, fold_manifest, fold_path, offline_index,
            offline_manifest, causal_store, zero_store)


def build_normalization_manifest(index_dir: Path, signal_dir: Path,
                                 subject: int) -> dict:
    (stage1, stage2, folds, fold_path, offline_index, offline_manifest,
     causal_store, zero_store) = load_sources(index_dir, signal_dir, subject)
    stores = {"causal": causal_store, "zero_phase": zero_store}
    pool = shared_training_pool(stage1, causal_store, zero_store)
    if not np.array_equal(pool[pool["is_task"]],
                          stage2[(stage2["session"] == 0) &
                                 formal_mask(causal_store, stage2) &
                                 formal_mask(zero_store, stage2)]):
        raise RuntimeError("共同Stage 2窗口不是共同Stage 1任务子集")

    entries = [
        build_entry(f"fold_{item['fold']}", item["train_runs"], pool,
                    stores, fold=int(item["fold"]))
        for item in folds["folds"]
    ]
    final_fit = build_entry("final_fit", folds["final_fit_runs"], pool, stores)
    manifest = {
        "protocol_id": NORMALIZATION_ID.format(subject=subject),
        "subject": subject,
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "method": "per_channel_zscore",
        "fit_scope": "shared_formal_stage1_training_windows_only",
        "aggregation_axes": "window_and_time",
        "overlap_weighting": "each_occurrence_in_overlapping_windows_counts_once",
        "std_policy": "reject_nonfinite_or_nonpositive_no_clipping",
        "domain_policy": "normalize_each_input_with_stats_from_same_filter_domain",
        "shared_pool": {
            "stage1_window_count": len(pool),
            "stage1_window_sha256": window_identity_hash(pool),
            "stage2_window_count": int(pool["is_task"].sum()),
            "stage2_window_sha256": window_identity_hash(pool[pool["is_task"]]),
        },
        "sources": {
            "offline_index_file": offline_index.name,
            "offline_index_sha256": file_hash(offline_index),
            "offline_manifest_file": offline_manifest.name,
            "offline_manifest_sha256": file_hash(offline_manifest),
            "fold_manifest_file": fold_path.name,
            "fold_manifest_sha256": file_hash(fold_path),
            "causal_manifest_file": f"{causal_store.manifest['protocol_id']}/manifest.json",
            "causal_manifest_sha256": file_hash(causal_store.manifest_path),
            "zero_phase_manifest_file": f"{zero_store.manifest['protocol_id']}/manifest.json",
            "zero_phase_manifest_sha256": file_hash(zero_store.manifest_path),
        },
        "folds": entries,
        "final_fit": final_fit,
    }
    validate_normalization_manifest(manifest)
    return manifest


def validate_normalization_manifest(manifest: dict) -> None:
    """验证统计结构和值域，读取器不得接受被篡改或静默clip的参数。"""
    subject = int(manifest.get("subject", -1))
    expected = {
        "protocol_id": NORMALIZATION_ID.format(subject=subject),
        "sampling_rate": FS,
        "channels": list(EEG_CHANNELS),
        "method": "per_channel_zscore",
        "fit_scope": "shared_formal_stage1_training_windows_only",
        "aggregation_axes": "window_and_time",
        "overlap_weighting": "each_occurrence_in_overlapping_windows_counts_once",
        "std_policy": "reject_nonfinite_or_nonpositive_no_clipping",
        "domain_policy": "normalize_each_input_with_stats_from_same_filter_domain",
    }
    if subject not in range(1, 10) or any(manifest.get(k) != v for k, v in expected.items()):
        raise RuntimeError("标准化清单顶层配置不匹配")
    folds = manifest.get("folds", [])
    if (not isinstance(folds, list) or len(folds) != 6 or
            [item.get("fold") for item in folds] != list(range(6))):
        raise RuntimeError("标准化清单必须包含六折和final_fit")
    shared = manifest.get("shared_pool", {})
    if (shared.get("stage1_window_count", 0) <= 0 or
            shared.get("stage2_window_count", 0) <= 0 or
            not is_sha256(shared.get("stage1_window_sha256")) or
            not is_sha256(shared.get("stage2_window_sha256"))):
        raise RuntimeError("共同训练窗口身份非法")
    sources = manifest.get("sources", {})
    expected_source_keys = {
        "offline_index_file", "offline_index_sha256",
        "offline_manifest_file", "offline_manifest_sha256",
        "fold_manifest_file", "fold_manifest_sha256",
        "causal_manifest_file", "causal_manifest_sha256",
        "zero_phase_manifest_file", "zero_phase_manifest_sha256",
    }
    if set(sources) != expected_source_keys:
        raise RuntimeError("标准化清单来源字段不完整")
    for key, value in sources.items():
        if key.endswith("_file"):
            path = Path(str(value))
            if not path.name or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("标准化清单包含非法来源路径")
        elif key.endswith("_sha256") and not is_sha256(value):
            raise RuntimeError("标准化清单包含非法来源哈希")

    entries = folds + [manifest.get("final_fit", {})]
    for entry in entries:
        fold = entry.get("fold")
        expected_runs = (list(range(6)) if fold is None else
                         [run for run in range(6) if run != fold])
        counts = entry.get("label_counts", {})
        if (entry.get("train_runs") != expected_runs or
                entry.get("stage1_window_count", 0) <= 0 or
                entry.get("stage2_window_count", 0) <= 0 or
                not is_sha256(entry.get("stage1_window_sha256")) or
                not is_sha256(entry.get("stage2_window_sha256")) or
                counts.get("stage1_idle", 0) + counts.get("stage1_task", 0) !=
                entry.get("stage1_window_count") or
                counts.get("stage1_task") != entry.get("stage2_window_count") or
                sum(counts.get("stage2_by_class", [])) != entry.get("stage2_window_count")):
            raise RuntimeError("标准化条目没有训练窗口")
        for domain in DOMAINS:
            stats = entry.get("statistics", {}).get(domain, {})
            mean = np.asarray(stats.get("mean_volts", []), dtype=np.float64)
            std = np.asarray(stats.get("std_volts", []), dtype=np.float64)
            if (mean.shape != (len(EEG_CHANNELS),) or std.shape != mean.shape or
                    not np.isfinite(mean).all() or not np.isfinite(std).all() or
                    np.any(std <= 0.0) or
                    stats.get("samples_per_window", 0) <= 0 or
                    stats.get("window_count") != entry["stage1_window_count"] or
                    stats.get("values_per_channel") !=
                    stats.get("window_count", 0) * stats.get("samples_per_window", 0)):
                raise RuntimeError(f"{entry.get('name')}的{domain}统计量非法")
    if (entries[-1].get("stage1_window_count") != shared["stage1_window_count"] or
            entries[-1].get("stage2_window_count") != shared["stage2_window_count"]):
        raise RuntimeError("final_fit没有覆盖全部共同训练窗口")


def save_normalization_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest['protocol_id']}.json"
    write_frozen_json(path, manifest)
    return path


def statistics_for(manifest: dict, domain: str, fold: int | None = None) -> dict:
    """按输入域取统计量；fold=None表示最终六run拟合参数。"""
    validate_normalization_manifest(manifest)
    if domain not in DOMAINS:
        raise ValueError(f"未知输入域: {domain}")
    entry = manifest["final_fit"] if fold is None else next(
        (item for item in manifest["folds"] if item["fold"] == fold), None
    )
    if entry is None:
        raise KeyError(f"未知fold: {fold}")
    # 运行时补上输入域、被试和store身份，使统一读取器能阻止拿错参数。
    return {
        **entry["statistics"][domain],
        "input_domain": domain,
        "subject": manifest["subject"],
        "store_protocol_id": Path(
            manifest["sources"][f"{domain}_manifest_file"]
        ).parts[0],
        "store_manifest_sha256": manifest["sources"][f"{domain}_manifest_sha256"],
        "normalization_protocol_id": manifest["protocol_id"],
        "fold": fold,
    }


def training_rows_for(manifest: dict, pool: np.ndarray, fold: int | None,
                      stage: int = 1) -> np.ndarray:
    """按冻结清单取训练行并核对身份；fold=None表示final-fit六run。"""
    validate_normalization_manifest(manifest)
    if stage not in (1, 2):
        raise ValueError("stage只能为1或2")
    entry = manifest["final_fit"] if fold is None else next(
        (item for item in manifest["folds"] if item["fold"] == fold), None
    )
    if entry is None:
        raise KeyError(f"未知fold: {fold}")
    rows = select_runs(pool, entry["train_runs"])
    if stage == 2:
        rows = rows[rows["is_task"]].copy()
    prefix = f"stage{stage}"
    if (len(rows) != entry[f"{prefix}_window_count"] or
            window_identity_hash(rows) != entry[f"{prefix}_window_sha256"]):
        raise RuntimeError("训练窗口与冻结标准化清单不一致")
    return rows


# ---------- 统一数据层：只读取窗口并应用既有参数，绝不在验证/测试时拟合 ----------
class NormalizedWindowDataset:
    """轻量NumPy数据集；未来训练器可直接包装为PyTorch Dataset。"""

    def __init__(self, rows: np.ndarray, store, statistics: dict,
                 expected_fold: int | None, label_field: str | None = None):
        self.rows = rows.copy()
        self.store = store
        self.mean = np.asarray(statistics["mean_volts"], dtype=np.float32)[:, None]
        self.std = np.asarray(statistics["std_volts"], dtype=np.float32)[:, None]
        self.samples = int(statistics["samples_per_window"])
        self.label_field = label_field
        if (self.mean.shape != (len(EEG_CHANNELS), 1) or self.std.shape != self.mean.shape or
                not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or
                np.any(self.std <= 0.0)):
            raise ValueError("数据集标准化参数非法")
        domain = statistics.get("input_domain")
        if (domain not in DOMAINS or
                store.manifest.get("preprocessing") != DOMAIN_PREPROCESSING[domain] or
                statistics.get("subject") != store.subject or
                statistics.get("store_protocol_id") != store.manifest.get("protocol_id") or
                statistics.get("store_manifest_sha256") != file_hash(store.manifest_path) or
                statistics.get("normalization_protocol_id") !=
                NORMALIZATION_ID.format(subject=store.subject) or
                statistics.get("fold") != expected_fold):
            raise ValueError("标准化参数与输入域、subject或信号存储不匹配")
        if label_field is not None and label_field not in (rows.dtype.names or ()):
            raise ValueError(f"窗口没有标签字段: {label_field}")
        if any(not store.window_is_formal(row) for row in self.rows):
            raise ValueError("数据集包含当前输入域的非正式窗口")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        signal = self.store.read_window(self.rows[index])
        if signal.shape != (len(EEG_CHANNELS), self.samples):
            raise RuntimeError(f"窗口形状与统计清单不一致: {signal.shape}")
        normalized = np.ascontiguousarray((signal - self.mean) / self.std,
                                          dtype=np.float32)
        if not np.isfinite(normalized).all():
            raise RuntimeError("标准化窗口出现非有限值")
        if self.label_field is None:
            return normalized
        return normalized, np.int64(self.rows[index][self.label_field])

    def metadata(self, index: int) -> np.void:
        return self.rows[index].copy()


def training_dataset_for(manifest: dict, pool: np.ndarray, store, domain: str,
                         fold: int | None, stage: int) -> NormalizedWindowDataset:
    """原子绑定同一fold的训练行和统计量，防止手工组合导致验证泄漏。"""
    rows = training_rows_for(manifest, pool, fold, stage=stage)
    statistics = statistics_for(manifest, domain, fold=fold)
    label_field = "stage1_label" if stage == 1 else "stage2_label"
    return NormalizedWindowDataset(
        rows, store, statistics, expected_fold=fold, label_field=label_field
    )


def evaluation_dataset_for(manifest: dict, rows: np.ndarray, store, domain: str,
                           fold: int | None,
                           label_field: str | None = None) -> NormalizedWindowDataset:
    """验证、测试和推理只应用指定fold参数，绝不在目标窗口上重新拟合。"""
    return NormalizedWindowDataset(
        rows, store, statistics_for(manifest, domain, fold=fold),
        expected_fold=fold, label_field=label_field,
    )


def main() -> None:
    args = parse_args()
    for subject in dict.fromkeys(args.subjects):
        manifest = build_normalization_manifest(args.index_dir, args.signal_dir, subject)
        path = save_normalization_manifest(args.output_dir, manifest)
        print(f"{manifest['protocol_id']}: shared={manifest['shared_pool']['stage1_window_count']} path={path}")


if __name__ == "__main__":
    main()
