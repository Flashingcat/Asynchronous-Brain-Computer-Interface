"""读取完全隔离于 BNCI2014001 测试 session 的 OOF 训练 bundle。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BUNDLE_ID = "bnci2014001_s{subject:02d}_oof_train_session0_native250_v1"
DOMAINS = ("causal", "zero_phase")
DOMAIN_PREPROCESSING = {
    "causal": "causal_bandpass_only_no_standardization_no_resampling",
    "zero_phase": "zero_phase_fir_only_no_standardization_no_resampling",
}
CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
    "P2", "POz",
)
WINDOW_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("segment", "u1"),
    ("window", "<u4"), ("start", "<i8"), ("stop", "<i8"), ("trial", "<i2"),
    ("final_label", "u1"), ("stage1_label", "u1"), ("stage2_label", "i1"),
    ("is_task", "?"),
])
IDENTITY_FIELDS = (
    "subject", "session", "run", "segment", "window", "start", "stop",
    "trial", "final_label", "stage1_label", "stage2_label", "is_task",
)


# ---------- 内容身份：bundle 自己携带全部训练行、统计量和 session0 信号哈希 ----------
def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def window_identity_hash(rows: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in rows:
        values = [int(row[name]) for name in IDENTITY_FIELDS]
        digest.update(("|".join(map(str, values)) + "\n").encode("ascii"))
    return digest.hexdigest()


def _safe_relative_file(value: object) -> Path:
    path = Path(str(value))
    if (path.is_absolute() or not path.name or ".." in path.parts or
            path == Path(".")):
        raise RuntimeError(f"bundle 含非法相对路径: {value}")
    return path


def validate_bundle_manifest(manifest: dict) -> None:
    """拒绝含测试 session、缺 fold 或无法绑定输入域的训练 bundle。"""
    subject = int(manifest.get("subject", -1))
    expected = {
        "protocol_id": BUNDLE_ID.format(subject=subject),
        "dataset": "BNCI2014001",
        "purpose": "session0_only_oof_training_and_validation",
        "included_session": 0,
        "sampling_rate": 250,
        "channels": list(CHANNELS),
        "window_shape": [len(CHANNELS), 500],
        "test_session_content_in_bundle": False,
    }
    if (subject not in range(1, 10) or
            any(manifest.get(key) != value for key, value in expected.items())):
        raise RuntimeError("OOF bundle 顶层身份错误")
    if (not is_sha256(manifest.get("index_sha256")) or
            not is_sha256(manifest.get("builder_source_sha256"))):
        raise RuntimeError("OOF bundle 索引或构建器哈希非法")
    _safe_relative_file(manifest.get("index_file"))

    pool = manifest.get("shared_pool", {})
    if (pool.get("stage1_window_count", 0) <= 0 or
            pool.get("stage2_window_count", 0) <= 0 or
            not is_sha256(pool.get("stage1_window_sha256")) or
            not is_sha256(pool.get("stage2_window_sha256"))):
        raise RuntimeError("OOF bundle 共同窗口身份非法")

    folds = manifest.get("folds")
    if (not isinstance(folds, list) or len(folds) != 6 or
            [item.get("fold") for item in folds] != list(range(6))):
        raise RuntimeError("OOF bundle 必须包含六个固定 fold")
    for entry in folds:
        fold = entry["fold"]
        expected_runs = [run for run in range(6) if run != fold]
        if (entry.get("train_runs") != expected_runs or
                entry.get("validation_runs") != [fold]):
            raise RuntimeError("OOF bundle 的 run 划分非法")
        for split in ("train", "validation"):
            for stage in (1, 2):
                identity = entry.get(f"{split}_stage{stage}", {})
                if (identity.get("window_count", 0) <= 0 or
                        not is_sha256(identity.get("window_sha256"))):
                    raise RuntimeError("OOF bundle 的 fold 窗口身份非法")
        for domain in DOMAINS:
            stats = entry.get("statistics", {}).get(domain, {})
            mean = np.asarray(stats.get("mean_volts", []), dtype=np.float64)
            std = np.asarray(stats.get("std_volts", []), dtype=np.float64)
            if (mean.shape != (len(CHANNELS),) or std.shape != mean.shape or
                    not np.isfinite(mean).all() or not np.isfinite(std).all() or
                    np.any(std <= 0) or stats.get("samples_per_window") != 500 or
                    stats.get("window_count") != entry["train_stage1"]["window_count"]):
                raise RuntimeError("OOF bundle 的 fold 标准化统计非法")

    domains = manifest.get("domains", {})
    if set(domains) != set(DOMAINS):
        raise RuntimeError("OOF bundle 输入域不完整")
    for domain in DOMAINS:
        info = domains[domain]
        if (info.get("preprocessing") != DOMAIN_PREPROCESSING[domain] or
                not is_sha256(info.get("source_manifest_sha256"))):
            raise RuntimeError(f"OOF bundle 的 {domain} 身份非法")
        records = info.get("segments")
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"OOF bundle 的 {domain} segment 为空")
        keys, files = set(), set()
        for record in records:
            key = (record.get("session"), record.get("run"), record.get("segment"))
            path = _safe_relative_file(record.get("file"))
            start, stop = int(record.get("start_native", -1)), int(record.get("stop_native", -1))
            formal_start = int(record.get("formal_start_native", -1))
            formal_stop = int(record.get("formal_stop_native", -1))
            if (key in keys or key[0] != 0 or key[1] not in range(6) or
                    start < 0 or not start <= formal_start <= formal_stop <= stop or
                    record.get("n_samples") != stop - start or
                    record.get("shape") != [len(CHANNELS), stop - start] or
                    not is_sha256(record.get("sha256")) or str(path) in files):
                raise RuntimeError(f"OOF bundle 的 {domain} segment 记录非法")
            keys.add(key)
            files.add(str(path))
        if info.get("segment_count") != len(records):
            raise RuntimeError(f"OOF bundle 的 {domain} segment 数不一致")

    provenance = manifest.get("source_provenance", {})
    if (not isinstance(provenance, dict) or not provenance or
            any(not is_sha256(value) for value in provenance.values())):
        raise RuntimeError("OOF bundle 上游 provenance 非法")


# ---------- 训练信号读取器：复合键只注册 session0，session1 行没有可访问路径 ----------
class BundleSignalStore:
    def __init__(self, root: Path, subject: int, domain: str,
                 info: dict, verify_hashes: bool = True):
        self.root = root
        self.subject = subject
        self.domain = domain
        self.info = info
        self._records = {
            (int(item["session"]), int(item["run"]), int(item["segment"])): item
            for item in info["segments"]
        }
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}
        if verify_hashes:
            for record in info["segments"]:
                path = self.root / record["file"]
                if not path.is_file() or file_hash(path) != record["sha256"]:
                    raise RuntimeError(f"bundle session0 信号缺失或哈希错误: {path}")

    def window_is_formal(self, row: np.void) -> bool:
        if int(row["subject"]) != self.subject or int(row["session"]) != 0:
            return False
        record = self._records.get(
            (int(row["session"]), int(row["run"]), int(row["segment"]))
        )
        return bool(record is not None and
                    record["formal_start_native"] <= int(row["start"]) and
                    int(row["stop"]) <= record["formal_stop_native"])

    def read_window(self, row: np.void) -> np.ndarray:
        if not self.window_is_formal(row):
            raise ValueError("窗口不属于 bundle 的正式 session0 区间")
        key = (int(row["session"]), int(row["run"]), int(row["segment"]))
        record = self._records[key]
        if key not in self._cache:
            path = self.root / record["file"]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != record["shape"]:
                raise RuntimeError(f"bundle 信号结构错误: {path}")
            self._cache[key] = array
        local_start = int(row["start"]) - record["start_native"]
        local_stop = int(row["stop"]) - record["start_native"]
        return np.ascontiguousarray(self._cache[key][:, local_start:local_stop])


@dataclass
class BundleContext:
    manifest: dict
    manifest_path: Path
    manifest_sha256: str
    rows: np.ndarray
    stores: dict[str, BundleSignalStore]


def load_bundle(manifest_path: Path, verify_hashes: bool = True) -> BundleContext:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_bundle_manifest(manifest)
    root = manifest_path.parent
    index_path = root / manifest["index_file"]
    if not index_path.is_file() or file_hash(index_path) != manifest["index_sha256"]:
        raise RuntimeError("OOF bundle 窗口索引缺失或哈希错误")
    with np.load(index_path, allow_pickle=False) as data:
        if set(data.files) != {"stage1_windows", "stage2_windows"}:
            raise RuntimeError("OOF bundle 索引字段非法")
        stage1, stage2 = data["stage1_windows"].copy(), data["stage2_windows"].copy()
    pool = manifest["shared_pool"]
    if (stage1.dtype != WINDOW_DTYPE or stage2.dtype != WINDOW_DTYPE or
            np.any(stage1["session"] != 0) or np.any(stage2["session"] != 0) or
            not np.array_equal(stage2, stage1[stage1["is_task"]]) or
            len(stage1) != pool["stage1_window_count"] or
            len(stage2) != pool["stage2_window_count"] or
            window_identity_hash(stage1) != pool["stage1_window_sha256"] or
            window_identity_hash(stage2) != pool["stage2_window_sha256"]):
        raise RuntimeError("OOF bundle 窗口内容与清单不一致")
    stores = {
        domain: BundleSignalStore(root, manifest["subject"], domain,
                                  manifest["domains"][domain], verify_hashes)
        for domain in DOMAINS
    }
    return BundleContext(manifest, manifest_path, file_hash(manifest_path),
                         stage1, stores)


# ---------- fold 数据集：统计量和窗口选择均来自同一 bundle 条目，不能手工串错 ----------
def fold_entry(manifest: dict, fold: int) -> dict:
    if fold not in range(6):
        raise KeyError(f"未知 fold: {fold}")
    return manifest["folds"][fold]


def rows_for(manifest: dict, pool: np.ndarray, fold: int,
             stage: int, split: str) -> np.ndarray:
    if stage not in (1, 2) or split not in ("train", "validation"):
        raise ValueError("stage 或 split 非法")
    entry = fold_entry(manifest, fold)
    runs = entry["train_runs"] if split == "train" else entry["validation_runs"]
    rows = pool[np.isin(pool["run"], runs)].copy()
    if stage == 2:
        rows = rows[rows["is_task"]].copy()
    identity = entry[f"{split}_stage{stage}"]
    if (len(rows) != identity["window_count"] or
            window_identity_hash(rows) != identity["window_sha256"] or
            np.any(rows["session"] != 0)):
        raise RuntimeError("bundle fold 窗口选择与冻结身份不一致")
    return rows


class BundleWindowDataset:
    def __init__(self, context: BundleContext, rows: np.ndarray,
                 domain: str, fold: int, stage: int):
        if domain not in DOMAINS or stage not in (1, 2):
            raise ValueError("输入域或 stage 非法")
        self.rows = rows.copy()
        self.store = context.stores[domain]
        stats = fold_entry(context.manifest, fold)["statistics"][domain]
        self.mean = np.asarray(stats["mean_volts"], np.float32)[:, None]
        self.std = np.asarray(stats["std_volts"], np.float32)[:, None]
        self.label_field = "stage1_label" if stage == 1 else "stage2_label"
        if (self.mean.shape != (len(CHANNELS), 1) or self.std.shape != self.mean.shape or
                not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or
                np.any(self.std <= 0) or any(not self.store.window_is_formal(row)
                                             for row in self.rows)):
            raise RuntimeError("bundle 数据集统计量或正式窗口非法")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        signal = self.store.read_window(self.rows[index])
        if signal.shape != (len(CHANNELS), 500):
            raise RuntimeError("bundle 窗口形状错误")
        normalized = np.ascontiguousarray((signal - self.mean) / self.std,
                                          dtype=np.float32)
        if not np.isfinite(normalized).all():
            raise RuntimeError("bundle 标准化结果出现非有限值")
        return normalized, np.int64(self.rows[index][self.label_field])
