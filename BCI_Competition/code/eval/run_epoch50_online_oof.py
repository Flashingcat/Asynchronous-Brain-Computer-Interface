"""使用固定 epoch 50 EEGNet checkpoint 运行单被试因果 OOF 在线基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# 训练时使用该配置；必须在导入 torch、初始化 CUDA 前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "code" / "train"
MODELS_DIR = PROJECT_ROOT / "code" / "models"
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
for source_dir in (TRAIN_DIR, MODELS_DIR, EVAL_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from model_factory import build_model  # noqa: E402
from oof_training_bundle import (  # noqa: E402
    BundleContext,
    artifact_contract,
    load_bundle,
    rows_for,
    window_identity_hash,
)
from protocol_metrics import (  # noqa: E402
    NATIVE_SAMPLING_RATE,
    NO_COMMAND,
    READY,
    STATEFUL_STRICT,
    STATELESS_DIAGNOSTIC,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)


FIXED_EPOCH = 50
INFERENCE_BATCH_SIZE = 256
FIXED_FOLDS = tuple(range(6))
KNOWN_SEEDS = (42, 43, 44)
KNOWN_SUBJECTS = tuple(range(1, 10))
LEGACY_INVENTORY_ID = "bnci2014001_s{subject:02d}_session0_causal_online_v1"
INVENTORY_ID = "bnci2014001_s{subject:02d}_session0_causal_online_v2"
OUTPUT_WINDOW_DTYPE = np.dtype([
    ("subject_id", "u1"), ("session_id", "u1"), ("run_id", "u1"),
    ("segment_id", "u1"), ("window_index", "<u4"),
    ("window_start_native", "<i8"), ("window_stop_native", "<i8"),
    ("window_start_model", "<i8"), ("window_stop_model", "<i8"),
    ("decision_time_seconds", "<f8"),
])


@dataclass
class OnlineInventory:
    """同一顺序下的评分对象和信号读取行；logit 必须与 windows 逐行对齐。"""

    segments: list[ScoringSegment]
    events: list[MIEvent]
    windows: list[ExpectedWindow]
    signal_rows: np.ndarray
    fully_warmup_excluded_segment_count: int
    fully_warmup_excluded_samples: int


@dataclass(frozen=True)
class SubjectPaths:
    """单被试运行的四类输入/输出路径，集中定义以避免各入口命名漂移。"""

    bundle_manifest: Path
    checkpoint_root: Path
    inventory_contract: Path
    output_root: Path


def default_subject_paths(subject: int) -> SubjectPaths:
    """解析既有 v1 复现实物；全新显式合同流程必须通过 CLI 传入 v2 路径。"""
    if type(subject) is not int or subject not in KNOWN_SUBJECTS:
        raise ValueError(f"subject 只能取 {KNOWN_SUBJECTS}")
    checkpoint_name = (
        "eegnet_oof_native250_v1" if subject == 1
        else f"eegnet_oof_extension_s{subject:02d}_native250_v1"
    )
    return SubjectPaths(
        bundle_manifest=(
            PROJECT_ROOT / "data" / "processed"
            / f"bnci2014001_s{subject:02d}_oof_train_session0_native250_v1"
            / "manifest.json"
        ),
        checkpoint_root=PROJECT_ROOT / "results" / "checkpoints" / checkpoint_name,
        inventory_contract=(
            PROJECT_ROOT / "config" / "evaluation"
            / f"bnci2014001_s{subject:02d}_session0_causal_online_v1.json"
        ),
        output_root=(
            PROJECT_ROOT / "results" / "tables"
            / f"s{subject:02d}_epoch50_causal_single_window_oof_v1"
        ),
    )


def bundle_contract_version(manifest: dict) -> str:
    """把 bundle 的伪迹绑定方式映射为唯一协议版本。"""
    subject = manifest.get("subject")
    if type(subject) is not int or subject not in KNOWN_SUBJECTS:
        raise RuntimeError("在线库存合同无法从非法被试身份派生")
    binding = artifact_contract(manifest)["artifact_policy_binding"]
    return "v2" if binding == "explicit_bundle_manifest" else "v1"


def inventory_contract_protocol_id(manifest: dict) -> str:
    """由 bundle 版本唯一确定在线库存合同协议名。"""
    subject = manifest["subject"]
    template = (
        INVENTORY_ID
        if bundle_contract_version(manifest) == "v2"
        else LEGACY_INVENTORY_ID
    )
    return template.format(subject=subject)


# ---------- 通用审计工具：所有外部输入和输出均以 SHA-256 绑定 ----------
def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def lf_normalized_hash(path: Path) -> str:
    """跨平台只放宽 CRLF/LF 差异，其他任意字节变化仍拒绝。"""
    normalized = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT.parent,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT.parent,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {"commit": commit, "dirty": dirty}


def verify_model_sources(checkpoint_root: Path, contract: dict) -> str:
    """核对推理模型和实际 bundle reader；换行差异须由覆盖该矩阵的快照证明。"""
    source_paths = {
        "oof_training_bundle_reader": TRAIN_DIR / "oof_training_bundle.py",
        "model_factory": MODELS_DIR / "model_factory.py",
        "eegnet": MODELS_DIR / "models" / "eegnet.py",
    }
    expected = contract["source_sha256"]
    if all(file_hash(path) == expected.get(role) for role, path in source_paths.items()):
        return "exact_training_bytes"

    candidates = (
        checkpoint_root / "posthoc_source_snapshot",
        PROJECT_ROOT / "results" / "checkpoints"
        / "eegnet_oof_native250_v1" / "posthoc_source_snapshot",
    )
    snapshot_root = next(
        (candidate for candidate in candidates if (candidate / "manifest.json").is_file()),
        None,
    )
    if snapshot_root is None:
        raise RuntimeError("模型源码字节不同且缺少 post-hoc source snapshot")
    snapshot_manifest_path = snapshot_root / "manifest.json"
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    records = {item["role"]: item for item in snapshot.get("source_files", [])}
    covered_roots = snapshot.get("job_config_audit", {}).get("covered_result_roots", [])
    if (
        snapshot.get("status") != "PASS_WITH_DISCLOSED_EOL_NORMALIZATION"
        or checkpoint_root.name not in covered_roots
    ):
        raise RuntimeError("post-hoc source snapshot 状态非法")
    for role, path in source_paths.items():
        record = records.get(role, {})
        snapshot_path = snapshot_root / record.get("snapshot_path", "missing")
        if (
            record.get("job_sha256") != expected.get(role)
            or not snapshot_path.is_file()
            or file_hash(snapshot_path) != expected.get(role)
            or lf_normalized_hash(path) != record.get("lf_normalized_sha256")
        ):
            raise RuntimeError(f"{role} 不是可证明的 CRLF/LF 等价源码")
    return "lf_normalized_equivalent_via_snapshot"


# ---------- 冻结库存：只从 session0-only bundle 派生，不读取联合索引或测试 session ----------
def build_online_inventory(context: BundleContext) -> OnlineInventory:
    manifest = context.manifest
    subject = manifest.get("subject")
    if (
        type(subject) is not int
        or subject not in KNOWN_SUBJECTS
        or manifest.get("included_session") != 0
        or manifest.get("test_session_content_in_bundle") is not False
        or np.any(context.rows["subject"] != subject)
        or np.any(context.rows["session"] != 0)
    ):
        raise RuntimeError("在线 OOF runner 只允许 Subject 1–9 的 session 0-only bundle")

    all_records = sorted(
        manifest["domains"]["causal"]["segments"],
        key=lambda item: (item["run"], item["segment"]),
    )
    if any(record["session"] != 0 for record in all_records):
        raise RuntimeError("因果信号清单意外包含测试 session")

    # 某些被试的 run 起点恰好被伪迹切出不足或等于 1 秒的短段；该段会被因果
    # warmup 完整排除。它不属于有效评分时间，但必须单独计数，不能静默消失。
    empty_records = [
        record for record in all_records
        if record["formal_start_native"] == record["formal_stop_native"]
    ]
    if any(
        record["formal_start_native"] > record["formal_stop_native"]
        or record["formal_stop_native"] != record["stop_native"]
        or not 0 < record["stop_native"] - record["start_native"] <= NATIVE_SAMPLING_RATE
        for record in all_records
        if record["formal_start_native"] >= record["formal_stop_native"]
    ):
        raise RuntimeError("因果 segment 的正式区间非法，不能解释为完整 warmup 排除")
    records = [
        record for record in all_records
        if record["formal_start_native"] < record["formal_stop_native"]
    ]

    segments = [
        ScoringSegment(
            subject, 0, record["run"], record["segment"],
            record["formal_start_native"], record["formal_stop_native"],
        )
        for record in records
    ]
    windows: list[ExpectedWindow] = []
    for segment in segments:
        starts = range(segment.start_sample, segment.stop_sample - 500 + 1, 125)
        windows.extend(
            ExpectedWindow(*segment.key, index, start, start + 500)
            for index, start in enumerate(starts)
        )

    # 训练 bundle 的 task 行由每个干净 trial 的 5 个任务窗构成；其并集恰好恢复 4 秒 MI 区间。
    task_rows = context.rows[context.rows["is_task"]]
    event_keys = sorted({
        (int(row["run"]), int(row["segment"]), int(row["trial"]))
        for row in task_rows
    })
    events: list[MIEvent] = []
    for run, segment, trial in event_keys:
        rows = task_rows[
            (task_rows["run"] == run)
            & (task_rows["segment"] == segment)
            & (task_rows["trial"] == trial)
        ]
        rows = np.sort(rows, order="start")
        starts = rows["start"].astype(np.int64).tolist()
        labels = set(int(value) for value in rows["stage2_label"])
        if (
            len(rows) != 5
            or starts != list(range(starts[0], starts[0] + 625, 125))
            or np.any(rows["stop"] - rows["start"] != 500)
            or len(labels) != 1
            or np.any(rows["final_label"] != rows["stage2_label"] + 1)
        ):
            raise RuntimeError("session0 task 行不能唯一恢复冻结 MI 事件")
        events.append(MIEvent(
            f"s0_r{run}_t{trial}", subject, 0, run, segment,
            int(rows["start"].min()), int(rows["stop"].max()), labels.pop() + 1,
        ))

    # BundleSignalStore 只读取身份和原生坐标，其余标签字段保持哨兵值，防止真值参与推理。
    signal_rows = np.zeros(len(windows), dtype=context.rows.dtype)
    signal_rows["trial"] = -1
    signal_rows["stage2_label"] = -1
    for index, window in enumerate(windows):
        signal_rows[index]["subject"] = window.subject_id
        signal_rows[index]["session"] = window.session_id
        signal_rows[index]["run"] = window.run_id
        signal_rows[index]["segment"] = window.segment_id
        signal_rows[index]["window"] = window.window_index
        signal_rows[index]["start"] = window.window_start_sample
        signal_rows[index]["stop"] = window.window_stop_sample
    return OnlineInventory(
        segments,
        events,
        windows,
        signal_rows,
        len(empty_records),
        sum(record["stop_native"] - record["start_native"] for record in empty_records),
    )


def verify_inventory_contract(
    context: BundleContext,
    inventory: OnlineInventory,
    contract: dict,
) -> dict:
    """用全 NO_COMMAND 轨迹计算可信库存哈希；缺事件或缺窗口会在正式推理前失败。"""
    subject = context.manifest["subject"]
    artifact_identity = artifact_contract(context.manifest)
    expected_protocol_id = inventory_contract_protocol_id(context.manifest)
    if (
        contract.get("protocol_id") != expected_protocol_id
        or contract.get("subject") != subject
        or contract.get("included_session") != 0
        or contract.get("test_session_access") != "forbidden"
        or contract.get("native_sampling_rate") != NATIVE_SAMPLING_RATE
        or contract.get("window_samples") != 500
        or contract.get("step_samples") != 125
        or contract.get("event_margin_samples") != 125
        or contract.get("source_bundle", {}).get("protocol_id")
        != context.manifest.get("protocol_id")
        or contract.get("source_bundle", {}).get("manifest_sha256") != context.manifest_sha256
        or contract.get("source_bundle", {}).get("index_sha256") != context.manifest["index_sha256"]
    ):
        raise RuntimeError("在线库存合同与 session0-only bundle 不匹配")
    # v2 把伪迹和连续 segment 语义写入库存合同；v1 保持历史文件逐字兼容，
    # 但其 legacy binding 会在下游运行清单中被显式披露。
    if (
        artifact_identity["artifact_policy_binding"] == "explicit_bundle_manifest"
        and any(contract.get(key) != value for key, value in artifact_identity.items())
    ):
        raise RuntimeError("v2 在线库存合同没有继承 bundle 的伪迹与 segment 身份")

    no_command = [
        DecisionRecord(
            *window.key, window.window_index,
            window.window_start_sample, window.window_stop_sample,
        )
        for window in inventory.windows
    ]
    result = evaluate_online_events(
        inventory.segments, inventory.events, inventory.windows, no_command,
        mode=STATELESS_DIAGNOSTIC,
    )
    expected = contract["inventory"]
    actual = {
        "segment_count": result["scoring_segment_count"],
        "segment_inventory_sha256": result["scoring_segment_inventory_sha256"],
        "fully_warmup_excluded_segment_count": inventory.fully_warmup_excluded_segment_count,
        "fully_warmup_excluded_samples": inventory.fully_warmup_excluded_samples,
        "zero_window_segment_count": result["zero_window_segment_count"],
        "zero_window_segment_samples": result["zero_window_segment_samples"],
        "trailing_unwindowed_samples": result["trailing_unwindowed_samples"],
        "window_count": result["expected_window_count"],
        "window_inventory_sha256": result["expected_window_inventory_sha256"],
        "event_count": result["event_count"],
        "event_inventory_sha256": result["event_inventory_sha256"],
        "valid_idle_seconds": result["valid_idle_seconds"],
    }
    per_run_windows = {
        str(run): sum(window.run_id == run for window in inventory.windows)
        for run in FIXED_FOLDS
    }
    per_run_events = {
        str(run): sum(event.run_id == run for event in inventory.events)
        for run in FIXED_FOLDS
    }
    if (
        actual != expected
        or per_run_windows != contract["per_run_window_count"]
        or per_run_events != contract["per_run_event_count"]
        or result["scorable_event_count"] != len(inventory.events)
        or result["unscorable_event_count"] != 0
        or result["miss_rate"] != 1.0
        or result["idle_false_command_count"] != 0
    ):
        raise RuntimeError("在线库存或 NO_COMMAND 手算控制与冻结合同不一致")
    return result


# ---------- 输入物化：每个 fold 只使用另外 5 个 run 拟合出的因果统计量 ----------
def materialize_rows(context: BundleContext, rows: np.ndarray, fold: int) -> np.ndarray:
    subject = context.manifest["subject"]
    if (
        len(rows) == 0
        or np.any(rows["subject"] != subject)
        or np.any(rows["session"] != 0)
        or set(rows["run"].tolist()) != {fold}
    ):
        raise RuntimeError("物化窗口必须是单个 held-out session0 run")
    entry = context.manifest["folds"][fold]
    if entry["fold"] != fold or entry["validation_runs"] != [fold] or fold in entry["train_runs"]:
        raise RuntimeError("fold 训练/验证 run 合同非法")
    statistics = entry["statistics"]["causal"]
    mean = np.asarray(statistics["mean_volts"], dtype=np.float32)[:, None]
    std = np.asarray(statistics["std_volts"], dtype=np.float32)[:, None]
    if mean.shape != (22, 1) or std.shape != mean.shape or np.any(std <= 0):
        raise RuntimeError("fold 因果标准化统计非法")

    features = np.empty((len(rows), 22, 500), dtype=np.float32)
    store = context.stores["causal"]
    for index, row in enumerate(rows):
        signal = store.read_window(row)
        features[index] = np.ascontiguousarray((signal - mean) / std, dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("连续窗口标准化后出现非有限值")
    return features


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    classes: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features), INFERENCE_BATCH_SIZE):
        batch = torch.from_numpy(features[start:start + INFERENCE_BATCH_SIZE]).to(
            device, non_blocking=device.type == "cuda",
        )
        chunks.append(model(batch).cpu().numpy())
    logits = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if logits.shape != (len(features), classes) or not np.isfinite(logits).all():
        raise RuntimeError("checkpoint 连续推理输出结构非法")
    return logits


# ---------- Checkpoint 合同：固定 final epoch 50，不进行 epoch 搜索或早停选择 ----------
def load_epoch50_job(
    context: BundleContext,
    checkpoint_root: Path,
    fold: int,
    seed: int,
    stage: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict]:
    job_name = f"stage{stage}_causal_fold{fold}_seed{seed}"
    job_dir = checkpoint_root / job_name
    config_path = job_dir / "job_config.json"
    completed_path = job_dir / "completed.json"
    checkpoint_path = job_dir / "latest.pt"
    oof_path = job_dir / "oof_predictions.npz"
    for path in (config_path, completed_path, checkpoint_path, oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"{job_name} 缺少产物: {path.name}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    contract = config.get("contract", {})
    classes = 2 if stage == 1 else 4
    subject = context.manifest["subject"]
    expected_job = {
        "subject": subject, "fold": fold, "stage": stage,
        "train_domain": "causal", "seed": seed,
    }
    fold_contract = context.manifest["folds"][fold]
    data_contract = contract.get("data", {})
    train_identity = fold_contract[f"train_stage{stage}"]
    validation_identity = fold_contract[f"validation_stage{stage}"]
    if (
        config.get("contract_sha256") != canonical_hash(contract)
        or contract.get("job") != expected_job
        or contract.get("optimization", {}).get("epochs") != FIXED_EPOCH
        or contract.get("model", {}).get("classes")
        != (["idle", "task"] if stage == 1 else ["left_hand", "right_hand", "feet", "tongue"])
        or contract.get("validation_domains") != ["causal"]
        or data_contract.get("session") != 0
        or data_contract.get("train_runs") != fold_contract["train_runs"]
        or data_contract.get("validation_runs") != fold_contract["validation_runs"]
        or data_contract.get("train_window_count") != train_identity["window_count"]
        or data_contract.get("train_window_sha256") != train_identity["window_sha256"]
        or data_contract.get("validation_window_count") != validation_identity["window_count"]
        or data_contract.get("validation_window_sha256") != validation_identity["window_sha256"]
        or data_contract.get("training_bundle_protocol_id") != context.manifest["protocol_id"]
        or data_contract.get("training_bundle_manifest_sha256") != context.manifest_sha256
        or contract.get("test_session_access") != "forbidden"
    ):
        raise RuntimeError(f"{job_name} 训练合同不是固定 epoch50 因果 OOF 作业")
    if (
        completed.get("status") != "complete"
        or completed.get("completed_epochs") != FIXED_EPOCH
        or completed.get("contract_sha256") != config["contract_sha256"]
        or completed.get("checkpoint_sha256") != file_hash(checkpoint_path)
        or completed.get("artifact_sha256", {}).get("oof_predictions.npz") != file_hash(oof_path)
        or completed.get("artifact_sha256", {}).get("job_config.json") != file_hash(config_path)
    ):
        raise RuntimeError(f"{job_name} 完成标记与 checkpoint/OOF 产物不一致")

    model_contract = contract.get("model", {})
    if (
        model_contract.get("name") != "eegnet"
        or model_contract.get("channels") != 22
        or model_contract.get("samples") != 500
    ):
        raise RuntimeError(f"{job_name} 模型结构合同非法")
    source_match_mode = verify_model_sources(checkpoint_root, contract)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("contract_sha256") != config["contract_sha256"]
        or checkpoint.get("epoch") != FIXED_EPOCH
        or len(checkpoint.get("history", [])) != FIXED_EPOCH
        or [item.get("epoch") for item in checkpoint.get("history", [])]
        != list(range(1, FIXED_EPOCH + 1))
    ):
        raise RuntimeError(f"{job_name} checkpoint 不是完整 epoch50 状态")
    model = build_model("eegnet", classes, 22, 500)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    expected_train_rows = rows_for(context.manifest, context.rows, fold, stage, "train")
    expected_validation_rows = rows_for(
        context.manifest, context.rows, fold, stage, "validation",
    )
    if (
        len(expected_train_rows) != data_contract["train_window_count"]
        or window_identity_hash(expected_train_rows) != data_contract["train_window_sha256"]
    ):
        raise RuntimeError(f"{job_name} 训练窗口不能从冻结 bundle 唯一恢复")

    with np.load(oof_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "epochs", "validation_rows", "validation_labels",
            "validation_window_sha256", "causal_logits",
        }:
            raise RuntimeError(f"{job_name} OOF 产物字段非法")
        epochs = archive["epochs"].copy()
        validation_rows = archive["validation_rows"].copy()
        validation_labels = archive["validation_labels"].copy()
        embedded_validation_hash = str(archive["validation_window_sha256"].item())
        all_logits = archive["causal_logits"].copy()
    epoch_positions = np.flatnonzero(epochs == FIXED_EPOCH)
    label_field = "stage1_label" if stage == 1 else "stage2_label"
    if (
        epoch_positions.tolist() != [FIXED_EPOCH - 1]
        or not np.array_equal(epochs, np.arange(1, FIXED_EPOCH + 1))
        or all_logits.ndim != 3
        or all_logits.shape[0] != FIXED_EPOCH
        or not np.array_equal(validation_rows, expected_validation_rows)
        or window_identity_hash(validation_rows) != validation_identity["window_sha256"]
        or embedded_validation_hash != validation_identity["window_sha256"]
        or validation_labels.shape != (len(validation_rows),)
        or not np.array_equal(
            validation_labels,
            validation_rows[label_field].astype(np.int64),
        )
    ):
        raise RuntimeError(f"{job_name} OOF 不能逐行绑定冻结验证窗口")
    saved_epoch50_logits = np.ascontiguousarray(all_logits[epoch_positions[0]], dtype=np.float32)
    if (
        saved_epoch50_logits.shape != (len(validation_rows), classes)
        or not np.isfinite(saved_epoch50_logits).all()
    ):
        raise RuntimeError(f"{job_name} epoch50 OOF logit 结构非法")
    metadata = {
        "job_name": job_name,
        "stage": stage,
        "fold": fold,
        "seed": seed,
        "checkpoint_epoch": FIXED_EPOCH,
        "epoch_selection_method": "none_fixed_final_epoch",
        "train_runs": data_contract["train_runs"],
        "train_window_count": data_contract["train_window_count"],
        "train_window_sha256": data_contract["train_window_sha256"],
        "validation_runs": data_contract["validation_runs"],
        "validation_window_count": data_contract["validation_window_count"],
        "validation_window_sha256": data_contract["validation_window_sha256"],
        "contract_sha256": config["contract_sha256"],
        "checkpoint_sha256": completed["checkpoint_sha256"],
        "oof_predictions_sha256": completed["artifact_sha256"]["oof_predictions.npz"],
        "model_source_match_mode": source_match_mode,
    }
    return model, validation_rows, saved_epoch50_logits, metadata


# ---------- 决策策略：只改变输出轨迹，不修改模型 logit 或评估器 ----------
def _argmax_predictions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if (
        stage1.shape != (len(windows), 2)
        or stage2.shape != (len(windows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise ValueError("Stage 1/2 logit 必须逐窗完整、有限且维度固定")
    identities = [(*window.key, window.window_index) for window in windows]
    if identities != sorted(identities):
        raise ValueError("决策窗口必须按 subject/session/run/segment/index 排序")
    # np.argmax 平局选择最小索引，因此 Stage 1 平局确定为 IDLE。
    return np.argmax(stage1, axis=1), np.argmax(stage2, axis=1) + 1


def stateless_argmax_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> list[DecisionRecord]:
    stage1, stage2 = _argmax_predictions(windows, stage1_logits, stage2_logits)
    return [
        DecisionRecord(
            *window.key, window.window_index,
            window.window_start_sample, window.window_stop_sample,
            int(stage2[index]) if stage1[index] == 1 else NO_COMMAND,
        )
        for index, window in enumerate(windows)
    ]


def stateful_argmax_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> list[DecisionRecord]:
    stage1, stage2 = _argmax_predictions(windows, stage1_logits, stage2_logits)
    decisions: list[DecisionRecord] = []
    current_key: tuple[int, int, int, int] | None = None
    state = READY
    for index, window in enumerate(windows):
        if window.key != current_key:
            current_key, state = window.key, READY
        before = state
        emitted = NO_COMMAND
        if state == READY and stage1[index] == 1:
            emitted, state = int(stage2[index]), WAIT_IDLE
        elif state == WAIT_IDLE and stage1[index] == 0:
            # 单窗基线用首个 IDLE argmax 窗复位；复位窗本身不允许同时输出新指令。
            state = READY
        decisions.append(DecisionRecord(
            *window.key, window.window_index,
            window.window_start_sample, window.window_stop_sample,
            emitted, before, state,
        ))
    return decisions


# ---------- 结果持久化：原始 logit、两种轨迹、完整指标和来源清单分开保存 ----------
def output_window_rows(windows: Sequence[ExpectedWindow]) -> np.ndarray:
    rows = np.zeros(len(windows), dtype=OUTPUT_WINDOW_DTYPE)
    for index, window in enumerate(windows):
        rows[index] = (
            window.subject_id, window.session_id, window.run_id, window.segment_id,
            window.window_index, window.window_start_sample, window.window_stop_sample,
            window.window_start_sample, window.window_stop_sample,
            window.window_stop_sample / NATIVE_SAMPLING_RATE,
        )
    return rows


def core_metrics(result: dict) -> dict:
    latency = result["correct_detection_latency_seconds"]
    return {
        "correct_event_rate": result["correct_event_rate"],
        "macro_correct_event_rate": result["macro_correct_event_rate"],
        "event_trigger_rate": result["event_trigger_rate"],
        "triggered_class_accuracy": result["triggered_class_accuracy"],
        "miss_rate": result["miss_rate"],
        "idle_false_commands_per_minute": result["idle_false_commands_per_minute"],
        "correct_latency_mean_seconds": latency["mean"],
        "correct_latency_median_seconds": latency["median"],
        "correct_latency_p90_seconds": latency["p90"],
    }


def summarize_seed_metrics(seed_results: dict[int, dict[str, dict]]) -> dict:
    summary: dict[str, dict] = {}
    for mode in (STATELESS_DIAGNOSTIC, STATEFUL_STRICT):
        by_seed = {seed: core_metrics(results[mode]) for seed, results in seed_results.items()}
        aggregate: dict[str, dict] = {}
        for field in next(iter(by_seed.values())):
            values = [metrics[field] for metrics in by_seed.values() if metrics[field] is not None]
            aggregate[field] = {
                "mean": None if not values else float(np.mean(values)),
                "population_std": None if not values else float(np.std(values, ddof=0)),
                "valid_seed_count": len(values),
            }
        summary[mode] = {"per_seed": by_seed, "aggregate": aggregate}
    return summary


def save_seed_result(
    output_root: Path,
    inventory: OnlineInventory,
    contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> tuple[dict[str, dict], dict]:
    stateless = stateless_argmax_decisions(inventory.windows, stage1_logits, stage2_logits)
    stateful = stateful_argmax_decisions(inventory.windows, stage1_logits, stage2_logits)
    metrics = {
        STATELESS_DIAGNOSTIC: evaluate_online_events(
            inventory.segments, inventory.events, inventory.windows, stateless,
            mode=STATELESS_DIAGNOSTIC,
        ),
        STATEFUL_STRICT: evaluate_online_events(
            inventory.segments, inventory.events, inventory.windows, stateful,
            mode=STATEFUL_STRICT,
        ),
    }
    for result in metrics.values():
        frozen = contract["inventory"]
        if (
            result["scoring_segment_inventory_sha256"] != frozen["segment_inventory_sha256"]
            or result["expected_window_inventory_sha256"] != frozen["window_inventory_sha256"]
            or result["event_inventory_sha256"] != frozen["event_inventory_sha256"]
        ):
            raise RuntimeError("模型决策结果与冻结在线库存哈希不一致")

    stateless_path = output_root / f"seed{seed}_stateless_metrics.json"
    stateful_path = output_root / f"seed{seed}_stateful_metrics.json"
    scores_path = output_root / f"seed{seed}_scores_and_decisions.npz"
    atomic_json(stateless_path, metrics[STATELESS_DIAGNOSTIC])
    atomic_json(stateful_path, metrics[STATEFUL_STRICT])
    state_code = {READY: 0, WAIT_IDLE: 1}
    atomic_npz(
        scores_path,
        window_rows=output_window_rows(inventory.windows),
        stage1_logits=np.asarray(stage1_logits, dtype=np.float32),
        stage2_logits=np.asarray(stage2_logits, dtype=np.float32),
        stateless_emitted=np.asarray([item.emitted_class for item in stateless], dtype=np.int8),
        stateful_emitted=np.asarray([item.emitted_class for item in stateful], dtype=np.int8),
        stateful_before=np.asarray(
            [state_code[item.decision_state_before] for item in stateful], dtype=np.uint8,
        ),
        stateful_after=np.asarray(
            [state_code[item.decision_state_after] for item in stateful], dtype=np.uint8,
        ),
        state_code_names=np.asarray([READY, WAIT_IDLE]),
    )
    artifacts = {
        "scores_and_decisions": {
            "file": scores_path.name, "sha256": file_hash(scores_path),
        },
        "stateless_metrics": {
            "file": stateless_path.name, "sha256": file_hash(stateless_path),
        },
        "stateful_metrics": {
            "file": stateful_path.name, "sha256": file_hash(stateful_path),
        },
    }
    return metrics, artifacts


# ---------- 端到端 OOF：每个 validation run 只使用对应 fold 的 Stage 1/2 checkpoint ----------
def run(args: argparse.Namespace) -> dict:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    subject = args.subject
    verbose = getattr(args, "verbose", True)
    defaults = default_subject_paths(subject)
    bundle_manifest = Path(args.bundle_manifest or defaults.bundle_manifest).resolve()
    checkpoint_root = Path(args.checkpoint_root or defaults.checkpoint_root).resolve()
    contract_path = Path(args.inventory_contract or defaults.inventory_contract).resolve()
    output_root = Path(args.output_root or defaults.output_root).resolve()
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 推理，但当前环境没有可用 GPU")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    context = load_bundle(bundle_manifest, verify_hashes=True)
    if context.manifest.get("subject") != subject:
        raise RuntimeError("--subject 与 bundle manifest 的被试编号不一致")
    artifact_identity = artifact_contract(context.manifest)
    output_protocol_version = bundle_contract_version(context.manifest)
    inventory = build_online_inventory(context)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    no_command_control = verify_inventory_contract(context, inventory, contract)

    seeds = tuple(sorted(set(int(seed) for seed in args.seeds)))
    if not seeds or any(seed not in KNOWN_SEEDS for seed in seeds):
        raise ValueError(f"seeds 只能取已训练集合 {KNOWN_SEEDS}")
    full_seed_grid = seeds == KNOWN_SEEDS
    scores = {
        seed: {
            1: np.full((len(inventory.windows), 2), np.nan, dtype=np.float32),
            2: np.full((len(inventory.windows), 4), np.nan, dtype=np.float32),
        }
        for seed in seeds
    }
    checkpoint_records: list[dict] = []
    run_ids = np.asarray([window.run_id for window in inventory.windows], dtype=np.int64)

    for fold in FIXED_FOLDS:
        online_indices = np.flatnonzero(run_ids == fold)
        online_rows = inventory.signal_rows[online_indices]
        online_features = materialize_rows(context, online_rows, fold)
        for seed in seeds:
            for stage in (1, 2):
                model, validation_rows, saved_logits, metadata = load_epoch50_job(
                    context, checkpoint_root, fold, seed, stage, device,
                )
                validation_features = materialize_rows(context, validation_rows, fold)
                reproduced = predict_logits(
                    model, validation_features, device, 2 if stage == 1 else 4,
                )
                difference = np.abs(reproduced - saved_logits)
                if not np.allclose(reproduced, saved_logits, rtol=1e-5, atol=1e-6):
                    raise RuntimeError(f"{metadata['job_name']} 无法复现已保存 epoch50 OOF logit")
                continuous = predict_logits(
                    model, online_features, device, 2 if stage == 1 else 4,
                )
                scores[seed][stage][online_indices] = continuous
                metadata["saved_oof_reproduction_max_abs_error"] = float(difference.max())
                metadata["saved_oof_reproduction_exact"] = bool(np.array_equal(reproduced, saved_logits))
                metadata["continuous_window_count"] = int(len(online_indices))
                checkpoint_records.append(metadata)
                if verbose:
                    print(
                        f"{metadata['job_name']}: saved_oof_max_abs={difference.max():.3g}, "
                        f"continuous_windows={len(online_indices)}",
                        flush=True,
                    )
                del model, validation_features, reproduced, continuous
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    seed_results: dict[int, dict[str, dict]] = {}
    seed_artifacts: dict[str, dict] = {}
    for seed in seeds:
        if not np.isfinite(scores[seed][1]).all() or not np.isfinite(scores[seed][2]).all():
            raise RuntimeError(f"seed {seed} 没有覆盖完整连续窗口母索引")
        metrics, artifacts = save_seed_result(
            output_root, inventory, contract, seed, scores[seed][1], scores[seed][2],
        )
        seed_results[seed] = metrics
        seed_artifacts[str(seed)] = artifacts

    current_git = git_state()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    runtime = {
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": None if device.type != "cuda" else torch.cuda.get_device_name(device),
        "git": current_git,
    }
    protocol_id = (
        f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_oof_"
        f"{output_protocol_version}"
        if full_seed_grid
        else f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_"
        f"seed_subset_{'_'.join(map(str, seeds))}_diagnostic_"
        f"{output_protocol_version}"
    )
    summary = summarize_seed_metrics(seed_results)
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": protocol_id,
        "subject": subject,
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        **artifact_identity,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "device": str(device),
        "seeds": list(seeds),
        "bundle_manifest": display_path(bundle_manifest),
        "bundle_manifest_sha256": context.manifest_sha256,
        "checkpoint_root": display_path(checkpoint_root),
        "inventory_contract": display_path(contract_path),
        "inventory_contract_sha256": file_hash(contract_path),
        "inventory": contract["inventory"],
        "checkpoint_verification": checkpoint_records,
        "seed_artifacts": seed_artifacts,
    })
    log_artifact = {"file": log_path.name, "sha256": file_hash(log_path)}
    manifest = {
        "status": "PASS",
        "claim_status": (
            "SEED_SUBSET_DIAGNOSTIC" if not full_seed_grid
            else "PRECOMMIT_DIAGNOSTIC" if current_git["dirty"] is not False
            else "CLEAN_COMMIT_FORMAL_CANDIDATE"
        ),
        "protocol_id": protocol_id,
        "subject": subject,
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        **artifact_identity,
        "training_checkpoint_rule": {
            "training_budget_epochs": FIXED_EPOCH,
            "checkpoint_epoch": FIXED_EPOCH,
            "checkpoint_rule": "fixed_final_epoch",
            "epoch_selection_method": "none",
            "selection_metric": None,
        },
        "decision_policy": {
            "stage1": "single_window_argmax_tie_to_idle",
            "stage2": "single_window_argmax_class_order",
            "stateful": "emit_once_then_first_idle_window_rearms_next_window",
            "aggregation": "none",
            "confidence_threshold": "none",
        },
        "inputs": {
            "bundle_manifest": display_path(bundle_manifest),
            "bundle_manifest_sha256": context.manifest_sha256,
            "checkpoint_root": display_path(checkpoint_root),
            "inventory_contract": display_path(contract_path),
            "inventory_contract_sha256": file_hash(contract_path),
        },
        "inventory_contract": contract,
        "no_command_control": core_metrics(no_command_control),
        "seeds": list(seeds),
        "checkpoint_records": checkpoint_records,
        "seed_artifacts": seed_artifacts,
        "run_log": log_artifact,
        "summary": summary,
        "source_sha256": {
            "runner": file_hash(Path(__file__)),
            "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
            "oof_training_bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
            "model_factory": file_hash(MODELS_DIR / "model_factory.py"),
            "eegnet": file_hash(MODELS_DIR / "models" / "eegnet.py"),
        },
        "runtime": runtime,
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    if verbose:
        print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=KNOWN_SUBJECTS, default=1)
    # 路径参数为空时由 subject 推导；显式覆盖主要用于预检和迁移验证。
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--inventory-contract", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(KNOWN_SEEDS))
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
