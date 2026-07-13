"""使用固定 epoch 50 EEGNet checkpoint 运行 Subject 1 的因果 OOF 在线基线。"""

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
from oof_training_bundle import BundleContext, load_bundle  # noqa: E402
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
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "data" / "processed"
    / "bnci2014001_s01_oof_train_session0_native250_v1" / "manifest.json"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "checkpoints" / "eegnet_oof_native250_v1"
DEFAULT_INVENTORY_CONTRACT = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_s01_session0_causal_online_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_epoch50_causal_single_window_oof_v1"
)
OUTPUT_WINDOW_DTYPE = np.dtype([
    ("subject_id", "u1"), ("session_id", "u1"), ("run_id", "u1"),
    ("segment_id", "u1"), ("window_index", "<u4"),
    ("window_start_sample", "<i8"), ("window_stop_sample", "<i8"),
    ("decision_time_seconds", "<f8"),
])


@dataclass
class OnlineInventory:
    """同一顺序下的评分对象和信号读取行；logit 必须与 windows 逐行对齐。"""

    segments: list[ScoringSegment]
    events: list[MIEvent]
    windows: list[ExpectedWindow]
    signal_rows: np.ndarray


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
    """优先要求训练字节完全一致；换行不同则由 post-hoc snapshot 证明内容等价。"""
    source_paths = {
        "model_factory": MODELS_DIR / "model_factory.py",
        "eegnet": MODELS_DIR / "models" / "eegnet.py",
    }
    expected = contract["source_sha256"]
    if all(file_hash(path) == expected.get(role) for role, path in source_paths.items()):
        return "exact_training_bytes"

    snapshot_root = checkpoint_root / "posthoc_source_snapshot"
    snapshot_manifest_path = snapshot_root / "manifest.json"
    if not snapshot_manifest_path.is_file():
        raise RuntimeError("模型源码字节不同且缺少 post-hoc source snapshot")
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    records = {item["role"]: item for item in snapshot.get("source_files", [])}
    if snapshot.get("status") != "PASS_WITH_DISCLOSED_EOL_NORMALIZATION":
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
    if (
        manifest.get("subject") != 1
        or manifest.get("included_session") != 0
        or manifest.get("test_session_content_in_bundle") is not False
        or np.any(context.rows["session"] != 0)
    ):
        raise RuntimeError("在线 OOF runner 只允许 Subject 1、session 0-only bundle")

    records = sorted(
        manifest["domains"]["causal"]["segments"],
        key=lambda item: (item["run"], item["segment"]),
    )
    if any(record["session"] != 0 for record in records):
        raise RuntimeError("因果信号清单意外包含测试 session")

    segments = [
        ScoringSegment(
            1, 0, record["run"], record["segment"],
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
            f"s0_r{run}_t{trial}", 1, 0, run, segment,
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
    return OnlineInventory(segments, events, windows, signal_rows)


def verify_inventory_contract(
    context: BundleContext,
    inventory: OnlineInventory,
    contract: dict,
) -> dict:
    """用全 NO_COMMAND 轨迹计算可信库存哈希；缺事件或缺窗口会在正式推理前失败。"""
    if (
        contract.get("protocol_id") != "bnci2014001_s01_session0_causal_online_v1"
        or contract.get("subject") != 1
        or contract.get("included_session") != 0
        or contract.get("test_session_access") != "forbidden"
        or contract.get("source_bundle", {}).get("manifest_sha256") != context.manifest_sha256
        or contract.get("source_bundle", {}).get("index_sha256") != context.manifest["index_sha256"]
    ):
        raise RuntimeError("在线库存合同与 session0-only bundle 不匹配")

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
    if len(rows) == 0 or np.any(rows["session"] != 0) or set(rows["run"].tolist()) != {fold}:
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
    expected_job = {
        "subject": 1, "fold": fold, "stage": stage,
        "train_domain": "causal", "seed": seed,
    }
    if (
        config.get("contract_sha256") != canonical_hash(contract)
        or contract.get("job") != expected_job
        or contract.get("optimization", {}).get("epochs") != FIXED_EPOCH
        or contract.get("model", {}).get("classes")
        != (["idle", "task"] if stage == 1 else ["left_hand", "right_hand", "feet", "tongue"])
        or contract.get("validation_domains") != ["causal"]
        or contract.get("data", {}).get("session") != 0
        or contract.get("data", {}).get("validation_runs") != [fold]
        or contract.get("data", {}).get("training_bundle_manifest_sha256") != context.manifest_sha256
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

    with np.load(oof_path, allow_pickle=False) as archive:
        epochs = archive["epochs"].copy()
        validation_rows = archive["validation_rows"].copy()
        all_logits = archive["causal_logits"].copy()
    epoch_positions = np.flatnonzero(epochs == FIXED_EPOCH)
    if (
        epoch_positions.tolist() != [FIXED_EPOCH - 1]
        or all_logits.ndim != 3
        or all_logits.shape[0] != FIXED_EPOCH
        or np.any(validation_rows["session"] != 0)
        or set(validation_rows["run"].tolist()) != {fold}
    ):
        raise RuntimeError(f"{job_name} 公开 OOF epoch 轴或验证 run 非法")
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
    output_root = Path(args.output_root).resolve()
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

    context = load_bundle(Path(args.bundle_manifest), verify_hashes=True)
    inventory = build_online_inventory(context)
    contract_path = Path(args.inventory_contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    no_command_control = verify_inventory_contract(context, inventory, contract)

    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds or any(seed not in KNOWN_SEEDS for seed in seeds):
        raise ValueError(f"seeds 只能取已训练集合 {KNOWN_SEEDS}")
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
                    context, Path(args.checkpoint_root), fold, seed, stage, device,
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
    runtime = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": None if device.type != "cuda" else torch.cuda.get_device_name(device),
        "git": current_git,
    }
    manifest = {
        "status": "PASS",
        "claim_status": (
            "PRECOMMIT_DIAGNOSTIC" if current_git["dirty"] is not False
            else "CLEAN_COMMIT_FORMAL_CANDIDATE"
        ),
        "protocol_id": "bnci2014001_s01_epoch50_causal_single_window_oof_v1",
        "subject": 1,
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
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
            "bundle_manifest": display_path(Path(args.bundle_manifest)),
            "bundle_manifest_sha256": context.manifest_sha256,
            "checkpoint_root": display_path(Path(args.checkpoint_root)),
            "inventory_contract": display_path(contract_path),
            "inventory_contract_sha256": file_hash(contract_path),
        },
        "inventory_contract": contract,
        "no_command_control": core_metrics(no_command_control),
        "seeds": list(seeds),
        "checkpoint_records": checkpoint_records,
        "seed_artifacts": seed_artifacts,
        "summary": summarize_seed_metrics(seed_results),
        "source_sha256": {
            "runner": file_hash(Path(__file__)),
            "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
            "model_factory": file_hash(MODELS_DIR / "model_factory.py"),
            "eegnet": file_hash(MODELS_DIR / "models" / "eegnet.py"),
        },
        "runtime": runtime,
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--inventory-contract", type=Path, default=DEFAULT_INVENTORY_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(KNOWN_SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
