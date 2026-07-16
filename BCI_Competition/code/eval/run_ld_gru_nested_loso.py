"""在冻结 session0 OOF logits 上训练并评估 LD-GRU-v1 嵌套被试 LOSO。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Sequence

# 必须在 torch 初始化 CUDA 之前设置，保证 GRU 训练可请求确定性算法。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from commit_reset_diagnostics import diagnose_commit_reset
from hard_vote_policy import stateful_hard_vote_decisions
from ld_gru_policy import (
    ABLATIONS,
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    IDLE_RESET,
    LEARNED_GRU_COMMIT,
    READY,
    TASK_CANDIDATE,
    TOKEN_MODES,
    WAIT_IDLE,
    CandidateInventory,
    CandidateSequence,
    TokenNormalizer,
    build_candidate_inventory,
    build_flow_inputs,
    fit_token_normalizer,
    ld_gru_decisions,
    tensorize_candidates,
)
from ld_gru_training import (
    TrainingHyperparameters,
    atomic_json,
    canonical_hash,
    file_hash,
    load_trained_model,
    tensor_set_hash,
    train_final_job,
    train_inner_pair_job,
)
from logit_candidate_strategies import logit_candidate_decisions
from online_truth_inventory import load_truth_inventory
from protocol_metrics import (
    STATEFUL_CANDIDATE,
    STATEFUL_STRICT,
    evaluate_online_events,
    hierarchical_5class_predictions,
)
from run_candidate_logit_matrix import _check_metric_inventory
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    _build_online_signal_inventory,
    atomic_npz,
    build_online_inventory,
    core_metrics,
    default_subject_paths,
    display_path,
    git_state,
    output_window_rows,
    stateful_argmax_decisions,
    verify_inventory_contract,
)
from run_hard_vote_matrix import (
    EXPECTED_INPUT_PROTOCOL,
    _atomic_csv,
    _load_seed_logits,
    _portable_path_text,
    _read_json,
    _safe_artifact,
    _seed_score_path,
    verify_input_root,
)
from run_oracle_ceiling_analysis import DEFAULT_ANCHOR_CONFIG, load_anchor
from oof_training_bundle import artifact_contract, load_bundle


EXPECTED_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_ld_gru_nested_loso_v1"
MASK_STAGE1_PROTOCOL = (
    "bnci2014001_s01_s09_epoch50_causal_ld_gru_mask_stage1_nested_loso_v1"
)
# 每个正式协议都绑定完整配置内容；新增消融不能借同一 protocol_id 偷换参数。
EXPECTED_CONFIG_IDENTITIES = {
    EXPECTED_PROTOCOL: {
        "canonical_sha256": "c343946eb6163f273f4fed01f667d3d7d06bb88e879e8c55f634329175b02f44",
        "token_mode": "full",
    },
    MASK_STAGE1_PROTOCOL: {
        "canonical_sha256": "3202e8e5896b3706ca248b9907e91c2cc04f02bb55f2b2d12fb75b7fc8210e0e",
        "token_mode": "mask_stage1",
    },
}
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_ld_gru_nested_loso_v1"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_ld_gru_v1.json"
)
MASK_STAGE1_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_ld_gru_mask_stage1_v1.json"
)
THRESHOLD_FIELDS = (
    "correct_event_rate",
    "idle_false_commands_per_minute",
    "correct_latency_p90_seconds",
    "triggered_class_accuracy",
)
SUMMARY_FIELDS = (
    "correct_event_rate",
    "macro_correct_event_rate",
    "event_trigger_rate",
    "triggered_class_accuracy",
    "miss_rate",
    "idle_false_commands_per_minute",
    "correct_latency_mean_seconds",
    "correct_latency_median_seconds",
    "correct_latency_p90_seconds",
    "too_early_command_count",
    "additional_event_command_count",
    "post_mi_spillover_per_valid_idle_minute",
)
STATE_CODE = {READY: 0, TASK_CANDIDATE: 1, WAIT_IDLE: 2}
REASON_CODE = {
    None: 0,
    CANDIDATE_OPEN: 1,
    CANDIDATE_ABORT_STAGE1: 2,
    CANDIDATE_TIMEOUT: 3,
    LEARNED_GRU_COMMIT: 4,
    IDLE_RESET: 5,
}


@dataclass
class SubjectData:
    subject: int
    inventory: object
    inventory_contract: dict
    stage1_logits: dict[int, np.ndarray]
    stage2_logits: dict[int, np.ndarray]
    score_artifacts: dict[int, dict]
    score_paths: dict[int, Path]
    candidate_inventories: dict[int, CandidateInventory]
    input_child_path: Path
    input_child_sha256: str
    bundle_manifest: Path
    bundle_sha256: str
    truth_manifest: Path
    truth_manifest_sha256: str
    truth_event_path: Path
    truth_event_sha256: str
    inventory_contract_path: Path
    inventory_contract_sha256: str
    artifact_identity: dict


@dataclass
class PreparedSplit:
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    normalizer: TokenNormalizer
    train_dataset: object
    validation_datasets: dict[int, object]


@dataclass(frozen=True)
class StateParameters:
    stage1_alpha: float
    task_on_probability: float
    task_hold_probability: float
    drop_abort: float
    idle_reset_probability: float
    max_after_open_windows: int


def state_parameters_from_config(config: dict) -> StateParameters:
    value = config["candidate_state"]
    result = StateParameters(
        stage1_alpha=float(value["stage1_alpha"]),
        task_on_probability=float(value["task_on_probability"]),
        task_hold_probability=float(value["task_hold_probability"]),
        drop_abort=float(value["drop_abort_delta"]),
        idle_reset_probability=float(value["idle_reset_probability"]),
        max_after_open_windows=int(value["max_after_open_windows"]),
    )
    if result != StateParameters(0.5, 0.5, 0.3, 0.2, 0.2, 8):
        raise RuntimeError("LD-GRU-v1 候选状态参数发生漂移")
    return result


# ---------- 配置与运行身份：参数改变必须升级或使用新的输出目录 ----------
def load_config(path: Path) -> dict:
    config = _read_json(path)
    training = config.get("training", {})
    expected_grid = [round(value / 20, 2) for value in range(1, 20)]
    protocol_id = config.get("protocol_id")
    identity = EXPECTED_CONFIG_IDENTITIES.get(protocol_id)
    token_mode = config.get("token_mode", "full")
    if (
        identity is None
        or canonical_hash(config) != identity["canonical_sha256"]
        or token_mode != identity["token_mode"]
        or token_mode not in TOKEN_MODES
        or config.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or config.get("included_session") != 0
        or config.get("test_session_access") != "forbidden"
        or tuple(config.get("subjects", [])) != KNOWN_SUBJECTS
        or tuple(config.get("base_seeds", [])) != KNOWN_SEEDS
        or tuple(config.get("ablations", [])) != ABLATIONS
        or config.get("decision_seed_offset") != 1000
        or config.get("model", {}).get("total_parameter_count") != 573
        or config.get("candidate_state", {}).get("max_after_open_windows") != 8
        or config.get("candidate_state", {}).get("opening_window_can_commit") is not False
        or config.get("threshold_selection", {}).get("grid") != expected_grid
        or training.get("batch_size") != 64
        or training.get("max_epochs") != 200
        or training.get("early_stopping_patience") != 20
    ):
        raise RuntimeError("LD-GRU-v1 配置与冻结协议不一致")
    return config


def token_mode_identity(token_mode: str) -> dict[str, str]:
    """旧 full 协议保持原合同形状；新屏蔽协议显式写入差异字段。"""
    if token_mode not in TOKEN_MODES:
        raise ValueError(f"token_mode 必须取 {TOKEN_MODES}")
    return {} if token_mode == "full" else {"token_mode": token_mode}


def hyperparameters_from_config(config: dict) -> TrainingHyperparameters:
    value = config["training"]
    result = TrainingHyperparameters(
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        batch_size=int(value["batch_size"]),
        max_epochs=int(value["max_epochs"]),
        gradient_clip_norm=float(value["gradient_clip_norm"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
        early_stopping_min_delta=float(value["early_stopping_min_delta"]),
    )
    result.validate()
    return result


def runtime_environment(training_device: torch.device, inference_device: torch.device) -> dict:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "environment_name": Path(sys.prefix).name,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "training_device": str(training_device),
        "inference_device": str(inference_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(training_device)
            if training_device.type == "cuda" else None
        ),
        "hostname": platform.node(),
    }


def source_hashes() -> dict[str, str]:
    names = {
        "runner": Path(__file__),
        "policy": EVAL_DIR / "ld_gru_policy.py",
        "trainer": EVAL_DIR / "ld_gru_training.py",
        "protocol_metrics": EVAL_DIR / "protocol_metrics.py",
        "candidate_state_policy": EVAL_DIR / "candidate_state_policy.py",
        "hard_vote_policy": EVAL_DIR / "hard_vote_policy.py",
        "logit_candidate_strategies": EVAL_DIR / "logit_candidate_strategies.py",
        "commit_reset_diagnostics": EVAL_DIR / "commit_reset_diagnostics.py",
        "online_truth_inventory": EVAL_DIR / "online_truth_inventory.py",
        "single_subject_input_runner": EVAL_DIR / "run_epoch50_online_oof.py",
        "multi_subject_input_runner": EVAL_DIR / "run_epoch50_online_oof_all_subjects.py",
        "input_contract_reader": EVAL_DIR / "run_hard_vote_matrix.py",
        "candidate_metric_contract": EVAL_DIR / "run_candidate_logit_matrix.py",
        "anchor_loader": EVAL_DIR / "run_oracle_ceiling_analysis.py",
        "anchor_contract_reader": EVAL_DIR / "run_commit_reset_matrix.py",
        "bundle_reader": TRAIN_DIR / "oof_training_bundle.py",
    }
    return {name: file_hash(path) for name, path in names.items()}


# ---------- 输入装载：先验证匿名窗口与冻结 logits，再打开 session0 独立真值侧车 ----------
def load_subject_data(
    input_root: Path,
    input_master: dict,
    input_children: dict[int, dict],
    seeds: tuple[int, ...],
    state_parameters: StateParameters,
) -> tuple[dict[int, SubjectData], dict]:
    subjects: dict[int, SubjectData] = {}
    common_artifact_identity: dict | None = None
    for subject in KNOWN_SUBJECTS:
        paths = default_subject_paths(subject)
        context = load_bundle(paths.bundle_manifest)
        identity = artifact_contract(context.manifest)
        if common_artifact_identity is None:
            common_artifact_identity = identity
        elif identity != common_artifact_identity:
            raise RuntimeError("九被试训练 bundle 的伪迹合同不一致")
        signal_inventory = _build_online_signal_inventory(context)
        expected_rows = output_window_rows(signal_inventory.windows)
        child_path = _safe_artifact(
            input_root, input_master["children"][str(subject)]["manifest"],
        )
        child = input_children[subject]
        if (
            _portable_path_text(child.get("inputs", {}).get("bundle_manifest", ""))
            != _portable_path_text(display_path(paths.bundle_manifest))
            or child.get("inputs", {}).get("bundle_manifest_sha256") != context.manifest_sha256
        ):
            raise RuntimeError(f"Subject {subject} 冻结 logits 与当前 bundle 不匹配")

        stage1: dict[int, np.ndarray] = {}
        stage2: dict[int, np.ndarray] = {}
        score_artifacts: dict[int, dict] = {}
        score_paths: dict[int, Path] = {}
        for seed in seeds:
            stage1[seed], stage2[seed], score_artifacts[seed] = _load_seed_logits(
                child_path.parent, child, seed, expected_rows,
            )
            # 不依赖 display_path 是项目相对还是绝对；始终以实际 input-root 解析。
            score_paths[seed] = _seed_score_path(child_path.parent, child, seed)
            if file_hash(score_paths[seed]) != score_artifacts[seed]["sha256"]:
                raise RuntimeError(f"Subject {subject} seed {seed} 绝对 score 路径哈希不一致")

        truth = load_truth_inventory(paths.truth_manifest, context)
        inventory = build_online_inventory(context, truth)
        if (
            inventory.segments != signal_inventory.segments
            or inventory.windows != signal_inventory.windows
            or not np.array_equal(inventory.signal_rows, signal_inventory.signal_rows)
        ):
            raise RuntimeError("加载真值改变了匿名推理窗口库存")
        inventory_contract = _read_json(paths.inventory_contract)
        verify_inventory_contract(context, inventory, inventory_contract)
        truth_payload = _read_json(paths.truth_manifest)
        truth_event_path = paths.truth_manifest.parent / truth_payload["event_file"]
        candidates = {
            seed: build_candidate_inventory(
                inventory.windows,
                inventory.events,
                build_flow_inputs(
                    inventory.windows,
                    stage1[seed],
                    stage2[seed],
                    stage1_alpha=state_parameters.stage1_alpha,
                ),
                task_on_probability=state_parameters.task_on_probability,
                task_hold_probability=state_parameters.task_hold_probability,
                drop_abort=state_parameters.drop_abort,
                max_after_open_windows=state_parameters.max_after_open_windows,
            )
            for seed in seeds
        }
        subjects[subject] = SubjectData(
            subject,
            inventory,
            inventory_contract,
            stage1,
            stage2,
            score_artifacts,
            score_paths,
            candidates,
            child_path,
            file_hash(child_path),
            paths.bundle_manifest,
            context.manifest_sha256,
            paths.truth_manifest,
            truth.manifest_sha256,
            truth_event_path,
            truth.event_file_sha256,
            paths.inventory_contract,
            file_hash(paths.inventory_contract),
            identity,
        )
        print(
            f"Subject {subject}: windows={len(inventory.windows)}, events={len(inventory.events)}, "
            + ", ".join(
                f"seed{seed} candidates={candidates[seed].open_count}"
                for seed in seeds
            ),
            flush=True,
        )
    if common_artifact_identity is None:
        raise RuntimeError("没有加载任何被试")
    return subjects, common_artifact_identity


def candidate_summary(subjects: dict[int, SubjectData], seeds: tuple[int, ...]) -> dict:
    result: dict[str, dict] = {}
    for subject, data in subjects.items():
        result[str(subject)] = {}
        for seed in seeds:
            inventory = data.candidate_inventories[seed]
            trainable = inventory.trainable_candidates
            result[str(subject)][str(seed)] = {
                "open_count": inventory.open_count,
                "abort_count": inventory.abort_count,
                "timeout_count": inventory.timeout_count,
                "segment_end_unresolved_count": inventory.unresolved_count,
                "single_token_zero_loss_count": inventory.open_count - len(trainable),
                "trainable_count": len(trainable),
                "positive_trainable_count": sum(item.positive for item in trainable),
                "negative_trainable_count": sum(not item.positive for item in trainable),
            }
    return result


# ---------- Split 物化：训练候选身份与标准化都只来自显式 train_subjects ----------
def _candidate_list(
    subjects: dict[int, SubjectData],
    seed: int,
    selected: Sequence[int],
    *,
    trainable_only: bool,
) -> tuple[CandidateSequence, ...]:
    expected = set(selected)
    rows = tuple(
        candidate
        for subject in selected
        for candidate in (
            subjects[subject].candidate_inventories[seed].trainable_candidates
            if trainable_only
            else subjects[subject].candidate_inventories[seed].candidates
        )
    )
    if {item.subject_id for item in rows} != expected:
        raise RuntimeError("候选集合的被试身份与 split 声明不一致")
    return rows


def prepare_split(
    subjects: dict[int, SubjectData],
    seed: int,
    train_subjects: tuple[int, ...],
    validation_subjects: tuple[int, ...],
) -> PreparedSplit:
    if set(train_subjects) & set(validation_subjects):
        raise ValueError("训练与验证被试不得重叠")
    normalizer_candidates = _candidate_list(
        subjects, seed, train_subjects, trainable_only=False,
    )
    train_candidates = _candidate_list(
        subjects, seed, train_subjects, trainable_only=True,
    )
    normalizer = fit_token_normalizer(normalizer_candidates)
    train_dataset = tensorize_candidates(train_candidates, normalizer)
    validation_datasets = {
        subject: tensorize_candidates(
            subjects[subject].candidate_inventories[seed].trainable_candidates,
            normalizer,
        )
        for subject in validation_subjects
    }
    return PreparedSplit(
        train_subjects, validation_subjects, normalizer, train_dataset, validation_datasets,
    )


def split_contract(
    split: PreparedSplit,
    *,
    protocol_id: str,
    token_mode: str,
    kind: str,
    base_seed: int,
    ablation: str,
    decision_seed: int,
    config_sha256: str,
    input_master_sha256: str,
    hyperparameters: TrainingHyperparameters,
    extra: dict,
) -> dict:
    return {
        "protocol_id": protocol_id,
        "job_kind": kind,
        "included_session": 0,
        "test_session_access": "forbidden",
        "base_seed": base_seed,
        "decision_seed": decision_seed,
        "ablation": ablation,
        "train_subjects": list(split.train_subjects),
        "validation_subjects": list(split.validation_subjects),
        "outer_test_subjects_used": [],
        "train_tensor_sha256": tensor_set_hash(split.train_dataset),
        "validation_tensor_sha256": {
            str(subject): tensor_set_hash(dataset)
            for subject, dataset in split.validation_datasets.items()
        },
        "token_normalizer": {
            "mean": split.normalizer.mean.tolist(),
            "std": split.normalizer.std.tolist(),
            "source_subjects": list(split.train_subjects),
        },
        "hyperparameters": asdict(hyperparameters),
        "config_sha256": config_sha256,
        "input_master_sha256": input_master_sha256,
        **token_mode_identity(token_mode),
        **extra,
    }


# ---------- 正式指标包装：候选策略统一运行同一个 evaluator 与复位/FAR 诊断 ----------
def evaluate_candidate_policy(data: SubjectData, decisions: Sequence) -> tuple[dict, dict]:
    result = evaluate_online_events(
        data.inventory.segments,
        data.inventory.events,
        data.inventory.windows,
        decisions,
        mode=STATEFUL_CANDIDATE,
    )
    _check_metric_inventory(result, data.inventory_contract)
    diagnostics = diagnose_commit_reset(
        data.inventory.segments,
        data.inventory.events,
        data.inventory.windows,
        decisions,
        result,
    )
    return result, diagnostics


def compact_metrics(result: dict, diagnostics: dict | None = None) -> dict:
    row = {
        **core_metrics(result),
        "too_early_command_count": result["too_early_command_count"],
        "additional_event_command_count": result["additional_event_command_count"],
        "emitted_command_count": result["emitted_command_count"],
        "post_mi_spillover_per_valid_idle_minute": None,
    }
    if diagnostics is not None:
        row["post_mi_spillover_per_valid_idle_minute"] = diagnostics[
            "idle_false_attribution"
        ]["post_mi_spillover_per_valid_idle_minute"]
    return row


def dual_ewma_baseline(
    data: SubjectData, seed: int, anchor,
) -> tuple[dict, dict]:
    replay = logit_candidate_decisions(
        data.inventory.windows,
        data.stage1_logits[seed],
        data.stage2_logits[seed],
        anchor.logit_config,
        idle_reset_consecutive_windows=anchor.idle_reset_consecutive_windows,
    )
    return evaluate_candidate_policy(data, replay.policy.decisions)


def reference_metrics(data: SubjectData, seed: int, anchor) -> dict:
    single_decisions = stateful_argmax_decisions(
        data.inventory.windows, data.stage1_logits[seed], data.stage2_logits[seed],
    )
    single = evaluate_online_events(
        data.inventory.segments, data.inventory.events, data.inventory.windows,
        single_decisions, mode=STATEFUL_STRICT,
    )
    labels = hierarchical_5class_predictions(
        data.stage1_logits[seed], data.stage2_logits[seed],
    )
    vote_decisions = stateful_hard_vote_decisions(
        data.inventory.windows, labels, window_count=5, vote_threshold=3,
    )
    vote = evaluate_online_events(
        data.inventory.segments, data.inventory.events, data.inventory.windows,
        vote_decisions, mode=STATEFUL_STRICT,
    )
    dual, dual_diagnostics = dual_ewma_baseline(data, seed, anchor)
    return {
        "single_window_stateful": compact_metrics(single),
        "n5_k3_hard_vote": compact_metrics(vote),
        "dual_ewma_drop_abort_c055_r020_l1": compact_metrics(dual, dual_diagnostics),
    }


# ---------- 内层阈值选择：完整状态回放后做等权被试 FAR 约束，不看 outer subject ----------
def _threshold_metrics(result: dict) -> dict:
    compact = core_metrics(result)
    return {field: compact[field] for field in THRESHOLD_FIELDS}


def _mean(values: Sequence[float | None], *, none_value: float | None = None) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return none_value if not finite else float(np.mean(finite))


def select_outer_hyperparameters(
    output_root: Path,
    subjects: dict[int, SubjectData],
    pair_jobs: dict[tuple[int, int], dict[str, object]],
    *,
    outer_subject: int,
    base_seed: int,
    ablation: str,
    token_mode: str,
    thresholds: tuple[float, ...],
    anchor,
    state_parameters: StateParameters,
    inference_device: torch.device,
    selection_contract: dict,
) -> tuple[dict, Path]:
    selection_dir = output_root / "selections" / f"seed{base_seed}" / ablation
    selection_path = selection_dir / f"outer_s{outer_subject:02d}.json"
    contract_sha256 = canonical_hash(selection_contract)
    if selection_path.exists():
        payload = _read_json(selection_path)
        if payload.get("selection_contract_sha256") != contract_sha256:
            raise RuntimeError("既有阈值选择产物与当前 outer 合同不一致")
        return payload, selection_path

    validation_subjects = tuple(subject for subject in KNOWN_SUBJECTS if subject != outer_subject)
    per_subject: dict[str, dict] = {}
    best_epochs: list[int] = []
    for validation_subject in validation_subjects:
        pair = tuple(sorted((outer_subject, validation_subject)))
        job = pair_jobs[pair]
        completed = job["completed"][ablation]
        best_record = completed["best_checkpoints"][str(validation_subject)]
        best_epochs.append(int(best_record["best_epoch"]))
        checkpoint = Path(job["job_dirs"][ablation]) / best_record["file"]
        model = load_trained_model(
            checkpoint, ablation, inference_device, token_mode=token_mode,
        )
        data = subjects[validation_subject]
        threshold_results: dict[str, dict] = {}
        for threshold in thresholds:
            replay = ld_gru_decisions(
                data.inventory.windows,
                data.stage1_logits[base_seed],
                data.stage2_logits[base_seed],
                model,
                job["split"].normalizer,
                threshold,
                inference_device,
                stage1_alpha=state_parameters.stage1_alpha,
                task_on_probability=state_parameters.task_on_probability,
                task_hold_probability=state_parameters.task_hold_probability,
                idle_reset_probability=state_parameters.idle_reset_probability,
                drop_abort=state_parameters.drop_abort,
                max_after_open_windows=state_parameters.max_after_open_windows,
            )
            evaluated, _ = evaluate_candidate_policy(data, replay.decisions)
            threshold_results[f"{threshold:.2f}"] = _threshold_metrics(evaluated)
        baseline, _ = dual_ewma_baseline(data, base_seed, anchor)
        per_subject[str(validation_subject)] = {
            "pair": list(pair),
            "train_subjects": list(job["split"].train_subjects),
            "best_epoch": best_record["best_epoch"],
            "best_validation_loss": best_record["best_validation_loss"],
            "baseline_far": baseline["idle_false_commands_per_minute"],
            "thresholds": threshold_results,
        }

    baseline_far = float(np.mean([
        per_subject[str(subject)]["baseline_far"] for subject in validation_subjects
    ]))
    aggregate: dict[str, dict] = {}
    feasible: list[float] = []
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        aggregate[key] = {
            field: _mean([
                per_subject[str(subject)]["thresholds"][key][field]
                for subject in validation_subjects
            ])
            for field in THRESHOLD_FIELDS
        }
        aggregate[key]["far_feasible"] = (
            aggregate[key]["idle_false_commands_per_minute"] <= baseline_far + 1e-12
        )
        if aggregate[key]["far_feasible"]:
            feasible.append(threshold)
    if not feasible:
        raise RuntimeError("阈值网格中没有满足内层 anchor FAR 的工作点")

    def ranking(threshold: float) -> tuple[float, float, float, float]:
        row = aggregate[f"{threshold:.2f}"]
        p90 = row["correct_latency_p90_seconds"]
        return (
            float(row["correct_event_rate"]),
            -float(row["idle_false_commands_per_minute"]),
            -math.inf if p90 is None else -float(p90),
            threshold,
        )

    selected_threshold = max(feasible, key=ranking)
    median_epoch = float(np.median(np.asarray(best_epochs, dtype=np.float64)))
    selected_epoch = int(math.floor(median_epoch + 0.5))
    payload = {
        "status": "PASS",
        "selection_contract_sha256": contract_sha256,
        "outer_subject": outer_subject,
        "outer_subject_access": "forbidden_and_not_used",
        "base_seed": base_seed,
        "ablation": ablation,
        **token_mode_identity(token_mode),
        "validation_subjects": list(validation_subjects),
        "baseline_equal_subject_mean_far": baseline_far,
        "best_epochs": best_epochs,
        "median_best_epoch": median_epoch,
        "selected_final_epoch_round_half_up": selected_epoch,
        "selected_stop_threshold": selected_threshold,
        "threshold_aggregate": aggregate,
        "per_validation_subject": per_subject,
    }
    atomic_json(selection_path, payload)
    return payload, selection_path


# ---------- 外层最终评估与逐窗审计轨迹 ----------
def save_final_result(
    result_root: Path,
    data: SubjectData,
    *,
    base_seed: int,
    ablation: str,
    token_mode: str,
    model,
    normalizer: TokenNormalizer,
    threshold: float,
    inference_device: torch.device,
    anchor,
    state_parameters: StateParameters,
    result_contract: dict,
) -> tuple[dict, Path]:
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "result_manifest.json"
    contract_sha256 = canonical_hash(result_contract)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("result_contract_sha256") != contract_sha256:
            raise RuntimeError("既有 outer 结果与当前合同不一致")
        for artifact in manifest["artifacts"].values():
            path = result_root / artifact["file"]
            if file_hash(path) != artifact["sha256"]:
                raise RuntimeError("outer 结果产物哈希不一致")
        return manifest, manifest_path

    replay = ld_gru_decisions(
        data.inventory.windows,
        data.stage1_logits[base_seed],
        data.stage2_logits[base_seed],
        model,
        normalizer,
        threshold,
        inference_device,
        stage1_alpha=state_parameters.stage1_alpha,
        task_on_probability=state_parameters.task_on_probability,
        task_hold_probability=state_parameters.task_hold_probability,
        idle_reset_probability=state_parameters.idle_reset_probability,
        drop_abort=state_parameters.drop_abort,
        max_after_open_windows=state_parameters.max_after_open_windows,
    )
    learned, diagnostics = evaluate_candidate_policy(data, replay.decisions)
    policies = reference_metrics(data, base_seed, anchor)
    policy_name = (
        f"ld_gru_{ablation}"
        if token_mode == "full"
        else f"ld_gru_{token_mode}_{ablation}"
    )
    policies[policy_name] = compact_metrics(learned, diagnostics)
    metrics_path = result_root / "metrics.json"
    atomic_json(metrics_path, {
        "outer_subject": data.subject,
        "base_seed": base_seed,
        "ablation": ablation,
        **token_mode_identity(token_mode),
        "stop_threshold": threshold,
        "policies": policies,
        "learned_full_evaluation": learned,
        "learned_commit_reset_diagnostics": diagnostics,
    })

    trace_path = result_root / "trajectory.npz"
    decisions = replay.decisions
    trace = replay.trace
    atomic_npz(
        trace_path,
        window_rows=output_window_rows(data.inventory.windows),
        raw_token=np.stack([item.raw_token for item in trace]),
        normalized_token=np.stack([item.normalized_token for item in trace]),
        hidden_state=np.stack([item.hidden for item in trace]),
        stop_logit=np.asarray([item.stop_logit for item in trace], dtype=np.float32),
        stop_score=np.asarray([item.stop_score for item in trace], dtype=np.float32),
        centered_stage2_logits=np.stack([item.centered_stage2 for item in trace]),
        class_correction=np.stack([item.class_correction for item in trace]),
        final_class_logits=np.stack([item.final_class_logits for item in trace]),
        gru_consumed=np.asarray([item.gru_consumed for item in trace], dtype=np.bool_),
        candidate_age=np.asarray([item.candidate_age for item in trace], dtype=np.int8),
        emitted_class=np.asarray([item.emitted_class for item in decisions], dtype=np.int8),
        state_before=np.asarray([STATE_CODE[item.decision_state_before] for item in decisions], dtype=np.uint8),
        state_after=np.asarray([STATE_CODE[item.decision_state_after] for item in decisions], dtype=np.uint8),
        transition_reason=np.asarray([REASON_CODE[item.transition_reason] for item in decisions], dtype=np.uint8),
        state_code_names=np.asarray([READY, TASK_CANDIDATE, WAIT_IDLE]),
        reason_code_names=np.asarray([
            "none", CANDIDATE_OPEN, CANDIDATE_ABORT_STAGE1, CANDIDATE_TIMEOUT,
            LEARNED_GRU_COMMIT, IDLE_RESET,
        ]),
    )
    artifacts = {
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectory": {"file": trace_path.name, "sha256": file_hash(trace_path)},
    }
    manifest = {
        "status": "PASS",
        "result_contract_sha256": contract_sha256,
        "outer_subject": data.subject,
        "base_seed": base_seed,
        "ablation": ablation,
        **token_mode_identity(token_mode),
        "stop_threshold": threshold,
        "policies": policies,
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    return manifest, manifest_path


# ---------- 汇总：基线按 subject/seed 去重，学习策略保留两个消融 ----------
def summarize_results(records: list[dict]) -> tuple[list[dict], dict]:
    rows_by_key: dict[tuple[int, int, str], dict] = {}
    for record in records:
        subject = record["outer_subject"]
        seed = record["base_seed"]
        for policy, metrics in record["policies"].items():
            key = (subject, seed, policy)
            row = {"subject": subject, "seed": seed, "policy": policy}
            row.update({field: metrics.get(field) for field in SUMMARY_FIELDS})
            if key in rows_by_key and rows_by_key[key] != row:
                raise RuntimeError("两个消融中的共享基线结果不一致")
            rows_by_key[key] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    policies = sorted({row["policy"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    summary: dict[str, dict] = {}
    for policy in policies:
        per_seed: dict[str, dict] = {}
        for seed in seeds:
            selected = [row for row in rows if row["policy"] == policy and row["seed"] == seed]
            per_seed[str(seed)] = {
                field: _mean([row[field] for row in selected])
                for field in SUMMARY_FIELDS
            }
        aggregate = {}
        for field in SUMMARY_FIELDS:
            values = [per_seed[str(seed)][field] for seed in seeds]
            finite = [float(value) for value in values if value is not None]
            aggregate[field] = {
                "mean": None if not finite else float(np.mean(finite)),
                "population_std": None if not finite else float(np.std(finite, ddof=0)),
                "valid_seed_count": len(finite),
            }
        summary[policy] = {"per_seed_equal_subject_macro": per_seed, "aggregate": aggregate}
    return rows, summary


def write_summary_csvs(output_root: Path, rows: list[dict], summary: dict) -> dict:
    row_path = output_root / "held_out_results.csv"
    _atomic_csv(row_path, ["subject", "seed", "policy", *SUMMARY_FIELDS], rows)
    aggregate_rows = []
    for policy, payload in summary.items():
        row = {"policy": policy}
        for field in SUMMARY_FIELDS:
            row[f"{field}_mean"] = payload["aggregate"][field]["mean"]
            row[f"{field}_population_std"] = payload["aggregate"][field]["population_std"]
        aggregate_rows.append(row)
    aggregate_fields = ["policy"]
    for field in SUMMARY_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_population_std"])
    aggregate_path = output_root / "aggregate_results.csv"
    _atomic_csv(aggregate_path, aggregate_fields, aggregate_rows)
    return {
        "held_out_results": {"file": row_path.name, "sha256": file_hash(row_path)},
        "aggregate_results": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


# ---------- 总入口：成对内层训练 -> outer 选择 -> 固定 epoch 重训 -> held-out 回放 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()
    anchor_config_path = Path(args.anchor_config).resolve()
    config = load_config(config_path)
    protocol_id = str(config["protocol_id"])
    token_mode = str(config.get("token_mode", "full"))
    state_parameters = state_parameters_from_config(config)
    hyperparameters = hyperparameters_from_config(config)
    outer_subjects = tuple(sorted(set(int(value) for value in args.subjects)))
    seeds = tuple(sorted(set(int(value) for value in args.base_seeds)))
    ablations = tuple(dict.fromkeys(args.ablations))
    if (
        not outer_subjects or any(value not in KNOWN_SUBJECTS for value in outer_subjects)
        or not seeds or any(value not in KNOWN_SEEDS for value in seeds)
        or not ablations or any(value not in ABLATIONS for value in ablations)
    ):
        raise ValueError("outer subjects、base seeds 或 ablations 超出冻结集合")
    training_device = torch.device(args.device)
    inference_device = torch.device(args.inference_device)
    if training_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本轮协议要求使用可用 CUDA GPU 训练 LD-GRU")
    torch.empty(1, device=training_device)

    input_master_path = input_root / "run_manifest.json"
    input_master_sha256 = file_hash(input_master_path)
    input_master, input_children = verify_input_root(input_root)
    config_sha256 = file_hash(config_path)
    anchor_config_sha256 = file_hash(anchor_config_path)
    anchor = load_anchor(anchor_config_path)
    sources = source_hashes()
    run_git = git_state()
    environment = runtime_environment(training_device, inference_device)
    scope_is_full = (
        outer_subjects == KNOWN_SUBJECTS and seeds == KNOWN_SEEDS and ablations == ABLATIONS
    )
    run_contract = {
        "protocol_id": protocol_id,
        "scope": "full" if scope_is_full else "subset_preflight",
        "outer_subjects": list(outer_subjects),
        "base_seeds": list(seeds),
        "ablations": list(ablations),
        "training_device": str(training_device),
        "inference_device": str(inference_device),
        "config_sha256": config_sha256,
        "anchor_config": {
            "file": display_path(anchor_config_path),
            "sha256": anchor_config_sha256,
            "cell_id": "c055_r020_l1",
        },
        "input_master_sha256": input_master_sha256,
        "source_sha256": sources,
        "hyperparameters": asdict(hyperparameters),
        **token_mode_identity(token_mode),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        if _read_json(contract_path) != run_contract:
            raise RuntimeError("输出目录已绑定不同 LD-GRU 运行合同")
    elif any(output_root.iterdir()):
        raise FileExistsError("非空输出目录缺少同一运行合同，拒绝覆盖")
    else:
        atomic_json(contract_path, run_contract)

    subjects, artifact_identity = load_subject_data(
        input_root, input_master, input_children, seeds, state_parameters,
    )
    candidate_summary_path = output_root / "candidate_summary.json"
    atomic_json(candidate_summary_path, candidate_summary(subjects, seeds))
    thresholds = tuple(float(value) for value in config["threshold_selection"]["grid"])
    decision_offset = int(config["decision_seed_offset"])
    result_records: list[dict] = []
    result_children: dict[str, dict] = {}

    for seed in seeds:
        split_cache: dict[tuple[int, ...], PreparedSplit] = {}
        needed_pairs = sorted({
            tuple(sorted((outer, other)))
            for outer in outer_subjects
            for other in KNOWN_SUBJECTS
            if other != outer
        })
        pair_jobs: dict[tuple[int, int], dict[str, object]] = {}
        for pair in needed_pairs:
            train_subjects = tuple(value for value in KNOWN_SUBJECTS if value not in pair)
            split = prepare_split(subjects, seed, train_subjects, pair)
            split_cache[train_subjects] = split
            completed_by_ablation: dict[str, dict] = {}
            job_dirs: dict[str, Path] = {}
            for ablation in ablations:
                job_dir = (
                    output_root / "jobs" / f"seed{seed}" / ablation
                    / f"inner_pair_s{pair[0]:02d}_s{pair[1]:02d}"
                )
                contract = split_contract(
                    split,
                    protocol_id=protocol_id,
                    token_mode=token_mode,
                    kind="inner_unordered_held_pair_shared_trajectory",
                    base_seed=seed,
                    ablation=ablation,
                    decision_seed=decision_offset + seed,
                    config_sha256=config_sha256,
                    input_master_sha256=input_master_sha256,
                    hyperparameters=hyperparameters,
                    extra={
                        "held_pair": list(pair),
                        "ordered_inner_folds_represented": [
                            {"outer": pair[0], "validation": pair[1]},
                            {"outer": pair[1], "validation": pair[0]},
                        ],
                    },
                )
                completed_by_ablation[ablation] = train_inner_pair_job(
                    job_dir,
                    split.train_dataset,
                    split.validation_datasets,
                    split.normalizer,
                    ablation=ablation,
                    decision_seed=decision_offset + seed,
                    hyperparameters=hyperparameters,
                    device=training_device,
                    contract=contract,
                    token_mode=token_mode,
                    verbose=not args.quiet,
                )
                job_dirs[ablation] = job_dir
            pair_jobs[pair] = {
                "split": split,
                "completed": completed_by_ablation,
                "job_dirs": job_dirs,
            }

        for outer in outer_subjects:
            for ablation in ablations:
                selection_contract = {
                    "protocol_id": protocol_id,
                    "outer_subject": outer,
                    "outer_subject_access": "forbidden",
                    "base_seed": seed,
                    "ablation": ablation,
                    "thresholds": list(thresholds),
                    "far_anchor": "dual_ewma_drop_abort_c055_r020_l1",
                    "input_master_sha256": input_master_sha256,
                    "config_sha256": config_sha256,
                    "anchor_config_sha256": anchor_config_sha256,
                    "anchor_cell_id": "c055_r020_l1",
                    **token_mode_identity(token_mode),
                    "inner_best_checkpoint_sha256": {
                        str(other): pair_jobs[tuple(sorted((outer, other)))][
                            "completed"
                        ][ablation]["best_checkpoints"][str(other)]["sha256"]
                        for other in KNOWN_SUBJECTS if other != outer
                    },
                }
                selection, selection_path = select_outer_hyperparameters(
                    output_root,
                    subjects,
                    pair_jobs,
                    outer_subject=outer,
                    base_seed=seed,
                    ablation=ablation,
                    token_mode=token_mode,
                    thresholds=thresholds,
                    anchor=anchor,
                    state_parameters=state_parameters,
                    inference_device=inference_device,
                    selection_contract=selection_contract,
                )
                train_subjects = tuple(value for value in KNOWN_SUBJECTS if value != outer)
                final_split = prepare_split(subjects, seed, train_subjects, tuple())
                fixed_epoch = int(selection["selected_final_epoch_round_half_up"])
                final_dir = (
                    output_root / "jobs" / f"seed{seed}" / ablation
                    / f"final_outer_s{outer:02d}"
                )
                final_contract = split_contract(
                    final_split,
                    protocol_id=protocol_id,
                    token_mode=token_mode,
                    kind="outer_final_fixed_epoch",
                    base_seed=seed,
                    ablation=ablation,
                    decision_seed=decision_offset + seed,
                    config_sha256=config_sha256,
                    input_master_sha256=input_master_sha256,
                    hyperparameters=hyperparameters,
                    extra={
                        "outer_test_subject": outer,
                        "outer_test_subject_used_for_training_or_selection": False,
                        "fixed_epoch": fixed_epoch,
                        "selection_file_sha256": file_hash(selection_path),
                        "selection_contract_sha256": selection["selection_contract_sha256"],
                    },
                )
                final_completed = train_final_job(
                    final_dir,
                    final_split.train_dataset,
                    final_split.normalizer,
                    ablation=ablation,
                    decision_seed=decision_offset + seed,
                    fixed_epochs=fixed_epoch,
                    hyperparameters=hyperparameters,
                    device=training_device,
                    contract=final_contract,
                    token_mode=token_mode,
                    verbose=not args.quiet,
                )
                checkpoint_path = final_dir / final_completed["final_checkpoint"]["file"]
                model = load_trained_model(
                    checkpoint_path, ablation, inference_device, token_mode=token_mode,
                )
                result_contract = {
                    "protocol_id": protocol_id,
                    "outer_subject": outer,
                    "base_seed": seed,
                    "ablation": ablation,
                    "selected_threshold": selection["selected_stop_threshold"],
                    "fixed_epoch": fixed_epoch,
                    "selection_sha256": file_hash(selection_path),
                    "final_checkpoint_sha256": file_hash(checkpoint_path),
                    "normalizer_source_subjects": list(train_subjects),
                    "outer_subject_excluded_from_normalizer": outer not in train_subjects,
                    "input_score_sha256": subjects[outer].score_artifacts[seed]["sha256"],
                    "inventory_contract_sha256": subjects[outer].inventory_contract_sha256,
                    "evaluator_source_sha256": sources["protocol_metrics"],
                    "anchor_config_sha256": anchor_config_sha256,
                    "anchor_cell_id": "c055_r020_l1",
                    **token_mode_identity(token_mode),
                }
                result_root = (
                    output_root / "outer_results" / f"seed{seed}" / ablation
                    / f"subject_{outer:02d}"
                )
                result_manifest, result_path = save_final_result(
                    result_root,
                    subjects[outer],
                    base_seed=seed,
                    ablation=ablation,
                    token_mode=token_mode,
                    model=model,
                    normalizer=final_split.normalizer,
                    threshold=float(selection["selected_stop_threshold"]),
                    inference_device=inference_device,
                    anchor=anchor,
                    state_parameters=state_parameters,
                    result_contract=result_contract,
                )
                result_records.append(result_manifest)
                result_key = f"s{outer:02d}_seed{seed}_{ablation}"
                result_children[result_key] = {
                    "manifest": str(result_path.relative_to(output_root)),
                    "manifest_sha256": file_hash(result_path),
                    "selection": str(selection_path.relative_to(output_root)),
                    "selection_sha256": file_hash(selection_path),
                    "final_checkpoint": str(checkpoint_path.relative_to(output_root)),
                    "final_checkpoint_sha256": file_hash(checkpoint_path),
                }
                print(
                    f"Outer S{outer} seed{seed} {ablation}: threshold="
                    f"{selection['selected_stop_threshold']:.2f}, epoch={fixed_epoch}, PASS",
                    flush=True,
                )

    rows, summary = summarize_results(result_records)
    csv_artifacts = write_summary_csvs(output_root, rows, summary)
    immutable_inputs = [
        (input_master_path, input_master_sha256),
        (config_path, config_sha256),
        (anchor_config_path, anchor_config_sha256),
    ]
    for data in subjects.values():
        immutable_inputs.extend([
            (data.bundle_manifest, data.bundle_sha256),
            (data.truth_manifest, data.truth_manifest_sha256),
            (data.truth_event_path, data.truth_event_sha256),
            (data.inventory_contract_path, data.inventory_contract_sha256),
            (data.input_child_path, data.input_child_sha256),
        ])
        for seed in seeds:
            immutable_inputs.append((
                data.score_paths[seed],
                data.score_artifacts[seed]["sha256"],
            ))
    if (
        git_state() != run_git
        or source_hashes() != sources
        or any(file_hash(path) != expected for path, expected in immutable_inputs)
    ):
        raise RuntimeError("LD-GRU 运行期间源码、Git 或冻结输入身份发生变化")
    completed_at = datetime.now(timezone.utc).isoformat()
    claim_status = (
        "PRECOMMIT_NESTED_LOSO_EXPERIMENT"
        if scope_is_full else "SUBSET_FULL_FLOW_PREFLIGHT"
    )
    run_log_path = output_root / "run_log.json"
    atomic_json(run_log_path, {
        "status": "PASS",
        "protocol_id": protocol_id,
        "claim_status": claim_status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "runtime_environment": environment,
        **token_mode_identity(token_mode),
        "expected_job_counts": {
            "inner_unique_training_trajectories": len({
                (seed, ablation, pair)
                for seed in seeds for ablation in ablations
                for pair in {
                    tuple(sorted((outer, other)))
                    for outer in outer_subjects for other in KNOWN_SUBJECTS if other != outer
                }
            }),
            "outer_final_models": len(outer_subjects) * len(seeds) * len(ablations),
        },
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": protocol_id,
        "claim_limit": config["claim_limit"],
        "scope": run_contract["scope"],
        "outer_subjects": list(outer_subjects),
        "base_seeds": list(seeds),
        "ablations": list(ablations),
        **token_mode_identity(token_mode),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        **artifact_identity,
        "config": {"file": display_path(config_path), "sha256": config_sha256},
        "anchor_config": {
            "file": display_path(anchor_config_path),
            "sha256": anchor_config_sha256,
            "cell_id": "c055_r020_l1",
        },
        "input_root_manifest": {
            "file": display_path(input_master_path),
            "sha256": input_master_sha256,
        },
        "candidate_summary": {
            "file": candidate_summary_path.name,
            "sha256": file_hash(candidate_summary_path),
        },
        "results": result_children,
        "csv_artifacts": csv_artifacts,
        "summary": summary,
        "run_log": {"file": run_log_path.name, "sha256": file_hash(run_log_path)},
        "run_contract": {"file": contract_path.name, "sha256": file_hash(contract_path)},
        "source_sha256": sources,
        "runtime_environment": environment,
        "runtime_git": run_git,
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--anchor-config", type=Path, default=DEFAULT_ANCHOR_CONFIG)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(KNOWN_SUBJECTS))
    parser.add_argument("--base-seeds", type=int, nargs="+", default=list(KNOWN_SEEDS))
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
