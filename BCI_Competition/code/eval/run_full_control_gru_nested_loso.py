"""训练并评估连续运行、负责提交与复位的 Full-Control-GRU-v1。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from continuous_oof_subject_loader import ContinuousSubjectData, load_continuous_subjects
from full_control_gru_diagnostics import diagnose_two_state_reset
from full_control_gru_policy import (
    READY,
    WAIT_IDLE,
    ContinuousNormalizer,
    ContinuousSequence,
    build_continuous_targets,
    build_continuous_tokens,
    fit_continuous_normalizer,
    full_control_decisions,
    split_continuous_sequences,
    tensorize_continuous_sequences,
)
from full_control_gru_training import (
    continuous_tensor_hash,
    load_trained_model,
    train_final_job,
    train_inner_pair_job,
)
from ld_gru_training import (
    TrainingHyperparameters,
    atomic_json,
    canonical_hash,
    file_hash,
)
from protocol_metrics import STATEFUL_STRICT, evaluate_online_events
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    atomic_npz,
    core_metrics,
    display_path,
    git_state,
    output_window_rows,
)
from run_hard_vote_matrix import (
    EXPECTED_INPUT_PROTOCOL,
    _check_metric_inventory as _check_strict_metric_inventory,
    verify_input_root,
)
from run_ld_gru_nested_loso import (
    SUMMARY_FIELDS,
    _mean,
    compact_metrics,
    reference_metrics,
    runtime_environment,
    summarize_results,
    write_summary_csvs,
)
from run_oracle_ceiling_analysis import DEFAULT_ANCHOR_CONFIG, load_anchor


EXPECTED_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_full_control_gru_nested_loso_v1"
EXPECTED_CONFIG_CANONICAL_SHA256 = "9fcf2ca76e39c1422d62d013046a8241bbd0a10759ee3dcd7e964b8355435eb4"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_full_control_gru_nested_loso_v1"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_full_control_gru_v1.json"
)
THRESHOLD_FIELDS = (
    "correct_event_rate",
    "idle_false_commands_per_minute",
    "correct_latency_p90_seconds",
    "triggered_class_accuracy",
)
STATE_CODE = {READY: 0, WAIT_IDLE: 1}


@dataclass
class PreparedSplit:
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    normalizer: ContinuousNormalizer
    train_dataset: object
    validation_datasets: dict[int, object]


# ---------- 冻结配置：任何模型、标签、阈值或 split 参数变化都必须升级协议 ----------
def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    training = config.get("training", {})
    expected_grid = [round(value / 10, 1) for value in range(3, 10)]
    if (
        canonical_hash(config) != EXPECTED_CONFIG_CANONICAL_SHA256
        or config.get("protocol_id") != EXPECTED_PROTOCOL
        or config.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or config.get("included_session") != 0
        or config.get("test_session_access") != "forbidden"
        or tuple(config.get("subjects", [])) != KNOWN_SUBJECTS
        or tuple(config.get("base_seeds", [])) != KNOWN_SEEDS
        or config.get("decision_seed_offset") != 2000
        or config.get("model", {}).get("total_parameter_count") != 1429
        or config.get("decision_shell", {}).get("handwritten_stage1_open_hold_drop_reset") is not False
        or config.get("threshold_selection", {}).get("commit_grid") != expected_grid
        or config.get("threshold_selection", {}).get("reset_grid") != expected_grid
        or training.get("batch_size") != 512
        or training.get("max_epochs") != 200
        or training.get("early_stopping_patience") != 20
    ):
        raise RuntimeError("Full-Control-GRU-v1 配置与冻结协议不一致")
    return config


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


def source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "policy": EVAL_DIR / "full_control_gru_policy.py",
        "trainer": EVAL_DIR / "full_control_gru_training.py",
        "two_state_diagnostics": EVAL_DIR / "full_control_gru_diagnostics.py",
        "subject_loader": EVAL_DIR / "continuous_oof_subject_loader.py",
        "protocol_metrics": EVAL_DIR / "protocol_metrics.py",
        "commit_reset_diagnostics": EVAL_DIR / "commit_reset_diagnostics.py",
        "reference_policy_runner": EVAL_DIR / "run_ld_gru_nested_loso.py",
        "online_truth_inventory": EVAL_DIR / "online_truth_inventory.py",
        "single_subject_input_runner": EVAL_DIR / "run_epoch50_online_oof.py",
        "input_contract_reader": EVAL_DIR / "run_hard_vote_matrix.py",
        "bundle_reader": TRAIN_DIR / "oof_training_bundle.py",
    }
    return {name: file_hash(path) for name, path in paths.items()}


# ---------- 连续序列物化：同一 segment 内全程保留顺序，边界窗只从 loss 中忽略 ----------
def build_sequence_cache(
    subjects: dict[int, ContinuousSubjectData],
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[int, int], tuple[ContinuousSequence, ...]], dict]:
    cache: dict[tuple[int, int], tuple[ContinuousSequence, ...]] = {}
    summary: dict[str, dict] = {}
    for subject, data in subjects.items():
        targets = build_continuous_targets(data.inventory.windows, data.inventory.events)
        counts = {str(value): int(np.sum(targets == value)) for value in (-100, 0, 1, 2, 3, 4)}
        summary[str(subject)] = {
            "target_counts": counts,
            "segment_count": len(data.inventory.segments),
            "window_count": len(data.inventory.windows),
        }
        for seed in seeds:
            tokens = build_continuous_tokens(
                data.inventory.windows,
                data.stage1_logits[seed],
                data.stage2_logits[seed],
            )
            sequences = split_continuous_sequences(data.inventory.windows, tokens, targets)
            if {item.subject_id for item in sequences} != {subject}:
                raise RuntimeError("连续序列被试身份错误")
            cache[(subject, seed)] = sequences
    return cache, summary


def prepare_split(
    cache: dict[tuple[int, int], tuple[ContinuousSequence, ...]],
    seed: int,
    train_subjects: tuple[int, ...],
    validation_subjects: tuple[int, ...],
) -> PreparedSplit:
    if set(train_subjects) & set(validation_subjects):
        raise ValueError("训练与验证被试不得重叠")
    train_sequences = tuple(
        sequence for subject in train_subjects for sequence in cache[(subject, seed)]
    )
    if {item.subject_id for item in train_sequences} != set(train_subjects):
        raise RuntimeError("连续训练集被试身份与 split 不一致")
    normalizer = fit_continuous_normalizer(train_sequences)
    train_dataset = tensorize_continuous_sequences(train_sequences, normalizer)
    validation_datasets = {
        subject: tensorize_continuous_sequences(cache[(subject, seed)], normalizer)
        for subject in validation_subjects
    }
    return PreparedSplit(
        train_subjects, validation_subjects, normalizer,
        train_dataset, validation_datasets,
    )


def split_contract(
    split: PreparedSplit,
    *,
    kind: str,
    base_seed: int,
    decision_seed: int,
    config_sha256: str,
    input_master_sha256: str,
    hyperparameters: TrainingHyperparameters,
    extra: dict,
) -> dict:
    return {
        "protocol_id": EXPECTED_PROTOCOL,
        "job_kind": kind,
        "included_session": 0,
        "test_session_access": "forbidden",
        "base_seed": base_seed,
        "decision_seed": decision_seed,
        "train_subjects": list(split.train_subjects),
        "validation_subjects": list(split.validation_subjects),
        "outer_test_subjects_used": [],
        "train_tensor_sha256": continuous_tensor_hash(split.train_dataset),
        "validation_tensor_sha256": {
            str(subject): continuous_tensor_hash(dataset)
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
        **extra,
    }


# ---------- 正式两状态评估：状态外壳只禁止重复输出，不提供候选态信息 ----------
def evaluate_policy(data: ContinuousSubjectData, decisions: list) -> tuple[dict, dict]:
    result = evaluate_online_events(
        data.inventory.segments,
        data.inventory.events,
        data.inventory.windows,
        decisions,
        mode=STATEFUL_STRICT,
    )
    _check_strict_metric_inventory(result, data.inventory_contract)
    diagnostics = diagnose_two_state_reset(
        data.inventory.segments,
        data.inventory.events,
        data.inventory.windows,
        decisions,
        result,
    )
    return result, diagnostics


def threshold_metrics(result: dict) -> dict:
    compact = core_metrics(result)
    return {field: compact[field] for field in THRESHOLD_FIELDS}


# ---------- 内层二维阈值选择：同一模型联合冻结提交与复位工作点 ----------
def select_outer_hyperparameters(
    output_root: Path,
    subjects: dict[int, ContinuousSubjectData],
    pair_jobs: dict[tuple[int, int], dict],
    *,
    outer_subject: int,
    base_seed: int,
    commit_grid: tuple[float, ...],
    reset_grid: tuple[float, ...],
    anchor,
    inference_device: torch.device,
    selection_contract: dict,
) -> tuple[dict, Path]:
    selection_path = output_root / "selections" / f"seed{base_seed}" / f"outer_s{outer_subject:02d}.json"
    contract_sha256 = canonical_hash(selection_contract)
    if selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if payload.get("selection_contract_sha256") != contract_sha256:
            raise RuntimeError("既有全控制阈值选择与当前合同不一致")
        return payload, selection_path

    validation_subjects = tuple(value for value in KNOWN_SUBJECTS if value != outer_subject)
    per_subject: dict[str, dict] = {}
    best_epochs: list[int] = []
    for validation_subject in validation_subjects:
        pair = tuple(sorted((outer_subject, validation_subject)))
        job = pair_jobs[pair]
        record = job["completed"]["best_checkpoints"][str(validation_subject)]
        best_epochs.append(int(record["best_epoch"]))
        checkpoint = Path(job["job_dir"]) / record["file"]
        model = load_trained_model(checkpoint, inference_device)
        data = subjects[validation_subject]
        cells: dict[str, dict] = {}
        for commit in commit_grid:
            for reset in reset_grid:
                decisions, _ = full_control_decisions(
                    data.inventory.windows,
                    data.stage1_logits[base_seed],
                    data.stage2_logits[base_seed],
                    model,
                    job["split"].normalizer,
                    commit,
                    reset,
                    inference_device,
                )
                evaluated, _ = evaluate_policy(data, decisions)
                cells[f"c{commit:.2f}_r{reset:.2f}"] = threshold_metrics(evaluated)
        baseline = reference_metrics(data, base_seed, anchor)["dual_ewma_drop_abort_c055_r020_l1"]
        per_subject[str(validation_subject)] = {
            "pair": list(pair),
            "train_subjects": list(job["split"].train_subjects),
            "best_epoch": record["best_epoch"],
            "best_validation_loss": record["best_validation_loss"],
            "baseline_far": baseline["idle_false_commands_per_minute"],
            "cells": cells,
        }

    baseline_far = float(np.mean([
        per_subject[str(subject)]["baseline_far"] for subject in validation_subjects
    ]))
    aggregate: dict[str, dict] = {}
    feasible: list[tuple[float, float]] = []
    for commit in commit_grid:
        for reset in reset_grid:
            key = f"c{commit:.2f}_r{reset:.2f}"
            aggregate[key] = {
                field: _mean([
                    per_subject[str(subject)]["cells"][key][field]
                    for subject in validation_subjects
                ])
                for field in THRESHOLD_FIELDS
            }
            aggregate[key]["far_feasible"] = (
                aggregate[key]["idle_false_commands_per_minute"] <= baseline_far + 1e-12
            )
            if aggregate[key]["far_feasible"]:
                feasible.append((commit, reset))
    if not feasible:
        raise RuntimeError("全控制二维阈值网格没有满足 anchor FAR 的工作点")

    def ranking(cell: tuple[float, float]) -> tuple[float, float, float, float, float]:
        commit, reset = cell
        row = aggregate[f"c{commit:.2f}_r{reset:.2f}"]
        p90 = row["correct_latency_p90_seconds"]
        return (
            float(row["correct_event_rate"]),
            -float(row["idle_false_commands_per_minute"]),
            -math.inf if p90 is None else -float(p90),
            commit,
            reset,
        )

    selected_commit, selected_reset = max(feasible, key=ranking)
    median_epoch = float(np.median(np.asarray(best_epochs, dtype=np.float64)))
    selected_epoch = int(math.floor(median_epoch + 0.5))
    payload = {
        "status": "PASS",
        "selection_contract_sha256": contract_sha256,
        "outer_subject": outer_subject,
        "outer_subject_access": "forbidden_and_not_used",
        "base_seed": base_seed,
        "validation_subjects": list(validation_subjects),
        "baseline_equal_subject_mean_far": baseline_far,
        "best_epochs": best_epochs,
        "median_best_epoch": median_epoch,
        "selected_final_epoch_round_half_up": selected_epoch,
        "selected_commit_threshold": selected_commit,
        "selected_reset_threshold": selected_reset,
        "threshold_aggregate": aggregate,
        "per_validation_subject": per_subject,
    }
    atomic_json(selection_path, payload)
    return payload, selection_path


# ---------- 外层结果：保存每窗联合概率、GRU 隐状态和两状态决策轨迹 ----------
def save_final_result(
    result_root: Path,
    data: ContinuousSubjectData,
    *,
    base_seed: int,
    model,
    normalizer: ContinuousNormalizer,
    commit_threshold: float,
    reset_threshold: float,
    inference_device: torch.device,
    anchor,
    result_contract: dict,
) -> tuple[dict, Path]:
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "result_manifest.json"
    contract_sha256 = canonical_hash(result_contract)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("result_contract_sha256") != contract_sha256:
            raise RuntimeError("既有全控制 outer 结果合同不一致")
        for artifact in manifest["artifacts"].values():
            if file_hash(result_root / artifact["file"]) != artifact["sha256"]:
                raise RuntimeError("全控制 outer 结果产物哈希不一致")
        return manifest, manifest_path

    decisions, trace = full_control_decisions(
        data.inventory.windows,
        data.stage1_logits[base_seed],
        data.stage2_logits[base_seed],
        model,
        normalizer,
        commit_threshold,
        reset_threshold,
        inference_device,
    )
    learned, diagnostics = evaluate_policy(data, decisions)
    policies = reference_metrics(data, base_seed, anchor)
    # 候选式 spillover 归因不适用于无候选 GRU；正式 FAR 仍完整保留。
    policies["full_control_gru_v1"] = compact_metrics(learned)
    metrics_path = result_root / "metrics.json"
    atomic_json(metrics_path, {
        "outer_subject": data.subject,
        "base_seed": base_seed,
        "commit_threshold": commit_threshold,
        "reset_threshold": reset_threshold,
        "policies": policies,
        "learned_full_evaluation": learned,
        "learned_two_state_reset_diagnostics": diagnostics,
    })
    trajectory_path = result_root / "trajectory.npz"
    atomic_npz(
        trajectory_path,
        window_rows=output_window_rows(data.inventory.windows),
        raw_token=np.stack([item.raw_token for item in trace]),
        normalized_token=np.stack([item.normalized_token for item in trace]),
        hidden_state=np.stack([item.hidden for item in trace]),
        output_logits=np.stack([item.logits for item in trace]),
        joint_probabilities=np.stack([item.probabilities for item in trace]),
        emitted_class=np.asarray([item.emitted_class for item in decisions], dtype=np.int8),
        state_before=np.asarray([STATE_CODE[item.decision_state_before] for item in decisions], dtype=np.uint8),
        state_after=np.asarray([STATE_CODE[item.decision_state_after] for item in decisions], dtype=np.uint8),
        state_code_names=np.asarray([READY, WAIT_IDLE]),
    )
    artifacts = {
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectory": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
    }
    manifest = {
        "status": "PASS",
        "result_contract_sha256": contract_sha256,
        "outer_subject": data.subject,
        "base_seed": base_seed,
        "commit_threshold": commit_threshold,
        "reset_threshold": reset_threshold,
        "policies": policies,
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    return manifest, manifest_path


# ---------- 总入口：成对内层训练 -> 二维阈值选择 -> 固定 epoch 重训 -> outer 回放 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()
    anchor_path = Path(args.anchor_config).resolve()
    config = load_config(config_path)
    hyperparameters = hyperparameters_from_config(config)
    outer_subjects = tuple(sorted(set(int(value) for value in args.subjects)))
    seeds = tuple(sorted(set(int(value) for value in args.base_seeds)))
    if (
        not outer_subjects or any(value not in KNOWN_SUBJECTS for value in outer_subjects)
        or not seeds or any(value not in KNOWN_SEEDS for value in seeds)
    ):
        raise ValueError("outer subjects 或 base seeds 超出冻结集合")
    training_device = torch.device(args.device)
    inference_device = torch.device(args.inference_device)
    if training_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-Control-GRU 正式训练要求可用 CUDA")
    torch.empty(1, device=training_device)

    input_master_path = input_root / "run_manifest.json"
    input_master_sha256 = file_hash(input_master_path)
    input_master, input_children = verify_input_root(input_root)
    config_sha256 = file_hash(config_path)
    anchor_sha256 = file_hash(anchor_path)
    anchor = load_anchor(anchor_path)
    sources = source_hashes()
    run_git = git_state()
    environment = runtime_environment(training_device, inference_device)
    full_scope = outer_subjects == KNOWN_SUBJECTS and seeds == KNOWN_SEEDS
    run_contract = {
        "protocol_id": EXPECTED_PROTOCOL,
        "scope": "full" if full_scope else "subset_preflight",
        "outer_subjects": list(outer_subjects),
        "base_seeds": list(seeds),
        "training_device": str(training_device),
        "inference_device": str(inference_device),
        "config_sha256": config_sha256,
        "anchor_config_sha256": anchor_sha256,
        "input_master_sha256": input_master_sha256,
        "source_sha256": sources,
        "hyperparameters": asdict(hyperparameters),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise RuntimeError("输出目录已绑定不同 Full-Control-GRU 合同")
    elif any(output_root.iterdir()):
        raise FileExistsError("非空输出目录缺少同一运行合同")
    else:
        atomic_json(contract_path, run_contract)

    subjects, artifact_identity = load_continuous_subjects(
        input_root, input_master, input_children, seeds,
    )
    sequence_cache, sequence_summary = build_sequence_cache(subjects, seeds)
    sequence_summary_path = output_root / "sequence_summary.json"
    atomic_json(sequence_summary_path, sequence_summary)
    commit_grid = tuple(float(value) for value in config["threshold_selection"]["commit_grid"])
    reset_grid = tuple(float(value) for value in config["threshold_selection"]["reset_grid"])
    decision_offset = int(config["decision_seed_offset"])
    records: list[dict] = []
    children: dict[str, dict] = {}

    for seed in seeds:
        needed_pairs = sorted({
            tuple(sorted((outer, other)))
            for outer in outer_subjects for other in KNOWN_SUBJECTS if other != outer
        })
        pair_jobs: dict[tuple[int, int], dict] = {}
        for pair in needed_pairs:
            train_subjects = tuple(value for value in KNOWN_SUBJECTS if value not in pair)
            split = prepare_split(sequence_cache, seed, train_subjects, pair)
            job_dir = output_root / "jobs" / f"seed{seed}" / f"inner_pair_s{pair[0]:02d}_s{pair[1]:02d}"
            contract = split_contract(
                split,
                kind="inner_unordered_held_pair_shared_trajectory",
                base_seed=seed,
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
            completed = train_inner_pair_job(
                job_dir,
                split.train_dataset,
                split.validation_datasets,
                split.normalizer,
                decision_seed=decision_offset + seed,
                hyperparameters=hyperparameters,
                device=training_device,
                contract=contract,
                verbose=not args.quiet,
            )
            pair_jobs[pair] = {"split": split, "completed": completed, "job_dir": job_dir}

        for outer in outer_subjects:
            selection_contract = {
                "protocol_id": EXPECTED_PROTOCOL,
                "outer_subject": outer,
                "outer_subject_access": "forbidden",
                "base_seed": seed,
                "commit_grid": list(commit_grid),
                "reset_grid": list(reset_grid),
                "far_anchor": "dual_ewma_drop_abort_c055_r020_l1",
                "input_master_sha256": input_master_sha256,
                "config_sha256": config_sha256,
                "anchor_config_sha256": anchor_sha256,
                "inner_best_checkpoint_sha256": {
                    str(other): pair_jobs[tuple(sorted((outer, other)))]["completed"]
                    ["best_checkpoints"][str(other)]["sha256"]
                    for other in KNOWN_SUBJECTS if other != outer
                },
            }
            selection, selection_path = select_outer_hyperparameters(
                output_root,
                subjects,
                pair_jobs,
                outer_subject=outer,
                base_seed=seed,
                commit_grid=commit_grid,
                reset_grid=reset_grid,
                anchor=anchor,
                inference_device=inference_device,
                selection_contract=selection_contract,
            )
            train_subjects = tuple(value for value in KNOWN_SUBJECTS if value != outer)
            final_split = prepare_split(sequence_cache, seed, train_subjects, tuple())
            fixed_epoch = int(selection["selected_final_epoch_round_half_up"])
            final_dir = output_root / "jobs" / f"seed{seed}" / f"final_outer_s{outer:02d}"
            final_contract = split_contract(
                final_split,
                kind="outer_final_fixed_epoch",
                base_seed=seed,
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
                decision_seed=decision_offset + seed,
                fixed_epochs=fixed_epoch,
                hyperparameters=hyperparameters,
                device=training_device,
                contract=final_contract,
                verbose=not args.quiet,
            )
            checkpoint = final_dir / final_completed["final_checkpoint"]["file"]
            model = load_trained_model(checkpoint, inference_device)
            result_contract = {
                "protocol_id": EXPECTED_PROTOCOL,
                "outer_subject": outer,
                "base_seed": seed,
                "commit_threshold": selection["selected_commit_threshold"],
                "reset_threshold": selection["selected_reset_threshold"],
                "fixed_epoch": fixed_epoch,
                "selection_sha256": file_hash(selection_path),
                "final_checkpoint_sha256": file_hash(checkpoint),
                "normalizer_source_subjects": list(train_subjects),
                "outer_subject_excluded_from_normalizer": outer not in train_subjects,
                "input_score_sha256": subjects[outer].score_artifacts[seed]["sha256"],
                "inventory_contract_sha256": subjects[outer].inventory_contract_sha256,
                "evaluator_source_sha256": sources["protocol_metrics"],
                "anchor_config_sha256": anchor_sha256,
            }
            result_root = output_root / "outer_results" / f"seed{seed}" / f"subject_{outer:02d}"
            result, result_path = save_final_result(
                result_root,
                subjects[outer],
                base_seed=seed,
                model=model,
                normalizer=final_split.normalizer,
                commit_threshold=float(selection["selected_commit_threshold"]),
                reset_threshold=float(selection["selected_reset_threshold"]),
                inference_device=inference_device,
                anchor=anchor,
                result_contract=result_contract,
            )
            records.append(result)
            key = f"s{outer:02d}_seed{seed}"
            children[key] = {
                "manifest": str(result_path.relative_to(output_root)),
                "manifest_sha256": file_hash(result_path),
                "selection": str(selection_path.relative_to(output_root)),
                "selection_sha256": file_hash(selection_path),
                "final_checkpoint": str(checkpoint.relative_to(output_root)),
                "final_checkpoint_sha256": file_hash(checkpoint),
            }
            print(
                f"Outer S{outer} seed{seed}: commit={selection['selected_commit_threshold']:.2f}, "
                f"reset={selection['selected_reset_threshold']:.2f}, epoch={fixed_epoch}, PASS",
                flush=True,
            )

    rows, summary = summarize_results(records)
    csv_artifacts = write_summary_csvs(output_root, rows, summary)
    immutable = [
        (input_master_path, input_master_sha256),
        (config_path, config_sha256),
        (anchor_path, anchor_sha256),
    ]
    for data in subjects.values():
        immutable.extend([
            (data.bundle_manifest, data.bundle_sha256),
            (data.truth_manifest, data.truth_manifest_sha256),
            (data.truth_event_path, data.truth_event_sha256),
            (data.inventory_contract_path, data.inventory_contract_sha256),
            (data.input_child_path, data.input_child_sha256),
        ])
        for seed in seeds:
            immutable.append((data.score_paths[seed], data.score_artifacts[seed]["sha256"]))
    if (
        git_state() != run_git
        or source_hashes() != sources
        or any(file_hash(path) != expected for path, expected in immutable)
    ):
        raise RuntimeError("Full-Control-GRU 运行期间源码、Git 或冻结输入发生变化")

    claim_status = "PRECOMMIT_NESTED_LOSO_EXPERIMENT" if full_scope else "SUBSET_FULL_FLOW_PREFLIGHT"
    run_log_path = output_root / "run_log.json"
    atomic_json(run_log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_PROTOCOL,
        "claim_status": claim_status,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "runtime_environment": environment,
        "expected_job_counts": {
            "inner_unique_training_trajectories": len({
                (seed, tuple(sorted((outer, other))))
                for seed in seeds for outer in outer_subjects
                for other in KNOWN_SUBJECTS if other != outer
            }),
            "outer_final_models": len(outer_subjects) * len(seeds),
        },
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_PROTOCOL,
        "claim_limit": config["claim_limit"],
        "scope": run_contract["scope"],
        "outer_subjects": list(outer_subjects),
        "base_seeds": list(seeds),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded_before_freeze",
        **artifact_identity,
        "config": {"file": display_path(config_path), "sha256": config_sha256},
        "anchor_config": {"file": display_path(anchor_path), "sha256": anchor_sha256},
        "input_root_manifest": {"file": display_path(input_master_path), "sha256": input_master_sha256},
        "sequence_summary": {"file": sequence_summary_path.name, "sha256": file_hash(sequence_summary_path)},
        "results": children,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
