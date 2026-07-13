"""从冻结的九被试连续 OOF logits 生成因果硬投票 N-K 诊断矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hard_vote_policy import VOTE_GRID, WINDOW_COUNTS, policy_id, stateful_hard_vote_decisions
from protocol_metrics import (
    READY,
    STATEFUL_STRICT,
    WAIT_IDLE,
    evaluate_online_events,
    hierarchical_5class_predictions,
)
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    atomic_json,
    atomic_npz,
    build_online_inventory,
    core_metrics,
    default_subject_paths,
    display_path,
    file_hash,
    git_state,
    output_window_rows,
    verify_inventory_contract,
)
from run_epoch50_online_oof_all_subjects import verify_child_artifacts
from oof_training_bundle import load_bundle


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_hard_vote_matrix_v1"
)
DEFAULT_POLICY_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_hard_vote_matrix_v1.json"
)
EXPECTED_INPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_single_window_oof_v1"
EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_hard_vote_matrix_v1"
# 历史正式输入绑定生成时源码，而不是要求它永远等于后续策略的当前源码。
FROZEN_INPUT_MASTER_SOURCE_SHA256 = {
    "multi_subject_runner": "37fa41eed2c9a8755510acb7e58d36f93b85f4827d8797de4a58027560ed5c1a",
    "single_subject_runner": "8552a3d8295c6a4c6125ca9fc1fb6c98e29011fde163089d8e39b64f84c6cdfb",
}
FROZEN_INPUT_CHILD_SOURCE_SHA256 = {
    "runner": "8552a3d8295c6a4c6125ca9fc1fb6c98e29011fde163089d8e39b64f84c6cdfb",
    "protocol_metrics": "098e8db59ae9a396f8d32a22f1e05b6841272a850fe348196304efb9f212ad7e",
    "oof_training_bundle_reader": "f5a2deb40b64187dcbbce34b5e4c382ebb637177851776713eb96c4e99e80f56",
    "model_factory": "9e6b6af936f088cf0ed3cb25f52cf59d460159019b50beaf7b2b7b7c93173a60",
    "eegnet": "73c97e1bae388ad599025c61cde27f4d451d12b3b83fa898232d6fe90d3fdaed",
}
EXPECTED_NPZ_FIELDS = {
    "window_rows", "stage1_logits", "stage2_logits",
    "stateless_emitted", "stateful_emitted", "stateful_before", "stateful_after",
    "state_code_names",
}
CORE_FIELDS = (
    "correct_event_rate", "macro_correct_event_rate", "event_trigger_rate",
    "triggered_class_accuracy", "miss_rate", "idle_false_commands_per_minute",
    "correct_latency_mean_seconds", "correct_latency_median_seconds",
    "correct_latency_p90_seconds",
)


# ---------- 输入合同：只接受已在干净提交上生成并闭合哈希的完整九被试基线 ----------
def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _safe_artifact(root: Path, relative_name: str) -> Path:
    relative = Path(str(relative_name))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"产物路径非法: {relative_name}")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"产物越出结果目录: {relative_name}")
    return resolved


def validate_policy_config(config: dict) -> tuple[tuple[int, int], ...]:
    """配置与代码常量双向核对，防止 JSON 写法和实际枚举网格不一致。"""
    # 标签语义和单窗锚点也是协议的一部分，不能只校验数值网格。
    expected_joint_hard_labels = {
        "0": "Stage 1 argmax is IDLE",
        "1_to_4": "Stage 1 argmax is Task and Stage 2 argmax class is 1..4",
        "argmax_tie_rule": "smallest class index",
    }
    thresholds = config.get("vote_thresholds", {})
    config_grid = tuple(
        (int(window_count), int(vote_threshold))
        for window_count in config.get("window_counts", [])
        for vote_threshold in thresholds.get(str(window_count), [])
    )
    if (
        config.get("protocol_id") != EXPECTED_OUTPUT_PROTOCOL
        or config.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or config.get("included_session") != 0
        or config.get("test_session_access") != "forbidden"
        or config.get("joint_hard_labels") != expected_joint_hard_labels
        or tuple(config.get("window_counts", [])) != WINDOW_COUNTS
        or config_grid != VOTE_GRID
        or config.get("require_full_buffer") is not True
        or config.get("same_n_k_for_mi_and_idle") is not True
        or config.get("single_window_reference")
        != "reuse frozen N=1 stateful strict result"
        or config.get("selection_status") != "none_all_cells_reported"
        or config.get("cache_reset") != [
            "segment_start",
            "ready_to_wait_idle_after_mi_output",
            "wait_idle_to_ready_after_idle_confirmation",
        ]
    ):
        raise RuntimeError("硬投票配置与冻结首版协议不一致")
    return config_grid


def runtime_environment() -> dict[str, str]:
    """记录实际解释器和核心依赖，结果清单不依赖聊天命令证明运行环境。"""
    prefix = Path(sys.prefix).resolve()
    return {
        "environment_name": prefix.name,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(prefix),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


def verify_input_root(input_root: Path) -> tuple[dict, dict[int, dict]]:
    """复核输入总清单、九个子清单及其全部原始分数产物。"""
    input_root = Path(input_root).resolve()
    master_path = input_root / "run_manifest.json"
    master = _read_json(master_path)
    master_log = _safe_artifact(input_root, master.get("run_log", {}).get("file", ""))
    if (
        master.get("status") != "PASS"
        or master.get("claim_status") != "CLEAN_COMMIT_FORMAL_CANDIDATE"
        or master.get("protocol_id") != EXPECTED_INPUT_PROTOCOL
        or tuple(master.get("subjects", [])) != KNOWN_SUBJECTS
        or tuple(master.get("seeds", [])) != KNOWN_SEEDS
        or master.get("included_session") != 0
        or master.get("test_session_access") != "forbidden_and_not_loaded"
        or master.get("runtime_git", {}).get("dirty") is not False
        or not master_log.is_file()
        or file_hash(master_log) != master.get("run_log", {}).get("sha256")
        or master.get("source_sha256") != FROZEN_INPUT_MASTER_SOURCE_SHA256
        or set(master.get("children", {})) != {str(subject) for subject in KNOWN_SUBJECTS}
    ):
        raise RuntimeError("输入九被试单窗正式候选清单不完整或与当前读取代码不兼容")

    children: dict[int, dict] = {}
    for subject in KNOWN_SUBJECTS:
        record = master["children"][str(subject)]
        child_path = _safe_artifact(input_root, record.get("manifest", ""))
        if not child_path.is_file() or file_hash(child_path) != record.get("manifest_sha256"):
            raise RuntimeError(f"Subject {subject} 输入子清单哈希不一致")
        child = _read_json(child_path)
        if (
            child.get("claim_status") != "CLEAN_COMMIT_FORMAL_CANDIDATE"
            or child.get("subject") != subject
            or child.get("runtime", {}).get("git") != master.get("runtime_git")
        ):
            raise RuntimeError(f"Subject {subject} 输入不是干净提交正式候选")
        verify_child_artifacts(
            child_path.parent,
            child,
            subject,
            KNOWN_SEEDS,
            expected_source_hashes=FROZEN_INPUT_CHILD_SOURCE_SHA256,
        )
        children[subject] = child
    return master, children


# ---------- 单被试重放：从 session0-only bundle 恢复真值库存，从 NPZ 只读取 logits ----------
def _load_subject_inventory(subject: int):
    paths = default_subject_paths(subject)
    context = load_bundle(paths.bundle_manifest)
    inventory = build_online_inventory(context)
    contract = _read_json(paths.inventory_contract)
    verify_inventory_contract(context, inventory, contract)
    return context, inventory, contract, paths


def _load_seed_logits(
    input_subject_root: Path,
    input_manifest: dict,
    seed: int,
    expected_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    artifact = input_manifest["seed_artifacts"][str(seed)]["scores_and_decisions"]
    path = _safe_artifact(input_subject_root, artifact["file"])
    if not path.is_file() or file_hash(path) != artifact.get("sha256"):
        raise RuntimeError(f"seed {seed} 原始分数文件哈希不一致")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_NPZ_FIELDS:
            raise RuntimeError(f"seed {seed} 原始分数字段集合漂移")
        rows = payload["window_rows"].copy()
        stage1 = payload["stage1_logits"].astype(np.float32, copy=True)
        stage2 = payload["stage2_logits"].astype(np.float32, copy=True)
    if (
        not np.array_equal(rows, expected_rows)
        or stage1.shape != (len(rows), 2)
        or stage2.shape != (len(rows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
        or np.any(rows["session_id"] != 0)
    ):
        raise RuntimeError(f"seed {seed} logit 与冻结窗口母索引不一致")
    return stage1, stage2, {"file": display_path(path), "sha256": artifact["sha256"]}


def _check_metric_inventory(result: dict, contract: dict) -> None:
    frozen = contract["inventory"]
    if (
        result["evaluation_mode"] != STATEFUL_STRICT
        or result["scoring_segment_inventory_sha256"] != frozen["segment_inventory_sha256"]
        or result["expected_window_inventory_sha256"] != frozen["window_inventory_sha256"]
        or result["event_inventory_sha256"] != frozen["event_inventory_sha256"]
        or result["event_count"] != frozen["event_count"]
    ):
        raise RuntimeError("硬投票指标与冻结在线库存不一致")


def _save_seed_matrix(
    output_root: Path,
    inventory,
    contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
) -> tuple[dict[str, dict], dict]:
    hard_labels = hierarchical_5class_predictions(stage1_logits, stage2_logits)
    state_code = {READY: 0, WAIT_IDLE: 1}
    arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "hard_labels": hard_labels.astype(np.int8),
        "grid_window_count": np.asarray([item[0] for item in VOTE_GRID], dtype=np.uint8),
        "grid_vote_threshold": np.asarray([item[1] for item in VOTE_GRID], dtype=np.uint8),
        "state_code_names": np.asarray([READY, WAIT_IDLE]),
    }
    metrics_payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "selection_status": "none_all_cells_reported",
        "grid": {},
    }
    core_by_policy: dict[str, dict] = {}

    for window_count, vote_threshold in VOTE_GRID:
        identifier = policy_id(window_count, vote_threshold)
        decisions = stateful_hard_vote_decisions(
            inventory.windows,
            hard_labels,
            window_count=window_count,
            vote_threshold=vote_threshold,
        )
        result = evaluate_online_events(
            inventory.segments,
            inventory.events,
            inventory.windows,
            decisions,
            mode=STATEFUL_STRICT,
        )
        _check_metric_inventory(result, contract)
        metrics_payload["grid"][identifier] = {
            "window_count": window_count,
            "vote_threshold": vote_threshold,
            "metrics": result,
        }
        core_by_policy[identifier] = core_metrics(result)
        arrays[f"{identifier}_emitted"] = np.asarray(
            [item.emitted_class for item in decisions], dtype=np.int8,
        )
        arrays[f"{identifier}_before"] = np.asarray(
            [state_code[item.decision_state_before] for item in decisions], dtype=np.uint8,
        )
        arrays[f"{identifier}_after"] = np.asarray(
            [state_code[item.decision_state_after] for item in decisions], dtype=np.uint8,
        )

    metrics_path = output_root / f"seed{seed}_hard_vote_metrics.json"
    trajectory_path = output_root / f"seed{seed}_hard_vote_trajectories.npz"
    atomic_json(metrics_path, metrics_payload)
    atomic_npz(trajectory_path, **arrays)
    return core_by_policy, {
        "input_scores": input_scores,
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
    }


# ---------- 汇总规则：每个策略和 seed 内先等权平均被试，再汇总三个配对 seed ----------
def _statistics(values: list[float]) -> dict:
    return {
        "mean": None if not values else float(np.mean(values)),
        "population_std": None if not values else float(np.std(values, ddof=0)),
        "valid_count": len(values),
    }


def summarize_subject(seed_results: dict[int, dict[str, dict]]) -> dict:
    result: dict[str, dict] = {}
    for window_count, vote_threshold in VOTE_GRID:
        identifier = policy_id(window_count, vote_threshold)
        per_seed = {str(seed): seed_results[seed][identifier] for seed in KNOWN_SEEDS}
        aggregate = {}
        for field in CORE_FIELDS:
            values = [float(row[field]) for row in per_seed.values() if row[field] is not None]
            aggregate[field] = _statistics(values)
        result[identifier] = {
            "window_count": window_count,
            "vote_threshold": vote_threshold,
            "per_seed": per_seed,
            "aggregate_across_seeds": aggregate,
        }
    return result


def aggregate_subject_matrix(subject_summaries: dict[int, dict]) -> dict:
    """所有指标均采用被试等权宏平均，禁止把不同被试事件池化。"""
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("硬投票汇总缺少 Subject 1–9")
    result: dict[str, dict] = {}
    for window_count, vote_threshold in VOTE_GRID:
        identifier = policy_id(window_count, vote_threshold)
        per_seed_subject_macro: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            row = {}
            for field in CORE_FIELDS:
                values = [
                    subject_summaries[subject][identifier]["per_seed"][str(seed)][field]
                    for subject in KNOWN_SUBJECTS
                ]
                finite = [float(value) for value in values if value is not None]
                row[field] = {
                    "mean": None if not finite else float(np.mean(finite)),
                    "valid_subject_count": len(finite),
                }
            per_seed_subject_macro[str(seed)] = row

        aggregate_across_seeds = {}
        across_subjects = {}
        for field in CORE_FIELDS:
            seed_values = [
                per_seed_subject_macro[str(seed)][field]["mean"]
                for seed in KNOWN_SEEDS
                if per_seed_subject_macro[str(seed)][field]["mean"] is not None
            ]
            aggregate_across_seeds[field] = _statistics(seed_values)
            by_subject = {
                str(subject): subject_summaries[subject][identifier]
                ["aggregate_across_seeds"][field]["mean"]
                for subject in KNOWN_SUBJECTS
            }
            finite = [float(value) for value in by_subject.values() if value is not None]
            across_subjects[field] = {
                **_statistics(finite),
                "per_subject_seed_mean": by_subject,
            }
        result[identifier] = {
            "window_count": window_count,
            "vote_threshold": vote_threshold,
            "per_seed_subject_macro": per_seed_subject_macro,
            "aggregate_across_seeds": aggregate_across_seeds,
            "across_subjects_from_seed_means": across_subjects,
        }
    return result


# ---------- 人类可读矩阵：N=1 仅作为冻结参考，N=2..5 是本轮全量诊断网格 ----------
def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_matrix_csvs(output_root: Path, summary: dict, single_reference: dict) -> dict:
    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    def append_policy(source: str, identifier: str, n: int, k: int, payload: dict) -> None:
        for seed in KNOWN_SEEDS:
            metrics = payload["per_seed_subject_macro"][str(seed)]
            per_seed_rows.append({
                "source": source, "policy_id": identifier,
                "window_count": n, "vote_threshold": k, "seed": seed,
                **{field: metrics[field]["mean"] for field in CORE_FIELDS},
            })
        aggregate_rows.append({
            "source": source, "policy_id": identifier,
            "window_count": n, "vote_threshold": k,
            **{
                f"{field}_{suffix}": payload["aggregate_across_seeds"][field][key]
                for field in CORE_FIELDS
                for suffix, key in (("mean", "mean"), ("population_std", "population_std"))
            },
        })

    append_policy("frozen_single_window_reference", "n1_k1", 1, 1, single_reference)
    for window_count, vote_threshold in VOTE_GRID:
        identifier = policy_id(window_count, vote_threshold)
        append_policy("hard_vote_matrix", identifier, window_count, vote_threshold, summary[identifier])

    per_seed_path = output_root / "hard_vote_matrix_per_seed.csv"
    aggregate_path = output_root / "hard_vote_matrix_aggregate.csv"
    _atomic_csv(
        per_seed_path,
        ["source", "policy_id", "window_count", "vote_threshold", "seed", *CORE_FIELDS],
        per_seed_rows,
    )
    aggregate_fields = ["source", "policy_id", "window_count", "vote_threshold"]
    for field in CORE_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_population_std"])
    _atomic_csv(aggregate_path, aggregate_fields, aggregate_rows)
    return {
        "per_seed_csv": {"file": per_seed_path.name, "sha256": file_hash(per_seed_path)},
        "aggregate_csv": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


def verify_matrix_child(child_root: Path, manifest: dict, subject: int) -> None:
    """顶层签名前重新读取子清单，复核日志与三 seed 的全部落盘文件。"""
    if (
        manifest.get("status") != "PASS"
        or manifest.get("subject") != subject
        or tuple(manifest.get("seeds", [])) != KNOWN_SEEDS
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or tuple(tuple(item) for item in manifest.get("vote_grid", [])) != VOTE_GRID
    ):
        raise RuntimeError(f"Subject {subject} 硬投票子清单合同非法")
    artifacts = [manifest.get("run_log", {})]
    for seed in KNOWN_SEEDS:
        roles = manifest.get("seed_artifacts", {}).get(str(seed), {})
        if set(roles) != {"input_scores", "metrics", "trajectories"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 产物角色不完整")
        artifacts.extend([roles["metrics"], roles["trajectories"]])
    for artifact in artifacts:
        path = _safe_artifact(child_root, artifact.get("file", ""))
        if not path.is_file() or file_hash(path) != artifact.get("sha256"):
            raise RuntimeError(f"Subject {subject} 硬投票子产物哈希不一致")


# ---------- 主入口：全部 cell 均报告，不在同一 OOF 矩阵上选择并宣称最优 ----------
def run(args: argparse.Namespace) -> dict:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.policy_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = _read_json(config_path)
    validate_policy_config(config)
    input_master, input_children = verify_input_root(input_root)
    run_git = git_state()
    run_environment = runtime_environment()
    claim_status = (
        "PRECOMMIT_DIAGNOSTIC_MATRIX"
        if run_git["dirty"] is not False
        else "CLEAN_COMMIT_DIAGNOSTIC_MATRIX"
    )

    children: dict[str, dict] = {}
    subject_summaries: dict[int, dict] = {}
    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=True)
        context, inventory, contract, paths = _load_subject_inventory(subject)
        expected_rows = output_window_rows(inventory.windows)
        input_child_path = _safe_artifact(
            input_root, input_master["children"][str(subject)]["manifest"],
        )
        input_child = input_children[subject]
        if (
            input_child.get("inputs", {}).get("bundle_manifest")
            != display_path(paths.bundle_manifest)
            or input_child.get("inputs", {}).get("bundle_manifest_sha256")
            != context.manifest_sha256
        ):
            raise RuntimeError(f"Subject {subject} 输入分数与当前 session0 bundle 不匹配")

        seed_results: dict[int, dict[str, dict]] = {}
        seed_artifacts: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = _load_seed_logits(
                input_child_path.parent, input_child, seed, expected_rows,
            )
            result, artifacts = _save_seed_matrix(
                child_root, inventory, contract, seed, stage1, stage2, input_scores,
            )
            seed_results[seed] = result
            seed_artifacts[str(seed)] = artifacts

        subject_summary = summarize_subject(seed_results)
        subject_summaries[subject] = subject_summary
        subject_completed = datetime.now(timezone.utc).isoformat()
        log_path = child_root / "run_log.json"
        atomic_json(log_path, {
            "status": "PASS",
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "claim_status": claim_status,
            "subject": subject,
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "started_at_utc": subject_started,
            "completed_at_utc": subject_completed,
            "input_child_manifest": display_path(input_child_path),
            "input_child_manifest_sha256": file_hash(input_child_path),
            "inventory_contract": display_path(paths.inventory_contract),
            "inventory_contract_sha256": file_hash(paths.inventory_contract),
            "vote_grid": [list(item) for item in VOTE_GRID],
            "runtime_environment": run_environment,
            "seed_artifacts": seed_artifacts,
        })
        manifest = {
            "status": "PASS",
            "claim_status": claim_status,
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "selection_status": "none_all_cells_reported",
            "subject": subject,
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "seeds": list(KNOWN_SEEDS),
            "vote_grid": [list(item) for item in VOTE_GRID],
            "policy_contract": config,
            "policy_contract_sha256": file_hash(config_path),
            "input_child_manifest": {
                "file": display_path(input_child_path),
                "sha256": file_hash(input_child_path),
                "source_git": input_child.get("runtime", {}).get("git"),
            },
            "inventory_contract": contract,
            "seed_artifacts": seed_artifacts,
            "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
            "summary": subject_summary,
            "source_sha256": {
                "hard_vote_policy": file_hash(EVAL_DIR / "hard_vote_policy.py"),
                "matrix_runner": file_hash(Path(__file__)),
                "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
                "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
                "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
            },
            "runtime": {
                "started_at_utc": subject_started,
                "completed_at_utc": subject_completed,
                "environment": run_environment,
                "git": run_git,
            },
        }
        manifest_path = child_root / "run_manifest.json"
        atomic_json(manifest_path, manifest)
        disk_manifest = _read_json(manifest_path)
        verify_matrix_child(child_root, disk_manifest, subject)
        children[str(subject)] = {
            "manifest": str(manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(manifest_path),
            "window_count": contract["inventory"]["window_count"],
            "event_count": contract["inventory"]["event_count"],
            "claim_status": claim_status,
        }
        print(f"Subject {subject}: PASS", flush=True)

    summary = aggregate_subject_matrix(subject_summaries)
    single_reference = input_master["summary"][STATEFUL_STRICT]
    csv_artifacts = write_matrix_csvs(output_root, summary, single_reference)
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    if git_state() != run_git:
        raise RuntimeError("硬投票运行期间 Git 身份发生变化")
    if runtime_environment() != run_environment:
        raise RuntimeError("硬投票运行期间解释器环境发生变化")
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "claim_status": claim_status,
        "selection_status": "none_all_cells_reported",
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime_environment": run_environment,
        "input_master_manifest": display_path(input_root / "run_manifest.json"),
        "input_master_manifest_sha256": file_hash(input_root / "run_manifest.json"),
        "children": children,
        "csv_artifacts": csv_artifacts,
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "selection_status": "none_all_cells_reported",
        "selection_warning": (
            "The matrix is descriptive. Selecting a cell and reporting the same OOF value "
            "as unbiased performance is forbidden."
        ),
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "input_master_manifest": {
            "file": display_path(input_root / "run_manifest.json"),
            "sha256": file_hash(input_root / "run_manifest.json"),
            "protocol_id": input_master["protocol_id"],
            "claim_status": input_master["claim_status"],
            "source_git": input_master["runtime_git"],
        },
        "policy_contract": config,
        "policy_contract_file": display_path(config_path),
        "policy_contract_sha256": file_hash(config_path),
        "aggregation_semantics": {
            "event_pooling_across_subjects": "forbidden",
            "primary_summary": "equal_subject_macro_within_paired_seed",
            "seed_summary": "mean_and_population_std_across_three_paired_seeds",
        },
        "single_window_reference": single_reference,
        "children": children,
        "csv_artifacts": csv_artifacts,
        "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
        "summary": summary,
        "source_sha256": {
            "hard_vote_policy": file_hash(EVAL_DIR / "hard_vote_policy.py"),
            "matrix_runner": file_hash(Path(__file__)),
            "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
            "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
            "single_window_multi_subject_runner": file_hash(
                EVAL_DIR / "run_epoch50_online_oof_all_subjects.py"
            ),
            "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
        },
        "runtime_environment": run_environment,
        "runtime_git": run_git,
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
