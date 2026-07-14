from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from commit_reset_diagnostics import diagnose_commit_reset
from logit_candidate_strategies import LogitStrategyConfig, logit_candidate_decisions
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    IDLE_RESET,
    READY,
    STATEFUL_CANDIDATE,
    TASK_CANDIDATE,
    WAIT_IDLE,
    evaluate_online_events,
)
from run_candidate_logit_matrix import (
    CANDIDATE_FIELDS,
    REASON_CODE,
    STATE_CODE,
    _check_metric_inventory,
    _strategy_metrics,
)
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    atomic_json,
    atomic_npz,
    display_path,
    file_hash,
    git_state,
    output_window_rows,
)
from run_hard_vote_matrix import (
    CORE_FIELDS,
    EXPECTED_INPUT_PROTOCOL,
    _atomic_csv,
    _load_seed_logits,
    _load_subject_inventory,
    _read_json,
    _safe_artifact,
    _statistics,
    runtime_environment,
    verify_input_root,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_commit_reset_matrix_v1"
)
DEFAULT_POLICY_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_commit_reset_matrix_v1.json"
)
EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_commit_reset_matrix_v1"
EXPECTED_PARAMETER_ORIGIN = (
    "fixed diagnostic grid after reviewing session0 OOF state trajectories"
)
EXPECTED_SEMANTICS = {
    "factorization": (
        "stage2 commit threshold crossed with independent stage1 WAIT_IDLE reset "
        "threshold and confirmation length"
    ),
    "stage1_score": "sigmoid of alpha0.5 EWMA over task_logit-minus-idle_logit",
    "stage2_score": (
        "softmax of candidate-local alpha0.5 EWMA over per-window centered logits"
    ),
    "candidate_history": (
        "excludes opening window and clears on every candidate exit and segment boundary"
    ),
    "reset_history": (
        "counts qualifying windows only while WAIT_IDLE and clears on failure, reset, "
        "or segment boundary"
    ),
    "decision_time": (
        "window stop at native 250 Hz with 500-sample windows and 125-sample steps"
    ),
    "truth_usage": (
        "labels and event boundaries are used only after inference for metrics and timing attribution"
    ),
}
EXPECTED_BASE = {
    "stage1_filter": "ewma_margin",
    "stage1_alpha": 0.5,
    "stage1_window": None,
    "task_on_probability": 0.5,
    "task_hold_probability": 0.3,
    "stage1_drop_abort": 0.2,
    "stage2_filter": "candidate_ewma_centered_logits",
    "stage2_alpha": 0.5,
    "stage2_min_candidate_windows": 2,
    "stage2_probability_gap": 0.15,
    "stage2_stable_windows": 1,
    "stage2_max_probability_curvature": None,
    "max_candidate_windows": 8,
}
EXPECTED_COMMITS = (("c055", 0.55), ("c0625", 0.625), ("c070", 0.70))
EXPECTED_RESETS = (
    ("r020_l1", 0.20, 1), ("r020_l2", 0.20, 2),
    ("r030_l1", 0.30, 1), ("r030_l2", 0.30, 2),
    ("r040_l1", 0.40, 1), ("r040_l2", 0.40, 2),
)
DIAGNOSTIC_FIELDS = (
    "wait_idle_mean_seconds",
    "wait_idle_median_seconds",
    "wait_idle_p90_seconds",
    "wait_idle_segment_end_unresolved_count",
    "reset_relative_event_offset_median_seconds",
    "reset_relative_event_offset_p90_seconds",
    "premature_reset_rate",
    "matched_command_without_reset_rate",
    "first_eligible_window_wait_idle_rate",
    "fully_wait_idle_event_rate",
    "fully_wait_idle_among_miss_rate",
    "post_mi_spillover_per_valid_idle_minute",
    "other_idle_false_per_valid_idle_minute",
)
SUMMARY_FIELDS = (*CORE_FIELDS, *CANDIDATE_FIELDS, *DIAGNOSTIC_FIELDS)


@dataclass(frozen=True)
class CommitResetCell:
    """把一个提交工作点和一个复位工作点显式绑定为矩阵 cell。"""

    cell_id: str
    commit_profile_id: str
    reset_profile_id: str
    stage2_top_probability: float
    idle_reset_probability: float
    idle_reset_consecutive_windows: int
    logit_config: LogitStrategyConfig

    def public_config(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "commit_profile_id": self.commit_profile_id,
            "reset_profile_id": self.reset_profile_id,
            "stage2_top_probability": self.stage2_top_probability,
            "idle_reset_probability": self.idle_reset_probability,
            "idle_reset_consecutive_windows": self.idle_reset_consecutive_windows,
            "fixed_logit_strategy": asdict(self.logit_config),
        }


# ---------- 配置合同：18 个 cell 必须是 3 个提交点与 6 个复位点的完整笛卡尔积 ----------
def load_commit_reset_contract(path: Path) -> tuple[dict, tuple[CommitResetCell, ...]]:
    payload = _read_json(path)
    if set(payload) != {
        "protocol_id", "input_protocol_id", "included_session", "test_session_access",
        "selection_status", "parameter_origin", "strategy_semantics", "base_strategy",
        "commit_profiles", "reset_profiles",
    }:
        raise RuntimeError("提交-复位矩阵配置字段不完整")
    if (
        payload["protocol_id"] != EXPECTED_OUTPUT_PROTOCOL
        or payload["input_protocol_id"] != EXPECTED_INPUT_PROTOCOL
        or payload["included_session"] != 0
        or payload["test_session_access"] != "forbidden"
        or payload["selection_status"] != "none_all_cells_reported"
        or payload["parameter_origin"] != EXPECTED_PARAMETER_ORIGIN
        or payload["strategy_semantics"] != EXPECTED_SEMANTICS
        or payload["base_strategy"] != EXPECTED_BASE
    ):
        raise RuntimeError("提交-复位矩阵配置与冻结诊断协议不一致")
    commits = tuple(
        (item.get("profile_id"), item.get("stage2_top_probability"))
        for item in payload["commit_profiles"]
        if isinstance(item, dict) and set(item) == {"profile_id", "stage2_top_probability"}
    )
    resets = tuple(
        (
            item.get("profile_id"), item.get("idle_reset_probability"),
            item.get("consecutive_windows"),
        )
        for item in payload["reset_profiles"]
        if isinstance(item, dict)
        and set(item) == {"profile_id", "idle_reset_probability", "consecutive_windows"}
    )
    if any(type(item.get("consecutive_windows")) is not int for item in payload["reset_profiles"]):
        raise RuntimeError("连续复位窗数必须是显式整数")
    if commits != EXPECTED_COMMITS or resets != EXPECTED_RESETS:
        raise RuntimeError("提交或复位轴不是冻结的 3x6 网格")

    cells: list[CommitResetCell] = []
    for commit_id, top_probability in commits:
        for reset_id, reset_probability, consecutive in resets:
            cell_id = f"{commit_id}_{reset_id}"
            config_payload = {
                "strategy_id": cell_id,
                **payload["base_strategy"],
                "idle_reset_probability": reset_probability,
                "stage2_top_probability": top_probability,
            }
            config = LogitStrategyConfig.from_dict(config_payload)
            cells.append(CommitResetCell(
                cell_id,
                commit_id,
                reset_id,
                float(top_probability),
                float(reset_probability),
                int(consecutive),
                config,
            ))
    if len(cells) != 18 or len({item.cell_id for item in cells}) != 18:
        raise RuntimeError("提交-复位矩阵必须生成 18 个唯一 cell")
    return payload, tuple(cells)


def _diagnostic_summary(diagnostics: dict) -> dict[str, float | None]:
    wait = diagnostics["wait_idle"]
    reset = diagnostics["reset_relative_to_matched_event_offset"]
    lock = diagnostics["event_lock"]
    false = diagnostics["idle_false_attribution"]
    matched = reset["matched_command_count"]
    return {
        "wait_idle_mean_seconds": wait["completed_duration_seconds"]["mean"],
        "wait_idle_median_seconds": wait["completed_duration_seconds"]["median"],
        "wait_idle_p90_seconds": wait["completed_duration_seconds"]["p90"],
        "wait_idle_segment_end_unresolved_count": wait["segment_end_unresolved_count"],
        "reset_relative_event_offset_median_seconds": reset["seconds"]["median"],
        "reset_relative_event_offset_p90_seconds": reset["seconds"]["p90"],
        "premature_reset_rate": reset["premature_reset_rate"],
        "matched_command_without_reset_rate": (
            None if not matched else reset["matched_command_without_reset_count"] / matched
        ),
        "first_eligible_window_wait_idle_rate": lock["first_eligible_window_wait_idle_rate"],
        "fully_wait_idle_event_rate": lock["fully_wait_idle_event_rate"],
        "fully_wait_idle_among_miss_rate": lock["fully_wait_idle_among_miss_rate"],
        "post_mi_spillover_per_valid_idle_minute": (
            false["post_mi_spillover_per_valid_idle_minute"]
        ),
        "other_idle_false_per_valid_idle_minute": (
            false["other_idle_false_per_valid_idle_minute"]
        ),
    }


# ---------- 单 seed：同一份冻结 logits 依次运行 18 个 cell，并保存可独立复算的轨迹 ----------
def _save_seed_matrix(
    output_root: Path,
    inventory,
    inventory_contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
    cells: tuple[CommitResetCell, ...],
) -> tuple[dict[str, dict], dict]:
    arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "cell_ids": np.asarray([item.cell_id for item in cells]),
        "state_code_names": np.asarray([READY, TASK_CANDIDATE, WAIT_IDLE]),
        "reason_code_names": np.asarray([
            "none", CANDIDATE_OPEN, CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT, COMMAND_COMMIT, IDLE_RESET,
        ]),
    }
    payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "selection_status": "none_all_cells_reported",
        "cells": {},
    }
    summary: dict[str, dict] = {}
    for cell in cells:
        strategy = logit_candidate_decisions(
            inventory.windows,
            stage1_logits,
            stage2_logits,
            cell.logit_config,
            idle_reset_consecutive_windows=cell.idle_reset_consecutive_windows,
        )
        evaluated = evaluate_online_events(
            inventory.segments,
            inventory.events,
            inventory.windows,
            strategy.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        _check_metric_inventory(evaluated, inventory_contract)
        diagnostics = diagnose_commit_reset(
            inventory.segments,
            inventory.events,
            inventory.windows,
            strategy.policy.decisions,
            evaluated,
        )
        payload["cells"][cell.cell_id] = {
            "config": cell.public_config(),
            "metrics": evaluated,
            "commit_reset_diagnostics": diagnostics,
        }
        summary[cell.cell_id] = {
            **_strategy_metrics(evaluated),
            **_diagnostic_summary(diagnostics),
        }

        decisions, policy_trace, score_trace = (
            strategy.policy.decisions, strategy.policy.trace, strategy.trace,
        )
        identifier = cell.cell_id
        for name, values, dtype in (
            ("emitted", [item.emitted_class for item in decisions], np.int8),
            ("before", [STATE_CODE[item.decision_state_before] for item in decisions], np.uint8),
            ("after", [STATE_CODE[item.decision_state_after] for item in decisions], np.uint8),
            ("reason", [REASON_CODE[item.transition_reason] for item in decisions], np.uint8),
            ("candidate_age_before", [item.candidate_windows_before for item in policy_trace], np.int64),
            ("candidate_age_after", [item.candidate_windows_after for item in policy_trace], np.int64),
            ("idle_reset", [item.evidence.idle_reset for item in score_trace], np.bool_),
            ("idle_reset_raw", [item.idle_reset_raw_condition for item in score_trace], np.bool_),
            ("idle_reset_streak", [item.idle_reset_consecutive_count for item in score_trace], np.int64),
            ("stage2_commit_class", [item.evidence.stage2_commit_class for item in score_trace], np.int8),
            ("stage2_candidate_count", [item.stage2_candidate_window_count for item in score_trace], np.int64),
        ):
            arrays[f"{identifier}_{name}"] = np.asarray(values, dtype=dtype)
        for name in (
            "stage1_filtered_task_probability", "stage1_filtered_delta",
            "stage2_top_probability", "stage2_probability_gap",
        ):
            arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item, name) for item in score_trace], dtype=np.float32,
            )

    metrics_path = output_root / f"seed{seed}_commit_reset_metrics.json"
    trajectory_path = output_root / f"seed{seed}_commit_reset_trajectories.npz"
    atomic_json(metrics_path, payload)
    atomic_npz(trajectory_path, **arrays)
    return summary, {
        "input_scores": input_scores,
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
    }


# ---------- 分层汇总：先同 seed 九被试等权，再汇总三个配对 seed ----------
def _aggregate_subjects(subject_summaries: dict[int, dict], cells: tuple[CommitResetCell, ...]) -> dict:
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("提交-复位矩阵汇总缺少 Subject 1-9")
    result: dict[str, dict] = {}
    for cell in cells:
        per_seed: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            row = {}
            for field in SUMMARY_FIELDS:
                values = [
                    subject_summaries[subject][str(seed)][cell.cell_id][field]
                    for subject in KNOWN_SUBJECTS
                    if subject_summaries[subject][str(seed)][cell.cell_id][field] is not None
                ]
                row[field] = {
                    "mean": None if not values else float(np.mean(values)),
                    "valid_subject_count": len(values),
                }
            per_seed[str(seed)] = row
        aggregate = {
            field: _statistics([
                per_seed[str(seed)][field]["mean"]
                for seed in KNOWN_SEEDS
                if per_seed[str(seed)][field]["mean"] is not None
            ])
            for field in SUMMARY_FIELDS
        }
        result[cell.cell_id] = {
            "config": cell.public_config(),
            "per_seed_subject_macro": per_seed,
            "aggregate_across_seeds": aggregate,
        }
    return result


def _write_csvs(output_root: Path, summary: dict) -> dict:
    config_fields = [
        "cell_id", "commit_profile_id", "reset_profile_id",
        "stage2_top_probability", "idle_reset_probability",
        "idle_reset_consecutive_windows",
    ]
    seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for cell_id, item in summary.items():
        config = {name: item["config"][name] for name in config_fields}
        for seed in KNOWN_SEEDS:
            seed_rows.append({
                **config,
                "seed": seed,
                **{
                    field: item["per_seed_subject_macro"][str(seed)][field]["mean"]
                    for field in SUMMARY_FIELDS
                },
            })
        aggregate_rows.append({
            **config,
            **{
                f"{field}_{suffix}": item["aggregate_across_seeds"][field][key]
                for field in SUMMARY_FIELDS
                for suffix, key in (("mean", "mean"), ("population_std", "population_std"))
            },
        })
    per_seed_path = output_root / "commit_reset_matrix_per_seed.csv"
    aggregate_path = output_root / "commit_reset_matrix_aggregate.csv"
    _atomic_csv(
        per_seed_path,
        [*config_fields, "seed", *SUMMARY_FIELDS],
        seed_rows,
    )
    aggregate_fields = list(config_fields)
    for field in SUMMARY_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_population_std"])
    _atomic_csv(aggregate_path, aggregate_fields, aggregate_rows)
    return {
        "per_seed_csv": {"file": per_seed_path.name, "sha256": file_hash(per_seed_path)},
        "aggregate_csv": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


def _verify_child(
    root: Path,
    manifest: dict,
    subject: int,
    cell_ids: tuple[str, ...],
    policy_sha256: str,
    source_hashes: dict[str, str],
) -> None:
    if (
        manifest.get("status") != "PASS"
        or manifest.get("subject") != subject
        or tuple(manifest.get("seeds", [])) != KNOWN_SEEDS
        or tuple(manifest.get("cell_ids", [])) != cell_ids
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or manifest.get("policy_contract_sha256") != policy_sha256
        or manifest.get("source_sha256") != source_hashes
    ):
        raise RuntimeError(f"Subject {subject} 提交-复位子清单合同非法")
    artifacts = [manifest["run_log"]]
    for seed in KNOWN_SEEDS:
        roles = manifest["seed_artifacts"].get(str(seed), {})
        if set(roles) != {"input_scores", "metrics", "trajectories"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 产物角色不完整")
        artifacts.extend([roles["metrics"], roles["trajectories"]])
    for artifact in artifacts:
        path = _safe_artifact(root, artifact["file"])
        if not path.is_file() or file_hash(path) != artifact["sha256"]:
            raise RuntimeError(f"Subject {subject} 提交-复位子产物哈希不一致")


# ---------- 主入口：只消费已冻结 session0 OOF 分数，全部 18 个 cell 并列报告 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.policy_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    policy_sha256 = file_hash(config_path)
    policy_contract, cells = load_commit_reset_contract(config_path)
    if file_hash(config_path) != policy_sha256:
        raise RuntimeError("读取提交-复位配置期间文件发生变化")
    input_master, input_children = verify_input_root(input_root)
    source_hashes = _source_hashes()
    run_git = git_state()
    environment = runtime_environment()
    claim_status = (
        "PRECOMMIT_DIAGNOSTIC_MATRIX"
        if run_git["dirty"] is not False
        else "CLEAN_COMMIT_DIAGNOSTIC_MATRIX"
    )
    cell_ids = tuple(item.cell_id for item in cells)
    children: dict[str, dict] = {}
    subject_summaries: dict[int, dict] = {}

    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=True)
        context, inventory, inventory_contract, paths = _load_subject_inventory(subject)
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
            raise RuntimeError(f"Subject {subject} 输入分数与 session0 bundle 不匹配")
        expected_rows = output_window_rows(inventory.windows)
        subject_summaries[subject] = {}
        seed_artifacts: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = _load_seed_logits(
                input_child_path.parent, input_child, seed, expected_rows,
            )
            summary, artifacts = _save_seed_matrix(
                child_root,
                inventory,
                inventory_contract,
                seed,
                stage1,
                stage2,
                input_scores,
                cells,
            )
            subject_summaries[subject][str(seed)] = summary
            seed_artifacts[str(seed)] = artifacts

        completed = datetime.now(timezone.utc).isoformat()
        log_path = child_root / "run_log.json"
        atomic_json(log_path, {
            "status": "PASS",
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "claim_status": claim_status,
            "subject": subject,
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "cell_ids": list(cell_ids),
            "started_at_utc": subject_started,
            "completed_at_utc": completed,
            "runtime_environment": environment,
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
            "cell_ids": list(cell_ids),
            "policy_contract_sha256": policy_sha256,
            "input_child_manifest": {
                "file": display_path(input_child_path),
                "sha256": file_hash(input_child_path),
            },
            "inventory_contract": inventory_contract,
            "seed_artifacts": seed_artifacts,
            "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
            "source_sha256": source_hashes,
            "runtime": {
                "started_at_utc": subject_started,
                "completed_at_utc": completed,
                "environment": environment,
                "git": run_git,
            },
        }
        manifest_path = child_root / "run_manifest.json"
        atomic_json(manifest_path, manifest)
        _verify_child(
            child_root, _read_json(manifest_path), subject,
            cell_ids, policy_sha256, source_hashes,
        )
        children[str(subject)] = {
            "manifest": str(manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(manifest_path),
            "window_count": inventory_contract["inventory"]["window_count"],
            "event_count": inventory_contract["inventory"]["event_count"],
        }
        print(f"Subject {subject}: PASS", flush=True)

    summary = _aggregate_subjects(subject_summaries, cells)
    csv_artifacts = _write_csvs(output_root, summary)
    completed_at = datetime.now(timezone.utc).isoformat()
    if (
        git_state() != run_git
        or runtime_environment() != environment
        or _source_hashes() != source_hashes
        or file_hash(config_path) != policy_sha256
    ):
        raise RuntimeError("矩阵运行期间 Git、源码、配置或解释器身份发生变化")
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "claim_status": claim_status,
        "selection_status": "none_all_cells_reported",
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "cell_ids": list(cell_ids),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_environment": environment,
        "children": children,
        "csv_artifacts": csv_artifacts,
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "selection_status": "none_all_cells_reported",
        "selection_warning": (
            "This session0 OOF matrix is descriptive; selecting and reporting the same "
            "cell as unbiased performance is forbidden."
        ),
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "cell_ids": list(cell_ids),
        "input_master_manifest": {
            "file": display_path(input_root / "run_manifest.json"),
            "sha256": file_hash(input_root / "run_manifest.json"),
            "protocol_id": input_master["protocol_id"],
            "claim_status": input_master["claim_status"],
        },
        "policy_contract": policy_contract,
        "policy_contract_file": display_path(config_path),
        "policy_contract_sha256": policy_sha256,
        "aggregation_semantics": {
            "event_pooling_across_subjects": "forbidden",
            "primary_summary": "equal_subject_macro_within_paired_seed",
            "seed_summary": "mean_and_population_std_across three paired seeds",
            "factor_comparison": "hold one axis fixed when interpreting the other",
        },
        "children": children,
        "csv_artifacts": csv_artifacts,
        "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
        "summary": summary,
        "source_sha256": source_hashes,
        "runtime_environment": environment,
        "runtime_git": run_git,
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def _source_hashes() -> dict[str, str]:
    return {
        "candidate_state_policy": file_hash(EVAL_DIR / "candidate_state_policy.py"),
        "logit_candidate_strategies": file_hash(EVAL_DIR / "logit_candidate_strategies.py"),
        "commit_reset_diagnostics": file_hash(EVAL_DIR / "commit_reset_diagnostics.py"),
        "commit_reset_matrix_runner": file_hash(Path(__file__)),
        "candidate_logit_matrix_helpers": file_hash(EVAL_DIR / "run_candidate_logit_matrix.py"),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "frozen_input_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
        # frozen_input_reader 会调用这里定义的子清单验证器，必须一并冻结源码身份。
        "single_window_multi_subject_verifier": file_hash(
            EVAL_DIR / "run_epoch50_online_oof_all_subjects.py"
        ),
        "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
        "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
