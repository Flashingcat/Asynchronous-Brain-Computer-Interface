"""在九被试三种子冻结 OOF logits 上运行完整 Oracle 上限诊断。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import scipy

from logit_candidate_strategies import logit_candidate_decisions
from oracle_ceiling_diagnostics import (
    MODEL_CELL_ID,
    ORACLE_CELLS,
    component_oracle_replays,
    shapley_component_contributions,
    stage2_evidence_ceiling,
)
from protocol_metrics import STATEFUL_CANDIDATE, evaluate_online_events
from run_candidate_logit_matrix import (
    REASON_CODE,
    STATE_CODE,
    _check_metric_inventory,
)
from run_commit_reset_matrix import load_commit_reset_contract
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    _build_online_signal_inventory,
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
from online_truth_inventory import load_truth_inventory
from run_hard_vote_matrix import (
    CORE_FIELDS,
    EXPECTED_INPUT_PROTOCOL,
    _atomic_csv,
    _load_seed_logits,
    _read_json,
    _safe_artifact,
    _statistics,
    runtime_environment,
    verify_input_root,
)
from oof_training_bundle import (
    ARTIFACT_POLICY,
    SEGMENT_POLICY,
    artifact_contract,
    load_bundle,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_oracle_ceiling_v1"
)
DEFAULT_ORACLE_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_oracle_ceiling_v1.json"
)
DEFAULT_ANCHOR_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_commit_reset_matrix_v1.json"
)
EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_oracle_ceiling_v1"
EXPECTED_TRUTH_LOAD_PHASE = "after_all_frozen_logits_and_model_only_anchor_traces"
ANCHOR_CELL_ID = "c055_r020_l1"


EXPECTED_CONTRACT = {
    "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
    "input_protocol_id": EXPECTED_INPUT_PROTOCOL,
    "included_session": 0,
    "test_session_access": "forbidden",
    "selection_status": "diagnostic_only_no_policy_selection",
    "anchor_cell": ANCHOR_CELL_ID,
    "truth_load_phase": EXPECTED_TRUTH_LOAD_PHASE,
    "component_axes": {
        "stage1": {
            "model": "anchor Stage 1 EWMA opening and holding evidence",
            "truth_oracle": "Task on every officially eligible event window; otherwise IDLE",
        },
        "commit": {
            "model": (
                "commit the first legal candidate-local Stage 2 crossing with at least two "
                "windows, top probability at least 0.55, gap at least 0.15, and current raw "
                "top1 equal to aggregate top1"
            ),
            "truth_oracle": (
                "dynamic program may skip crossings and commit only the same legal crossing "
                "whose model aggregate top1 equals the event true class"
            ),
        },
        "reset": {
            "model": "anchor Stage 1 Task probability at most 0.20 for one WAIT_IDLE window",
            "truth_oracle": (
                "first WAIT_IDLE window whose decision time is strictly after true MI end; "
                "an idle false command may reset on its following true-IDLE window"
            ),
        },
    },
    "dynamic_program_objective": [
        "maximize correctly submitted unique events over each full segment",
        "minimize total correct-event latency in native samples",
        "minimize emitted commands",
        "choose lexicographically earlier command windows",
    ],
    "stage2_evidence_ceilings": {
        "raw_top1": (
            "any officially eligible raw Stage 2 window has fixed-order top1 equal to truth"
        ),
        "any_start_causal_ewma_top1": (
            "alpha 0.5 causal EWMA from any officially eligible start has fixed-order top1 "
            "equal to truth"
        ),
        "convex_logit_top1": (
            "some nonnegative sum-to-one weighting of the eligible prefix makes truth a co-top class"
        ),
    },
    "event_matching": {
        "native_sampling_rate": 250,
        "window_samples": 500,
        "step_samples": 125,
        "minimum_overlap_samples": 125,
        "window_stop_must_not_exceed_event_offset": True,
    },
    "aggregation": (
        "equal-subject macro within each paired seed, then mean and population standard "
        "deviation across seeds 42 43 44"
    ),
    "claim_limit": (
        "truth-aware post-inference upper-bound diagnostic; not deployable performance, FAR, "
        "or a policy-selection result"
    ),
}

EVIDENCE_IDS = (
    "raw_top1",
    "any_start_causal_ewma_top1",
    "raw_or_any_start_ewma_top1",
    "convex_logit_top1",
)
EVIDENCE_VALUE_FIELDS = tuple(
    f"{identifier}_{field}"
    for identifier in EVIDENCE_IDS
    for field in (
        "coverage_rate",
        "gain_over_baseline_correct_event_rate",
        "recoverable_baseline_miss_rate",
        "recoverable_baseline_wrong_rate",
        "earliest_latency_mean_seconds",
        "earliest_latency_median_seconds",
    )
)
EVIDENCE_FIELDS = (
    "baseline_correct_event_rate",
    *EVIDENCE_VALUE_FIELDS,
    "convex_only_beyond_any_top1_rate",
)
SHAPLEY_FIELDS = (
    "baseline_correct_event_rate",
    "all_truth_correct_event_rate",
    "total_oracle_gap",
    "stage1_contribution",
    "commit_contribution",
    "reset_contribution",
)


def load_oracle_contract(path: Path) -> dict:
    """配置逐字段逐值冻结，Oracle 口径变化必须显式升级协议版本。"""
    payload = _read_json(path)
    if payload != EXPECTED_CONTRACT:
        raise RuntimeError("Oracle 配置与冻结 v1 合同不一致")
    return payload


def load_anchor(path: Path):
    """只接受提交-复位矩阵中明确命名的 c055/r020/l1 历史锚点。"""
    _, cells = load_commit_reset_contract(path)
    matches = [cell for cell in cells if cell.cell_id == ANCHOR_CELL_ID]
    if len(matches) != 1:
        raise RuntimeError("Oracle 锚点必须唯一对应 c055/r020/l1")
    return matches[0]


def _compact_evaluation(result: dict) -> dict:
    """保存正式指标、库存哈希和逐事件匹配；大体积候选区间明细留在轨迹 NPZ。"""
    return {
        "metrics": core_metrics(result),
        "scorable_event_count": result["scorable_event_count"],
        "correct_event_count": result["correct_event_count"],
        "emitted_command_count": result["emitted_command_count"],
        "idle_false_command_count": result["idle_false_command_count"],
        "too_early_command_count": result["too_early_command_count"],
        "additional_event_command_count": result["additional_event_command_count"],
        "scoring_segment_inventory_sha256": result["scoring_segment_inventory_sha256"],
        "expected_window_inventory_sha256": result["expected_window_inventory_sha256"],
        "event_inventory_sha256": result["event_inventory_sha256"],
        "decision_inventory_sha256": result["decision_inventory_sha256"],
        "event_matches": result["event_matches"],
    }


def _flat_evidence(summary: dict) -> dict[str, float | None]:
    row: dict[str, float | None] = {
        "baseline_correct_event_rate": summary["baseline_correct_event_rate"],
        "convex_only_beyond_any_top1_rate": (
            summary["convex_only_beyond_any_top1_count"] / summary["event_count"]
        ),
    }
    for identifier in EVIDENCE_IDS:
        item = summary[identifier]
        row[f"{identifier}_coverage_rate"] = item["coverage_rate"]
        row[f"{identifier}_gain_over_baseline_correct_event_rate"] = (
            item["gain_over_baseline_correct_event_rate"]
        )
        row[f"{identifier}_recoverable_baseline_miss_rate"] = (
            item["recoverable_baseline_miss_rate"]
        )
        row[f"{identifier}_recoverable_baseline_wrong_rate"] = (
            item["recoverable_baseline_wrong_rate"]
        )
        row[f"{identifier}_earliest_latency_mean_seconds"] = (
            item["earliest_latency_seconds"]["mean"]
        )
        row[f"{identifier}_earliest_latency_median_seconds"] = (
            item["earliest_latency_seconds"]["median"]
        )
    return row


def _flat_shapley(payload: dict) -> dict[str, float]:
    return {
        "baseline_correct_event_rate": payload["baseline_value"],
        "all_truth_correct_event_rate": payload["all_truth_value"],
        "total_oracle_gap": payload["total_oracle_gap"],
        "stage1_contribution": payload["contributions"]["stage1"],
        "commit_contribution": payload["contributions"]["commit"],
        "reset_contribution": payload["contributions"]["reset"],
    }


# ---------- 单 seed：八条完整轨迹、事件证据上限、Shapley 分解一次性闭合 ----------
def _run_seed(
    output_root: Path,
    inventory,
    inventory_contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
    anchor_cell,
    pretruth_anchor,
) -> tuple[dict, dict]:
    replays = component_oracle_replays(
        inventory.windows,
        inventory.events,
        stage1_logits,
        stage2_logits,
        anchor_cell.logit_config,
        idle_reset_consecutive_windows=anchor_cell.idle_reset_consecutive_windows,
        pretruth_anchor=pretruth_anchor,
    )
    evaluations: dict[str, dict] = {}
    compact: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "cell_ids": np.asarray([cell.cell_id for cell in ORACLE_CELLS]),
        "state_code_names": np.asarray(list(STATE_CODE)),
        "reason_code_names": np.asarray([
            "none", "candidate_open", "candidate_abort_stage1",
            "candidate_timeout", "command_commit", "idle_reset",
        ]),
    }
    for cell in ORACLE_CELLS:
        replay = replays[cell.cell_id]
        result = evaluate_online_events(
            inventory.segments,
            inventory.events,
            inventory.windows,
            replay.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        _check_metric_inventory(result, inventory_contract)
        if cell.commit_truth:
            latency_samples = sum(
                int(round(match["latency_seconds"] * 250))
                for match in result["event_matches"]
                if match["outcome"] == "correct"
            )
            if (
                result["correct_event_count"] != replay.optimized_correct_count
                or latency_samples != replay.optimized_latency_samples
                or result["idle_false_command_count"] != 0
                or result["too_early_command_count"] != 0
                or result["additional_event_command_count"] != 0
            ):
                raise RuntimeError("真值提交动态规划目标与正式事件评估结果不一致")
        evaluations[cell.cell_id] = result
        compact[cell.cell_id] = {
            "definition": cell.public_definition(),
            "oracle_optimizer": {
                "optimized_correct_count": replay.optimized_correct_count,
                "optimized_latency_samples": replay.optimized_latency_samples,
                "selected_command_identities": replay.selected_command_identities,
            },
            "formal_evaluation": _compact_evaluation(result),
        }
        arrays[f"{cell.cell_id}_emitted"] = np.asarray(
            [row.emitted_class for row in replay.decisions], dtype=np.int8,
        )
        arrays[f"{cell.cell_id}_before"] = np.asarray(
            [STATE_CODE[row.decision_state_before] for row in replay.decisions], dtype=np.uint8,
        )
        arrays[f"{cell.cell_id}_after"] = np.asarray(
            [STATE_CODE[row.decision_state_after] for row in replay.decisions], dtype=np.uint8,
        )
        arrays[f"{cell.cell_id}_reason"] = np.asarray(
            [REASON_CODE[row.transition_reason] for row in replay.decisions], dtype=np.uint8,
        )

    baseline = evaluations[MODEL_CELL_ID]
    evidence = stage2_evidence_ceiling(
        inventory.windows,
        inventory.events,
        stage2_logits,
        baseline["event_matches"],
    )
    shapley = shapley_component_contributions({
        cell.cell_id: evaluations[cell.cell_id]["correct_event_rate"]
        for cell in ORACLE_CELLS
    })
    payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "anchor": anchor_cell.public_config(),
        "truth_usage": "post_model_anchor_component_oracle_and_evidence_ceiling_only",
        "cells": compact,
        "stage2_evidence_ceiling": evidence,
        "shapley_correct_event_rate": shapley,
    }
    metrics_path = output_root / f"seed{seed}_oracle_metrics.json"
    trajectory_path = output_root / f"seed{seed}_oracle_trajectories.npz"
    atomic_json(metrics_path, payload)
    atomic_npz(trajectory_path, **arrays)
    summary = {
        "cells": {
            cell.cell_id: core_metrics(evaluations[cell.cell_id])
            for cell in ORACLE_CELLS
        },
        "evidence": _flat_evidence(evidence["summary"]),
        "shapley": _flat_shapley(shapley),
    }
    return summary, {
        "input_scores": input_scores,
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {
            "file": trajectory_path.name,
            "sha256": file_hash(trajectory_path),
        },
    }


# ---------- 九被试分层汇总：禁止把 2328 个事件直接混成一个微平均 ----------
def _aggregate(subject_summaries: dict[int, dict[int, dict]]) -> dict:
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("Oracle 汇总缺少 Subject 1-9")

    component: dict[str, dict] = {}
    for cell in ORACLE_CELLS:
        per_seed: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            per_seed[str(seed)] = {
                field: _statistics([
                    subject_summaries[subject][seed]["cells"][cell.cell_id][field]
                    for subject in KNOWN_SUBJECTS
                    if subject_summaries[subject][seed]["cells"][cell.cell_id][field] is not None
                ])
                for field in CORE_FIELDS
            }
        component[cell.cell_id] = {
            "definition": cell.public_definition(),
            "per_seed_equal_subject_macro": per_seed,
            "across_paired_seeds": {
                field: _statistics([
                    per_seed[str(seed)][field]["mean"]
                    for seed in KNOWN_SEEDS
                    if per_seed[str(seed)][field]["mean"] is not None
                ])
                for field in CORE_FIELDS
            },
        }

    def aggregate_section(section: str, fields: tuple[str, ...]) -> dict:
        per_seed: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            per_seed[str(seed)] = {
                field: _statistics([
                    subject_summaries[subject][seed][section][field]
                    for subject in KNOWN_SUBJECTS
                    if subject_summaries[subject][seed][section][field] is not None
                ])
                for field in fields
            }
        return {
            "per_seed_equal_subject_macro": per_seed,
            "across_paired_seeds": {
                field: _statistics([
                    per_seed[str(seed)][field]["mean"]
                    for seed in KNOWN_SEEDS
                    if per_seed[str(seed)][field]["mean"] is not None
                ])
                for field in fields
            },
        }

    return {
        "aggregation": (
            "equal_subject_macro_within_seed_then_mean_and_population_std_"
            "across_three_paired_seeds"
        ),
        "component_cells": component,
        "stage2_evidence": aggregate_section("evidence", EVIDENCE_FIELDS),
        "shapley": aggregate_section("shapley", SHAPLEY_FIELDS),
    }


def _write_csvs(output_root: Path, subject_summaries: dict[int, dict[int, dict]], summary: dict) -> dict:
    component_subject = output_root / "oracle_component_per_subject_seed.csv"
    component_aggregate = output_root / "oracle_component_aggregate.csv"
    evidence_subject = output_root / "oracle_evidence_per_subject_seed.csv"
    evidence_aggregate = output_root / "oracle_evidence_aggregate.csv"
    shapley_subject = output_root / "oracle_shapley_per_subject_seed.csv"
    shapley_aggregate = output_root / "oracle_shapley_aggregate.csv"

    component_rows = []
    for subject in KNOWN_SUBJECTS:
        for seed in KNOWN_SEEDS:
            for cell in ORACLE_CELLS:
                component_rows.append({
                    "subject": subject,
                    "seed": seed,
                    "cell_id": cell.cell_id,
                    "stage1_source": "truth" if cell.stage1_truth else "model",
                    "commit_source": "truth" if cell.commit_truth else "model",
                    "reset_source": "truth" if cell.reset_truth else "model",
                    **subject_summaries[subject][seed]["cells"][cell.cell_id],
                })
    _atomic_csv(
        component_subject,
        [
            "subject", "seed", "cell_id", "stage1_source", "commit_source",
            "reset_source", *CORE_FIELDS,
        ],
        component_rows,
    )

    aggregate_rows = []
    for cell in ORACLE_CELLS:
        row = {"cell_id": cell.cell_id}
        for field in CORE_FIELDS:
            statistics = summary["component_cells"][cell.cell_id]["across_paired_seeds"][field]
            row[f"{field}_mean"] = statistics["mean"]
            row[f"{field}_population_std"] = statistics["population_std"]
        aggregate_rows.append(row)
    aggregate_fields = ["cell_id"]
    for field in CORE_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_population_std"])
    _atomic_csv(component_aggregate, aggregate_fields, aggregate_rows)

    evidence_rows = [
        {"subject": subject, "seed": seed, **subject_summaries[subject][seed]["evidence"]}
        for subject in KNOWN_SUBJECTS for seed in KNOWN_SEEDS
    ]
    shapley_rows = [
        {"subject": subject, "seed": seed, **subject_summaries[subject][seed]["shapley"]}
        for subject in KNOWN_SUBJECTS for seed in KNOWN_SEEDS
    ]
    _atomic_csv(evidence_subject, ["subject", "seed", *EVIDENCE_FIELDS], evidence_rows)
    _atomic_csv(shapley_subject, ["subject", "seed", *SHAPLEY_FIELDS], shapley_rows)

    # 聚合 CSV 每行一个指标，便于不解析嵌套 manifest 也能核对均值、seed 波动和有效数。
    evidence_aggregate_rows = [
        {
            "metric": field,
            **summary["stage2_evidence"]["across_paired_seeds"][field],
        }
        for field in EVIDENCE_FIELDS
    ]
    shapley_aggregate_rows = [
        {
            "metric": field,
            **summary["shapley"]["across_paired_seeds"][field],
        }
        for field in SHAPLEY_FIELDS
    ]
    statistics_fields = ["metric", "mean", "population_std", "valid_count"]
    _atomic_csv(evidence_aggregate, statistics_fields, evidence_aggregate_rows)
    _atomic_csv(shapley_aggregate, statistics_fields, shapley_aggregate_rows)
    return {
        "component_per_subject_seed": {
            "file": component_subject.name, "sha256": file_hash(component_subject),
        },
        "component_aggregate": {
            "file": component_aggregate.name, "sha256": file_hash(component_aggregate),
        },
        "evidence_per_subject_seed": {
            "file": evidence_subject.name, "sha256": file_hash(evidence_subject),
        },
        "evidence_aggregate": {
            "file": evidence_aggregate.name, "sha256": file_hash(evidence_aggregate),
        },
        "shapley_per_subject_seed": {
            "file": shapley_subject.name, "sha256": file_hash(shapley_subject),
        },
        "shapley_aggregate": {
            "file": shapley_aggregate.name, "sha256": file_hash(shapley_aggregate),
        },
    }


def _verify_child(
    root: Path,
    manifest: dict,
    *,
    subject: int,
    claim_status: str,
    source_hashes: dict[str, str],
    oracle_config_sha256: str,
    anchor_config_sha256: str,
) -> None:
    """顶层签名前重新读盘，确保每个被试的真值绑定和三 seed 产物均闭合。"""
    if (
        manifest.get("status") != "PASS"
        or manifest.get("claim_status") != claim_status
        or manifest.get("protocol_id") != EXPECTED_OUTPUT_PROTOCOL
        or manifest.get("subject") != subject
        or tuple(manifest.get("seeds", [])) != KNOWN_SEEDS
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or manifest.get("truth_load_phase") != EXPECTED_TRUTH_LOAD_PHASE
        or manifest.get("artifact_policy") != ARTIFACT_POLICY
        or manifest.get("segment_policy") != SEGMENT_POLICY
        or manifest.get("source_sha256") != source_hashes
        or manifest.get("oracle_config_sha256") != oracle_config_sha256
        or manifest.get("anchor_config_sha256") != anchor_config_sha256
        or manifest.get("all_model_anchor_exact") is not True
    ):
        raise RuntimeError(f"Subject {subject} Oracle 子清单合同非法")
    artifacts = [manifest.get("run_log", {})]
    for seed in KNOWN_SEEDS:
        roles = manifest.get("seed_artifacts", {}).get(str(seed), {})
        if set(roles) != {"input_scores", "metrics", "trajectories"}:
            raise RuntimeError(f"Subject {subject} seed {seed} Oracle 产物角色不完整")
        artifacts.extend([roles["metrics"], roles["trajectories"]])
    for artifact in artifacts:
        path = _safe_artifact(root, artifact.get("file", ""))
        if not path.is_file() or file_hash(path) != artifact.get("sha256"):
            raise RuntimeError(f"Subject {subject} Oracle 子产物哈希不一致")


def _source_hashes() -> dict[str, str]:
    """冻结会影响 logits 验证、真值读取、反事实轨迹、LP 和正式事件评分的源码。"""
    return {
        "oracle_diagnostics": file_hash(EVAL_DIR / "oracle_ceiling_diagnostics.py"),
        "oracle_runner": file_hash(Path(__file__)),
        "candidate_state_policy": file_hash(EVAL_DIR / "candidate_state_policy.py"),
        "logit_candidate_strategies": file_hash(EVAL_DIR / "logit_candidate_strategies.py"),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "online_truth_inventory": file_hash(EVAL_DIR / "online_truth_inventory.py"),
        "commit_reset_anchor_loader": file_hash(EVAL_DIR / "run_commit_reset_matrix.py"),
        "frozen_input_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
        "single_window_multi_subject_verifier": file_hash(
            EVAL_DIR / "run_epoch50_online_oof_all_subjects.py"
        ),
        "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
        "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
    }


# ---------- 主入口：先完成全部无真值工作，再打开独立真值侧车并运行 Oracle ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    oracle_config_path = Path(args.oracle_config).resolve()
    anchor_config_path = Path(args.anchor_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    oracle_config_sha256 = file_hash(oracle_config_path)
    anchor_config_sha256 = file_hash(anchor_config_path)
    input_master_path = input_root / "run_manifest.json"
    input_master_sha256 = file_hash(input_master_path)
    contract = load_oracle_contract(oracle_config_path)
    anchor_cell = load_anchor(anchor_config_path)
    input_master, input_children = verify_input_root(input_root)
    input_child_hashes = {
        subject: input_master["children"][str(subject)]["manifest_sha256"]
        for subject in KNOWN_SUBJECTS
    }
    source_hashes = _source_hashes()
    run_git = git_state()
    environment = {**runtime_environment(), "scipy_version": scipy.__version__}
    claim_status = (
        "PRECOMMIT_ORACLE_DIAGNOSTIC_ONLY"
        if run_git["dirty"] is not False
        else "CLEAN_COMMIT_ORACLE_DIAGNOSTIC_ONLY"
    )

    subject_summaries: dict[int, dict[int, dict]] = {}
    children: dict[str, dict] = {}
    immutable_subject_files: list[tuple[Path, str]] = []
    matrix_artifact_identity: dict[str, str] | None = None
    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=False)
        paths = default_subject_paths(subject)
        context = load_bundle(paths.bundle_manifest)
        subject_identity = artifact_contract(context.manifest)
        if matrix_artifact_identity is None:
            matrix_artifact_identity = subject_identity
        elif matrix_artifact_identity != subject_identity:
            raise RuntimeError("九被试 Oracle bundle 的伪迹合同绑定方式不一致")

        # 此阶段只建立匿名 segment/window 库存，标签字段仍为哨兵值。
        signal_inventory = _build_online_signal_inventory(context)
        expected_rows = output_window_rows(signal_inventory.windows)
        input_child_path = _safe_artifact(
            input_root, input_master["children"][str(subject)]["manifest"],
        )
        input_child = input_children[subject]
        if (
            input_child.get("inputs", {}).get("bundle_manifest") != display_path(paths.bundle_manifest)
            or input_child.get("inputs", {}).get("bundle_manifest_sha256") != context.manifest_sha256
        ):
            raise RuntimeError(f"Subject {subject} 冻结 logits 与当前 bundle 不匹配")

        # 三个 seed 的 logits 与纯模型锚点必须全部先生成；这一段不打开真值文件。
        seed_inputs: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
        pretruth_anchors = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = _load_seed_logits(
                input_child_path.parent, input_child, seed, expected_rows,
            )
            score_path = Path(input_scores["file"])
            if not score_path.is_absolute():
                score_path = PROJECT_ROOT / score_path
            immutable_subject_files.append((score_path.resolve(), input_scores["sha256"]))
            seed_inputs[seed] = (stage1, stage2, input_scores)
            pretruth_anchors[seed] = logit_candidate_decisions(
                signal_inventory.windows,
                stage1,
                stage2,
                anchor_cell.logit_config,
                idle_reset_consecutive_windows=anchor_cell.idle_reset_consecutive_windows,
            )

        # 只有上述工作全部完成后才加载独立 session0 真值和 v3/v4 库存合同。
        truth = load_truth_inventory(paths.truth_manifest, context)
        inventory = build_online_inventory(context, truth)
        if (
            inventory.segments != signal_inventory.segments
            or inventory.windows != signal_inventory.windows
            or not np.array_equal(inventory.signal_rows, signal_inventory.signal_rows)
        ):
            raise RuntimeError("Subject 真值加载改变了匿名推理库存")
        inventory_contract = _read_json(paths.inventory_contract)
        verify_inventory_contract(context, inventory, inventory_contract)
        truth_payload = _read_json(paths.truth_manifest)
        truth_event_path = paths.truth_manifest.parent / truth_payload["event_file"]
        immutable_subject_files.extend([
            (paths.bundle_manifest, context.manifest_sha256),
            (paths.truth_manifest, truth.manifest_sha256),
            (truth_event_path, truth.event_file_sha256),
            (paths.inventory_contract, file_hash(paths.inventory_contract)),
            (input_child_path, input_child_hashes[subject]),
        ])

        subject_summaries[subject] = {}
        seed_artifacts: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = seed_inputs[seed]
            summary, artifacts = _run_seed(
                child_root,
                inventory,
                inventory_contract,
                seed,
                stage1,
                stage2,
                input_scores,
                anchor_cell,
                pretruth_anchors[seed],
            )
            subject_summaries[subject][seed] = summary
            seed_artifacts[str(seed)] = artifacts

        completed = datetime.now(timezone.utc).isoformat()
        log_path = child_root / "run_log.json"
        atomic_json(log_path, {
            "status": "PASS",
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "claim_status": claim_status,
            "subject": subject,
            "seeds": list(KNOWN_SEEDS),
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "truth_load_phase": EXPECTED_TRUTH_LOAD_PHASE,
            **subject_identity,
            "started_at_utc": subject_started,
            "completed_at_utc": completed,
            "runtime_environment": environment,
        })
        child_manifest = {
            "status": "PASS",
            "claim_status": claim_status,
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "selection_status": contract["selection_status"],
            "subject": subject,
            "seeds": list(KNOWN_SEEDS),
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "truth_load_phase": EXPECTED_TRUTH_LOAD_PHASE,
            "all_model_anchor_exact": True,
            **subject_identity,
            "truth_inventory": {
                "protocol_id": truth.manifest["protocol_id"],
                "manifest": display_path(paths.truth_manifest),
                "manifest_sha256": truth.manifest_sha256,
                "event_file_sha256": truth.event_file_sha256,
            },
            "inventory_contract": {
                "file": display_path(paths.inventory_contract),
                "sha256": file_hash(paths.inventory_contract),
                "protocol_id": inventory_contract["protocol_id"],
            },
            "oracle_config_sha256": oracle_config_sha256,
            "anchor_config_sha256": anchor_config_sha256,
            "anchor": anchor_cell.public_config(),
            "input_child_manifest": {
                "file": display_path(input_child_path),
                "sha256": file_hash(input_child_path),
            },
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
        child_manifest_path = child_root / "run_manifest.json"
        atomic_json(child_manifest_path, child_manifest)
        _verify_child(
            child_root,
            _read_json(child_manifest_path),
            subject=subject,
            claim_status=claim_status,
            source_hashes=source_hashes,
            oracle_config_sha256=oracle_config_sha256,
            anchor_config_sha256=anchor_config_sha256,
        )
        children[str(subject)] = {
            "manifest": str(child_manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(child_manifest_path),
            "window_count": len(inventory.windows),
            "event_count": len(inventory.events),
        }
        print(f"Subject {subject}: PASS", flush=True)

    if matrix_artifact_identity is None:
        raise RuntimeError("Oracle 没有加载任何被试")
    summary = _aggregate(subject_summaries)
    csv_artifacts = _write_csvs(output_root, subject_summaries, summary)
    completed_at = datetime.now(timezone.utc).isoformat()
    if (
        git_state() != run_git
        or {**runtime_environment(), "scipy_version": scipy.__version__} != environment
        or _source_hashes() != source_hashes
        or file_hash(oracle_config_path) != oracle_config_sha256
        or file_hash(anchor_config_path) != anchor_config_sha256
        or file_hash(input_master_path) != input_master_sha256
        or any(file_hash(path) != expected for path, expected in immutable_subject_files)
    ):
        raise RuntimeError("Oracle 运行期间 Git、源码、配置、输入 logits 或真值身份发生变化")

    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "claim_status": claim_status,
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "truth_load_phase": EXPECTED_TRUTH_LOAD_PHASE,
        **matrix_artifact_identity,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_environment": environment,
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "selection_status": contract["selection_status"],
        "claim_limit": contract["claim_limit"],
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "truth_load_phase": EXPECTED_TRUTH_LOAD_PHASE,
        "all_model_anchor_exact": True,
        **matrix_artifact_identity,
        "oracle_config": {
            "file": display_path(oracle_config_path),
            "sha256": oracle_config_sha256,
        },
        "anchor_config": {
            "file": display_path(anchor_config_path),
            "sha256": anchor_config_sha256,
            "cell_id": ANCHOR_CELL_ID,
        },
        "input_root_manifest": {
            "file": display_path(input_master_path),
            "sha256": input_master_sha256,
            "protocol_id": input_master["protocol_id"],
        },
        "cell_definitions": [cell.public_definition() for cell in ORACLE_CELLS],
        "children": children,
        "csv_artifacts": csv_artifacts,
        "summary": summary,
        "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
        "source_sha256": source_hashes,
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
    parser.add_argument("--oracle-config", type=Path, default=DEFAULT_ORACLE_CONFIG)
    parser.add_argument("--anchor-config", type=Path, default=DEFAULT_ANCHOR_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
