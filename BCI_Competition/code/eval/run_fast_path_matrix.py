"""在冻结的九被试 session0 OOF logits 上运行 Fast-0/Fast-1/慢通道回退矩阵。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from commit_reset_diagnostics import diagnose_commit_reset
from fast_path_candidate_strategies import FastPathConfig, fast_path_candidate_decisions
from fast_path_diagnostics import diagnose_against_anchor, diagnose_fast_paths
from logit_candidate_strategies import logit_candidate_decisions
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    FAST0_COMMAND_COMMIT,
    FAST1_COMMAND_COMMIT,
    IDLE_RESET,
    READY,
    STATEFUL_CANDIDATE,
    TASK_CANDIDATE,
    WAIT_IDLE,
    evaluate_online_events,
)
from run_candidate_logit_matrix import CANDIDATE_FIELDS, _check_metric_inventory, _strategy_metrics
from run_commit_reset_matrix import DIAGNOSTIC_FIELDS, _diagnostic_summary
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
    / "s01_s09_epoch50_causal_fast_path_matrix_v1"
)
DEFAULT_POLICY_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_fast_path_matrix_v1.json"
)
EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_fast_path_matrix_v1"
EXPECTED_PARAMETER_ORIGIN = (
    "fixed descriptive grid after reviewing session0 OOF score scales and post-inference attribution"
)
EXPECTED_SEMANTICS = {
    "fast0": "opening-window raw Stage 2 evidence may atomically commit from READY to WAIT_IDLE",
    "fast1": "opening window and immediately following window use an isolated two-window EWMA and raw-class agreement",
    "slow_fallback": "historical candidate-local Stage 2 EWMA excludes opening window and remains bit-exact when fast paths do not commit",
    "priority": "Stage 1 abort before Fast-1 before slow commit before timeout; Fast-0 requires the Stage 1 opening condition",
    "cache_reset": "all fast and slow caches reset at segment boundary and every candidate or command exit",
    "decision_time": "window stop at native 250 Hz with 500-sample windows and 125-sample steps",
    "truth_usage": "labels and event boundaries are used only after inference for metrics and timing attribution",
}
EXPECTED_BASE = {
    "stage1_filter": "ewma_margin",
    "stage1_alpha": 0.5,
    "stage1_window": None,
    "task_on_probability": 0.5,
    "task_hold_probability": 0.3,
    "idle_reset_probability": 0.2,
    "stage1_drop_abort": 0.2,
    "stage2_filter": "candidate_ewma_centered_logits",
    "stage2_alpha": 0.5,
    "stage2_min_candidate_windows": 2,
    "stage2_top_probability": 0.55,
    "stage2_probability_gap": 0.15,
    "stage2_stable_windows": 1,
    "stage2_max_probability_curvature": None,
    "max_candidate_windows": 8,
}
EXPECTED_FAST0 = {
    "balanced": {
        "min_stage1_probability": 0.5,
        "min_stage1_delta": 0.03,
        "min_stage2_top_probability": 0.9,
        "min_stage2_probability_gap": 0.8,
    },
    "strict": {
        "min_stage1_probability": 0.5,
        "min_stage1_delta": 0.03,
        "min_stage2_top_probability": 0.95,
        "min_stage2_probability_gap": 0.9,
    },
}
EXPECTED_FAST1 = {
    "balanced": {
        "min_stage1_probability": 0.6,
        "stage2_alpha": 0.5,
        "min_stage2_top_probability": 0.9,
        "min_stage2_probability_gap": 0.8,
        "require_same_raw_class": True,
    },
    "strict": {
        "min_stage1_probability": 0.6,
        "stage2_alpha": 0.5,
        "min_stage2_top_probability": 0.95,
        "min_stage2_probability_gap": 0.9,
        "require_same_raw_class": True,
    },
}
EXPECTED_CELLS = (
    ("anchor_no_fast", None, None),
    ("f0_balanced", "balanced", None),
    ("f1_balanced", None, "balanced"),
    ("f01_balanced", "balanced", "balanced"),
    ("f0_strict", "strict", None),
    ("f1_strict", None, "strict"),
    ("f01_strict", "strict", "strict"),
)
ANCHOR_CELL_ID = "anchor_no_fast"

PATH_FIELDS = tuple(
    f"{path}_{name}"
    for path in ("fast0", "fast1", "slow")
    for name in (
        "command_share",
        "correct_among_all_commands",
        "triggered_class_accuracy",
        "correct_event_rate",
        "idle_false_per_valid_idle_minute",
        "correct_latency_median_seconds",
    )
)
FAST_TOTAL_FIELDS = (
    "fast_command_share",
    "fast_correct_event_rate",
    "fast_idle_false_per_valid_idle_minute",
)
PAIRED_FIELDS = (
    "anchor_correct_harmed_rate",
    "anchor_miss_rescued_correct_rate",
    "anchor_noncorrect_rescued_correct_rate",
    "both_correct_earlier_rate",
    "anchor_minus_current_correct_latency_median_seconds",
)
SUMMARY_FIELDS = (
    *CORE_FIELDS,
    *CANDIDATE_FIELDS,
    *DIAGNOSTIC_FIELDS,
    *PATH_FIELDS,
    *FAST_TOTAL_FIELDS,
    *PAIRED_FIELDS,
)

STATE_CODE = {READY: 0, TASK_CANDIDATE: 1, WAIT_IDLE: 2}
REASON_CODE = {
    None: 0,
    CANDIDATE_OPEN: 1,
    CANDIDATE_ABORT_STAGE1: 2,
    CANDIDATE_TIMEOUT: 3,
    COMMAND_COMMIT: 4,
    IDLE_RESET: 5,
    FAST0_COMMAND_COMMIT: 6,
    FAST1_COMMAND_COMMIT: 7,
}
PATH_CODE = {"none": 0, "fast0": 1, "fast1": 2, "slow": 3}


@dataclass(frozen=True)
class FastPathCell:
    """矩阵单元显式记录它引用的 Fast-0/Fast-1 阈值档位。"""

    cell_id: str
    fast0_profile: str | None
    fast1_profile: str | None
    config: FastPathConfig

    def public_config(self) -> dict:
        f0, f1 = self.config.fast0, self.config.fast1
        return {
            "cell_id": self.cell_id,
            "fast0_profile": self.fast0_profile,
            "fast1_profile": self.fast1_profile,
            "fast0_enabled": f0 is not None,
            "fast1_enabled": f1 is not None,
            "fast0_min_stage1_delta": None if f0 is None else f0.min_stage1_delta,
            "fast0_min_stage2_top_probability": None if f0 is None else f0.min_stage2_top_probability,
            "fast0_min_stage2_probability_gap": None if f0 is None else f0.min_stage2_probability_gap,
            "fast1_min_stage1_probability": None if f1 is None else f1.min_stage1_probability,
            "fast1_min_stage2_top_probability": None if f1 is None else f1.min_stage2_top_probability,
            "fast1_min_stage2_probability_gap": None if f1 is None else f1.min_stage2_probability_gap,
            "full_config": asdict(self.config),
        }


# ---------- 配置合同：7 个 cell 全部报告，锚点必须关闭两条快速通道 ----------
def load_fast_path_contract(path: Path) -> tuple[dict, tuple[FastPathCell, ...]]:
    payload = _read_json(path)
    expected_keys = {
        "protocol_id",
        "input_protocol_id",
        "included_session",
        "test_session_access",
        "selection_status",
        "parameter_origin",
        "strategy_semantics",
        "base_logit_strategy",
        "idle_reset_consecutive_windows",
        "fast0_profiles",
        "fast1_profiles",
        "cells",
    }
    cells_raw = tuple(
        (item.get("cell_id"), item.get("fast0_profile"), item.get("fast1_profile"))
        for item in payload.get("cells", [])
        if isinstance(item, dict)
        and set(item) == {"cell_id", "fast0_profile", "fast1_profile"}
    )
    if (
        set(payload) != expected_keys
        or payload.get("protocol_id") != EXPECTED_OUTPUT_PROTOCOL
        or payload.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or payload.get("included_session") != 0
        or payload.get("test_session_access") != "forbidden"
        or payload.get("selection_status") != "none_all_cells_reported"
        or payload.get("parameter_origin") != EXPECTED_PARAMETER_ORIGIN
        or payload.get("strategy_semantics") != EXPECTED_SEMANTICS
        or payload.get("base_logit_strategy") != EXPECTED_BASE
        or payload.get("idle_reset_consecutive_windows") != 1
        or type(payload.get("idle_reset_consecutive_windows")) is not int
        or payload.get("fast0_profiles") != EXPECTED_FAST0
        or payload.get("fast1_profiles") != EXPECTED_FAST1
        or cells_raw != EXPECTED_CELLS
    ):
        raise RuntimeError("Fast-0/Fast-1 矩阵配置与冻结诊断合同不一致")

    cells = []
    for cell_id, fast0_id, fast1_id in EXPECTED_CELLS:
        config = FastPathConfig.from_dict({
            "strategy_id": cell_id,
            "base_logit_strategy": {"strategy_id": cell_id, **EXPECTED_BASE},
            "idle_reset_consecutive_windows": 1,
            "fast0": None if fast0_id is None else EXPECTED_FAST0[fast0_id],
            "fast1": None if fast1_id is None else EXPECTED_FAST1[fast1_id],
        })
        cells.append(FastPathCell(cell_id, fast0_id, fast1_id, config))
    if cells[0].cell_id != ANCHOR_CELL_ID or cells[0].config.fast0 is not None or cells[0].config.fast1 is not None:
        raise RuntimeError("快速通道矩阵缺少第一个无快速通道锚点")
    return payload, tuple(cells)


def _flatten_path_diagnostics(payload: dict) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for path in ("fast0", "fast1", "slow"):
        row = payload["paths"][path]
        for name in (
            "command_share",
            "correct_among_all_commands",
            "triggered_class_accuracy",
            "correct_event_rate",
            "idle_false_per_valid_idle_minute",
        ):
            result[f"{path}_{name}"] = row[name]
        result[f"{path}_correct_latency_median_seconds"] = row["correct_latency_seconds"]["median"]
    commands = sum(payload["paths"][path]["command_count"] for path in ("fast0", "fast1", "slow"))
    fast_commands = sum(payload["paths"][path]["command_count"] for path in ("fast0", "fast1"))
    path_rates = [payload["paths"][path]["correct_event_rate"] for path in ("fast0", "fast1")]
    idle_rates = [
        payload["paths"][path]["idle_false_per_valid_idle_minute"]
        for path in ("fast0", "fast1")
    ]
    result["fast_command_share"] = None if not commands else fast_commands / commands
    result["fast_correct_event_rate"] = (
        None if any(value is None for value in path_rates) else float(sum(path_rates))
    )
    result["fast_idle_false_per_valid_idle_minute"] = (
        None if any(value is None for value in idle_rates) else float(sum(idle_rates))
    )
    return result


def _flatten_paired(payload: dict) -> dict[str, float | None]:
    return {
        "anchor_correct_harmed_rate": payload["anchor_correct_harmed_rate"],
        "anchor_miss_rescued_correct_rate": payload["anchor_miss_rescued_correct_rate"],
        "anchor_noncorrect_rescued_correct_rate": payload["anchor_noncorrect_rescued_correct_rate"],
        "both_correct_earlier_rate": payload["both_correct_earlier_rate"],
        "anchor_minus_current_correct_latency_median_seconds": (
            payload["anchor_minus_current_correct_latency_seconds"]["median"]
        ),
    }


def _assert_anchor_equivalence(strategy, slow_anchor) -> None:
    """关闭快速通道后，状态轨迹和慢通道核心分数必须逐窗相同。"""
    if strategy.policy != slow_anchor.policy:
        raise RuntimeError("无快速通道锚点的状态轨迹与历史慢通道不一致")
    fast_count = [item.slow_candidate_window_count for item in strategy.trace]
    slow_count = [item.stage2_candidate_window_count for item in slow_anchor.trace]
    if fast_count != slow_count:
        raise RuntimeError("无快速通道锚点的慢通道窗数漂移")
    for fast_name, slow_name in (
        ("stage1_filtered_task_probability", "stage1_filtered_task_probability"),
        ("stage1_filtered_delta", "stage1_filtered_delta"),
        ("slow_top_probability", "stage2_top_probability"),
        ("slow_probability_gap", "stage2_probability_gap"),
    ):
        fast_values = np.asarray([getattr(item, fast_name) for item in strategy.trace])
        slow_values = np.asarray([getattr(item, slow_name) for item in slow_anchor.trace])
        if not np.array_equal(fast_values, slow_values):
            raise RuntimeError(f"无快速通道锚点的 {fast_name} 与历史慢通道不一致")


# ---------- 单 seed 完整回放：先生成所有轨迹，再与第一个锚点逐事件配对 ----------
def _save_seed_matrix(
    output_root: Path,
    inventory,
    inventory_contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
    cells: tuple[FastPathCell, ...],
) -> tuple[dict[str, dict], dict]:
    arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "cell_ids": np.asarray([item.cell_id for item in cells]),
        "state_code_names": np.asarray([READY, TASK_CANDIDATE, WAIT_IDLE]),
        "reason_code_names": np.asarray([
            "none",
            CANDIDATE_OPEN,
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT,
            COMMAND_COMMIT,
            IDLE_RESET,
            FAST0_COMMAND_COMMIT,
            FAST1_COMMAND_COMMIT,
        ]),
        "commit_path_code_names": np.asarray(["none", "fast0", "fast1", "slow"]),
    }
    payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "selection_status": "none_all_cells_reported",
        "cells": {},
    }
    strategies = {}
    evaluated_by_cell = {}
    path_by_cell = {}
    reset_by_cell = {}
    for cell in cells:
        strategy = fast_path_candidate_decisions(
            inventory.windows,
            stage1_logits,
            stage2_logits,
            cell.config,
        )
        if cell.cell_id == ANCHOR_CELL_ID:
            slow_anchor = logit_candidate_decisions(
                inventory.windows,
                stage1_logits,
                stage2_logits,
                cell.config.base_logit_strategy,
                idle_reset_consecutive_windows=cell.config.idle_reset_consecutive_windows,
            )
            _assert_anchor_equivalence(strategy, slow_anchor)
        evaluated = evaluate_online_events(
            inventory.segments,
            inventory.events,
            inventory.windows,
            strategy.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        _check_metric_inventory(evaluated, inventory_contract)
        reset = diagnose_commit_reset(
            inventory.segments,
            inventory.events,
            inventory.windows,
            strategy.policy.decisions,
            evaluated,
        )
        path_diagnostics = diagnose_fast_paths(
            inventory.events,
            strategy.policy.decisions,
            evaluated,
        )
        strategies[cell.cell_id] = strategy
        evaluated_by_cell[cell.cell_id] = evaluated
        path_by_cell[cell.cell_id] = path_diagnostics
        reset_by_cell[cell.cell_id] = reset

    anchor = evaluated_by_cell[ANCHOR_CELL_ID]
    summary: dict[str, dict] = {}
    for cell in cells:
        identifier = cell.cell_id
        paired = diagnose_against_anchor(anchor, evaluated_by_cell[identifier])
        payload["cells"][identifier] = {
            "config": cell.public_config(),
            "metrics": evaluated_by_cell[identifier],
            "commit_reset_diagnostics": reset_by_cell[identifier],
            "path_diagnostics": path_by_cell[identifier],
            "paired_anchor_diagnostics": paired,
        }
        summary[identifier] = {
            **_strategy_metrics(evaluated_by_cell[identifier]),
            **_diagnostic_summary(reset_by_cell[identifier]),
            **_flatten_path_diagnostics(path_by_cell[identifier]),
            **_flatten_paired(paired),
        }

        strategy = strategies[identifier]
        decisions, policy_trace, score_trace = (
            strategy.policy.decisions,
            strategy.policy.trace,
            strategy.trace,
        )
        integer_arrays = (
            ("emitted", [item.emitted_class for item in decisions], np.int8),
            ("before", [STATE_CODE[item.decision_state_before] for item in decisions], np.uint8),
            ("after", [STATE_CODE[item.decision_state_after] for item in decisions], np.uint8),
            ("reason", [REASON_CODE[item.transition_reason] for item in decisions], np.uint8),
            ("candidate_age_before", [item.candidate_windows_before for item in policy_trace], np.int16),
            ("candidate_age_after", [item.candidate_windows_after for item in policy_trace], np.int16),
            ("fast_commit_class", [item.evidence.fast_commit_class for item in score_trace], np.int8),
            ("slow_commit_class", [item.evidence.stage2_commit_class for item in score_trace], np.int8),
            ("slow_candidate_count", [item.slow_candidate_window_count for item in score_trace], np.int16),
            ("proposed_path", [PATH_CODE[item.proposed_commit_path] for item in score_trace], np.uint8),
            ("raw_stage2_top_class", [item.raw_stage2_top_class for item in score_trace], np.int8),
            ("fast1_top_class", [item.fast1_top_class for item in score_trace], np.int8),
            ("slow_top_class", [item.slow_top_class for item in score_trace], np.int8),
            (
                "idle_reset_consecutive_count",
                [item.idle_reset_consecutive_count for item in score_trace],
                np.int16,
            ),
        )
        bool_arrays = (
            ("fast0_evaluated", [item.fast0_evaluated for item in score_trace]),
            ("fast0_pass", [item.fast0_pass for item in score_trace]),
            ("fast1_evaluated", [item.fast1_evaluated for item in score_trace]),
            ("fast1_same_raw_class", [item.fast1_same_raw_class for item in score_trace]),
            ("fast1_pass", [item.fast1_pass for item in score_trace]),
            ("idle_reset_raw_condition", [item.idle_reset_raw_condition for item in score_trace]),
            ("idle_reset", [item.evidence.idle_reset for item in score_trace]),
        )
        float_names = (
            "stage1_filtered_task_probability",
            "stage1_filtered_delta",
            "raw_stage2_top_probability",
            "raw_stage2_probability_gap",
            "fast1_top_probability",
            "fast1_probability_gap",
            "slow_top_probability",
            "slow_probability_gap",
        )
        for name, values, dtype in integer_arrays:
            arrays[f"{identifier}_{name}"] = np.asarray(values, dtype=dtype)
        for name, values in bool_arrays:
            arrays[f"{identifier}_{name}"] = np.asarray(values, dtype=np.bool_)
        for name in float_names:
            arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item, name) for item in score_trace],
                dtype=np.float32,
            )

    metrics_path = output_root / f"seed{seed}_fast_path_metrics.json"
    trajectory_path = output_root / f"seed{seed}_fast_path_trajectories.npz"
    atomic_json(metrics_path, payload)
    atomic_npz(trajectory_path, **arrays)
    return summary, {
        "input_scores": input_scores,
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
    }


# ---------- 分层汇总：先在同 seed 内对九被试等权，再汇总三个配对 seed ----------
def _aggregate_subjects(subject_summaries: dict[int, dict], cells: tuple[FastPathCell, ...]) -> dict:
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("快速通道矩阵汇总缺少 Subject 1-9")
    result = {}
    for cell in cells:
        per_seed = {}
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


CSV_CONFIG_FIELDS = (
    "cell_id",
    "fast0_profile",
    "fast1_profile",
    "fast0_enabled",
    "fast1_enabled",
    "fast0_min_stage1_delta",
    "fast0_min_stage2_top_probability",
    "fast0_min_stage2_probability_gap",
    "fast1_min_stage1_probability",
    "fast1_min_stage2_top_probability",
    "fast1_min_stage2_probability_gap",
)


def _write_csvs(output_root: Path, summary: dict) -> dict:
    per_seed_rows, aggregate_rows = [], []
    for item in summary.values():
        config = {name: item["config"][name] for name in CSV_CONFIG_FIELDS}
        for seed in KNOWN_SEEDS:
            row = {**config, "seed": seed}
            for field in SUMMARY_FIELDS:
                value = item["per_seed_subject_macro"][str(seed)][field]
                row[field] = value["mean"]
                row[f"{field}_valid_subject_count"] = value["valid_subject_count"]
            per_seed_rows.append(row)

        # 条件路径指标可能只在部分被试有定义；CSV 必须同时给出有效分母。
        aggregate_row = dict(config)
        for field in SUMMARY_FIELDS:
            stats = item["aggregate_across_seeds"][field]
            counts = [
                item["per_seed_subject_macro"][str(seed)][field]["valid_subject_count"]
                for seed in KNOWN_SEEDS
            ]
            aggregate_row[f"{field}_mean"] = stats["mean"]
            aggregate_row[f"{field}_population_std"] = stats["population_std"]
            aggregate_row[f"{field}_valid_seed_count"] = stats["valid_count"]
            aggregate_row[f"{field}_valid_subject_count_min"] = min(counts)
            aggregate_row[f"{field}_valid_subject_count_max"] = max(counts)
        aggregate_rows.append(aggregate_row)
    per_seed_path = output_root / "fast_path_matrix_per_seed.csv"
    aggregate_path = output_root / "fast_path_matrix_aggregate.csv"
    per_seed_fields = [*CSV_CONFIG_FIELDS, "seed"]
    for field in SUMMARY_FIELDS:
        per_seed_fields.extend([field, f"{field}_valid_subject_count"])
    _atomic_csv(per_seed_path, per_seed_fields, per_seed_rows)
    aggregate_fields = list(CSV_CONFIG_FIELDS)
    for field in SUMMARY_FIELDS:
        aggregate_fields.extend([
            f"{field}_mean",
            f"{field}_population_std",
            f"{field}_valid_seed_count",
            f"{field}_valid_subject_count_min",
            f"{field}_valid_subject_count_max",
        ])
    _atomic_csv(aggregate_path, aggregate_fields, aggregate_rows)
    return {
        "per_seed_csv": {"file": per_seed_path.name, "sha256": file_hash(per_seed_path)},
        "aggregate_csv": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


def _source_hashes() -> dict[str, str]:
    return {
        "candidate_state_policy": file_hash(EVAL_DIR / "candidate_state_policy.py"),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "logit_candidate_strategies": file_hash(EVAL_DIR / "logit_candidate_strategies.py"),
        "fast_path_candidate_strategies": file_hash(EVAL_DIR / "fast_path_candidate_strategies.py"),
        "fast_path_diagnostics": file_hash(EVAL_DIR / "fast_path_diagnostics.py"),
        "commit_reset_diagnostics": file_hash(EVAL_DIR / "commit_reset_diagnostics.py"),
        "fast_path_matrix_runner": file_hash(Path(__file__)),
        "candidate_logit_matrix_helpers": file_hash(EVAL_DIR / "run_candidate_logit_matrix.py"),
        "commit_reset_matrix_helpers": file_hash(EVAL_DIR / "run_commit_reset_matrix.py"),
        "frozen_input_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
        "single_window_multi_subject_verifier": file_hash(
            EVAL_DIR / "run_epoch50_online_oof_all_subjects.py"
        ),
        "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
        "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
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
        raise RuntimeError(f"Subject {subject} 快速通道子清单合同非法")
    artifacts = [manifest["run_log"]]
    for seed in KNOWN_SEEDS:
        roles = manifest["seed_artifacts"].get(str(seed), {})
        if set(roles) != {"input_scores", "metrics", "trajectories"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 快速通道产物角色不完整")
        artifacts.extend([roles["metrics"], roles["trajectories"]])
    for artifact in artifacts:
        path = _safe_artifact(root, artifact["file"])
        if not path.is_file() or file_hash(path) != artifact["sha256"]:
            raise RuntimeError(f"Subject {subject} 快速通道子产物哈希不一致")


# ---------- 主入口：只消费 session0 OOF 冻结分数，禁止覆盖旧结果 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.policy_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    policy_sha256 = file_hash(config_path)
    policy_contract, cells = load_fast_path_contract(config_path)
    if file_hash(config_path) != policy_sha256:
        raise RuntimeError("读取快速通道配置期间文件发生变化")
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
    children, subject_summaries = {}, {}

    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=True)
        context, inventory, inventory_contract, paths = _load_subject_inventory(subject)
        input_child_path = _safe_artifact(
            input_root,
            input_master["children"][str(subject)]["manifest"],
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
        seed_artifacts = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = _load_seed_logits(
                input_child_path.parent,
                input_child,
                seed,
                expected_rows,
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
            child_root,
            _read_json(manifest_path),
            subject,
            cell_ids,
            policy_sha256,
            source_hashes,
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
            "seed_summary": "mean_and_population_std_across_three_paired_seeds",
            "anchor_comparison": "paired_event_outcomes_against_anchor_no_fast",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
