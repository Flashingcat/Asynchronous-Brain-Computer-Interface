"""从冻结九被试 OOF logits 运行可撤销候选态的多种因果分数策略。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
from logit_candidate_strategies import LogitStrategyConfig, logit_candidate_decisions
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    atomic_json,
    atomic_npz,
    core_metrics,
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
from oof_training_bundle import (  # noqa: E402
    ARTIFACT_POLICY,
    SEGMENT_POLICY,
    artifact_contract,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_candidate_logit_matrix_v1"
)
DEFAULT_POLICY_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_candidate_logit_strategies_v1.json"
)
EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_candidate_logit_matrix_v1"
EXPECTED_SEMANTICS = {
    "stage1_binary_score": (
        "task_logit_minus_idle_logit_then_sigmoid, except ewma_probability"
    ),
    "stage2_shift_invariance": "subtract each window mean before causal aggregation",
    "stage2_history": (
        "candidate-local, excludes opening window, clears on every candidate exit"
    ),
    "temporal_reset": "every frozen segment",
    "stage1_startup": (
        "EWMA starts from the first score; rolling mean uses the available causal prefix"
    ),
    "stage2_stability": (
        "consecutive raw top1 classes after opening and equal to aggregate top1"
    ),
    "threshold_comparison": (
        "on, hold and commit use >=; idle reset and curvature use <=; "
        "drop abort uses delta <= -threshold"
    ),
    "curvature": (
        "L2 norm of second difference of the latest three candidate-local raw-window "
        "Stage 2 softmax vectors; unavailable values cannot pass"
    ),
}
CANDIDATE_FIELDS = (
    "candidate_opens_per_valid_minute",
    "candidate_conversion_rate",
    "candidate_stage1_abort_rate",
    "candidate_timeout_rate",
    "candidate_unresolved_rate",
    "candidate_dwell_mean_seconds",
    "candidate_dwell_median_seconds",
    "miss_event_with_candidate_timeout_rate",
)
SUMMARY_FIELDS = (*CORE_FIELDS, *CANDIDATE_FIELDS)
SCALE_FIELDS = (
    "stage1_margin",
    "stage1_task_probability",
    "stage2_top_logit_margin",
    "stage2_top_probability",
    "stage1_task_probability_delta",
    "stage2_centered_delta_l2",
    "stage2_probability_second_difference_l2",
)
SCALE_QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
STATE_CODE = {READY: 0, TASK_CANDIDATE: 1, WAIT_IDLE: 2}
REASON_CODE = {
    None: 0,
    CANDIDATE_OPEN: 1,
    CANDIDATE_ABORT_STAGE1: 2,
    CANDIDATE_TIMEOUT: 3,
    COMMAND_COMMIT: 4,
    IDLE_RESET: 5,
}


# ---------- 配置合同：允许并列探索多种策略，但禁止隐藏选择或读取测试 session ----------
def load_strategy_contract(path: Path) -> tuple[dict, tuple[LogitStrategyConfig, ...]]:
    payload = _read_json(path)
    if (
        set(payload) != {
            "protocol_id", "input_protocol_id", "included_session",
            "test_session_access", "selection_status", "parameter_origin",
            "strategy_semantics", "strategies",
        }
        or payload.get("protocol_id") != EXPECTED_OUTPUT_PROTOCOL
        or payload.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or payload.get("included_session") != 0
        or payload.get("test_session_access") != "forbidden"
        or payload.get("selection_status") != "none_all_cells_reported"
        or payload.get("parameter_origin")
        != "fixed after unlabeled scale inspection of the same session0 OOF logits"
        or payload.get("strategy_semantics") != EXPECTED_SEMANTICS
        or not isinstance(payload.get("strategies"), list)
        or not payload["strategies"]
    ):
        raise RuntimeError("候选 logit 策略总配置与诊断协议不一致")
    configs = tuple(LogitStrategyConfig.from_dict(item) for item in payload["strategies"])
    identifiers = [item.strategy_id for item in configs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("候选策略编号不得重复")
    return payload, configs


def _strategy_metrics(result: dict) -> dict[str, float | None]:
    candidate = result["candidate_diagnostics"]
    dwell = candidate["completed_candidate_dwell_seconds"]
    return {
        **core_metrics(result),
        "candidate_opens_per_valid_minute": candidate["candidate_opens_per_valid_minute"],
        "candidate_conversion_rate": candidate["candidate_conversion_rate"],
        "candidate_stage1_abort_rate": candidate["candidate_stage1_abort_rate"],
        "candidate_timeout_rate": candidate["candidate_timeout_rate"],
        "candidate_unresolved_rate": candidate["candidate_unresolved_rate"],
        "candidate_dwell_mean_seconds": dwell["mean"],
        "candidate_dwell_median_seconds": dwell["median"],
        "miss_event_with_candidate_timeout_rate": (
            candidate["miss_event_with_candidate_timeout_rate"]
        ),
    }


def _check_metric_inventory(result: dict, contract: dict) -> None:
    frozen = contract["inventory"]
    if (
        result["evaluation_mode"] != STATEFUL_CANDIDATE
        or result["scoring_segment_inventory_sha256"] != frozen["segment_inventory_sha256"]
        or result["expected_window_inventory_sha256"] != frozen["window_inventory_sha256"]
        or result["event_inventory_sha256"] != frozen["event_inventory_sha256"]
        or result["event_count"] != frozen["event_count"]
    ):
        raise RuntimeError("候选策略指标与冻结在线库存不一致")


def _accumulate_logit_scale(
    samples: dict[str, list[np.ndarray]],
    windows,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> None:
    """只从原始分数和 segment 边界统计尺度，不读取事件或类别真值。"""
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        margin = stage1[:, 1] - stage1[:, 0]
        offsets = stage2 - stage2[:, :1]
        centered = offsets - np.mean(offsets, axis=1, keepdims=True)
        sorted_logits = np.sort(stage2, axis=1)
        top_logit_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    if (
        not np.isfinite(margin).all()
        or not np.isfinite(centered).all()
        or not np.isfinite(top_logit_margin).all()
    ):
        raise RuntimeError("无标签 logit 尺度派生发生溢出")
    task_probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))
    shifted = centered - np.max(centered, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probability = exp / np.sum(exp, axis=1, keepdims=True)
    sorted_probability = np.sort(probability, axis=1)
    samples["stage1_margin"].append(margin)
    samples["stage1_task_probability"].append(task_probability)
    samples["stage2_top_logit_margin"].append(top_logit_margin)
    samples["stage2_top_probability"].append(sorted_probability[:, -1])

    probability_delta: list[float] = []
    centered_delta: list[float] = []
    probability_second: list[float] = []
    for index, window in enumerate(windows):
        if index and window.key == windows[index - 1].key:
            probability_delta.append(float(task_probability[index] - task_probability[index - 1]))
            centered_delta.append(float(np.linalg.norm(centered[index] - centered[index - 1])))
        if (
            index >= 2
            and window.key == windows[index - 1].key == windows[index - 2].key
        ):
            probability_second.append(float(np.linalg.norm(
                probability[index] - 2.0 * probability[index - 1] + probability[index - 2]
            )))
    samples["stage1_task_probability_delta"].append(np.asarray(probability_delta))
    samples["stage2_centered_delta_l2"].append(np.asarray(centered_delta))
    samples["stage2_probability_second_difference_l2"].append(
        np.asarray(probability_second)
    )


def _summarize_logit_scale(samples: dict[str, list[np.ndarray]]) -> dict:
    fields: dict[str, dict] = {}
    for name in SCALE_FIELDS:
        values = np.concatenate(samples[name])
        if not len(values) or not np.isfinite(values).all():
            raise RuntimeError(f"logit 尺度样本非法: {name}")
        fields[name] = {
            "sample_count": int(len(values)),
            "quantile_probabilities": list(SCALE_QUANTILES),
            "quantile_values": [
                float(value) for value in np.quantile(values, SCALE_QUANTILES)
            ],
        }
    return {
        "labels_or_events_used": False,
        "pooling": "all 9 subjects x 3 seeds model-window rows from session0 OOF",
        "fields": fields,
    }


# ---------- 单 seed 重放：原始 logits 不改写，策略分数、证据和轨迹分层保存 ----------
def _save_seed_matrix(
    output_root: Path,
    inventory,
    contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
    configs: tuple[LogitStrategyConfig, ...],
) -> tuple[dict[str, dict], dict]:
    arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "strategy_ids": np.asarray([item.strategy_id for item in configs]),
        "state_code_names": np.asarray([READY, TASK_CANDIDATE, WAIT_IDLE]),
        "reason_code_names": np.asarray([
            "none", CANDIDATE_OPEN, CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT, COMMAND_COMMIT, IDLE_RESET,
        ]),
    }
    metrics_payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "selection_status": "none_all_cells_reported",
        "strategies": {},
    }
    summary: dict[str, dict] = {}

    for config in configs:
        strategy = logit_candidate_decisions(
            inventory.windows, stage1_logits, stage2_logits, config,
        )
        result = evaluate_online_events(
            inventory.segments,
            inventory.events,
            inventory.windows,
            strategy.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        _check_metric_inventory(result, contract)
        identifier = config.strategy_id
        metrics_payload["strategies"][identifier] = {
            "config": asdict(config),
            "metrics": result,
        }
        summary[identifier] = _strategy_metrics(result)

        decisions = strategy.policy.decisions
        policy_trace = strategy.policy.trace
        score_trace = strategy.trace
        arrays[f"{identifier}_emitted"] = np.asarray(
            [item.emitted_class for item in decisions], dtype=np.int8,
        )
        arrays[f"{identifier}_before"] = np.asarray(
            [STATE_CODE[item.decision_state_before] for item in decisions], dtype=np.uint8,
        )
        arrays[f"{identifier}_after"] = np.asarray(
            [STATE_CODE[item.decision_state_after] for item in decisions], dtype=np.uint8,
        )
        arrays[f"{identifier}_reason"] = np.asarray(
            [REASON_CODE[item.transition_reason] for item in decisions], dtype=np.uint8,
        )
        arrays[f"{identifier}_candidate_age_before"] = np.asarray(
            [item.candidate_windows_before for item in policy_trace], dtype=np.int64,
        )
        arrays[f"{identifier}_candidate_age_after"] = np.asarray(
            [item.candidate_windows_after for item in policy_trace], dtype=np.int64,
        )
        for name, dtype in (
            ("task_on", np.bool_), ("task_hold", np.bool_), ("idle_reset", np.bool_),
        ):
            arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item.evidence, name) for item in score_trace], dtype=dtype,
            )
        arrays[f"{identifier}_stage2_commit_class"] = np.asarray(
            [item.evidence.stage2_commit_class for item in score_trace], dtype=np.int8,
        )
        for name in (
            "stage1_raw_margin", "stage1_raw_task_probability",
            "stage1_filtered_task_probability", "stage1_filtered_delta",
            "stage2_top_probability", "stage2_probability_gap",
            "stage2_probability_curvature",
        ):
            arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item, name) for item in score_trace], dtype=np.float32,
            )
        for name, dtype in (
            ("stage2_candidate_window_count", np.int64),
            ("stage2_top_class", np.int8),
            ("stage2_stable_windows", np.int64),
        ):
            arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item, name) for item in score_trace], dtype=dtype,
            )

    metrics_path = output_root / f"seed{seed}_candidate_logit_metrics.json"
    trajectory_path = output_root / f"seed{seed}_candidate_logit_trajectories.npz"
    atomic_json(metrics_path, metrics_payload)
    atomic_npz(trajectory_path, **arrays)
    return summary, {
        "input_scores": input_scores,
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
    }


# ---------- 汇总：同 seed 九被试等权，再对三个配对 seed 描述性汇总 ----------
def summarize_subject(
    seed_results: dict[int, dict[str, dict]],
    configs: tuple[LogitStrategyConfig, ...],
) -> dict:
    result: dict[str, dict] = {}
    for config in configs:
        identifier = config.strategy_id
        per_seed = {str(seed): seed_results[seed][identifier] for seed in KNOWN_SEEDS}
        aggregate = {
            field: _statistics([
                float(row[field]) for row in per_seed.values() if row[field] is not None
            ])
            for field in SUMMARY_FIELDS
        }
        result[identifier] = {
            "config": asdict(config),
            "per_seed": per_seed,
            "aggregate_across_seeds": aggregate,
        }
    return result


def aggregate_subject_matrix(
    subject_summaries: dict[int, dict],
    configs: tuple[LogitStrategyConfig, ...],
) -> dict:
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("候选策略汇总缺少 Subject 1–9")
    result: dict[str, dict] = {}
    for config in configs:
        identifier = config.strategy_id
        per_seed_subject_macro: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            row: dict[str, dict] = {}
            for field in SUMMARY_FIELDS:
                values = [
                    float(subject_summaries[subject][identifier]["per_seed"][str(seed)][field])
                    for subject in KNOWN_SUBJECTS
                    if subject_summaries[subject][identifier]["per_seed"][str(seed)][field]
                    is not None
                ]
                row[field] = {
                    "mean": float(np.mean(values)) if values else None,
                    "valid_subject_count": len(values),
                }
            per_seed_subject_macro[str(seed)] = row
        aggregate = {
            field: _statistics([
                float(per_seed_subject_macro[str(seed)][field]["mean"])
                for seed in KNOWN_SEEDS
                if per_seed_subject_macro[str(seed)][field]["mean"] is not None
            ])
            for field in SUMMARY_FIELDS
        }
        result[identifier] = {
            "config": asdict(config),
            "per_seed_subject_macro": per_seed_subject_macro,
            "aggregate_across_seeds": aggregate,
        }
    return result


def write_matrix_csvs(output_root: Path, summary: dict) -> dict:
    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for identifier, payload in summary.items():
        for seed in KNOWN_SEEDS:
            row = payload["per_seed_subject_macro"][str(seed)]
            per_seed_rows.append({
                "strategy_id": identifier,
                "seed": seed,
                **{field: row[field]["mean"] for field in SUMMARY_FIELDS},
            })
        aggregate_rows.append({
            "strategy_id": identifier,
            **{
                f"{field}_{suffix}": payload["aggregate_across_seeds"][field][key]
                for field in SUMMARY_FIELDS
                for suffix, key in (("mean", "mean"), ("population_std", "population_std"))
            },
        })
    per_seed_path = output_root / "candidate_logit_matrix_per_seed.csv"
    aggregate_path = output_root / "candidate_logit_matrix_aggregate.csv"
    _atomic_csv(per_seed_path, ["strategy_id", "seed", *SUMMARY_FIELDS], per_seed_rows)
    aggregate_fields = ["strategy_id"]
    for field in SUMMARY_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_population_std"])
    _atomic_csv(aggregate_path, aggregate_fields, aggregate_rows)
    return {
        "per_seed_csv": {"file": per_seed_path.name, "sha256": file_hash(per_seed_path)},
        "aggregate_csv": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


def verify_matrix_child(
    child_root: Path,
    manifest: dict,
    subject: int,
    strategy_ids: tuple[str, ...],
    expected_source_hashes: dict[str, str],
    expected_policy_sha256: str,
) -> None:
    if (
        manifest.get("status") != "PASS"
        or manifest.get("subject") != subject
        or tuple(manifest.get("seeds", [])) != KNOWN_SEEDS
        or tuple(manifest.get("strategy_ids", [])) != strategy_ids
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or manifest.get("artifact_policy") != ARTIFACT_POLICY
        or manifest.get("segment_policy") != SEGMENT_POLICY
        or manifest.get("artifact_policy_binding") not in {
            "explicit_bundle_manifest", "legacy_v1_protocol_contract",
        }
        or manifest.get("selection_status") != "none_all_cells_reported"
        or manifest.get("source_sha256") != expected_source_hashes
        or manifest.get("policy_contract_sha256") != expected_policy_sha256
    ):
        raise RuntimeError(f"Subject {subject} 候选策略子清单合同非法")
    artifacts = [manifest.get("run_log", {})]
    for seed in KNOWN_SEEDS:
        roles = manifest.get("seed_artifacts", {}).get(str(seed), {})
        if set(roles) != {"input_scores", "metrics", "trajectories"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 产物角色不完整")
        artifacts.extend([roles["metrics"], roles["trajectories"]])
    for artifact in artifacts:
        path = _safe_artifact(child_root, artifact.get("file", ""))
        if not path.is_file() or file_hash(path) != artifact.get("sha256"):
            raise RuntimeError(f"Subject {subject} 候选策略子产物哈希不一致")


# ---------- 主入口：全策略并列报告；同一 OOF 上不得挑 cell 后声称无偏性能 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.policy_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    policy_sha256 = file_hash(config_path)
    policy_contract, configs = load_strategy_contract(config_path)
    if file_hash(config_path) != policy_sha256:
        raise RuntimeError("读取候选策略配置期间文件发生变化")
    strategy_ids = tuple(item.strategy_id for item in configs)
    source_hashes = _source_hashes()
    input_master, input_children = verify_input_root(input_root)
    run_git = git_state()
    environment = runtime_environment()
    claim_status = (
        "PRECOMMIT_DIAGNOSTIC_MATRIX"
        if run_git["dirty"] is not False
        else "CLEAN_COMMIT_DIAGNOSTIC_MATRIX"
    )
    children: dict[str, dict] = {}
    subject_summaries: dict[int, dict] = {}
    scale_samples = {name: [] for name in SCALE_FIELDS}
    matrix_artifact_identity: dict[str, str] | None = None

    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=True)
        context, inventory, contract, paths = _load_subject_inventory(subject)
        subject_artifact_identity = artifact_contract(context.manifest)
        if matrix_artifact_identity is None:
            matrix_artifact_identity = subject_artifact_identity
        elif matrix_artifact_identity != subject_artifact_identity:
            raise RuntimeError("九被试 OOF bundle 的伪迹合同绑定方式不一致")
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
            seed_results[seed], seed_artifacts[str(seed)] = _save_seed_matrix(
                child_root,
                inventory,
                contract,
                seed,
                stage1,
                stage2,
                input_scores,
                configs,
            )
            _accumulate_logit_scale(scale_samples, inventory.windows, stage1, stage2)
        subject_summary = summarize_subject(seed_results, configs)
        subject_summaries[subject] = subject_summary
        completed = datetime.now(timezone.utc).isoformat()
        log_path = child_root / "run_log.json"
        atomic_json(log_path, {
            "status": "PASS",
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "claim_status": claim_status,
            "subject": subject,
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            **subject_artifact_identity,
            "strategy_ids": list(strategy_ids),
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
            **subject_artifact_identity,
            "seeds": list(KNOWN_SEEDS),
            "strategy_ids": list(strategy_ids),
            "policy_contract": policy_contract,
            "policy_contract_sha256": policy_sha256,
            "input_child_manifest": {
                "file": display_path(input_child_path),
                "sha256": file_hash(input_child_path),
                "source_git": input_child.get("runtime", {}).get("git"),
            },
            "inventory_contract": contract,
            "seed_artifacts": seed_artifacts,
            "run_log": {"file": log_path.name, "sha256": file_hash(log_path)},
            "summary": subject_summary,
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
        verify_matrix_child(
            child_root,
            _read_json(manifest_path),
            subject,
            strategy_ids,
            source_hashes,
            policy_sha256,
        )
        children[str(subject)] = {
            "manifest": str(manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(manifest_path),
            "window_count": contract["inventory"]["window_count"],
            "event_count": contract["inventory"]["event_count"],
            "claim_status": claim_status,
        }
        print(f"Subject {subject}: PASS", flush=True)

    if matrix_artifact_identity is None:
        raise RuntimeError("候选 logit 矩阵没有加载任何被试伪迹合同")
    summary = aggregate_subject_matrix(subject_summaries, configs)
    scale_path = output_root / "input_logit_scale_summary.json"
    atomic_json(scale_path, _summarize_logit_scale(scale_samples))
    scale_artifact = {"file": scale_path.name, "sha256": file_hash(scale_path)}
    csv_artifacts = write_matrix_csvs(output_root, summary)
    completed_at = datetime.now(timezone.utc).isoformat()
    if (
        git_state() != run_git
        or runtime_environment() != environment
        or _source_hashes() != source_hashes
        or file_hash(config_path) != policy_sha256
    ):
        raise RuntimeError("候选策略运行期间 Git、源码、配置或解释器身份发生变化")
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "claim_status": claim_status,
        "selection_status": "none_all_cells_reported",
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "strategy_ids": list(strategy_ids),
        **matrix_artifact_identity,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_environment": environment,
        "children": children,
        "input_logit_scale_summary": scale_artifact,
        "csv_artifacts": csv_artifacts,
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "selection_status": "none_all_cells_reported",
        "selection_warning": (
            "This session0 OOF matrix is descriptive; selecting and reporting the same cell "
            "as unbiased performance is forbidden."
        ),
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        **matrix_artifact_identity,
        "strategy_ids": list(strategy_ids),
        "input_master_manifest": {
            "file": display_path(input_root / "run_manifest.json"),
            "sha256": file_hash(input_root / "run_manifest.json"),
            "protocol_id": input_master["protocol_id"],
            "claim_status": input_master["claim_status"],
            "source_git": input_master["runtime_git"],
        },
        "policy_contract": policy_contract,
        "policy_contract_file": display_path(config_path),
        "policy_contract_sha256": policy_sha256,
        "aggregation_semantics": {
            "event_pooling_across_subjects": "forbidden",
            "primary_summary": "equal_subject_macro_within_paired_seed",
            "seed_summary": "mean_and_population_std_across_three_paired_seeds",
        },
        "single_window_reference": input_master["summary"]["stateful_strict"],
        "children": children,
        "input_logit_scale_summary": scale_artifact,
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
        "matrix_runner": file_hash(Path(__file__)),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "frozen_input_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
        "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
        "single_window_multi_subject_verifier": file_hash(
            EVAL_DIR / "run_epoch50_online_oof_all_subjects.py"
        ),
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
