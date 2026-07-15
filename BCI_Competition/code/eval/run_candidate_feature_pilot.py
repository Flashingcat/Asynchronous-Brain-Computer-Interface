"""在 S1/seed42 session0 OOF 上并列运行首批隐藏特征候选门控。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from feature_candidate_strategies import (
    FEATURE_DIM,
    FeatureStrategyConfig,
    feature_candidate_decisions,
)
from logit_candidate_strategies import logit_candidate_decisions
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
from run_candidate_logit_matrix import _check_metric_inventory, _strategy_metrics
from run_epoch50_online_oof import (
    EVAL_DIR,
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
    _load_subject_inventory,
    _read_json,
    _safe_artifact,
    runtime_environment,
)
from oof_training_bundle import artifact_contract  # noqa: E402


SUBJECT = 1
SEED = 42
EXPECTED_PROTOCOL = "bnci2014001_s01_seed42_candidate_feature_pilot_v1"
EXPECTED_INPUT_PROTOCOL = "bnci2014001_s01_seed42_epoch50_feature_preflight_v1"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_seed42_epoch50_feature_preflight_clean_8737c75_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_seed42_candidate_feature_pilot_v1"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_candidate_feature_strategies_pilot_v1.json"
)
EXPECTED_SEMANTICS = {
    "base_logit_policy": "dual_ewma_drop_abort from the frozen logit strategy matrix",
    "feature_source": "Stage 2 flattened EEGNet block2 output before the linear classifier",
    "feature_normalization": "each candidate window is independently L2-normalized",
    "feature_history": (
        "candidate-local, excludes opening window, clears on every candidate exit "
        "and segment boundary"
    ),
    "feature_role": (
        "veto Stage 2 commit only; never opens a candidate and never overrides Stage 1 abort"
    ),
    "metric_order": "compare current feature with causal history before adding the current feature",
    "threshold_comparison": "feature change uses <= and required consecutive passes use >=",
    "reporting_scope": "S1 seed42 mechanism pilot only; no selected or unbiased strategy claim",
}
EXPECTED_BASE_LOGIT_STRATEGY = {
    "strategy_id": "dual_ewma_drop_abort",
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
EXPECTED_FEATURE_CELLS = (
    {
        "strategy_id": "logit_only_reference", "feature_metric": "none",
        "feature_max_change": None, "feature_required_consecutive": 0,
    },
    {
        "strategy_id": "velocity_loose", "feature_metric": "unit_velocity_l2",
        "feature_max_change": 1.2, "feature_required_consecutive": 1,
    },
    {
        "strategy_id": "velocity_strict", "feature_metric": "unit_velocity_l2",
        "feature_max_change": 0.9, "feature_required_consecutive": 1,
    },
    {
        "strategy_id": "velocity_consecutive", "feature_metric": "unit_velocity_l2",
        "feature_max_change": 1.2, "feature_required_consecutive": 2,
    },
    {
        "strategy_id": "prototype_loose",
        "feature_metric": "unit_prototype_cosine_distance",
        "feature_max_change": 0.7, "feature_required_consecutive": 1,
    },
    {
        "strategy_id": "prototype_strict",
        "feature_metric": "unit_prototype_cosine_distance",
        "feature_max_change": 0.4, "feature_required_consecutive": 1,
    },
    {
        "strategy_id": "acceleration_loose", "feature_metric": "unit_acceleration_l2",
        "feature_max_change": 2.0, "feature_required_consecutive": 1,
    },
    {
        "strategy_id": "acceleration_strict", "feature_metric": "unit_acceleration_l2",
        "feature_max_change": 1.5, "feature_required_consecutive": 1,
    },
)
STATE_CODE = {READY: 0, TASK_CANDIDATE: 1, WAIT_IDLE: 2}
REASON_CODE = {
    None: 0,
    CANDIDATE_OPEN: 1,
    CANDIDATE_ABORT_STAGE1: 2,
    CANDIDATE_TIMEOUT: 3,
    COMMAND_COMMIT: 4,
    IDLE_RESET: 5,
}
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# ---------- 配置与输入合同：pilot 身份、完整基线和全部并列 cell 都必须显式 ----------
def load_contract(path: Path) -> tuple[dict, tuple[FeatureStrategyConfig, ...]]:
    payload = _read_json(path)
    if (
        set(payload) != {
            "protocol_id", "input_protocol_id", "subject", "seed", "included_session",
            "test_session_access", "selection_status", "parameter_origin",
            "strategy_semantics", "base_logit_strategy", "strategies",
        }
        or payload.get("protocol_id") != EXPECTED_PROTOCOL
        or payload.get("input_protocol_id") != EXPECTED_INPUT_PROTOCOL
        or payload.get("subject") != SUBJECT
        or payload.get("seed") != SEED
        or payload.get("included_session") != 0
        or payload.get("test_session_access") != "forbidden"
        or payload.get("selection_status") != "none_all_cells_reported"
        or payload.get("parameter_origin")
        != "fixed after unlabeled S1 seed42 candidate-local Stage 2 feature scale inspection"
        or payload.get("strategy_semantics") != EXPECTED_SEMANTICS
        or payload.get("base_logit_strategy") != EXPECTED_BASE_LOGIT_STRATEGY
        or payload.get("strategies") != list(EXPECTED_FEATURE_CELLS)
    ):
        raise RuntimeError("隐藏特征 pilot 配置与冻结机制合同不一致")
    configs = tuple(FeatureStrategyConfig.from_dict({
        **cell,
        "base_logit_strategy": payload["base_logit_strategy"],
    }) for cell in payload["strategies"])
    identifiers = [item.strategy_id for item in configs]
    if (
        len(identifiers) != len(set(identifiers))
        or configs[0].strategy_id != "logit_only_reference"
        or configs[0].feature_metric != "none"
        or any(item.base_logit_strategy != configs[0].base_logit_strategy for item in configs)
    ):
        raise RuntimeError("feature cell 必须唯一、首项为无门控参考并共用同一 logit 骨架")
    return payload, configs


def _frozen_feature_input_contract() -> dict:
    """绑定 8737c75 产生的历史输入，不要求历史源码永远等于当前工作树。"""
    return {
        "manifest_sha256": "3e4f1dd32631fdd0af62c139e53a1f624a5c5b1290037c21b181e6787d9a1195",
        "runtime_git": {
            "commit": "8737c755ffac2ad8a11ad81721d11f902b7d3e5c",
            "dirty": False,
        },
        "source_sha256": {
            "feature_preflight_runner": "e5c5bdaf40ad69975709555f289a3e4e0e9a185e231239962d0cafef69e43ecd",
            "epoch50_online_runner": "8552a3d8295c6a4c6125ca9fc1fb6c98e29011fde163089d8e39b64f84c6cdfb",
            "protocol_metrics": "405fe28d62cd61a5145fbce3380a769339b38226fa0b18639c84f78baa86c25b",
            "oof_training_bundle_reader": "f5a2deb40b64187dcbbce34b5e4c382ebb637177851776713eb96c4e99e80f56",
            "model_factory": "9e6b6af936f088cf0ed3cb25f52cf59d460159019b50beaf7b2b7b7c93173a60",
            "eegnet": "73c97e1bae388ad599025c61cde27f4d451d12b3b83fa898232d6fe90d3fdaed",
        },
    }


def load_feature_input(
    input_root: Path,
    expected_rows: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    manifest_path = input_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    frozen = _frozen_feature_input_contract()
    artifact = manifest.get("artifact", {})
    artifact_path = _safe_artifact(input_root, artifact.get("file", ""))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("claim_status") != "FEATURE_EXTRACTION_PREFLIGHT_ONLY"
        or manifest.get("protocol_id") != EXPECTED_INPUT_PROTOCOL
        or manifest.get("subject") != SUBJECT
        or manifest.get("seed") != SEED
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or manifest.get("job_count") != 12
        or manifest.get("feature_contract", {}).get("dimension_per_stage") != FEATURE_DIM
        or manifest.get("feature_contract", {}).get("strategy_or_threshold_selection") != "none"
        or manifest.get("feature_contract", {}).get("decision_generation") != "none"
        or file_hash(manifest_path) != frozen["manifest_sha256"]
        or manifest.get("runtime", {}).get("git") != frozen["runtime_git"]
        or manifest.get("source_sha256") != frozen["source_sha256"]
        or not artifact_path.is_file()
        or file_hash(artifact_path) != artifact.get("sha256")
        or artifact.get("window_count") != len(expected_rows)
    ):
        raise RuntimeError("隐藏特征输入不是干净提交产生的冻结 S1/seed42 预检产物")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "window_rows", "stage1_logits", "stage2_logits",
            "stage1_features", "stage2_features",
        }:
            raise RuntimeError("隐藏特征输入字段集合非法")
        arrays = {name: archive[name].copy() for name in archive.files}
    if (
        not np.array_equal(arrays["window_rows"], expected_rows)
        or arrays["stage1_logits"].shape != (len(expected_rows), 2)
        or arrays["stage2_logits"].shape != (len(expected_rows), 4)
        or arrays["stage1_features"].shape != (len(expected_rows), FEATURE_DIM)
        or arrays["stage2_features"].shape != (len(expected_rows), FEATURE_DIM)
        or any(
            values.dtype != np.float32 or not np.isfinite(values).all()
            for name, values in arrays.items() if name != "window_rows"
        )
    ):
        raise RuntimeError("隐藏特征输入不能逐窗绑定在线库存或 dtype/shape 非法")
    identity = {
        "manifest": display_path(manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        "artifact": display_path(artifact_path),
        "artifact_file": artifact_path.name,
        "artifact_sha256": file_hash(artifact_path),
        "source_git": manifest["runtime"]["git"],
    }
    return arrays, manifest, identity


# ---------- 无标签尺度：segment 内变化与参考候选内变化分开统计 ----------
def _quantile_field(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise RuntimeError("隐藏特征尺度样本为空或包含非有限值")
    return {
        "sample_count": int(len(array)),
        "quantile_probabilities": list(QUANTILES),
        "quantile_values": [float(value) for value in np.quantile(array, QUANTILES)],
    }


def summarize_feature_scale(windows, stage1: np.ndarray, stage2: np.ndarray, reference) -> dict:
    fields: dict[str, dict] = {}
    for stage, values in ((1, stage1), (2, stage2)):
        raw = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(raw, axis=1)
        unit = raw / norms[:, None]
        velocity, acceleration = [], []
        for index, window in enumerate(windows):
            if index and window.key == windows[index - 1].key:
                velocity.append(float(np.linalg.norm(unit[index] - unit[index - 1])))
            if index >= 2 and window.key == windows[index - 1].key == windows[index - 2].key:
                acceleration.append(float(np.linalg.norm(
                    unit[index] - 2.0 * unit[index - 1] + unit[index - 2]
                )))
        fields[f"stage{stage}_feature_norm"] = _quantile_field(norms)
        fields[f"stage{stage}_segment_unit_velocity_l2"] = _quantile_field(velocity)
        fields[f"stage{stage}_segment_unit_acceleration_l2"] = _quantile_field(acceleration)

    unit = np.asarray(stage2, dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    candidate_velocity, candidate_prototype, candidate_acceleration = [], [], []
    history: list[np.ndarray] = []
    current_key = None
    for index, (window, decision) in enumerate(zip(windows, reference.decisions)):
        if window.key != current_key:
            current_key, history = window.key, []
        if decision.decision_state_before != TASK_CANDIDATE:
            history = []
            continue
        if history:
            candidate_velocity.append(float(np.linalg.norm(unit[index] - history[-1])))
            prototype = np.sum(history, axis=0)
            prototype /= np.linalg.norm(prototype)
            candidate_prototype.append(float(np.clip(1.0 - np.dot(unit[index], prototype), 0, 2)))
        if len(history) >= 2:
            candidate_acceleration.append(float(np.linalg.norm(
                unit[index] - 2.0 * history[-1] + history[-2]
            )))
        history.append(unit[index])
    fields["reference_candidate_unit_velocity_l2"] = _quantile_field(candidate_velocity)
    fields["reference_candidate_unit_prototype_cosine_distance"] = _quantile_field(
        candidate_prototype
    )
    fields["reference_candidate_unit_acceleration_l2"] = _quantile_field(
        candidate_acceleration
    )
    return {
        "labels_or_events_used": False,
        "scope": "S1 seed42 session0 OOF; reference candidates come from logit-only policy",
        "fields": fields,
    }


# ---------- 轨迹和指标分层保存，参考 cell 还必须与原 logit 实现逐窗完全一致 ----------
def run_strategies(inventory, contract: dict, arrays: dict, configs) -> tuple[dict, dict]:
    output_arrays: dict[str, np.ndarray] = {
        "window_rows": output_window_rows(inventory.windows),
        "strategy_ids": np.asarray([item.strategy_id for item in configs]),
        "state_code_names": np.asarray([READY, TASK_CANDIDATE, WAIT_IDLE]),
        "reason_code_names": np.asarray([
            "none", CANDIDATE_OPEN, CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT, COMMAND_COMMIT, IDLE_RESET,
        ]),
    }
    metrics_payload = {
        "subject": SUBJECT,
        "seed": SEED,
        "selection_status": "none_all_cells_reported",
        "strategies": {},
    }
    summary: dict[str, dict] = {}
    strategy_results = {}
    for config in configs:
        result = feature_candidate_decisions(
            inventory.windows,
            arrays["stage1_logits"],
            arrays["stage2_logits"],
            arrays["stage2_features"],
            config,
        )
        evaluated = evaluate_online_events(
            inventory.segments, inventory.events, inventory.windows,
            result.policy.decisions, mode=STATEFUL_CANDIDATE,
        )
        _check_metric_inventory(evaluated, contract)
        identifier = config.strategy_id
        strategy_results[identifier] = result
        metrics_payload["strategies"][identifier] = {
            "config": asdict(config),
            "metrics": evaluated,
        }
        summary[identifier] = _strategy_metrics(evaluated)
        decisions, policy_trace, trace = result.policy.decisions, result.policy.trace, result.trace
        for name, values, dtype in (
            ("emitted", [item.emitted_class for item in decisions], np.int8),
            ("before", [STATE_CODE[item.decision_state_before] for item in decisions], np.uint8),
            ("after", [STATE_CODE[item.decision_state_after] for item in decisions], np.uint8),
            ("reason", [REASON_CODE[item.transition_reason] for item in decisions], np.uint8),
            ("candidate_age_before", [item.candidate_windows_before for item in policy_trace], np.int64),
            ("candidate_age_after", [item.candidate_windows_after for item in policy_trace], np.int64),
            ("task_on", [item.evidence.task_on for item in trace], np.bool_),
            ("task_hold", [item.evidence.task_hold for item in trace], np.bool_),
            ("idle_reset", [item.evidence.idle_reset for item in trace], np.bool_),
            ("stage2_commit_class", [item.evidence.stage2_commit_class for item in trace], np.int8),
            ("stage2_candidate_count", [item.stage2_candidate_window_count for item in trace], np.int64),
            ("stage2_top_class", [item.stage2_top_class for item in trace], np.int8),
            ("base_logit_commit_class", [item.base_logit_commit_class for item in trace], np.int8),
            ("feature_metric_available", [item.feature_metric_available for item in trace], np.bool_),
            ("feature_pass", [item.feature_pass for item in trace], np.bool_),
            ("feature_pass_streak", [item.feature_pass_streak for item in trace], np.int64),
        ):
            output_arrays[f"{identifier}_{name}"] = np.asarray(values, dtype=dtype)
        for name in (
            "stage1_filtered_task_probability", "stage1_filtered_delta",
            "stage2_top_probability", "stage2_probability_gap", "feature_metric_value",
        ):
            output_arrays[f"{identifier}_{name}"] = np.asarray(
                [getattr(item, name) for item in trace], dtype=np.float32,
            )

    logit_reference = logit_candidate_decisions(
        inventory.windows,
        arrays["stage1_logits"],
        arrays["stage2_logits"],
        configs[0].base_logit_strategy,
    )
    reference_exact = (
        strategy_results["logit_only_reference"].policy == logit_reference.policy
    )
    if not reference_exact:
        raise RuntimeError("无特征门控参考不能复现原 logit 候选策略")
    return {
        "metrics": metrics_payload,
        "summary": summary,
        "logit_reference_policy_exact": reference_exact,
        "reference_policy": logit_reference.policy,
    }, output_arrays


def _source_hashes() -> dict[str, str]:
    return {
        "candidate_state_policy": file_hash(EVAL_DIR / "candidate_state_policy.py"),
        "logit_candidate_strategies": file_hash(EVAL_DIR / "logit_candidate_strategies.py"),
        "feature_candidate_strategies": file_hash(EVAL_DIR / "feature_candidate_strategies.py"),
        "feature_pilot_runner": file_hash(Path(__file__)),
        "candidate_logit_matrix_runner": file_hash(EVAL_DIR / "run_candidate_logit_matrix.py"),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "feature_preflight_runner": file_hash(EVAL_DIR / "run_epoch50_feature_preflight.py"),
        "frozen_inventory_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
        "single_window_runner": file_hash(EVAL_DIR / "run_epoch50_online_oof.py"),
        "bundle_reader": file_hash(TRAIN_DIR / "oof_training_bundle.py"),
    }


def _verify_runtime_identity(
    input_root: Path,
    config_path: Path,
    run_git: dict,
    environment: dict,
    sources: dict,
    config_sha256: str,
    input_identity: dict,
) -> None:
    """计算完成后和正式产物写完后各复核一次，变化时不写 PASS 清单。"""
    if (
        git_state() != run_git
        or runtime_environment() != environment
        or _source_hashes() != sources
        or file_hash(config_path) != config_sha256
        or file_hash(input_root / "run_manifest.json") != input_identity["manifest_sha256"]
        or file_hash(input_root / input_identity["artifact_file"])
        != input_identity["artifact_sha256"]
    ):
        raise RuntimeError("pilot 运行期间 Git、源码、配置、解释器或输入身份发生变化")


# ---------- 主入口：只做一个机制 pilot；运行中任何身份变化都会使结果失败 ----------
def run(args: argparse.Namespace) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.policy_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_git = git_state()
    environment = runtime_environment()
    sources = _source_hashes()
    config_sha256 = file_hash(config_path)
    contract, configs = load_contract(config_path)
    context, inventory, inventory_contract, _ = _load_subject_inventory(SUBJECT)
    artifact_identity = artifact_contract(context.manifest)
    expected_rows = output_window_rows(inventory.windows)
    arrays, _, input_identity = load_feature_input(input_root, expected_rows)
    result, trajectory_arrays = run_strategies(
        inventory, inventory_contract, arrays, configs,
    )
    scale = summarize_feature_scale(
        inventory.windows,
        arrays["stage1_features"],
        arrays["stage2_features"],
        result["reference_policy"],
    )
    completed = datetime.now(timezone.utc).isoformat()

    _verify_runtime_identity(
        input_root, config_path, run_git, environment,
        sources, config_sha256, input_identity,
    )

    metrics_path = output_root / "candidate_feature_metrics.json"
    trajectory_path = output_root / "candidate_feature_trajectories.npz"
    scale_path = output_root / "input_feature_scale_summary.json"
    atomic_json(metrics_path, result["metrics"])
    atomic_npz(trajectory_path, **trajectory_arrays)
    atomic_json(scale_path, scale)
    artifacts = {
        "metrics": {"file": metrics_path.name, "sha256": file_hash(metrics_path)},
        "trajectories": {"file": trajectory_path.name, "sha256": file_hash(trajectory_path)},
        "input_feature_scale": {"file": scale_path.name, "sha256": file_hash(scale_path)},
    }
    # 产物写出也属于运行过程；只有写完后身份仍一致才允许落 PASS manifest。
    _verify_runtime_identity(
        input_root, config_path, run_git, environment,
        sources, config_sha256, input_identity,
    )
    manifest = {
        "status": "PASS",
        "claim_status": "S01_SEED42_MECHANISM_PILOT_ONLY",
        "protocol_id": EXPECTED_PROTOCOL,
        "selection_status": "none_all_cells_reported",
        "selection_warning": (
            "All cells use the same session0 OOF pilot; no cell is an unbiased selected result."
        ),
        "subject": SUBJECT,
        "seed": SEED,
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        **artifact_identity,
        "strategy_ids": [item.strategy_id for item in configs],
        "policy_contract": contract,
        "policy_contract_file": display_path(config_path),
        "policy_contract_sha256": config_sha256,
        "input_feature_preflight": input_identity,
        "inventory_contract": inventory_contract,
        "logit_reference_policy_exact": result["logit_reference_policy_exact"],
        "artifacts": artifacts,
        "summary": result["summary"],
        "source_sha256": sources,
        "runtime": {
            "started_at_utc": started,
            "completed_at_utc": completed,
            "environment": environment,
            "git": run_git,
        },
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
