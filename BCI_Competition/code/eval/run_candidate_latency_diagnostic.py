from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from candidate_latency_diagnostics import diagnose_candidate_latency
from logit_candidate_strategies import logit_candidate_decisions
from protocol_metrics import STATEFUL_CANDIDATE, evaluate_online_events
from run_candidate_logit_matrix import _check_metric_inventory
from run_commit_reset_matrix import (
    DEFAULT_POLICY_CONFIG as DEFAULT_ANCHOR_CONFIG,
    load_commit_reset_contract,
)
from run_epoch50_online_oof import (
    EVAL_DIR,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    TRAIN_DIR,
    atomic_json,
    core_metrics,
    display_path,
    file_hash,
    git_state,
    output_window_rows,
)
from run_hard_vote_matrix import (
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


EXPECTED_OUTPUT_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_candidate_latency_diagnostic_v1"
ANCHOR_CELL_ID = "c055_r020_l1"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_candidate_latency_diagnostic_v1"
)
DEFAULT_DIAGNOSTIC_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation"
    / "bnci2014001_candidate_latency_diagnostic_v1.json"
)

EXPECTED_CONTRACT = {
    "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
    "input_protocol_id": EXPECTED_INPUT_PROTOCOL,
    "included_session": 0,
    "test_session_access": "forbidden",
    "selection_status": "none_diagnostic_only",
    "anchor": {
        "source_config": "bnci2014001_commit_reset_matrix_v1.json",
        "cell_id": ANCHOR_CELL_ID,
        "reason": "historical dual_ewma_drop_abort regression anchor, not selected from this diagnostic",
    },
    "latency_clock": {
        "sampling_rate": 250,
        "window_samples": 500,
        "step_samples": 125,
        "definition": "decision_sample/250 - MI_onset_sample/250",
        "causal_filter_group_delay": "included_and_not_subtracted",
        "wall_clock_compute": "not_measured",
    },
    "truth_usage": "post_inference_oracle_diagnostics_only",
    "oracles": [
        "raw_correct_top1",
        "raw_correct_confident",
        "open_ewma_correct_top1",
        "open_ewma_correct_confident_min1",
        "open_ewma_correct_confident_min2",
    ],
    "label_free_first_crossings": [
        "raw_first_confident",
        "open_ewma_first_confident_min1",
        "open_ewma_first_confident_min2",
    ],
    "label_free_scope": (
        "class-label-free selection within post-inference truth-defined event "
        "onset/offset and margin"
    ),
    "required_pairing": [
        "same baseline-correct event latency headroom",
        "baseline-miss correct-class post-MI spillover truth-aware rescue",
        "baseline correct rate reported beside latency",
        "WAIT_IDLE locked-event rates",
        "truth oracle preceded by an earlier wrong label-free crossing",
    ],
}

SUMMARY_FIELDS = (
    "baseline_correct_event_rate",
    "baseline_correct_latency_median_seconds",
    "baseline_stage1_open_latency_median_seconds",
    "baseline_candidate_dwell_median_seconds",
    "baseline_extra_after_fixed_wait_median_seconds",
    "oracle_raw_confident_coverage_rate",
    "oracle_open_ewma_min1_coverage_rate",
    "oracle_open_ewma_min2_coverage_rate",
    "paired_min1_earlier_rate",
    "paired_min1_headroom_median_seconds",
    "paired_min2_earlier_rate",
    "paired_min2_headroom_median_seconds",
    "label_free_min1_coverage_rate",
    "label_free_min1_class_accuracy",
    "label_free_min1_correct_event_rate",
    "label_free_min1_earlier_correct_rate",
    "label_free_min1_wrong_on_baseline_correct_rate",
    "truth_min1_preceded_by_wrong_crossing_rate",
    "label_free_min2_coverage_rate",
    "label_free_min2_class_accuracy",
    "label_free_min2_correct_event_rate",
    "label_free_min2_earlier_correct_rate",
    "label_free_min2_wrong_on_baseline_correct_rate",
    "truth_min2_preceded_by_wrong_crossing_rate",
    "correct_class_spillover_event_rate",
    "spillover_min1_rescuable_rate",
    "spillover_min2_rescuable_rate",
    "first_eligible_window_wait_idle_rate",
    "fully_wait_idle_event_rate",
)


# ---------- 冻结合同：只接受历史锚点，不在本诊断结果上反向选择策略 ----------
def load_latency_contract(path: Path) -> dict:
    payload = _read_json(path)
    if payload != EXPECTED_CONTRACT:
        raise RuntimeError("候选延迟诊断配置与冻结首版合同不一致")
    return payload


def load_anchor(path: Path):
    payload, cells = load_commit_reset_contract(path)
    matches = [cell for cell in cells if cell.cell_id == ANCHOR_CELL_ID]
    if len(matches) != 1:
        raise RuntimeError("历史提交-复位配置缺少唯一延迟诊断锚点")
    anchor = matches[0]
    if (
        anchor.stage2_top_probability != 0.55
        or anchor.idle_reset_probability != 0.2
        or anchor.idle_reset_consecutive_windows != 1
        or anchor.logit_config.stage2_alpha != 0.5
        or anchor.logit_config.stage2_probability_gap != 0.15
        or anchor.logit_config.stage2_min_candidate_windows != 2
    ):
        raise RuntimeError("延迟诊断锚点参数发生漂移")
    return payload, anchor


def _flat_summary(diagnostics: dict) -> dict[str, float | None]:
    """抽取可跨被试等权汇总的字段；完整事件级证据仍保存在子产物中。"""
    summary = diagnostics["summary"]
    oracles = summary["truth_aware_oracles"]
    paired = summary["paired_baseline_correct"]
    crossings = summary["label_free_first_crossings"]
    spillover = summary["correct_class_post_mi_spillover"]
    min1 = "open_ewma_correct_confident_min1"
    min2 = "open_ewma_correct_confident_min2"
    crossing_min1 = "open_ewma_first_confident_min1"
    crossing_min2 = "open_ewma_first_confident_min2"
    return {
        "baseline_correct_event_rate": summary["baseline_correct_event_rate"],
        "baseline_correct_latency_median_seconds": summary["baseline_correct_latency_seconds"]["median"],
        "baseline_stage1_open_latency_median_seconds": (
            summary["baseline_correct_stage1_open_latency_seconds"]["median"]
        ),
        "baseline_candidate_dwell_median_seconds": (
            summary["baseline_correct_candidate_dwell_seconds"]["median"]
        ),
        "baseline_extra_after_fixed_wait_median_seconds": (
            summary["baseline_correct_extra_after_fixed_wait_seconds"]["median"]
        ),
        "oracle_raw_confident_coverage_rate": (
            oracles["raw_correct_confident"]["event_coverage_rate"]
        ),
        "oracle_open_ewma_min1_coverage_rate": oracles[min1]["event_coverage_rate"],
        "oracle_open_ewma_min2_coverage_rate": oracles[min2]["event_coverage_rate"],
        "paired_min1_earlier_rate": paired[min1]["earlier_rate_among_baseline_correct"],
        "paired_min1_headroom_median_seconds": paired[min1]["positive_headroom_seconds"]["median"],
        "paired_min2_earlier_rate": paired[min2]["earlier_rate_among_baseline_correct"],
        "paired_min2_headroom_median_seconds": paired[min2]["positive_headroom_seconds"]["median"],
        "label_free_min1_coverage_rate": crossings[crossing_min1]["opportunity_coverage_rate"],
        "label_free_min1_class_accuracy": crossings[crossing_min1]["class_accuracy_at_first_crossing"],
        "label_free_min1_correct_event_rate": crossings[crossing_min1]["correct_event_rate"],
        "label_free_min1_earlier_correct_rate": (
            crossings[crossing_min1]["earlier_and_correct_rate_among_baseline_correct"]
        ),
        "label_free_min1_wrong_on_baseline_correct_rate": (
            crossings[crossing_min1]["wrong_crossing_rate_among_baseline_correct"]
        ),
        "truth_min1_preceded_by_wrong_crossing_rate": (
            crossings[crossing_min1]["truth_oracle_preceded_by_wrong_crossing_rate"]
        ),
        "label_free_min2_coverage_rate": crossings[crossing_min2]["opportunity_coverage_rate"],
        "label_free_min2_class_accuracy": crossings[crossing_min2]["class_accuracy_at_first_crossing"],
        "label_free_min2_correct_event_rate": crossings[crossing_min2]["correct_event_rate"],
        "label_free_min2_earlier_correct_rate": (
            crossings[crossing_min2]["earlier_and_correct_rate_among_baseline_correct"]
        ),
        "label_free_min2_wrong_on_baseline_correct_rate": (
            crossings[crossing_min2]["wrong_crossing_rate_among_baseline_correct"]
        ),
        "truth_min2_preceded_by_wrong_crossing_rate": (
            crossings[crossing_min2]["truth_oracle_preceded_by_wrong_crossing_rate"]
        ),
        "correct_class_spillover_event_rate": spillover["event_rate"],
        "spillover_min1_rescuable_rate": spillover["rescue"][min1]["truth_aware_rescuable_rate"],
        "spillover_min2_rescuable_rate": spillover["rescue"][min2]["truth_aware_rescuable_rate"],
        "first_eligible_window_wait_idle_rate": summary["first_eligible_window_wait_idle_rate"],
        "fully_wait_idle_event_rate": summary["fully_wait_idle_event_rate"],
    }


# ---------- 单 seed：重放唯一锚点，再以真值只读方式生成事件级 oracle 诊断 ----------
def _run_seed(
    output_root: Path,
    inventory,
    inventory_contract: dict,
    seed: int,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    input_scores: dict,
    anchor,
) -> tuple[dict, dict]:
    strategy = logit_candidate_decisions(
        inventory.windows,
        stage1_logits,
        stage2_logits,
        anchor.logit_config,
        idle_reset_consecutive_windows=anchor.idle_reset_consecutive_windows,
    )
    evaluated = evaluate_online_events(
        inventory.segments,
        inventory.events,
        inventory.windows,
        strategy.policy.decisions,
        mode=STATEFUL_CANDIDATE,
    )
    _check_metric_inventory(evaluated, inventory_contract)
    diagnostics = diagnose_candidate_latency(
        inventory.events,
        inventory.windows,
        stage2_logits,
        strategy,
        evaluated,
        stage2_alpha=float(anchor.logit_config.stage2_alpha),
        stage2_top_probability=anchor.stage2_top_probability,
        stage2_probability_gap=anchor.logit_config.stage2_probability_gap,
        baseline_min_candidate_windows=anchor.logit_config.stage2_min_candidate_windows,
    )
    payload = {
        "subject": inventory.windows[0].subject_id,
        "seed": seed,
        "input_scores": input_scores,
        "anchor": anchor.public_config(),
        "baseline_metrics": core_metrics(evaluated),
        "diagnostics": diagnostics,
    }
    path = output_root / f"seed{seed}_candidate_latency_diagnostics.json"
    atomic_json(path, payload)
    return _flat_summary(diagnostics), {
        "input_scores": input_scores,
        "diagnostics": {"file": path.name, "sha256": file_hash(path)},
    }


# ---------- 分层汇总：每个 seed 先九被试等权，再汇总三个配对 seed ----------
def _aggregate(subject_summaries: dict[int, dict[int, dict]]) -> dict:
    if set(subject_summaries) != set(KNOWN_SUBJECTS):
        raise RuntimeError("候选延迟诊断缺少 Subject 1-9")
    per_seed: dict[str, dict] = {}
    for seed in KNOWN_SEEDS:
        rows = [subject_summaries[subject][seed] for subject in KNOWN_SUBJECTS]
        fields: dict[str, dict] = {}
        for field in SUMMARY_FIELDS:
            values = [float(row[field]) for row in rows if row[field] is not None]
            fields[field] = _statistics(values)
        per_seed[str(seed)] = {"equal_subject_macro": fields}

    across_seeds: dict[str, dict] = {}
    for field in SUMMARY_FIELDS:
        values = [
            per_seed[str(seed)]["equal_subject_macro"][field]["mean"]
            for seed in KNOWN_SEEDS
            if per_seed[str(seed)]["equal_subject_macro"][field]["mean"] is not None
        ]
        across_seeds[field] = _statistics(values)
    return {
        "aggregation": "equal_subject_macro_within_seed_then_mean_and_population_std_across_three_paired_seeds",
        "per_seed": per_seed,
        "across_seeds": across_seeds,
    }


def _write_csv(output_root: Path, subject_summaries: dict[int, dict[int, dict]], summary: dict) -> dict:
    per_subject_path = output_root / "candidate_latency_per_subject_seed.csv"
    aggregate_path = output_root / "candidate_latency_aggregate.csv"
    rows = [
        {"subject": subject, "seed": seed, **subject_summaries[subject][seed]}
        for subject in KNOWN_SUBJECTS for seed in KNOWN_SEEDS
    ]
    _atomic_csv(per_subject_path, ["subject", "seed", *SUMMARY_FIELDS], rows)
    aggregate_rows = []
    for field in SUMMARY_FIELDS:
        statistics = summary["across_seeds"][field]
        aggregate_rows.append({"metric": field, **statistics})
    _atomic_csv(
        aggregate_path,
        ["metric", "mean", "population_std", "valid_count"],
        aggregate_rows,
    )
    return {
        "per_subject_seed": {"file": per_subject_path.name, "sha256": file_hash(per_subject_path)},
        "aggregate": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
    }


def _verify_child(
    root: Path,
    manifest: dict,
    *,
    subject: int,
    source_hashes: dict[str, str],
    diagnostic_config_sha256: str,
    anchor_config_sha256: str,
    claim_status: str,
    anchor_public_config: dict,
) -> None:
    if (
        manifest.get("status") != "PASS"
        or manifest.get("claim_status") != claim_status
        or manifest.get("protocol_id") != EXPECTED_OUTPUT_PROTOCOL
        or manifest.get("selection_status") != "none_diagnostic_only"
        or manifest.get("truth_usage") != "post_inference_oracle_diagnostics_only"
        or manifest.get("subject") != subject
        or tuple(manifest.get("seeds", [])) != KNOWN_SEEDS
        or manifest.get("included_session") != 0
        or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        or manifest.get("source_sha256") != source_hashes
        or manifest.get("diagnostic_config_sha256") != diagnostic_config_sha256
        or manifest.get("anchor_config_sha256") != anchor_config_sha256
        or manifest.get("anchor") != anchor_public_config
    ):
        raise RuntimeError(f"Subject {subject} 候选延迟子清单合同非法")
    artifacts = [manifest.get("run_log", {})]
    for seed in KNOWN_SEEDS:
        roles = manifest.get("seed_artifacts", {}).get(str(seed), {})
        if set(roles) != {"input_scores", "diagnostics"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 诊断产物角色不完整")
        artifacts.append(roles["diagnostics"])
    for artifact in artifacts:
        path = _safe_artifact(root, artifact.get("file", ""))
        if not path.is_file() or file_hash(path) != artifact.get("sha256"):
            raise RuntimeError(f"Subject {subject} 候选延迟子产物哈希不一致")


# ---------- 主入口：只消费冻结 session0 OOF logits，输出不具备策略成绩含义 ----------
def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    diagnostic_config = Path(args.diagnostic_config).resolve()
    anchor_config = Path(args.anchor_config).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    diagnostic_config_sha256 = file_hash(diagnostic_config)
    anchor_config_sha256 = file_hash(anchor_config)
    input_master_path = input_root / "run_manifest.json"
    input_master_sha256 = file_hash(input_master_path)
    contract = load_latency_contract(diagnostic_config)
    _, anchor = load_anchor(anchor_config)
    if (
        file_hash(diagnostic_config) != diagnostic_config_sha256
        or file_hash(anchor_config) != anchor_config_sha256
    ):
        raise RuntimeError("读取延迟诊断配置期间文件发生变化")
    input_master, input_children = verify_input_root(input_root)
    input_child_identities = {
        subject: (
            _safe_artifact(input_root, input_master["children"][str(subject)]["manifest"]),
            input_master["children"][str(subject)]["manifest_sha256"],
        )
        for subject in KNOWN_SUBJECTS
    }
    source_hashes = _source_hashes()
    run_git = git_state()
    environment = runtime_environment()
    claim_status = (
        "PRECOMMIT_DIAGNOSTIC_ONLY" if run_git["dirty"] is not False
        else "CLEAN_COMMIT_DIAGNOSTIC_ONLY"
    )

    subject_summaries: dict[int, dict[int, dict]] = {}
    children: dict[str, dict] = {}
    for subject in KNOWN_SUBJECTS:
        subject_started = datetime.now(timezone.utc).isoformat()
        child_root = output_root / f"subject_{subject:02d}"
        child_root.mkdir(parents=True, exist_ok=False)
        context, inventory, inventory_contract, _ = _load_subject_inventory(subject)
        expected_rows = output_window_rows(inventory.windows)
        input_child_path = _safe_artifact(
            input_root, input_master["children"][str(subject)]["manifest"],
        )
        input_child = input_children[subject]
        seed_summaries: dict[int, dict] = {}
        seed_artifacts: dict[str, dict] = {}
        for seed in KNOWN_SEEDS:
            stage1, stage2, input_scores = _load_seed_logits(
                input_child_path.parent, input_child, seed, expected_rows,
            )
            seed_summaries[seed], seed_artifacts[str(seed)] = _run_seed(
                child_root, inventory, inventory_contract, seed,
                stage1, stage2, input_scores, anchor,
            )
        subject_summaries[subject] = seed_summaries
        completed = datetime.now(timezone.utc).isoformat()
        log_path = child_root / "run_log.json"
        atomic_json(log_path, {
            "status": "PASS",
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "claim_status": claim_status,
            "subject": subject,
            "seeds": list(KNOWN_SEEDS),
            "started_at_utc": subject_started,
            "completed_at_utc": completed,
            "runtime_environment": environment,
        })
        manifest = {
            "status": "PASS",
            "claim_status": claim_status,
            "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
            "selection_status": "none_diagnostic_only",
            "truth_usage": contract["truth_usage"],
            "subject": subject,
            "included_session": 0,
            "test_session_access": "forbidden_and_not_loaded",
            "seeds": list(KNOWN_SEEDS),
            "anchor": anchor.public_config(),
            "diagnostic_config_sha256": diagnostic_config_sha256,
            "anchor_config_sha256": anchor_config_sha256,
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
            child_root, _read_json(manifest_path), subject=subject,
            source_hashes=source_hashes,
            diagnostic_config_sha256=diagnostic_config_sha256,
            anchor_config_sha256=anchor_config_sha256,
            claim_status=claim_status,
            anchor_public_config=anchor.public_config(),
        )
        children[str(subject)] = {
            "manifest": str(manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(manifest_path),
            "window_count": inventory_contract["inventory"]["window_count"],
            "event_count": inventory_contract["inventory"]["event_count"],
        }
        print(f"Subject {subject}: PASS", flush=True)

    summary = _aggregate(subject_summaries)
    csv_artifacts = _write_csv(output_root, subject_summaries, summary)
    completed_at = datetime.now(timezone.utc).isoformat()
    if (
        git_state() != run_git
        or runtime_environment() != environment
        or _source_hashes() != source_hashes
        or file_hash(diagnostic_config) != diagnostic_config_sha256
        or file_hash(anchor_config) != anchor_config_sha256
        or file_hash(input_master_path) != input_master_sha256
        or any(file_hash(path) != expected for path, expected in input_child_identities.values())
    ):
        raise RuntimeError("延迟诊断期间 Git、源码、配置或解释器身份发生变化")
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "claim_status": claim_status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_environment": environment,
    })
    manifest = {
        "status": "PASS",
        "claim_status": claim_status,
        "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
        "selection_status": "none_diagnostic_only",
        "selection_warning": "Truth-aware oracle coverage is not a deployable policy score and cannot select a test policy.",
        "subjects": list(KNOWN_SUBJECTS),
        "seeds": list(KNOWN_SEEDS),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "truth_usage": contract["truth_usage"],
        "latency_clock": contract["latency_clock"],
        "anchor": anchor.public_config(),
        "diagnostic_config": {
            "file": display_path(diagnostic_config),
            "sha256": diagnostic_config_sha256,
        },
        "anchor_config": {
            "file": display_path(anchor_config),
            "sha256": anchor_config_sha256,
        },
        "input_root_manifest": {
            "file": display_path(input_master_path),
            "sha256": input_master_sha256,
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
    """冻结所有会实际影响输入验证、轨迹、事件匹配和诊断的本地源码。"""
    return {
        "candidate_state_policy": file_hash(EVAL_DIR / "candidate_state_policy.py"),
        "logit_candidate_strategies": file_hash(EVAL_DIR / "logit_candidate_strategies.py"),
        "candidate_latency_diagnostics": file_hash(EVAL_DIR / "candidate_latency_diagnostics.py"),
        "candidate_latency_runner": file_hash(Path(__file__)),
        "commit_reset_matrix_anchor_loader": file_hash(EVAL_DIR / "run_commit_reset_matrix.py"),
        "commit_reset_diagnostics_import": file_hash(EVAL_DIR / "commit_reset_diagnostics.py"),
        "candidate_logit_matrix_helpers": file_hash(EVAL_DIR / "run_candidate_logit_matrix.py"),
        "protocol_metrics": file_hash(EVAL_DIR / "protocol_metrics.py"),
        "frozen_input_reader": file_hash(EVAL_DIR / "run_hard_vote_matrix.py"),
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
    parser.add_argument("--diagnostic-config", type=Path, default=DEFAULT_DIAGNOSTIC_CONFIG)
    parser.add_argument("--anchor-config", type=Path, default=DEFAULT_ANCHOR_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
