"""严格配对比较 LD-GRU 保留/屏蔽三个 Stage 1 token 的正式结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ld_gru_training import atomic_json, file_hash
from run_hard_vote_matrix import _atomic_csv
from run_ld_gru_nested_loso import SUMMARY_FIELDS


FULL_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_ld_gru_nested_loso_v1"
MASK_PROTOCOL = "bnci2014001_s01_s09_epoch50_causal_ld_gru_mask_stage1_nested_loso_v1"
SUBJECTS = tuple(range(1, 10))
SEEDS = (42, 43, 44)
ABLATIONS = ("stop_only", "stop_residual")
REFERENCES = (
    "single_window_stateful",
    "n5_k3_hard_vote",
    "dual_ewma_drop_abort_c055_r020_l1",
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def read_rows(path: Path) -> dict[tuple[int, int, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["subject"]), int(row["seed"]), row["policy"])
        if key in result:
            raise RuntimeError(f"汇总表包含重复键: {key}")
        result[key] = row
    return result


def numeric(value: str) -> float | None:
    return None if value == "" else float(value)


def mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


# ---------- 顶层兼容性：轴、输入、评估器、伪迹规则与参考策略必须一致 ----------
def verify_manifests(full: dict, masked: dict) -> None:
    if (
        full.get("status") != "PASS"
        or full.get("scope") != "full"
        or full.get("protocol_id") != FULL_PROTOCOL
    ):
        raise RuntimeError("保留 token 的正式结果未 PASS 或协议身份错误")
    if (
        masked.get("status") != "PASS"
        or masked.get("scope") != "full"
        or masked.get("protocol_id") != MASK_PROTOCOL
        or masked.get("token_mode") != "mask_stage1"
    ):
        raise RuntimeError("屏蔽 token 的正式结果未 PASS 或协议身份错误")
    for payload in (full, masked):
        if (
            tuple(payload.get("outer_subjects", [])) != SUBJECTS
            or tuple(payload.get("base_seeds", [])) != SEEDS
            or tuple(payload.get("ablations", [])) != ABLATIONS
            or len(payload.get("results", {})) != 54
        ):
            raise RuntimeError("正式结果没有覆盖完整配对实验轴")
    if full["input_root_manifest"]["sha256"] != masked["input_root_manifest"]["sha256"]:
        raise RuntimeError("两组没有消费同一个冻结 OOF 输入母清单")
    if full["anchor_config"]["sha256"] != masked["anchor_config"]["sha256"]:
        raise RuntimeError("两组 FAR anchor 配置不一致")
    for field in ("artifact_policy", "artifact_policy_binding", "segment_policy"):
        if full.get(field) != masked.get(field):
            raise RuntimeError(f"两组 {field} 不一致")
    shared_sources = set(full["source_sha256"]) & set(masked["source_sha256"])
    for name in sorted(shared_sources - {"runner", "policy", "trainer"}):
        if full["source_sha256"][name] != masked["source_sha256"][name]:
            raise RuntimeError(f"共享评估依赖源码不一致: {name}")
    if full["source_sha256"]["protocol_metrics"] != masked["source_sha256"]["protocol_metrics"]:
        raise RuntimeError("正式 evaluator 源码不一致")


def expected_axis(policy: str) -> set[tuple[int, int, str]]:
    return {(subject, seed, policy) for subject in SUBJECTS for seed in SEEDS}


def run(full_root: Path, masked_root: Path, output_root: Path) -> dict:
    full_root = full_root.resolve()
    masked_root = masked_root.resolve()
    output_root = output_root.resolve()
    full_manifest_path = full_root / "run_manifest.json"
    masked_manifest_path = masked_root / "run_manifest.json"
    full_manifest = read_json(full_manifest_path)
    masked_manifest = read_json(masked_manifest_path)
    verify_manifests(full_manifest, masked_manifest)

    full_table = full_root / full_manifest["csv_artifacts"]["held_out_results"]["file"]
    masked_table = masked_root / masked_manifest["csv_artifacts"]["held_out_results"]["file"]
    full_rows = read_rows(full_table)
    masked_rows = read_rows(masked_table)

    # 参考策略是输入和 evaluator 的端到端哨兵，逐字符串相同才允许比较学习策略。
    for policy in REFERENCES:
        axis = expected_axis(policy)
        if not axis <= set(full_rows) or not axis <= set(masked_rows):
            raise RuntimeError(f"参考策略 {policy} 配对轴不完整")
        for key in axis:
            for field in SUMMARY_FIELDS:
                if full_rows[key][field] != masked_rows[key][field]:
                    raise RuntimeError(f"参考结果发生变化: {key}, {field}")

    paired_rows: list[dict] = []
    seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for ablation in ABLATIONS:
        full_policy = f"ld_gru_{ablation}"
        mask_policy = f"ld_gru_mask_stage1_{ablation}"
        if set(full_rows) & expected_axis(full_policy) != expected_axis(full_policy):
            raise RuntimeError(f"保留组 {ablation} 轴不完整")
        if set(masked_rows) & expected_axis(mask_policy) != expected_axis(mask_policy):
            raise RuntimeError(f"屏蔽组 {ablation} 轴不完整")
        per_metric_seed: dict[str, list[dict]] = {field: [] for field in SUMMARY_FIELDS}
        for seed in SEEDS:
            for field in SUMMARY_FIELDS:
                full_values: list[float | None] = []
                mask_values: list[float | None] = []
                deltas: list[float | None] = []
                for subject in SUBJECTS:
                    full_value = numeric(full_rows[(subject, seed, full_policy)][field])
                    mask_value = numeric(masked_rows[(subject, seed, mask_policy)][field])
                    delta = None if full_value is None or mask_value is None else mask_value - full_value
                    full_values.append(full_value)
                    mask_values.append(mask_value)
                    deltas.append(delta)
                    paired_rows.append({
                        "subject": subject,
                        "seed": seed,
                        "ablation": ablation,
                        "metric": field,
                        "full": full_value,
                        "mask_stage1": mask_value,
                        "delta_mask_minus_full": delta,
                    })
                seed_row = {
                    "seed": seed,
                    "ablation": ablation,
                    "metric": field,
                    "full_equal_subject_mean": mean(full_values),
                    "mask_equal_subject_mean": mean(mask_values),
                    "paired_delta_equal_subject_mean": mean(deltas),
                }
                seed_rows.append(seed_row)
                per_metric_seed[field].append(seed_row)
        for field in SUMMARY_FIELDS:
            cells = per_metric_seed[field]
            full_seed = [row["full_equal_subject_mean"] for row in cells]
            mask_seed = [row["mask_equal_subject_mean"] for row in cells]
            delta_seed = [row["paired_delta_equal_subject_mean"] for row in cells]
            finite_delta = np.asarray([value for value in delta_seed if value is not None], dtype=np.float64)
            aggregate_rows.append({
                "ablation": ablation,
                "metric": field,
                "full_mean_over_paired_seeds": mean(full_seed),
                "mask_mean_over_paired_seeds": mean(mask_seed),
                "paired_delta_mean_over_seeds": mean(delta_seed),
                "paired_delta_population_std_over_seeds": (
                    None if finite_delta.size == 0 else float(np.std(finite_delta, ddof=0))
                ),
            })

    contract = {
        "full_manifest_sha256": file_hash(full_manifest_path),
        "masked_manifest_sha256": file_hash(masked_manifest_path),
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "ablations": list(ABLATIONS),
        "metrics": list(SUMMARY_FIELDS),
        "delta_direction": "mask_stage1_minus_full",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "comparison_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise RuntimeError("输出目录已绑定不同的配对比较合同")
    elif any(output_root.iterdir()):
        raise FileExistsError("非空比较目录缺少同一合同")
    else:
        atomic_json(contract_path, contract)
    paired_path = output_root / "paired_subject_seed_metrics.csv"
    seed_path = output_root / "paired_seed_macro.csv"
    aggregate_path = output_root / "paired_aggregate.csv"
    _atomic_csv(
        paired_path,
        ["subject", "seed", "ablation", "metric", "full", "mask_stage1", "delta_mask_minus_full"],
        paired_rows,
    )
    _atomic_csv(
        seed_path,
        ["seed", "ablation", "metric", "full_equal_subject_mean", "mask_equal_subject_mean", "paired_delta_equal_subject_mean"],
        seed_rows,
    )
    _atomic_csv(
        aggregate_path,
        ["ablation", "metric", "full_mean_over_paired_seeds", "mask_mean_over_paired_seeds", "paired_delta_mean_over_seeds", "paired_delta_population_std_over_seeds"],
        aggregate_rows,
    )
    manifest = {
        "status": "PASS",
        "comparison": "paired_stage1_token_visibility_ablation",
        "contract": {"file": contract_path.name, "sha256": file_hash(contract_path)},
        "artifacts": {
            "paired_subject_seed_metrics": {"file": paired_path.name, "sha256": file_hash(paired_path)},
            "paired_seed_macro": {"file": seed_path.name, "sha256": file_hash(seed_path)},
            "paired_aggregate": {"file": aggregate_path.name, "sha256": file_hash(aggregate_path)},
        },
        "aggregate_rows": aggregate_rows,
    }
    atomic_json(output_root / "comparison_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_root", type=Path)
    parser.add_argument("masked_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.full_root, args.masked_root, args.output_root)
