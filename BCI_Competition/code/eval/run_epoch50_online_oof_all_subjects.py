"""顺序运行 9 个被试的固定 epoch50 因果 OOF 基线，并做等权被试宏汇总。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_epoch50_online_oof import (
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    PROJECT_ROOT,
    STATEFUL_STRICT,
    STATELESS_DIAGNOSTIC,
    atomic_json,
    file_hash,
    git_state,
    run as run_single_subject,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_v1"
)


# ---------- 宏汇总：先在每个被试内部保留 seed，再对被试等权平均 ----------
def _metric_statistics(values: list[float]) -> dict:
    return {
        "mean": None if not values else float(np.mean(values)),
        "population_std": None if not values else float(np.std(values, ddof=0)),
        "valid_count": len(values),
    }


def verify_child_artifacts(
    child_root: Path,
    manifest: dict,
    subject: int,
    seeds: tuple[int, ...],
    *,
    expected_source_hashes: dict[str, str] | None = None,
) -> None:
    """复核子清单；新运行绑定当前源码，历史消费者可传入冻结源码合同。"""
    expected_jobs = {
        (fold, seed, stage)
        for fold in range(6) for seed in seeds for stage in (1, 2)
    }
    records = manifest.get("checkpoint_records", [])
    actual_jobs = {
        (record.get("fold"), record.get("seed"), record.get("stage"))
        for record in records
    }
    window_count = manifest.get("inventory_contract", {}).get("inventory", {}).get(
        "window_count",
    )
    expected_sources = {
        "runner": PROJECT_ROOT / "code" / "eval" / "run_epoch50_online_oof.py",
        "protocol_metrics": PROJECT_ROOT / "code" / "eval" / "protocol_metrics.py",
        "oof_training_bundle_reader": (
            PROJECT_ROOT / "code" / "train" / "oof_training_bundle.py"
        ),
        "model_factory": PROJECT_ROOT / "code" / "models" / "model_factory.py",
        "eegnet": PROJECT_ROOT / "code" / "models" / "models" / "eegnet.py",
    }
    current_source_hashes = {
        role: file_hash(path)
        for role, path in expected_sources.items()
    }
    source_contract = (
        current_source_hashes
        if expected_source_hashes is None
        else expected_source_hashes
    )
    if set(source_contract) != set(expected_sources) or any(
        not isinstance(value, str) or len(value) != 64
        for value in source_contract.values()
    ):
        raise RuntimeError(f"Subject {subject} 源码哈希合同不完整")
    source_hashes = manifest.get("source_sha256", {})
    if (
        actual_jobs != expected_jobs
        or len(records) != len(expected_jobs)
        or source_hashes != source_contract
        or any(
            record.get("validation_runs") != [record.get("fold")]
            or record.get("fold") in record.get("train_runs", [])
            or not np.isfinite(record.get("saved_oof_reproduction_max_abs_error", np.nan))
            for record in records
        )
        or any(
            sum(
                record["continuous_window_count"]
                for record in records
                if record["seed"] == seed and record["stage"] == stage
            ) != window_count
            for seed in seeds for stage in (1, 2)
        )
    ):
        raise RuntimeError(f"Subject {subject} checkpoint 网格或 reader 来源不完整")

    artifacts: list[dict] = [manifest.get("run_log", {})]
    seed_artifacts = manifest.get("seed_artifacts", {})
    if set(seed_artifacts) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"Subject {subject} seed 产物集合不完整")
    for seed in seeds:
        roles = seed_artifacts[str(seed)]
        if set(roles) != {"scores_and_decisions", "stateless_metrics", "stateful_metrics"}:
            raise RuntimeError(f"Subject {subject} seed {seed} 产物角色不完整")
        artifacts.extend(roles.values())

    for artifact in artifacts:
        relative = Path(str(artifact.get("file", "")))
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != str(relative):
            raise RuntimeError(f"Subject {subject} 含非法产物路径")
        path = child_root / relative
        if not path.is_file() or file_hash(path) != artifact.get("sha256"):
            raise RuntimeError(f"Subject {subject} 子产物哈希不一致: {relative}")

    log = json.loads((child_root / manifest["run_log"]["file"]).read_text(encoding="utf-8"))
    if (
        log.get("status") != "PASS"
        or log.get("protocol_id") != manifest.get("protocol_id")
        or log.get("subject") != subject
        or tuple(log.get("seeds", [])) != seeds
    ):
        raise RuntimeError(f"Subject {subject} 运行日志与 manifest 不一致")


def aggregate_subject_summaries(
    manifests: dict[int, dict],
    subjects: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict:
    """生成不按事件数加权的被试宏平均，并保留被试间与 seed 间两种波动。"""
    if set(manifests) != set(subjects):
        raise RuntimeError("被试 manifest 集合与请求范围不一致")

    result: dict[str, dict] = {}
    for subject in subjects:
        manifest = manifests[subject]
        expected_protocol = (
            f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_oof_v1"
            if seeds == KNOWN_SEEDS
            else f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_"
            f"seed_subset_{'_'.join(map(str, seeds))}_diagnostic_v1"
        )
        if (
            manifest.get("status") != "PASS"
            or manifest.get("subject") != subject
            or manifest.get("protocol_id") != expected_protocol
            or tuple(manifest.get("seeds", [])) != seeds
            or manifest.get("included_session") != 0
            or manifest.get("test_session_access") != "forbidden_and_not_loaded"
        ):
            raise RuntimeError(f"Subject {subject} 单被试 manifest 合同非法")

    for mode in (STATELESS_DIAGNOSTIC, STATEFUL_STRICT):
        first_metrics = manifests[subjects[0]]["summary"][mode]["per_seed"][str(seeds[0])]
        fields = tuple(first_metrics)

        # 同一 seed 下先对 9 个被试等权平均，避免伪迹较少的被试因事件更多而占更大权重。
        per_seed_subject_macro: dict[str, dict] = {}
        for seed in seeds:
            metrics: dict[str, dict] = {}
            for field in fields:
                values = [
                    manifests[subject]["summary"][mode]["per_seed"][str(seed)][field]
                    for subject in subjects
                ]
                finite = [float(value) for value in values if value is not None]
                metrics[field] = {
                    "mean": None if not finite else float(np.mean(finite)),
                    "valid_subject_count": len(finite),
                }
            per_seed_subject_macro[str(seed)] = metrics

        # 三个配对 seed 的被试宏平均再给出描述性均值和总体标准差。
        aggregate_across_seeds: dict[str, dict] = {}
        for field in fields:
            values = [
                per_seed_subject_macro[str(seed)][field]["mean"]
                for seed in seeds
                if per_seed_subject_macro[str(seed)][field]["mean"] is not None
            ]
            aggregate_across_seeds[field] = _metric_statistics(values)

        # 另行报告被试间波动：每个被试先对 seed 求均值，再在被试间求标准差。
        across_subjects: dict[str, dict] = {}
        for field in fields:
            by_subject = {
                str(subject): manifests[subject]["summary"][mode]["aggregate"][field]["mean"]
                for subject in subjects
            }
            finite = [float(value) for value in by_subject.values() if value is not None]
            across_subjects[field] = {
                **_metric_statistics(finite),
                "per_subject_seed_mean": by_subject,
            }

        result[mode] = {
            "per_seed_subject_macro": per_seed_subject_macro,
            "aggregate_across_seeds": aggregate_across_seeds,
            "across_subjects_from_seed_means": across_subjects,
        }
    return result


# ---------- 总入口：每个被试独立落盘，跨被试清单只引用子清单哈希 ----------
def run(args: argparse.Namespace) -> dict:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    subjects = tuple(dict.fromkeys(int(value) for value in args.subjects))
    seeds = tuple(sorted(set(int(value) for value in args.seeds)))
    if not subjects or any(subject not in KNOWN_SUBJECTS for subject in subjects):
        raise ValueError(f"subjects 只能取 {KNOWN_SUBJECTS}")
    if not seeds or any(seed not in KNOWN_SEEDS for seed in seeds):
        raise ValueError(f"seeds 只能取 {KNOWN_SEEDS}")

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifests: dict[int, dict] = {}
    children: dict[str, dict] = {}
    for subject in subjects:
        child_root = output_root / f"subject_{subject:02d}"
        print(f"Subject {subject}: 开始连续 OOF 推理", flush=True)
        child_args = argparse.Namespace(
            subject=subject,
            bundle_manifest=None,
            checkpoint_root=None,
            inventory_contract=None,
            output_root=child_root,
            device=args.device,
            seeds=list(seeds),
            verbose=False,
        )
        run_single_subject(child_args)
        child_manifest_path = child_root / "run_manifest.json"
        # 聚合消费真实落盘 JSON，而非内存对象，顺便统一 JSON 对整数键的字符串化语义。
        manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
        verify_child_artifacts(child_root, manifest, subject, seeds)
        manifests[subject] = manifest
        children[str(subject)] = {
            "manifest": str(child_manifest_path.relative_to(output_root)),
            "manifest_sha256": file_hash(child_manifest_path),
            "window_count": manifest["inventory_contract"]["inventory"]["window_count"],
            "event_count": manifest["inventory_contract"]["inventory"]["event_count"],
            "claim_status": manifest["claim_status"],
        }
        print(
            f"Subject {subject}: PASS, windows={children[str(subject)]['window_count']}, "
            f"events={children[str(subject)]['event_count']}",
            flush=True,
        )

    summary = aggregate_subject_summaries(manifests, subjects, seeds)
    current_git = git_state()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    full_scope = subjects == KNOWN_SUBJECTS and seeds == KNOWN_SEEDS
    scope_id = "s01_s09" if full_scope else "subset_" + "_".join(
        f"s{subject:02d}" for subject in subjects
    ) + "_seeds_" + "_".join(map(str, seeds))
    protocol_id = f"bnci2014001_{scope_id}_epoch50_causal_single_window_oof_v1"
    log_path = output_root / "run_log.json"
    atomic_json(log_path, {
        "status": "PASS",
        "protocol_id": protocol_id,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "children": children,
        "summary": summary,
    })
    log_artifact = {"file": log_path.name, "sha256": file_hash(log_path)}
    manifest = {
        "status": "PASS",
        "claim_status": (
            "SUBSET_DIAGNOSTIC" if not full_scope
            else "PRECOMMIT_DIAGNOSTIC" if current_git["dirty"] is not False
            else "CLEAN_COMMIT_FORMAL_CANDIDATE"
        ),
        "protocol_id": protocol_id,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "aggregation_semantics": {
            "event_and_window_pooling_across_subjects": "forbidden",
            "primary_cross_subject_summary": "equal_weight_subject_macro_per_paired_seed",
            "seed_summary": "mean_and_population_std_of_paired_seed_subject_macros",
            "subject_variability": "population_std_of_per_subject_seed_means",
        },
        "children": children,
        "run_log": log_artifact,
        "summary": summary,
        "runtime_git": current_git,
        "source_sha256": {
            "multi_subject_runner": file_hash(Path(__file__)),
            "single_subject_runner": file_hash(
                PROJECT_ROOT / "code" / "eval" / "run_epoch50_online_oof.py",
            ),
        },
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(KNOWN_SUBJECTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(KNOWN_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
