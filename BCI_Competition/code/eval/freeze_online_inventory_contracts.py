"""从各被试 session0-only bundle 冻结连续在线评分库存合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_metrics import (
    STATELESS_DIAGNOSTIC,
    DecisionRecord,
    evaluate_online_events,
)
from run_epoch50_online_oof import (
    KNOWN_SUBJECTS,
    atomic_json,
    build_online_inventory,
    canonical_hash,
    default_subject_paths,
    inventory_contract_protocol_id,
    verify_inventory_contract,
)
from oof_training_bundle import BUNDLE_ID, artifact_contract, load_bundle


# ---------- 路径版本：默认复核历史 v1，显式 bundle-root 则读取新建的 v2 bundle ----------
def resolve_bundle_manifest(subject: int, bundle_root: Path | None) -> Path:
    if bundle_root is None:
        return default_subject_paths(subject).bundle_manifest.resolve()
    return (Path(bundle_root).resolve() / BUNDLE_ID.format(subject=subject) / "manifest.json")


# ---------- 合同派生：只使用 session0 冻结 bundle，不访问原始 MAT 或测试 session ----------
def derive_contract(subject: int, bundle_manifest: Path | None = None) -> dict:
    manifest_path = (
        default_subject_paths(subject).bundle_manifest
        if bundle_manifest is None else Path(bundle_manifest)
    ).resolve()
    context = load_bundle(manifest_path, verify_hashes=True)
    artifact_identity = artifact_contract(context.manifest)
    inventory = build_online_inventory(context)
    no_command = [
        DecisionRecord(
            *window.key, window.window_index,
            window.window_start_sample, window.window_stop_sample,
        )
        for window in inventory.windows
    ]
    result = evaluate_online_events(
        inventory.segments,
        inventory.events,
        inventory.windows,
        no_command,
        mode=STATELESS_DIAGNOSTIC,
    )
    if (
        result["scorable_event_count"] != len(inventory.events)
        or result["unscorable_event_count"] != 0
        or result["miss_rate"] != 1.0
        or result["idle_false_command_count"] != 0
    ):
        raise RuntimeError(f"Subject {subject} 的 NO_COMMAND 库存控制失败")

    contract = {
        "protocol_id": inventory_contract_protocol_id(context.manifest),
        "subject": subject,
        "included_session": 0,
        "test_session_access": "forbidden",
        "native_sampling_rate": 250,
        "window_samples": 500,
        "step_samples": 125,
        "event_margin_samples": 125,
        "source_bundle": {
            "protocol_id": context.manifest["protocol_id"],
            "manifest_sha256": context.manifest_sha256,
            "index_sha256": context.manifest["index_sha256"],
        },
        "derivation": {
            "segments": "causal segment formal_start_native to formal_stop_native",
            "windows": "start at each formal segment start; 500 samples; 125-sample step; reindex from zero",
            "events": "group the five clean task windows by run, segment and trial; onset=min(start); offset=max(stop)",
            "event_id": "s0_r{run}_t{trial}",
        },
        "inventory": {
            "segment_count": result["scoring_segment_count"],
            "segment_inventory_sha256": result["scoring_segment_inventory_sha256"],
            "fully_warmup_excluded_segment_count": (
                inventory.fully_warmup_excluded_segment_count
            ),
            "fully_warmup_excluded_samples": inventory.fully_warmup_excluded_samples,
            "zero_window_segment_count": result["zero_window_segment_count"],
            "zero_window_segment_samples": result["zero_window_segment_samples"],
            "trailing_unwindowed_samples": result["trailing_unwindowed_samples"],
            "window_count": result["expected_window_count"],
            "window_inventory_sha256": result["expected_window_inventory_sha256"],
            "event_count": result["event_count"],
            "event_inventory_sha256": result["event_inventory_sha256"],
            "valid_idle_seconds": result["valid_idle_seconds"],
        },
        "per_run_window_count": {
            str(run): sum(window.run_id == run for window in inventory.windows)
            for run in range(6)
        },
        "per_run_event_count": {
            str(run): sum(event.run_id == run for event in inventory.events)
            for run in range(6)
        },
    }
    # 历史 v1 文件保持逐字兼容；新建 v2 必须把伪迹和 segment 合同写入库存本身。
    if artifact_identity["artifact_policy_binding"] == "explicit_bundle_manifest":
        contract.update(artifact_identity)
    return contract


# ---------- 写入策略：v1/v2 使用不同文件名；已有合同仍必须逐字段完全相等 ----------
def freeze_or_verify(
    subject: int,
    write_missing: bool,
    bundle_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    bundle_manifest = resolve_bundle_manifest(subject, bundle_root)
    derived = derive_contract(subject, bundle_manifest)
    contract_dir = (
        default_subject_paths(subject).inventory_contract.parent
        if output_dir is None else Path(output_dir)
    ).resolve()
    contract_path = contract_dir / f"{derived['protocol_id']}.json"
    if contract_path.exists():
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved != derived:
            raise RuntimeError(f"Subject {subject} 已冻结合同与当前派生结果不一致")
        action = "verified_existing"
    else:
        if not write_missing:
            raise FileNotFoundError(
                f"Subject {subject} 合同不存在；确认后使用 --write-missing 创建",
            )
        atomic_json(contract_path, derived)
        action = "created_missing"

    # 写后再次走正式运行器的合同验证，防止生成器与消费端接口漂移。
    context = load_bundle(bundle_manifest, verify_hashes=True)
    verify_inventory_contract(context, build_online_inventory(context), derived)
    return {
        "subject": subject,
        "action": action,
        "bundle_manifest": str(bundle_manifest),
        "contract": str(contract_path),
        "contract_canonical_sha256": canonical_hash(derived),
        **derived["inventory"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(KNOWN_SUBJECTS))
    parser.add_argument(
        "--bundle-root", type=Path,
        help="新建 v2 bundle 的父目录；缺省时只复核仓库既有 v1 复现实物",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="库存合同输出目录；文件名由 v1/v2 协议身份自动确定",
    )
    parser.add_argument("--write-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subjects = tuple(dict.fromkeys(args.subjects))
    if not subjects or any(subject not in KNOWN_SUBJECTS for subject in subjects):
        raise ValueError(f"subjects 只能取 {KNOWN_SUBJECTS}")
    records = [
        freeze_or_verify(
            subject, args.write_missing, args.bundle_root, args.output_dir,
        )
        for subject in subjects
    ]
    print(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False))
