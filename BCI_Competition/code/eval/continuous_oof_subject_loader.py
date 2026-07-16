"""装载九被试连续 OOF logits 与 session0 真值，不构造任何候选态。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from online_truth_inventory import load_truth_inventory
from run_epoch50_online_oof import (
    KNOWN_SUBJECTS,
    _build_online_signal_inventory,
    build_online_inventory,
    default_subject_paths,
    display_path,
    output_window_rows,
    verify_inventory_contract,
)
from run_hard_vote_matrix import (
    _load_seed_logits,
    _portable_path_text,
    _read_json,
    _safe_artifact,
    _seed_score_path,
)
from oof_training_bundle import artifact_contract, load_bundle
from ld_gru_training import file_hash


@dataclass
class ContinuousSubjectData:
    subject: int
    inventory: object
    inventory_contract: dict
    stage1_logits: dict[int, np.ndarray]
    stage2_logits: dict[int, np.ndarray]
    score_artifacts: dict[int, dict]
    score_paths: dict[int, Path]
    input_child_path: Path
    input_child_sha256: str
    bundle_manifest: Path
    bundle_sha256: str
    truth_manifest: Path
    truth_manifest_sha256: str
    truth_event_path: Path
    truth_event_sha256: str
    inventory_contract_path: Path
    inventory_contract_sha256: str
    artifact_identity: dict


# ---------- 输入身份验证：匿名信号先核验，真值只在建立监督/计分侧车时打开 ----------
def load_continuous_subjects(
    input_root: Path,
    input_master: dict,
    input_children: dict[int, dict],
    seeds: tuple[int, ...],
) -> tuple[dict[int, ContinuousSubjectData], dict]:
    subjects: dict[int, ContinuousSubjectData] = {}
    common_artifact_identity: dict | None = None
    for subject in KNOWN_SUBJECTS:
        paths = default_subject_paths(subject)
        context = load_bundle(paths.bundle_manifest)
        identity = artifact_contract(context.manifest)
        if common_artifact_identity is None:
            common_artifact_identity = identity
        elif identity != common_artifact_identity:
            raise RuntimeError("九被试训练 bundle 的伪迹合同不一致")

        signal_inventory = _build_online_signal_inventory(context)
        expected_rows = output_window_rows(signal_inventory.windows)
        child_path = _safe_artifact(
            input_root, input_master["children"][str(subject)]["manifest"],
        )
        child = input_children[subject]
        if (
            _portable_path_text(child.get("inputs", {}).get("bundle_manifest", ""))
            != _portable_path_text(display_path(paths.bundle_manifest))
            or child.get("inputs", {}).get("bundle_manifest_sha256") != context.manifest_sha256
        ):
            raise RuntimeError(f"Subject {subject} 冻结 logits 与当前 bundle 不匹配")

        stage1: dict[int, np.ndarray] = {}
        stage2: dict[int, np.ndarray] = {}
        score_artifacts: dict[int, dict] = {}
        score_paths: dict[int, Path] = {}
        for seed in seeds:
            stage1[seed], stage2[seed], score_artifacts[seed] = _load_seed_logits(
                child_path.parent, child, seed, expected_rows,
            )
            score_paths[seed] = _seed_score_path(child_path.parent, child, seed)
            if file_hash(score_paths[seed]) != score_artifacts[seed]["sha256"]:
                raise RuntimeError(f"Subject {subject} seed {seed} score 哈希不一致")

        truth = load_truth_inventory(paths.truth_manifest, context)
        inventory = build_online_inventory(context, truth)
        if (
            inventory.segments != signal_inventory.segments
            or inventory.windows != signal_inventory.windows
            or not np.array_equal(inventory.signal_rows, signal_inventory.signal_rows)
        ):
            raise RuntimeError("加载真值改变了匿名推理窗口库存")
        inventory_contract = _read_json(paths.inventory_contract)
        verify_inventory_contract(context, inventory, inventory_contract)
        truth_payload = _read_json(paths.truth_manifest)
        truth_event_path = paths.truth_manifest.parent / truth_payload["event_file"]
        subjects[subject] = ContinuousSubjectData(
            subject,
            inventory,
            inventory_contract,
            stage1,
            stage2,
            score_artifacts,
            score_paths,
            child_path,
            file_hash(child_path),
            paths.bundle_manifest,
            context.manifest_sha256,
            paths.truth_manifest,
            truth.manifest_sha256,
            truth_event_path,
            truth.event_file_sha256,
            paths.inventory_contract,
            file_hash(paths.inventory_contract),
            identity,
        )
        print(
            f"Subject {subject}: windows={len(inventory.windows)}, "
            f"events={len(inventory.events)}, segments={len(inventory.segments)}",
            flush=True,
        )
    if common_artifact_identity is None:
        raise RuntimeError("没有加载任何被试")
    return subjects, common_artifact_identity
