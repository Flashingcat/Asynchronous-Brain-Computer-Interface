"""验证固定 epoch 50 EEGNet 的连续隐藏特征可被可信、可复核地提取。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# 与训练和既有连续推理入口保持相同的确定性 CUDA 配置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "code" / "train"
MODELS_DIR = PROJECT_ROOT / "code" / "models"
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
for source_dir in (TRAIN_DIR, MODELS_DIR, EVAL_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from oof_training_bundle import load_bundle  # noqa: E402
from run_epoch50_online_oof import (  # noqa: E402
    FIXED_EPOCH,
    FIXED_FOLDS,
    INFERENCE_BATCH_SIZE,
    KNOWN_SEEDS,
    KNOWN_SUBJECTS,
    atomic_json,
    atomic_npz,
    build_online_inventory,
    default_subject_paths,
    display_path,
    file_hash,
    git_state,
    load_epoch50_job,
    materialize_rows,
    output_window_rows,
    verify_inventory_contract,
)


EEGNET_FEATURE_DIM = 240
FROZEN_SCORE_ROOT = (
    PROJECT_ROOT / "results" / "tables"
    / "s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2"
)


# ---------- 特征接口：包装层与底层模型必须逐元素给出相同 logit ----------
@torch.inference_mode()
def predict_logits_and_features(
    model: nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """提取分类头前特征，同时验证没有因绕过包装层而改变模型输出。"""
    values = np.asarray(inputs)
    backbone = getattr(model, "model", None)
    classifier = getattr(backbone, "classifier_block", None)
    if (
        values.ndim != 3
        or values.shape[1:] != (22, 500)
        or values.dtype != np.float32
        or len(values) == 0
        or not np.isfinite(values).all()
        or backbone is None
        or getattr(model, "logits_index", "missing") is not None
        or not isinstance(classifier, nn.Sequential)
        or len(classifier) != 1
        or not isinstance(classifier[0], nn.Linear)
        or classifier[0].in_features != EEGNET_FEATURE_DIM
        or classifier[0].out_features != classes
    ):
        raise RuntimeError("只允许对 22x500 的固定 EEGNet LogitAdapter 提取 240 维特征")

    logit_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(values), INFERENCE_BATCH_SIZE):
        batch = torch.from_numpy(values[start:start + INFERENCE_BATCH_SIZE]).to(
            device, non_blocking=device.type == "cuda",
        )
        adapter_logits = model(batch)
        direct_output = backbone(batch, return_features=True)
        if (
            not isinstance(direct_output, tuple)
            or len(direct_output) != 2
            or not torch.equal(adapter_logits, direct_output[0])
        ):
            raise RuntimeError("底层特征接口改变了 LogitAdapter 的 logit")
        logit_chunks.append(direct_output[0].cpu().numpy())
        feature_chunks.append(direct_output[1].cpu().numpy())

    logits = np.ascontiguousarray(np.concatenate(logit_chunks), dtype=np.float32)
    features = np.ascontiguousarray(np.concatenate(feature_chunks), dtype=np.float32)
    if (
        logits.shape != (len(values), classes)
        or features.shape != (len(values), EEGNET_FEATURE_DIM)
        or not np.isfinite(logits).all()
        or not np.isfinite(features).all()
    ):
        raise RuntimeError("EEGNet logit 或隐藏特征的结构非法")
    return logits, features


# ---------- 冻结连续分数：只读 session0 基线，作为逐窗 logit 对照 ----------
def load_frozen_scores(
    path: Path,
    expected_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        fields = tuple(sorted(archive.files))
        required = {"window_rows", "stage1_logits", "stage2_logits"}
        if not required.issubset(fields):
            raise RuntimeError("冻结连续分数缺少窗口或 Stage 1/2 logit")
        rows = archive["window_rows"].copy()
        stage1 = np.ascontiguousarray(archive["stage1_logits"], dtype=np.float32)
        stage2 = np.ascontiguousarray(archive["stage2_logits"], dtype=np.float32)
    if (
        not np.array_equal(rows, expected_rows)
        or stage1.shape != (len(expected_rows), 2)
        or stage2.shape != (len(expected_rows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise RuntimeError("冻结连续分数不能逐窗绑定当前 session0 在线库存")
    return stage1, stage2, fields


# ---------- 来源闭包：运行前后重复哈希，拒绝执行中源码或 checkpoint 漂移 ----------
def source_hashes() -> dict[str, str]:
    paths = {
        "feature_preflight_runner": Path(__file__),
        "epoch50_online_runner": EVAL_DIR / "run_epoch50_online_oof.py",
        "protocol_metrics": EVAL_DIR / "protocol_metrics.py",
        "oof_training_bundle_reader": TRAIN_DIR / "oof_training_bundle.py",
        "model_factory": MODELS_DIR / "model_factory.py",
        "eegnet": MODELS_DIR / "models" / "eegnet.py",
    }
    return {role: file_hash(path) for role, path in paths.items()}


def checkpoint_input_hashes(checkpoint_root: Path, seed: int) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for fold in FIXED_FOLDS:
        for stage in (1, 2):
            job_name = f"stage{stage}_causal_fold{fold}_seed{seed}"
            job_dir = checkpoint_root / job_name
            records[job_name] = {
                name: file_hash(job_dir / name)
                for name in ("job_config.json", "completed.json", "latest.pt", "oof_predictions.npz")
            }
    return records


def configure_runtime(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 提取特征，但当前环境没有可用 GPU")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return device


# ---------- 端到端预检：不生成决策、不读取测试 session、不选择任何策略 ----------
def run(args: argparse.Namespace) -> dict:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    subject, seed = int(args.subject), int(args.seed)
    if subject not in KNOWN_SUBJECTS or seed not in KNOWN_SEEDS:
        raise ValueError("subject 或 seed 不在已训练的固定集合中")
    defaults = default_subject_paths(subject)
    bundle_manifest = Path(args.bundle_manifest or defaults.bundle_manifest).resolve()
    checkpoint_root = Path(args.checkpoint_root or defaults.checkpoint_root).resolve()
    # 该预检绑定既有 v1 分数产物，只允许显式走历史事件恢复合同。
    inventory_contract = Path(
        args.inventory_contract or defaults.legacy_inventory_contract,
    ).resolve()
    frozen_scores = Path(
        args.frozen_scores
        or FROZEN_SCORE_ROOT / f"subject_{subject:02d}" / f"seed{seed}_scores_and_decisions.npz"
    ).resolve()
    output_root = Path(
        args.output_root
        or PROJECT_ROOT / "results" / "tables"
        / f"s{subject:02d}_seed{seed}_epoch50_feature_preflight_v1"
    ).resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出路径不是空目录，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    device = configure_runtime(args.device)
    sources_at_start = source_hashes()
    checkpoints_at_start = checkpoint_input_hashes(checkpoint_root, seed)
    fixed_inputs_at_start = {
        "bundle_manifest": file_hash(bundle_manifest),
        "inventory_contract": file_hash(inventory_contract),
        "frozen_scores": file_hash(frozen_scores),
    }

    context = load_bundle(bundle_manifest, verify_hashes=True)
    if context.manifest.get("subject") != subject:
        raise RuntimeError("--subject 与 session0-only bundle 不一致")
    inventory = build_online_inventory(
        context, allow_legacy_event_reconstruction=True,
    )
    contract = json.loads(inventory_contract.read_text(encoding="utf-8"))
    verify_inventory_contract(context, inventory, contract)
    expected_rows = output_window_rows(inventory.windows)
    frozen_stage1, frozen_stage2, frozen_fields = load_frozen_scores(
        frozen_scores, expected_rows,
    )

    stage_logits = {
        1: np.full((len(expected_rows), 2), np.nan, dtype=np.float32),
        2: np.full((len(expected_rows), 4), np.nan, dtype=np.float32),
    }
    stage_features = {
        1: np.full((len(expected_rows), EEGNET_FEATURE_DIM), np.nan, dtype=np.float32),
        2: np.full((len(expected_rows), EEGNET_FEATURE_DIM), np.nan, dtype=np.float32),
    }
    frozen_by_stage = {1: frozen_stage1, 2: frozen_stage2}
    run_ids = expected_rows["run_id"].astype(np.int64)
    records: list[dict] = []

    for fold in FIXED_FOLDS:
        online_indices = np.flatnonzero(run_ids == fold)
        online_inputs = materialize_rows(context, inventory.signal_rows[online_indices], fold)
        for stage in (1, 2):
            classes = 2 if stage == 1 else 4
            model, validation_rows, saved_logits, metadata = load_epoch50_job(
                context, checkpoint_root, fold, seed, stage, device,
            )
            validation_inputs = materialize_rows(context, validation_rows, fold)
            validation_logits, validation_features = predict_logits_and_features(
                model, validation_inputs, device, classes,
            )
            validation_error = np.abs(validation_logits - saved_logits)
            if not np.allclose(validation_logits, saved_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"{metadata['job_name']} 无法复现已保存 epoch50 OOF logit")

            continuous_logits, continuous_features = predict_logits_and_features(
                model, online_inputs, device, classes,
            )
            frozen_fold_logits = frozen_by_stage[stage][online_indices]
            continuous_error = np.abs(continuous_logits - frozen_fold_logits)
            if not np.allclose(continuous_logits, frozen_fold_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"{metadata['job_name']} 无法对齐冻结连续 logit")
            stage_logits[stage][online_indices] = continuous_logits
            stage_features[stage][online_indices] = continuous_features

            metadata.update({
                "adapter_vs_feature_api_logits_exact": True,
                "saved_oof_logits_max_abs_error": float(validation_error.max()),
                "saved_oof_logits_exact": bool(np.array_equal(validation_logits, saved_logits)),
                "frozen_continuous_logits_max_abs_error": float(continuous_error.max()),
                "frozen_continuous_logits_exact": bool(
                    np.array_equal(continuous_logits, frozen_fold_logits)
                ),
                "validation_feature_shape": list(validation_features.shape),
                "continuous_feature_shape": list(continuous_features.shape),
            })
            records.append(metadata)
            if args.verbose:
                print(
                    f"{metadata['job_name']}: validation_max={validation_error.max():.3g}, "
                    f"continuous_max={continuous_error.max():.3g}, "
                    f"features={continuous_features.shape}",
                    flush=True,
                )
            del model, validation_inputs, validation_logits, validation_features
            del continuous_logits, continuous_features
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if any(not np.isfinite(values).all() for values in (*stage_logits.values(), *stage_features.values())):
        raise RuntimeError("并非所有连续窗口都获得了有限 logit 和隐藏特征")
    if source_hashes() != sources_at_start:
        raise RuntimeError("执行期间来源代码发生变化，拒绝写出预检产物")
    if checkpoint_input_hashes(checkpoint_root, seed) != checkpoints_at_start:
        raise RuntimeError("执行期间 checkpoint 输入发生变化，拒绝写出预检产物")
    fixed_inputs_at_end = {
        "bundle_manifest": file_hash(bundle_manifest),
        "inventory_contract": file_hash(inventory_contract),
        "frozen_scores": file_hash(frozen_scores),
    }
    if fixed_inputs_at_end != fixed_inputs_at_start:
        raise RuntimeError("执行期间固定输入发生变化，拒绝写出预检产物")

    feature_path = output_root / "features_and_logits.npz"
    atomic_npz(
        feature_path,
        window_rows=expected_rows,
        stage1_logits=stage_logits[1],
        stage2_logits=stage_logits[2],
        stage1_features=stage_features[1],
        stage2_features=stage_features[2],
    )
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "PASS",
        "claim_status": "FEATURE_EXTRACTION_PREFLIGHT_ONLY",
        "protocol_id": f"bnci2014001_s{subject:02d}_seed{seed}_epoch50_feature_preflight_v1",
        "subject": subject,
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "checkpoint_epoch": FIXED_EPOCH,
        "seed": seed,
        "job_count": len(records),
        "feature_contract": {
            "model": "eegnet",
            "layer": "flattened_block2_output_immediately_before_classifier",
            "dimension_per_stage": EEGNET_FEATURE_DIM,
            "dropout_mode": "evaluation_identity",
            "stage_spaces_are_separate": True,
            "fold_spaces_are_separate": True,
            "allowed_temporal_comparison_scope": "within_same_run_and_segment_only",
            "normalization": "none_raw_hidden_features",
            "strategy_or_threshold_selection": "none",
            "decision_generation": "none",
        },
        "inputs": {
            "bundle_manifest": display_path(bundle_manifest),
            "inventory_contract": display_path(inventory_contract),
            "checkpoint_root": display_path(checkpoint_root),
            "frozen_scores": display_path(frozen_scores),
            "frozen_score_fields": list(frozen_fields),
            "fixed_input_sha256": fixed_inputs_at_start,
            "checkpoint_input_sha256": checkpoints_at_start,
        },
        "inventory": contract["inventory"],
        "checkpoint_records": records,
        "artifact": {
            "file": feature_path.name,
            "sha256": file_hash(feature_path),
            "window_count": len(expected_rows),
            "stage1_feature_shape": list(stage_features[1].shape),
            "stage2_feature_shape": list(stage_features[2].shape),
        },
        "source_sha256": sources_at_start,
        "runtime": {
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": None if device.type != "cuda" else torch.cuda.get_device_name(device),
            "git": git_state(),
        },
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    if args.verbose:
        print(json.dumps(manifest["artifact"], ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=KNOWN_SUBJECTS, default=1)
    parser.add_argument("--seed", type=int, choices=KNOWN_SEEDS, default=42)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--inventory-contract", type=Path)
    parser.add_argument("--frozen-scores", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
