"""连续五分类 GRU 的可恢复训练、早停和 checkpoint 合同。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from full_control_gru_policy import (
    ContinuousNormalizer,
    ContinuousTensorSet,
    FullControlGRU,
    balanced_full_control_loss,
)
from ld_gru_training import (
    TrainingHyperparameters,
    _clone_state,
    _initialize_job,
    _move_optimizer,
    atomic_json,
    atomic_torch,
    configure_determinism,
    file_hash,
    update_early_stopping_endpoint,
)


# ---------- 数据身份：padding 数组与 segment key 全部进入训练合同 ----------
def continuous_tensor_hash(dataset: ContinuousTensorSet) -> str:
    digest = hashlib.sha256()
    for value in (dataset.tokens, dataset.targets, dataset.valid_mask):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(json.dumps(dataset.keys, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def model_parameter_counts(model: FullControlGRU) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _device_batch(
    dataset: ContinuousTensorSet,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(dataset.tokens).to(device),
        torch.from_numpy(dataset.targets).to(device),
        torch.from_numpy(dataset.valid_mask).to(device),
    )


# ---------- 每个 epoch 使用完整 segment 集合，loss 自身平衡 IDLE/TASK 与四个 MI 类 ----------
def train_epoch(
    model: FullControlGRU,
    optimizer: torch.optim.Optimizer,
    dataset: ContinuousTensorSet,
    device: torch.device,
    hyperparameters: TrainingHyperparameters,
) -> dict[str, float]:
    if len(dataset.keys) > hyperparameters.batch_size:
        raise RuntimeError("连续 GRU split 的 segment 数超过冻结 full-batch 上限")
    model.train()
    tokens, targets, valid = _device_batch(dataset, device)
    optimizer.zero_grad(set_to_none=True)
    _, logits = model(tokens)
    loss, parts = balanced_full_control_loss(logits, targets, valid)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), hyperparameters.gradient_clip_norm,
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "state_loss": float(parts["state_loss"].detach().cpu()),
        "class_loss": float(parts["class_loss"].detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
    }


def evaluate_loss(
    model: FullControlGRU,
    dataset: ContinuousTensorSet,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        tokens, targets, valid = _device_batch(dataset, device)
        _, logits = model(tokens)
        loss, parts = balanced_full_control_loss(logits, targets, valid)
    return {
        "loss": float(loss.cpu()),
        "state_loss": float(parts["state_loss"].cpu()),
        "class_loss": float(parts["class_loss"].cpu()),
    }


# ---------- 七被试训练、两名端点独立早停；断电后从 latest 精确恢复 ----------
def train_inner_pair_job(
    job_dir: Path,
    train_dataset: ContinuousTensorSet,
    validation_datasets: Mapping[int, ContinuousTensorSet],
    normalizer: ContinuousNormalizer,
    *,
    decision_seed: int,
    hyperparameters: TrainingHyperparameters,
    device: torch.device,
    contract: dict,
    verbose: bool = True,
) -> dict:
    hyperparameters.validate()
    held_subjects = tuple(sorted(validation_datasets))
    if len(held_subjects) != 2:
        raise ValueError("连续 GRU 内层作业必须恰好留出两名验证被试")
    contract_hash, config_path, normalizer_path = _initialize_job(
        job_dir, contract, normalizer,
    )
    completed_path = job_dir / "completed.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("status") != "complete" or completed.get("contract_sha256") != contract_hash:
            raise RuntimeError("连续 GRU 完成标记与当前合同不一致")
        for subject in held_subjects:
            artifact = completed["best_checkpoints"][str(subject)]
            if file_hash(job_dir / artifact["file"]) != artifact["sha256"]:
                raise RuntimeError("连续 GRU 内层最佳 checkpoint 哈希不一致")
        for artifact in completed.get("artifacts", {}).values():
            if file_hash(job_dir / artifact["file"]) != artifact["sha256"]:
                raise RuntimeError("连续 GRU 内层审计产物哈希不一致")
        return completed

    configure_determinism(decision_seed)
    model = FullControlGRU().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    best = {
        subject: {
            "loss": float("inf"), "epoch": 0, "patience": 0, "state": None,
            "closed": False, "closed_epoch": None,
        }
        for subject in held_subjects
    }
    history: list[dict] = []
    start_epoch = 1
    latest_path = job_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != contract_hash:
            raise RuntimeError("连续 GRU latest checkpoint 合同不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer(optimizer, device)
        history = checkpoint["history"]
        best = checkpoint["best"]
        start_epoch = checkpoint["epoch"] + 1

    # latest 已记录两端点关闭时，恢复只物化 completed，不允许再多训练一轮。
    stop_before_training = all(best[subject]["closed"] for subject in held_subjects)
    stop_epoch = start_epoch if stop_before_training else hyperparameters.max_epochs + 1
    for epoch in range(start_epoch, stop_epoch):
        training = train_epoch(model, optimizer, train_dataset, device, hyperparameters)
        validation = {
            subject: evaluate_loss(model, validation_datasets[subject], device)
            for subject in held_subjects
        }
        state = _clone_state(model)
        for subject in held_subjects:
            update_early_stopping_endpoint(
                best[subject],
                current_loss=float(validation[subject]["loss"]),
                epoch=epoch,
                model_state=state,
                min_delta=hyperparameters.early_stopping_min_delta,
                patience_limit=hyperparameters.early_stopping_patience,
            )
        history.append({
            "epoch": epoch,
            "training": training,
            "validation": {str(subject): validation[subject] for subject in held_subjects},
        })
        atomic_torch(latest_path, {
            "format_version": 1,
            "contract_sha256": contract_hash,
            "epoch": epoch,
            "model_state_dict": _clone_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best": best,
            "history": history,
        })
        if verbose and (epoch == 1 or epoch % 10 == 0):
            losses = ", ".join(
                f"S{subject}={validation[subject]['loss']:.4f}" for subject in held_subjects
            )
            print(f"{job_dir.name}: epoch {epoch}, train={training['loss']:.4f}, {losses}", flush=True)
        if all(best[subject]["closed"] for subject in held_subjects):
            break

    if any(best[subject]["state"] is None for subject in held_subjects):
        raise RuntimeError("连续 GRU 内层训练未产生最佳状态")
    best_artifacts: dict[str, dict] = {}
    for subject in held_subjects:
        path = job_dir / f"best_for_subject_{subject:02d}.pt"
        atomic_torch(path, {
            "format_version": 1,
            "contract_sha256": contract_hash,
            "validation_subject": subject,
            "best_epoch": best[subject]["epoch"],
            "best_validation_loss": best[subject]["loss"],
            "endpoint_closed_epoch": best[subject]["closed_epoch"],
            "model_state_dict": best[subject]["state"],
        })
        best_artifacts[str(subject)] = {
            "file": path.name,
            "sha256": file_hash(path),
            "best_epoch": best[subject]["epoch"],
            "best_validation_loss": best[subject]["loss"],
            "endpoint_closed_epoch": best[subject]["closed_epoch"],
        }
    total, trainable = model_parameter_counts(model)
    completed = {
        "status": "complete",
        "contract_sha256": contract_hash,
        "completed_epochs": len(history),
        "best_checkpoints": best_artifacts,
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "artifacts": {
            "job_config": {"file": config_path.name, "sha256": file_hash(config_path)},
            "normalizer": {"file": normalizer_path.name, "sha256": file_hash(normalizer_path)},
            "latest": {"file": latest_path.name, "sha256": file_hash(latest_path)},
        },
    }
    atomic_json(completed_path, completed)
    return completed


# ---------- 外层最终模型：epoch 已由八个内层端点冻结，训练中不再看 outer subject ----------
def train_final_job(
    job_dir: Path,
    train_dataset: ContinuousTensorSet,
    normalizer: ContinuousNormalizer,
    *,
    decision_seed: int,
    fixed_epochs: int,
    hyperparameters: TrainingHyperparameters,
    device: torch.device,
    contract: dict,
    verbose: bool = True,
) -> dict:
    hyperparameters.validate()
    if type(fixed_epochs) is not int or not 1 <= fixed_epochs <= hyperparameters.max_epochs:
        raise ValueError("连续 GRU 最终训练 epoch 非法")
    contract_hash, config_path, normalizer_path = _initialize_job(job_dir, contract, normalizer)
    completed_path = job_dir / "completed.json"
    final_path = job_dir / "final.pt"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "complete"
            or completed.get("contract_sha256") != contract_hash
            or file_hash(final_path) != completed.get("final_checkpoint", {}).get("sha256")
        ):
            raise RuntimeError("连续 GRU 最终模型完成标记或哈希不一致")
        return completed

    configure_determinism(decision_seed)
    model = FullControlGRU().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    latest_path = job_dir / "latest.pt"
    history: list[dict] = []
    start_epoch = 1
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != contract_hash:
            raise RuntimeError("连续 GRU 最终 latest checkpoint 合同不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer(optimizer, device)
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, fixed_epochs + 1):
        training = train_epoch(model, optimizer, train_dataset, device, hyperparameters)
        history.append({"epoch": epoch, "training": training})
        atomic_torch(latest_path, {
            "format_version": 1,
            "contract_sha256": contract_hash,
            "epoch": epoch,
            "model_state_dict": _clone_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        })
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == fixed_epochs):
            print(f"{job_dir.name}: epoch {epoch}/{fixed_epochs}, loss={training['loss']:.4f}", flush=True)
    atomic_torch(final_path, {
        "format_version": 1,
        "contract_sha256": contract_hash,
        "fixed_epochs": fixed_epochs,
        "model_state_dict": _clone_state(model),
    })
    total, trainable = model_parameter_counts(model)
    completed = {
        "status": "complete",
        "contract_sha256": contract_hash,
        "fixed_epochs": fixed_epochs,
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "final_checkpoint": {"file": final_path.name, "sha256": file_hash(final_path)},
        "artifacts": {
            "job_config": {"file": config_path.name, "sha256": file_hash(config_path)},
            "normalizer": {"file": normalizer_path.name, "sha256": file_hash(normalizer_path)},
            "latest": {"file": latest_path.name, "sha256": file_hash(latest_path)},
        },
    }
    atomic_json(completed_path, completed)
    return completed


def load_trained_model(checkpoint_path: Path, device: torch.device) -> FullControlGRU:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FullControlGRU()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
