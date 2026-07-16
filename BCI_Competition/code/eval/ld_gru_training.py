"""LD-GRU-v1 的确定性、可恢复训练器；只消费已构造好的候选 token。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Mapping

import numpy as np
import torch

from ld_gru_policy import (
    ABLATIONS,
    TOKEN_MODES,
    CandidateTensorSet,
    TinyLDGRU,
    TokenNormalizer,
    balanced_batch_indices,
    model_parameter_counts,
    set_valued_commit_loss,
)


@dataclass(frozen=True)
class TrainingHyperparameters:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-8

    def validate(self) -> None:
        if (
            not 0.0 < self.learning_rate < 1.0
            or not 0.0 <= self.weight_decay < 1.0
            or self.batch_size < 2
            or self.batch_size % 2
            or self.max_epochs < 1
            or self.gradient_clip_norm <= 0.0
            or self.early_stopping_patience < 1
            or self.early_stopping_min_delta < 0.0
        ):
            raise ValueError("LD-GRU 训练超参数非法")


# ---------- 原子产物与哈希：中断只能留下旧完整文件或新完整文件 ----------
def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_normalizer(path: Path, normalizer: TokenNormalizer) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, mean=normalizer.mean, std=normalizer.std)
    temporary.replace(path)


def tensor_set_hash(dataset: CandidateTensorSet) -> str:
    digest = hashlib.sha256()
    for array in (
        dataset.normalized_tokens,
        dataset.centered_stage2,
        dataset.valid_mask,
        dataset.correct_mask,
        dataset.positive_mask,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update("\n".join(dataset.candidate_ids).encode("utf-8"))
    return digest.hexdigest()


# ---------- 可复现执行：每个 epoch 自带确定 batch seed，恢复时无需猜测 RNG 位置 ----------
def configure_determinism(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("decision seed 必须是非负整数")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _to_device(
    dataset: CandidateTensorSet,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(dataset.normalized_tokens[indices]).to(device),
        torch.from_numpy(dataset.centered_stage2[indices]).to(device),
        torch.from_numpy(dataset.valid_mask[indices]).to(device),
        torch.from_numpy(dataset.correct_mask[indices]).to(device),
    )


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _clone_state(model: TinyLDGRU) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def update_early_stopping_endpoint(
    record: dict,
    *,
    current_loss: float,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    min_delta: float,
    patience_limit: int,
) -> None:
    """独立更新一个验证端点；一旦 closed，另一端不能使它重新参与选择。"""
    if record.get("closed") is True:
        return
    if current_loss < record["loss"] - min_delta:
        record.update({
            "loss": current_loss,
            "epoch": epoch,
            "patience": 0,
            "state": {name: value.detach().cpu().clone() for name, value in model_state.items()},
        })
    else:
        record["patience"] += 1
    if record["patience"] >= patience_limit:
        record["closed"] = True
        record["closed_epoch"] = epoch


def train_epoch(
    model: TinyLDGRU,
    optimizer: torch.optim.Optimizer,
    dataset: CandidateTensorSet,
    device: torch.device,
    hyperparameters: TrainingHyperparameters,
    *,
    epoch_seed: int,
) -> dict[str, float | int]:
    model.train()
    batches = balanced_batch_indices(
        dataset, hyperparameters.batch_size, np.random.default_rng(epoch_seed),
    )
    totals: list[float] = []
    positives: list[float] = []
    negatives: list[float] = []
    for indices in batches:
        tokens, centered, valid, correct = _to_device(dataset, indices, device)
        optimizer.zero_grad(set_to_none=True)
        _, stop_logits, _, class_logits = model(tokens, centered, valid)
        loss, parts = set_valued_commit_loss(stop_logits, class_logits, valid, correct)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            hyperparameters.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("LD-GRU 梯度范数出现非有限值")
        optimizer.step()
        totals.append(float(loss.detach().cpu()))
        positives.append(float(parts["positive_mean"].detach().cpu()))
        negatives.append(float(parts["negative_mean"].detach().cpu()))
    return {
        "loss": float(np.mean(totals)),
        "positive_loss": float(np.mean(positives)),
        "negative_loss": float(np.mean(negatives)),
        "batch_count": len(batches),
    }


@torch.inference_mode()
def evaluate_loss(
    model: TinyLDGRU,
    dataset: CandidateTensorSet,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    indices = np.arange(len(dataset.candidate_ids), dtype=np.int64)
    tokens, centered, valid, correct = _to_device(dataset, indices, device)
    _, stop_logits, _, class_logits = model(tokens, centered, valid)
    loss, parts = set_valued_commit_loss(stop_logits, class_logits, valid, correct)
    return {
        "loss": float(loss.cpu()),
        "positive_loss": float(parts["positive_mean"].cpu()),
        "negative_loss": float(parts["negative_mean"].cpu()),
        "positive_count": int(parts["positive_count"]),
        "negative_count": int(parts["negative_count"]),
    }


def _initialize_job(
    job_dir: Path,
    contract: dict,
    normalizer: TokenNormalizer,
) -> tuple[str, Path, Path]:
    job_dir.mkdir(parents=True, exist_ok=True)
    contract_sha256 = canonical_hash(contract)
    config_path = job_dir / "job_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != {"contract_sha256": contract_sha256, "contract": contract}:
            raise RuntimeError(f"训练目录已绑定不同合同: {job_dir}")
    else:
        atomic_json(config_path, {"contract_sha256": contract_sha256, "contract": contract})
    normalizer_path = job_dir / "token_normalizer.npz"
    if normalizer_path.exists():
        with np.load(normalizer_path, allow_pickle=False) as payload:
            if (
                set(payload.files) != {"mean", "std"}
                or not np.array_equal(payload["mean"], normalizer.mean)
                or not np.array_equal(payload["std"], normalizer.std)
            ):
                raise RuntimeError("恢复作业的 token 标准化参数发生变化")
    else:
        atomic_normalizer(normalizer_path, normalizer)
    return contract_sha256, config_path, normalizer_path


# ---------- 内层成对留出：一条七被试训练轨迹分别按两名验证被试保存最佳状态 ----------
def train_inner_pair_job(
    job_dir: Path,
    train_dataset: CandidateTensorSet,
    validation_datasets: Mapping[int, CandidateTensorSet],
    normalizer: TokenNormalizer,
    *,
    ablation: str,
    decision_seed: int,
    hyperparameters: TrainingHyperparameters,
    device: torch.device,
    contract: dict,
    token_mode: str = "full",
    verbose: bool = True,
) -> dict:
    hyperparameters.validate()
    held_subjects = tuple(sorted(validation_datasets))
    if (
        ablation not in ABLATIONS
        or token_mode not in TOKEN_MODES
        or contract.get("token_mode", "full") != token_mode
        or len(held_subjects) != 2
    ):
        raise ValueError("内层作业必须指定一种消融和两名留出被试")
    contract_hash, config_path, normalizer_path = _initialize_job(job_dir, contract, normalizer)
    completed_path = job_dir / "completed.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "complete"
            or completed.get("contract_sha256") != contract_hash
        ):
            raise RuntimeError("完成标记与当前内层合同不一致")
        for subject in held_subjects:
            best_path = job_dir / completed["best_checkpoints"][str(subject)]["file"]
            if file_hash(best_path) != completed["best_checkpoints"][str(subject)]["sha256"]:
                raise RuntimeError("内层最佳 checkpoint 哈希不一致")
        for artifact in completed.get("artifacts", {}).values():
            path = job_dir / artifact["file"]
            if not path.is_file() or file_hash(path) != artifact["sha256"]:
                raise RuntimeError("内层完成作业的审计产物哈希不一致")
        return completed

    configure_determinism(decision_seed)
    model = TinyLDGRU(ablation, token_mode).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
            raise RuntimeError("latest checkpoint 与当前内层合同不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer(optimizer, device)
        history = checkpoint["history"]
        best = checkpoint["best"]
        start_epoch = checkpoint["epoch"] + 1

    # latest.pt 可能已记录“两端点均关闭”，但进程在写 completed.json
    # 之前中断。此时恢复必须直接物化完成产物，不能多训练一个 epoch。
    stop_before_training = all(best[subject]["closed"] for subject in held_subjects)
    stop_epoch = start_epoch if stop_before_training else hyperparameters.max_epochs + 1
    for epoch in range(start_epoch, stop_epoch):
        training = train_epoch(
            model,
            optimizer,
            train_dataset,
            device,
            hyperparameters,
            epoch_seed=decision_seed * 1000 + epoch,
        )
        validation = {
            subject: evaluate_loss(model, validation_datasets[subject], device)
            for subject in held_subjects
        }
        current_state = _clone_state(model)
        for subject in held_subjects:
            update_early_stopping_endpoint(
                best[subject],
                current_loss=float(validation[subject]["loss"]),
                epoch=epoch,
                model_state=current_state,
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
        raise RuntimeError("内层训练未产生两名验证被试的最佳状态")
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
        "stop_rule": (
            "each_endpoint_freezes_at_its_own_first_patience_limit; "
            "shared_training_stops_when_both_closed_or_at_max_epochs"
        ),
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


# ---------- 外层最终模型：内层先冻结 epoch，本作业不再查看任何验证或测试指标 ----------
def train_final_job(
    job_dir: Path,
    train_dataset: CandidateTensorSet,
    normalizer: TokenNormalizer,
    *,
    ablation: str,
    decision_seed: int,
    fixed_epochs: int,
    hyperparameters: TrainingHyperparameters,
    device: torch.device,
    contract: dict,
    token_mode: str = "full",
    verbose: bool = True,
) -> dict:
    hyperparameters.validate()
    if (
        ablation not in ABLATIONS
        or token_mode not in TOKEN_MODES
        or contract.get("token_mode", "full") != token_mode
        or type(fixed_epochs) is not int
        or not 1 <= fixed_epochs <= hyperparameters.max_epochs
    ):
        raise ValueError("最终作业消融或冻结 epoch 非法")
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
            raise RuntimeError("最终模型完成标记或 checkpoint 哈希不一致")
        for artifact in completed.get("artifacts", {}).values():
            path = job_dir / artifact["file"]
            if not path.is_file() or file_hash(path) != artifact["sha256"]:
                raise RuntimeError("最终模型完成作业的审计产物哈希不一致")
        return completed

    configure_determinism(decision_seed)
    model = TinyLDGRU(ablation, token_mode).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    latest_path = job_dir / "latest.pt"
    history: list[dict] = []
    start_epoch = 1
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != contract_hash:
            raise RuntimeError("最终模型 latest checkpoint 合同不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer(optimizer, device)
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, fixed_epochs + 1):
        training = train_epoch(
            model,
            optimizer,
            train_dataset,
            device,
            hyperparameters,
            epoch_seed=decision_seed * 1000 + epoch,
        )
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
            print(
                f"{job_dir.name}: epoch {epoch}/{fixed_epochs}, loss={training['loss']:.4f}",
                flush=True,
            )
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
        "epoch_selection_method": "inner_subject_loso_median_round_half_up",
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


def load_trained_model(
    checkpoint_path: Path,
    ablation: str,
    device: torch.device,
    token_mode: str = "full",
) -> TinyLDGRU:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if token_mode not in TOKEN_MODES:
        raise ValueError(f"token_mode 必须取 {TOKEN_MODES}")
    model = TinyLDGRU(ablation, token_mode)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
