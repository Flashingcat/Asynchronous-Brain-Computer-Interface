"""构造并校验可审计的训练实验身份。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


# 规范化序列化保证相同配置在不同机器上得到同一个短身份。
def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def experiment_id_for(experiment: dict) -> str:
    digest = hashlib.sha256(canonical_json(experiment).encode("utf-8")).hexdigest()[:12]
    return f"{experiment['model']}-s{experiment['seed']}-{digest}"


def validate_experiment(experiment: dict) -> None:
    required = {"schema_version", "trainer", "model", "seed", "augmentation", "training", "preprocessing", "dataset_sha256", "code_commit"}
    if required.difference(experiment):
        raise RuntimeError("experiment metadata is incomplete")
    training = {"binary_epochs", "mi_epochs", "batch_size", "learning_rate", "class_weight"}
    preprocessing = {
        "schema_version", "sampling_rate", "window_samples", "stride_samples", "bandpass_hz",
        "filter_order", "artifact_policy", "window_label_policy",
    }
    if training.difference(experiment["training"]) or preprocessing.difference(experiment["preprocessing"]):
        raise RuntimeError("experiment configuration is incomplete")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 正式训练只接受已提交源码，并把精确 commit 与数据内容哈希写入身份。
def clean_git_commit(project_root: Path) -> str:
    repo = project_root.resolve().parent
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("formal training requires a clean git worktree")
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def dataset_identity(path: Path) -> tuple[dict, str]:
    with np.load(path, allow_pickle=False) as data:
        if "preprocessing_config" not in data.files:
            raise RuntimeError("dataset is missing preprocessing_config; rebuild it")
        preprocessing = json.loads(str(data["preprocessing_config"].item()))
    return preprocessing, file_sha256(path)


def build_experiment(trainer: str, model: str, seed: int, augmentation: list[str], args, data_file: Path, project_root: Path) -> tuple[str, dict]:
    preprocessing, dataset_hash = dataset_identity(data_file)
    experiment = {
        "schema_version": SCHEMA_VERSION,
        "trainer": trainer,
        "model": model,
        "seed": int(seed),
        "augmentation": list(augmentation),
        "training": {
            "binary_epochs": int(args.binary_epochs),
            "mi_epochs": int(args.mi_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "class_weight": args.class_weight,
        },
        "preprocessing": preprocessing,
        "dataset_sha256": dataset_hash,
        "code_commit": clean_git_commit(project_root),
    }
    return experiment_id_for(experiment), experiment


# 评估前重算身份并核对冗余字段，旧检查点或被手工改写的元数据直接拒绝。
def validate_experiment_identity(checkpoint: dict) -> tuple[str, dict]:
    if "experiment_id" not in checkpoint or "experiment" not in checkpoint:
        raise RuntimeError("checkpoint has no experiment identity; retrain it")
    experiment_id, experiment = checkpoint["experiment_id"], checkpoint["experiment"]
    validate_experiment(experiment)
    if experiment["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unsupported experiment identity schema")
    if experiment_id_for(experiment) != experiment_id:
        raise RuntimeError("checkpoint experiment_id does not match its metadata")
    fields = ("model", "seed", "augmentation")
    if any(checkpoint.get(field) != experiment.get(field) for field in fields):
        raise RuntimeError("checkpoint fields do not match its experiment metadata")
    return experiment_id, experiment
