"""Model registry for the asynchronous BNCI2014001 experiment."""

from __future__ import annotations

import hashlib
import importlib.util
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

# 以源码位置定位模型，避免换电脑或从其他工作目录启动时依赖当前 cwd。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_SPECS = {
    "eegnet": ("eegnet.py", "EEGNetClassifier"),
    "shallowconvnet": ("shallowconvnet.py", "ShallowConvNetClassifier"),
    "deepcnn": ("deepcnn.py", "DeepCNNClassifier"),
    "conformer": ("conformer.py", "ConformerClassifier"),
    "deformer": ("deformer.py", "DeformerClassifier"),
    "dbconformer": ("DBConfrmer.py", "DBConformer"),
}


class LogitAdapter(nn.Module):
    """Normalize model outputs so training always receives logits."""

    def __init__(self, model: nn.Module, logits_index: int | None = None):
        super().__init__()
        self.model = model
        self.logits_index = logits_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.logits_index is None:
            return output
        return output[self.logits_index]


def available_models() -> tuple[str, ...]:
    """Return model names accepted by the training CLI."""
    return tuple(MODEL_SPECS)


def model_source(model_name: str) -> Path:
    """Return the source file for a registered model."""
    file_name, _ = MODEL_SPECS[normalize_model_name(model_name)]
    return MODEL_DIR / file_name


# 模型身份同时覆盖注册表和实际网络实现；评估必须与训练时保存的值一致。
def model_source_id(model_name: str) -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), model_source(model_name)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def build_model(model_name: str, num_classes: int, chans: int, samples: int) -> nn.Module:
    """Create one of the local EEG backbones by name."""
    model_name = normalize_model_name(model_name)
    model_class = _load_model_class(model_name)

    if model_name == "eegnet":
        model = model_class(
            num_classes=num_classes,
            chans=chans,
            samples=samples,
            kernLenght=64,
            F1=8,
            D=2,
            F2=16,
            dropoutRate=0.5,
            norm_rate=0.25,
        )
        return LogitAdapter(model)

    if model_name == "dbconformer":
        args = Namespace(
            data_name="BNCI2014001",
            chn=chans,
            patch_size=_choose_patch_size(samples),
            time_sample_num=samples,
            class_num=num_classes,
            spa_dim=16,
            gate_flag=False,
            posemb_flag=True,
            branch="all",
            chn_atten_flag=True,
        )
        return LogitAdapter(model_class(args, chn=chans, n_classes=num_classes), logits_index=1)

    return LogitAdapter(model_class(num_classes=num_classes, chans=chans, samples=samples))


def normalize_model_name(model_name: str) -> str:
    """Normalize user input while keeping argparse choices readable."""
    key = model_name.lower().replace("-", "").replace("_", "")
    aliases = {
        "shallow": "shallowconvnet",
        "deep": "deepcnn",
        "dbconfrmer": "dbconformer",
    }
    key = aliases.get(key, key)
    if key not in MODEL_SPECS:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}")
    return key


def _load_model_class(model_name: str) -> type[nn.Module]:
    file_name, class_name = MODEL_SPECS[model_name]
    model_file = MODEL_DIR / file_name
    spec = importlib.util.spec_from_file_location(f"bci_{model_name}", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model source: {model_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        dependency = exc.name or "a model dependency"
        raise ModuleNotFoundError(
            f"Model '{model_name}' requires missing dependency '{dependency}'. "
            "Install dependencies from BCI_Competition/environment-bciml-repro.yml and try again."
        ) from exc
    return getattr(module, class_name)


def _choose_patch_size(samples: int) -> int:
    for patch_size in (128, 64, 32, 16, 8, 4, 2, 1):
        if samples % patch_size == 0:
            return patch_size
    return 1
