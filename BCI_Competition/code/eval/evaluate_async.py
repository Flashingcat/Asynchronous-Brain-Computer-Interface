"""Evaluate two-stage asynchronous BNCI2014001 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from models.model_factory import available_models, normalize_model_name


TABLE_DIR = PROJECT_ROOT / "results" / "tables"
BINARY_CLASS_NAMES = ["idle", "task"]
MI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_CLASS_NAMES = ["idle", *MI_CLASS_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="eegnet", choices=available_models(), help="model name used during training")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def load_predictions(prediction_file: Path) -> dict[str, np.ndarray]:
    if not prediction_file.is_file():
        raise FileNotFoundError(f"Prediction file not found; train first: {prediction_file}")

    with np.load(prediction_file) as prediction_data:
        required_keys = ("y_true", "y_pred", "binary_true", "binary_pred", "mi_true", "mi_pred")
        missing_keys = [key for key in required_keys if key not in prediction_data.files]
        if missing_keys:
            raise RuntimeError(f"Prediction file is not a two-stage output; missing keys: {missing_keys}")
        return {key: prediction_data[key] for key in required_keys}


def main() -> None:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    run_name = f"hierarchical_{model_name}"
    prediction_file = TABLE_DIR / f"{run_name}_async_predictions.npz"
    metrics_file = TABLE_DIR / f"{run_name}_async_metrics.json"

    predictions = load_predictions(prediction_file)
    report = {
        "final_5class": compute_metrics(predictions["y_true"], predictions["y_pred"], FINAL_CLASS_NAMES),
        "stage1_binary": compute_metrics(
            predictions["binary_true"],
            predictions["binary_pred"],
            BINARY_CLASS_NAMES,
        ),
        "stage2_mi_on_true_task_windows": compute_metrics(
            predictions["mi_true"],
            predictions["mi_pred"],
            MI_CLASS_NAMES,
        ),
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "Final 5-class "
        f"accuracy={report['final_5class']['accuracy']:.3f}; "
        f"balanced_accuracy={report['final_5class']['balanced_accuracy']:.3f}"
    )
    print(
        "Stage1 binary "
        f"accuracy={report['stage1_binary']['accuracy']:.3f}; "
        f"balanced_accuracy={report['stage1_binary']['balanced_accuracy']:.3f}"
    )
    print(
        "Stage2 MI "
        f"accuracy={report['stage2_mi_on_true_task_windows']['accuracy']:.3f}; "
        f"balanced_accuracy={report['stage2_mi_on_true_task_windows']['balanced_accuracy']:.3f}"
    )
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    main()
