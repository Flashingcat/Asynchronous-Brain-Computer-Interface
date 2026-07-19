"""Single-window hierarchical argmax baseline."""

from __future__ import annotations

import numpy as np


def predict(stage1_logits: np.ndarray, stage2_logits: np.ndarray) -> np.ndarray:
    """Return five-class labels: 0=idle, 1..4=MI."""
    stage1 = np.asarray(stage1_logits)
    stage2 = np.asarray(stage2_logits)
    if stage1.ndim != 2 or stage1.shape[1] != 2 or stage2.shape != (len(stage1), 4):
        raise ValueError("expected stage1 [windows,2] and stage2 [windows,4] logits")
    return np.where(stage1.argmax(axis=1) == 1, stage2.argmax(axis=1) + 1, 0).astype(np.int64)
