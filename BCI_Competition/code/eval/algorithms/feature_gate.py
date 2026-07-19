"""240-dimensional hidden-feature-gated candidate command policy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .candidate import CandidateConfig, PolicyOutput, commands as candidate_commands


FEATURE_DIM = 240


def commands(
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    stage2_features: np.ndarray,
    config: CandidateConfig = CandidateConfig(),
    *,
    run_ids: np.ndarray | None = None,
) -> PolicyOutput:
    """Require stable EEGNet Stage-2 hidden features before committing a command."""
    features = np.asarray(stage2_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected EEGNet Stage-2 features [windows,{FEATURE_DIM}]")
    return candidate_commands(
        stage1_logits,
        stage2_logits,
        replace(config, fast0=False, fast1=False, feature_gate=True),
        stage2_features=features,
        run_ids=run_ids,
    )
