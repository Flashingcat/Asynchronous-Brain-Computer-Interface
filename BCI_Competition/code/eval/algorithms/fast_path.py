"""Fast-0/Fast-1 policy: an accelerated variant of the candidate state machine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .candidate import CandidateConfig, PolicyOutput, commands as candidate_commands


def commands(
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    config: CandidateConfig = CandidateConfig(),
    *,
    run_ids: np.ndarray | None = None,
) -> PolicyOutput:
    """Enable Fast-0 in the opening window and Fast-1 in the next window."""
    return candidate_commands(
        stage1_logits,
        stage2_logits,
        replace(config, fast0=True, fast1=True, feature_gate=False),
        run_ids=run_ids,
    )
