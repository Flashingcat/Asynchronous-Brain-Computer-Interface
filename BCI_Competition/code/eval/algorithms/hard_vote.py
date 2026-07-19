"""Strict-majority sliding-window command policy."""

from __future__ import annotations

from collections import deque

import numpy as np

from .argmax import predict


def commands(stage1_logits: np.ndarray, stage2_logits: np.ndarray, *, window_count: int = 3, vote_threshold: int = 2, run_ids: np.ndarray | None = None) -> np.ndarray:
    """Emit at most one MI command until a strict idle vote re-arms the policy."""
    if window_count < 2 or vote_threshold <= window_count // 2 or vote_threshold > window_count:
        raise ValueError("require 2 <= N and N/2 < K <= N")
    labels = predict(stage1_logits, stage2_logits)
    runs = None if run_ids is None else np.asarray(run_ids)
    if runs is not None and runs.shape != labels.shape:
        raise ValueError("run_ids must align with logits")
    result = np.full(len(labels), -1, dtype=np.int64)
    votes: deque[int] = deque(maxlen=window_count)
    waiting_for_idle = False
    for index, label in enumerate(labels):
        if index and runs is not None and runs[index] != runs[index - 1]:
            votes.clear()
            waiting_for_idle = False
        votes.append(int(label))
        if len(votes) < window_count:
            continue
        counts = np.bincount(np.asarray(votes), minlength=5)
        if waiting_for_idle:
            if counts[0] >= vote_threshold:
                waiting_for_idle = False
                votes.clear()
            continue
        winners = np.flatnonzero(counts[1:] >= vote_threshold) + 1
        if len(winners) == 1:
            result[index] = int(winners[0])
            waiting_for_idle = True
            votes.clear()
    return result
