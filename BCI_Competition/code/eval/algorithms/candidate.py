"""Candidate, Fast-0/Fast-1, and feature-gated online command policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NO_COMMAND = -1
READY, CANDIDATE, WAIT_IDLE = "READY", "CANDIDATE", "WAIT_IDLE"


@dataclass(frozen=True)
class CandidateConfig:
    task_on_probability: float = 0.60
    task_hold_probability: float = 0.50
    idle_reset_probability: float = 0.40
    min_candidate_windows: int = 2
    max_candidate_windows: int = 4
    top_probability: float = 0.50
    probability_gap: float = 0.10
    stable_windows: int = 2
    stage2_aggregation: str = "candidate_mean"  # current, candidate_mean, candidate_ewma
    stage2_alpha: float = 0.5
    fast0: bool = False
    fast1: bool = False
    fast_probability: float = 0.75
    fast_gap: float = 0.25
    feature_gate: bool = False
    feature_max_change: float = 0.50
    feature_consecutive: int = 2


@dataclass(frozen=True)
class PolicyOutput:
    commands: np.ndarray
    state_before: tuple[str, ...]
    state_after: tuple[str, ...]
    reasons: tuple[str | None, ...]
    candidate_age: np.ndarray


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _valid(config: CandidateConfig, features: np.ndarray | None) -> None:
    if not 0 < config.idle_reset_probability < config.task_on_probability <= 1:
        raise ValueError("require 0 < idle_reset < task_on <= 1")
    if not 0 < config.task_hold_probability <= config.task_on_probability:
        raise ValueError("require 0 < task_hold <= task_on")
    if not 1 <= config.stable_windows <= config.min_candidate_windows <= config.max_candidate_windows:
        raise ValueError("require stable <= min_candidate <= max_candidate")
    if config.feature_gate and features is None:
        raise ValueError("feature policy requires Stage-2 hidden features")
    if config.stage2_aggregation not in {"current", "candidate_mean", "candidate_ewma"}:
        raise ValueError("stage2_aggregation must be current, candidate_mean, or candidate_ewma")
    if not 0 < config.stage2_alpha <= 1:
        raise ValueError("stage2_alpha must be in (0,1]")


def commands(
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    config: CandidateConfig = CandidateConfig(),
    *,
    stage2_features: np.ndarray | None = None,
    run_ids: np.ndarray | None = None,
) -> PolicyOutput:
    """Run READY → CANDIDATE → WAIT_IDLE causally over one continuous run."""
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    features = None if stage2_features is None else np.asarray(stage2_features, dtype=np.float64)
    if stage1.ndim != 2 or stage1.shape[1] != 2 or stage2.shape != (len(stage1), 4):
        raise ValueError("expected stage1 [windows,2] and stage2 [windows,4] logits")
    if features is not None and (features.ndim != 2 or len(features) != len(stage1) or not np.isfinite(features).all()):
        raise ValueError("features must be finite and aligned with windows")
    runs = None if run_ids is None else np.asarray(run_ids)
    if runs is not None and runs.shape != (len(stage1),):
        raise ValueError("run_ids must align with logits")
    _valid(config, features)

    output = np.full(len(stage1), NO_COMMAND, dtype=np.int64)
    before, after, reasons, ages = [], [], [], np.zeros(len(stage1), dtype=np.int64)
    state, age, stable, last_class = READY, 0, 0, NO_COMMAND
    candidate_logits: list[np.ndarray] = []
    ewma_logits: np.ndarray | None = None
    previous_feature: np.ndarray | None = None
    feature_streak = 0
    open_stage2: np.ndarray | None = None

    for i, (gate, classes) in enumerate(zip(stage1, stage2)):
        if i and runs is not None and runs[i] != runs[i - 1]:
            state, age, stable, last_class = READY, 0, 0, NO_COMMAND
            candidate_logits, ewma_logits, previous_feature, feature_streak, open_stage2 = [], None, None, 0, None
        before.append(state)
        reason: str | None = None
        task_probability = float(_softmax(gate)[1])
        centered = classes - classes.mean()
        raw_probability = _softmax(centered)
        raw_class = int(raw_probability.argmax()) + 1
        raw_gap = float(np.partition(raw_probability, -1)[-1] - np.partition(raw_probability, -2)[-2])

        if state == READY:
            if task_probability >= config.task_on_probability:
                if config.fast0 and raw_probability.max() >= config.fast_probability and raw_gap >= config.fast_gap:
                    output[i], state, reason = raw_class, WAIT_IDLE, "fast0_commit"
                else:
                    state, reason, open_stage2 = CANDIDATE, "candidate_open", centered
                    age = stable = 0
                    candidate_logits, ewma_logits, previous_feature, feature_streak = [], None, None, 0
        elif state == WAIT_IDLE:
            if task_probability <= config.idle_reset_probability:
                state, reason = READY, "idle_reset"
        else:
            age += 1
            if task_probability < config.task_hold_probability:
                state, reason = READY, "candidate_abort"
            else:
                candidate_logits.append(centered)
                ewma_logits = centered if ewma_logits is None else (
                    config.stage2_alpha * centered + (1.0 - config.stage2_alpha) * ewma_logits
                )
                aggregate = (
                    centered if config.stage2_aggregation == "current"
                    else np.mean(candidate_logits, axis=0) if config.stage2_aggregation == "candidate_mean"
                    else ewma_logits
                )
                probability = _softmax(aggregate)
                top_class = int(probability.argmax()) + 1
                stable = stable + 1 if top_class == last_class else 1
                last_class = top_class
                ordered = np.sort(probability)
                feature_ok = True
                if config.feature_gate:
                    assert features is not None
                    vector = features[i]
                    norm = float(np.linalg.norm(vector))
                    if not np.isfinite(norm) or norm <= 1e-12:
                        raise ValueError("Stage-2 feature vector must be non-zero")
                    unit = vector / norm
                    change = 0.0 if previous_feature is None else float(np.linalg.norm(unit - previous_feature))
                    feature_streak = feature_streak + 1 if change <= config.feature_max_change else 0
                    previous_feature = unit
                    feature_ok = feature_streak >= config.feature_consecutive
                fast1_ok = (
                    config.fast1 and age == 1 and open_stage2 is not None
                    and task_probability >= config.task_hold_probability
                    and raw_class == int(_softmax(open_stage2).argmax()) + 1
                    and _softmax(0.5 * (open_stage2 + centered)).max() >= config.fast_probability
                )
                slow_ok = (
                    age >= config.min_candidate_windows and stable >= config.stable_windows
                    and probability.max() >= config.top_probability
                    and ordered[-1] - ordered[-2] >= config.probability_gap and feature_ok
                )
                if fast1_ok:
                    output[i], state, reason = raw_class, WAIT_IDLE, "fast1_commit"
                elif slow_ok:
                    output[i], state, reason = top_class, WAIT_IDLE, "candidate_commit"
                elif age >= config.max_candidate_windows:
                    state, reason = READY, "candidate_timeout"
        ages[i] = age
        after.append(state)
        reasons.append(reason)
    return PolicyOutput(output, tuple(before), tuple(after), tuple(reasons), ages)
