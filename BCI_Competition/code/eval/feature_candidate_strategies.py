"""用 Stage 2 EEGNet 隐藏特征的因果时间变化门控候选态提交。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from candidate_state_policy import (
    CandidateEvidence,
    CandidatePolicyResult,
    candidate_state_decisions,
    candidate_transition,
)
from logit_candidate_strategies import (
    LogitStrategyConfig,
    _Stage1Accumulator,
    _Stage2Accumulator,
)
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    ExpectedWindow,
)


FEATURE_DIM = 240
FEATURE_METRICS = {
    "none",
    "unit_velocity_l2",
    "unit_prototype_cosine_distance",
    "unit_acceleration_l2",
}
CONFIG_FIELDS = {
    "strategy_id",
    "base_logit_strategy",
    "feature_metric",
    "feature_max_change",
    "feature_required_consecutive",
}


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} 必须为有限数值")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须为有限数值")
    return result


@dataclass(frozen=True)
class FeatureStrategyConfig:
    """一个 feature gate cell；logit 骨架也完整嵌入，禁止隐式继承。"""

    strategy_id: str
    base_logit_strategy: LogitStrategyConfig
    feature_metric: str
    feature_max_change: float | None
    feature_required_consecutive: int

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureStrategyConfig":
        if not isinstance(payload, dict) or set(payload) != CONFIG_FIELDS:
            raise ValueError("feature 策略字段必须与冻结 schema 完全一致")
        identifier = payload["strategy_id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None:
            raise ValueError("strategy_id 只能使用小写字母、数字和下划线")
        base = LogitStrategyConfig.from_dict(payload["base_logit_strategy"])
        metric = payload["feature_metric"]
        if not isinstance(metric, str) or metric not in FEATURE_METRICS:
            raise ValueError("未知的隐藏特征时间变化指标")
        threshold = payload["feature_max_change"]
        consecutive = payload["feature_required_consecutive"]
        if type(consecutive) is not int:
            raise TypeError("feature_required_consecutive 必须为整数")
        if metric == "none":
            if threshold is not None or consecutive != 0:
                raise ValueError("无特征门控时阈值必须为空且连续窗数必须为 0")
            parsed_threshold = None
        else:
            parsed_threshold = _finite_float(threshold, "feature_max_change")
            if parsed_threshold < 0.0 or not 1 <= consecutive <= base.max_candidate_windows:
                raise ValueError("特征阈值须非负，连续通过窗数须落在候选长度内")
        return cls(identifier, base, metric, parsed_threshold, consecutive)


@dataclass(frozen=True)
class FeatureWindowTrace:
    """保存特征门控前后的证据，便于逐窗独立复算。"""

    evidence: CandidateEvidence
    stage1_filtered_task_probability: float
    stage1_filtered_delta: float
    stage2_candidate_window_count: int
    stage2_top_class: int
    stage2_top_probability: float
    stage2_probability_gap: float
    base_logit_commit_class: int
    feature_metric_value: float
    feature_metric_available: bool
    feature_pass: bool
    feature_pass_streak: int


@dataclass(frozen=True)
class FeatureStrategyResult:
    policy: CandidatePolicyResult
    trace: tuple[FeatureWindowTrace, ...]


# ---------- 候选局部特征历史：先计算当前对历史的变化，再把当前窗写入历史 ----------
class _FeatureAccumulator:
    def __init__(self, config: FeatureStrategyConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.history: list[np.ndarray] = []
        self.prototype_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.pass_streak = 0

    def update(self, feature: np.ndarray) -> tuple[float, bool, bool, int, int]:
        vector = np.asarray(feature, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(vector))
        if (
            vector.shape != (FEATURE_DIM,)
            or not np.isfinite(vector).all()
            or not np.isfinite(norm)
            or not norm > 1e-12
        ):
            raise ValueError("Stage 2 隐藏特征必须是非零、有限的 240 维向量")
        unit = vector / norm
        metric = self.config.feature_metric
        value, available = -1.0, False
        if metric == "none":
            value, available = 0.0, True
        elif metric == "unit_velocity_l2" and self.history:
            value, available = float(np.linalg.norm(unit - self.history[-1])), True
        elif metric == "unit_prototype_cosine_distance" and self.history:
            prototype_norm = float(np.linalg.norm(self.prototype_sum))
            if not prototype_norm > 1e-12:
                raise ValueError("候选特征原型退化为零向量")
            cosine = float(np.dot(unit, self.prototype_sum / prototype_norm))
            value, available = float(np.clip(1.0 - cosine, 0.0, 2.0)), True
        elif metric == "unit_acceleration_l2" and len(self.history) >= 2:
            value = float(np.linalg.norm(unit - 2.0 * self.history[-1] + self.history[-2]))
            available = True
        if available and not np.isfinite(value):
            raise ValueError("隐藏特征时间变化指标不是有限数")

        passed = (
            available
            and (
                metric == "none"
                or value <= float(self.config.feature_max_change)
            )
        )
        self.pass_streak = self.pass_streak + 1 if passed else 0
        self.history.append(unit)
        self.prototype_sum += unit
        return value, available, passed, self.pass_streak, len(self.history)


# ---------- 完整候选策略：Stage 1 撤销优先，特征只门控 Stage 2 的最终提交 ----------
def feature_candidate_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    stage2_features: np.ndarray,
    config: FeatureStrategyConfig,
) -> FeatureStrategyResult:
    if not isinstance(config, FeatureStrategyConfig):
        raise TypeError("config 必须为 FeatureStrategyConfig")
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    features = np.asarray(stage2_features, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        feature_norms = np.linalg.norm(features, axis=1)
    if (
        stage1.shape != (len(windows), 2)
        or stage2.shape != (len(windows), 4)
        or features.shape != (len(windows), FEATURE_DIM)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
        or not np.isfinite(features).all()
        or not np.isfinite(feature_norms).all()
        or np.any(feature_norms <= 1e-12)
    ):
        raise ValueError("logit/特征必须逐窗对齐、有限，特征还必须是非零 240 维")

    base = config.base_logit_strategy
    neutral = CandidateEvidence(False, True, NO_COMMAND, False)
    candidate_state_decisions(
        windows, [neutral] * len(windows), max_candidate_windows=base.max_candidate_windows,
    )
    evidence_rows: list[CandidateEvidence] = []
    trace: list[FeatureWindowTrace] = []
    current_key: tuple[int, int, int, int] | None = None
    state, candidate_age = READY, 0
    stage1_accumulator = _Stage1Accumulator(base)
    stage2_accumulator = _Stage2Accumulator(base)
    feature_accumulator = _FeatureAccumulator(config)

    for index, window in enumerate(windows):
        if window.key != current_key:
            current_key = window.key
            state, candidate_age = READY, 0
            stage1_accumulator = _Stage1Accumulator(base)
            stage2_accumulator.reset()
            feature_accumulator.reset()
        with np.errstate(over="ignore", invalid="ignore"):
            margin = float(stage1[index, 1] - stage1[index, 0])
        if not np.isfinite(margin):
            raise ValueError("Stage 1 task-idle margin 溢出")
        _, filtered_probability, delta = stage1_accumulator.update(margin)

        top_class, top_probability, gap = NO_COMMAND, -1.0, -1.0
        candidate_count, base_commit = 0, NO_COMMAND
        feature_value, feature_available, feature_pass = -1.0, False, False
        feature_streak = 0
        if state == TASK_CANDIDATE:
            (
                top_class,
                top_probability,
                gap,
                stable_windows,
                curvature,
                candidate_count,
            ) = stage2_accumulator.update(stage2[index])
            curvature_ok = (
                base.stage2_max_probability_curvature is None
                or (
                    curvature >= 0.0
                    and curvature <= base.stage2_max_probability_curvature
                )
            )
            if (
                candidate_count >= base.stage2_min_candidate_windows
                and top_probability >= base.stage2_top_probability
                and gap >= base.stage2_probability_gap
                and stable_windows >= base.stage2_stable_windows
                and curvature_ok
            ):
                base_commit = top_class
            (
                feature_value,
                feature_available,
                feature_pass,
                feature_streak,
                feature_count,
            ) = feature_accumulator.update(features[index])
            if feature_count != candidate_count:
                raise RuntimeError("logit 与特征候选历史发生错位")

        feature_gate = (
            config.feature_metric == "none"
            or feature_streak >= config.feature_required_consecutive
        )
        commit_class = base_commit if base_commit != NO_COMMAND and feature_gate else NO_COMMAND
        drop_abort = base.stage1_drop_abort is not None and delta <= -base.stage1_drop_abort
        evidence = CandidateEvidence(
            filtered_probability >= base.task_on_probability,
            filtered_probability >= base.task_hold_probability and not drop_abort,
            commit_class,
            filtered_probability <= base.idle_reset_probability,
        )
        transition = candidate_transition(
            state, candidate_age, evidence, max_candidate_windows=base.max_candidate_windows,
        )
        evidence_rows.append(evidence)
        trace.append(FeatureWindowTrace(
            evidence,
            filtered_probability,
            delta,
            candidate_count,
            top_class,
            top_probability,
            gap,
            base_commit,
            feature_value,
            feature_available,
            feature_pass,
            feature_streak,
        ))
        if transition.transition_reason == CANDIDATE_OPEN:
            stage2_accumulator.reset()
            feature_accumulator.reset()
        elif transition.transition_reason in {
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT,
            COMMAND_COMMIT,
        }:
            stage2_accumulator.reset()
            feature_accumulator.reset()
        state = transition.state_after
        candidate_age = transition.candidate_windows_after

    policy = candidate_state_decisions(
        windows, evidence_rows, max_candidate_windows=base.max_candidate_windows,
    )
    return FeatureStrategyResult(policy, tuple(trace))
