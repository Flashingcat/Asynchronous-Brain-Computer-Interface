"""在历史慢通道前增加 Fast-0/Fast-1，并保证未命中时逐窗回退原策略。"""

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
    _centered_stage2_logits,
    _softmax,
)
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    FAST0_COMMAND_COMMIT,
    FAST1_COMMAND_COMMIT,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    ExpectedWindow,
)


FAST0_FIELDS = {
    "min_stage1_probability",
    "min_stage1_delta",
    "min_stage2_top_probability",
    "min_stage2_probability_gap",
}
FAST1_FIELDS = {
    "min_stage1_probability",
    "stage2_alpha",
    "min_stage2_top_probability",
    "min_stage2_probability_gap",
    "require_same_raw_class",
}
CONFIG_FIELDS = {
    "strategy_id",
    "base_logit_strategy",
    "idle_reset_consecutive_windows",
    "fast0",
    "fast1",
}
COMMIT_PATHS = {"none", "fast0", "fast1", "slow"}


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} 必须为有限数值")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须为有限数值")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]")
    return result


@dataclass(frozen=True)
class Fast0Config:
    """Fast-0 只在开门窗上使用当前窗 Stage 1/2 证据。"""

    min_stage1_probability: float
    min_stage1_delta: float
    min_stage2_top_probability: float
    min_stage2_probability_gap: float

    @classmethod
    def from_dict(cls, payload: dict) -> "Fast0Config":
        if not isinstance(payload, dict) or set(payload) != FAST0_FIELDS:
            raise ValueError("Fast-0 字段必须与冻结 schema 完全一致")
        delta = _finite_float(payload["min_stage1_delta"], "Fast-0 Stage 1 delta")
        if delta < 0.0:
            raise ValueError("Fast-0 Stage 1 delta 不得为负")
        return cls(
            _probability(payload["min_stage1_probability"], "Fast-0 Stage 1 概率"),
            delta,
            _probability(payload["min_stage2_top_probability"], "Fast-0 Stage 2 top 概率"),
            _probability(payload["min_stage2_probability_gap"], "Fast-0 Stage 2 gap"),
        )


@dataclass(frozen=True)
class Fast1Config:
    """Fast-1 只聚合开门窗与紧随的下一窗，不写入慢通道缓存。"""

    min_stage1_probability: float
    stage2_alpha: float
    min_stage2_top_probability: float
    min_stage2_probability_gap: float
    require_same_raw_class: bool

    @classmethod
    def from_dict(cls, payload: dict) -> "Fast1Config":
        if not isinstance(payload, dict) or set(payload) != FAST1_FIELDS:
            raise ValueError("Fast-1 字段必须与冻结 schema 完全一致")
        alpha = _finite_float(payload["stage2_alpha"], "Fast-1 Stage 2 alpha")
        same_class = payload["require_same_raw_class"]
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Fast-1 Stage 2 alpha 必须位于 (0, 1]")
        if type(same_class) is not bool:
            raise TypeError("Fast-1 require_same_raw_class 必须为布尔值")
        return cls(
            _probability(payload["min_stage1_probability"], "Fast-1 Stage 1 概率"),
            alpha,
            _probability(payload["min_stage2_top_probability"], "Fast-1 Stage 2 top 概率"),
            _probability(payload["min_stage2_probability_gap"], "Fast-1 Stage 2 gap"),
            same_class,
        )


@dataclass(frozen=True)
class FastPathConfig:
    """一个完整快速通道 cell；None 表示关闭对应通道而非使用隐式默认值。"""

    strategy_id: str
    base_logit_strategy: LogitStrategyConfig
    idle_reset_consecutive_windows: int
    fast0: Fast0Config | None
    fast1: Fast1Config | None

    @classmethod
    def from_dict(cls, payload: dict) -> "FastPathConfig":
        if not isinstance(payload, dict) or set(payload) != CONFIG_FIELDS:
            raise ValueError("快速通道策略字段必须与冻结 schema 完全一致")
        identifier = payload["strategy_id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None:
            raise ValueError("strategy_id 只能使用小写字母、数字和下划线")
        base = LogitStrategyConfig.from_dict(payload["base_logit_strategy"])
        reset_windows = payload["idle_reset_consecutive_windows"]
        if type(reset_windows) is not int or reset_windows < 1:
            raise ValueError("idle_reset_consecutive_windows 必须为正整数")
        fast0 = None if payload["fast0"] is None else Fast0Config.from_dict(payload["fast0"])
        fast1 = None if payload["fast1"] is None else Fast1Config.from_dict(payload["fast1"])
        if fast0 is not None and fast0.min_stage1_probability < base.task_on_probability:
            raise ValueError("Fast-0 Stage 1 阈值不得低于慢通道开门阈值")
        if fast1 is not None and fast1.min_stage1_probability < base.task_hold_probability:
            raise ValueError("Fast-1 Stage 1 阈值不得低于慢通道保持阈值")
        return cls(identifier, base, reset_windows, fast0, fast1)


@dataclass(frozen=True)
class FastPathWindowTrace:
    """保存每窗快/慢证据，可独立核对快速通道没有污染慢通道。"""

    evidence: CandidateEvidence
    stage1_filtered_task_probability: float
    stage1_filtered_delta: float
    raw_stage2_top_class: int
    raw_stage2_top_probability: float
    raw_stage2_probability_gap: float
    fast0_evaluated: bool
    fast0_pass: bool
    fast1_evaluated: bool
    fast1_same_raw_class: bool
    fast1_top_class: int
    fast1_top_probability: float
    fast1_probability_gap: float
    fast1_pass: bool
    slow_candidate_window_count: int
    slow_top_class: int
    slow_top_probability: float
    slow_probability_gap: float
    proposed_commit_path: str
    idle_reset_raw_condition: bool
    idle_reset_consecutive_count: int


@dataclass(frozen=True)
class FastPathResult:
    policy: CandidatePolicyResult
    trace: tuple[FastPathWindowTrace, ...]


def _top_summary(centered_logits: np.ndarray) -> tuple[int, float, float]:
    probability = _softmax(centered_logits)
    order = np.sort(probability)
    return int(np.argmax(probability)) + 1, float(order[-1]), float(order[-1] - order[-2])


# ---------- 完整因果主循环：快速缓存与慢通道缓存物理分离 ----------
def fast_path_candidate_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    config: FastPathConfig,
) -> FastPathResult:
    if not isinstance(config, FastPathConfig):
        raise TypeError("config 必须为 FastPathConfig")
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if (
        stage1.shape != (len(windows), 2)
        or stage2.shape != (len(windows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise ValueError("Stage 1/2 logits 必须逐窗对齐、有限且维度为 2/4")

    base = config.base_logit_strategy
    neutral = CandidateEvidence(False, True, NO_COMMAND, False)
    candidate_state_decisions(
        windows,
        [neutral] * len(windows),
        max_candidate_windows=base.max_candidate_windows,
    )
    evidence_rows: list[CandidateEvidence] = []
    trace: list[FastPathWindowTrace] = []
    first_pass_transitions: list[tuple[str, str, str | None, int]] = []
    current_key: tuple[int, int, int, int] | None = None
    state, candidate_age, idle_reset_count = READY, 0, 0
    stage1_accumulator = _Stage1Accumulator(base)
    slow_accumulator = _Stage2Accumulator(base)
    fast_open_logits: np.ndarray | None = None
    fast_open_raw_class = NO_COMMAND
    fast_open_stage1_probability = -1.0

    for index, window in enumerate(windows):
        if window.key != current_key:
            current_key = window.key
            state, candidate_age, idle_reset_count = READY, 0, 0
            stage1_accumulator = _Stage1Accumulator(base)
            slow_accumulator.reset()
            fast_open_logits = None
            fast_open_raw_class = NO_COMMAND
            fast_open_stage1_probability = -1.0

        with np.errstate(over="ignore", invalid="ignore"):
            margin = float(stage1[index, 1] - stage1[index, 0])
        if not np.isfinite(margin):
            raise ValueError("Stage 1 task-idle margin 溢出")
        _, filtered_probability, delta = stage1_accumulator.update(margin)
        centered = _centered_stage2_logits(stage2[index])
        raw_class, raw_top, raw_gap = _top_summary(centered)

        drop_abort = base.stage1_drop_abort is not None and delta <= -base.stage1_drop_abort
        task_on = filtered_probability >= base.task_on_probability
        task_hold = filtered_probability >= base.task_hold_probability and not drop_abort

        # 慢通道仍只消费开门后的窗，所有原锚点语义保持不变。
        slow_class, slow_top, slow_gap = NO_COMMAND, -1.0, -1.0
        slow_count, slow_stability, slow_curvature = 0, 0, -1.0
        slow_commit = NO_COMMAND
        if state == TASK_CANDIDATE:
            (
                slow_class,
                slow_top,
                slow_gap,
                slow_stability,
                slow_curvature,
                slow_count,
            ) = slow_accumulator.update(stage2[index])
            curvature_ok = (
                base.stage2_max_probability_curvature is None
                or (
                    slow_curvature >= 0.0
                    and slow_curvature <= base.stage2_max_probability_curvature
                )
            )
            if (
                slow_count >= base.stage2_min_candidate_windows
                and slow_top >= base.stage2_top_probability
                and slow_gap >= base.stage2_probability_gap
                and slow_stability >= base.stage2_stable_windows
                and curvature_ok
            ):
                slow_commit = slow_class

        # Fast-0 只在 READY 且 Stage 1 已可开门的同一窗检查。
        fast0_evaluated = state == READY and task_on and config.fast0 is not None
        fast0_pass = False
        if fast0_evaluated:
            gate0 = config.fast0
            if gate0 is None:  # pragma: no cover - 上方条件已缩小类型
                raise RuntimeError("Fast-0 配置意外为空")
            fast0_pass = (
                filtered_probability >= gate0.min_stage1_probability
                and delta >= gate0.min_stage1_delta
                and raw_top >= gate0.min_stage2_top_probability
                and raw_gap >= gate0.min_stage2_probability_gap
            )

        # Fast-1 只在开门后第 1 个窗检查一次；失败后立即丢弃快速缓存。
        fast1_evaluated = (
            state == TASK_CANDIDATE
            and candidate_age == 0
            and config.fast1 is not None
            and fast_open_logits is not None
        )
        fast1_same_class = False
        fast1_class, fast1_top, fast1_gap = NO_COMMAND, -1.0, -1.0
        fast1_pass = False
        if fast1_evaluated:
            gate1 = config.fast1
            if gate1 is None:  # pragma: no cover - 上方条件已缩小类型
                raise RuntimeError("Fast-1 配置意外为空")
            fast1_same_class = raw_class == fast_open_raw_class
            aggregate = gate1.stage2_alpha * centered + (1.0 - gate1.stage2_alpha) * fast_open_logits
            fast1_class, fast1_top, fast1_gap = _top_summary(aggregate)
            fast1_pass = (
                task_hold
                and min(filtered_probability, fast_open_stage1_probability)
                >= gate1.min_stage1_probability
                and (fast1_same_class or not gate1.require_same_raw_class)
                and fast1_top >= gate1.min_stage2_top_probability
                and fast1_gap >= gate1.min_stage2_probability_gap
            )

        fast_commit = NO_COMMAND
        proposed_path = "none"
        if fast0_pass:
            fast_commit, proposed_path = raw_class, "fast0"
        elif fast1_pass:
            fast_commit, proposed_path = fast1_class, "fast1"
        elif slow_commit != NO_COMMAND:
            proposed_path = "slow"
        if proposed_path not in COMMIT_PATHS:
            raise RuntimeError("未知的提交路径")

        idle_reset_raw = filtered_probability <= base.idle_reset_probability
        if state == WAIT_IDLE and idle_reset_raw:
            idle_reset_count += 1
        else:
            idle_reset_count = 0
        idle_reset_confirmed = (
            state == WAIT_IDLE
            and idle_reset_count >= config.idle_reset_consecutive_windows
        )
        evidence = CandidateEvidence(
            task_on,
            task_hold,
            slow_commit,
            idle_reset_confirmed,
            fast_commit,
        )
        before = state
        transition = candidate_transition(
            state,
            candidate_age,
            evidence,
            max_candidate_windows=base.max_candidate_windows,
        )
        evidence_rows.append(evidence)
        first_pass_transitions.append((
            before,
            transition.state_after,
            transition.transition_reason,
            transition.emitted_class,
        ))
        trace.append(FastPathWindowTrace(
            evidence,
            filtered_probability,
            delta,
            raw_class,
            raw_top,
            raw_gap,
            fast0_evaluated,
            fast0_pass,
            fast1_evaluated,
            fast1_same_class,
            fast1_class,
            fast1_top,
            fast1_gap,
            fast1_pass,
            slow_count,
            slow_class,
            slow_top,
            slow_gap,
            proposed_path,
            idle_reset_raw,
            idle_reset_count,
        ))

        first_candidate_window = state == TASK_CANDIDATE and candidate_age == 0
        reason = transition.transition_reason
        if reason == CANDIDATE_OPEN:
            slow_accumulator.reset()
            if config.fast1 is not None:
                fast_open_logits = centered.copy()
                fast_open_raw_class = raw_class
                fast_open_stage1_probability = filtered_probability
        elif reason in {
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT,
            COMMAND_COMMIT,
            FAST0_COMMAND_COMMIT,
            FAST1_COMMAND_COMMIT,
        }:
            slow_accumulator.reset()
            fast_open_logits = None
            fast_open_raw_class = NO_COMMAND
            fast_open_stage1_probability = -1.0
        elif first_candidate_window:
            fast_open_logits = None
            fast_open_raw_class = NO_COMMAND
            fast_open_stage1_probability = -1.0

        state = transition.state_after
        candidate_age = transition.candidate_windows_after
        if state != WAIT_IDLE:
            idle_reset_count = 0

    # 用纯状态机对已保存证据重放，防止分数循环与正式轨迹分叉。
    policy = candidate_state_decisions(
        windows,
        evidence_rows,
        max_candidate_windows=base.max_candidate_windows,
    )
    replayed = [
        (
            item.decision_state_before,
            item.decision_state_after,
            item.transition_reason,
            item.emitted_class,
        )
        for item in policy.decisions
    ]
    if replayed != first_pass_transitions:
        raise RuntimeError("快速通道分数循环与状态机重放结果不一致")
    return FastPathResult(policy, tuple(trace))
