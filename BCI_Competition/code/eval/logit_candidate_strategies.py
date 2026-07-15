"""从逐窗 Stage 1/2 logits 因果构造可撤销候选态证据。"""

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
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    ExpectedWindow,
)


STAGE1_FILTERS = {"raw_margin", "ewma_margin", "rolling_margin", "ewma_probability"}
STAGE2_FILTERS = {
    "current_centered_logits",
    "candidate_mean_centered_logits",
    "candidate_ewma_centered_logits",
}
CONFIG_FIELDS = {
    "strategy_id",
    "stage1_filter",
    "stage1_alpha",
    "stage1_window",
    "task_on_probability",
    "task_hold_probability",
    "idle_reset_probability",
    "stage1_drop_abort",
    "stage2_filter",
    "stage2_alpha",
    "stage2_min_candidate_windows",
    "stage2_top_probability",
    "stage2_probability_gap",
    "stage2_stable_windows",
    "stage2_max_probability_curvature",
    "max_candidate_windows",
}


# ---------- 配置 schema：所有阈值显式入 JSON，不允许 bool 冒充数值或整数 ----------
def _strict_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} 必须为有限数值")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须为有限数值")
    return result


def _probability(value: object, name: str, *, minimum: float = 0.0) -> float:
    result = _strict_float(value, name)
    if not minimum <= result <= 1.0:
        raise ValueError(f"{name} 必须位于 [{minimum}, 1]")
    return result


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _strict_float(value, name)


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} 必须为整数")
    return value


@dataclass(frozen=True)
class LogitStrategyConfig:
    """一个完整、无隐式默认值的描述性策略 cell。"""

    strategy_id: str
    stage1_filter: str
    stage1_alpha: float | None
    stage1_window: int | None
    task_on_probability: float
    task_hold_probability: float
    idle_reset_probability: float
    stage1_drop_abort: float | None
    stage2_filter: str
    stage2_alpha: float | None
    stage2_min_candidate_windows: int
    stage2_top_probability: float
    stage2_probability_gap: float
    stage2_stable_windows: int
    stage2_max_probability_curvature: float | None
    max_candidate_windows: int

    @classmethod
    def from_dict(cls, payload: dict) -> "LogitStrategyConfig":
        if not isinstance(payload, dict) or set(payload) != CONFIG_FIELDS:
            raise ValueError("策略字段集合必须与冻结 schema 完全一致")
        strategy_id = payload["strategy_id"]
        if not isinstance(strategy_id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", strategy_id) is None:
            raise ValueError("strategy_id 只能使用小写字母、数字和下划线")
        stage1_filter = payload["stage1_filter"]
        stage2_filter = payload["stage2_filter"]
        if (
            not isinstance(stage1_filter, str)
            or not isinstance(stage2_filter, str)
            or stage1_filter not in STAGE1_FILTERS
            or stage2_filter not in STAGE2_FILTERS
        ):
            raise ValueError("未知的 Stage 1 或 Stage 2 因果滤波方式")

        stage1_alpha = _optional_float(payload["stage1_alpha"], "stage1_alpha")
        stage1_window = payload["stage1_window"]
        stage2_alpha = _optional_float(payload["stage2_alpha"], "stage2_alpha")
        if stage1_filter in {"ewma_margin", "ewma_probability"}:
            if stage1_alpha is None or not 0.0 < stage1_alpha <= 1.0 or stage1_window is not None:
                raise ValueError("Stage 1 EWMA 只接受 (0,1] alpha，且 window 必须为空")
        elif stage1_filter == "rolling_margin":
            if type(stage1_window) is not int or stage1_window < 1 or stage1_alpha is not None:
                raise ValueError("rolling_margin 需要正整数 window，且 alpha 必须为空")
        elif stage1_alpha is not None or stage1_window is not None:
            raise ValueError("raw_margin 的 alpha/window 必须为空")
        if stage2_filter == "candidate_ewma_centered_logits":
            if stage2_alpha is None or not 0.0 < stage2_alpha <= 1.0:
                raise ValueError("Stage 2 EWMA 需要 (0,1] alpha")
        elif stage2_alpha is not None:
            raise ValueError("非 EWMA Stage 2 的 alpha 必须为空")

        on = _probability(payload["task_on_probability"], "task_on_probability")
        hold = _probability(payload["task_hold_probability"], "task_hold_probability")
        reset = _probability(payload["idle_reset_probability"], "idle_reset_probability")
        # task_hold 只控制候选态是否继续，idle_reset 只控制命令发出后的复位；
        # 二者职责分离后不再强制排序，但复位阈值必须低于重新开门阈值以保留滞回区。
        if not hold <= on:
            raise ValueError("Stage 1 阈值必须满足 task_hold <= task_on")
        if not reset < on:
            raise ValueError("Stage 1 阈值必须满足 idle_reset < task_on")
        drop = _optional_float(payload["stage1_drop_abort"], "stage1_drop_abort")
        if drop is not None and not 0.0 < drop <= 1.0:
            raise ValueError("stage1_drop_abort 必须位于 (0,1]")

        minimum = _strict_int(
            payload["stage2_min_candidate_windows"], "stage2_min_candidate_windows",
        )
        stable = _strict_int(payload["stage2_stable_windows"], "stage2_stable_windows")
        maximum = _strict_int(payload["max_candidate_windows"], "max_candidate_windows")
        if not 1 <= stable <= minimum <= maximum:
            raise ValueError("候选窗数必须满足 1 <= stable <= minimum <= maximum")
        curvature = _optional_float(
            payload["stage2_max_probability_curvature"],
            "stage2_max_probability_curvature",
        )
        if curvature is not None and (not 0.0 <= curvature <= 3.0 or minimum < 3):
            raise ValueError("曲率阈值需位于 [0,3]，且最少候选窗数不得小于 3")

        return cls(
            strategy_id=strategy_id,
            stage1_filter=stage1_filter,
            stage1_alpha=stage1_alpha,
            stage1_window=stage1_window,
            task_on_probability=on,
            task_hold_probability=hold,
            idle_reset_probability=reset,
            stage1_drop_abort=drop,
            stage2_filter=stage2_filter,
            stage2_alpha=stage2_alpha,
            stage2_min_candidate_windows=minimum,
            stage2_top_probability=_probability(
                payload["stage2_top_probability"], "stage2_top_probability", minimum=0.25,
            ),
            stage2_probability_gap=_probability(
                payload["stage2_probability_gap"], "stage2_probability_gap",
            ),
            stage2_stable_windows=stable,
            stage2_max_probability_curvature=curvature,
            max_candidate_windows=maximum,
        )


@dataclass(frozen=True)
class LogitWindowTrace:
    """保存每窗真正送入状态机前的因果分数，方便独立重算。"""

    evidence: CandidateEvidence
    stage1_raw_margin: float
    stage1_raw_task_probability: float
    stage1_filtered_task_probability: float
    stage1_filtered_delta: float
    idle_reset_raw_condition: bool
    idle_reset_consecutive_count: int
    stage2_candidate_window_count: int
    stage2_top_class: int
    stage2_top_probability: float
    stage2_probability_gap: float
    stage2_stable_windows: int
    stage2_probability_curvature: float


@dataclass(frozen=True)
class LogitStrategyResult:
    policy: CandidatePolicyResult
    trace: tuple[LogitWindowTrace, ...]


def _sigmoid(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("sigmoid 输入必须有限")
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Stage 2 softmax 输入必须是四维有限向量")
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - np.max(values)
    if not np.isfinite(shifted).all():
        raise ValueError("Stage 2 softmax 类别差值溢出")
    exp = np.exp(shifted)
    probability = exp / np.sum(exp)
    if not np.isfinite(probability).all():
        raise ValueError("Stage 2 softmax 派生概率非有限")
    return probability


def _centered_stage2_logits(logits: np.ndarray) -> np.ndarray:
    """消除四类 logit 的公共平移，并对极端有限数做溢出保护。"""
    raw = np.asarray(logits, dtype=np.float64)
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError("Stage 2 单窗 logits 必须是四维有限向量")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = raw - float(np.mean(raw))
    if not np.isfinite(centered).all():
        # 公共超大平移本应可消除；先减参考值可避免四个 1e308 求和溢出。
        with np.errstate(over="ignore", invalid="ignore"):
            offsets = raw - raw[0]
            centered = offsets - float(np.mean(offsets))
    if not np.isfinite(centered).all():
        raise ValueError("Stage 2 中心化 logits 溢出")
    return centered


# ---------- Stage 1 因果历史：margin-EWMA 与 probability-EWMA 的运算顺序不同 ----------
class _Stage1Accumulator:
    def __init__(self, config: LogitStrategyConfig) -> None:
        self.config = config
        self.values: list[float] = []
        self.filtered: float | None = None
        self.previous_probability: float | None = None

    def update(self, margin: float) -> tuple[float, float, float]:
        margin = _strict_float(margin, "stage1_margin")
        raw_probability = _sigmoid(margin)
        kind = self.config.stage1_filter
        if kind == "raw_margin":
            filtered_probability = raw_probability
        elif kind == "rolling_margin":
            self.values.append(margin)
            self.values = self.values[-int(self.config.stage1_window):]
            filtered_probability = _sigmoid(float(np.mean(self.values)))
        else:
            source = raw_probability if kind == "ewma_probability" else margin
            alpha = float(self.config.stage1_alpha)
            self.filtered = source if self.filtered is None else alpha * source + (1 - alpha) * self.filtered
            filtered_probability = self.filtered if kind == "ewma_probability" else _sigmoid(self.filtered)
        if not np.isfinite(filtered_probability):
            raise ValueError("Stage 1 因果聚合产生非有限分数")
        delta = (
            0.0
            if self.previous_probability is None
            else float(filtered_probability - self.previous_probability)
        )
        if not np.isfinite(delta):
            raise ValueError("Stage 1 概率差分产生非有限分数")
        self.previous_probability = float(filtered_probability)
        return raw_probability, float(filtered_probability), delta


# ---------- Stage 2 候选历史：raw top1 管稳定性，聚合 logits 管类别与置信度 ----------
class _Stage2Accumulator:
    def __init__(self, config: LogitStrategyConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum_logits = np.zeros(4, dtype=np.float64)
        self.ewma_logits: np.ndarray | None = None
        self.probability_history: list[np.ndarray] = []
        self.last_raw_class = NO_COMMAND
        self.stable_count = 0

    def update(self, logits: np.ndarray) -> tuple[int, float, float, int, float, int]:
        centered = _centered_stage2_logits(logits)
        raw_probability = _softmax(centered)
        raw_class = int(np.argmax(raw_probability)) + 1
        self.stable_count = self.stable_count + 1 if raw_class == self.last_raw_class else 1
        self.last_raw_class = raw_class
        self.count += 1
        with np.errstate(over="ignore", invalid="ignore"):
            self.sum_logits += centered
        if not np.isfinite(self.sum_logits).all():
            raise ValueError("Stage 2 候选均值累加溢出")
        if self.config.stage2_filter == "candidate_ewma_centered_logits":
            alpha = float(self.config.stage2_alpha)
            self.ewma_logits = (
                centered.copy()
                if self.ewma_logits is None
                else alpha * centered + (1 - alpha) * self.ewma_logits
            )
            if not np.isfinite(self.ewma_logits).all():
                raise ValueError("Stage 2 候选 EWMA 产生非有限分数")
        if self.config.stage2_filter == "current_centered_logits":
            aggregate = centered
        elif self.config.stage2_filter == "candidate_mean_centered_logits":
            aggregate = self.sum_logits / self.count
        else:
            aggregate = self.ewma_logits
        if aggregate is None or not np.isfinite(aggregate).all():
            raise ValueError("Stage 2 聚合 logits 产生非有限分数")
        probability = _softmax(aggregate)
        order = np.sort(probability)
        top_class = int(np.argmax(probability)) + 1
        effective_stability = self.stable_count if top_class == raw_class else 0
        curvature = -1.0
        self.probability_history.append(raw_probability)
        self.probability_history = self.probability_history[-3:]
        if len(self.probability_history) == 3:
            curvature = float(np.linalg.norm(
                self.probability_history[2]
                - 2.0 * self.probability_history[1]
                + self.probability_history[0]
            ))
            if not np.isfinite(curvature):
                raise ValueError("Stage 2 单窗 raw softmax 二阶曲率非有限")
        return (
            top_class,
            float(order[-1]),
            float(order[-1] - order[-2]),
            effective_stability,
            curvature,
            self.count,
        )


def logit_candidate_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    config: LogitStrategyConfig,
    *,
    idle_reset_consecutive_windows: int = 1,
) -> LogitStrategyResult:
    """逐 segment 因果更新分数；候选专属 Stage 2 历史绝不包含开门窗。"""
    if not isinstance(config, LogitStrategyConfig):
        raise TypeError("config 必须为 LogitStrategyConfig")
    if type(idle_reset_consecutive_windows) is not int or idle_reset_consecutive_windows < 1:
        raise ValueError("idle_reset_consecutive_windows 必须为正整数")
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if (
        stage1.shape != (len(windows), 2)
        or stage2.shape != (len(windows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise ValueError("Stage 1/2 logits 必须逐窗对齐、有限且维度为 2/4")

    neutral = CandidateEvidence(False, True, NO_COMMAND, False)
    candidate_state_decisions(
        windows,
        [neutral] * len(windows),
        max_candidate_windows=config.max_candidate_windows,
    )
    evidence_rows: list[CandidateEvidence] = []
    trace: list[LogitWindowTrace] = []
    current_key: tuple[int, int, int, int] | None = None
    state = READY
    candidate_age = 0
    idle_reset_count = 0
    stage1_accumulator = _Stage1Accumulator(config)
    stage2_accumulator = _Stage2Accumulator(config)

    # ---------- 逐窗主循环：Stage 1 始终更新，Stage 2 仅消费开门后的候选窗 ----------
    for index, window in enumerate(windows):
        if window.key != current_key:
            current_key = window.key
            state, candidate_age, idle_reset_count = READY, 0, 0
            stage1_accumulator = _Stage1Accumulator(config)
            stage2_accumulator.reset()
        with np.errstate(over="ignore", invalid="ignore"):
            margin = float(stage1[index, 1] - stage1[index, 0])
        if not np.isfinite(margin):
            raise ValueError("Stage 1 task-idle margin 溢出")
        raw_probability, filtered_probability, delta = stage1_accumulator.update(margin)

        top_class, top_probability, gap = NO_COMMAND, -1.0, -1.0
        stable_windows, curvature, candidate_count = 0, -1.0, 0
        commit_class = NO_COMMAND
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
                config.stage2_max_probability_curvature is None
                or (
                    curvature >= 0.0
                    and curvature <= config.stage2_max_probability_curvature
                )
            )
            if (
                candidate_count >= config.stage2_min_candidate_windows
                and top_probability >= config.stage2_top_probability
                and gap >= config.stage2_probability_gap
                and stable_windows >= config.stage2_stable_windows
                and curvature_ok
            ):
                commit_class = top_class

        drop_abort = (
            config.stage1_drop_abort is not None
            and delta <= -config.stage1_drop_abort
        )

        # 连续复位证据只在 WAIT_IDLE 内累计；任一不满足窗口、离开等待态或
        # segment 切换都会清零。这里保存原始条件与连续计数，方便独立复算。
        idle_reset_raw = filtered_probability <= config.idle_reset_probability
        if state == WAIT_IDLE and idle_reset_raw:
            idle_reset_count += 1
        else:
            idle_reset_count = 0
        idle_reset_confirmed = (
            state == WAIT_IDLE
            and idle_reset_count >= idle_reset_consecutive_windows
        )
        evidence = CandidateEvidence(
            filtered_probability >= config.task_on_probability,
            filtered_probability >= config.task_hold_probability and not drop_abort,
            commit_class,
            idle_reset_confirmed,
        )
        transition = candidate_transition(
            state,
            candidate_age,
            evidence,
            max_candidate_windows=config.max_candidate_windows,
        )
        evidence_rows.append(evidence)
        trace.append(LogitWindowTrace(
            evidence,
            margin,
            raw_probability,
            filtered_probability,
            delta,
            idle_reset_raw,
            idle_reset_count,
            candidate_count,
            top_class,
            top_probability,
            gap,
            stable_windows,
            curvature,
        ))

        if transition.transition_reason == CANDIDATE_OPEN:
            stage2_accumulator.reset()
        elif transition.transition_reason in {
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT,
            COMMAND_COMMIT,
        }:
            stage2_accumulator.reset()
        state = transition.state_after
        candidate_age = transition.candidate_windows_after
        if state != WAIT_IDLE:
            idle_reset_count = 0

    policy = candidate_state_decisions(
        windows,
        evidence_rows,
        max_candidate_windows=config.max_candidate_windows,
    )
    return LogitStrategyResult(policy, tuple(trace))
