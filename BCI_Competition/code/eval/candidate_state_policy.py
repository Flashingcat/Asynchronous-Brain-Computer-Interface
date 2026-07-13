"""Stage 1 门控、Stage 2 提交与重复触发锁定相分离的三状态决策内核。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    IDLE_RESET,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
)


@dataclass(frozen=True)
class CandidateEvidence:
    """调用方已因果计算好的四项证据；本轮不规定其阈值或聚合算法。"""

    task_on: bool
    task_hold: bool
    stage2_commit_class: int
    idle_reset: bool

    def __post_init__(self) -> None:
        for name in ("task_on", "task_hold", "idle_reset"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} 必须为布尔证据")
            object.__setattr__(self, name, bool(value))
        value = self.stage2_commit_class
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("stage2_commit_class 必须为整数 -1 或 1..4")
        value = int(value)
        if value not in (NO_COMMAND, 1, 2, 3, 4):
            raise ValueError("stage2_commit_class 必须为 -1 或 1..4")
        object.__setattr__(self, "stage2_commit_class", value)


@dataclass(frozen=True)
class CandidateTraceRecord:
    """保留每窗输入证据和候选年龄，便于核对状态机是否按声明运行。"""

    decision: DecisionRecord
    evidence: CandidateEvidence
    candidate_windows_before: int
    candidate_windows_after: int


@dataclass(frozen=True)
class CandidatePolicyResult:
    decisions: tuple[DecisionRecord, ...]
    trace: tuple[CandidateTraceRecord, ...]


@dataclass(frozen=True)
class CandidateTransition:
    """单窗纯状态转移结果，供分数策略和批量轨迹生成共享同一语义。"""

    emitted_class: int
    state_after: str
    transition_reason: str | None
    candidate_windows_after: int


def _validate_max_candidate_windows(max_candidate_windows: int) -> None:
    if type(max_candidate_windows) is not int or max_candidate_windows < 1:
        raise ValueError("max_candidate_windows 必须为正整数")


def candidate_transition(
    state: str,
    candidate_windows: int,
    evidence: CandidateEvidence,
    *,
    max_candidate_windows: int,
) -> CandidateTransition:
    """执行一个窗口的固定优先级；不读取窗口、logit 或真值。"""
    _validate_max_candidate_windows(max_candidate_windows)
    if state not in (READY, TASK_CANDIDATE, WAIT_IDLE):
        raise ValueError("state 必须是 READY、TASK_CANDIDATE 或 WAIT_IDLE")
    if type(candidate_windows) is not int or candidate_windows < 0:
        raise ValueError("candidate_windows 必须为非负整数")
    if not isinstance(evidence, CandidateEvidence):
        raise TypeError("evidence 必须为 CandidateEvidence")
    if state != TASK_CANDIDATE and candidate_windows != 0:
        raise ValueError("非候选态的 candidate_windows 必须为 0")
    if state == TASK_CANDIDATE and candidate_windows >= max_candidate_windows:
        raise ValueError("候选年龄不得在进入本窗前达到最大候选窗数")

    emitted = NO_COMMAND
    reason: str | None = None
    age_after = candidate_windows
    state_after = state
    if state == READY:
        if evidence.task_on:
            state_after = TASK_CANDIDATE
            reason = CANDIDATE_OPEN
            age_after = 0
    elif state == TASK_CANDIDATE:
        age_after += 1
        if not evidence.task_hold:
            state_after = READY
            reason = CANDIDATE_ABORT_STAGE1
            age_after = 0
        elif evidence.stage2_commit_class != NO_COMMAND:
            emitted = evidence.stage2_commit_class
            state_after = WAIT_IDLE
            reason = COMMAND_COMMIT
            age_after = 0
        elif age_after >= max_candidate_windows:
            state_after = READY
            reason = CANDIDATE_TIMEOUT
            age_after = 0
    elif evidence.idle_reset:
        state_after = READY
        reason = IDLE_RESET

    return CandidateTransition(emitted, state_after, reason, age_after)


def _validate_inputs(
    windows: Sequence[ExpectedWindow],
    evidence: Sequence[CandidateEvidence],
    max_candidate_windows: int,
) -> None:
    """拒绝错位、乱序或隐式类型转换，正式策略必须逐窗消费完整母索引。"""
    _validate_max_candidate_windows(max_candidate_windows)
    if len(windows) != len(evidence):
        raise ValueError("证据数量必须与窗口数量完全一致")
    if any(not isinstance(window, ExpectedWindow) for window in windows):
        raise TypeError("windows 必须全部为 ExpectedWindow")
    if any(not isinstance(item, CandidateEvidence) for item in evidence):
        raise TypeError("evidence 必须全部为 CandidateEvidence")

    identities = [(*window.key, window.window_index) for window in windows]
    if identities != sorted(identities):
        raise ValueError("候选策略窗口必须按 subject/session/run/segment/index 排序")
    previous_key: tuple[int, int, int, int] | None = None
    previous_index = -1
    previous_start = -1
    previous_stop = -1
    for window in windows:
        if window.window_stop_sample <= window.window_start_sample:
            raise ValueError("每个窗口必须满足 window_stop_sample > window_start_sample")
        if window.key != previous_key:
            if window.window_index != 0:
                raise ValueError("每个 segment 的 window_index 必须从 0 开始")
            if (
                previous_key is not None
                and window.key[:3] == previous_key[:3]
                and window.window_start_sample < previous_stop
            ):
                raise ValueError("同一 run 的后续 segment 不得早于上一 segment 的末窗")
            previous_key = window.key
            previous_start = window.window_start_sample
            previous_stop = window.window_stop_sample
        elif window.window_index != previous_index + 1:
            raise ValueError("同一 segment 的 window_index 必须连续")
        elif (
            window.window_start_sample <= previous_start
            or window.window_stop_sample <= previous_stop
        ):
            raise ValueError("同一 segment 的窗口起止时间必须严格递增")
        else:
            previous_start = window.window_start_sample
            previous_stop = window.window_stop_sample
        previous_index = window.window_index


def candidate_state_decisions(
    windows: Sequence[ExpectedWindow],
    evidence: Sequence[CandidateEvidence],
    *,
    max_candidate_windows: int,
) -> CandidatePolicyResult:
    """运行可撤销候选态；Stage 2 只在进入候选态后的窗口具有提交权限。"""
    _validate_inputs(windows, evidence, max_candidate_windows)
    decisions: list[DecisionRecord] = []
    traces: list[CandidateTraceRecord] = []
    current_key: tuple[int, int, int, int] | None = None
    state = READY
    candidate_windows = 0

    # 优先级固定为 Stage 1 撤销 > Stage 2 提交 > 候选超时；提交在最后一个
    # 允许窗口仍有机会成功。WAIT_IDLE 只响应复位，复位窗口不得同时重新开门。
    for window, item in zip(windows, evidence):
        if window.key != current_key:
            current_key = window.key
            state = READY
            candidate_windows = 0

        before = state
        age_before = candidate_windows
        transition = candidate_transition(
            state,
            candidate_windows,
            item,
            max_candidate_windows=max_candidate_windows,
        )
        state = transition.state_after
        candidate_windows = transition.candidate_windows_after

        decision = DecisionRecord(
            *window.key,
            window.window_index,
            window.window_start_sample,
            window.window_stop_sample,
            transition.emitted_class,
            before,
            state,
            transition.transition_reason,
        )
        decisions.append(decision)
        traces.append(CandidateTraceRecord(
            decision,
            item,
            age_before,
            candidate_windows,
        ))

    return CandidatePolicyResult(tuple(decisions), tuple(traces))
