"""联合五分类硬标签上的因果多窗口投票策略。"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from protocol_metrics import (
    NO_COMMAND,
    READY,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
)


WINDOW_COUNTS = (2, 3, 4, 5)
VOTE_GRID = tuple(
    (window_count, vote_threshold)
    for window_count in WINDOW_COUNTS
    for vote_threshold in range(window_count // 2 + 1, window_count + 1)
)


def policy_id(window_count: int, vote_threshold: int) -> str:
    """生成稳定、可直接用于 JSON 与 NPZ 字段名的策略编号。"""
    _validate_policy(window_count, vote_threshold)
    return f"n{window_count}_k{vote_threshold}"


def _validate_policy(window_count: int, vote_threshold: int) -> None:
    """首版只允许 N=2..5，K 从严格多数到全票。"""
    if type(window_count) is not int or window_count not in WINDOW_COUNTS:
        raise ValueError(f"window_count 只能取 {WINDOW_COUNTS}")
    if (
        type(vote_threshold) is not int
        or vote_threshold <= window_count // 2
        or vote_threshold > window_count
    ):
        raise ValueError("vote_threshold 必须大于 N/2 且不超过 N")


def stateful_hard_vote_decisions(
    windows: Sequence[ExpectedWindow],
    hard_labels: Sequence[int] | np.ndarray,
    *,
    window_count: int,
    vote_threshold: int,
) -> list[DecisionRecord]:
    """按最近 N 个联合硬标签投票，并在每次状态转换后清空缓存。"""
    _validate_policy(window_count, vote_threshold)
    labels = np.asarray(hard_labels)
    if (
        labels.ndim != 1
        or labels.shape[0] != len(windows)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels > 4))
    ):
        raise ValueError("hard_labels 必须是与窗口逐行对齐的整数 0..4")

    identities = [(*window.key, window.window_index) for window in windows]
    if identities != sorted(identities):
        raise ValueError("投票窗口必须按 subject/session/run/segment/index 排序")

    decisions: list[DecisionRecord] = []
    votes: deque[int] = deque(maxlen=window_count)
    current_key: tuple[int, int, int, int] | None = None
    state = READY

    # 每个 segment 独立初始化。只有缓存已含完整 N 票时才允许转换，避免把短缓存
    # 偷换成另一个窗口数量；K 对 MI 输出和 IDLE 复位采用完全相同的定义。
    for index, window in enumerate(windows):
        if window.key != current_key:
            current_key, state = window.key, READY
            votes.clear()

        before = state
        emitted = NO_COMMAND
        votes.append(int(labels[index]))
        if len(votes) == window_count:
            counts = np.bincount(np.asarray(votes), minlength=5)
            if state == READY:
                winners = np.flatnonzero(counts[1:] >= vote_threshold) + 1
                # K>N/2 保证最多只有一个类别过阈值；保留断言防止规则以后漂移。
                if len(winners) > 1:
                    raise RuntimeError("严格多数规则不应产生多个 MI 获胜类别")
                if len(winners) == 1:
                    emitted = int(winners[0])
                    state = WAIT_IDLE
                    votes.clear()
            elif counts[0] >= vote_threshold:
                state = READY
                votes.clear()

        decisions.append(DecisionRecord(
            *window.key,
            window.window_index,
            window.window_start_sample,
            window.window_stop_sample,
            emitted,
            before,
            state,
        ))
    return decisions
