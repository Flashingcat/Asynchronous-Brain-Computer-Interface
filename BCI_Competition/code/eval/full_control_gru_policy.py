"""连续五分类 GRU：模型负责 MI 提交类别与 IDLE 复位证据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from protocol_metrics import (
    NO_COMMAND,
    READY,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
)


TOKEN_DIM = 10
HIDDEN_DIM = 16
CLASS_COUNT = 5
IGNORE_TARGET = -100


# ---------- 连续因果 token：只使用当前/过去窗口，segment 边界处变化量清零 ----------
def build_continuous_tokens(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
) -> np.ndarray:
    """构造 margin、四类中心化 logit 及各自一阶变化，共 10 维。"""
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    count = len(windows)
    if (
        stage1.shape != (count, 2)
        or stage2.shape != (count, 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise ValueError("连续 GRU 的 Stage 1/2 logits 形状或有限性非法")

    result = np.zeros((count, TOKEN_DIM), dtype=np.float32)
    previous_key: tuple[int, int, int, int] | None = None
    previous_margin = 0.0
    previous_centered = np.zeros(4, dtype=np.float64)
    for index, window in enumerate(windows):
        margin = float(stage1[index, 1] - stage1[index, 0])
        centered = stage2[index] - np.mean(stage2[index])
        if window.key == previous_key:
            margin_delta = margin - previous_margin
            centered_delta = centered - previous_centered
        else:
            margin_delta = 0.0
            centered_delta = np.zeros(4, dtype=np.float64)
        result[index] = np.asarray([
            margin,
            margin_delta,
            *centered.tolist(),
            *centered_delta.tolist(),
        ], dtype=np.float32)
        previous_key = window.key
        previous_margin = margin
        previous_centered = centered
    return result


# ---------- 监督标签：合法首次提交窗标四类 MI，干净 IDLE 标 0，边界窗忽略 ----------
def build_continuous_targets(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    *,
    margin_samples: int = 125,
) -> np.ndarray:
    """标签资格与正式 evaluator 的 overlap/offset 规则保持一致。"""
    if type(margin_samples) is not int or margin_samples < 1:
        raise ValueError("margin_samples 必须是正整数")
    events_by_key: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    for event in events:
        events_by_key.setdefault(event.key, []).append(event)

    targets = np.full(len(windows), IGNORE_TARGET, dtype=np.int64)
    for index, window in enumerate(windows):
        overlaps: list[tuple[MIEvent, int]] = []
        eligible: list[MIEvent] = []
        for event in events_by_key.get(window.key, []):
            overlap = max(
                0,
                min(window.window_stop_sample, event.offset_sample)
                - max(window.window_start_sample, event.onset_sample),
            )
            if overlap > 0:
                overlaps.append((event, overlap))
            if window.window_stop_sample <= event.offset_sample and overlap >= margin_samples:
                eligible.append(event)
        if len(eligible) > 1:
            raise ValueError("一个窗口不得同时成为多个事件的合法提交点")
        if eligible:
            targets[index] = eligible[0].true_class
        elif not overlaps:
            targets[index] = 0
        # 与 MI 仅部分相交但不具备正式提交资格的窗口保持 IGNORE_TARGET。
    return targets


@dataclass(frozen=True)
class ContinuousSequence:
    key: tuple[int, int, int, int]
    subject_id: int
    tokens: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.targets)
        if (
            len(self.key) != 4
            or self.key[0] != self.subject_id
            or np.asarray(self.tokens).shape != (length, TOKEN_DIM)
            or np.asarray(self.targets).shape != (length,)
            or length < 1
            or not np.isfinite(self.tokens).all()
            or np.any(~np.isin(self.targets, [IGNORE_TARGET, 0, 1, 2, 3, 4]))
        ):
            raise ValueError("连续 segment 序列非法")


def split_continuous_sequences(
    windows: Sequence[ExpectedWindow],
    tokens: np.ndarray,
    targets: np.ndarray,
) -> tuple[ContinuousSequence, ...]:
    """按冻结 segment 分组，禁止跨伪迹排除区间传播 GRU 隐状态。"""
    if np.asarray(tokens).shape != (len(windows), TOKEN_DIM) or np.asarray(targets).shape != (len(windows),):
        raise ValueError("窗口、token 与 target 数量不一致")
    groups: list[ContinuousSequence] = []
    start = 0
    while start < len(windows):
        key = windows[start].key
        stop = start + 1
        while stop < len(windows) and windows[stop].key == key:
            stop += 1
        groups.append(ContinuousSequence(
            key,
            key[0],
            np.asarray(tokens[start:stop], dtype=np.float32).copy(),
            np.asarray(targets[start:stop], dtype=np.int64).copy(),
        ))
        start = stop
    return tuple(groups)


# ---------- 标准化与 padding：统计量只允许来自当前训练被试 ----------
@dataclass(frozen=True)
class ContinuousNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if (
            np.asarray(self.mean).shape != (TOKEN_DIM,)
            or np.asarray(self.std).shape != (TOKEN_DIM,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("连续 token 标准化参数非法")

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        value = np.asarray(tokens, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != TOKEN_DIM or not np.isfinite(value).all():
            raise ValueError("待标准化连续 token 必须为 [T,10] 有限数组")
        return ((value - self.mean) / self.std).astype(np.float32)


def fit_continuous_normalizer(sequences: Iterable[ContinuousSequence]) -> ContinuousNormalizer:
    rows = [item.tokens for item in sequences]
    if not rows:
        raise ValueError("至少需要一个训练 segment 拟合标准化")
    values = np.concatenate(rows, axis=0).astype(np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ContinuousNormalizer(mean.astype(np.float32), std.astype(np.float32))


@dataclass(frozen=True)
class ContinuousTensorSet:
    tokens: np.ndarray
    targets: np.ndarray
    valid_mask: np.ndarray
    keys: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        count = len(self.keys)
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[0] != count
            or self.tokens.shape[2] != TOKEN_DIM
            or self.targets.shape != self.tokens.shape[:2]
            or self.valid_mask.shape != self.tokens.shape[:2]
            or not np.isfinite(self.tokens).all()
            or np.any(self.valid_mask.sum(axis=1) < 1)
            or np.any(~np.isin(self.targets, [IGNORE_TARGET, 0, 1, 2, 3, 4]))
            or any(len(key) != 4 for key in self.keys)
        ):
            raise ValueError("padding 后的连续 TensorSet 非法")


def tensorize_continuous_sequences(
    sequences: Sequence[ContinuousSequence],
    normalizer: ContinuousNormalizer,
) -> ContinuousTensorSet:
    if not sequences:
        raise ValueError("连续训练集不得为空")
    maximum = max(len(item.targets) for item in sequences)
    tokens = np.zeros((len(sequences), maximum, TOKEN_DIM), dtype=np.float32)
    targets = np.full((len(sequences), maximum), IGNORE_TARGET, dtype=np.int64)
    valid = np.zeros((len(sequences), maximum), dtype=np.bool_)
    for row, sequence in enumerate(sequences):
        length = len(sequence.targets)
        tokens[row, :length] = normalizer.transform(sequence.tokens)
        targets[row, :length] = sequence.targets
        valid[row, :length] = True
    return ContinuousTensorSet(tokens, targets, valid, tuple(item.key for item in sequences))


# ---------- 联合五类 GRU：一维 TASK 状态头 + 四维条件类别头 ----------
class FullControlGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(TOKEN_DIM, HIDDEN_DIM, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, CLASS_COUNT)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError("FullControlGRU 输入必须为 [B,T,10]")
        hidden_rows, final_hidden = self.gru(tokens)
        return hidden_rows, self.head(hidden_rows)

    def step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token.ndim != 2 or token.shape[-1] != TOKEN_DIM:
            raise ValueError("逐窗 token 必须为 [B,10]")
        row, next_hidden = self.gru(token[:, None, :], hidden)
        return next_hidden, self.head(row[:, 0])


def balanced_full_control_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """分别等权 IDLE/TASK 和四个 MI 类，避免连续 IDLE 数量淹没任务窗。"""
    if logits.shape[:2] != targets.shape or targets.shape != valid_mask.shape or logits.shape[-1] != 5:
        raise ValueError("连续五分类 loss 输入形状不一致")
    labeled = valid_mask & (targets != IGNORE_TARGET)
    idle = labeled & (targets == 0)
    task = labeled & (targets > 0)
    if not torch.any(idle) or not torch.any(task):
        raise ValueError("每个正式 split 必须同时包含 IDLE 和 TASK 标签")

    # 第一维是独立 TASK logit，避免普通五类 softmax 因有四个 MI 类而自带 80% TASK 先验。
    binary_targets = (targets > 0).to(logits.dtype)
    binary_nll = F.binary_cross_entropy_with_logits(
        logits[..., 0], binary_targets, reduction="none",
    )
    state_loss = 0.5 * binary_nll[idle].mean() + 0.5 * binary_nll[task].mean()

    class_nll = F.cross_entropy(
        logits[..., 1:].reshape(-1, 4),
        (targets - 1).clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    class_parts = [class_nll[labeled & (targets == class_id)].mean() for class_id in range(1, 5)]
    if any(not torch.isfinite(value) for value in class_parts):
        raise ValueError("当前 split 缺少至少一个 MI 类标签")
    class_loss = torch.stack(class_parts).mean()
    total = 0.5 * state_loss + 0.5 * class_loss
    return total, {"state_loss": state_loss, "class_loss": class_loss}


@dataclass(frozen=True)
class FullControlTrace:
    raw_token: np.ndarray
    normalized_token: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


# ---------- 在线回放：GRU 连续运行，二状态外壳只防止一次 MI 内重复输出 ----------
def full_control_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    model: FullControlGRU,
    normalizer: ContinuousNormalizer,
    commit_threshold: float,
    reset_threshold: float,
    device: torch.device,
) -> tuple[list[DecisionRecord], list[FullControlTrace]]:
    if not 0.0 < commit_threshold < 1.0 or not 0.0 < reset_threshold < 1.0:
        raise ValueError("提交与复位阈值必须位于 (0,1)")
    raw = build_continuous_tokens(windows, stage1_logits, stage2_logits)
    normalized = normalizer.transform(raw)
    decisions: list[DecisionRecord] = []
    traces: list[FullControlTrace] = []
    hidden = torch.zeros(1, 1, HIDDEN_DIM, device=device)
    state = READY
    current_key: tuple[int, int, int, int] | None = None
    model.eval()
    with torch.no_grad():
        for index, window in enumerate(windows):
            if window.key != current_key:
                current_key = window.key
                hidden.zero_()
                state = READY
            token = torch.from_numpy(normalized[index:index + 1]).to(device)
            hidden, logits = model.step(token, hidden)
            task_probability = torch.sigmoid(logits[:, :1])
            conditional_class = torch.softmax(logits[:, 1:], dim=-1)
            probabilities = torch.cat((
                1.0 - task_probability,
                task_probability * conditional_class,
            ), dim=-1)
            before = state
            emitted = NO_COMMAND
            if state == READY and float(1.0 - probabilities[0, 0]) >= commit_threshold:
                emitted = int(torch.argmax(probabilities[0, 1:]).item()) + 1
                state = WAIT_IDLE
            elif state == WAIT_IDLE and float(probabilities[0, 0]) >= reset_threshold:
                state = READY
            decisions.append(DecisionRecord(
                *window.key,
                window.window_index,
                window.window_start_sample,
                window.window_stop_sample,
                emitted,
                before,
                state,
                None,
            ))
            traces.append(FullControlTrace(
                raw[index].copy(),
                normalized[index].copy(),
                hidden[0, 0].detach().cpu().numpy().copy(),
                logits[0].detach().cpu().numpy().copy(),
                probabilities[0].detach().cpu().numpy().copy(),
            ))
    return decisions, traces
