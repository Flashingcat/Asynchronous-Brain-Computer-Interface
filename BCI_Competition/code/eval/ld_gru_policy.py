"""LD-GRU-v1：只在固定 Stage 1 候选态内学习提交时机与可选类别修正。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from candidate_state_policy import CandidateEvidence, candidate_state_decisions, candidate_transition
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    IDLE_RESET,
    LEARNED_GRU_COMMIT,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    classify_counterfactual_first_commit,
)


TOKEN_DIM = 12
STANDARDIZED_DIM = 11
HIDDEN_DIM = 8
MAX_AFTER_OPEN_WINDOWS = 8
MAX_SEQUENCE_TOKENS = 1 + MAX_AFTER_OPEN_WINDOWS
ABLATIONS = ("stop_only", "stop_residual")
TOKEN_MODES = ("full", "mask_stage1")


# ---------- 候选流输入：所有量只由当前及更早窗口计算，segment 边界清空历史 ----------
@dataclass(frozen=True)
class FlowInputs:
    task_margin: np.ndarray
    task_probability: np.ndarray
    task_probability_delta: np.ndarray
    centered_stage2: np.ndarray
    centered_stage2_delta: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.task_margin)
        expected = ((count,), (count,), (count,), (count, 4), (count, 4))
        actual = tuple(
            np.asarray(value).shape
            for value in (
                self.task_margin,
                self.task_probability,
                self.task_probability_delta,
                self.centered_stage2,
                self.centered_stage2_delta,
            )
        )
        if actual != expected or not all(
            np.isfinite(np.asarray(value)).all()
            for value in (
                self.task_margin,
                self.task_probability,
                self.task_probability_delta,
                self.centered_stage2,
                self.centered_stage2_delta,
            )
        ):
            raise ValueError("LD-GRU 流输入形状错误或含非有限值")


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))


def _center_stage2(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Stage 2 单窗 logits 必须是四维有限向量")
    centered = value - float(np.mean(value))
    if not np.isfinite(centered).all():
        offsets = value - value[0]
        centered = offsets - float(np.mean(offsets))
    if not np.isfinite(centered).all():
        raise ValueError("Stage 2 中心化 logits 溢出")
    return centered


def build_flow_inputs(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    *,
    stage1_alpha: float = 0.5,
) -> FlowInputs:
    """生成 11 维非年龄特征；EWMA 和一阶差分均不跨 segment。"""
    if not 0.0 < stage1_alpha <= 1.0:
        raise ValueError("stage1_alpha 必须位于 (0,1]")
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if (
        stage1.shape != (len(windows), 2)
        or stage2.shape != (len(windows), 4)
        or not np.isfinite(stage1).all()
        or not np.isfinite(stage2).all()
    ):
        raise ValueError("Stage 1/2 logits 必须逐窗对齐且有限")
    neutral = CandidateEvidence(False, True, NO_COMMAND, False)
    candidate_state_decisions(
        windows, [neutral] * len(windows), max_candidate_windows=MAX_AFTER_OPEN_WINDOWS,
    )

    margin = np.empty(len(windows), dtype=np.float32)
    probability = np.empty(len(windows), dtype=np.float32)
    probability_delta = np.empty(len(windows), dtype=np.float32)
    centered = np.empty((len(windows), 4), dtype=np.float32)
    centered_delta = np.empty((len(windows), 4), dtype=np.float32)
    previous_key: tuple[int, int, int, int] | None = None
    ewma_margin = 0.0
    previous_probability = 0.0
    previous_centered = np.zeros(4, dtype=np.float64)
    for index, window in enumerate(windows):
        current_margin = float(stage1[index, 1] - stage1[index, 0])
        current_centered = _center_stage2(stage2[index])
        if window.key != previous_key:
            ewma_margin = current_margin
            current_probability = _sigmoid(ewma_margin)
            probability_change = 0.0
            center_change = np.zeros(4, dtype=np.float64)
            previous_key = window.key
        else:
            ewma_margin = stage1_alpha * current_margin + (1.0 - stage1_alpha) * ewma_margin
            current_probability = _sigmoid(ewma_margin)
            probability_change = current_probability - previous_probability
            center_change = current_centered - previous_centered
        margin[index] = current_margin
        probability[index] = current_probability
        probability_delta[index] = probability_change
        centered[index] = current_centered
        centered_delta[index] = center_change
        previous_probability = current_probability
        previous_centered = current_centered
    return FlowInputs(margin, probability, probability_delta, centered, centered_delta)


def token_at(flow: FlowInputs, index: int, age: int) -> np.ndarray:
    """把固定流特征与系统自身候选年龄拼成一个 12 维 token。"""
    if type(index) is not int or not 0 <= index < len(flow.task_margin):
        raise IndexError("token 窗口下标越界")
    if type(age) is not int or not 0 <= age <= MAX_AFTER_OPEN_WINDOWS:
        raise ValueError("候选年龄必须是 0..8 的整数")
    return np.asarray([
        flow.task_margin[index],
        flow.task_probability[index],
        flow.task_probability_delta[index],
        *flow.centered_stage2[index].tolist(),
        *flow.centered_stage2_delta[index].tolist(),
        age / MAX_AFTER_OPEN_WINDOWS,
    ], dtype=np.float32)


# ---------- 训练候选：开门窗进入序列但正确掩码恒为 0，最早下一窗才能提交 ----------
@dataclass(frozen=True)
class CandidateSequence:
    candidate_id: str
    subject_id: int
    window_positions: np.ndarray
    tokens: np.ndarray
    centered_stage2: np.ndarray
    correct_mask: np.ndarray
    exit_reason: str

    def __post_init__(self) -> None:
        length = len(self.window_positions)
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or type(self.subject_id) is not int
            or not 1 <= length <= MAX_SEQUENCE_TOKENS
            or np.asarray(self.window_positions).shape != (length,)
            or np.asarray(self.tokens).shape != (length, TOKEN_DIM)
            or np.asarray(self.centered_stage2).shape != (length, 4)
            or np.asarray(self.correct_mask).shape != (length, 4)
            or np.asarray(self.correct_mask).dtype != np.bool_
            or not np.isfinite(self.tokens).all()
            or not np.isfinite(self.centered_stage2).all()
            or np.any(self.correct_mask[0])
            or self.exit_reason not in {
                CANDIDATE_ABORT_STAGE1, CANDIDATE_TIMEOUT, "segment_end_unresolved",
            }
        ):
            raise ValueError("候选序列合同非法")

    @property
    def positive(self) -> bool:
        return bool(np.any(self.correct_mask))


@dataclass(frozen=True)
class CandidateInventory:
    candidates: tuple[CandidateSequence, ...]
    open_count: int
    abort_count: int
    timeout_count: int
    unresolved_count: int

    @property
    def trainable_candidates(self) -> tuple[CandidateSequence, ...]:
        # 单 token 候选只有被强制为 q0=0 的开门窗，其损失恒为零，不进入梯度批次。
        return tuple(item for item in self.candidates if len(item.window_positions) >= 2)


def build_candidate_inventory(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    flow: FlowInputs,
    *,
    task_on_probability: float = 0.5,
    task_hold_probability: float = 0.3,
    drop_abort: float = 0.2,
    max_after_open_windows: int = MAX_AFTER_OPEN_WINDOWS,
    margin_samples: int = 125,
) -> CandidateInventory:
    """关闭 Stage 2 提交，用固定 Stage 1 在完整正式窗口流上切出候选序列。"""
    if len(windows) != len(flow.task_margin):
        raise ValueError("flow 与窗口数量不一致")
    if max_after_open_windows != MAX_AFTER_OPEN_WINDOWS:
        raise ValueError("LD-GRU-v1 固定开门后最多 8 窗")
    candidates: list[CandidateSequence] = []
    events_by_key: dict[tuple[int, int, int, int], tuple[MIEvent, ...]] = {}
    for event in events:
        events_by_key.setdefault(event.key, tuple())
        events_by_key[event.key] = (*events_by_key[event.key], event)
    current_positions: list[int] = []
    current_key: tuple[int, int, int, int] | None = None
    age = 0
    state = READY

    def finish(reason: str) -> None:
        nonlocal current_positions
        if not current_positions:
            raise RuntimeError("候选结束时缺少开门窗口")
        token_rows = np.stack([
            token_at(flow, position, local_age)
            for local_age, position in enumerate(current_positions)
        ])
        mask = np.zeros((len(current_positions), 4), dtype=np.bool_)
        for local_index, position in enumerate(current_positions[1:], start=1):
            for class_id in range(1, 5):
                result = classify_counterfactual_first_commit(
                    windows[position],
                    class_id,
                    events_by_key.get(windows[position].key, tuple()),
                    margin_samples=margin_samples,
                )
                mask[local_index, class_id - 1] = result.outcome == "correct"
        opening = windows[current_positions[0]]
        identifier = (
            f"s{opening.subject_id:02d}_sess{opening.session_id}_r{opening.run_id}_"
            f"seg{opening.segment_id}_w{opening.window_index}"
        )
        candidates.append(CandidateSequence(
            identifier,
            opening.subject_id,
            np.asarray(current_positions, dtype=np.int64),
            token_rows,
            flow.centered_stage2[current_positions].copy(),
            mask,
            reason,
        ))
        current_positions = []

    for position, window in enumerate(windows):
        if window.key != current_key:
            if state == TASK_CANDIDATE:
                finish("segment_end_unresolved")
            current_key, state, age = window.key, READY, 0
        if state == READY:
            if flow.task_probability[position] >= task_on_probability:
                state, age = TASK_CANDIDATE, 0
                current_positions = [position]
            continue

        drop = flow.task_probability_delta[position] <= -drop_abort
        hold = flow.task_probability[position] >= task_hold_probability and not drop
        if not hold:
            finish(CANDIDATE_ABORT_STAGE1)
            state, age = READY, 0
            continue
        age += 1
        current_positions.append(position)
        if age >= max_after_open_windows:
            finish(CANDIDATE_TIMEOUT)
            state, age = READY, 0
    if state == TASK_CANDIDATE:
        finish("segment_end_unresolved")

    reasons = [item.exit_reason for item in candidates]
    return CandidateInventory(
        tuple(candidates),
        len(candidates),
        reasons.count(CANDIDATE_ABORT_STAGE1),
        reasons.count(CANDIDATE_TIMEOUT),
        reasons.count("segment_end_unresolved"),
    )


# ---------- 训练标准化：只拟合当前训练被试，年龄列保持 0..1 原值 ----------
@dataclass(frozen=True)
class TokenNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if (
            np.asarray(self.mean).shape != (STANDARDIZED_DIM,)
            or np.asarray(self.std).shape != (STANDARDIZED_DIM,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("token 标准化参数非法")

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        value = np.asarray(tokens, dtype=np.float32).copy()
        if value.ndim != 2 or value.shape[1] != TOKEN_DIM or not np.isfinite(value).all():
            raise ValueError("待标准化 token 必须为 [T,12] 有限数组")
        value[:, :STANDARDIZED_DIM] = (
            value[:, :STANDARDIZED_DIM] - self.mean
        ) / self.std
        return value


def fit_token_normalizer(candidates: Iterable[CandidateSequence]) -> TokenNormalizer:
    rows = [item.tokens[:, :STANDARDIZED_DIM] for item in candidates]
    if not rows:
        raise ValueError("至少需要一个训练候选拟合 token 标准化")
    values = np.concatenate(rows, axis=0).astype(np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    # 常量特征不携带判别信息，置 1 避免除零；阈值固定而非依赖测试数据。
    std = np.where(std < 1e-6, 1.0, std)
    return TokenNormalizer(mean.astype(np.float32), std.astype(np.float32))


# ---------- 573 参数 Tiny GRU：stop_only 仍保留零修正头，但不训练也不使用 ----------
class TinyLDGRU(nn.Module):
    def __init__(self, ablation: str, token_mode: str = "full") -> None:
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"ablation 必须取 {ABLATIONS}")
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode 必须取 {TOKEN_MODES}")
        self.ablation = ablation
        self.token_mode = token_mode
        self.gru = nn.GRUCell(TOKEN_DIM, HIDDEN_DIM)
        self.stop_head = nn.Linear(HIDDEN_DIM, 1)
        self.class_correction = nn.Linear(HIDDEN_DIM, 4)
        nn.init.constant_(self.stop_head.bias, -3.0)
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)
        if ablation == "stop_only":
            for parameter in self.class_correction.parameters():
                parameter.requires_grad_(False)

    def _visible_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """屏蔽标准化后的前三维，使 0 精确表示训练被试均值且不改变其余证据。"""
        if self.token_mode == "full":
            return tokens
        return torch.cat((torch.zeros_like(tokens[..., :3]), tokens[..., 3:]), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        centered_stage2: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            tokens.ndim != 3
            or tokens.shape[2] != TOKEN_DIM
            or centered_stage2.shape != (*tokens.shape[:2], 4)
            or valid_mask.shape != tokens.shape[:2]
        ):
            raise ValueError("TinyLDGRU batch 形状非法")
        model_tokens = self._visible_tokens(tokens)
        hidden = torch.zeros(tokens.shape[0], HIDDEN_DIM, device=tokens.device, dtype=tokens.dtype)
        hidden_rows: list[torch.Tensor] = []
        for step in range(tokens.shape[1]):
            updated = self.gru(model_tokens[:, step], hidden)
            hidden = torch.where(valid_mask[:, step, None], updated, hidden)
            hidden_rows.append(hidden)
        hidden_sequence = torch.stack(hidden_rows, dim=1)
        stop_logits = self.stop_head(hidden_sequence).squeeze(-1)
        if self.ablation == "stop_residual":
            correction = self.class_correction(hidden_sequence)
        else:
            correction = torch.zeros_like(centered_stage2)
        return hidden_sequence, stop_logits, correction, centered_stage2 + correction

    def step(
        self,
        token: torch.Tensor,
        centered_stage2: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单窗部署接口；调用方负责候选边界处的 hidden 清零。"""
        next_hidden = self.gru(self._visible_tokens(token), hidden)
        stop_logit = self.stop_head(next_hidden).squeeze(-1)
        correction = (
            self.class_correction(next_hidden)
            if self.ablation == "stop_residual"
            else torch.zeros_like(centered_stage2)
        )
        return next_hidden, stop_logit, correction, centered_stage2 + correction


def model_parameter_counts(model: TinyLDGRU) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


# ---------- Padding 与集合式 hazard 损失：正负候选分别求均值后等权 ----------
@dataclass(frozen=True)
class CandidateTensorSet:
    normalized_tokens: np.ndarray
    centered_stage2: np.ndarray
    valid_mask: np.ndarray
    correct_mask: np.ndarray
    positive_mask: np.ndarray
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.candidate_ids)
        if (
            self.normalized_tokens.shape != (count, MAX_SEQUENCE_TOKENS, TOKEN_DIM)
            or self.centered_stage2.shape != (count, MAX_SEQUENCE_TOKENS, 4)
            or self.valid_mask.shape != (count, MAX_SEQUENCE_TOKENS)
            or self.correct_mask.shape != (count, MAX_SEQUENCE_TOKENS, 4)
            or self.positive_mask.shape != (count,)
            or self.valid_mask.dtype != np.bool_
            or self.correct_mask.dtype != np.bool_
            or self.positive_mask.dtype != np.bool_
        ):
            raise ValueError("候选 TensorSet 结构非法")


def tensorize_candidates(
    candidates: Iterable[CandidateSequence],
    normalizer: TokenNormalizer,
    *,
    trainable_only: bool = True,
) -> CandidateTensorSet:
    selected = [
        item for item in candidates
        if not trainable_only or len(item.window_positions) >= 2
    ]
    if not selected:
        raise ValueError("没有可张量化的候选序列")
    count = len(selected)
    tokens = np.zeros((count, MAX_SEQUENCE_TOKENS, TOKEN_DIM), dtype=np.float32)
    centered = np.zeros((count, MAX_SEQUENCE_TOKENS, 4), dtype=np.float32)
    valid = np.zeros((count, MAX_SEQUENCE_TOKENS), dtype=np.bool_)
    correct = np.zeros((count, MAX_SEQUENCE_TOKENS, 4), dtype=np.bool_)
    for row, candidate in enumerate(selected):
        length = len(candidate.window_positions)
        tokens[row, :length] = normalizer.transform(candidate.tokens)
        centered[row, :length] = candidate.centered_stage2
        valid[row, :length] = True
        correct[row, :length] = candidate.correct_mask
    positive = np.any(correct, axis=(1, 2))
    return CandidateTensorSet(
        tokens, centered, valid, correct, positive,
        tuple(item.candidate_id for item in selected),
    )


def set_valued_commit_loss(
    stop_logits: torch.Tensor,
    class_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    correct_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """稳定计算首次正确提交概率与整段不提交概率，不指定唯一目标窗口。"""
    if (
        stop_logits.ndim != 2
        or class_logits.shape != (*stop_logits.shape, 4)
        or valid_mask.shape != stop_logits.shape
        or correct_mask.shape != class_logits.shape
    ):
        raise ValueError("集合式损失输入形状非法")
    committable = valid_mask.clone()
    committable[:, 0] = False
    effective_correct = correct_mask & committable[:, :, None]
    positive = torch.any(effective_correct.flatten(1), dim=1)
    if not torch.any(positive) or torch.all(positive):
        raise ValueError("一个损失批次必须同时含正候选和负候选")

    log_wait = torch.where(committable, F.logsigmoid(-stop_logits), torch.zeros_like(stop_logits))
    log_submit = torch.where(
        committable,
        F.logsigmoid(stop_logits),
        torch.full_like(stop_logits, -torch.inf),
    )
    survival_before = torch.cumsum(log_wait, dim=1) - log_wait
    log_first_class = survival_before[:, :, None] + log_submit[:, :, None]
    log_first_class = log_first_class + F.log_softmax(class_logits, dim=-1)
    masked = torch.where(
        effective_correct,
        log_first_class,
        torch.full_like(log_first_class, -torch.inf),
    )
    positive_loss = -torch.logsumexp(masked[positive].flatten(1), dim=1)
    negative_loss = -torch.sum(log_wait[~positive], dim=1)
    total = 0.5 * positive_loss.mean() + 0.5 * negative_loss.mean()
    if not torch.isfinite(total):
        raise FloatingPointError("集合式候选损失出现非有限值")
    return total, {
        "positive_mean": positive_loss.mean(),
        "negative_mean": negative_loss.mean(),
        "positive_count": int(positive.sum().item()),
        "negative_count": int((~positive).sum().item()),
    }


def balanced_batch_indices(
    dataset: CandidateTensorSet,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """每批正负各半；少数类循环重排，避免 IDLE 候选数量压倒正候选。"""
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size 必须是至少 2 的偶数")
    positive = np.flatnonzero(dataset.positive_mask)
    negative = np.flatnonzero(~dataset.positive_mask)
    if not len(positive) or not len(negative):
        raise ValueError("训练集必须同时含正候选和负候选")
    half = batch_size // 2
    batch_count = math.ceil(max(len(positive), len(negative)) / half)

    def draw(pool: np.ndarray, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < count:
            pieces.append(rng.permutation(pool))
        return np.concatenate(pieces)[:count]

    positive_draw = draw(positive, batch_count * half)
    negative_draw = draw(negative, batch_count * half)
    batches = []
    for batch in range(batch_count):
        indices = np.concatenate([
            positive_draw[batch * half:(batch + 1) * half],
            negative_draw[batch * half:(batch + 1) * half],
        ])
        batches.append(rng.permutation(indices))
    return tuple(batches)


# ---------- 正式在线回放：模型不读真值，所有退出路径都清空候选局部 hidden ----------
@dataclass(frozen=True)
class LDGRUTrace:
    raw_token: np.ndarray
    normalized_token: np.ndarray
    hidden: np.ndarray
    stop_logit: float
    stop_score: float
    centered_stage2: np.ndarray
    class_correction: np.ndarray
    final_class_logits: np.ndarray
    gru_consumed: bool
    candidate_age: int


@dataclass(frozen=True)
class LDGRUReplayResult:
    decisions: tuple[DecisionRecord, ...]
    trace: tuple[LDGRUTrace, ...]
    flow: FlowInputs


@torch.inference_mode()
def ld_gru_decisions(
    windows: Sequence[ExpectedWindow],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    model: TinyLDGRU,
    normalizer: TokenNormalizer,
    stop_threshold: float,
    device: torch.device,
    *,
    stage1_alpha: float = 0.5,
    task_on_probability: float = 0.5,
    task_hold_probability: float = 0.3,
    idle_reset_probability: float = 0.2,
    drop_abort: float = 0.2,
    max_after_open_windows: int = MAX_AFTER_OPEN_WINDOWS,
) -> LDGRUReplayResult:
    """按正式窗口顺序运行 LD-GRU；开门窗 q 强制为 0，不存在 Fast-0。"""
    if not isinstance(model, TinyLDGRU):
        raise TypeError("model 必须为 TinyLDGRU")
    if not 0.0 < stop_threshold < 1.0:
        raise ValueError("stop_threshold 必须位于 (0,1)")
    if max_after_open_windows != MAX_AFTER_OPEN_WINDOWS:
        raise ValueError("LD-GRU-v1 固定开门后最多 8 窗")
    flow = build_flow_inputs(
        windows, stage1_logits, stage2_logits, stage1_alpha=stage1_alpha,
    )
    model = model.to(device).eval()
    decisions: list[DecisionRecord] = []
    traces: list[LDGRUTrace] = []
    state, age = READY, 0
    current_key: tuple[int, int, int, int] | None = None
    hidden = torch.zeros(1, HIDDEN_DIM, dtype=torch.float32, device=device)

    def inactive_trace(position: int) -> LDGRUTrace:
        """未消费 GRU 的窗口仍保存全部基础分数；年龄 -1 是显式非候选哨兵。"""
        raw = np.asarray([
            flow.task_margin[position],
            flow.task_probability[position],
            flow.task_probability_delta[position],
            *flow.centered_stage2[position].tolist(),
            *flow.centered_stage2_delta[position].tolist(),
            -1.0,
        ], dtype=np.float32)
        normalized = normalizer.transform(raw[None])[0]
        centered = flow.centered_stage2[position].copy()
        return LDGRUTrace(
            raw,
            normalized,
            np.zeros(HIDDEN_DIM, dtype=np.float32),
            0.0,
            0.0,
            centered,
            np.zeros(4, dtype=np.float32),
            centered.copy(),
            False,
            -1,
        )

    for position, window in enumerate(windows):
        if window.key != current_key:
            current_key, state, age = window.key, READY, 0
            hidden.zero_()
        before = state
        evidence = CandidateEvidence(False, True, NO_COMMAND, False)
        trace_row = inactive_trace(position)

        if state == READY:
            task_on = bool(flow.task_probability[position] >= task_on_probability)
            evidence = CandidateEvidence(task_on, True, NO_COMMAND, False)
            transition = candidate_transition(
                state, age, evidence, max_candidate_windows=max_after_open_windows,
            )
            if transition.transition_reason == CANDIDATE_OPEN:
                raw = token_at(flow, position, 0)
                normalized = normalizer.transform(raw[None])[0]
                raw_tensor = torch.from_numpy(normalized[None]).to(device)
                center_tensor = torch.from_numpy(flow.centered_stage2[position][None]).to(device)
                hidden, stop_logit, correction, final = model.step(
                    raw_tensor, center_tensor, hidden,
                )
                trace_row = LDGRUTrace(
                    raw, normalized, hidden[0].cpu().numpy().copy(),
                    float(stop_logit.item()), 0.0,
                    flow.centered_stage2[position].copy(),
                    correction[0].cpu().numpy().copy(), final[0].cpu().numpy().copy(),
                    True, 0,
                )
        elif state == TASK_CANDIDATE:
            drop = bool(flow.task_probability_delta[position] <= -drop_abort)
            hold = bool(flow.task_probability[position] >= task_hold_probability and not drop)
            commit_class = NO_COMMAND
            if hold:
                next_age = age + 1
                raw = token_at(flow, position, next_age)
                normalized = normalizer.transform(raw[None])[0]
                raw_tensor = torch.from_numpy(normalized[None]).to(device)
                center_tensor = torch.from_numpy(flow.centered_stage2[position][None]).to(device)
                hidden, stop_logit, correction, final = model.step(
                    raw_tensor, center_tensor, hidden,
                )
                score = float(torch.sigmoid(stop_logit).item())
                if score >= stop_threshold:
                    commit_class = int(torch.argmax(final, dim=-1).item()) + 1
                trace_row = LDGRUTrace(
                    raw, normalized, hidden[0].cpu().numpy().copy(),
                    float(stop_logit.item()), score,
                    flow.centered_stage2[position].copy(),
                    correction[0].cpu().numpy().copy(), final[0].cpu().numpy().copy(),
                    True, next_age,
                )
            evidence = CandidateEvidence(False, hold, commit_class, False)
            transition = candidate_transition(
                state, age, evidence, max_candidate_windows=max_after_open_windows,
            )
        else:
            idle_reset = bool(flow.task_probability[position] <= idle_reset_probability)
            evidence = CandidateEvidence(False, True, NO_COMMAND, idle_reset)
            transition = candidate_transition(
                state, age, evidence, max_candidate_windows=max_after_open_windows,
            )

        reason = (
            LEARNED_GRU_COMMIT
            if transition.transition_reason == COMMAND_COMMIT
            else transition.transition_reason
        )
        decision = DecisionRecord(
            *window.key,
            window.window_index,
            window.window_start_sample,
            window.window_stop_sample,
            transition.emitted_class,
            before,
            transition.state_after,
            reason,
        )
        decisions.append(decision)
        traces.append(trace_row)
        state, age = transition.state_after, transition.candidate_windows_after
        if reason in {
            LEARNED_GRU_COMMIT, CANDIDATE_ABORT_STAGE1, CANDIDATE_TIMEOUT,
        }:
            hidden.zero_()
        if reason == IDLE_RESET:
            hidden.zero_()
    return LDGRUReplayResult(tuple(decisions), tuple(traces), flow)
