"""在冻结 logits 上计算决策部件 Oracle 与 Stage 2 证据上限。

这里的 Oracle 会显式读取事件边界或类别真值，只用于回答“现有分数还有多少
可恢复空间”，绝不是可部署策略，也不能用于选择测试集工作点。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from logit_candidate_strategies import (
    LogitStrategyConfig,
    LogitStrategyResult,
    _Stage2Accumulator,
    logit_candidate_decisions,
)
from protocol_metrics import (
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    IDLE_RESET,
    MIN_OVERLAP_SECONDS,
    NATIVE_SAMPLING_RATE,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    _eligible,
)


MODEL = "model"
TRUTH = "truth_oracle"
ORACLE_AXIS_NAMES = ("stage1", "commit", "reset")


@dataclass(frozen=True)
class OracleCell:
    """三个二元决策轴组成的一个完整反事实轨迹。"""

    stage1_truth: bool
    commit_truth: bool
    reset_truth: bool

    @property
    def cell_id(self) -> str:
        source = lambda value: "truth" if value else "model"
        return (
            f"stage1_{source(self.stage1_truth)}__"
            f"commit_{source(self.commit_truth)}__"
            f"reset_{source(self.reset_truth)}"
        )

    @property
    def bits(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in (
            self.stage1_truth, self.commit_truth, self.reset_truth,
        ))

    def public_definition(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "stage1_source": (
                "truth_eligible_task_windows" if self.stage1_truth
                else "anchor_stage1_ewma"
            ),
            "commit_source": (
                "truth_selected_legal_stage2_crossing_with_raw_aggregate_top1_agreement"
                if self.commit_truth
                else "first_legal_stage2_crossing_with_raw_aggregate_top1_agreement"
            ),
            "reset_source": (
                "first_decision_time_after_true_mi_end" if self.reset_truth
                else "anchor_stage1_idle_threshold"
            ),
        }


ORACLE_CELLS = tuple(
    OracleCell(*bits) for bits in product((False, True), repeat=3)
)
MODEL_CELL_ID = OracleCell(False, False, False).cell_id
ALL_TRUTH_CELL_ID = OracleCell(True, True, True).cell_id


@dataclass(frozen=True)
class Crossing:
    """从候选打开后的历史严格复算出的当前 Stage 2 证据。"""

    legal: bool
    top_class: int
    top_probability: float
    probability_gap: float
    candidate_count: int


@dataclass(frozen=True)
class OracleReplay:
    """一个 cell 的逐窗轨迹及其 Oracle 优化审计信息。"""

    decisions: tuple[DecisionRecord, ...]
    selected_command_identities: tuple[tuple[int, int, int, int, int], ...]
    optimized_correct_count: int | None
    optimized_latency_samples: int | None


@dataclass(frozen=True)
class _Path:
    """动态规划路径；只保存稀疏提交点，避免逐窗复制完整轨迹。"""

    state: str
    candidate_start: int
    last_matched_event: int
    correct_count: int
    latency_samples: int
    commands: tuple[tuple[int, int, int], ...]  # global_window, class, event_index


def _validate_anchor(config: LogitStrategyConfig, reset_windows: int) -> None:
    """Oracle v1 只围绕已冻结的 c055/r020/l1 锚点，不接受静默换参。"""
    expected = {
        "stage1_filter": "ewma_margin",
        "stage1_alpha": 0.5,
        "task_on_probability": 0.5,
        "task_hold_probability": 0.3,
        "idle_reset_probability": 0.2,
        "stage1_drop_abort": 0.2,
        "stage2_filter": "candidate_ewma_centered_logits",
        "stage2_alpha": 0.5,
        "stage2_min_candidate_windows": 2,
        "stage2_top_probability": 0.55,
        "stage2_probability_gap": 0.15,
        "stage2_stable_windows": 1,
        "stage2_max_probability_curvature": None,
        "max_candidate_windows": 8,
    }
    if any(getattr(config, name) != value for name, value in expected.items()):
        raise ValueError("Oracle v1 必须使用冻结的 c055/r020/l1 logit 锚点")
    if reset_windows != 1:
        raise ValueError("Oracle v1 的模型复位轴固定为连续 1 窗")


def _validate_inputs(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    anchor: LogitStrategyResult,
) -> None:
    """拒绝乱序、跨被试或与预真值锚点错位的输入。"""
    if not windows or not events:
        raise ValueError("Oracle 需要非空窗口和事件库存")
    if len(anchor.policy.decisions) != len(windows) or len(anchor.trace) != len(windows):
        raise ValueError("预真值锚点轨迹与窗口数量不一致")
    if stage1_logits.shape != (len(windows), 2) or not np.isfinite(stage1_logits).all():
        raise ValueError("Stage 1 logits 必须逐窗对齐且为有限二维向量")
    if stage2_logits.shape != (len(windows), 4) or not np.isfinite(stage2_logits).all():
        raise ValueError("Stage 2 logits 必须逐窗对齐且为有限四维向量")
    identities = [(*window.key, window.window_index) for window in windows]
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("Oracle 窗口必须唯一并按流及 window_index 排序")
    if any(decision.identity != window.identity for decision, window in zip(
        anchor.policy.decisions, windows,
    )):
        raise ValueError("预真值锚点决策与窗口身份不一致")
    subjects = {window.subject_id for window in windows} | {event.subject_id for event in events}
    sessions = {window.session_id for window in windows} | {event.session_id for event in events}
    if len(subjects) != 1 or sessions != {0}:
        raise ValueError("Oracle 单次只接受一个被试的 session 0")
    event_identities = [(*event.key, event.event_id) for event in events]
    if len(set(event_identities)) != len(event_identities):
        raise ValueError("Oracle 事件身份不得重复")


def _event_maps(
    windows: Sequence[ExpectedWindow], events: Sequence[MIEvent],
) -> tuple[list[int | None], list[bool]]:
    """按正式 0.5 秒 margin 建立事件映射，并另建决策时刻真 IDLE 标记。"""
    margin = int(round(NATIVE_SAMPLING_RATE * MIN_OVERLAP_SECONDS))
    by_stream: dict[tuple[int, int, int, int], list[tuple[int, MIEvent]]] = {}
    for event_index, event in enumerate(events):
        by_stream.setdefault(event.key, []).append((event_index, event))

    eligible_event: list[int | None] = []
    true_idle_at_decision: list[bool] = []
    for window in windows:
        stream_events = by_stream.get(window.key, [])
        eligible = [
            index for index, event in stream_events
            if _eligible(window, event, margin)
        ]
        if len(eligible) > 1:
            raise RuntimeError("同一窗口同时匹配多个 MI 事件，Oracle 合同不再唯一")
        eligible_event.append(None if not eligible else eligible[0])

        # offset 时刻仍视作刚结束的 MI 决策点；真值复位只能从其后的窗口发生。
        inside_mi = any(
            event.onset_sample <= window.window_stop_sample <= event.offset_sample
            for _, event in stream_events
        )
        true_idle_at_decision.append(not inside_mi)
    return eligible_event, true_idle_at_decision


def _crossing(
    stage2_logits: np.ndarray,
    candidate_start: int,
    current: int,
    config: LogitStrategyConfig,
    cache: dict[tuple[int, int], Crossing],
) -> Crossing:
    """复用正式 Stage 2 累加器，精确保持开门窗不进入慢通道的语义。"""
    key = (candidate_start, current)
    if key in cache:
        return cache[key]
    if current <= candidate_start:
        raise ValueError("候选证据必须来自开门后的窗口")
    accumulator = _Stage2Accumulator(config)
    top_class, top_probability, gap = NO_COMMAND, -1.0, -1.0
    stable, curvature, count = 0, -1.0, 0
    for index in range(candidate_start + 1, current + 1):
        top_class, top_probability, gap, stable, curvature, count = accumulator.update(
            stage2_logits[index],
        )
    curvature_ok = (
        config.stage2_max_probability_curvature is None
        or (
            curvature >= 0.0
            and curvature <= config.stage2_max_probability_curvature
        )
    )
    legal = (
        count >= config.stage2_min_candidate_windows
        and top_probability >= config.stage2_top_probability
        and gap >= config.stage2_probability_gap
        and stable >= config.stage2_stable_windows
        and curvature_ok
    )
    result = Crossing(bool(legal), top_class, top_probability, gap, count)
    cache[key] = result
    return result


def _model_evidence(anchor: LogitStrategyResult, index: int, config: LogitStrategyConfig) -> tuple[bool, bool, bool]:
    """仅从预真值锚点保存的 Stage 1 因果轨迹重建开门、维持和复位证据。"""
    trace = anchor.trace[index]
    probability = trace.stage1_filtered_task_probability
    drop_abort = (
        config.stage1_drop_abort is not None
        and trace.stage1_filtered_delta <= -config.stage1_drop_abort
    )
    return (
        probability >= config.task_on_probability,
        probability >= config.task_hold_probability and not drop_abort,
        probability <= config.idle_reset_probability,
    )


def _transition_options(
    path: _Path,
    index: int,
    cell: OracleCell,
    events: Sequence[MIEvent],
    eligible_event: Sequence[int | None],
    true_idle: Sequence[bool],
    anchor: LogitStrategyResult,
    stage2_logits: np.ndarray,
    config: LogitStrategyConfig,
    cache: dict[tuple[int, int], Crossing],
) -> list[_Path]:
    """推进一个窗口；真值提交轴在合法正确 crossing 上保留“提交/跳过”两支。"""
    active_event = eligible_event[index]
    last_event = path.last_matched_event if path.last_matched_event == active_event else -1
    model_on, model_hold, model_reset = _model_evidence(anchor, index, config)
    task_on = active_event is not None if cell.stage1_truth else model_on
    task_hold = active_event is not None if cell.stage1_truth else model_hold
    idle_reset = true_idle[index] if cell.reset_truth else model_reset

    if path.state == READY:
        return [_Path(
            TASK_CANDIDATE if task_on else READY,
            index if task_on else -1,
            last_event,
            path.correct_count,
            path.latency_samples,
            path.commands,
        )]
    if path.state == WAIT_IDLE:
        return [_Path(
            READY if idle_reset else WAIT_IDLE,
            -1,
            last_event,
            path.correct_count,
            path.latency_samples,
            path.commands,
        )]

    if not task_hold:
        return [_Path(
            READY, -1, last_event, path.correct_count, path.latency_samples, path.commands,
        )]
    crossing = _crossing(stage2_logits, path.candidate_start, index, config, cache)
    age = index - path.candidate_start
    if not cell.commit_truth:
        if crossing.legal:
            return [_Path(
                WAIT_IDLE, -1, last_event,
                path.correct_count, path.latency_samples,
                (*path.commands, (index, crossing.top_class, -1)),
            )]
        return [_Path(
            READY if age >= config.max_candidate_windows else TASK_CANDIDATE,
            -1 if age >= config.max_candidate_windows else path.candidate_start,
            last_event,
            path.correct_count,
            path.latency_samples,
            path.commands,
        )]

    options: list[_Path] = []
    correct_crossing = (
        crossing.legal
        and active_event is not None
        and active_event != last_event
        and crossing.top_class == events[active_event].true_class
    )
    if correct_crossing:
        event = events[active_event]
        options.append(_Path(
            WAIT_IDLE, -1, active_event,
            path.correct_count + 1,
            path.latency_samples + (anchor.policy.decisions[index].window_stop_sample - event.onset_sample),
            (*path.commands, (index, crossing.top_class, active_event)),
        ))
    # Oracle 可以跳过错误 crossing，也可以跳过当前正确 crossing 以避免阻塞后续事件。
    options.append(_Path(
        READY if age >= config.max_candidate_windows else TASK_CANDIDATE,
        -1 if age >= config.max_candidate_windows else path.candidate_start,
        last_event,
        path.correct_count,
        path.latency_samples,
        path.commands,
    ))
    return options


def _better(left: _Path, right: _Path | None) -> bool:
    """字典序目标：正确事件最多、总延迟最少、提交更少、提交点更早。"""
    if right is None:
        return True
    left_key = (-left.correct_count, left.latency_samples, len(left.commands), left.commands)
    right_key = (-right.correct_count, right.latency_samples, len(right.commands), right.commands)
    return left_key < right_key


def _truth_selected_commands(
    segment_indices: Sequence[int],
    cell: OracleCell,
    events: Sequence[MIEvent],
    eligible_event: Sequence[int | None],
    true_idle: Sequence[bool],
    anchor: LogitStrategyResult,
    stage2_logits: np.ndarray,
    config: LogitStrategyConfig,
    cache: dict[tuple[int, int], Crossing],
) -> _Path:
    """在完整 segment 上做精确动态规划，不把每个事件独立贪心拼接。"""
    states: dict[tuple[str, int, int], _Path] = {
        (READY, -1, -1): _Path(READY, -1, -1, 0, 0, ()),
    }
    for index in segment_indices:
        next_states: dict[tuple[str, int, int], _Path] = {}
        for path in states.values():
            for candidate in _transition_options(
                path, index, cell, events, eligible_event, true_idle,
                anchor, stage2_logits, config, cache,
            ):
                key = (candidate.state, candidate.candidate_start, candidate.last_matched_event)
                if _better(candidate, next_states.get(key)):
                    next_states[key] = candidate
        states = next_states
    best: _Path | None = None
    for path in states.values():
        if _better(path, best):
            best = path
    if best is None:
        raise RuntimeError("Oracle 动态规划没有可达终态")
    return best


def _replay(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    eligible_event: Sequence[int | None],
    true_idle: Sequence[bool],
    anchor: LogitStrategyResult,
    stage2_logits: np.ndarray,
    config: LogitStrategyConfig,
    cell: OracleCell,
) -> OracleReplay:
    """先在各 segment 求最优提交点，再按同一规则重放并生成可交给正式评估器的轨迹。"""
    by_segment: dict[tuple[int, int, int, int], list[int]] = {}
    for index, window in enumerate(windows):
        by_segment.setdefault(window.key, []).append(index)
    crossing_cache: dict[tuple[int, int], Crossing] = {}
    selected: dict[int, tuple[int, int]] = {}
    optimized_correct = 0
    optimized_latency = 0
    if cell.commit_truth:
        for indices in by_segment.values():
            best = _truth_selected_commands(
                indices, cell, events, eligible_event, true_idle, anchor,
                stage2_logits, config, crossing_cache,
            )
            optimized_correct += best.correct_count
            optimized_latency += best.latency_samples
            for index, class_id, event_index in best.commands:
                if index in selected:
                    raise RuntimeError("Oracle 在同一窗口选择了两次命令")
                selected[index] = (class_id, event_index)

    decisions: list[DecisionRecord] = []
    consumed: set[int] = set()
    for indices in by_segment.values():
        state, candidate_start = READY, -1
        for index in indices:
            window = windows[index]
            before = state
            reason: str | None = None
            emitted = NO_COMMAND
            model_on, model_hold, model_reset = _model_evidence(anchor, index, config)
            task_on = eligible_event[index] is not None if cell.stage1_truth else model_on
            task_hold = eligible_event[index] is not None if cell.stage1_truth else model_hold
            idle_reset = true_idle[index] if cell.reset_truth else model_reset

            if state == READY:
                if task_on:
                    state, candidate_start, reason = TASK_CANDIDATE, index, CANDIDATE_OPEN
            elif state == WAIT_IDLE:
                if idle_reset:
                    state, reason = READY, IDLE_RESET
            else:
                age = index - candidate_start
                if not task_hold:
                    state, candidate_start, reason = READY, -1, CANDIDATE_ABORT_STAGE1
                else:
                    crossing = _crossing(
                        stage2_logits, candidate_start, index, config, crossing_cache,
                    )
                    should_commit = crossing.legal if not cell.commit_truth else index in selected
                    if should_commit:
                        if cell.commit_truth:
                            expected_class, event_index = selected[index]
                            if (
                                not crossing.legal
                                or crossing.top_class != expected_class
                                or eligible_event[index] != event_index
                                or events[event_index].true_class != expected_class
                            ):
                                raise RuntimeError("Oracle 选择点不再满足正确合法 crossing")
                            consumed.add(index)
                        emitted = crossing.top_class
                        state, candidate_start, reason = WAIT_IDLE, -1, COMMAND_COMMIT
                    elif age >= config.max_candidate_windows:
                        state, candidate_start, reason = READY, -1, CANDIDATE_TIMEOUT
            decisions.append(DecisionRecord(
                *window.key,
                window.window_index,
                window.window_start_sample,
                window.window_stop_sample,
                emitted,
                before,
                state,
                reason,
            ))
    if consumed != set(selected):
        raise RuntimeError("Oracle 动态规划选择点未被完整重放")
    command_identities = tuple(decisions[index].identity for index in sorted(selected))
    return OracleReplay(
        tuple(decisions),
        command_identities,
        optimized_correct if cell.commit_truth else None,
        optimized_latency if cell.commit_truth else None,
    )


def component_oracle_replays(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    config: LogitStrategyConfig,
    *,
    idle_reset_consecutive_windows: int = 1,
    pretruth_anchor: LogitStrategyResult | None = None,
) -> dict[str, OracleReplay]:
    """生成 2x2x2 完整轨迹；全模型 cell 必须逐窗等于预真值锚点。"""
    _validate_anchor(config, idle_reset_consecutive_windows)
    recomputed_anchor = logit_candidate_decisions(
        windows, stage1_logits, stage2_logits, config,
        idle_reset_consecutive_windows=idle_reset_consecutive_windows,
    )
    if pretruth_anchor is not None and (
        pretruth_anchor.policy != recomputed_anchor.policy
        or pretruth_anchor.trace != recomputed_anchor.trace
    ):
        raise RuntimeError("预真值锚点不是由当前传入的冻结 logits 逐窗生成")
    anchor = pretruth_anchor or recomputed_anchor
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    _validate_inputs(windows, events, stage1, stage2, anchor)
    eligible_event, true_idle = _event_maps(windows, events)

    results: dict[str, OracleReplay] = {}
    for cell in ORACLE_CELLS:
        replay = _replay(
            windows, events, eligible_event, true_idle, anchor, stage2, config, cell,
        )
        results[cell.cell_id] = replay
    if results[MODEL_CELL_ID].decisions != anchor.policy.decisions:
        raise RuntimeError("全模型 Oracle cell 没有逐窗复现 c055/r020/l1 锚点")
    return results


# ---------- 事件级 Stage 2 证据上限：不改变状态机，只回答分数中是否存在正确类别 ----------
def _top_class(logits: np.ndarray) -> int:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("top-1 输入必须是四维有限 logits")
    return int(np.argmax(values)) + 1


def _max_convex_margin(logits: np.ndarray, true_class: int) -> float:
    """线性规划最大化“真类 logit - 最强其他类 logit”的最小差。"""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        raise ValueError("凸组合 Oracle 需要非空有限四类 logits")
    if true_class not in (1, 2, 3, 4):
        raise ValueError("true_class 必须为 1..4")
    centered = values - np.mean(values, axis=1, keepdims=True)
    true_index = true_class - 1
    others = [index for index in range(4) if index != true_index]
    differences = centered[:, [true_index]] - centered[:, others]

    # 变量为每窗权重 w 与公共 margin gamma；约束 D^T w >= gamma。
    objective = np.zeros(len(values) + 1, dtype=np.float64)
    objective[-1] = -1.0
    inequalities = np.column_stack((-differences.T, np.ones(3, dtype=np.float64)))
    equality = np.zeros((1, len(values) + 1), dtype=np.float64)
    equality[0, :len(values)] = 1.0
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(3, dtype=np.float64),
        A_eq=equality,
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, 1.0)] * len(values) + [(None, None)],
        method="highs",
    )
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"凸组合 Oracle 线性规划失败: {result.message}")
    return float(result.x[-1])


def _latency_distribution(samples: list[int]) -> dict:
    if not samples:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    seconds = np.asarray(samples, dtype=np.float64) / NATIVE_SAMPLING_RATE
    return {
        "count": len(samples),
        "mean": float(np.mean(seconds)),
        "median": float(np.median(seconds)),
        "p90": float(np.quantile(seconds, 0.90)),
    }


def stage2_evidence_ceiling(
    windows: Sequence[ExpectedWindow],
    events: Sequence[MIEvent],
    stage2_logits: np.ndarray,
    baseline_matches: Sequence[dict],
    *,
    ewma_alpha: float = 0.5,
    convex_tolerance: float = 1e-9,
) -> dict:
    """计算 raw、任意因果 EWMA 起点和任意凸组合三层标签知情上限。"""
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if stage2.shape != (len(windows), 4) or not np.isfinite(stage2).all():
        raise ValueError("证据上限的 Stage 2 logits 与窗口不对齐")
    if not 0.0 < ewma_alpha <= 1.0 or convex_tolerance < 0.0:
        raise ValueError("EWMA alpha 或凸可行容差非法")
    match_map = {
        (
            item["subject_id"], item["session_id"], item["run_id"],
            item["segment_id"], item["event_id"],
        ): item
        for item in baseline_matches
    }
    onset_map = {(*event.key, event.event_id): event.onset_sample for event in events}
    margin = int(round(NATIVE_SAMPLING_RATE * MIN_OVERLAP_SECONDS))
    by_stream: dict[tuple[int, int, int, int], list[int]] = {}
    for index, window in enumerate(windows):
        by_stream.setdefault(window.key, []).append(index)

    rows: list[dict] = []
    for event in events:
        identity = (*event.key, event.event_id)
        if identity not in match_map:
            raise ValueError("基线事件匹配表缺少 Oracle 事件")
        eligible = [
            index for index in by_stream.get(event.key, [])
            if _eligible(windows[index], event, margin)
        ]
        if not eligible:
            raise RuntimeError("冻结库存事件在 Oracle 中变成不可评分")

        raw_sample: int | None = None
        for index in eligible:
            if _top_class(stage2[index]) == event.true_class:
                raw_sample = windows[index].window_stop_sample
                break

        # 允许候选在任一合法事件窗之前打开；每个起点仍只向前做 alpha=0.5 因果 EWMA。
        ewma_sample: int | None = None
        for start_position in range(len(eligible)):
            aggregate: np.ndarray | None = None
            for index in eligible[start_position:]:
                aggregate = (
                    stage2[index].copy() if aggregate is None
                    else ewma_alpha * stage2[index] + (1.0 - ewma_alpha) * aggregate
                )
                if _top_class(aggregate) == event.true_class:
                    sample = windows[index].window_stop_sample
                    ewma_sample = sample if ewma_sample is None else min(ewma_sample, sample)
                    break

        convex_sample: int | None = None
        full_margin: float | None = None
        for prefix_stop in range(1, len(eligible) + 1):
            prefix = stage2[eligible[:prefix_stop]]
            maximum_margin = _max_convex_margin(prefix, event.true_class)
            if prefix_stop == len(eligible):
                full_margin = maximum_margin
            if convex_sample is None and maximum_margin >= -convex_tolerance:
                convex_sample = windows[eligible[prefix_stop - 1]].window_stop_sample
        if full_margin is None:
            raise RuntimeError("凸组合 Oracle 未计算完整事件 margin")
        any_top1_sample = min(
            sample for sample in (raw_sample, ewma_sample) if sample is not None
        ) if raw_sample is not None or ewma_sample is not None else None
        if raw_sample is not None and convex_sample is None:
            raise RuntimeError("凸组合上限不应低于单窗 raw 上限")

        match = match_map[identity]
        rows.append({
            "event_id": event.event_id,
            "subject_id": event.subject_id,
            "session_id": event.session_id,
            "run_id": event.run_id,
            "segment_id": event.segment_id,
            "true_class": event.true_class,
            "eligible_window_count": len(eligible),
            "baseline_outcome": match["outcome"],
            "baseline_latency_seconds": match["latency_seconds"],
            "raw_top1_earliest_sample": raw_sample,
            "any_start_ewma_top1_earliest_sample": ewma_sample,
            "any_top1_earliest_sample": any_top1_sample,
            "convex_top1_earliest_sample": convex_sample,
            "convex_full_event_max_margin": full_margin,
            "raw_top1_latency_seconds": (
                None if raw_sample is None
                else (raw_sample - event.onset_sample) / NATIVE_SAMPLING_RATE
            ),
            "any_start_ewma_top1_latency_seconds": (
                None if ewma_sample is None
                else (ewma_sample - event.onset_sample) / NATIVE_SAMPLING_RATE
            ),
            "any_top1_latency_seconds": (
                None if any_top1_sample is None
                else (any_top1_sample - event.onset_sample) / NATIVE_SAMPLING_RATE
            ),
            "convex_top1_latency_seconds": (
                None if convex_sample is None
                else (convex_sample - event.onset_sample) / NATIVE_SAMPLING_RATE
            ),
        })

    baseline_correct = sum(row["baseline_outcome"] == "correct" for row in rows)
    baseline_miss = sum(row["baseline_outcome"] == "miss" for row in rows)
    baseline_wrong = sum(row["baseline_outcome"] == "wrong_class" for row in rows)

    def summarize(name: str, sample_field: str) -> dict:
        available = [row for row in rows if row[sample_field] is not None]
        miss_available = [row for row in available if row["baseline_outcome"] == "miss"]
        wrong_available = [row for row in available if row["baseline_outcome"] == "wrong_class"]
        latency_samples = [
            int(row[sample_field])
            - onset_map[(
                row["subject_id"], row["session_id"], row["run_id"],
                row["segment_id"], row["event_id"],
            )]
            for row in available
        ]
        return {
            "ceiling_id": name,
            "available_count": len(available),
            "coverage_rate": len(available) / len(rows),
            "gain_over_baseline_correct_event_rate": (len(available) - baseline_correct) / len(rows),
            "recoverable_baseline_miss_count": len(miss_available),
            "recoverable_baseline_miss_rate": (
                None if baseline_miss == 0 else len(miss_available) / baseline_miss
            ),
            "recoverable_baseline_wrong_count": len(wrong_available),
            "recoverable_baseline_wrong_rate": (
                None if baseline_wrong == 0 else len(wrong_available) / baseline_wrong
            ),
            "earliest_latency_seconds": _latency_distribution(latency_samples),
        }

    summary = {
        "event_count": len(rows),
        "baseline_correct_count": baseline_correct,
        "baseline_correct_event_rate": baseline_correct / len(rows),
        "baseline_miss_count": baseline_miss,
        "baseline_wrong_class_count": baseline_wrong,
        "raw_top1": summarize("raw_top1", "raw_top1_earliest_sample"),
        "any_start_causal_ewma_top1": summarize(
            "any_start_causal_ewma_top1", "any_start_ewma_top1_earliest_sample",
        ),
        "raw_or_any_start_ewma_top1": summarize(
            "raw_or_any_start_ewma_top1", "any_top1_earliest_sample",
        ),
        "convex_logit_top1": summarize("convex_logit_top1", "convex_top1_earliest_sample"),
        "convex_only_beyond_any_top1_count": sum(
            row["convex_top1_earliest_sample"] is not None
            and row["any_top1_earliest_sample"] is None
            for row in rows
        ),
        "truth_usage": "event_boundary_and_true_class_aware_post_inference_ceiling_only",
        "far_interpretation": "not_defined_as_deployable_far",
    }
    return {"summary": summary, "event_rows": rows}


def shapley_component_contributions(values_by_cell: dict[str, float]) -> dict:
    """用完整 2x2x2 矩阵公平分摊交互增益；三项之和等于全真值减全模型。"""
    by_bits = {
        cell.bits: float(values_by_cell[cell.cell_id])
        for cell in ORACLE_CELLS
    }
    if set(values_by_cell) != {cell.cell_id for cell in ORACLE_CELLS}:
        raise ValueError("Shapley 分解必须提供完整且仅含 8 个 Oracle cell")
    contributions: dict[str, float] = {}
    n = len(ORACLE_AXIS_NAMES)
    for axis, axis_name in enumerate(ORACLE_AXIS_NAMES):
        contribution = 0.0
        for bits in product((0, 1), repeat=n):
            if bits[axis]:
                continue
            subset_size = sum(bits)
            weight = factorial(subset_size) * factorial(n - subset_size - 1) / factorial(n)
            enabled = list(bits)
            enabled[axis] = 1
            contribution += weight * (by_bits[tuple(enabled)] - by_bits[bits])
        contributions[axis_name] = float(contribution)
    total_gap = by_bits[(1, 1, 1)] - by_bits[(0, 0, 0)]
    if not np.isclose(sum(contributions.values()), total_gap, rtol=0.0, atol=1e-12):
        raise RuntimeError("Shapley 分量没有严格加回 Oracle 总增益")
    return {
        "baseline_value": by_bits[(0, 0, 0)],
        "all_truth_value": by_bits[(1, 1, 1)],
        "total_oracle_gap": float(total_gap),
        "contributions": contributions,
        "interpretation": "descriptive_factorial_attribution_with_interactions_shared_by_shapley",
    }
