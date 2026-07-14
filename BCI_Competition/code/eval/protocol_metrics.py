"""与模型和决策策略解耦的窗口级、事件级基础评估器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


MI_CLASS_COUNT = 4
NO_COMMAND = -1
NATIVE_SAMPLING_RATE = 250.0
WINDOW_SAMPLES = 500
STEP_SAMPLES = 125
MIN_OVERLAP_SECONDS = 0.5
READY = "READY"
TASK_CANDIDATE = "TASK_CANDIDATE"
WAIT_IDLE = "WAIT_IDLE"
STATEFUL_STRICT = "stateful_strict"
STATEFUL_CANDIDATE = "stateful_candidate"
STATELESS_DIAGNOSTIC = "stateless_diagnostic"
CANDIDATE_OPEN = "candidate_open"
CANDIDATE_ABORT_STAGE1 = "candidate_abort_stage1"
CANDIDATE_TIMEOUT = "candidate_timeout"
COMMAND_COMMIT = "command_commit"
FAST0_COMMAND_COMMIT = "fast0_command_commit"
FAST1_COMMAND_COMMIT = "fast1_command_commit"
IDLE_RESET = "idle_reset"
STAGE1_CLASS_NAMES = ("idle", "task")
STAGE2_CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
FINAL_CLASS_NAMES = ("idle", *STAGE2_CLASS_NAMES)


def _python_int(value: int, field_name: str) -> int:
    """把冻结索引中的 NumPy 整数规范为可 JSON 序列化的 Python int。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} 必须为整数")
    return int(value)


def _finite_float(value: float, field_name: str) -> float:
    """把数值配置规范为 Python float，并拒绝布尔值、字符串和非有限数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{field_name} 必须为有限数")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} 必须为有限数") from error
    if not np.isfinite(result):
        raise ValueError(f"{field_name} 必须为有限数")
    return result


# 数据类只保存评分必需的原生采样坐标，不允许评估器回读 EEG 或模型状态。
@dataclass(frozen=True)
class ScoringSegment:
    subject_id: int
    session_id: int
    run_id: int
    segment_id: int
    start_sample: int
    stop_sample: int

    def __post_init__(self) -> None:
        for name in ("subject_id", "session_id", "run_id", "segment_id", "start_sample", "stop_sample"):
            object.__setattr__(self, name, _python_int(getattr(self, name), name))
        if self.subject_id < 1:
            raise ValueError("subject_id 必须从 1 开始")
        if min(self.session_id, self.run_id, self.segment_id, self.start_sample, self.stop_sample) < 0:
            raise ValueError("session/run/segment 和原生采样坐标不得为负")

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.subject_id, self.session_id, self.run_id, self.segment_id


@dataclass(frozen=True)
class MIEvent:
    event_id: str
    subject_id: int
    session_id: int
    run_id: int
    segment_id: int
    onset_sample: int
    offset_sample: int
    true_class: int

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise TypeError("event_id 必须为非空字符串")
        for name in (
            "subject_id", "session_id", "run_id", "segment_id",
            "onset_sample", "offset_sample", "true_class",
        ):
            object.__setattr__(self, name, _python_int(getattr(self, name), name))
        if self.subject_id < 1:
            raise ValueError("subject_id 必须从 1 开始")
        if min(
            self.session_id, self.run_id, self.segment_id,
            self.onset_sample, self.offset_sample,
        ) < 0:
            raise ValueError("session/run/segment 和原生采样坐标不得为负")

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.subject_id, self.session_id, self.run_id, self.segment_id


@dataclass(frozen=True)
class DecisionRecord:
    subject_id: int
    session_id: int
    run_id: int
    segment_id: int
    window_index: int
    window_start_sample: int
    window_stop_sample: int
    emitted_class: int = NO_COMMAND
    decision_state_before: str | None = None
    decision_state_after: str | None = None
    transition_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "subject_id", "session_id", "run_id", "segment_id", "window_index",
            "window_start_sample", "window_stop_sample", "emitted_class",
        ):
            object.__setattr__(self, name, _python_int(getattr(self, name), name))
        if self.subject_id < 1:
            raise ValueError("subject_id 必须从 1 开始")
        if min(
            self.session_id, self.run_id, self.segment_id, self.window_index,
            self.window_start_sample, self.window_stop_sample,
        ) < 0:
            raise ValueError("身份索引和原生采样坐标不得为负")
        for name in ("decision_state_before", "decision_state_after", "transition_reason"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} 必须为字符串或 None")

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.subject_id, self.session_id, self.run_id, self.segment_id

    @property
    def identity(self) -> tuple[tuple[int, int, int, int], int]:
        return self.key, self.window_index


@dataclass(frozen=True)
class ExpectedWindow:
    """来自冻结母索引的正式窗口身份，不含任何模型输出。"""

    subject_id: int
    session_id: int
    run_id: int
    segment_id: int
    window_index: int
    window_start_sample: int
    window_stop_sample: int

    def __post_init__(self) -> None:
        for name in (
            "subject_id", "session_id", "run_id", "segment_id", "window_index",
            "window_start_sample", "window_stop_sample",
        ):
            object.__setattr__(self, name, _python_int(getattr(self, name), name))
        if self.subject_id < 1:
            raise ValueError("subject_id 必须从 1 开始")
        if min(
            self.session_id, self.run_id, self.segment_id, self.window_index,
            self.window_start_sample, self.window_stop_sample,
        ) < 0:
            raise ValueError("身份索引和原生采样坐标不得为负")

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.subject_id, self.session_id, self.run_id, self.segment_id

    @property
    def identity(self) -> tuple[tuple[int, int, int, int], int]:
        return self.key, self.window_index


# 窗口指标统一从原始 logit 计算，避免调用方提前舍弃置信度信息。
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _validated_targets(
    y_true: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    raw_true = np.asarray(y_true)
    names = list(class_names)
    if raw_true.ndim != 1 or raw_true.size == 0:
        raise ValueError("y_true 必须是一维非空数组")
    contains_bool = raw_true.ndim == 1 and any(
        isinstance(item, (bool, np.bool_)) for item in raw_true.tolist()
    )
    try:
        integer_true = raw_true.astype(np.int64)
        valid_integer = (
            not contains_bool
            and np.all(np.isfinite(raw_true))
            and np.all(raw_true == integer_true)
        )
    except (TypeError, ValueError):
        valid_integer = False
    if not valid_integer:
        raise ValueError("y_true 必须只包含有限整数标签")
    if not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError("class_names 必须非空且不重复")
    if np.any(integer_true < 0) or np.any(integer_true >= len(names)):
        raise ValueError("y_true 包含 class_names 之外的标签")
    return integer_true, names


def _hard_metrics(
    y: np.ndarray,
    predictions: np.ndarray,
    names: list[str],
    require_all_classes: bool,
) -> dict:
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    if require_all_classes and np.any(support == 0):
        raise ValueError("正式宏平均要求每个固定类别至少有一个真值样本")

    per_class: dict[str, dict] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(names):
        true_positive = int(confusion[index, index])
        precision = true_positive / predicted[index] if predicted[index] else 0.0
        recall = true_positive / support[index] if support[index] else None
        f1 = None if recall is None else (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        per_class[name] = {
            "support": int(support[index]),
            "predicted": int(predicted[index]),
            "precision": float(precision),
            "recall": None if recall is None else float(recall),
            "f1": None if f1 is None else float(f1),
        }
        if recall is not None:
            recalls.append(float(recall))
            f1_values.append(float(f1))

    complete = len(recalls) == len(names)
    return {
        "sample_count": int(y.size),
        "class_names": names,
        "accuracy": float(np.mean(predictions == y)),
        "macro_metrics_computable": complete,
        "missing_true_classes": [names[index] for index, count in enumerate(support) if count == 0],
        "balanced_accuracy": float(np.mean(recalls)) if complete else None,
        "balanced_accuracy_class_count": len(recalls),
        "macro_f1": float(np.mean(f1_values)) if complete else None,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _hard_label_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    *,
    require_all_classes: bool = True,
) -> dict:
    """计算没有概率语义的硬标签指标，供两阶段五分类级联使用。"""
    y, names = _validated_targets(y_true, class_names)
    raw_predictions = np.asarray(y_pred)
    contains_bool = raw_predictions.ndim == 1 and any(
        isinstance(item, (bool, np.bool_)) for item in raw_predictions.tolist()
    )
    try:
        predictions = raw_predictions.astype(np.int64)
        valid_integer = (
            not contains_bool
            and np.all(np.isfinite(raw_predictions))
            and np.all(raw_predictions == predictions)
        )
    except (TypeError, ValueError):
        valid_integer = False
    if raw_predictions.shape != y.shape or not valid_integer:
        raise ValueError("y_pred 必须是与 y_true 同形的一维有限整数数组")
    if np.any(predictions < 0) or np.any(predictions >= len(names)):
        raise ValueError("y_pred 包含 class_names 之外的标签")
    return _hard_metrics(y, predictions, names, require_all_classes)


def _window_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    *,
    require_all_classes: bool = True,
) -> dict:
    """计算固定标签顺序的分类、置信度和逐类指标。"""
    y, names = _validated_targets(y_true, class_names)
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (y.size, len(names)) or not np.all(np.isfinite(scores)):
        raise ValueError("logits 的形状、样本数或有限性不合法")
    probabilities = _softmax(scores)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    result = _hard_metrics(y, predictions, names, require_all_classes)
    # 在平移后的坐标中同时计算分母和真实类项，避免大共同偏置造成灾难性消减。
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    logsumexp_shifted = np.log(np.sum(np.exp(shifted_scores), axis=1))
    true_shifted = shifted_scores[np.arange(y.size), y]
    nll = np.mean(logsumexp_shifted - true_shifted)
    if not np.isfinite(nll):
        raise ValueError("有限 logits 产生了不可表示的 NLL")
    one_hot = np.eye(len(names), dtype=np.float64)[y]
    result.update({
        "nll": float(nll),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    })
    return result


def hierarchical_5class_predictions(stage1_logits: np.ndarray, stage2_logits: np.ndarray) -> np.ndarray:
    """按 Stage 1 硬门控生成 IDLE+四类 MI 的五分类输出。"""
    stage1 = np.asarray(stage1_logits, dtype=np.float64)
    stage2 = np.asarray(stage2_logits, dtype=np.float64)
    if stage1.ndim != 2 or stage1.shape[1] != 2:
        raise ValueError("stage1_logits 必须为 [window, 2]")
    if stage2.shape != (stage1.shape[0], MI_CLASS_COUNT):
        raise ValueError("stage2_logits 必须为 [window, 4] 且与 Stage 1 行数相同")
    if not np.all(np.isfinite(stage1)) or not np.all(np.isfinite(stage2)):
        raise ValueError("级联 logit 不得包含 NaN 或无穷值")
    task_mask = np.argmax(stage1, axis=1) == 1
    result = np.zeros(stage1.shape[0], dtype=np.int64)
    result[task_mask] = 1 + np.argmax(stage2[task_mask], axis=1)
    return result


# 正式入口固定类别身份；底层通用函数保持私有，防止调用方交换标签顺序。
def stage1_window_metrics(
    y_true: Sequence[int] | np.ndarray,
    logits: np.ndarray,
    *,
    require_all_classes: bool = True,
) -> dict:
    return _window_classification_metrics(y_true, logits, STAGE1_CLASS_NAMES, require_all_classes=require_all_classes)


def stage2_window_metrics(
    y_true: Sequence[int] | np.ndarray,
    logits: np.ndarray,
    *,
    require_all_classes: bool = True,
) -> dict:
    return _window_classification_metrics(y_true, logits, STAGE2_CLASS_NAMES, require_all_classes=require_all_classes)


def final_5class_window_metrics(
    y_true: Sequence[int] | np.ndarray,
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    *,
    require_all_classes: bool = True,
) -> dict:
    predictions = hierarchical_5class_predictions(stage1_logits, stage2_logits)
    return _hard_label_classification_metrics(
        y_true,
        predictions,
        FINAL_CLASS_NAMES,
        require_all_classes=require_all_classes,
    )


def _overlap_samples(record: DecisionRecord | ExpectedWindow, event: MIEvent) -> int:
    return max(
        0,
        min(record.window_stop_sample, event.offset_sample)
        - max(record.window_start_sample, event.onset_sample),
    )


def _eligible(record: DecisionRecord | ExpectedWindow, event: MIEvent, margin_samples: int) -> bool:
    return (
        record.key == event.key
        and record.window_stop_sample <= event.offset_sample
        and _overlap_samples(record, event) >= margin_samples
    )


def _inventory_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 所有结构约束在评分前一次性失败，防止坏轨迹被静默计入正式结果。
def _validate_online_inputs(
    segments: Sequence[ScoringSegment],
    events: Sequence[MIEvent],
    expected_windows: Sequence[ExpectedWindow],
    decisions: Sequence[DecisionRecord],
    *,
    mode: str,
    window_samples: int,
    step_samples: int,
) -> tuple[
    dict[tuple[int, int, int, int], ScoringSegment],
    dict[tuple[tuple[int, int, int, int], int], ExpectedWindow],
    dict[tuple[tuple[int, int, int, int], int], DecisionRecord],
]:
    segment_map: dict[tuple[int, int, int, int], ScoringSegment] = {}
    for segment in segments:
        if segment.stop_sample <= segment.start_sample or segment.key in segment_map:
            raise ValueError("segment 坐标必须递增且身份不得重复")
        segment_map[segment.key] = segment
    if not segment_map:
        raise ValueError("至少需要一个有效评分 segment")
    segments_by_run: dict[tuple[int, int, int], list[ScoringSegment]] = {}
    for segment in segments:
        segments_by_run.setdefault(segment.key[:3], []).append(segment)
    for run_segments in segments_by_run.values():
        ordered = sorted(run_segments, key=lambda item: item.start_sample)
        if any(right.start_sample < left.stop_sample for left, right in zip(ordered, ordered[1:])):
            raise ValueError("同一 run 的有效 segment 不得重叠")

    # 冻结窗口清单必须自行构成完整网格，不能由实际提交的预测反推可评分事件。
    expected_map: dict[tuple[tuple[int, int, int, int], int], ExpectedWindow] = {}
    windows_by_stream: dict[tuple[int, int, int, int], list[ExpectedWindow]] = {}
    for window in expected_windows:
        segment = segment_map.get(window.key)
        if segment is None or window.identity in expected_map:
            raise ValueError("冻结窗口必须属于唯一的有效 segment")
        if window.window_index < 0:
            raise ValueError("window_index 必须从 0 开始且非负")
        if window.window_stop_sample - window.window_start_sample != window_samples:
            raise ValueError("冻结窗口长度与正式配置不一致")
        if not (segment.start_sample <= window.window_start_sample < window.window_stop_sample <= segment.stop_sample):
            raise ValueError("冻结窗口必须完整位于对应 segment 内")
        expected_map[window.identity] = window
        windows_by_stream.setdefault(window.key, []).append(window)
    for key, segment in segment_map.items():
        ordered = sorted(windows_by_stream.get(key, []), key=lambda item: item.window_index)
        if not ordered:
            if segment.stop_sample - segment.start_sample >= window_samples:
                raise ValueError("长度足够的 segment 不得缺少冻结窗口")
            # 短 clean segment 仍属于真实运行时间，只是无法形成一次完整决策。
            continue
        if [item.window_index for item in ordered] != list(range(len(ordered))):
            raise ValueError("冻结窗口的 window_index 必须从 0 连续编号")
        if ordered[0].window_start_sample != segment.start_sample:
            raise ValueError("冻结窗口网格必须从 segment 首个有效采样点开始")
        if any(
            right.window_start_sample - left.window_start_sample != step_samples
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("冻结窗口起点必须严格采用固定步长")
        tail_samples = segment.stop_sample - ordered[-1].window_stop_sample
        if not 0 <= tail_samples < step_samples:
            raise ValueError("segment 尾部与完整冻结窗口网格不一致")

    event_ids: set[tuple[tuple[int, int, int, int], str]] = set()
    by_stream: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    for event in events:
        segment = segment_map.get(event.key)
        identity = (event.key, event.event_id)
        if segment is None or identity in event_ids:
            raise ValueError("MI 事件必须属于唯一的有效 segment")
        if not (segment.start_sample <= event.onset_sample < event.offset_sample <= segment.stop_sample):
            raise ValueError("MI 事件必须完整位于对应 segment 内")
        if not 1 <= event.true_class <= MI_CLASS_COUNT:
            raise ValueError("MI 事件类别必须为 1..4")
        event_ids.add(identity)
        by_stream.setdefault(event.key, []).append(event)
    for stream_events in by_stream.values():
        ordered = sorted(stream_events, key=lambda item: item.onset_sample)
        if any(right.onset_sample < left.offset_sample for left, right in zip(ordered, ordered[1:])):
            raise ValueError("同一 segment 内的 MI 事件不得重叠")

    decision_map: dict[tuple[tuple[int, int, int, int], int], DecisionRecord] = {}
    records_by_stream: dict[tuple[int, int, int, int], list[DecisionRecord]] = {}
    for record in decisions:
        if record.key not in segment_map or record.identity in decision_map:
            raise ValueError("每个决策窗口必须属于唯一的有效 segment")
        if record.emitted_class not in (NO_COMMAND, 1, 2, 3, 4):
            raise ValueError("emitted_class 必须为 -1 或 1..4")
        decision_map[record.identity] = record
        records_by_stream.setdefault(record.key, []).append(record)
    missing = sorted(set(expected_map) - set(decision_map))
    extra = sorted(set(decision_map) - set(expected_map))
    if missing or extra:
        raise ValueError(f"决策轨迹与冻结窗口清单不一致：missing={len(missing)}, extra={len(extra)}")
    for identity, expected in expected_map.items():
        actual = decision_map[identity]
        if (
            actual.window_start_sample != expected.window_start_sample
            or actual.window_stop_sample != expected.window_stop_sample
        ):
            raise ValueError("决策窗口坐标与冻结母索引不一致")

    # 模式必须显式选择；两状态基线、三状态候选策略和无状态诊断分别校验。
    for stream_records in records_by_stream.values():
        if mode == STATELESS_DIAGNOSTIC:
            if any(
                record.decision_state_before is not None or record.decision_state_after is not None
                or record.transition_reason is not None
                for record in stream_records
            ):
                raise ValueError("无状态诊断模式不得混入状态字段或转换原因")
            continue

        allowed_states = (
            (READY, WAIT_IDLE)
            if mode == STATEFUL_STRICT
            else (READY, TASK_CANDIDATE, WAIT_IDLE)
        )
        if any(
            record.decision_state_before not in allowed_states
            or record.decision_state_after not in allowed_states
            for record in stream_records
        ):
            required = "/".join(allowed_states)
            raise ValueError(f"{mode} 要求每个窗口完整记录 {required}")
        ordered = sorted(stream_records, key=lambda item: (item.window_stop_sample, item.window_index))
        if ordered[0].decision_state_before != READY:
            raise ValueError("每个 segment 的决策状态必须从 READY 开始")
        for index, record in enumerate(ordered):
            transition = (record.decision_state_before, record.decision_state_after)
            if mode == STATEFUL_STRICT:
                if record.transition_reason is not None:
                    raise ValueError("两状态严格基线不得混入候选态转换原因")
                if record.emitted_class != NO_COMMAND:
                    if transition != (READY, WAIT_IDLE):
                        raise ValueError("MI 指令只能从 READY 发出并立即进入 WAIT_IDLE")
                elif transition not in {(READY, READY), (WAIT_IDLE, WAIT_IDLE), (WAIT_IDLE, READY)}:
                    raise ValueError("无输出窗口包含非法状态转换")
            else:
                # 候选态的原因字段与状态转换一一绑定，避免轨迹声称的退出原因和行为不符。
                no_command_contract = {
                    (READY, READY): {None},
                    (READY, TASK_CANDIDATE): {CANDIDATE_OPEN},
                    (TASK_CANDIDATE, TASK_CANDIDATE): {None},
                    (TASK_CANDIDATE, READY): {
                        CANDIDATE_ABORT_STAGE1,
                        CANDIDATE_TIMEOUT,
                    },
                    (WAIT_IDLE, WAIT_IDLE): {None},
                    (WAIT_IDLE, READY): {IDLE_RESET},
                }
                if record.emitted_class != NO_COMMAND:
                    # Fast-0 在开门窗原子提交；Fast-1 与慢通道都从候选态提交。
                    allowed_command_transitions = {
                        (READY, WAIT_IDLE): {FAST0_COMMAND_COMMIT},
                        (TASK_CANDIDATE, WAIT_IDLE): {
                            COMMAND_COMMIT,
                            FAST1_COMMAND_COMMIT,
                        },
                    }
                    if (
                        transition not in allowed_command_transitions
                        or record.transition_reason not in allowed_command_transitions[transition]
                    ):
                        raise ValueError("候选策略包含非法的命令提交路径")
                    if (
                        record.transition_reason == FAST1_COMMAND_COMMIT
                        and (
                            index == 0
                            or ordered[index - 1].transition_reason != CANDIDATE_OPEN
                        )
                    ):
                        raise ValueError("Fast-1 必须紧接候选打开窗口提交")
                elif (
                    transition not in no_command_contract
                    or record.transition_reason not in no_command_contract[transition]
                ):
                    raise ValueError("候选策略包含非法状态转换或转换原因")
            if index and ordered[index - 1].decision_state_after != record.decision_state_before:
                raise ValueError("相邻窗口的决策状态不连续")
    return segment_map, expected_map, decision_map


def _latency_summary(latencies_seconds: list[float]) -> dict:
    if not latencies_seconds:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "p90": None}
    values = np.asarray(latencies_seconds, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _candidate_diagnostics(
    segments: Sequence[ScoringSegment],
    events: Sequence[MIEvent],
    decisions: Sequence[DecisionRecord],
    matches: Sequence[dict],
    *,
    sampling_rate: float,
    margin_samples: int,
) -> dict:
    """从已完成的三状态轨迹统计候选区间，不让真值反向参与状态转换。"""
    intervals: list[dict] = []
    completed_dwell: list[float] = []
    by_stream: dict[tuple[int, int, int, int], list[DecisionRecord]] = {}
    for record in decisions:
        by_stream.setdefault(record.key, []).append(record)

    # 候选时长从 READY->TASK_CANDIDATE 的决策时刻算到退出候选态的决策时刻。
    for key, stream in sorted(by_stream.items()):
        ordered = sorted(stream, key=lambda item: (item.window_stop_sample, item.window_index))
        opened: DecisionRecord | None = None
        for record in ordered:
            if record.transition_reason == CANDIDATE_OPEN:
                opened = record
                continue
            if record.transition_reason not in {
                CANDIDATE_ABORT_STAGE1,
                CANDIDATE_TIMEOUT,
                COMMAND_COMMIT,
                FAST1_COMMAND_COMMIT,
            }:
                continue
            if opened is None:  # 前置状态校验已保证不会发生，保留断言防止以后漂移。
                raise RuntimeError("候选退出缺少对应的候选打开记录")
            dwell = (record.window_stop_sample - opened.window_stop_sample) / sampling_rate
            completed_dwell.append(float(dwell))
            intervals.append({
                "subject_id": key[0],
                "session_id": key[1],
                "run_id": key[2],
                "segment_id": key[3],
                "open_window_index": opened.window_index,
                "open_decision_sample": opened.window_stop_sample,
                "exit_window_index": record.window_index,
                "exit_decision_sample": record.window_stop_sample,
                "outcome": record.transition_reason,
                "duration_seconds": float(dwell),
            })
            opened = None
        if opened is not None:
            observed = (ordered[-1].window_stop_sample - opened.window_stop_sample) / sampling_rate
            intervals.append({
                "subject_id": key[0],
                "session_id": key[1],
                "run_id": key[2],
                "segment_id": key[3],
                "open_window_index": opened.window_index,
                "open_decision_sample": opened.window_stop_sample,
                "exit_window_index": None,
                "exit_decision_sample": None,
                "outcome": "segment_end_unresolved",
                "duration_seconds": float(observed),
            })

    outcome_counts = {
        reason: sum(item["outcome"] == reason for item in intervals)
        for reason in (
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_TIMEOUT,
            COMMAND_COMMIT,
            FAST1_COMMAND_COMMIT,
            "segment_end_unresolved",
        )
    }
    open_count = len(intervals)
    abort_count = outcome_counts[CANDIDATE_ABORT_STAGE1] + outcome_counts[CANDIDATE_TIMEOUT]
    valid_seconds = sum(item.stop_sample - item.start_sample for item in segments) / sampling_rate

    # “超时相关 MISS”只表示同一事件可评分区间内出现过超时，不声称超时是 MISS 的因果来源。
    event_lookup = {
        (*event.key, event.event_id): event
        for event in events
    }
    timeout_records = [
        record for record in decisions
        if record.transition_reason == CANDIDATE_TIMEOUT
    ]
    miss_matches = [
        item for item in matches
        if item["scorable"] and item["outcome"] == "miss"
    ]
    timeout_related_miss_events: list[dict] = []
    for item in miss_matches:
        identity = (
            item["subject_id"], item["session_id"], item["run_id"], item["segment_id"],
            item["event_id"],
        )
        event = event_lookup[identity]
        if any(_eligible(record, event, margin_samples) for record in timeout_records):
            timeout_related_miss_events.append({
                "subject_id": item["subject_id"],
                "session_id": item["session_id"],
                "run_id": item["run_id"],
                "segment_id": item["segment_id"],
                "event_id": item["event_id"],
            })

    def ratio(numerator: int) -> float | None:
        return None if open_count == 0 else numerator / open_count

    return {
        "candidate_open_count": open_count,
        "candidate_opens_per_valid_minute": (
            None if valid_seconds <= 0 else open_count / (valid_seconds / 60.0)
        ),
        "candidate_command_count": (
            outcome_counts[COMMAND_COMMIT] + outcome_counts[FAST1_COMMAND_COMMIT]
        ),
        "candidate_conversion_rate": ratio(
            outcome_counts[COMMAND_COMMIT] + outcome_counts[FAST1_COMMAND_COMMIT]
        ),
        "candidate_abort_count": abort_count,
        "candidate_abort_rate": ratio(abort_count),
        "candidate_stage1_abort_count": outcome_counts[CANDIDATE_ABORT_STAGE1],
        "candidate_stage1_abort_rate": ratio(outcome_counts[CANDIDATE_ABORT_STAGE1]),
        "candidate_timeout_count": outcome_counts[CANDIDATE_TIMEOUT],
        "candidate_timeout_rate": ratio(outcome_counts[CANDIDATE_TIMEOUT]),
        "candidate_unresolved_count": outcome_counts["segment_end_unresolved"],
        "candidate_unresolved_rate": ratio(outcome_counts["segment_end_unresolved"]),
        "completed_candidate_dwell_seconds": _latency_summary(completed_dwell),
        "miss_event_with_candidate_timeout_count": len(timeout_related_miss_events),
        "miss_event_with_candidate_timeout_rate": (
            None if not miss_matches else len(timeout_related_miss_events) / len(miss_matches)
        ),
        "miss_event_with_candidate_timeout_events": timeout_related_miss_events,
        "timeout_miss_interpretation": "associated_with_timeout_not_proven_causal",
        "candidate_intervals": intervals,
    }


def evaluate_online_events(
    segments: Sequence[ScoringSegment],
    events: Sequence[MIEvent],
    expected_windows: Sequence[ExpectedWindow],
    decisions: Sequence[DecisionRecord],
    *,
    mode: str,
    sampling_rate: float = NATIVE_SAMPLING_RATE,
    min_overlap_seconds: float = MIN_OVERLAP_SECONDS,
    window_samples: int = WINDOW_SAMPLES,
    step_samples: int = STEP_SAMPLES,
) -> dict:
    """按 0.5 秒证据 margin、首次输出和有效 IDLE 时长计算事件指标。"""
    if mode not in (STATEFUL_STRICT, STATEFUL_CANDIDATE, STATELESS_DIAGNOSTIC):
        raise ValueError(
            "mode 必须显式选择 stateful_strict、stateful_candidate 或 stateless_diagnostic"
        )
    sampling_rate = _finite_float(sampling_rate, "sampling_rate")
    min_overlap_seconds = _finite_float(min_overlap_seconds, "min_overlap_seconds")
    if sampling_rate != NATIVE_SAMPLING_RATE:
        raise ValueError("事件评分坐标固定使用 BNCI2014001 原生 250 Hz 时钟")
    if min_overlap_seconds != MIN_OVERLAP_SECONDS:
        raise ValueError("正式事件匹配 margin 固定为 0.5 秒")
    margin_float = sampling_rate * min_overlap_seconds
    margin_samples = int(round(margin_float))
    if min_overlap_seconds <= 0 or not np.isclose(margin_float, margin_samples, rtol=0.0, atol=1e-9):
        raise ValueError("min_overlap_seconds 必须对应整数个原生采样点")
    window_samples = _python_int(window_samples, "window_samples")
    step_samples = _python_int(step_samples, "step_samples")
    if window_samples != WINDOW_SAMPLES or step_samples != STEP_SAMPLES:
        raise ValueError("原生窗口网格固定为 500 点窗长和 125 点步长")

    segment_map, expected_map, decision_map = _validate_online_inputs(
        segments,
        events,
        expected_windows,
        decisions,
        mode=mode,
        window_samples=window_samples,
        step_samples=step_samples,
    )
    ordered_decisions = sorted(decisions, key=lambda item: (*item.key, item.window_stop_sample, item.window_index))
    expected_by_stream: dict[tuple[int, int, int, int], list[ExpectedWindow]] = {}
    events_by_stream: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    for window in sorted(expected_windows, key=lambda item: (*item.key, item.window_index)):
        expected_by_stream.setdefault(window.key, []).append(window)
    for event in events:
        events_by_stream.setdefault(event.key, []).append(event)

    # 无窗口短 segment 和长 segment 末尾不足一步的采样仍计入有效运行时长，并显式审计。
    windowless_segment_samples = 0
    trailing_unwindowed_samples = 0
    for key, segment in segment_map.items():
        stream_windows = expected_by_stream.get(key, [])
        if not stream_windows:
            windowless_segment_samples += segment.stop_sample - segment.start_sample
        else:
            trailing_unwindowed_samples += segment.stop_sample - stream_windows[-1].window_stop_sample

    matches: list[dict] = []
    used_records: set[DecisionRecord] = set()
    for event in sorted(events, key=lambda item: (*item.key, item.onset_sample)):
        eligible_expected = [
            window for window in expected_by_stream.get(event.key, [])
            if _eligible(window, event, margin_samples)
        ]
        eligible = [decision_map[window.identity] for window in eligible_expected]
        emitted = [record for record in eligible if record.emitted_class != NO_COMMAND]
        matched = emitted[0] if emitted else None
        if matched is not None:
            used_records.add(matched)
        outcome = "unscorable" if not eligible else "miss"
        if matched is not None:
            outcome = "correct" if matched.emitted_class == event.true_class else "wrong_class"
        matches.append({
            "event_id": event.event_id,
            "subject_id": event.subject_id,
            "session_id": event.session_id,
            "run_id": event.run_id,
            "segment_id": event.segment_id,
            "true_class": event.true_class,
            "scorable": bool(eligible_expected),
            "outcome": outcome,
            "predicted_class": None if matched is None else matched.emitted_class,
            "window_index": None if matched is None else matched.window_index,
            "decision_sample": None if matched is None else matched.window_stop_sample,
            "latency_seconds": None if matched is None else (matched.window_stop_sample - event.onset_sample) / sampling_rate,
        })

    # 未匹配指令按真实时间位置拆成 IDLE 误触发、证据不足的过早输出和事件内额外输出。
    idle_false_commands = 0
    too_early_commands = 0
    additional_event_commands = 0
    for record in (item for item in ordered_decisions if item.emitted_class != NO_COMMAND and item not in used_records):
        candidates = [
            event for event in events_by_stream.get(record.key, [])
            if event.onset_sample <= record.window_stop_sample <= event.offset_sample
        ]
        if not candidates:
            idle_false_commands += 1
            continue
        event = max(candidates, key=lambda item: _overlap_samples(record, item))
        if _overlap_samples(record, event) < margin_samples:
            too_early_commands += 1
        else:
            additional_event_commands += 1

    scorable = [item for item in matches if item["scorable"]]
    correct = [item for item in scorable if item["outcome"] == "correct"]
    triggered = [item for item in scorable if item["predicted_class"] is not None]
    confusion = np.zeros((MI_CLASS_COUNT, MI_CLASS_COUNT + 1), dtype=np.int64)
    per_class: dict[str, dict] = {}
    class_rates: list[float] = []
    for item in scorable:
        column = MI_CLASS_COUNT if item["predicted_class"] is None else item["predicted_class"] - 1
        confusion[item["true_class"] - 1, column] += 1
    for class_id in range(1, MI_CLASS_COUNT + 1):
        total = sum(item["true_class"] == class_id for item in scorable)
        count = sum(item["true_class"] == class_id and item["outcome"] == "correct" for item in scorable)
        rate = count / total if total else None
        per_class[STAGE2_CLASS_NAMES[class_id - 1]] = {
            "event_count": total,
            "correct_count": count,
            "correct_event_rate": rate,
        }
        if rate is not None:
            class_rates.append(rate)

    idle_samples = sum(segment.stop_sample - segment.start_sample for segment in segment_map.values())
    idle_samples -= sum(event.offset_sample - event.onset_sample for event in events)
    idle_seconds = idle_samples / sampling_rate
    false_per_minute = None if idle_seconds <= 0 else idle_false_commands / (idle_seconds / 60.0)
    correct_latencies = [item["latency_seconds"] for item in correct]
    segment_rows = [
        {
            "subject_id": item.subject_id,
            "session_id": item.session_id,
            "run_id": item.run_id,
            "segment_id": item.segment_id,
            "start_sample": item.start_sample,
            "stop_sample": item.stop_sample,
        }
        for item in sorted(segments, key=lambda value: value.key)
    ]
    window_rows = [
        {
            "subject_id": item.subject_id,
            "session_id": item.session_id,
            "run_id": item.run_id,
            "segment_id": item.segment_id,
            "window_index": item.window_index,
            "window_start_sample": item.window_start_sample,
            "window_stop_sample": item.window_stop_sample,
        }
        for item in sorted(expected_windows, key=lambda value: (*value.key, value.window_index))
    ]
    event_rows = [
        {
            "event_id": item.event_id,
            "subject_id": item.subject_id,
            "session_id": item.session_id,
            "run_id": item.run_id,
            "segment_id": item.segment_id,
            "onset_sample": item.onset_sample,
            "offset_sample": item.offset_sample,
            "true_class": item.true_class,
        }
        for item in sorted(events, key=lambda value: (*value.key, value.onset_sample, value.event_id))
    ]
    decision_rows = [
        {
            "subject_id": item.subject_id,
            "session_id": item.session_id,
            "run_id": item.run_id,
            "segment_id": item.segment_id,
            "window_index": item.window_index,
            "window_start_sample": item.window_start_sample,
            "window_stop_sample": item.window_stop_sample,
            "emitted_class": item.emitted_class,
            "decision_state_before": item.decision_state_before,
            "decision_state_after": item.decision_state_after,
        }
        for item in sorted(decisions, key=lambda value: (*value.key, value.window_index))
    ]
    if mode == STATEFUL_CANDIDATE:
        for row, item in zip(
            decision_rows,
            sorted(decisions, key=lambda value: (*value.key, value.window_index)),
        ):
            row["transition_reason"] = item.transition_reason

    result = {
        "evaluation_mode": mode,
        "sampling_rate": float(sampling_rate),
        "min_overlap_seconds": float(min_overlap_seconds),
        "min_overlap_samples": margin_samples,
        "window_samples": window_samples,
        "step_samples": step_samples,
        "scoring_segment_count": len(segment_map),
        "scoring_segment_inventory_sha256": _inventory_sha256(segment_rows),
        "zero_window_segment_count": sum(key not in expected_by_stream for key in segment_map),
        "zero_window_segment_samples": windowless_segment_samples,
        "trailing_unwindowed_samples": trailing_unwindowed_samples,
        "expected_window_count": len(expected_map),
        "expected_window_inventory_sha256": _inventory_sha256(window_rows),
        "event_count": len(events),
        "event_inventory_sha256": _inventory_sha256(event_rows),
        "decision_inventory_sha256": _inventory_sha256(decision_rows),
        "scorable_event_count": len(scorable),
        "unscorable_event_count": len(events) - len(scorable),
        "correct_event_count": len(correct),
        "correct_event_rate": None if not scorable else len(correct) / len(scorable),
        "macro_correct_event_rate": None if len(class_rates) != MI_CLASS_COUNT else float(np.mean(class_rates)),
        "event_trigger_rate": None if not scorable else len(triggered) / len(scorable),
        "triggered_class_accuracy": None if not triggered else len(correct) / len(triggered),
        "miss_rate": None if not scorable else (len(scorable) - len(triggered)) / len(scorable),
        "event_confusion_matrix": confusion.tolist(),
        "event_confusion_rows": list(STAGE2_CLASS_NAMES),
        "event_confusion_columns": [*STAGE2_CLASS_NAMES, "MISS"],
        "per_class": per_class,
        "correct_detection_latency_seconds": _latency_summary(correct_latencies),
        "valid_idle_seconds": float(idle_seconds),
        "idle_false_command_count": idle_false_commands,
        "idle_false_commands_per_minute": false_per_minute,
        "too_early_command_count": too_early_commands,
        "additional_event_command_count": additional_event_commands,
        "additional_event_command_interpretation": (
            "possible_output_after_premature_rearm"
            if mode in (STATEFUL_STRICT, STATEFUL_CANDIDATE)
            else "multiple_stateless_outputs_within_one_event"
        ),
        "emitted_command_count": sum(item.emitted_class != NO_COMMAND for item in ordered_decisions),
        "event_matches": matches,
    }
    if mode == STATEFUL_CANDIDATE:
        result["candidate_diagnostics"] = _candidate_diagnostics(
            segments,
            events,
            decisions,
            matches,
            sampling_rate=sampling_rate,
            margin_samples=margin_samples,
        )
    return result
