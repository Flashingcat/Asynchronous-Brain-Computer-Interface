"""基础窗口指标与严格在线事件匹配的独立回归测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from protocol_metrics import (  # noqa: E402
    FINAL_CLASS_NAMES,
    READY,
    STATEFUL_STRICT,
    STATELESS_DIAGNOSTIC,
    STAGE1_CLASS_NAMES,
    STAGE2_CLASS_NAMES,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
    final_5class_window_metrics,
    hierarchical_5class_predictions,
    stage1_window_metrics,
    stage2_window_metrics,
)


class WindowMetricTests(unittest.TestCase):
    """锁定三个正式入口的类别语义、公式和异常行为。"""

    def test_stage1_metrics_include_accuracy_calibration_and_counts(self) -> None:
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
        logits = np.asarray([[3, 0], [0, 3], [0, 3], [0, 3]], dtype=np.float64)
        result = stage1_window_metrics(y_true, logits)

        self.assertEqual(result["class_names"], list(STAGE1_CLASS_NAMES))
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertGreater(result["nll"], 0.0)
        self.assertGreater(result["brier_multiclass"], 0.0)
        self.assertEqual(result["per_class"]["idle"]["support"], 2)

    def test_missing_class_is_failure_formally_and_none_diagnostically(self) -> None:
        with self.assertRaisesRegex(ValueError, "每个固定类别"):
            stage1_window_metrics([0, 0], np.zeros((2, 2)))

        diagnostic = stage1_window_metrics(
            [0, 0],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            require_all_classes=False,
        )
        self.assertFalse(diagnostic["macro_metrics_computable"])
        self.assertEqual(diagnostic["missing_true_classes"], ["task"])
        self.assertIsNone(diagnostic["balanced_accuracy"])
        self.assertIsNone(diagnostic["macro_f1"])
        self.assertIsNone(diagnostic["per_class"]["task"]["recall"])

        with self.assertRaisesRegex(ValueError, "有限性"):
            stage1_window_metrics([0, 1], [[0, 1], [np.nan, 0]])
        with self.assertRaisesRegex(ValueError, "有限整数标签"):
            stage1_window_metrics([False, True], [[1, 0], [0, 1]])

    def test_nll_does_not_clip_extreme_but_finite_logits(self) -> None:
        # 两个样本都以 1000 logit margin 自信地预测错，真实 NLL 应接近 1000。
        result = stage1_window_metrics([0, 1], [[-1000.0, 0.0], [0.0, -1000.0]])
        self.assertAlmostEqual(result["nll"], 1000.0)

        # 分类概率只由 logit 差决定，共同加上巨大偏置后 NLL 仍必须为 log(2)。
        offset = stage1_window_metrics([0, 1], [[1e16, 1e16], [1e16, 1e16]])
        self.assertAlmostEqual(offset["nll"], np.log(2.0))

    def test_formal_stage2_and_final_wrappers_freeze_label_order(self) -> None:
        stage2 = stage2_window_metrics([0, 1, 2, 3], np.eye(4) * 4.0)
        self.assertEqual(stage2["class_names"], list(STAGE2_CLASS_NAMES))
        self.assertEqual(stage2["accuracy"], 1.0)

        stage1_logits = np.asarray([[3, 0], [0, 3], [0, 3], [0, 3], [0, 3]])
        stage2_logits = np.asarray([
            [4, 0, 0, 0],
            [4, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 4],
        ])
        final = final_5class_window_metrics(
            [0, 1, 2, 3, 4],
            stage1_logits,
            stage2_logits,
        )
        self.assertEqual(final["class_names"], list(FINAL_CLASS_NAMES))
        self.assertEqual(final["accuracy"], 1.0)
        self.assertNotIn("nll", final)

    def test_hierarchical_prediction_keeps_stage1_error_propagation(self) -> None:
        stage1 = np.asarray([[2, 0], [0, 2], [0, 2]], dtype=np.float64)
        stage2 = np.asarray([[0, 5, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]], dtype=np.float64)
        prediction = hierarchical_5class_predictions(stage1, stage2)
        np.testing.assert_array_equal(prediction, [0, 2, 3])


class OnlineEventMetricTests(unittest.TestCase):
    """用完整母索引测试事件匹配；测试数据不读取 EEG 或模型。"""

    @staticmethod
    def _segment(stop_sample: int = 7500) -> ScoringSegment:
        return ScoringSegment(1, 0, 0, 0, 0, stop_sample)

    @staticmethod
    def _events() -> list[MIEvent]:
        return [
            MIEvent("e1", 1, 0, 0, 0, 1000, 2000, 1),
            MIEvent("e2", 1, 0, 0, 0, 2500, 3500, 2),
            MIEvent("e3", 1, 0, 0, 0, 4000, 5000, 3),
            MIEvent("e4", 1, 0, 0, 0, 5500, 6500, 4),
        ]

    @staticmethod
    def _expected(segment: ScoringSegment) -> list[ExpectedWindow]:
        starts = range(segment.start_sample, segment.stop_sample - 500 + 1, 125)
        return [
            ExpectedWindow(*segment.key, index, start, start + 500)
            for index, start in enumerate(starts)
        ]

    @staticmethod
    def _stateless(
        expected: list[ExpectedWindow],
        outputs: dict[int, int] | None = None,
    ) -> list[DecisionRecord]:
        outputs = outputs or {}
        return [
            DecisionRecord(
                *window.key,
                window.window_index,
                window.window_start_sample,
                window.window_stop_sample,
                outputs.get(window.window_index, -1),
            )
            for window in expected
        ]

    @staticmethod
    def _stateful(
        expected: list[ExpectedWindow],
        overrides: dict[int, tuple[int, str, str]],
    ) -> list[DecisionRecord]:
        records: list[DecisionRecord] = []
        for window in expected:
            emitted, before, after = overrides.get(window.window_index, (-1, READY, READY))
            records.append(DecisionRecord(
                *window.key,
                window.window_index,
                window.window_start_sample,
                window.window_stop_sample,
                emitted,
                before,
                after,
            ))
        return records

    def test_margin_first_command_and_idle_denominator_are_exact(self) -> None:
        segment = self._segment()
        expected = self._expected(segment)
        decisions = self._stateless(expected, {
            # e1 起点没有 MI 证据；首个合法输出为错误类别，后续正确输出不能补救。
            4: 1,
            6: 2,
            8: 1,
            # 该输出的决策时刻位于两个事件之间，因此属于 IDLE 误触发。
            14: 1,
            # e2 无输出；e3 在终点输出；e4 在起点后 2 秒输出。
            36: 3,
            44: 4,
        })
        result = evaluate_online_events(
            [segment],
            self._events(),
            expected,
            decisions,
            mode=STATELESS_DIAGNOSTIC,
        )

        self.assertEqual(result["evaluation_mode"], STATELESS_DIAGNOSTIC)
        self.assertEqual(result["min_overlap_samples"], 125)
        self.assertEqual(result["expected_window_count"], 57)
        self.assertEqual(result["scorable_event_count"], 4)
        self.assertEqual(result["correct_event_count"], 2)
        self.assertEqual(result["correct_event_rate"], 0.5)
        self.assertEqual(result["macro_correct_event_rate"], 0.5)
        self.assertEqual(result["event_trigger_rate"], 0.75)
        self.assertAlmostEqual(result["triggered_class_accuracy"], 2 / 3)
        self.assertEqual(result["miss_rate"], 0.25)
        self.assertEqual(result["event_confusion_matrix"], [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ])
        self.assertEqual(result["too_early_command_count"], 1)
        self.assertEqual(result["additional_event_command_count"], 1)
        self.assertEqual(result["idle_false_command_count"], 1)
        self.assertEqual(result["valid_idle_seconds"], 14.0)
        self.assertAlmostEqual(result["idle_false_commands_per_minute"], 60 / 14)
        self.assertEqual(result["correct_detection_latency_seconds"], {
            "count": 2,
            "mean": 3.0,
            "median": 3.0,
            "q25": 2.5,
            "q75": 3.5,
            "p90": 3.8,
        })
        self.assertEqual(result["event_matches"][0]["outcome"], "wrong_class")
        self.assertEqual(result["event_matches"][0]["predicted_class"], 2)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_event_without_eligible_window_is_unscorable_not_miss(self) -> None:
        segment = self._segment()
        expected = self._expected(segment)
        event = MIEvent("short", 1, 0, 0, 0, 1000, 1100, 1)
        result = evaluate_online_events(
            [segment],
            [event],
            expected,
            self._stateless(expected),
            mode=STATELESS_DIAGNOSTIC,
        )

        self.assertEqual(result["scorable_event_count"], 0)
        self.assertEqual(result["unscorable_event_count"], 1)
        self.assertIsNone(result["correct_event_rate"])
        self.assertEqual(result["valid_idle_seconds"], (7500 - 100) / 250)

    def test_decision_trace_must_equal_complete_frozen_window_grid(self) -> None:
        segment = self._segment(750)
        expected = self._expected(segment)
        decisions = self._stateless(expected)

        with self.assertRaisesRegex(ValueError, "missing=1"):
            evaluate_online_events(
                [segment], [], expected, decisions[:-1], mode=STATELESS_DIAGNOSTIC,
            )

        # 同时删去母索引和决策也不能躲过检查，因为 segment 尾部不再闭合。
        with self.assertRaisesRegex(ValueError, "尾部"):
            evaluate_online_events(
                [segment], [], expected[:-1], decisions[:-1], mode=STATELESS_DIAGNOSTIC,
            )

        shifted = list(decisions)
        last = shifted[-1]
        shifted[-1] = DecisionRecord(
            *last.key,
            last.window_index,
            last.window_start_sample + 1,
            last.window_stop_sample + 1,
        )
        with self.assertRaisesRegex(ValueError, "冻结母索引"):
            evaluate_online_events(
                [segment], [], expected, shifted, mode=STATELESS_DIAGNOSTIC,
            )

    def test_clean_segment_shorter_than_one_window_is_kept_in_time(self) -> None:
        short = self._segment(400)
        result = evaluate_online_events(
            [short], [], [], [], mode=STATELESS_DIAGNOSTIC,
        )

        self.assertEqual(result["zero_window_segment_count"], 1)
        self.assertEqual(result["zero_window_segment_samples"], 400)
        self.assertEqual(result["trailing_unwindowed_samples"], 0)
        self.assertEqual(result["expected_window_count"], 0)
        self.assertEqual(result["valid_idle_seconds"], 1.6)

        # 只要长度足以形成窗口，空母索引就属于静默删窗，必须失败。
        with self.assertRaisesRegex(ValueError, "不得缺少冻结窗口"):
            evaluate_online_events(
                [self._segment(500)], [], [], [], mode=STATELESS_DIAGNOSTIC,
            )

    def test_stateful_and_stateless_modes_are_explicit_and_disjoint(self) -> None:
        segment = self._segment(750)
        expected = self._expected(segment)

        valid = self._stateful(expected, {
            0: (1, READY, WAIT_IDLE),
            1: (-1, WAIT_IDLE, READY),
        })
        result = evaluate_online_events(
            [segment], [], expected, valid, mode=STATEFUL_STRICT,
        )
        self.assertEqual(result["evaluation_mode"], STATEFUL_STRICT)
        self.assertEqual(result["idle_false_command_count"], 1)

        forbidden = self._stateful(expected, {
            0: (1, READY, WAIT_IDLE),
            1: (2, WAIT_IDLE, WAIT_IDLE),
            2: (-1, WAIT_IDLE, WAIT_IDLE),
        })
        with self.assertRaisesRegex(ValueError, "MI 指令只能"):
            evaluate_online_events(
                [segment], [], expected, forbidden, mode=STATEFUL_STRICT,
            )

        wrong_after = self._stateful(expected, {0: (1, READY, READY)})
        with self.assertRaisesRegex(ValueError, "进入 WAIT_IDLE"):
            evaluate_online_events(
                [segment], [], expected, wrong_after, mode=STATEFUL_STRICT,
            )

        discontinuous = self._stateful(expected, {
            0: (1, READY, WAIT_IDLE),
            1: (-1, READY, READY),
        })
        with self.assertRaisesRegex(ValueError, "状态不连续"):
            evaluate_online_events(
                [segment], [], expected, discontinuous, mode=STATEFUL_STRICT,
            )

        with self.assertRaisesRegex(ValueError, "不得混入状态字段"):
            evaluate_online_events(
                [segment], [], expected, valid, mode=STATELESS_DIAGNOSTIC,
            )
        with self.assertRaisesRegex(ValueError, "mode 必须显式选择"):
            evaluate_online_events(
                [segment], [], expected, self._stateless(expected), mode="auto",
            )

    def test_numpy_integer_inputs_produce_strict_json(self) -> None:
        segment = ScoringSegment(*(np.int64(value) for value in (1, 0, 0, 0, 0, 750)))
        expected = self._expected(segment)
        result = evaluate_online_events(
            [segment],
            [],
            expected,
            self._stateless(expected),
            mode=STATELESS_DIAGNOSTIC,
            sampling_rate=np.float64(250.0),
        )
        self.assertIsInstance(segment.subject_id, int)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_native_coordinates_and_clock_configuration_are_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "不得为负"):
            ScoringSegment(1, 0, 0, 0, -750, 0)

        segment = self._segment(750)
        expected = self._expected(segment)
        decisions = self._stateless(expected)
        with self.assertRaisesRegex(ValueError, "原生 250 Hz"):
            evaluate_online_events(
                [segment], [], expected, decisions,
                mode=STATELESS_DIAGNOSTIC, sampling_rate=128,
            )
        with self.assertRaisesRegex(ValueError, "500 点窗长"):
            evaluate_online_events(
                [segment], [], expected, decisions,
                mode=STATELESS_DIAGNOSTIC, window_samples=256, step_samples=64,
            )
        with self.assertRaisesRegex(ValueError, "固定为 0.5 秒"):
            evaluate_online_events(
                [segment], [], expected, decisions,
                mode=STATELESS_DIAGNOSTIC, min_overlap_seconds=1.0,
            )
        with self.assertRaisesRegex(TypeError, "必须为有限数"):
            evaluate_online_events(
                [segment], [], expected, decisions,
                mode=STATELESS_DIAGNOSTIC, sampling_rate="250",  # type: ignore[arg-type]
            )

    def test_overlapping_segments_are_rejected_before_idle_is_counted(self) -> None:
        overlapping = [
            ScoringSegment(1, 0, 0, 0, 0, 1000),
            ScoringSegment(1, 0, 0, 1, 900, 2000),
        ]
        with self.assertRaisesRegex(ValueError, "segment 不得重叠"):
            evaluate_online_events(
                overlapping, [], [], [], mode=STATELESS_DIAGNOSTIC,
            )


if __name__ == "__main__":
    unittest.main()
