"""隐藏特征候选门控的手算、重置、因果性和参考兼容测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from feature_candidate_strategies import (  # noqa: E402
    FEATURE_DIM,
    FeatureStrategyConfig,
    feature_candidate_decisions,
)
from logit_candidate_strategies import (  # noqa: E402
    LogitStrategyConfig,
    logit_candidate_decisions,
)
from protocol_metrics import (  # noqa: E402
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    COMMAND_COMMIT,
    NO_COMMAND,
    TASK_CANDIDATE,
    ExpectedWindow,
)


def base_payload() -> dict:
    return {
        "strategy_id": "dual_ewma_drop_abort",
        "stage1_filter": "ewma_margin",
        "stage1_alpha": 0.5,
        "stage1_window": None,
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


def config(metric: str = "unit_velocity_l2", threshold: float | None = 0.5,
           consecutive: int = 1) -> FeatureStrategyConfig:
    return FeatureStrategyConfig.from_dict({
        "strategy_id": "unit_test",
        "base_logit_strategy": base_payload(),
        "feature_metric": metric,
        "feature_max_change": threshold,
        "feature_required_consecutive": consecutive,
    })


def windows(count: int, segment: int = 0, offset: int = 0) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment, index, offset + 125 * index, offset + 125 * index + 500)
        for index in range(count)
    ]


def confident_logits(count: int) -> tuple[np.ndarray, np.ndarray]:
    stage1 = np.tile(np.asarray([[0.0, 4.0]]), (count, 1))
    stage2 = np.tile(np.asarray([[5.0, 0.0, 0.0, 0.0]]), (count, 1))
    return stage1, stage2


def features(vectors: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros((len(vectors), FEATURE_DIM), dtype=np.float64)
    for index, (first, second) in enumerate(vectors):
        result[index, :2] = (first, second)
    return result


class FeatureGateTests(unittest.TestCase):
    def test_opening_window_is_excluded_and_velocity_gates_commit(self) -> None:
        ws = windows(4)
        stage1, stage2 = confident_logits(4)
        hidden = features([(1, 0), (1, 0), (0, 1), (0, 1)])
        result = feature_candidate_decisions(ws, stage1, stage2, hidden, config())

        self.assertEqual(result.policy.decisions[0].transition_reason, CANDIDATE_OPEN)
        self.assertEqual(result.trace[0].stage2_candidate_window_count, 0)
        self.assertFalse(result.trace[1].feature_metric_available)
        self.assertEqual(result.trace[2].base_logit_commit_class, 1)
        self.assertGreater(result.trace[2].feature_metric_value, 0.5)
        self.assertEqual(result.policy.decisions[2].emitted_class, NO_COMMAND)
        self.assertEqual(result.policy.decisions[3].transition_reason, COMMAND_COMMIT)
        self.assertEqual(result.policy.decisions[3].emitted_class, 1)

    def test_stage1_drop_abort_has_priority_over_feature_pass(self) -> None:
        ws = windows(3)
        stage1 = np.asarray([[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]])
        stage2 = confident_logits(3)[1]
        hidden = features([(1, 0), (1, 0), (1, 0)])
        result = feature_candidate_decisions(ws, stage1, stage2, hidden, config(threshold=1.0))

        self.assertEqual(result.trace[2].base_logit_commit_class, 1)
        self.assertTrue(result.trace[2].feature_pass)
        self.assertEqual(result.policy.decisions[2].transition_reason, CANDIDATE_ABORT_STAGE1)
        self.assertEqual(result.policy.decisions[2].emitted_class, NO_COMMAND)

    def test_segment_reset_prevents_cross_segment_feature_history(self) -> None:
        ws = windows(2) + windows(2, segment=1, offset=2000)
        stage1, stage2 = confident_logits(4)
        hidden = features([(1, 0), (1, 0), (0, 1), (0, 1)])
        result = feature_candidate_decisions(ws, stage1, stage2, hidden, config())

        self.assertEqual(result.policy.decisions[2].transition_reason, CANDIDATE_OPEN)
        self.assertEqual(result.policy.decisions[2].decision_state_after, TASK_CANDIDATE)
        self.assertFalse(result.trace[3].feature_metric_available)

    def test_none_gate_exactly_matches_logit_only_policy(self) -> None:
        ws = windows(20)
        rng = np.random.default_rng(17)
        stage1 = rng.normal(size=(20, 2))
        stage2 = rng.normal(size=(20, 4))
        hidden = rng.normal(size=(20, FEATURE_DIM))
        reference = logit_candidate_decisions(
            ws, stage1, stage2, LogitStrategyConfig.from_dict(base_payload()),
        )
        gated = feature_candidate_decisions(
            ws, stage1, stage2, hidden, config("none", None, 0),
        )
        self.assertEqual(gated.policy, reference.policy)

    def test_future_suffix_cannot_change_prefix(self) -> None:
        ws = windows(8)
        stage1, stage2 = confident_logits(8)
        hidden = features([(1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1)])
        original = feature_candidate_decisions(ws, stage1, stage2, hidden, config())
        changed1, changed2, changed_hidden = stage1.copy(), stage2.copy(), hidden.copy()
        changed1[5:] *= -100
        changed2[5:] *= -100
        changed_hidden[5:] = np.random.default_rng(3).normal(size=(3, FEATURE_DIM))
        changed = feature_candidate_decisions(ws, changed1, changed2, changed_hidden, config())
        self.assertEqual(original.policy.decisions[:5], changed.policy.decisions[:5])
        self.assertEqual(original.trace[:5], changed.trace[:5])

    def test_prototype_and_acceleration_match_hand_values(self) -> None:
        ws = windows(4)
        stage1, stage2 = confident_logits(4)
        # 前两个候选窗保持四类平局，避免在收集到第三个特征前提前提交。
        stage2[1:3] = 0.0
        hidden = features([(1, 0), (1, 0), (0, 1), (-1, 0)])
        prototype = feature_candidate_decisions(
            ws, stage1, stage2, hidden,
            config("unit_prototype_cosine_distance", 2.0, 1),
        )
        acceleration = feature_candidate_decisions(
            ws, stage1, stage2, hidden,
            config("unit_acceleration_l2", 3.0, 1),
        )
        self.assertAlmostEqual(prototype.trace[2].feature_metric_value, 1.0)
        self.assertAlmostEqual(prototype.trace[3].feature_metric_value, 1.0 + 2 ** -0.5)
        self.assertFalse(acceleration.trace[2].feature_metric_available)
        self.assertAlmostEqual(acceleration.trace[3].feature_metric_value, 2.0)


class FeatureConfigTests(unittest.TestCase):
    def test_invalid_gate_contract_and_zero_feature_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "阈值必须为空"):
            config("none", 0.5, 0)
        with self.assertRaisesRegex(ValueError, "连续通过窗数"):
            config("unit_velocity_l2", 0.5, 0)
        stage1, stage2 = confident_logits(2)
        with self.assertRaisesRegex(ValueError, "非零 240 维"):
            feature_candidate_decisions(
                windows(2), stage1, stage2, np.zeros((2, FEATURE_DIM)), config(),
            )
        extreme = np.full((2, FEATURE_DIM), 1e308)
        with self.assertRaisesRegex(ValueError, "非零 240 维"):
            feature_candidate_decisions(windows(2), stage1, stage2, extreme, config())


if __name__ == "__main__":
    unittest.main()
