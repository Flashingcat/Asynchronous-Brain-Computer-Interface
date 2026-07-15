"""Fast-0/Fast-1 原子转换、慢通道回退与因果性的手算测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from candidate_state_policy import CandidateEvidence, candidate_transition  # noqa: E402
from fast_path_candidate_strategies import (  # noqa: E402
    FastPathConfig,
    fast_path_candidate_decisions,
)
from logit_candidate_strategies import (  # noqa: E402
    LogitStrategyConfig,
    logit_candidate_decisions,
)
from protocol_metrics import (  # noqa: E402
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    COMMAND_COMMIT,
    FAST0_COMMAND_COMMIT,
    FAST1_COMMAND_COMMIT,
    NO_COMMAND,
    READY,
    STATEFUL_CANDIDATE,
    TASK_CANDIDATE,
    WAIT_IDLE,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)


def make_windows(count: int, *, segment: int = 0, offset: int = 0) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment, index, offset + index * 125, offset + index * 125 + 500)
        for index in range(count)
    ]


def margin(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def stage1(probabilities: list[float]) -> np.ndarray:
    return np.asarray([[0.0, margin(value)] for value in probabilities], dtype=np.float64)


def base_config() -> dict:
    return {
        "strategy_id": "slow_anchor",
        "stage1_filter": "raw_margin",
        "stage1_alpha": None,
        "stage1_window": None,
        "task_on_probability": 0.5,
        "task_hold_probability": 0.3,
        "idle_reset_probability": 0.2,
        "stage1_drop_abort": None,
        "stage2_filter": "candidate_ewma_centered_logits",
        "stage2_alpha": 0.5,
        "stage2_min_candidate_windows": 2,
        "stage2_top_probability": 0.55,
        "stage2_probability_gap": 0.15,
        "stage2_stable_windows": 1,
        "stage2_max_probability_curvature": None,
        "max_candidate_windows": 8,
    }


def fast0_config() -> dict:
    return {
        "min_stage1_probability": 0.5,
        "min_stage1_delta": 0.0,
        "min_stage2_top_probability": 0.9,
        "min_stage2_probability_gap": 0.8,
    }


def fast1_config() -> dict:
    return {
        "min_stage1_probability": 0.5,
        "stage2_alpha": 0.5,
        "min_stage2_top_probability": 0.9,
        "min_stage2_probability_gap": 0.8,
        "require_same_raw_class": True,
    }


def make_config(*, fast0=None, fast1=None, identifier: str = "fast_test") -> FastPathConfig:
    return FastPathConfig.from_dict({
        "strategy_id": identifier,
        "base_logit_strategy": base_config(),
        "idle_reset_consecutive_windows": 1,
        "fast0": fast0,
        "fast1": fast1,
    })


class FastPathCandidateStrategyTests(unittest.TestCase):
    def test_fast0_commits_atomically_without_opening_candidate(self) -> None:
        windows = make_windows(2)
        result = fast_path_candidate_decisions(
            windows,
            stage1([0.9, 0.1]),
            np.asarray([[8.0, 0.0, 0.0, 0.0], [0.0] * 4]),
            make_config(fast0=fast0_config()),
        )
        first = result.policy.decisions[0]
        self.assertEqual(
            (first.decision_state_before, first.decision_state_after),
            (READY, WAIT_IDLE),
        )
        self.assertEqual(first.transition_reason, FAST0_COMMAND_COMMIT)
        self.assertEqual(first.emitted_class, 1)
        self.assertTrue(result.trace[0].fast0_pass)
        self.assertEqual(result.trace[0].slow_candidate_window_count, 0)

        metrics = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 625)],
            [MIEvent("e", 1, 0, 0, 0, 0, 625, 1)],
            windows,
            result.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        self.assertEqual(metrics["correct_event_count"], 1)
        self.assertEqual(metrics["candidate_diagnostics"]["candidate_open_count"], 0)

    def test_fast1_uses_open_and_next_window_then_closes_candidate(self) -> None:
        windows = make_windows(3)
        result = fast_path_candidate_decisions(
            windows,
            stage1([0.9, 0.9, 0.9]),
            np.asarray([
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 7.0, 0.0, 0.0],
                [0.0] * 4,
            ]),
            make_config(fast1=fast1_config()),
        )
        self.assertEqual(
            [item.transition_reason for item in result.policy.decisions],
            [CANDIDATE_OPEN, FAST1_COMMAND_COMMIT, None],
        )
        self.assertEqual(result.policy.decisions[1].emitted_class, 2)
        self.assertTrue(result.trace[1].fast1_evaluated)
        self.assertTrue(result.trace[1].fast1_same_raw_class)
        self.assertTrue(result.trace[1].fast1_pass)
        self.assertEqual(result.trace[1].slow_candidate_window_count, 1)

        metrics = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 750)],
            [],
            windows,
            result.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        candidate = metrics["candidate_diagnostics"]
        self.assertEqual(candidate["candidate_command_count"], 1)
        self.assertEqual(candidate["candidate_intervals"][0]["outcome"], FAST1_COMMAND_COMMIT)

    def test_failed_fast1_does_not_pollute_slow_fallback(self) -> None:
        windows = make_windows(3)
        logits = np.asarray([
            [8.0, 0.0, 0.0, 0.0],  # 开门窗支持类 1，仅进 Fast-1 缓存。
            [0.0, 8.0, 0.0, 0.0],  # 与开门窗不一致，Fast-1 失败。
            [0.0, 8.0, 0.0, 0.0],  # 慢通道只聚合后两窗并提交类 2。
        ])
        result = fast_path_candidate_decisions(
            windows,
            stage1([0.9, 0.9, 0.9]),
            logits,
            make_config(fast1=fast1_config()),
        )
        self.assertEqual(
            [item.transition_reason for item in result.policy.decisions],
            [CANDIDATE_OPEN, None, COMMAND_COMMIT],
        )
        self.assertFalse(result.trace[1].fast1_pass)
        self.assertFalse(result.trace[2].fast1_evaluated)
        self.assertEqual(result.trace[2].slow_candidate_window_count, 2)
        self.assertEqual(result.policy.decisions[2].emitted_class, 2)

    def test_disabled_fast_paths_are_bit_exact_with_slow_anchor_policy(self) -> None:
        windows = make_windows(7)
        s1 = stage1([0.1, 0.9, 0.8, 0.8, 0.1, 0.1, 0.9])
        s2 = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
            [0.0, 7.0, 0.0, 0.0],
            [0.0] * 4,
            [0.0] * 4,
            [0.0, 0.0, 8.0, 0.0],
        ])
        fast = fast_path_candidate_decisions(windows, s1, s2, make_config())
        slow = logit_candidate_decisions(
            windows,
            s1,
            s2,
            LogitStrategyConfig.from_dict(base_config()),
            idle_reset_consecutive_windows=1,
        )
        self.assertEqual(fast.policy, slow.policy)
        self.assertEqual(
            [item.slow_candidate_window_count for item in fast.trace],
            [item.stage2_candidate_window_count for item in slow.trace],
        )

    def test_future_suffix_and_common_logit_shift_do_not_change_prefix(self) -> None:
        windows = make_windows(5)
        s1 = stage1([0.9] * 5)
        s2 = np.asarray([
            [8.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
        ])
        config = make_config(fast0=fast0_config(), fast1=fast1_config())
        original = fast_path_candidate_decisions(windows, s1, s2, config)
        shifted = fast_path_candidate_decisions(
            windows,
            s1 + 11.0,
            s2 + np.arange(5)[:, None] * 13.0,
            config,
        )
        self.assertEqual(original.policy, shifted.policy)
        self.assertEqual(
            [item.proposed_commit_path for item in original.trace],
            [item.proposed_commit_path for item in shifted.trace],
        )
        np.testing.assert_allclose(
            [item.stage1_filtered_task_probability for item in original.trace],
            [item.stage1_filtered_task_probability for item in shifted.trace],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            [item.raw_stage2_top_probability for item in original.trace],
            [item.raw_stage2_top_probability for item in shifted.trace],
            rtol=0.0,
            atol=1e-14,
        )

        changed = s2.copy()
        changed[3:] = np.asarray([[50.0, -50.0, 0.0, 0.0], [-50.0, 50.0, 0.0, 0.0]])
        future = fast_path_candidate_decisions(windows, s1, changed, config)
        self.assertEqual(original.policy.decisions[:3], future.policy.decisions[:3])
        self.assertEqual(original.trace[:3], future.trace[:3])

    def test_fast_priority_and_invalid_state_evidence_are_explicit(self) -> None:
        aborted = candidate_transition(
            TASK_CANDIDATE,
            0,
            CandidateEvidence(False, False, 2, False, 3),
            max_candidate_windows=2,
        )
        self.assertEqual(aborted.transition_reason, CANDIDATE_ABORT_STAGE1)
        self.assertEqual(aborted.emitted_class, NO_COMMAND)
        with self.assertRaisesRegex(ValueError, "同时满足 Stage 1"):
            candidate_transition(
                READY,
                0,
                CandidateEvidence(False, True, NO_COMMAND, False, 1),
                max_candidate_windows=2,
            )
        with self.assertRaisesRegex(ValueError, "WAIT_IDLE"):
            candidate_transition(
                WAIT_IDLE,
                0,
                CandidateEvidence(False, True, NO_COMMAND, False, 1),
                max_candidate_windows=2,
            )

    def test_configuration_rejects_hidden_or_weaker_gate_values(self) -> None:
        payload = {
            "strategy_id": "bad",
            "base_logit_strategy": base_config(),
            "idle_reset_consecutive_windows": 1,
            "fast0": {**fast0_config(), "min_stage1_probability": 0.4},
            "fast1": None,
        }
        with self.assertRaisesRegex(ValueError, "不得低于"):
            FastPathConfig.from_dict(payload)
        payload["fast0"] = {**fast0_config(), "unexpected": 1}
        with self.assertRaisesRegex(ValueError, "schema"):
            FastPathConfig.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
