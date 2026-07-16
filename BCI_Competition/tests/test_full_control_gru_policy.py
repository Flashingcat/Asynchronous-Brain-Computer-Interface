"""连续五分类 GRU 的标签、因果 token、损失和状态外壳手算测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from full_control_gru_policy import (  # noqa: E402
    HIDDEN_DIM,
    IGNORE_TARGET,
    ContinuousNormalizer,
    FullControlGRU,
    balanced_full_control_loss,
    build_continuous_targets,
    build_continuous_tokens,
    full_control_decisions,
)
from full_control_gru_diagnostics import diagnose_two_state_reset  # noqa: E402
from protocol_metrics import (  # noqa: E402
    NO_COMMAND,
    READY,
    STATEFUL_STRICT,
    WAIT_IDLE,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)


def windows(count: int, *, segment: int = 0, offset: int = 0) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment, index, offset + index * 125, offset + index * 125 + 500)
        for index in range(count)
    ]


class TokenAndTargetTests(unittest.TestCase):
    def test_token_deltas_use_only_previous_window_and_reset_at_segment(self) -> None:
        ws = [*windows(2), *windows(2, segment=1, offset=2000)]
        stage1 = np.asarray([[0, 1], [0, 3], [0, 9], [0, 8]], dtype=np.float32)
        stage2 = np.asarray([
            [1, 0, 0, 0],
            [3, 0, 0, 0],
            [8, 0, 0, 0],
            [7, 0, 0, 0],
        ], dtype=np.float32)
        token = build_continuous_tokens(ws, stage1, stage2)
        self.assertEqual(token.shape, (4, 10))
        self.assertEqual(token[1, 1], 2.0)
        np.testing.assert_allclose(token[2, [1, 6, 7, 8, 9]], 0.0)
        self.assertEqual(token[3, 1], -1.0)

    def test_targets_follow_formal_margin_and_ignore_both_mi_boundaries(self) -> None:
        ws = [
            ExpectedWindow(1, 0, 0, 0, 0, 0, 500),
            ExpectedWindow(1, 0, 0, 0, 1, 125, 625),
            ExpectedWindow(1, 0, 0, 0, 2, 1125, 1625),
            ExpectedWindow(1, 0, 0, 0, 3, 1500, 2000),
        ]
        event = MIEvent("e0", 1, 0, 0, 0, 500, 1500, 3)
        targets = build_continuous_targets(ws, [event])
        np.testing.assert_array_equal(targets, [0, 3, IGNORE_TARGET, 0])


class ModelAndLossTests(unittest.TestCase):
    def test_online_step_matches_batched_gru_output(self) -> None:
        torch.manual_seed(20260716)
        model = FullControlGRU()
        token = torch.randn(1, 5, 10)
        hidden_rows, logits = model(token)
        hidden = torch.zeros(1, 1, HIDDEN_DIM)
        for index in range(token.shape[1]):
            hidden, row_logits = model.step(token[:, index], hidden)
            torch.testing.assert_close(hidden[0], hidden_rows[:, index])
            torch.testing.assert_close(row_logits, logits[:, index])

    def test_balanced_zero_logit_loss_matches_hand_calculation(self) -> None:
        logits = torch.zeros(1, 5, 5, dtype=torch.float64)
        targets = torch.tensor([[0, 1, 2, 3, 4]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        loss, parts = balanced_full_control_loss(logits, targets, valid)
        self.assertAlmostEqual(float(parts["state_loss"]), math.log(2.0), places=12)
        self.assertAlmostEqual(float(parts["class_loss"]), math.log(4.0), places=12)
        self.assertAlmostEqual(float(loss), 0.5 * math.log(8.0), places=12)


class ScriptedModel:
    """用预定五分类 logit 验证状态外壳，不把模型质量混入状态测试。"""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.index = 0

    def eval(self):
        return self

    def step(self, token: torch.Tensor, hidden: torch.Tensor):
        logits = torch.tensor([self.rows[self.index]], dtype=token.dtype, device=token.device)
        self.index += 1
        return hidden + 1.0, logits


class OnlineShellTests(unittest.TestCase):
    def test_gru_controls_submit_class_and_reset_without_duplicate_command(self) -> None:
        ws = windows(5)
        stage1 = np.zeros((5, 2), dtype=np.float32)
        stage2 = np.zeros((5, 4), dtype=np.float32)
        # 第一维是 TASK logit；后四维是条件类别 logit。
        # IDLE -> 类2 -> 类2 -> IDLE -> 类1；WAIT_IDLE 内的第二个类2不得重复输出。
        model = ScriptedModel([
            [-8, 0, 0, 0, 0],
            [8, 0, 8, 0, 0],
            [8, 0, 8, 0, 0],
            [-8, 0, 0, 0, 0],
            [8, 8, 0, 0, 0],
        ])
        normalizer = ContinuousNormalizer(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32))
        decisions, traces = full_control_decisions(
            ws, stage1, stage2, model, normalizer,
            commit_threshold=0.8, reset_threshold=0.8, device=torch.device("cpu"),
        )
        self.assertEqual(
            [item.emitted_class for item in decisions],
            [NO_COMMAND, 2, NO_COMMAND, NO_COMMAND, 1],
        )
        self.assertEqual(
            [(item.decision_state_before, item.decision_state_after) for item in decisions],
            [
                (READY, READY),
                (READY, WAIT_IDLE),
                (WAIT_IDLE, WAIT_IDLE),
                (WAIT_IDLE, READY),
                (READY, WAIT_IDLE),
            ],
        )
        self.assertTrue(all(item.transition_reason is None for item in decisions))
        np.testing.assert_allclose([item.hidden[0] for item in traces], [1, 2, 3, 4, 5])
        evaluated = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 1000)],
            [MIEvent("e0", 1, 0, 0, 0, 500, 1000, 2)],
            ws,
            decisions,
            mode=STATEFUL_STRICT,
        )
        self.assertEqual(evaluated["correct_event_count"], 1)
        self.assertEqual(evaluated["additional_event_command_count"], 1)
        diagnostic = diagnose_two_state_reset(
            [ScoringSegment(1, 0, 0, 0, 0, 1000)],
            [MIEvent("e0", 1, 0, 0, 0, 500, 1000, 2)],
            ws,
            decisions,
            evaluated,
        )
        self.assertEqual(diagnostic["wait_idle"]["command_count"], 2)
        self.assertEqual(diagnostic["wait_idle"]["reset_count"], 1)
        self.assertEqual(diagnostic["wait_idle"]["segment_end_unresolved_count"], 1)
        self.assertAlmostEqual(
            diagnostic["wait_idle"]["completed_duration_seconds"]["mean"], 1.0,
        )
        self.assertAlmostEqual(
            diagnostic["reset_relative_to_matched_event_offset"]["seconds"]["mean"],
            -0.5,
        )
        self.assertEqual(
            diagnostic["idle_false_attribution"]["status"],
            "not_defined_for_candidate_free_gru_v1",
        )


if __name__ == "__main__":
    unittest.main()
