"""LD-GRU-v1 的反事实监督、集合式损失和因果状态回放手算测试。"""

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

from ld_gru_policy import (  # noqa: E402
    CandidateSequence,
    TinyLDGRU,
    TokenNormalizer,
    build_candidate_inventory,
    build_flow_inputs,
    ld_gru_decisions,
    model_parameter_counts,
    set_valued_commit_loss,
)
from logit_candidate_strategies import LogitStrategyConfig, logit_candidate_decisions  # noqa: E402
from protocol_metrics import (  # noqa: E402
    CANDIDATE_ABORT_STAGE1,
    LEARNED_GRU_COMMIT,
    NO_COMMAND,
    STATEFUL_CANDIDATE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    classify_counterfactual_first_commit,
    evaluate_online_events,
)


def windows(count: int) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
        for index in range(count)
    ]


def stage1_probabilities(probabilities: list[float]) -> np.ndarray:
    margins = [math.log(value / (1.0 - value)) for value in probabilities]
    return np.asarray([[0.0, value] for value in margins], dtype=np.float32)


def anchor_config() -> LogitStrategyConfig:
    return LogitStrategyConfig.from_dict({
        "strategy_id": "ld_gru_test_anchor",
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
    })


class CounterfactualCommitTests(unittest.TestCase):
    def test_public_helper_matches_single_command_evaluator_outcomes(self) -> None:
        ws = windows(8)
        event = MIEvent("event0", 1, 0, 0, 0, 500, 1250, 2)
        self.assertEqual(
            classify_counterfactual_first_commit(ws[0], 2, [event]).outcome,
            "too_early",
        )
        self.assertEqual(
            classify_counterfactual_first_commit(ws[1], 2, [event]).outcome,
            "correct",
        )
        self.assertEqual(
            classify_counterfactual_first_commit(ws[1], 1, [event]).outcome,
            "wrong_class",
        )
        self.assertEqual(
            classify_counterfactual_first_commit(
                ws[1], 2, [event], used_event_ids=frozenset({"event0"}),
            ).outcome,
            "additional_event",
        )
        self.assertEqual(
            classify_counterfactual_first_commit(ws[7], 2, [event]).outcome,
            "idle_false",
        )

        for index, class_id, expected in ((1, 2, "correct"), (1, 1, "wrong_class"), (0, 2, "too_early"), (7, 2, "idle_false")):
            decisions = [
                DecisionRecord(
                    *window.key, window.window_index,
                    window.window_start_sample, window.window_stop_sample,
                    class_id if position == index else NO_COMMAND,
                )
                for position, window in enumerate(ws)
            ]
            evaluated = evaluate_online_events(
                [ScoringSegment(1, 0, 0, 0, 0, 1375)],
                [event], ws, decisions, mode="stateless_diagnostic",
            )
            if expected in {"correct", "wrong_class"}:
                self.assertEqual(evaluated["event_matches"][0]["outcome"], expected)
            elif expected == "too_early":
                self.assertEqual(evaluated["too_early_command_count"], 1)
            else:
                self.assertEqual(evaluated["idle_false_command_count"], 1)


class CandidateAndTokenTests(unittest.TestCase):
    def test_flow_stage1_trace_matches_existing_anchor_math(self) -> None:
        ws = windows(6)
        stage1 = stage1_probabilities([0.8, 0.7, 0.4, 0.9, 0.6, 0.2])
        stage2 = np.arange(24, dtype=np.float32).reshape(6, 4)
        flow = build_flow_inputs(ws, stage1, stage2)
        anchor = logit_candidate_decisions(ws, stage1, stage2, anchor_config())
        np.testing.assert_allclose(
            flow.task_probability,
            [item.stage1_filtered_task_probability for item in anchor.trace],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            flow.task_probability_delta,
            [item.stage1_filtered_delta for item in anchor.trace],
            rtol=1e-6,
        )
        np.testing.assert_allclose(flow.centered_stage2.sum(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(flow.centered_stage2_delta[0], 0.0)

    def test_candidate_contains_opening_but_only_later_windows_can_be_correct(self) -> None:
        ws = windows(10)
        stage1 = stage1_probabilities([0.95] * 10)
        stage2 = np.zeros((10, 4), dtype=np.float32)
        stage2[:, 1] = 3.0
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1500, 2)
        flow = build_flow_inputs(ws, stage1, stage2)
        inventory = build_candidate_inventory(ws, [event], flow)
        first = inventory.candidates[0]
        self.assertEqual(len(first.window_positions), 9)
        self.assertEqual(first.tokens[0, -1], 0.0)
        self.assertEqual(first.tokens[-1, -1], 1.0)
        self.assertFalse(np.any(first.correct_mask[0]))
        self.assertTrue(np.any(first.correct_mask[1:, 1]))
        self.assertFalse(np.any(first.correct_mask[:, [0, 2, 3]]))


class ModelAndLossTests(unittest.TestCase):
    def test_model_has_exact_parameter_count_and_zero_residual_initialization(self) -> None:
        residual = TinyLDGRU("stop_residual")
        stop_only = TinyLDGRU("stop_only")
        self.assertEqual(model_parameter_counts(residual), (573, 573))
        self.assertEqual(model_parameter_counts(stop_only), (573, 537))
        self.assertTrue(torch.equal(residual.class_correction.weight, torch.zeros_like(residual.class_correction.weight)))
        self.assertTrue(torch.equal(residual.class_correction.bias, torch.zeros_like(residual.class_correction.bias)))
        self.assertAlmostEqual(
            float(torch.sigmoid(residual.stop_head.bias.detach())[0]), 0.04742587, places=6,
        )

    def test_full_token_mode_is_exactly_the_legacy_default(self) -> None:
        """显式 full 必须与旧默认路径逐元素一致，避免消融改写原基线。"""
        torch.manual_seed(20260716)
        legacy = TinyLDGRU("stop_residual")
        explicit = TinyLDGRU("stop_residual", token_mode="full")
        explicit.load_state_dict(legacy.state_dict())
        tokens = torch.randn(3, 5, 12)
        centered = torch.randn(3, 5, 4)
        valid = torch.tensor([
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ])
        for legacy_value, explicit_value in zip(
            legacy(tokens, centered, valid),
            explicit(tokens, centered, valid),
        ):
            torch.testing.assert_close(legacy_value, explicit_value, rtol=0.0, atol=0.0)

    def test_mask_stage1_removes_only_the_first_three_token_dimensions(self) -> None:
        """屏蔽组应对 Stage 1 三维不变，但仍能感知 Stage 2 和候选年龄。"""
        torch.manual_seed(20260716)
        model = TinyLDGRU("stop_residual", token_mode="mask_stage1")
        base = torch.randn(2, 4, 12)
        changed_stage1 = base.clone()
        changed_stage1[..., :3] += torch.randn_like(changed_stage1[..., :3]) * 100.0
        changed_stage2 = base.clone()
        changed_stage2[..., 3] += 1.0
        centered = torch.randn(2, 4, 4)
        valid = torch.ones(2, 4, dtype=torch.bool)

        reference = model(base, centered, valid)
        masked_change = model(changed_stage1, centered, valid)
        stage2_change = model(changed_stage2, centered, valid)
        for reference_value, masked_value in zip(reference, masked_change):
            torch.testing.assert_close(reference_value, masked_value, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(reference[0], stage2_change[0]))

    def test_mask_stage1_step_matches_batched_forward(self) -> None:
        """在线逐窗接口与训练批接口必须使用同一屏蔽规则。"""
        torch.manual_seed(20260716)
        model = TinyLDGRU("stop_residual", token_mode="mask_stage1")
        tokens = torch.randn(1, 4, 12)
        centered = torch.randn(1, 4, 4)
        valid = torch.ones(1, 4, dtype=torch.bool)
        batch_hidden, batch_stop, batch_correction, batch_class = model(
            tokens, centered, valid,
        )

        hidden = torch.zeros(1, batch_hidden.shape[-1])
        for index in range(tokens.shape[1]):
            hidden, stop, correction, class_logits = model.step(
                tokens[:, index], centered[:, index], hidden,
            )
            torch.testing.assert_close(hidden, batch_hidden[:, index])
            torch.testing.assert_close(stop, batch_stop[:, index])
            torch.testing.assert_close(correction, batch_correction[:, index])
            torch.testing.assert_close(class_logits, batch_class[:, index])

    def test_set_valued_loss_matches_two_candidate_hand_calculation(self) -> None:
        stop = torch.zeros((2, 2), dtype=torch.float64)
        classes = torch.zeros((2, 2, 4), dtype=torch.float64)
        valid = torch.ones((2, 2), dtype=torch.bool)
        correct = torch.zeros((2, 2, 4), dtype=torch.bool)
        correct[0, 1, 0] = True
        loss, parts = set_valued_commit_loss(stop, classes, valid, correct)
        positive = -math.log(0.5 * 0.25)
        negative = -math.log(0.5)
        self.assertAlmostEqual(float(parts["positive_mean"]), positive)
        self.assertAlmostEqual(float(parts["negative_mean"]), negative)
        self.assertAlmostEqual(float(loss), 0.5 * positive + 0.5 * negative)


class OnlineReplayTests(unittest.TestCase):
    @staticmethod
    def always_submit_model() -> TinyLDGRU:
        model = TinyLDGRU("stop_only")
        with torch.no_grad():
            for parameter in model.gru.parameters():
                parameter.zero_()
            model.stop_head.weight.zero_()
            model.stop_head.bias.fill_(10.0)
        return model

    def test_opening_window_never_commits_and_next_window_uses_learned_reason(self) -> None:
        ws = windows(4)
        stage1 = stage1_probabilities([0.95, 0.95, 0.05, 0.05])
        stage2 = np.zeros((4, 4), dtype=np.float32)
        stage2[:, 2] = 5.0
        normalizer = TokenNormalizer(np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32))
        replay = ld_gru_decisions(
            ws, stage1, stage2, self.always_submit_model(), normalizer,
            0.5, torch.device("cpu"),
        )
        self.assertEqual(replay.decisions[0].emitted_class, NO_COMMAND)
        self.assertEqual(replay.trace[0].stop_score, 0.0)
        self.assertEqual(replay.decisions[1].emitted_class, 3)
        self.assertEqual(replay.decisions[1].transition_reason, LEARNED_GRU_COMMIT)
        for row in replay.trace:
            self.assertTrue(np.isfinite(row.raw_token).all())
            self.assertTrue(np.isfinite(row.normalized_token).all())
            self.assertTrue(np.isfinite(row.hidden).all())
            self.assertTrue(np.isfinite(row.centered_stage2).all())
            self.assertTrue(np.isfinite(row.class_correction).all())
            self.assertTrue(np.isfinite(row.final_class_logits).all())
            self.assertTrue(math.isfinite(row.stop_logit))
            self.assertTrue(math.isfinite(row.stop_score))
        self.assertEqual(replay.trace[2].candidate_age, -1)
        self.assertFalse(replay.trace[2].gru_consumed)
        np.testing.assert_allclose(replay.trace[2].hidden, 0.0)
        np.testing.assert_allclose(
            replay.trace[2].final_class_logits, replay.trace[2].centered_stage2,
        )
        evaluated = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 875)],
            [MIEvent("event0", 1, 0, 0, 0, 0, 875, 3)],
            ws, replay.decisions, mode=STATEFUL_CANDIDATE,
        )
        self.assertEqual(evaluated["correct_event_count"], 1)

    def test_stage1_drop_abort_precedes_high_gru_stop_score(self) -> None:
        ws = windows(3)
        stage1 = np.asarray([[0.0, 8.0], [0.0, -12.0], [0.0, -12.0]], dtype=np.float32)
        stage2 = np.zeros((3, 4), dtype=np.float32)
        normalizer = TokenNormalizer(np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32))
        replay = ld_gru_decisions(
            ws, stage1, stage2, self.always_submit_model(), normalizer,
            0.05, torch.device("cpu"),
        )
        self.assertEqual(replay.decisions[1].transition_reason, CANDIDATE_ABORT_STAGE1)
        self.assertEqual(replay.decisions[1].emitted_class, NO_COMMAND)
        self.assertFalse(replay.trace[1].gru_consumed)


if __name__ == "__main__":
    unittest.main()
