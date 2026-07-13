"""三状态候选决策内核、严格轨迹合同和候选区间指标的手算测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from candidate_state_policy import (  # noqa: E402
    CandidateEvidence,
    candidate_state_decisions,
)
from protocol_metrics import (  # noqa: E402
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    CANDIDATE_TIMEOUT,
    COMMAND_COMMIT,
    IDLE_RESET,
    NO_COMMAND,
    READY,
    STATEFUL_CANDIDATE,
    STATEFUL_STRICT,
    TASK_CANDIDATE,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)


def make_windows(
    count: int,
    *,
    segment_id: int = 0,
    offset: int = 0,
) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment_id, index, offset + index * 125, offset + index * 125 + 500)
        for index in range(count)
    ]


class CandidateStatePolicyTests(unittest.TestCase):
    @staticmethod
    def hand_evidence() -> list[CandidateEvidence]:
        return [
            CandidateEvidence(False, False, NO_COMMAND, False),  # READY 保持
            CandidateEvidence(True, False, 2, False),            # 开门；本窗 Stage 2 不可提交
            CandidateEvidence(False, True, NO_COMMAND, False),   # 候选第 1 窗
            CandidateEvidence(False, False, 2, False),           # Stage 1 撤销优先于类别提交
            CandidateEvidence(True, False, NO_COMMAND, False),   # 第二次开门
            CandidateEvidence(False, True, 3, False),            # 提交 Feet
            CandidateEvidence(True, True, 1, False),             # WAIT_IDLE 忽略新类别
            CandidateEvidence(True, True, 4, True),              # 只复位，不在同窗重新开门
            CandidateEvidence(True, False, NO_COMMAND, False),   # 第三次开门
            CandidateEvidence(False, True, NO_COMMAND, False),   # 候选第 1 窗
            CandidateEvidence(False, True, NO_COMMAND, False),   # 候选第 2 窗后超时
        ]

    def test_hand_trace_freezes_priorities_and_timeout(self) -> None:
        result = candidate_state_decisions(
            make_windows(11),
            self.hand_evidence(),
            max_candidate_windows=2,
        )
        decisions = result.decisions
        self.assertEqual([item.emitted_class for item in decisions], [
            -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1,
        ])
        self.assertEqual([item.transition_reason for item in decisions], [
            None,
            CANDIDATE_OPEN,
            None,
            CANDIDATE_ABORT_STAGE1,
            CANDIDATE_OPEN,
            COMMAND_COMMIT,
            None,
            IDLE_RESET,
            CANDIDATE_OPEN,
            None,
            CANDIDATE_TIMEOUT,
        ])
        self.assertEqual(
            (decisions[1].decision_state_before, decisions[1].decision_state_after),
            (READY, TASK_CANDIDATE),
        )
        self.assertEqual(
            (decisions[5].decision_state_before, decisions[5].decision_state_after),
            (TASK_CANDIDATE, WAIT_IDLE),
        )
        self.assertEqual(
            (decisions[7].decision_state_before, decisions[7].decision_state_after),
            (WAIT_IDLE, READY),
        )
        self.assertEqual(
            [result.trace[index].candidate_windows_after for index in (1, 2, 3, 8, 9, 10)],
            [0, 1, 0, 0, 1, 0],
        )

    def test_candidate_metrics_match_hand_calculation(self) -> None:
        windows = make_windows(11)
        decisions = candidate_state_decisions(
            windows,
            self.hand_evidence(),
            max_candidate_windows=2,
        ).decisions
        metrics = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 1750)],
            [
                MIEvent("e1", 1, 0, 0, 0, 500, 1250, 3),
                MIEvent("e2", 1, 0, 0, 0, 1250, 1750, 1),
            ],
            windows,
            decisions,
            mode=STATEFUL_CANDIDATE,
        )
        candidate = metrics["candidate_diagnostics"]
        self.assertEqual(metrics["correct_event_count"], 1)
        self.assertEqual(metrics["miss_rate"], 0.5)
        self.assertEqual(candidate["candidate_open_count"], 3)
        self.assertTrue(math.isclose(candidate["candidate_opens_per_valid_minute"], 180 / 7))
        self.assertEqual(candidate["candidate_command_count"], 1)
        self.assertEqual(candidate["candidate_conversion_rate"], 1 / 3)
        self.assertEqual(candidate["candidate_abort_count"], 2)
        self.assertEqual(candidate["candidate_abort_rate"], 2 / 3)
        self.assertEqual(candidate["candidate_stage1_abort_count"], 1)
        self.assertEqual(candidate["candidate_timeout_count"], 1)
        self.assertEqual(candidate["candidate_unresolved_count"], 0)
        self.assertEqual(candidate["completed_candidate_dwell_seconds"]["count"], 3)
        self.assertTrue(math.isclose(
            candidate["completed_candidate_dwell_seconds"]["mean"],
            5 / 6,
        ))
        self.assertEqual(candidate["completed_candidate_dwell_seconds"]["median"], 1.0)
        self.assertEqual(candidate["miss_event_with_candidate_timeout_count"], 1)
        self.assertEqual(candidate["miss_event_with_candidate_timeout_events"], [{
            "subject_id": 1,
            "session_id": 0,
            "run_id": 0,
            "segment_id": 0,
            "event_id": "e2",
        }])
        self.assertEqual(
            [item["outcome"] for item in candidate["candidate_intervals"]],
            [CANDIDATE_ABORT_STAGE1, COMMAND_COMMIT, CANDIDATE_TIMEOUT],
        )

    def test_segment_boundary_resets_state_and_reports_unresolved_candidate(self) -> None:
        windows = make_windows(2) + make_windows(1, segment_id=1, offset=1000)
        evidence = [
            CandidateEvidence(True, False, NO_COMMAND, False),
            CandidateEvidence(False, True, NO_COMMAND, False),
            CandidateEvidence(False, False, NO_COMMAND, False),
        ]
        result = candidate_state_decisions(windows, evidence, max_candidate_windows=3)
        self.assertEqual(result.decisions[2].decision_state_before, READY)
        metrics = evaluate_online_events(
            [
                ScoringSegment(1, 0, 0, 0, 0, 625),
                ScoringSegment(1, 0, 0, 1, 1000, 1500),
            ],
            [],
            windows,
            result.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        candidate = metrics["candidate_diagnostics"]
        self.assertEqual(candidate["candidate_open_count"], 1)
        self.assertEqual(candidate["candidate_unresolved_count"], 1)
        self.assertEqual(candidate["completed_candidate_dwell_seconds"]["count"], 0)
        self.assertEqual(candidate["candidate_intervals"][0]["duration_seconds"], 0.5)

    def test_invalid_evidence_limits_and_window_order_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "布尔证据"):
            CandidateEvidence(1, True, NO_COMMAND, False)
        with self.assertRaisesRegex(ValueError, "-1 或 1..4"):
            CandidateEvidence(True, True, 0, False)
        with self.assertRaisesRegex(ValueError, "正整数"):
            candidate_state_decisions([], [], max_candidate_windows=True)
        with self.assertRaisesRegex(ValueError, "证据数量"):
            candidate_state_decisions(make_windows(1), [], max_candidate_windows=1)
        with self.assertRaisesRegex(ValueError, "必须连续"):
            candidate_state_decisions(
                [
                    ExpectedWindow(1, 0, 0, 0, 0, 0, 500),
                    ExpectedWindow(1, 0, 0, 0, 2, 250, 750),
                ],
                [
                    CandidateEvidence(False, False, NO_COMMAND, False),
                    CandidateEvidence(False, False, NO_COMMAND, False),
                ],
                max_candidate_windows=1,
            )
        with self.assertRaisesRegex(ValueError, "window_stop_sample"):
            candidate_state_decisions(
                [ExpectedWindow(1, 0, 0, 0, 0, 500, 500)],
                [CandidateEvidence(False, False, NO_COMMAND, False)],
                max_candidate_windows=1,
            )
        with self.assertRaisesRegex(ValueError, "起止时间必须严格递增"):
            candidate_state_decisions(
                [
                    ExpectedWindow(1, 0, 0, 0, 0, 125, 625),
                    ExpectedWindow(1, 0, 0, 0, 1, 0, 500),
                ],
                [
                    CandidateEvidence(False, False, NO_COMMAND, False),
                    CandidateEvidence(False, False, NO_COMMAND, False),
                ],
                max_candidate_windows=1,
            )
        with self.assertRaisesRegex(ValueError, "后续 segment"):
            candidate_state_decisions(
                [
                    ExpectedWindow(1, 0, 0, 0, 0, 1000, 1500),
                    ExpectedWindow(1, 0, 0, 1, 0, 0, 500),
                ],
                [
                    CandidateEvidence(False, False, NO_COMMAND, False),
                    CandidateEvidence(False, False, NO_COMMAND, False),
                ],
                max_candidate_windows=1,
            )

    def test_commit_on_last_allowed_candidate_window_precedes_timeout(self) -> None:
        windows = make_windows(3)
        evidence = [
            CandidateEvidence(True, False, NO_COMMAND, False),
            CandidateEvidence(False, True, NO_COMMAND, False),
            CandidateEvidence(False, True, 4, False),
        ]
        result = candidate_state_decisions(windows, evidence, max_candidate_windows=2)
        self.assertEqual(result.decisions[-1].emitted_class, 4)
        self.assertEqual(result.decisions[-1].transition_reason, COMMAND_COMMIT)
        self.assertEqual(result.decisions[-1].decision_state_after, WAIT_IDLE)

    def test_future_evidence_cannot_change_earlier_trace(self) -> None:
        windows = make_windows(6)
        original = [
            CandidateEvidence(True, False, NO_COMMAND, False),
            CandidateEvidence(False, True, NO_COMMAND, False),
            CandidateEvidence(False, True, 2, False),
            CandidateEvidence(False, False, NO_COMMAND, False),
            CandidateEvidence(False, False, NO_COMMAND, True),
            CandidateEvidence(False, False, NO_COMMAND, False),
        ]
        changed = list(original)
        changed[4:] = [
            CandidateEvidence(True, True, 4, True),
            CandidateEvidence(True, True, 1, True),
        ]
        first = candidate_state_decisions(windows, original, max_candidate_windows=3)
        second = candidate_state_decisions(windows, changed, max_candidate_windows=3)
        self.assertEqual(first.decisions[:4], second.decisions[:4])
        self.assertEqual(first.trace[:4], second.trace[:4])

    def test_empty_window_segment_has_zero_candidate_activity(self) -> None:
        policy = candidate_state_decisions([], [], max_candidate_windows=1)
        self.assertEqual(policy.decisions, ())
        metrics = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 250)],
            [],
            [],
            policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        candidate = metrics["candidate_diagnostics"]
        self.assertEqual(candidate["candidate_open_count"], 0)
        self.assertEqual(candidate["candidate_opens_per_valid_minute"], 0.0)
        self.assertIsNone(candidate["candidate_conversion_rate"])


class CandidateEvaluatorContractTests(unittest.TestCase):
    def test_candidate_mode_rejects_direct_command_and_wrong_reason(self) -> None:
        windows = make_windows(1)
        segment = [ScoringSegment(1, 0, 0, 0, 0, 500)]
        direct = [DecisionRecord(1, 0, 0, 0, 0, 0, 500, 1, READY, WAIT_IDLE, COMMAND_COMMIT)]
        with self.assertRaisesRegex(ValueError, "只能从 TASK_CANDIDATE"):
            evaluate_online_events(segment, [], windows, direct, mode=STATEFUL_CANDIDATE)

        wrong_reason = [DecisionRecord(
            1, 0, 0, 0, 0, 0, 500, NO_COMMAND, READY, TASK_CANDIDATE, None,
        )]
        with self.assertRaisesRegex(ValueError, "非法状态转换或转换原因"):
            evaluate_online_events(segment, [], windows, wrong_reason, mode=STATEFUL_CANDIDATE)

    def test_candidate_and_legacy_stateful_modes_remain_disjoint(self) -> None:
        windows = make_windows(1)
        segment = [ScoringSegment(1, 0, 0, 0, 0, 500)]
        candidate = candidate_state_decisions(
            windows,
            [CandidateEvidence(True, False, NO_COMMAND, False)],
            max_candidate_windows=1,
        ).decisions
        with self.assertRaisesRegex(ValueError, "READY/WAIT_IDLE"):
            evaluate_online_events(segment, [], windows, candidate, mode=STATEFUL_STRICT)

        legacy = [DecisionRecord(1, 0, 0, 0, 0, 0, 500, NO_COMMAND, READY, READY)]
        result = evaluate_online_events(segment, [], windows, legacy, mode=STATEFUL_STRICT)
        self.assertNotIn("candidate_diagnostics", result)


if __name__ == "__main__":
    unittest.main()
