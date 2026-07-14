from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from commit_reset_diagnostics import diagnose_commit_reset  # noqa: E402
from logit_candidate_strategies import (  # noqa: E402
    LogitStrategyConfig,
    logit_candidate_decisions,
)
from protocol_metrics import (  # noqa: E402
    CANDIDATE_OPEN,
    COMMAND_COMMIT,
    IDLE_RESET,
    MIEvent,
    ScoringSegment,
    STATEFUL_CANDIDATE,
    ExpectedWindow,
    evaluate_online_events,
)
from run_commit_reset_matrix import (  # noqa: E402
    DEFAULT_POLICY_CONFIG,
    EXPECTED_OUTPUT_PROTOCOL,
    _source_hashes,
    _verify_child,
    load_commit_reset_contract,
)
from run_epoch50_online_oof import file_hash  # noqa: E402


def windows(count: int) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
        for index in range(count)
    ]


def stage1(probabilities: list[float]) -> np.ndarray:
    margins = [math.log(value / (1.0 - value)) for value in probabilities]
    return np.asarray([[0.0, value] for value in margins], dtype=np.float64)


def config(**changes) -> LogitStrategyConfig:
    payload = {
        "strategy_id": "unit_cell",
        "stage1_filter": "raw_margin",
        "stage1_alpha": None,
        "stage1_window": None,
        "task_on_probability": 0.5,
        "task_hold_probability": 0.3,
        "idle_reset_probability": 0.3,
        "stage1_drop_abort": None,
        "stage2_filter": "current_centered_logits",
        "stage2_alpha": None,
        "stage2_min_candidate_windows": 1,
        "stage2_top_probability": 0.55,
        "stage2_probability_gap": 0.15,
        "stage2_stable_windows": 1,
        "stage2_max_probability_curvature": None,
        "max_candidate_windows": 8,
    }
    payload.update(changes)
    return LogitStrategyConfig.from_dict(payload)


class CommitResetContractTests(unittest.TestCase):
    def test_repository_contract_is_exact_three_by_six_cartesian_product(self) -> None:
        payload, cells = load_commit_reset_contract(DEFAULT_POLICY_CONFIG)
        self.assertEqual(payload["protocol_id"], EXPECTED_OUTPUT_PROTOCOL)
        self.assertEqual(len(cells), 18)
        self.assertEqual(len({item.cell_id for item in cells}), 18)
        self.assertEqual(
            {item.stage2_top_probability for item in cells},
            {0.55, 0.625, 0.70},
        )
        self.assertEqual(
            {(item.idle_reset_probability, item.idle_reset_consecutive_windows) for item in cells},
            {(0.2, 1), (0.2, 2), (0.3, 1), (0.3, 2), (0.4, 1), (0.4, 2)},
        )

    def test_contract_tampering_and_test_session_are_rejected(self) -> None:
        payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            payload["included_session"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结诊断协议"):
                load_commit_reset_contract(path)

            payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
            payload["commit_profiles"][1]["stage2_top_probability"] = 0.63
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "3x6"):
                load_commit_reset_contract(path)

            payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
            payload["reset_profiles"][0]["consecutive_windows"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "显式整数"):
                load_commit_reset_contract(path)

    def test_child_manifest_freezes_transitive_input_verifier_source(self) -> None:
        source_hashes = _source_hashes()
        self.assertIn("single_window_multi_subject_verifier", source_hashes)
        self.assertEqual(len(source_hashes), 10)

        # 构造最小完整子清单，证明源码合同原样可通过、缺少传递依赖时会被拒绝。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def artifact(name: str) -> dict[str, str]:
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                return {"file": name, "sha256": file_hash(path)}

            manifest = {
                "status": "PASS",
                "subject": 1,
                "seeds": [42, 43, 44],
                "cell_ids": ["cell"],
                "included_session": 0,
                "test_session_access": "forbidden_and_not_loaded",
                "policy_contract_sha256": "b" * 64,
                "source_sha256": source_hashes,
                "run_log": artifact("run_log.json"),
                "seed_artifacts": {},
            }
            for seed in (42, 43, 44):
                manifest["seed_artifacts"][str(seed)] = {
                    "input_scores": {"file": "external.npz", "sha256": "c" * 64},
                    "metrics": artifact(f"seed{seed}_metrics.json"),
                    "trajectories": artifact(f"seed{seed}_trajectories.npz"),
                }
            _verify_child(root, manifest, 1, ("cell",), "b" * 64, source_hashes)

            incomplete = dict(source_hashes)
            incomplete.pop("single_window_multi_subject_verifier")
            changed = {**manifest, "source_sha256": incomplete}
            with self.assertRaisesRegex(RuntimeError, "合同非法"):
                _verify_child(root, changed, 1, ("cell",), "b" * 64, source_hashes)


class ConsecutiveResetTests(unittest.TestCase):
    def test_reset_threshold_may_exceed_hold_but_must_remain_below_open(self) -> None:
        accepted = config(idle_reset_probability=0.4)
        self.assertEqual(accepted.idle_reset_probability, 0.4)
        with self.assertRaisesRegex(ValueError, "idle_reset < task_on"):
            config(idle_reset_probability=0.5)

    def test_two_window_reset_streak_clears_on_failure(self) -> None:
        ws = windows(7)
        probabilities = [0.9, 0.9, 0.4, 0.29, 0.35, 0.28, 0.27]
        stage2 = np.asarray([[8.0, 0.0, 0.0, 0.0]] * len(ws))
        result = logit_candidate_decisions(
            ws,
            stage1(probabilities),
            stage2,
            config(),
            idle_reset_consecutive_windows=2,
        )
        reasons = [item.transition_reason for item in result.policy.decisions]
        self.assertEqual(reasons[0], CANDIDATE_OPEN)
        self.assertEqual(reasons[1], COMMAND_COMMIT)
        self.assertIsNone(reasons[3])
        self.assertEqual(result.trace[3].idle_reset_consecutive_count, 1)
        self.assertEqual(result.trace[4].idle_reset_consecutive_count, 0)
        self.assertEqual(result.trace[5].idle_reset_consecutive_count, 1)
        self.assertEqual(result.trace[6].idle_reset_consecutive_count, 2)
        self.assertEqual(reasons[6], IDLE_RESET)

    def test_reset_streak_and_wait_state_clear_at_segment_boundary(self) -> None:
        ws = windows(3) + [ExpectedWindow(1, 0, 0, 1, 0, 1000, 1500)]
        result = logit_candidate_decisions(
            ws,
            stage1([0.9, 0.9, 0.29, 0.29]),
            np.asarray([[8.0, 0.0, 0.0, 0.0]] * len(ws)),
            config(),
            idle_reset_consecutive_windows=2,
        )
        # 前一 segment 以 WAIT_IDLE 且仅一窗复位证据结束；新 segment 必须独立 READY。
        self.assertEqual(result.trace[2].idle_reset_consecutive_count, 1)
        self.assertEqual(result.trace[3].idle_reset_consecutive_count, 0)
        self.assertEqual(result.policy.decisions[3].decision_state_before, "READY")
        self.assertEqual(result.policy.decisions[3].decision_state_after, "READY")

    def test_invalid_confirmation_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "正整数"):
            logit_candidate_decisions(
                windows(1), stage1([0.9]), np.zeros((1, 4)), config(),
                idle_reset_consecutive_windows=0,
            )


class CommitResetDiagnosticTests(unittest.TestCase):
    def test_post_mi_delayed_commit_is_attributed_without_changing_inference(self) -> None:
        ws = windows(4)
        segment = ScoringSegment(1, 0, 0, 0, 0, 875)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 700, 1)
        # 开门后最少等待两个候选窗，故命令在 sample 750 提交，晚于 MI offset=700。
        result = logit_candidate_decisions(
            ws,
            stage1([0.9] * 4),
            np.asarray([[8.0, 0.0, 0.0, 0.0]] * 4),
            config(stage2_min_candidate_windows=2),
        )
        evaluated = evaluate_online_events(
            [segment], [event], ws, result.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        diagnostics = diagnose_commit_reset(
            [segment], [event], ws, result.policy.decisions, evaluated,
        )
        self.assertEqual(evaluated["idle_false_command_count"], 1)
        self.assertEqual(
            diagnostics["idle_false_attribution"]["post_mi_spillover_count"],
            1,
        )
        self.assertEqual(
            diagnostics["wait_idle"]["segment_end_unresolved_count"],
            1,
        )
        self.assertEqual(
            diagnostics["event_lock"]["fully_wait_idle_event_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
