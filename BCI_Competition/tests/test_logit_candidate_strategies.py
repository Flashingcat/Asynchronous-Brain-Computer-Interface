"""候选 logit 策略的手算、因果性和配置合同测试。"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from logit_candidate_strategies import (  # noqa: E402
    LogitStrategyConfig,
    logit_candidate_decisions,
)
from protocol_metrics import (  # noqa: E402
    CANDIDATE_ABORT_STAGE1,
    CANDIDATE_OPEN,
    COMMAND_COMMIT,
    NO_COMMAND,
    READY,
    TASK_CANDIDATE,
    WAIT_IDLE,
    ExpectedWindow,
    ScoringSegment,
    STATEFUL_CANDIDATE,
    evaluate_online_events,
)
from run_candidate_logit_matrix import (  # noqa: E402
    DEFAULT_POLICY_CONFIG,
    EXPECTED_OUTPUT_PROTOCOL,
    SCALE_FIELDS,
    _accumulate_logit_scale,
    _save_seed_matrix,
    _source_hashes,
    _summarize_logit_scale,
    load_strategy_contract,
    verify_matrix_child,
)
from run_epoch50_online_oof import file_hash  # noqa: E402


def make_windows(count: int, *, segment: int = 0, offset: int = 0) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment, index, offset + index * 125, offset + index * 125 + 500)
        for index in range(count)
    ]


def margin_for_probability(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def stage1_from_probabilities(probabilities: list[float]) -> np.ndarray:
    return np.asarray([[0.0, margin_for_probability(value)] for value in probabilities])


def make_config(**changes) -> LogitStrategyConfig:
    payload = {
        "strategy_id": "unit_strategy",
        "stage1_filter": "raw_margin",
        "stage1_alpha": None,
        "stage1_window": None,
        "task_on_probability": 0.5,
        "task_hold_probability": 0.3,
        "idle_reset_probability": 0.2,
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


class LogitCandidateStrategyTests(unittest.TestCase):
    def test_opening_window_stage2_is_excluded_from_commit_history(self) -> None:
        windows = make_windows(3)
        stage1 = stage1_from_probabilities([0.9, 0.9, 0.9])
        stage2 = np.asarray([
            [8.0, 0.0, 0.0, 0.0],  # 开门窗强烈支持 Left，但不得进入候选历史。
            [0.0, 8.0, 0.0, 0.0],  # 首个候选窗支持 Right，应提交 Right。
            [0.0, 0.0, 8.0, 0.0],
        ])
        result = logit_candidate_decisions(windows, stage1, stage2, make_config())
        self.assertEqual(
            [item.transition_reason for item in result.policy.decisions],
            [CANDIDATE_OPEN, COMMAND_COMMIT, None],
        )
        self.assertEqual(
            [item.emitted_class for item in result.policy.decisions],
            [NO_COMMAND, 2, NO_COMMAND],
        )
        self.assertEqual(result.trace[0].stage2_candidate_window_count, 0)
        self.assertEqual(result.trace[1].stage2_candidate_window_count, 1)

    def test_stage1_probability_drop_vetoes_simultaneous_stage2_commit(self) -> None:
        config = make_config(
            stage1_filter="ewma_probability",
            stage1_alpha=1.0,
            stage1_drop_abort=0.2,
        )
        windows = make_windows(2)
        stage1 = stage1_from_probabilities([0.9, 0.6])
        stage2 = np.asarray([[0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 8.0, 0.0]])
        result = logit_candidate_decisions(windows, stage1, stage2, config)
        second_evidence = result.trace[1].evidence
        self.assertFalse(second_evidence.task_hold)
        self.assertEqual(second_evidence.stage2_commit_class, 3)
        self.assertEqual(result.policy.decisions[1].emitted_class, NO_COMMAND)
        self.assertEqual(
            result.policy.decisions[1].transition_reason,
            CANDIDATE_ABORT_STAGE1,
        )

    def test_candidate_mean_and_segment_reset_do_not_inherit_old_stage2(self) -> None:
        config = make_config(
            stage2_filter="candidate_mean_centered_logits",
            stage2_min_candidate_windows=2,
        )
        windows = make_windows(3) + make_windows(2, segment=1, offset=2000)
        stage1 = stage1_from_probabilities([0.9] * 5)
        stage2 = np.asarray([
            [0.0, 0.0, 0.0, 9.0],
            [8.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
        ])
        result = logit_candidate_decisions(windows, stage1, stage2, config)
        self.assertEqual(result.policy.decisions[2].emitted_class, 1)
        self.assertEqual(result.policy.decisions[3].decision_state_before, READY)
        self.assertEqual(result.policy.decisions[3].decision_state_after, TASK_CANDIDATE)
        self.assertEqual(result.trace[3].stage2_candidate_window_count, 0)
        self.assertEqual(result.trace[4].stage2_candidate_window_count, 1)
        self.assertEqual(result.policy.decisions[4].emitted_class, NO_COMMAND)

    def test_logit_common_shift_and_future_suffix_cannot_change_prior_trace(self) -> None:
        config = make_config(
            stage1_filter="ewma_margin",
            stage1_alpha=0.5,
            stage2_filter="candidate_ewma_centered_logits",
            stage2_alpha=0.5,
            stage2_min_candidate_windows=2,
        )
        windows = make_windows(6)
        stage1 = stage1_from_probabilities([0.8, 0.7, 0.7, 0.1, 0.1, 0.8])
        stage2 = np.asarray([
            [3.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ])
        original = logit_candidate_decisions(windows, stage1, stage2, config)
        shifted = logit_candidate_decisions(
            windows,
            stage1 + np.asarray([[5.0, 5.0]]),
            stage2 + np.arange(6, dtype=float)[:, None] * 7.0,
            config,
        )
        self.assertEqual(original.policy, shifted.policy)
        self.assertEqual(original.trace, shifted.trace)

        changed_stage1, changed_stage2 = stage1.copy(), stage2.copy()
        changed_stage1[4:] = np.asarray([[100.0, -100.0], [-100.0, 100.0]])
        changed_stage2[4:] = np.asarray([[50.0, -50.0, 0.0, 0.0], [0.0, 0.0, -50.0, 50.0]])
        changed = logit_candidate_decisions(windows, changed_stage1, changed_stage2, config)
        self.assertEqual(original.policy.decisions[:4], changed.policy.decisions[:4])
        self.assertEqual(original.trace[:4], changed.trace[:4])

    def test_probability_curvature_matches_independent_second_difference(self) -> None:
        config = make_config(
            stage2_filter="candidate_mean_centered_logits",
            stage2_min_candidate_windows=3,
            stage2_stable_windows=1,
            stage2_max_probability_curvature=3.0,
        )
        windows = make_windows(4)
        stage1 = stage1_from_probabilities([0.9] * 4)
        stage2 = np.asarray([
            [9.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
        ])
        result = logit_candidate_decisions(windows, stage1, stage2, config)

        def softmax(row: np.ndarray) -> np.ndarray:
            exp = np.exp(row - np.max(row))
            return exp / np.sum(exp)

        probabilities = [softmax(stage2[index]) for index in (1, 2, 3)]
        expected = np.linalg.norm(probabilities[2] - 2 * probabilities[1] + probabilities[0])
        self.assertAlmostEqual(result.trace[3].stage2_probability_curvature, expected)

    def test_ewma_formulas_and_rolling_prefix_reset_match_hand_calculation(self) -> None:
        windows = make_windows(3)
        stage1 = stage1_from_probabilities([0.8, 0.4, 0.4])
        stage2 = np.asarray([
            [9.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ])
        config = make_config(
            stage1_filter="ewma_margin",
            stage1_alpha=0.5,
            stage2_filter="candidate_ewma_centered_logits",
            stage2_alpha=0.5,
            stage2_min_candidate_windows=2,
        )
        result = logit_candidate_decisions(windows, stage1, stage2, config)
        expected_margin = 0.5 * margin_for_probability(0.4) + 0.5 * margin_for_probability(0.8)
        expected_probability = 1.0 / (1.0 + math.exp(-expected_margin))
        self.assertAlmostEqual(result.trace[1].stage1_filtered_task_probability, expected_probability)

        first = stage2[1] - np.mean(stage2[1])
        second = stage2[2] - np.mean(stage2[2])
        aggregate = 0.5 * second + 0.5 * first
        exp = np.exp(aggregate - np.max(aggregate))
        expected_top_probability = float(np.max(exp / np.sum(exp)))
        self.assertAlmostEqual(result.trace[2].stage2_top_probability, expected_top_probability)

        reset_windows = make_windows(2) + make_windows(1, segment=1, offset=2000)
        rolling = logit_candidate_decisions(
            reset_windows,
            stage1_from_probabilities([0.9, 0.9, 0.2]),
            np.zeros((3, 4)),
            make_config(stage1_filter="rolling_margin", stage1_window=3),
        )
        self.assertAlmostEqual(rolling.trace[2].stage1_filtered_task_probability, 0.2)

    def test_finite_extremes_fail_fast_but_common_stage2_shift_remains_valid(self) -> None:
        config = make_config()
        with self.assertRaisesRegex(ValueError, "margin 溢出"):
            logit_candidate_decisions(
                make_windows(1),
                np.asarray([[1e308, -1e308]]),
                np.zeros((1, 4)),
                config,
            )
        with self.assertRaisesRegex(ValueError, "溢出"):
            logit_candidate_decisions(
                make_windows(2),
                stage1_from_probabilities([0.9, 0.9]),
                np.asarray([[0.0] * 4, [1e308, -1e308, 0.0, 0.0]]),
                config,
            )
        accepted = logit_candidate_decisions(
            make_windows(2),
            stage1_from_probabilities([0.9, 0.9]),
            np.asarray([[0.0] * 4, [1e308] * 4]),
            config,
        )
        self.assertEqual(accepted.trace[1].stage2_top_probability, 0.25)
        with self.assertRaisesRegex(TypeError, "LogitStrategyConfig"):
            logit_candidate_decisions(
                make_windows(1), np.zeros((1, 2)), np.zeros((1, 4)), {},
            )


class LogitCandidateConfigTests(unittest.TestCase):
    def test_repository_contract_loads_all_unique_cells(self) -> None:
        payload, configs = load_strategy_contract(DEFAULT_POLICY_CONFIG)
        self.assertEqual(payload["protocol_id"], EXPECTED_OUTPUT_PROTOCOL)
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({item.strategy_id for item in configs}), 8)
        self.assertEqual(payload["included_session"], 0)

    def test_invalid_schema_filters_and_window_relationships_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema"):
            LogitStrategyConfig.from_dict({"strategy_id": "missing_fields"})
        with self.assertRaisesRegex(ValueError, "EWMA"):
            make_config(stage1_filter="ewma_margin", stage1_alpha=None)
        with self.assertRaisesRegex(ValueError, "stable"):
            make_config(stage2_min_candidate_windows=1, stage2_stable_windows=2)
        with self.assertRaisesRegex(ValueError, "曲率"):
            make_config(stage2_max_probability_curvature=0.2)

    def test_test_session_or_duplicate_strategy_tampering_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            payload["included_session"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "总配置"):
                load_strategy_contract(path)

            payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
            payload["strategies"].append(payload["strategies"][0])
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "不得重复"):
                load_strategy_contract(path)

    def test_unlabeled_scale_summary_resets_differences_at_segment_boundary(self) -> None:
        windows = make_windows(3) + make_windows(1, segment=1, offset=2000)
        stage1 = stage1_from_probabilities([0.2, 0.8, 0.8, 0.1])
        stage2 = np.asarray([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
        ])
        samples = {name: [] for name in SCALE_FIELDS}
        _accumulate_logit_scale(samples, windows, stage1, stage2)
        summary = _summarize_logit_scale(samples)
        self.assertFalse(summary["labels_or_events_used"])
        self.assertEqual(
            summary["fields"]["stage1_task_probability_delta"]["sample_count"],
            2,
        )
        self.assertAlmostEqual(
            summary["fields"]["stage1_task_probability_delta"]["quantile_values"][-1],
            0.6,
        )
        self.assertEqual(
            summary["fields"]["stage2_probability_second_difference_l2"]["sample_count"],
            1,
        )

    def test_audit_counts_over_255_are_saved_without_uint8_wraparound(self) -> None:
        config = make_config(strategy_id="long_candidate", max_candidate_windows=300)
        windows = make_windows(302)
        stage1 = stage1_from_probabilities([0.9] * len(windows))
        stage2 = np.zeros((len(windows), 4))
        segments = [ScoringSegment(1, 0, 0, 0, 0, windows[-1].window_stop_sample)]
        strategy = logit_candidate_decisions(windows, stage1, stage2, config)
        probe = evaluate_online_events(
            segments, [], windows, strategy.policy.decisions, mode=STATEFUL_CANDIDATE,
        )
        contract = {"inventory": {
            "segment_inventory_sha256": probe["scoring_segment_inventory_sha256"],
            "window_inventory_sha256": probe["expected_window_inventory_sha256"],
            "event_inventory_sha256": probe["event_inventory_sha256"],
            "event_count": 0,
        }}
        inventory = SimpleNamespace(windows=windows, segments=segments, events=[])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _save_seed_matrix(
                root,
                inventory,
                contract,
                42,
                stage1,
                stage2,
                {"file": "frozen_input.npz", "sha256": "a" * 64},
                (config,),
            )
            with np.load(root / "seed42_candidate_logit_trajectories.npz") as payload:
                for name, expected_maximum in (
                    ("long_candidate_candidate_age_before", 299),
                    ("long_candidate_stage2_candidate_window_count", 300),
                    ("long_candidate_stage2_stable_windows", 300),
                ):
                    self.assertEqual(payload[name].dtype, np.dtype(np.int64))
                    self.assertEqual(int(np.max(payload[name])), expected_maximum)

    def test_child_manifest_uses_frozen_source_and_policy_identity(self) -> None:
        source_hashes = _source_hashes()
        self.assertIn("single_window_multi_subject_verifier", source_hashes)
        self.assertEqual(len(source_hashes), 8)
        policy_sha = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def artifact(name: str) -> dict[str, str]:
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                return {"file": name, "sha256": file_hash(path)}

            manifest = {
                "status": "PASS",
                "selection_status": "none_all_cells_reported",
                "subject": 1,
                "seeds": [42, 43, 44],
                "strategy_ids": ["raw_current"],
                "included_session": 0,
                "test_session_access": "forbidden_and_not_loaded",
                "source_sha256": source_hashes,
                "policy_contract_sha256": policy_sha,
                "run_log": artifact("run_log.json"),
                "seed_artifacts": {},
            }
            for seed in (42, 43, 44):
                manifest["seed_artifacts"][str(seed)] = {
                    "input_scores": {"file": "external.npz", "sha256": "c" * 64},
                    "metrics": artifact(f"seed{seed}_metrics.json"),
                    "trajectories": artifact(f"seed{seed}_trajectory.npz"),
                }
            verify_matrix_child(
                root, manifest, 1, ("raw_current",), source_hashes, policy_sha,
            )
            changed = dict(manifest)
            changed["source_sha256"] = {**source_hashes, "matrix_runner": "f" * 64}
            with self.assertRaisesRegex(RuntimeError, "合同非法"):
                verify_matrix_child(
                    root, changed, 1, ("raw_current",), source_hashes, policy_sha,
                )


if __name__ == "__main__":
    unittest.main()
