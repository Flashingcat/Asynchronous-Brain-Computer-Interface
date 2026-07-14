"""多窗口硬投票状态轨迹、配置网格和跨被试汇总的回归测试。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from hard_vote_policy import VOTE_GRID, policy_id, stateful_hard_vote_decisions  # noqa: E402
from protocol_metrics import (  # noqa: E402
    READY,
    STATEFUL_STRICT,
    WAIT_IDLE,
    ExpectedWindow,
    ScoringSegment,
    evaluate_online_events,
)
from run_epoch50_online_oof import KNOWN_SEEDS, KNOWN_SUBJECTS  # noqa: E402
from run_hard_vote_matrix import (  # noqa: E402
    CORE_FIELDS,
    FROZEN_INPUT_CHILD_SOURCE_SHA256,
    FROZEN_INPUT_MASTER_SOURCE_SHA256,
    aggregate_subject_matrix,
    runtime_environment,
    validate_policy_config,
    verify_matrix_child,
)
from run_epoch50_online_oof import file_hash  # noqa: E402
from run_epoch50_online_oof_all_subjects import verify_child_artifacts  # noqa: E402


def make_windows(count: int, *, segment: int = 0, offset: int = 0) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, segment, index, offset + index * 125, offset + index * 125 + 500)
        for index in range(count)
    ]


class HardVotePolicyTests(unittest.TestCase):
    def test_frozen_grid_is_strict_majority_to_unanimity(self) -> None:
        self.assertEqual(
            VOTE_GRID,
            ((2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)),
        )

    def test_cache_is_cleared_after_both_state_transitions(self) -> None:
        windows = make_windows(10)
        # n3_k2：第 3/4 窗形成 MI=2；输出后旧票清空，2/0/0 三张新票确认 IDLE；
        # 再清空后必须重新收满 3 张 MI=3，不能借用复位前的任何标签。
        labels = np.asarray([0, 0, 2, 2, 2, 0, 0, 3, 3, 3], dtype=np.int8)
        decisions = stateful_hard_vote_decisions(
            windows, labels, window_count=3, vote_threshold=2,
        )
        self.assertEqual(
            [item.emitted_class for item in decisions],
            [-1, -1, -1, 2, -1, -1, -1, -1, -1, 3],
        )
        self.assertEqual(
            (decisions[3].decision_state_before, decisions[3].decision_state_after),
            (READY, WAIT_IDLE),
        )
        self.assertEqual(
            (decisions[6].decision_state_before, decisions[6].decision_state_after),
            (WAIT_IDLE, READY),
        )

        # 独立基础评估器必须接受完整状态轨迹；这里所有命令都位于 IDLE，只用于手算。
        result = evaluate_online_events(
            [ScoringSegment(1, 0, 0, 0, 0, 1625)],
            [], windows, decisions, mode=STATEFUL_STRICT,
        )
        self.assertEqual(result["emitted_command_count"], 2)
        self.assertEqual(result["idle_false_command_count"], 2)

    def test_segment_start_resets_state_and_requires_full_buffer(self) -> None:
        first = make_windows(4, segment=0)
        second = make_windows(3, segment=1, offset=2000)
        labels = np.asarray([1, 1, 1, 1, 4, 4, 4], dtype=np.int8)
        decisions = stateful_hard_vote_decisions(
            first + second, labels, window_count=3, vote_threshold=2,
        )
        self.assertEqual(decisions[4].decision_state_before, READY)
        self.assertEqual([item.emitted_class for item in decisions[4:]], [-1, -1, 4])

    def test_full_n_votes_are_required_even_if_k_votes_arrive_early(self) -> None:
        decisions = stateful_hard_vote_decisions(
            make_windows(4),
            np.asarray([1, 1, 1, 0], dtype=np.int8),
            window_count=5,
            vote_threshold=3,
        )
        self.assertTrue(all(item.emitted_class == -1 for item in decisions))

    def test_invalid_policy_or_label_type_is_rejected(self) -> None:
        windows = make_windows(3)
        with self.assertRaisesRegex(ValueError, "大于 N/2"):
            stateful_hard_vote_decisions(
                windows, np.asarray([1, 1, 1]), window_count=3, vote_threshold=1,
            )
        with self.assertRaisesRegex(ValueError, "整数 0..4"):
            stateful_hard_vote_decisions(
                windows, np.asarray([1.0, 1.0, 1.0]), window_count=3, vote_threshold=2,
            )


class HardVoteMatrixContractTests(unittest.TestCase):
    def test_child_manifest_requires_artifact_and_segment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def artifact(name: str) -> dict[str, str]:
                path = root / name
                path.write_text(name, encoding="utf-8")
                return {"file": name, "sha256": file_hash(path)}

            manifest = {
                "status": "PASS",
                "subject": 1,
                "seeds": list(KNOWN_SEEDS),
                "included_session": 0,
                "test_session_access": "forbidden_and_not_loaded",
                "artifact_policy": "official_trial_exclusion",
                "segment_policy": "separate_clean_segments_no_time_compression",
                "artifact_policy_binding": "legacy_v1_protocol_contract",
                "vote_grid": [list(item) for item in VOTE_GRID],
                "run_log": artifact("run_log.json"),
                "seed_artifacts": {},
            }
            for seed in KNOWN_SEEDS:
                manifest["seed_artifacts"][str(seed)] = {
                    "input_scores": {"file": "external.npz", "sha256": "c" * 64},
                    "metrics": artifact(f"seed{seed}_metrics.json"),
                    "trajectories": artifact(f"seed{seed}_trajectories.npz"),
                }
            verify_matrix_child(root, manifest, 1)
            changed = {**manifest, "artifact_policy": "unknown"}
            with self.assertRaisesRegex(RuntimeError, "合同非法"):
                verify_matrix_child(root, changed, 1)

    def test_historical_source_contract_is_checked_without_current_source_equality(self) -> None:
        self.assertEqual(
            FROZEN_INPUT_MASTER_SOURCE_SHA256["single_subject_runner"],
            FROZEN_INPUT_CHILD_SOURCE_SHA256["runner"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_id = "historical_input_fixture"

            def artifact(name: str, payload: str) -> dict[str, str]:
                path = root / name
                path.write_text(payload, encoding="utf-8")
                return {"file": name, "sha256": file_hash(path)}

            log = artifact("run_log.json", json.dumps({
                "status": "PASS",
                "protocol_id": protocol_id,
                "subject": 1,
                "seeds": [42],
            }))
            seed_artifacts = {
                "42": {
                    role: artifact(f"{role}.txt", role)
                    for role in ("scores_and_decisions", "stateless_metrics", "stateful_metrics")
                }
            }
            records = [
                {
                    "fold": fold,
                    "seed": 42,
                    "stage": stage,
                    "validation_runs": [fold],
                    "train_runs": [],
                    "saved_oof_reproduction_max_abs_error": 0.0,
                    "continuous_window_count": 2,
                }
                for fold in range(6)
                for stage in (1, 2)
            ]
            frozen = {role: str(index) * 64 for index, role in enumerate(
                ("runner", "protocol_metrics", "oof_training_bundle_reader", "model_factory", "eegnet"),
                start=1,
            )}
            manifest = {
                "protocol_id": protocol_id,
                "inventory_contract": {"inventory": {"window_count": 12}},
                "checkpoint_records": records,
                "source_sha256": frozen,
                "run_log": log,
                "seed_artifacts": seed_artifacts,
            }
            verify_child_artifacts(
                root, manifest, 1, (42,), expected_source_hashes=frozen,
            )
            changed = copy.deepcopy(manifest)
            changed["source_sha256"]["protocol_metrics"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "来源不完整"):
                verify_child_artifacts(
                    root, changed, 1, (42,), expected_source_hashes=frozen,
                )

    def test_runtime_environment_records_actual_interpreter(self) -> None:
        runtime = runtime_environment()
        self.assertEqual(Path(runtime["python_executable"]), Path(sys.executable).resolve())
        self.assertEqual(Path(runtime["python_prefix"]), Path(sys.prefix).resolve())
        self.assertEqual(runtime["environment_name"], Path(sys.prefix).resolve().name)
        self.assertTrue(runtime["python_version"])
        self.assertTrue(runtime["numpy_version"])

    def test_checked_in_policy_config_matches_code_grid(self) -> None:
        path = PROJECT_ROOT / "config" / "evaluation" / "bnci2014001_hard_vote_matrix_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_policy_config(config), VOTE_GRID)
        changed = copy.deepcopy(config)
        changed["require_full_buffer"] = False
        with self.assertRaisesRegex(RuntimeError, "冻结首版协议"):
            validate_policy_config(changed)

        # 文本合同同样参与校验，避免配置声称的标签或锚点语义与代码漂移。
        for field in ("joint_hard_labels", "single_window_reference"):
            changed = copy.deepcopy(config)
            changed[field] = "错误的协议语义"
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "冻结首版协议",
            ):
                validate_policy_config(changed)

    def test_cross_subject_summary_is_equal_weight_within_seed(self) -> None:
        summaries = {}
        for subject in KNOWN_SUBJECTS:
            subject_summary = {}
            for window_count, vote_threshold in VOTE_GRID:
                identifier = policy_id(window_count, vote_threshold)
                per_seed = {
                    str(seed): {field: float(subject) for field in CORE_FIELDS}
                    for seed in KNOWN_SEEDS
                }
                subject_summary[identifier] = {
                    "per_seed": per_seed,
                    "aggregate_across_seeds": {
                        field: {"mean": float(subject)} for field in CORE_FIELDS
                    },
                }
            summaries[subject] = subject_summary

        result = aggregate_subject_matrix(summaries)
        cell = result["n3_k2"]
        self.assertEqual(
            cell["per_seed_subject_macro"]["42"]["correct_event_rate"],
            {"mean": 5.0, "valid_subject_count": 9},
        )
        self.assertEqual(
            cell["across_subjects_from_seed_means"]["correct_event_rate"]["valid_count"],
            9,
        )


if __name__ == "__main__":
    unittest.main()
