"""Fast-0/Fast-1 冻结配置、路径归因和最小端到端产物测试。"""

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

from fast_path_candidate_strategies import fast_path_candidate_decisions  # noqa: E402
from fast_path_diagnostics import diagnose_against_anchor, diagnose_fast_paths  # noqa: E402
from protocol_metrics import (  # noqa: E402
    NO_COMMAND,
    STATEFUL_CANDIDATE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)
from run_epoch50_online_oof import file_hash  # noqa: E402
from run_fast_path_matrix import (  # noqa: E402
    ANCHOR_CELL_ID,
    DEFAULT_POLICY_CONFIG,
    EXPECTED_OUTPUT_PROTOCOL,
    _aggregate_subjects,
    _save_seed_matrix,
    _source_hashes,
    _write_csvs,
    load_fast_path_contract,
)


def windows(count: int) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
        for index in range(count)
    ]


def stage1(probabilities: list[float]) -> np.ndarray:
    margins = [math.log(value / (1.0 - value)) for value in probabilities]
    return np.asarray([[0.0, value] for value in margins], dtype=np.float64)


class FastPathMatrixContractTests(unittest.TestCase):
    def test_repository_contract_has_anchor_and_six_factor_cells(self) -> None:
        payload, cells = load_fast_path_contract(DEFAULT_POLICY_CONFIG)
        self.assertEqual(payload["protocol_id"], EXPECTED_OUTPUT_PROTOCOL)
        self.assertEqual(len(cells), 7)
        self.assertEqual(cells[0].cell_id, ANCHOR_CELL_ID)
        self.assertIsNone(cells[0].config.fast0)
        self.assertIsNone(cells[0].config.fast1)
        self.assertEqual(sum(item.config.fast0 is not None for item in cells), 4)
        self.assertEqual(sum(item.config.fast1 is not None for item in cells), 4)

    def test_contract_tampering_and_test_session_are_rejected(self) -> None:
        payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fast.json"
            payload["included_session"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结诊断合同"):
                load_fast_path_contract(path)

            payload = json.loads(DEFAULT_POLICY_CONFIG.read_text(encoding="utf-8"))
            payload["fast1_profiles"]["balanced"]["require_same_raw_class"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结诊断合同"):
                load_fast_path_contract(path)

    def test_source_manifest_covers_transitive_helpers(self) -> None:
        hashes = _source_hashes()
        self.assertIn("fast_path_candidate_strategies", hashes)
        self.assertIn("fast_path_diagnostics", hashes)
        self.assertIn("candidate_logit_matrix_helpers", hashes)
        self.assertIn("commit_reset_matrix_helpers", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


class FastPathMatrixEndToEndTests(unittest.TestCase):
    def test_path_attribution_and_paired_anchor_are_hand_checkable(self) -> None:
        ws = windows(3)
        event = MIEvent("e", 1, 0, 0, 0, 0, 750, 1)
        segment = ScoringSegment(1, 0, 0, 0, 0, 750)
        anchor = [
            DecisionRecord(1, 0, 0, 0, 0, 0, 500, NO_COMMAND, "READY", "READY"),
            DecisionRecord(1, 0, 0, 0, 1, 125, 625, NO_COMMAND, "READY", "READY"),
            DecisionRecord(1, 0, 0, 0, 2, 250, 750, NO_COMMAND, "READY", "READY"),
        ]
        current = [
            DecisionRecord(
                1, 0, 0, 0, 0, 0, 500, 1,
                "READY", "WAIT_IDLE", "fast0_command_commit",
            ),
            DecisionRecord(1, 0, 0, 0, 1, 125, 625, NO_COMMAND, "WAIT_IDLE", "WAIT_IDLE"),
            DecisionRecord(1, 0, 0, 0, 2, 250, 750, NO_COMMAND, "WAIT_IDLE", "WAIT_IDLE"),
        ]
        anchor_eval = evaluate_online_events(
            [segment], [event], ws, anchor, mode=STATEFUL_CANDIDATE,
        )
        current_eval = evaluate_online_events(
            [segment], [event], ws, current, mode=STATEFUL_CANDIDATE,
        )
        paths = diagnose_fast_paths([event], current, current_eval)
        paired = diagnose_against_anchor(anchor_eval, current_eval)
        self.assertEqual(paths["paths"]["fast0"]["correct_event_count"], 1)
        self.assertEqual(paths["paths"]["slow"]["command_count"], 0)
        self.assertEqual(paired["anchor_miss_rescued_correct_count"], 1)

    def test_minimal_seed_run_writes_recomputable_metrics_and_trajectories(self) -> None:
        payload, cells = load_fast_path_contract(DEFAULT_POLICY_CONFIG)
        self.assertEqual(payload["included_session"], 0)
        ws = windows(8)
        segment = ScoringSegment(1, 0, 0, 0, 0, 1375)
        events = [MIEvent("e", 1, 0, 0, 0, 0, 1000, 1)]
        s1 = stage1([0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9])
        s2 = np.asarray([[8.0, 0.0, 0.0, 0.0]] * len(ws), dtype=np.float64)
        anchor_strategy = fast_path_candidate_decisions(ws, s1, s2, cells[0].config)
        reference = evaluate_online_events(
            [segment], events, ws, anchor_strategy.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        contract = {"inventory": {
            "segment_inventory_sha256": reference["scoring_segment_inventory_sha256"],
            "window_inventory_sha256": reference["expected_window_inventory_sha256"],
            "event_inventory_sha256": reference["event_inventory_sha256"],
            "event_count": reference["event_count"],
        }}
        inventory = SimpleNamespace(
            windows=ws,
            segments=[segment],
            events=events,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, artifacts = _save_seed_matrix(
                root,
                inventory,
                contract,
                42,
                s1,
                s2,
                {"file": "frozen_input.npz", "sha256": "a" * 64},
                cells,
            )
            self.assertEqual(set(summary), {item.cell_id for item in cells})
            for role in ("metrics", "trajectories"):
                path = root / artifacts[role]["file"]
                self.assertTrue(path.is_file())
                self.assertEqual(file_hash(path), artifacts[role]["sha256"])
            metrics = json.loads((root / artifacts["metrics"]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["cells"][ANCHOR_CELL_ID]["paired_anchor_diagnostics"]["anchor_correct_harmed_count"],
                0,
            )
            with np.load(root / artifacts["trajectories"]["file"], allow_pickle=False) as arrays:
                self.assertIn("anchor_no_fast_reason", arrays.files)
                self.assertIn("f01_balanced_fast0_pass", arrays.files)
                self.assertIn("f01_strict_raw_stage2_top_class", arrays.files)
                self.assertIn("f01_strict_fast1_top_class", arrays.files)
                self.assertIn("f01_strict_slow_top_class", arrays.files)
                self.assertIn("f01_strict_idle_reset_raw_condition", arrays.files)
                self.assertIn("f01_strict_idle_reset_consecutive_count", arrays.files)
                self.assertEqual(len(arrays["window_rows"]), len(ws))

            all_subjects = {
                subject: {str(seed): summary for seed in (42, 43, 44)}
                for subject in range(1, 10)
            }
            csvs = _write_csvs(root, _aggregate_subjects(all_subjects, cells))
            per_seed_header = (root / csvs["per_seed_csv"]["file"]).read_text()
            aggregate_header = (root / csvs["aggregate_csv"]["file"]).read_text()
            self.assertIn("fast1_triggered_class_accuracy_valid_subject_count", per_seed_header)
            self.assertIn("fast1_triggered_class_accuracy_valid_seed_count", aggregate_header)
            self.assertIn("fast1_triggered_class_accuracy_valid_subject_count_min", aggregate_header)


if __name__ == "__main__":
    unittest.main()
