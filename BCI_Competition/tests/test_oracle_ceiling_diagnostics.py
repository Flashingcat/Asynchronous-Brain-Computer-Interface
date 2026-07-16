from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from logit_candidate_strategies import LogitStrategyConfig, logit_candidate_decisions  # noqa: E402
from oracle_ceiling_diagnostics import (  # noqa: E402
    ALL_TRUTH_CELL_ID,
    MODEL_CELL_ID,
    ORACLE_CELLS,
    _max_convex_margin,
    component_oracle_replays,
    shapley_component_contributions,
    stage2_evidence_ceiling,
)
from protocol_metrics import (  # noqa: E402
    STATEFUL_CANDIDATE,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
    evaluate_online_events,
)
from run_oracle_ceiling_analysis import (  # noqa: E402
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_ORACLE_CONFIG,
    EXPECTED_OUTPUT_PROTOCOL,
    _run_seed,
    _source_hashes,
    load_anchor,
    load_oracle_contract,
)


def anchor_config() -> LogitStrategyConfig:
    """单测也使用正式 c055/r020/l1 参数，避免为测试另造 Oracle 语义。"""
    return LogitStrategyConfig.from_dict({
        "strategy_id": "oracle_anchor",
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


def windows(count: int) -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
        for index in range(count)
    ]


def stage1(probabilities: list[float]) -> np.ndarray:
    margins = [math.log(value / (1.0 - value)) for value in probabilities]
    return np.asarray([[0.0, margin] for margin in margins], dtype=np.float64)


class ComponentOracleTests(unittest.TestCase):
    def test_aggregate_top1_without_current_raw_agreement_is_not_a_legal_crossing(self) -> None:
        ws = windows(6)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1250, 1)
        s1 = stage1([0.95] * len(ws))
        s2 = np.zeros((len(ws), 4), dtype=np.float64)
        # w1 建立真类 1 证据；w2 的候选 EWMA 仍高置信指向类 1，但 raw top-1
        # 已转为类 2。stable_windows=1 要求二者一致，所以 w2 不得提交。
        s2[1, 0] = 10.0
        s2[2, 1] = 6.0
        s2[3:, 0] = 10.0
        anchor = logit_candidate_decisions(ws, s1, s2, anchor_config())
        self.assertEqual(anchor.trace[2].stage2_top_class, 1)
        self.assertGreater(anchor.trace[2].stage2_top_probability, 0.55)
        self.assertGreater(anchor.trace[2].stage2_probability_gap, 0.15)
        self.assertEqual(anchor.trace[2].stage2_stable_windows, 0)
        self.assertEqual(anchor.policy.decisions[2].emitted_class, -1)
        self.assertEqual(anchor.policy.decisions[3].emitted_class, 1)

    def test_all_model_cell_is_exact_anchor_and_truth_commit_skips_wrong_crossing(self) -> None:
        ws = windows(8)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1500, 1)
        s1 = stage1([0.95] * len(ws))
        s2 = np.zeros((len(ws), 4), dtype=np.float64)
        # w0 为开门窗；w1+w2 的 EWMA 先高置信提交错误类别 2，w3 再转为真类 1。
        s2[1:3, 1] = 8.0
        s2[3:, 0] = 20.0
        config = anchor_config()
        anchor = logit_candidate_decisions(ws, s1, s2, config)

        results = component_oracle_replays(
            ws, [event], s1, s2, config, pretruth_anchor=anchor,
        )
        self.assertEqual(set(results), {cell.cell_id for cell in ORACLE_CELLS})
        self.assertEqual(results[MODEL_CELL_ID].decisions, anchor.policy.decisions)
        model_commands = [row for row in results[MODEL_CELL_ID].decisions if row.emitted_class != -1]
        truth_commands = [row for row in results[ALL_TRUTH_CELL_ID].decisions if row.emitted_class != -1]
        self.assertEqual(model_commands[0].emitted_class, 2)
        self.assertEqual(truth_commands[0].emitted_class, 1)
        self.assertGreater(truth_commands[0].window_index, model_commands[0].window_index)
        self.assertEqual(results[ALL_TRUTH_CELL_ID].optimized_correct_count, 1)

    def test_truth_selected_dynamic_program_does_not_emit_same_event_twice(self) -> None:
        ws = windows(14)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1800, 1)
        # Stage 1 的模型复位会在 MI 中间短暂降至 IDLE，随后可再次开门。
        probabilities = [0.95, 0.95, 0.95, 0.05, 0.05] + [0.95] * 9
        s1 = stage1(probabilities)
        s2 = np.zeros((len(ws), 4), dtype=np.float64)
        s2[:, 0] = 10.0
        results = component_oracle_replays(ws, [event], s1, s2, anchor_config())
        truth_commit_cell = next(
            cell.cell_id for cell in ORACLE_CELLS
            if not cell.stage1_truth and cell.commit_truth and not cell.reset_truth
        )
        commands = [
            row for row in results[truth_commit_cell].decisions
            if row.emitted_class != -1
        ]
        self.assertEqual(len(commands), 1)
        self.assertEqual(results[truth_commit_cell].optimized_correct_count, 1)


class EvidenceCeilingTests(unittest.TestCase):
    def test_convex_combination_can_recover_class_absent_from_all_raw_top1(self) -> None:
        logits = np.asarray([
            [0.0, 1.0, -2.0, -2.0],
            [0.0, -2.0, 1.0, -2.0],
            [0.0, -2.0, -2.0, 1.0],
        ])
        self.assertTrue(all(int(np.argmax(row)) != 0 for row in logits))
        self.assertGreater(_max_convex_margin(logits, 1), 0.0)

        ws = windows(3)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1000, 1)
        baseline = [{
            "event_id": "event0", "subject_id": 1, "session_id": 0,
            "run_id": 0, "segment_id": 0, "outcome": "miss",
            "latency_seconds": None,
        }]
        result = stage2_evidence_ceiling(ws, [event], logits, baseline)
        self.assertEqual(result["summary"]["raw_top1"]["available_count"], 0)
        self.assertEqual(result["summary"]["convex_logit_top1"]["available_count"], 1)
        self.assertEqual(
            result["summary"]["convex_logit_top1"]["recoverable_baseline_miss_count"],
            1,
        )

    def test_shapley_contributions_share_interactions_and_sum_to_total_gap(self) -> None:
        weights = (0.1, 0.2, 0.3)
        values = {
            cell.cell_id: 0.25 + sum(bit * weight for bit, weight in zip(cell.bits, weights))
            for cell in ORACLE_CELLS
        }
        result = shapley_component_contributions(values)
        self.assertAlmostEqual(result["total_oracle_gap"], 0.6)
        for name, expected in zip(("stage1", "commit", "reset"), weights):
            self.assertAlmostEqual(result["contributions"][name], expected)
        self.assertAlmostEqual(
            sum(result["contributions"].values()), result["total_oracle_gap"],
        )


class OracleRunnerContractTests(unittest.TestCase):
    def test_repository_contract_and_anchor_are_exact(self) -> None:
        contract = load_oracle_contract(DEFAULT_ORACLE_CONFIG)
        self.assertEqual(contract["protocol_id"], EXPECTED_OUTPUT_PROTOCOL)
        self.assertEqual(load_anchor(DEFAULT_ANCHOR_CONFIG).cell_id, "c055_r020_l1")
        hashes = _source_hashes()
        self.assertIn("online_truth_inventory", hashes)
        self.assertIn("frozen_input_reader", hashes)
        self.assertIn("bundle_reader", hashes)

    def test_contract_tampering_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_ORACLE_CONFIG.read_text(encoding="utf-8"))
        payload["component_axes"]["reset"]["truth_oracle"] = "reset immediately"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oracle.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结 v1 合同"):
                load_oracle_contract(path)

    def test_minimal_seed_run_writes_metrics_and_trajectory_files(self) -> None:
        ws = windows(8)
        events = [MIEvent("event0", 1, 0, 0, 0, 0, 1250, 1)]
        segments = [ScoringSegment(1, 0, 0, 0, 0, 1375)]
        inventory = SimpleNamespace(windows=ws, events=events, segments=segments)
        s1 = stage1([0.95] * len(ws))
        s2 = np.zeros((len(ws), 4), dtype=np.float64)
        s2[:, 0] = 10.0
        anchor_cell = load_anchor(DEFAULT_ANCHOR_CONFIG)
        anchor = logit_candidate_decisions(ws, s1, s2, anchor_cell.logit_config)
        baseline = evaluate_online_events(
            segments, events, ws, anchor.policy.decisions, mode=STATEFUL_CANDIDATE,
        )
        contract = {"inventory": {
            "segment_inventory_sha256": baseline["scoring_segment_inventory_sha256"],
            "window_inventory_sha256": baseline["expected_window_inventory_sha256"],
            "event_inventory_sha256": baseline["event_inventory_sha256"],
            "event_count": baseline["event_count"],
        }}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, artifacts = _run_seed(
                root, inventory, contract, 42, s1, s2,
                {"file": "external_scores.npz", "sha256": "a" * 64},
                anchor_cell, anchor,
            )
            self.assertEqual(set(summary["cells"]), {cell.cell_id for cell in ORACLE_CELLS})
            self.assertTrue((root / artifacts["metrics"]["file"]).is_file())
            trajectory = root / artifacts["trajectories"]["file"]
            self.assertTrue(trajectory.is_file())
            with np.load(trajectory, allow_pickle=False) as payload:
                self.assertIn(f"{MODEL_CELL_ID}_emitted", payload.files)
                self.assertEqual(len(payload[f"{MODEL_CELL_ID}_emitted"]), len(ws))


if __name__ == "__main__":
    unittest.main()
