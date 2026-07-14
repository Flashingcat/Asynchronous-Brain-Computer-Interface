from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from candidate_latency_diagnostics import diagnose_candidate_latency  # noqa: E402
from logit_candidate_strategies import (  # noqa: E402
    LogitStrategyConfig,
    logit_candidate_decisions,
)
from protocol_metrics import (  # noqa: E402
    MIEvent,
    STATEFUL_CANDIDATE,
    ExpectedWindow,
    ScoringSegment,
    evaluate_online_events,
)
from run_candidate_latency_diagnostic import (  # noqa: E402
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_DIAGNOSTIC_CONFIG,
    EXPECTED_OUTPUT_PROTOCOL,
    _source_hashes,
    _verify_child,
    load_anchor,
    load_latency_contract,
)
from run_epoch50_online_oof import file_hash  # noqa: E402


def _windows() -> list[ExpectedWindow]:
    return [
        ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
        for index in range(5)
    ]


def _stage1(probabilities: list[float]) -> np.ndarray:
    margins = [math.log(value / (1.0 - value)) for value in probabilities]
    return np.asarray([[0.0, margin] for margin in margins], dtype=np.float64)


def _config() -> LogitStrategyConfig:
    return LogitStrategyConfig.from_dict({
        "strategy_id": "latency_anchor",
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
    })


def _diagnose(
    event_offset: int,
    stage2_class: int = 1,
    stage2_logits: np.ndarray | None = None,
) -> tuple[dict, tuple, tuple]:
    windows = _windows()
    segment = ScoringSegment(1, 0, 0, 0, 0, 1000)
    event = MIEvent("event0", 1, 0, 0, 0, 0, event_offset, 1)
    stage2 = np.zeros((len(windows), 4), dtype=np.float64)
    stage2[:, stage2_class - 1] = 8.0
    if stage2_logits is not None:
        stage2 = np.asarray(stage2_logits, dtype=np.float64)
    strategy = logit_candidate_decisions(
        windows,
        _stage1([0.9, 0.9, 0.9, 0.1, 0.1]),
        stage2,
        _config(),
        idle_reset_consecutive_windows=1,
    )
    evaluated = evaluate_online_events(
        [segment], [event], windows, strategy.policy.decisions,
        mode=STATEFUL_CANDIDATE,
    )
    before = tuple(strategy.policy.decisions)
    diagnostics = diagnose_candidate_latency(
        [event], windows, stage2, strategy, evaluated,
        stage2_alpha=0.5,
        stage2_top_probability=0.55,
        stage2_probability_gap=0.15,
        baseline_min_candidate_windows=2,
    )
    return diagnostics, before, tuple(strategy.policy.decisions)


class CandidateLatencyDiagnosticTests(unittest.TestCase):
    def test_opening_window_oracles_expose_half_and_one_second_headroom(self) -> None:
        diagnostics, before, after = _diagnose(event_offset=1000)
        row = diagnostics["event_rows"][0]
        summary = diagnostics["summary"]
        self.assertEqual(row["baseline_outcome"], "correct")
        self.assertEqual(row["baseline_candidate_open_sample"], 500)
        self.assertEqual(row["baseline_decision_sample"], 750)
        self.assertEqual(row["open_ewma_correct_confident_min1_sample"], 500)
        self.assertEqual(row["open_ewma_correct_confident_min2_sample"], 625)
        self.assertEqual(summary["fixed_exclude_open_plus_min2_wait_seconds"], 1.0)
        self.assertEqual(
            summary["paired_baseline_correct"]["open_ewma_correct_confident_min1"]
            ["positive_headroom_seconds"]["median"],
            1.0,
        )
        self.assertEqual(
            summary["paired_baseline_correct"]["open_ewma_correct_confident_min2"]
            ["positive_headroom_seconds"]["median"],
            0.5,
        )
        # 诊断只读完整轨迹，不能反向改写任何线上决策。
        self.assertEqual(before, after)

    def test_correct_class_post_mi_spillover_can_be_truth_aware_rescued(self) -> None:
        diagnostics, _, _ = _diagnose(event_offset=700)
        row = diagnostics["event_rows"][0]
        spillover = diagnostics["summary"]["correct_class_post_mi_spillover"]
        self.assertEqual(row["baseline_outcome"], "miss")
        self.assertEqual(row["correct_class_spillover_sample"], 750)
        self.assertEqual(spillover["event_count"], 1)
        self.assertEqual(
            spillover["rescue"]["open_ewma_correct_confident_min1"]
            ["truth_aware_rescuable_count"],
            1,
        )
        self.assertEqual(
            spillover["rescue"]["open_ewma_correct_confident_min2"]
            ["truth_aware_rescuable_count"],
            1,
        )

    def test_already_triggered_event_is_not_counted_as_rescued_miss(self) -> None:
        windows = [
            ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
            for index in range(8)
        ]
        segment = ScoringSegment(1, 0, 0, 0, 0, 1375)
        event = MIEvent("event0", 1, 0, 0, 0, 0, 1200, 1)
        stage2 = np.zeros((len(windows), 4), dtype=np.float64)
        stage2[:, 0] = 8.0
        strategy = logit_candidate_decisions(
            windows,
            _stage1([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1]),
            stage2,
            _config(),
            idle_reset_consecutive_windows=1,
        )
        evaluated = evaluate_online_events(
            [segment], [event], windows, strategy.policy.decisions,
            mode=STATEFUL_CANDIDATE,
        )
        diagnostics = diagnose_candidate_latency(
            [event], windows, stage2, strategy, evaluated,
            stage2_alpha=0.5,
            stage2_top_probability=0.55,
            stage2_probability_gap=0.15,
            baseline_min_candidate_windows=2,
        )
        spillover = diagnostics["summary"]["correct_class_post_mi_spillover"]
        self.assertEqual(diagnostics["event_rows"][0]["baseline_outcome"], "correct")
        self.assertEqual(spillover["event_count"], 1)
        self.assertEqual(spillover["baseline_miss_event_count"], 0)
        self.assertEqual(spillover["already_triggered_event_count"], 1)
        self.assertEqual(
            spillover["rescue"]["open_ewma_correct_confident_min1"]
            ["truth_aware_rescuable_count"],
            0,
        )
        self.assertIsNone(
            spillover["rescue"]["open_ewma_correct_confident_min1"]
            ["truth_aware_rescuable_rate"],
        )

    def test_wrong_top_class_does_not_become_truth_aware_oracle(self) -> None:
        diagnostics, _, _ = _diagnose(event_offset=1000, stage2_class=2)
        row = diagnostics["event_rows"][0]
        self.assertEqual(row["baseline_outcome"], "wrong_class")
        self.assertIsNone(row["raw_correct_top1_sample"])
        self.assertIsNone(row["open_ewma_correct_confident_min1_sample"])
        self.assertEqual(row["open_ewma_first_confident_min1_predicted_class"], 2)
        self.assertFalse(row["open_ewma_first_confident_min1_correct"])

    def test_label_free_first_crossing_exposes_wrong_evidence_before_truth_oracle(self) -> None:
        stage2 = np.zeros((5, 4), dtype=np.float64)
        stage2[0:2, 1] = 8.0
        stage2[2:, 0] = 30.0
        diagnostics, _, _ = _diagnose(event_offset=1000, stage2_logits=stage2)
        row = diagnostics["event_rows"][0]
        crossings = diagnostics["summary"]["label_free_first_crossings"]
        self.assertEqual(row["baseline_outcome"], "correct")
        self.assertEqual(row["open_ewma_first_confident_min1_sample"], 500)
        self.assertEqual(row["open_ewma_first_confident_min2_sample"], 625)
        self.assertFalse(row["open_ewma_first_confident_min1_correct"])
        self.assertFalse(row["open_ewma_first_confident_min2_correct"])
        self.assertEqual(row["open_ewma_correct_confident_min2_sample"], 750)
        self.assertEqual(
            crossings["open_ewma_first_confident_min2"]
            ["truth_oracle_preceded_by_wrong_crossing_count"],
            1,
        )

    def test_filter_delay_is_included_but_wall_clock_is_not_claimed(self) -> None:
        diagnostics, _, _ = _diagnose(event_offset=1000)
        clock = diagnostics["latency_clock"]
        self.assertEqual(
            clock["causal_filter_group_delay"],
            "included_in_end_to_end_score_timing_and_not_subtracted",
        )
        self.assertEqual(
            clock["filter_model_state_machine_wall_clock_compute"],
            "not_measured",
        )
        semantics = diagnostics["counterfactual_semantics"]["label_free_first_crossings"]
        self.assertIn("without true_class", semantics)
        self.assertIn("truth-defined event", semantics)


class CandidateLatencyRunnerContractTests(unittest.TestCase):
    def test_repository_contract_and_historical_anchor_are_exact(self) -> None:
        contract = load_latency_contract(DEFAULT_DIAGNOSTIC_CONFIG)
        _, anchor = load_anchor(DEFAULT_ANCHOR_CONFIG)
        self.assertEqual(contract["protocol_id"], EXPECTED_OUTPUT_PROTOCOL)
        self.assertEqual(anchor.cell_id, "c055_r020_l1")
        self.assertEqual(anchor.logit_config.stage2_min_candidate_windows, 2)
        source_hashes = _source_hashes()
        self.assertEqual(len(source_hashes), 12)
        self.assertIn("single_window_multi_subject_verifier", source_hashes)

    def test_test_session_or_oracle_semantic_tampering_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_DIAGNOSTIC_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            payload["included_session"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结首版合同"):
                load_latency_contract(path)

            payload = json.loads(DEFAULT_DIAGNOSTIC_CONFIG.read_text(encoding="utf-8"))
            payload["oracles"].remove("open_ewma_correct_confident_min1")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结首版合同"):
                load_latency_contract(path)

    def test_child_manifest_rejects_source_contract_drift(self) -> None:
        source_hashes = _source_hashes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def artifact(name: str) -> dict[str, str]:
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                return {"file": name, "sha256": file_hash(path)}

            manifest = {
                "status": "PASS",
                "claim_status": "PRECOMMIT_DIAGNOSTIC_ONLY",
                "protocol_id": EXPECTED_OUTPUT_PROTOCOL,
                "selection_status": "none_diagnostic_only",
                "truth_usage": "post_inference_oracle_diagnostics_only",
                "subject": 1,
                "seeds": [42, 43, 44],
                "included_session": 0,
                "test_session_access": "forbidden_and_not_loaded",
                "source_sha256": source_hashes,
                "diagnostic_config_sha256": "a" * 64,
                "anchor_config_sha256": "b" * 64,
                "anchor": {"cell_id": "anchor"},
                "run_log": artifact("run_log.json"),
                "seed_artifacts": {},
            }
            for seed in (42, 43, 44):
                manifest["seed_artifacts"][str(seed)] = {
                    "input_scores": {"file": "external.npz", "sha256": "c" * 64},
                    "diagnostics": artifact(f"seed{seed}_diagnostics.json"),
                }
            _verify_child(
                root, manifest, subject=1, source_hashes=source_hashes,
                diagnostic_config_sha256="a" * 64,
                anchor_config_sha256="b" * 64,
                claim_status="PRECOMMIT_DIAGNOSTIC_ONLY",
                anchor_public_config={"cell_id": "anchor"},
            )
            changed = {**manifest, "source_sha256": {**source_hashes, "protocol_metrics": "f" * 64}}
            with self.assertRaisesRegex(RuntimeError, "合同非法"):
                _verify_child(
                    root, changed, subject=1, source_hashes=source_hashes,
                    diagnostic_config_sha256="a" * 64,
                    anchor_config_sha256="b" * 64,
                    claim_status="PRECOMMIT_DIAGNOSTIC_ONLY",
                    anchor_public_config={"cell_id": "anchor"},
                )


if __name__ == "__main__":
    unittest.main()
