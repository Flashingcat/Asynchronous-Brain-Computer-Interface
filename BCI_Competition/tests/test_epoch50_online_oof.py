"""固定 epoch50 连续 OOF runner 的库存和单窗决策回归测试。"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from protocol_metrics import (  # noqa: E402
    READY,
    STATEFUL_STRICT,
    STATELESS_DIAGNOSTIC,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    ScoringSegment,
    evaluate_online_events,
)
from run_epoch50_online_oof import (  # noqa: E402
    build_online_inventory,
    bundle_contract_version,
    default_subject_paths,
    inventory_contract_protocol_id,
    output_window_rows,
    stateful_argmax_decisions,
    stateless_argmax_decisions,
    verify_inventory_contract,
)


WINDOW_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"), ("segment", "u1"),
    ("window", "<u4"), ("start", "<i8"), ("stop", "<i8"),
    ("trial", "<i2"), ("final_label", "u1"), ("stage1_label", "u1"),
    ("stage2_label", "i1"), ("is_task", "?"),
])


def fake_context(subject: int = 1) -> SimpleNamespace:
    """构造一个不含测试 session、只含一个干净 MI 事件的最小 bundle。"""
    rows = np.zeros(5, dtype=WINDOW_DTYPE)
    for index, start in enumerate(range(500, 1125, 125)):
        rows[index] = (subject, 0, 0, 0, index, start, start + 500, 7, 3, 1, 2, True)
    manifest = {
        "protocol_id": f"bnci2014001_s{subject:02d}_oof_train_session0_native250_v2",
        "subject": subject,
        "included_session": 0,
        "artifact_policy": "official_trial_exclusion",
        "segment_policy": "separate_clean_segments_no_time_compression",
        "test_session_content_in_bundle": False,
        "index_sha256": "b" * 64,
        "domains": {
            "causal": {
                "segments": [{
                    "session": 0, "run": 0, "segment": 0,
                    "formal_start_native": 0, "formal_stop_native": 2000,
                }],
            },
        },
    }
    return SimpleNamespace(manifest=manifest, manifest_sha256="a" * 64, rows=rows)


class InventoryTests(unittest.TestCase):
    def test_bundle_contract_version_separates_legacy_and_explicit_manifests(self) -> None:
        explicit = fake_context().manifest
        self.assertEqual(bundle_contract_version(explicit), "v2")
        self.assertTrue(inventory_contract_protocol_id(explicit).endswith("_v2"))

        legacy = copy.deepcopy(explicit)
        legacy["protocol_id"] = "bnci2014001_s01_oof_train_session0_native250_v1"
        legacy.pop("artifact_policy")
        legacy.pop("segment_policy")
        self.assertEqual(bundle_contract_version(legacy), "v1")
        self.assertTrue(inventory_contract_protocol_id(legacy).endswith("_v1"))

    def test_session0_bundle_deterministically_restores_inventory(self) -> None:
        inventory = build_online_inventory(fake_context())

        self.assertEqual(len(inventory.segments), 1)
        self.assertEqual(len(inventory.windows), 13)
        self.assertEqual(len(inventory.events), 1)
        self.assertEqual(inventory.events[0].event_id, "s0_r0_t7")
        self.assertEqual(inventory.events[0].onset_sample, 500)
        self.assertEqual(inventory.events[0].offset_sample, 1500)
        self.assertEqual(inventory.events[0].true_class, 3)
        self.assertTrue(np.all(inventory.signal_rows["session"] == 0))
        self.assertTrue(np.all(inventory.signal_rows["trial"] == -1))
        self.assertTrue(np.all(inventory.signal_rows["stage2_label"] == -1))

    def test_subject9_uses_its_own_identity_and_paths(self) -> None:
        inventory = build_online_inventory(fake_context(subject=9))
        self.assertTrue(all(item.subject_id == 9 for item in inventory.segments))
        self.assertTrue(all(item.subject_id == 9 for item in inventory.events))
        self.assertTrue(all(item.subject_id == 9 for item in inventory.windows))
        self.assertTrue(np.all(inventory.signal_rows["subject"] == 9))

        paths = default_subject_paths(9)
        self.assertIn("s09_oof_train_session0", str(paths.bundle_manifest))
        self.assertIn("extension_s09", str(paths.checkpoint_root))
        self.assertIn("s09_session0_causal_online", str(paths.inventory_contract))

    def test_fully_warmup_excluded_segment_is_counted_not_scored(self) -> None:
        context = fake_context()
        context.rows["segment"] = 1
        context.manifest["domains"]["causal"]["segments"] = [
            {
                "session": 0, "run": 0, "segment": 0,
                "start_native": 0, "stop_native": 250,
                "formal_start_native": 250, "formal_stop_native": 250,
            },
            {
                "session": 0, "run": 0, "segment": 1,
                "start_native": 250, "stop_native": 2000,
                "formal_start_native": 250, "formal_stop_native": 2000,
            },
        ]
        inventory = build_online_inventory(context)
        self.assertEqual(len(inventory.segments), 1)
        self.assertEqual(inventory.segments[0].segment_id, 1)
        self.assertEqual(inventory.fully_warmup_excluded_segment_count, 1)
        self.assertEqual(inventory.fully_warmup_excluded_samples, 250)
        self.assertTrue(all(event.segment_id == 1 for event in inventory.events))

    def test_frozen_contract_detects_inventory_change(self) -> None:
        context = fake_context()
        inventory = build_online_inventory(context)
        decisions = [
            DecisionRecord(
                *window.key, window.window_index,
                window.window_start_sample, window.window_stop_sample,
            )
            for window in inventory.windows
        ]
        baseline = evaluate_online_events(
            inventory.segments, inventory.events, inventory.windows, decisions,
            mode=STATELESS_DIAGNOSTIC,
        )
        contract = {
            "protocol_id": "bnci2014001_s01_session0_causal_online_v2",
            "subject": 1,
            "included_session": 0,
            "test_session_access": "forbidden",
            "native_sampling_rate": 250,
            "window_samples": 500,
            "step_samples": 125,
            "event_margin_samples": 125,
            "artifact_policy": "official_trial_exclusion",
            "segment_policy": "separate_clean_segments_no_time_compression",
            "artifact_policy_binding": "explicit_bundle_manifest",
            "source_bundle": {
                "protocol_id": context.manifest["protocol_id"],
                "manifest_sha256": context.manifest_sha256,
                "index_sha256": context.manifest["index_sha256"],
            },
            "inventory": {
                "segment_count": baseline["scoring_segment_count"],
                "segment_inventory_sha256": baseline["scoring_segment_inventory_sha256"],
                "fully_warmup_excluded_segment_count": 0,
                "fully_warmup_excluded_samples": 0,
                "zero_window_segment_count": baseline["zero_window_segment_count"],
                "zero_window_segment_samples": baseline["zero_window_segment_samples"],
                "trailing_unwindowed_samples": baseline["trailing_unwindowed_samples"],
                "window_count": baseline["expected_window_count"],
                "window_inventory_sha256": baseline["expected_window_inventory_sha256"],
                "event_count": baseline["event_count"],
                "event_inventory_sha256": baseline["event_inventory_sha256"],
                "valid_idle_seconds": baseline["valid_idle_seconds"],
            },
            "per_run_window_count": {str(run): 13 if run == 0 else 0 for run in range(6)},
            "per_run_event_count": {str(run): 1 if run == 0 else 0 for run in range(6)},
        }
        verified = verify_inventory_contract(context, inventory, contract)
        self.assertEqual(verified["miss_rate"], 1.0)

        changed = copy.deepcopy(contract)
        changed["inventory"]["window_count"] -= 1
        with self.assertRaisesRegex(RuntimeError, "冻结合同"):
            verify_inventory_contract(context, inventory, changed)


class SingleWindowPolicyTests(unittest.TestCase):
    @staticmethod
    def windows() -> list[ExpectedWindow]:
        first = [ExpectedWindow(1, 0, 0, 0, index, index * 125, index * 125 + 500)
                 for index in range(5)]
        second = [ExpectedWindow(1, 0, 0, 1, index, 2000 + index * 125, 2500 + index * 125)
                  for index in range(2)]
        return first + second

    def test_stateless_and_stateful_argmax_have_distinct_outputs(self) -> None:
        windows = self.windows()
        # Stage 1 序列：IDLE、Task、Task、IDLE、Task；新 segment 首窗重新从 READY 开始。
        stage1 = np.asarray([
            [0, 0], [0, 2], [0, 2], [2, 0], [0, 2], [0, 2], [2, 0],
        ], dtype=np.float32)
        stage2_class = [1, 2, 3, 1, 4, 2, 3]
        stage2 = np.full((7, 4), -1.0, dtype=np.float32)
        for index, class_id in enumerate(stage2_class):
            stage2[index, class_id - 1] = 2.0

        stateless = stateless_argmax_decisions(windows, stage1, stage2)
        stateful = stateful_argmax_decisions(windows, stage1, stage2)
        self.assertEqual([item.emitted_class for item in stateless], [-1, 2, 3, -1, 4, 2, -1])
        self.assertEqual([item.emitted_class for item in stateful], [-1, 2, -1, -1, 4, 2, -1])
        self.assertEqual(
            [(item.decision_state_before, item.decision_state_after) for item in stateful[:5]],
            [
                (READY, READY),
                (READY, WAIT_IDLE),
                (WAIT_IDLE, WAIT_IDLE),
                (WAIT_IDLE, READY),
                (READY, WAIT_IDLE),
            ],
        )
        self.assertEqual(stateful[5].decision_state_before, READY)
        self.assertEqual(stateful[5].decision_state_after, WAIT_IDLE)

        # 两个 segment 的完整状态轨迹还必须通过独立基础评估器的严格检查。
        result = evaluate_online_events(
            [
                # 第一段 5 个窗，第二段 2 个窗，尾部均小于一个步长。
                ScoringSegment(1, 0, 0, 0, 0, 1000),
                ScoringSegment(1, 0, 0, 1, 2000, 2625),
            ],
            [], windows, stateful, mode=STATEFUL_STRICT,
        )
        self.assertEqual(result["emitted_command_count"], 3)

    def test_invalid_logit_shape_fails_before_decision(self) -> None:
        windows = self.windows()
        with self.assertRaisesRegex(ValueError, "逐窗完整"):
            stateful_argmax_decisions(windows, np.zeros((6, 2)), np.zeros((7, 4)))

    def test_output_rows_explicitly_store_decision_time(self) -> None:
        rows = output_window_rows(self.windows())
        self.assertEqual(rows.dtype.names, (
            "subject_id", "session_id", "run_id", "segment_id", "window_index",
            "window_start_native", "window_stop_native",
            "window_start_model", "window_stop_model", "decision_time_seconds",
        ))
        np.testing.assert_array_equal(
            rows["window_start_native"], rows["window_start_model"],
        )
        np.testing.assert_array_equal(
            rows["window_stop_native"], rows["window_stop_model"],
        )
        np.testing.assert_array_equal(
            rows["decision_time_seconds"], rows["window_stop_native"] / 250.0,
        )


if __name__ == "__main__":
    unittest.main()
