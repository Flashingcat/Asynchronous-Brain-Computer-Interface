"""隐藏特征 pilot 的配置、输入来源和无标签尺度合同测试。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from protocol_metrics import READY, TASK_CANDIDATE, ExpectedWindow  # noqa: E402
from run_candidate_feature_pilot import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_INPUT_PROTOCOL,
    EXPECTED_BASE_LOGIT_STRATEGY,
    EXPECTED_FEATURE_CELLS,
    load_contract,
    load_feature_input,
    summarize_feature_scale,
)
from run_epoch50_online_oof import file_hash  # noqa: E402


class PilotConfigTests(unittest.TestCase):
    def test_checked_in_contract_freezes_all_eight_cells(self) -> None:
        _, configs = load_contract(DEFAULT_CONFIG)
        self.assertEqual(
            [item.strategy_id for item in configs],
            [
                "logit_only_reference", "velocity_loose", "velocity_strict",
                "velocity_consecutive", "prototype_loose", "prototype_strict",
                "acceleration_loose", "acceleration_strict",
            ],
        )
        self.assertEqual(configs[0].feature_metric, "none")
        self.assertTrue(all(
            item.base_logit_strategy.strategy_id == "dual_ewma_drop_abort"
            for item in configs
        ))

    def test_protocol_identity_rejects_semantic_parameter_and_grid_tampering(self) -> None:
        original = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        changed_payloads = []
        semantic = copy.deepcopy(original)
        semantic["strategy_semantics"]["feature_role"] = "changed"
        changed_payloads.append(semantic)
        base = copy.deepcopy(original)
        base["base_logit_strategy"]["task_on_probability"] = 0.9
        changed_payloads.append(base)
        threshold = copy.deepcopy(original)
        threshold["strategies"][1]["feature_max_change"] = 0.123
        changed_payloads.append(threshold)
        shortened = copy.deepcopy(original)
        shortened["strategies"].pop()
        changed_payloads.append(shortened)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            for payload in changed_payloads:
                with self.subTest(change=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "机制合同"):
                        load_contract(path)

        self.assertEqual(original["base_logit_strategy"], EXPECTED_BASE_LOGIT_STRATEGY)
        self.assertEqual(original["strategies"], list(EXPECTED_FEATURE_CELLS))


class PilotInputTests(unittest.TestCase):
    def test_input_loader_binds_clean_manifest_hash_and_exact_fields(self) -> None:
        rows = np.zeros(2, dtype=[("window_index", "<u4")])
        rows[1]["window_index"] = 1
        fake_sources = {"runner": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features_and_logits.npz"
            np.savez_compressed(
                artifact_path,
                window_rows=rows,
                stage1_logits=np.ones((2, 2), dtype=np.float32),
                stage2_logits=np.ones((2, 4), dtype=np.float32),
                stage1_features=np.ones((2, 240), dtype=np.float32),
                stage2_features=np.ones((2, 240), dtype=np.float32),
            )
            manifest = {
                "status": "PASS",
                "claim_status": "FEATURE_EXTRACTION_PREFLIGHT_ONLY",
                "protocol_id": EXPECTED_INPUT_PROTOCOL,
                "subject": 1,
                "seed": 42,
                "included_session": 0,
                "test_session_access": "forbidden_and_not_loaded",
                "job_count": 12,
                "feature_contract": {
                    "dimension_per_stage": 240,
                    "strategy_or_threshold_selection": "none",
                    "decision_generation": "none",
                },
                "runtime": {"git": {"commit": "b" * 40, "dirty": False}},
                "source_sha256": fake_sources,
                "artifact": {
                    "file": artifact_path.name,
                    "sha256": file_hash(artifact_path),
                    "window_count": 2,
                },
            }
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            frozen = {
                "manifest_sha256": file_hash(manifest_path),
                "runtime_git": manifest["runtime"]["git"],
                "source_sha256": fake_sources,
            }
            with patch(
                "run_candidate_feature_pilot._frozen_feature_input_contract",
                return_value=frozen,
            ):
                arrays, _, identity = load_feature_input(root, rows)
                self.assertEqual(arrays["stage2_features"].shape, (2, 240))
                self.assertEqual(identity["artifact_sha256"], file_hash(artifact_path))

                changed = copy.deepcopy(manifest)
                changed["runtime"]["git"]["dirty"] = True
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "干净提交"):
                    load_feature_input(root, rows)


class FeatureScaleTests(unittest.TestCase):
    def test_segment_and_candidate_histories_reset_independently(self) -> None:
        windows = []
        decisions = []
        for segment, offset in ((0, 0), (1, 2000)):
            for index in range(4):
                windows.append(ExpectedWindow(
                    1, 0, 0, segment, index,
                    offset + 125 * index, offset + 125 * index + 500,
                ))
                decisions.append(SimpleNamespace(
                    decision_state_before=READY if index == 0 else TASK_CANDIDATE,
                ))
        stage1 = np.ones((8, 240), dtype=np.float64)
        stage2 = np.zeros((8, 240), dtype=np.float64)
        for index in range(8):
            stage2[index, index % 4] = 1.0
        scale = summarize_feature_scale(
            windows, stage1, stage2, SimpleNamespace(decisions=decisions),
        )
        fields = scale["fields"]
        self.assertFalse(scale["labels_or_events_used"])
        self.assertEqual(fields["stage2_segment_unit_velocity_l2"]["sample_count"], 6)
        self.assertEqual(fields["stage2_segment_unit_acceleration_l2"]["sample_count"], 4)
        self.assertEqual(fields["reference_candidate_unit_velocity_l2"]["sample_count"], 4)
        self.assertEqual(
            fields["reference_candidate_unit_prototype_cosine_distance"]["sample_count"], 4,
        )
        self.assertEqual(fields["reference_candidate_unit_acceleration_l2"]["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
