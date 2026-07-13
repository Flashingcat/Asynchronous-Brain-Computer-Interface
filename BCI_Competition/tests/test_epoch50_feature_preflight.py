"""EEGNet 隐藏特征预检的接口、结构和冻结分数绑定测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
MODELS_DIR = PROJECT_ROOT / "code" / "models"
for source_dir in (EVAL_DIR, MODELS_DIR):
    sys.path.insert(0, str(source_dir))

from model_factory import build_model  # noqa: E402
from run_epoch50_feature_preflight import (  # noqa: E402
    EEGNET_FEATURE_DIM,
    load_frozen_scores,
    predict_logits_and_features,
)
from run_epoch50_online_oof import OUTPUT_WINDOW_DTYPE  # noqa: E402


class FeatureInterfaceTests(unittest.TestCase):
    def test_eegnet_feature_api_preserves_adapter_logits(self) -> None:
        torch.manual_seed(7)
        model = build_model("eegnet", 4, 22, 500).eval()
        inputs = np.random.default_rng(7).normal(size=(3, 22, 500)).astype(np.float32)

        logits, features = predict_logits_and_features(
            model, inputs, torch.device("cpu"), classes=4,
        )

        self.assertEqual(logits.shape, (3, 4))
        self.assertEqual(features.shape, (3, EEGNET_FEATURE_DIM))
        self.assertTrue(np.isfinite(features).all())
        with torch.inference_mode():
            expected = model(torch.from_numpy(inputs)).numpy()
        np.testing.assert_array_equal(logits, expected)

    def test_non_eegnet_wrapper_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "固定 EEGNet"):
            predict_logits_and_features(
                torch.nn.Linear(5, 2),
                np.zeros((1, 22, 500), dtype=np.float32),
                torch.device("cpu"),
                classes=2,
            )


class FrozenScoreBindingTests(unittest.TestCase):
    @staticmethod
    def rows() -> np.ndarray:
        rows = np.zeros(2, dtype=OUTPUT_WINDOW_DTYPE)
        rows[0]["window_index"] = 0
        rows[1]["window_index"] = 1
        return rows

    def test_frozen_scores_require_exact_window_rows(self) -> None:
        rows = self.rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez_compressed(
                path,
                window_rows=rows,
                stage1_logits=np.zeros((2, 2), dtype=np.float32),
                stage2_logits=np.zeros((2, 4), dtype=np.float32),
                diagnostic=np.ones(2, dtype=np.int8),
            )
            stage1, stage2, fields = load_frozen_scores(path, rows.copy())
            self.assertEqual(stage1.shape, (2, 2))
            self.assertEqual(stage2.shape, (2, 4))
            self.assertIn("diagnostic", fields)

            changed = rows.copy()
            changed[1]["window_index"] = 2
            with self.assertRaisesRegex(RuntimeError, "逐窗绑定"):
                load_frozen_scores(path, changed)


if __name__ == "__main__":
    unittest.main()
