"""连续五分类 GRU 训练、恢复和产物哈希测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from full_control_gru_policy import (  # noqa: E402
    ContinuousNormalizer,
    ContinuousTensorSet,
)
from full_control_gru_training import (  # noqa: E402
    continuous_tensor_hash,
    file_hash,
    load_trained_model,
    train_final_job,
    train_inner_pair_job,
)
from ld_gru_training import TrainingHyperparameters  # noqa: E402
from full_control_gru_policy import ContinuousSequence  # noqa: E402
from run_full_control_gru_nested_loso import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    prepare_split,
)


def small_dataset() -> ContinuousTensorSet:
    tokens = np.zeros((2, 5, 10), dtype=np.float32)
    tokens[0, :, 0] = np.linspace(-1.0, 1.0, 5)
    tokens[1, :, 2] = np.linspace(1.0, -1.0, 5)
    targets = np.asarray([[0, 1, 2, 3, 4], [0, 4, 3, 2, 1]], dtype=np.int64)
    valid = np.ones((2, 5), dtype=np.bool_)
    return ContinuousTensorSet(
        tokens,
        targets,
        valid,
        ((1, 0, 0, 0), (1, 0, 1, 0)),
    )


class FullControlTrainingTests(unittest.TestCase):
    def test_config_is_hash_frozen_and_rejects_threshold_drift(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        modified = json.loads(json.dumps(config))
        modified["threshold_selection"]["commit_grid"][0] = 0.2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结协议"):
                load_config(path)

    def test_split_normalizer_never_uses_validation_subject(self) -> None:
        targets = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        cache = {}
        for subject, value in ((1, 0.0), (2, 2.0), (3, 100.0)):
            sequence = ContinuousSequence(
                (subject, 0, 0, 0),
                subject,
                np.full((5, 10), value, dtype=np.float32),
                targets.copy(),
            )
            cache[(subject, 42)] = (sequence,)
        split = prepare_split(cache, 42, (1, 2), (3,))
        np.testing.assert_allclose(split.normalizer.mean, 1.0)
        self.assertEqual(split.train_subjects, (1, 2))
        self.assertEqual(split.validation_subjects, (3,))

    def test_tensor_hash_binds_segment_identity_and_values(self) -> None:
        dataset = small_dataset()
        changed = ContinuousTensorSet(
            dataset.tokens.copy(), dataset.targets.copy(), dataset.valid_mask.copy(),
            ((1, 0, 0, 1), (1, 0, 1, 0)),
        )
        self.assertNotEqual(continuous_tensor_hash(dataset), continuous_tensor_hash(changed))

    def test_inner_and_final_jobs_resume_and_load(self) -> None:
        dataset = small_dataset()
        normalizer = ContinuousNormalizer(
            np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
        )
        hyper = TrainingHyperparameters(
            batch_size=64,
            max_epochs=2,
            early_stopping_patience=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"kind": "full_control_unit_inner", "subjects": [1, 2]}
            inner = train_inner_pair_job(
                root / "inner", dataset, {1: dataset, 2: dataset}, normalizer,
                decision_seed=2042, hyperparameters=hyper,
                device=torch.device("cpu"), contract=contract, verbose=False,
            )
            repeated = train_inner_pair_job(
                root / "inner", dataset, {1: dataset, 2: dataset}, normalizer,
                decision_seed=2042, hyperparameters=hyper,
                device=torch.device("cpu"), contract=contract, verbose=False,
            )
            self.assertEqual(inner, repeated)
            self.assertEqual(inner["parameter_count"], 1429)
            for artifact in inner["best_checkpoints"].values():
                self.assertEqual(
                    file_hash(root / "inner" / artifact["file"]), artifact["sha256"],
                )

            final = train_final_job(
                root / "final", dataset, normalizer,
                decision_seed=2042, fixed_epochs=1, hyperparameters=hyper,
                device=torch.device("cpu"),
                contract={"kind": "full_control_unit_final"}, verbose=False,
            )
            checkpoint = root / "final" / final["final_checkpoint"]["file"]
            self.assertEqual(file_hash(checkpoint), final["final_checkpoint"]["sha256"])
            model = load_trained_model(checkpoint, torch.device("cpu"))
            self.assertEqual(sum(value.numel() for value in model.parameters()), 1429)


if __name__ == "__main__":
    unittest.main()
