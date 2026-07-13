"""EEGNet OOF 训练矩阵、持久化与可复现基础组件测试。"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TRAIN_DIR = Path(__file__).resolve().parents[1] / "code" / "train"
sys.path.insert(0, str(TRAIN_DIR))

from oof_training_bundle import WINDOW_DTYPE, window_identity_hash  # noqa: E402
from train_eegnet_oof import (  # noqa: E402
    JobArrays,
    JobSpec,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    balanced_class_weights,
    build_job_specs,
    canonical_hash,
    capture_rng,
    classification_metrics,
    execution_fingerprint,
    restore_rng,
    run_job,
)


class MatrixTests(unittest.TestCase):
    def test_default_matrix_contains_72_unique_training_trajectories(self) -> None:
        specs = build_job_specs(
            1, list(range(6)), [1, 2], ["causal", "zero_phase"], [42, 43, 44]
        )
        self.assertEqual(len(specs), 72)
        self.assertEqual(len({spec.name for spec in specs}), 72)
        self.assertTrue(all(spec.subject == 1 for spec in specs))

    def test_validation_domains_do_not_create_duplicate_training_jobs(self) -> None:
        causal, zero = build_job_specs(
            1, [0], [1], ["causal", "zero_phase"], [42]
        )
        self.assertEqual(causal.validation_domains, ("causal",))
        self.assertEqual(zero.validation_domains, ("zero_phase", "causal"))

    def test_invalid_or_duplicate_matrix_axes_are_rejected(self) -> None:
        for arguments in (
            (1, [0, 0], [1], ["causal"], [42]),
            (1, [6], [1], ["causal"], [42]),
            (1, [0], [3], ["causal"], [42]),
            (1, [0], [1], ["bad"], [42]),
            (10, [0], [1], ["causal"], [42]),
        ):
            with self.assertRaises(ValueError):
                build_job_specs(*arguments)


class MetricAndStateTests(unittest.TestCase):
    def test_balanced_weights_and_metrics_are_independently_checkable(self) -> None:
        labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
        weights = balanced_class_weights(labels, 2, torch.device("cpu")).numpy()
        np.testing.assert_allclose(weights, [2 / 3, 2.0])
        logits = np.asarray([[2, 0], [2, 0], [0, 2], [0, 2]], np.float32)
        metrics = classification_metrics(labels, logits, 2)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (2 / 3 + 1) / 2)

    def test_rng_capture_restore_replays_python_numpy_and_torch(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng(torch.device("cpu"))
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state, torch.device("cpu"))
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_canonical_hash_ignores_dictionary_insertion_order(self) -> None:
        self.assertEqual(canonical_hash({"a": 1, "b": 2}),
                         canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))


class AtomicArtifactTests(unittest.TestCase):
    def test_json_and_npz_are_complete_and_leave_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "state.json", {"status": "PASS", "epoch": 2})
            atomic_npz(root / "scores.npz", {
                "logits": np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            })
            self.assertEqual(
                json.loads((root / "state.json").read_text(encoding="utf-8")),
                {"status": "PASS", "epoch": 2},
            )
            with np.load(root / "scores.npz", allow_pickle=False) as data:
                self.assertEqual(data["logits"].shape, (2, 3, 2))
                self.assertEqual(data["logits"].dtype, np.float32)
            self.assertEqual([path for path in root.iterdir() if path.suffix == ".tmp"], [])

    def test_torch_checkpoint_roundtrip_uses_windows_compatible_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "latest.pt"
            payload = {"epoch": 1, "weight": torch.arange(6).reshape(2, 3)}
            atomic_torch_save(path, payload)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["epoch"], 1)
            self.assertTrue(torch.equal(loaded["weight"], payload["weight"]))
            self.assertEqual([item for item in root.iterdir() if item.suffix == ".tmp"], [])


class SyntheticResumeTests(unittest.TestCase):
    @staticmethod
    def rows(count: int) -> np.ndarray:
        values = []
        for index in range(count):
            label = index % 2
            values.append((1, 0, 0, 0, index, index * 16, index * 16 + 64,
                           -1, label, label, -1, bool(label)))
        return np.asarray(values, dtype=WINDOW_DTYPE)

    def setUp(self) -> None:
        generator = np.random.default_rng(123)
        validation_rows = self.rows(8)
        self.arrays = JobArrays(
            train_x=generator.normal(size=(16, 22, 64)).astype(np.float32),
            train_y=np.asarray([0, 1] * 8, dtype=np.int64),
            validation_x={
                "causal": generator.normal(size=(8, 22, 64)).astype(np.float32)
            },
            validation_y=np.asarray([0, 1] * 4, dtype=np.int64),
            validation_rows=validation_rows,
        )
        fold = {
            "fold": 0, "train_runs": [1, 2, 3, 4, 5], "validation_runs": [0],
            "train_stage1": {"window_count": 16,
                             "window_sha256": "1" * 64},
        }
        self.context = SimpleNamespace(
            manifest={
                "protocol_id": "synthetic_session0_bundle",
                "normalization_protocol_id": "synthetic_normalization",
                "source_provenance": {"source_sha256": "2" * 64},
                "folds": [fold],
            },
            manifest_sha256="3" * 64,
        )
        self.spec = JobSpec(1, 0, 1, "causal", 42)

    @staticmethod
    def args(output: Path, stop_after_epoch: int | None = None) -> Namespace:
        return Namespace(
            output_root=output, epochs=3, batch_size=4,
            validation_batch_size=8, learning_rate=1e-3,
            weight_decay=1e-4, no_resume=False,
            stop_after_epoch=stop_after_epoch,
        )

    @staticmethod
    def file_bytes(root: Path) -> dict[str, bytes]:
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def test_cpu_resume_matches_continuous_and_completed_rerun_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous, resumed = root / "continuous", root / "resumed"
            device = torch.device("cpu")
            self.assertEqual(
                run_job(self.context, self.spec, self.arrays,
                        self.args(continuous), device), "complete"
            )
            self.assertEqual(
                run_job(self.context, self.spec, self.arrays,
                        self.args(resumed, 1), device), "paused"
            )
            self.assertEqual(
                run_job(self.context, self.spec, self.arrays,
                        self.args(resumed), device), "complete"
            )
            left = continuous / self.spec.name
            right = resumed / self.spec.name
            left_completed = json.loads((left / "completed.json").read_text())
            right_completed = json.loads((right / "completed.json").read_text())
            self.assertEqual(left_completed["model_tensor_sha256"],
                             right_completed["model_tensor_sha256"])
            with np.load(left / "oof_predictions.npz", allow_pickle=False) as a, \
                    np.load(right / "oof_predictions.npz", allow_pickle=False) as b:
                self.assertEqual(a.files, b.files)
                for key in a.files:
                    self.assertTrue(np.array_equal(a[key], b[key]), key)

            before = self.file_bytes(left)
            self.assertEqual(
                run_job(self.context, self.spec, self.arrays,
                        self.args(continuous), device), "already_complete"
            )
            self.assertEqual(self.file_bytes(left), before)

    def test_missing_derived_artifact_is_rebuilt_but_checkpoint_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = self.args(output)
            device = torch.device("cpu")
            run_job(self.context, self.spec, self.arrays, args, device)
            job = output / self.spec.name
            (job / "history.json").unlink()
            self.assertEqual(run_job(self.context, self.spec, self.arrays, args, device),
                             "already_complete")
            self.assertTrue((job / "history.json").is_file())
            self.assertEqual(json.loads((job / "status.json").read_text())["status"],
                             "complete")

            checkpoint = job / "latest.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaises(RuntimeError):
                run_job(self.context, self.spec, self.arrays, args, device)

    def test_full_checkpoint_without_completed_marker_is_finalized_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = self.args(output)
            device = torch.device("cpu")
            run_job(self.context, self.spec, self.arrays, args, device)
            completed = output / self.spec.name / "completed.json"
            completed.unlink()
            self.assertEqual(run_job(self.context, self.spec, self.arrays, args, device),
                             "complete")
            self.assertTrue(completed.is_file())

    def test_contract_binds_execution_fingerprint_and_rejects_hyperparameter_change(self) -> None:
        fingerprint = execution_fingerprint(torch.device("cpu"))
        self.assertEqual(fingerprint["device"], "cpu")
        self.assertIn("torch", fingerprint)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = self.args(output)
            run_job(self.context, self.spec, self.arrays, args, torch.device("cpu"))
            changed = self.args(output)
            changed.learning_rate = 2e-3
            with self.assertRaises(RuntimeError):
                run_job(self.context, self.spec, self.arrays, changed,
                        torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
