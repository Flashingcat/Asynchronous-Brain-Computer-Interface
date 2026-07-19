"""Regression tests for causal timeline, artifact identity, and grouped evaluation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import savemat


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path[:0] = [str(CODE_ROOT), str(CODE_ROOT / "eval"), str(CODE_ROOT / "preprocessing"), str(CODE_ROOT / "train")]

import build_oof_windows as preprocessing
import evaluate_test_session as evaluation
import metric
import train_hierarchical_oof as training
from algorithms.hard_vote import commands as hard_vote_commands


# 连续时间轴契约：边界窗口保留、坏试次切段、在线状态不得跨段。
class TimelineTests(unittest.TestCase):
    def test_boundary_windows_are_kept_with_real_segment_positions(self) -> None:
        signal = np.zeros((22, 2000), dtype=np.float32)
        events = [
            preprocessing.TaskEvent(0, 400, 800, 1, "left_hand"),
            preprocessing.TaskEvent(1, 1500, 1900, 2, "right_hand"),
        ]

        # 用一段伪迹将原 run 分成两个连续片段，滤波在本测试中保持恒等。
        with (
            patch.object(preprocessing, "eeg_data", return_value=signal),
            patch.object(preprocessing, "task_events", return_value=events),
            patch.object(preprocessing, "artifact_intervals", return_value=[(800, 1000)]),
            patch.object(preprocessing, "causal_filter_segment", side_effect=lambda value: value),
        ):
            arrays, returned_events, info = preprocessing.build_run_windows(object())

        np.testing.assert_array_equal(arrays["window_start"], [0, 125, 250, 1000, 1125, 1250, 1375, 1500])
        np.testing.assert_array_equal(arrays["segment"], [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(arrays["y"], [1, 1, 1, 0, 2, 2, 2, 0])
        np.testing.assert_array_equal(arrays["event"], [0, 0, 0, -1, 1, 1, 1, -1])
        self.assertEqual(arrays["X"].shape, (8, 22, 500))
        self.assertEqual(returned_events, events)
        self.assertEqual(info, {"segments": 2, "task_events": 2, "excluded_events": 0, "artifacts": 1, "windows": 8})

    def test_trial_artifact_flag_removes_the_full_trial(self) -> None:
        class Run:
            trial = np.asarray([251, 2254, 4172])
            artifacts = np.asarray([0, 1, 0])
            X = np.zeros((6000, 25))

        self.assertEqual(preprocessing.trial_artifact_intervals(Run()), [(2253, 4171)])

    def test_source_mat_artifacts_are_loaded_for_train_and_test_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for suffix in ("T", "E"):
                path = Path(directory) / f"A01{suffix}.mat"
                runs = np.empty(1, dtype=object)
                runs[0] = {
                    "X": np.zeros((100, 25)), "trial": np.asarray([1, 51]), "artifacts": np.asarray([0, 1]),
                }
                savemat(path, {"data": runs})
                paths.append(str(path))

            class Dataset:
                @staticmethod
                def data_path(subject: int) -> list[str]:
                    if subject != 1:
                        raise ValueError(subject)
                    return paths

            actual = preprocessing.load_subject_trial_artifacts(Dataset(), 1)
            self.assertEqual(actual, {(0, 0): [(50, 100)], (1, 0): [(50, 100)]})

    def test_event_overlapping_bad_trial_is_excluded(self) -> None:
        signal = np.zeros((22, 1500), dtype=np.float32)
        event = preprocessing.TaskEvent(0, 400, 800, 1, "left_hand")
        with (
            patch.object(preprocessing, "eeg_data", return_value=signal),
            patch.object(preprocessing, "task_events", return_value=[event]),
            patch.object(preprocessing, "artifact_intervals", return_value=[]),
            patch.object(preprocessing, "causal_filter_segment", side_effect=lambda value: value),
        ):
            arrays, events, info = preprocessing.build_run_windows(object(), [(750, 1000)])
        self.assertEqual(events, [])
        self.assertEqual(info["excluded_events"], 1)
        self.assertTrue(np.all(arrays["y"] == 0))

    def test_continuous_ids_reset_at_run_or_segment_boundaries(self) -> None:
        actual = evaluation.continuous_ids(
            np.asarray([0, 0, 0, 1, 1]),
            np.asarray([0, 0, 1, 0, 0]),
        )
        np.testing.assert_array_equal(actual, [0, 0, 1, 2, 2])

    def test_hard_vote_does_not_cross_segment_boundary(self) -> None:
        stage1 = np.asarray([[0.0, 1.0], [0.0, 1.0]])
        stage2 = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        reset_ids = evaluation.continuous_ids(np.asarray([0, 0]), np.asarray([0, 1]))
        output = hard_vote_commands(stage1, stage2, window_count=2, vote_threshold=2, run_ids=reset_ids)
        np.testing.assert_array_equal(output, [-1, -1])


# 事件评估契约：同类事件分离，并用原始采样点计算首指令延迟。
class EventMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.asarray([1, 1, 1, 1, 0])
        self.runs = np.asarray([0, 0, 0, 0, 0])
        self.window_events = np.asarray([0, 0, 1, 1, -1])
        self.decisions = np.asarray([600, 700, 1500, 1600, 1700])
        self.events = {
            "run": np.asarray([0, 0, 0]),
            "event": np.asarray([0, 1, 2]),
            "label": np.asarray([1, 1, 3]),
            "start": np.asarray([500, 1400, 2000]),
        }

    def test_same_class_events_stay_separate_and_use_real_time(self) -> None:
        report = metric.event_metrics(
            self.truth,
            np.asarray([1, -1, -1, 1, 2]),
            self.runs,
            self.window_events,
            self.decisions,
            self.events,
            sampling_rate=1000,
        )
        self.assertEqual(report["event_count"], 3)
        self.assertEqual(report["event_correct"], 2)
        self.assertEqual(report["event_miss"], 1)
        self.assertEqual(report["idle_false_commands"], 1)
        self.assertAlmostEqual(report["event_hit_rate"], 2 / 3)
        self.assertAlmostEqual(report["mean_latency_seconds"], 0.15)

    def test_wrong_first_command_does_not_enter_correct_latency(self) -> None:
        report = metric.event_metrics(
            self.truth,
            np.asarray([2, 1, -1, 1, -1]),
            self.runs,
            self.window_events,
            self.decisions,
            self.events,
            sampling_rate=1000,
        )
        self.assertEqual(report["event_wrong_class"], 1)
        self.assertEqual(report["event_correct"], 1)
        self.assertEqual(report["additional_event_commands"], 1)
        self.assertAlmostEqual(report["mean_latency_seconds"], 0.2)
        self.assertAlmostEqual(report["mean_wrong_command_latency_seconds"], 0.1)

    def test_unknown_window_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown events"):
            metric.event_metrics(
                self.truth,
                np.full(5, -1),
                self.runs,
                np.asarray([0, 0, 99, 99, -1]),
                self.decisions,
                self.events,
                sampling_rate=1000,
            )


# 产物身份契约：禁止静默覆盖，只汇总真正可比的多 seed checkpoint。
class ArtifactIdentityTests(unittest.TestCase):
    @staticmethod
    def report(subject: int, seed: int, learning_rate: float = 1e-3) -> dict:
        return {
            "checkpoint": f"s{subject:02d}_seed{seed}.pt",
            "subject": subject,
            "model": "eegnet",
            "seed": seed,
            "training_config": {"model": "eegnet", "seed": seed, "learning_rate": learning_rate},
            "accuracy": 0.7 + seed / 10000,
            "balanced_accuracy": 0.6,
            "event_hit_rate": 0.5,
            "mean_latency_seconds": 0.75,
        }

    def test_summary_groups_only_comparable_checkpoints(self) -> None:
        summary = metric.grouped_summary([self.report(1, 42), self.report(1, 43), self.report(2, 42)])
        self.assertEqual(summary["group_count"], 2)
        groups = {item["subject"]: item for item in summary["groups"]}
        self.assertEqual(groups[1]["seeds"], [42, 43])
        self.assertEqual(groups[1]["seed_count"], 2)
        self.assertEqual(groups[2]["seeds"], [42])

    def test_duplicate_seed_in_comparable_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            metric.grouped_summary([self.report(1, 42), self.report(1, 42)])

    def test_different_source_ids_are_not_grouped(self) -> None:
        first, second = self.report(1, 42), self.report(1, 43)
        first["training_config"]["source_id"] = "source_a"
        second["training_config"]["source_id"] = "source_b"
        self.assertEqual(metric.grouped_summary([first, second])["group_count"], 2)

    def test_existing_artifact_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            target.touch()
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                training.ensure_writable([target], overwrite=False)
            training.ensure_writable([target], overwrite=True)

    def test_duplicate_planned_artifact_is_rejected(self) -> None:
        target = Path("duplicate.pt")
        with self.assertRaisesRegex(ValueError, "duplicate artifact"):
            training.ensure_writable([target, target], overwrite=False)

    def test_fingerprint_changes_with_seed(self) -> None:
        first = training.config_fingerprint({"model": "eegnet", "seed": 42})
        second = training.config_fingerprint({"model": "eegnet", "seed": 43})
        self.assertNotEqual(first, second)
        self.assertEqual(first, training.config_fingerprint({"seed": 42, "model": "eegnet"}))

    def test_subject_run_reseeds_before_data_checks(self) -> None:
        arrays = {"subject": np.asarray([1]), "split": np.asarray([2])}
        with patch.object(training, "set_seed") as reseed, self.assertRaisesRegex(RuntimeError, "No train-session"):
            training.run_subject(1, arrays, SimpleNamespace(seed=9, model="eegnet"), torch.device("cpu"), {}, "test")
        reseed.assert_called_once_with(9)

    def test_evaluation_rejects_changed_model_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "windows.npz"
            one = np.asarray([1], dtype=np.int64)
            empty = np.empty(0, dtype=np.int64)
            np.savez(
                data_file, X=np.zeros((1, 22, 500), dtype=np.float32), y=one, subject=one,
                session=one, split=np.asarray([2]), run=np.asarray([0]), segment=np.asarray([0]),
                window_stop=np.asarray([500]), event=np.asarray([-1]), event_subject=empty,
                event_session=empty, event_run=empty, event_id=empty, event_label=empty, event_start=empty,
                schema_version=np.asarray(evaluation.REQUIRED_SCHEMA), dataset_id=np.asarray("data"), sampling_rate=np.asarray(250),
            )
            checkpoint = {
                "run_id": "run", "model": "eegnet", "subject": 1, "seed": 42,
                "training_config": {"dataset_id": "data", "data_sha256": "hash", "model": "eegnet", "seed": 42,
                                    "model_source_id": "old"},
                "binary_state_dict": {}, "mi_state_dict": {}, "mean": np.zeros((1, 22, 1)), "std": np.ones((1, 22, 1)),
            }
            with self.assertRaisesRegex(RuntimeError, "different model source"):
                evaluation.evaluate_checkpoint(
                    Path("checkpoint.pt"), checkpoint, SimpleNamespace(data=data_file), torch.device("cpu"),
                    "hash", "checkpoint_hash", "current",
                )

    def test_training_rejects_pre_repair_dataset_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_windows.npz"
            values = np.asarray([0], dtype=np.int64)
            np.savez(path, X=np.zeros((1, 22, 500), dtype=np.float32), y=values, subject=values,
                     session=values, run=values, fold=values, split=values)
            with self.assertRaisesRegex(RuntimeError, "schema_version"):
                training.load_arrays(path)


if __name__ == "__main__":
    unittest.main()
