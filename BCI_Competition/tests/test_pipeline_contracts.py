"""Regression tests for the repaired timeline, identity, and evaluation contracts."""

from __future__ import annotations

import json
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


class PipelineContractTests(unittest.TestCase):
    # 时间轴：边界窗口保留，决策采样点与事件延迟使用同一坐标定义。
    def test_causal_windows_keep_segments_and_decision_time(self) -> None:
        signal = np.zeros((22, 2000), dtype=np.float32)
        events = [
            preprocessing.TaskEvent(0, 400, 800, 1, "left_hand"),
            preprocessing.TaskEvent(1, 1500, 1900, 2, "right_hand"),
        ]
        with (
            patch.object(preprocessing, "eeg_data", return_value=signal),
            patch.object(preprocessing, "task_events", return_value=events),
            patch.object(preprocessing, "artifact_intervals", return_value=[(800, 1000)]),
            patch.object(preprocessing, "causal_filter_segment", side_effect=lambda value: value),
        ):
            arrays, returned, _ = preprocessing.build_run_windows(object())
        np.testing.assert_array_equal(arrays["window_start"], [0, 125, 250, 1000, 1125, 1250, 1375, 1500])
        np.testing.assert_array_equal(arrays["segment"], [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(arrays["decision_sample"], arrays["window_stop"] - 1)
        np.testing.assert_array_equal(arrays["event"], [0, 0, 0, -1, 1, 1, 1, -1])
        commands = np.full(len(arrays["y"]), -1); commands[0] = 1
        report = metric.event_metrics(
            arrays["y"], commands, np.zeros(len(commands)), arrays["event"], arrays["decision_sample"],
            {"run": np.asarray([0, 0]), "event": np.asarray([0, 1]), "label": np.asarray([1, 2]),
             "start": np.asarray([400, 1500])}, sampling_rate=250,
        )
        self.assertEqual(returned, events)
        self.assertAlmostEqual(report["mean_correct_latency_seconds"], 0.396)

    # 源 MAT 坏试次必须进入真实预处理链路，并整体排除重叠事件。
    def test_source_artifact_removes_full_trial_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for suffix in ("T", "E"):
                path = Path(directory) / f"A01{suffix}.mat"
                runs = np.empty(1, dtype=object)
                runs[0] = {"X": np.zeros((1500, 25)), "trial": np.asarray([1, 751]), "artifacts": np.asarray([0, 1])}
                savemat(path, {"data": runs}); paths.append(str(path))

            dataset = SimpleNamespace(data_path=lambda subject: paths)
            artifacts = preprocessing.load_subject_trial_artifacts(dataset, 1)
            self.assertEqual(artifacts, {(0, 0): [(750, 1500)], (1, 0): [(750, 1500)]})
            with (
                patch.object(preprocessing, "eeg_data", return_value=np.zeros((22, 1500), dtype=np.float32)),
                patch.object(preprocessing, "task_events", return_value=[preprocessing.TaskEvent(0, 800, 1200, 1, "left_hand")]),
                patch.object(preprocessing, "artifact_intervals", return_value=[]),
                patch.object(preprocessing, "causal_filter_segment", side_effect=lambda value: value),
            ):
                arrays, events, info = preprocessing.build_run_windows(object(), artifacts[(0, 0)])
            self.assertEqual(events, [])
            self.assertEqual(info["excluded_events"], 1)
            self.assertTrue(np.all(arrays["y"] == 0))

    def test_policy_resets_at_segment_boundary(self) -> None:
        reset_ids = evaluation.continuous_ids(np.asarray([0, 0]), np.asarray([0, 1]))
        output = hard_vote_commands(
            np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
            window_count=2, vote_threshold=2, run_ids=reset_ids,
        )
        np.testing.assert_array_equal(output, [-1, -1])

    def test_event_metric_separates_first_commands(self) -> None:
        events = {"run": np.asarray([0, 0, 0]), "event": np.asarray([0, 1, 2]),
                  "label": np.asarray([1, 1, 3]), "start": np.asarray([500, 1400, 2000])}
        data = {"y": np.asarray([1, 1, 1, 1, 0]), "run": np.zeros(5),
                "event": np.asarray([0, 0, 1, 1, -1]),
                "decision_sample": np.asarray([600, 700, 1500, 1600, 1700]),
                "events": events, "sampling_rate": 1000}
        report = evaluation.command_metrics(data, np.asarray([2, 1, -1, 1, 2]))
        self.assertNotIn("balanced_accuracy", report)
        self.assertEqual((report["event_correct"], report["event_wrong_class"], report["event_miss"]), (1, 1, 1))
        self.assertAlmostEqual(report["mean_correct_latency_seconds"], 0.2)
        self.assertAlmostEqual(report["mean_wrong_command_latency_seconds"], 0.1)
        self.assertEqual((report["additional_event_commands"], report["idle_false_commands"]), (1, 1))

    # 汇总只能合并除 seed 外身份一致的结果，并显式报告缺失延迟。
    def test_grouped_summary_rejects_incomparable_or_duplicate_runs(self) -> None:
        def report(seed: int, source: str = "same", latency: float | None = 0.75) -> dict:
            return {"checkpoint": f"{seed}.pt", "subject": 1, "model": "eegnet", "seed": seed,
                    "training_config": {"model": "eegnet", "seed": seed, "source_id": source},
                    "window_classification": {"accuracy": 0.7, "balanced_accuracy": 0.6},
                    "command_policy": {"event_hit_rate": 0.5, "event_wrong_class_rate": 0.2,
                                       "event_miss_rate": 0.3, "mean_correct_latency_seconds": latency,
                                       "idle_false_commands": 1, "additional_event_commands": 0}}
        summary = metric.grouped_summary([report(42, latency=None), report(43), report(44, source="other")])
        self.assertEqual(summary["group_count"], 2)
        latency = next(group for group in summary["groups"] if group["seed_count"] == 2)["metrics"]["command_policy"]["mean_correct_latency_seconds"]
        self.assertEqual((latency["valid_count"], latency["missing_count"]), (1, 1))
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            metric.grouped_summary([report(42), report(42)])

    def test_semantic_identity_ignores_paths_and_checkpoint_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.npz", Path(directory) / "second.npz"
            first.write_bytes(b"same data"); second.write_bytes(first.read_bytes())
            args = dict(model="eegnet", binary_epochs=1, mi_epochs=1, batch_size=2,
                        learning_rate=1e-3, seed=42, class_weight="none")
            arrays = {"dataset_id": np.asarray("data"), "dataset_config": np.asarray("{}")}
            one = training.effective_training_config(SimpleNamespace(data_file=first, **args), arrays)
            two = training.effective_training_config(SimpleNamespace(data_file=second, **args), arrays)
            self.assertEqual(one, two)

            np.savez(first, dataset_id=np.asarray("data"))
            eval_args = SimpleNamespace(data=first, algorithm="argmax", batch_size=8, device="cpu")
            records = [
                {"path": Path("a.pt"), "subject": 2, "model": "eegnet", "seed": 43, "sha256": "b", "model_source_id": "src"},
                {"path": Path("b.pt"), "subject": 1, "model": "eegnet", "seed": 42, "sha256": "a", "model_source_id": "src"},
            ]
            config = evaluation.evaluation_config(eval_args, records)
            records.reverse(); records[0]["path"] = Path("moved.pt")
            self.assertEqual(config, evaluation.evaluation_config(eval_args, records))

    def test_checkpoint_manifest_streams_and_sorts(self) -> None:
        state = {"live": 0, "peak": 0}

        class Checkpoint(dict):
            def __init__(self, seed: int):
                super().__init__(model="eegnet", subject=1, seed=seed, training_config={})
                state["live"] += 1; state["peak"] = max(state["peak"], state["live"])

            def __del__(self):
                state["live"] -= 1

        with (
            patch.object(evaluation.torch, "load", side_effect=lambda path, **_: Checkpoint(int(path.stem))),
            patch.object(evaluation, "file_sha256", side_effect=lambda path: path.stem),
            patch.object(evaluation, "model_source_id", return_value="source"),
        ):
            manifest = evaluation.checkpoint_manifest([Path("43.pt"), Path("42.pt")])
        self.assertEqual([item["seed"] for item in manifest], [42, 43])
        self.assertEqual(state["peak"], 1)

    # 防御性检查只保留会阻止错误实验继续运行的契约。
    def test_invalid_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            one, empty = np.asarray([1]), np.empty(0, dtype=np.int64)
            np.savez(
                path, X=np.zeros((1, 22, 500)), y=one, subject=one, session=one, split=np.asarray([2]),
                run=np.asarray([0]), segment=np.asarray([0]), decision_sample=np.asarray([499]), event=np.asarray([-1]),
                event_subject=empty, event_session=empty, event_run=empty, event_id=empty,
                event_label=empty, event_start=empty, schema_version=np.asarray(evaluation.REQUIRED_SCHEMA),
                dataset_id=np.asarray("data"), sampling_rate=np.asarray(250),
            )
            with self.assertRaisesRegex(RuntimeError, "no test-session events"):
                evaluation.load_subject_test_data(path, 1)
            path.touch()
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                training.ensure_writable([path], overwrite=False)

        checkpoint = {"run_id": "run", "model": "eegnet", "subject": 1, "seed": 42,
                      "training_config": {"dataset_id": "data", "data_sha256": "hash", "model": "eegnet",
                                          "seed": 42, "model_source_id": "old"},
                      "binary_state_dict": {}, "mi_state_dict": {},
                      "mean": np.zeros((1, 22, 1)), "std": np.ones((1, 22, 1))}
        with patch.object(evaluation, "load_subject_test_data", return_value={"dataset_id": "data"}):
            with self.assertRaisesRegex(RuntimeError, "different model source"):
                evaluation.evaluate_checkpoint(Path("checkpoint.pt"), checkpoint, SimpleNamespace(data=Path("data.npz")),
                                               torch.device("cpu"), "hash", "checkpoint_hash", "current")


if __name__ == "__main__":
    unittest.main()
