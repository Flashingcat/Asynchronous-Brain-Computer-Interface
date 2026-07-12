"""原仓库风格离线窗口视图的真实数据回归测试。"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_offline_view.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_offline_view import OFFLINE_ID, build_offline_view, load_base, save_offline_view  # noqa: E402
from build_protocol_index import FS, STEP, WINDOW, build_subject, file_hash, save_subject  # noqa: E402


class RealOfflineViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.data_root = Path(value)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls.temporary.name)
        cls.base, cls.base_manifests, cls.offline = {}, {}, {}
        for subject in range(1, 10):
            built = build_subject(cls.data_root, subject)
            base_manifest = save_subject(cls.index_dir, subject, built)
            cls.base[subject] = built
            cls.base_manifests[subject] = base_manifest
            cls.offline[subject] = build_offline_view(built[0], built[1], built[2], base_manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_subject1_golden_counts(self) -> None:
        summary = self.offline[1][2]
        self.assertEqual(
            (summary["0train"]["stage1_windows"], summary["0train"]["idle_windows"],
             summary["0train"]["task_windows"], summary["0train"]["task_windows_per_class"]),
            (2496, 1131, 1365, [345, 345, 340, 335]),
        )
        self.assertEqual(
            (summary["1test"]["stage1_windows"], summary["1test"]["idle_windows"],
             summary["1test"]["task_windows"], summary["1test"]["task_windows_per_class"]),
            (2565, 1160, 1405, [355, 350, 345, 355]),
        )

    def test_every_clean_trial_has_five_event_anchored_windows(self) -> None:
        expected_offsets = [0, 125, 250, 375, 500]
        for subject in range(1, 10):
            events, segments, _, _ = self.base[subject]
            _, stage2, _ = self.offline[subject]
            counts = Counter((int(row["session"]), int(row["run"]), int(row["trial"])) for row in stage2)
            for event in events:
                key = (int(event["session"]), int(event["run"]), int(event["trial"]))
                if event["artifact"]:
                    self.assertNotIn(key, counts)
                else:
                    self.assertEqual(counts[key], 5)
                    selected = stage2[(stage2["session"] == event["session"]) &
                                      (stage2["run"] == event["run"]) &
                                      (stage2["trial"] == event["trial"])]
                    self.assertEqual((selected["start"] - event["mi_start"]).tolist(), expected_offsets)
                    self.assertTrue(np.all(selected["stop"] - selected["start"] == WINDOW))
                    self.assertTrue(np.all(selected["start"] >= event["mi_start"]))
                    self.assertTrue(np.all(selected["stop"] <= event["mi_stop"]))
                    self.assertTrue(np.all(selected["final_label"] == event["class_id"]))
                    for row in selected:
                        matching = segments[(segments["session"] == row["session"]) &
                                            (segments["run"] == row["run"]) &
                                            (segments["segment"] == row["segment"])]
                        self.assertEqual(len(matching), 1)
                        self.assertLessEqual(int(matching[0]["start"]), int(row["start"]))
                        self.assertLessEqual(int(row["stop"]), int(matching[0]["stop"]))

    def test_idle_windows_are_run_anchored_and_pure(self) -> None:
        for subject in range(1, 10):
            events, segments, _, _ = self.base[subject]
            stage1, _, _ = self.offline[subject]
            for row in stage1[~stage1["is_task"]]:
                self.assertEqual(int(row["start"]) % STEP, 0)
                matching_segments = segments[(segments["session"] == row["session"]) &
                                             (segments["run"] == row["run"]) &
                                             (segments["segment"] == row["segment"])]
                self.assertEqual(len(matching_segments), 1)
                self.assertLessEqual(int(matching_segments[0]["start"]), int(row["start"]))
                self.assertLessEqual(int(row["stop"]), int(matching_segments[0]["stop"]))
                run_events = events[(events["session"] == row["session"]) &
                                    (events["run"] == row["run"])]
                overlap = ((row["start"] < run_events["mi_stop"]) &
                           (row["stop"] > run_events["mi_start"]))
                self.assertFalse(np.any(overlap), f"Subject {subject} 出现非纯IDLE窗口")
                self.assertEqual(int(row["stop"] - row["start"]), WINDOW)

    def test_labels_stage2_view_keys_and_order(self) -> None:
        for stage1, stage2, _ in self.offline.values():
            self.assertTrue(np.array_equal(stage2, stage1[stage1["is_task"]]))
            self.assertTrue(np.all(stage1[~stage1["is_task"]]["final_label"] == 0))
            self.assertTrue(np.all(stage1[~stage1["is_task"]]["stage2_label"] == -1))
            self.assertTrue(np.all(stage2["stage1_label"] == 1))
            self.assertTrue(np.all(stage2["stage2_label"] == stage2["final_label"] - 1))
            keys = [(int(row["subject"]), int(row["session"]), int(row["run"]), int(row["window"]))
                    for row in stage1]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(keys, sorted(keys))
            for session in (0, 1):
                for run in range(6):
                    selected = stage1[(stage1["session"] == session) & (stage1["run"] == run)]
                    self.assertTrue(np.array_equal(selected["window"], np.arange(len(selected))))

    def test_run_grid_accounting(self) -> None:
        for _, _, summary in self.offline.values():
            for session_name in ("0train", "1test"):
                for run in summary[session_name]["runs"]:
                    accounted = (run["idle_windows"] + run["pure_mi_grid_windows"] +
                                 run["boundary_grid_windows"] + run["artifact_or_gap_grid_windows"])
                    self.assertEqual(accounted, run["run_grid_candidates"])
                    self.assertEqual(run["task_windows"], run["clean_events"] * 5)

    def test_rejects_wrong_base_identity_and_configuration(self) -> None:
        """首次冻结前必须拒绝被试、协议参数或数组身份不一致。"""
        subject = 1
        base_id = f"bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
        source_index = self.index_dir / f"{base_id}.npz"
        source_manifest = self.index_dir / f"{base_id}_manifest.json"

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            shutil.copy2(source_index, copied / source_index.name)
            original = json.loads(source_manifest.read_text(encoding="utf-8"))
            for field, bad_value in (("subject", 9), ("sampling_rate", 128),
                                     ("window_samples", 256), ("step_samples", 64)):
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = bad_value
                    (copied / source_manifest.name).write_text(
                        json.dumps(changed, ensure_ascii=False), encoding="utf-8"
                    )
                    with self.assertRaises(RuntimeError):
                        load_base(copied, subject)

            changed = copy.deepcopy(original)
            changed["summaries"]["0train"]["runs"][0]["original_samples"] -= 1
            (copied / source_manifest.name).write_text(
                json.dumps(changed, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_base(copied, subject)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            with np.load(source_index, allow_pickle=False) as data:
                events = data["events"].copy()
                segments = data["segments"].copy()
                online_windows = data["windows"].copy()
            events["subject"] = 2
            changed_index = copied / source_index.name
            np.savez_compressed(changed_index, events=events, segments=segments, windows=online_windows)
            changed_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            changed_manifest["index_sha256"] = file_hash(changed_index)
            (copied / source_manifest.name).write_text(
                json.dumps(changed_manifest, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_base(copied, subject)

    def test_rejects_incomplete_or_shifted_mi_interval(self) -> None:
        """Task窗口只能来自 trial 后2至6秒的完整MI区间。"""
        events, segments, online_windows, _ = self.base[1]
        for field, delta in (("mi_stop", -750), ("mi_start", 1)):
            with self.subTest(field=field):
                changed = events.copy()
                clean_index = int(np.flatnonzero(~changed["artifact"])[0])
                changed[field][clean_index] += delta
                with self.assertRaises(RuntimeError):
                    build_offline_view(changed, segments, online_windows, self.base_manifests[1])

    def test_rejects_dtype_label_trial_and_subject_mutations(self) -> None:
        """字段类型、类别域、trial时序和被试范围均属于冻结协议。"""
        events, segments, online_windows, _ = self.base[1]
        manifest = self.base_manifests[1]

        artifact_int_dtype = np.dtype([
            (name, "i1" if name == "artifact" else events.dtype.fields[name][0])
            for name in events.dtype.names
        ])
        artifact_int_events = np.empty(events.shape, dtype=artifact_int_dtype)
        for name in events.dtype.names:
            artifact_int_events[name] = events[name]

        extra_field_events = np.empty(events.shape, dtype=np.dtype(events.dtype.descr + [("extra", "u1")]))
        for name in events.dtype.names:
            extra_field_events[name] = events[name]
        extra_field_events["extra"] = 0
        for name, changed in (("artifact_int8", artifact_int_events),
                              ("extra_field", extra_field_events)):
            with self.subTest(case=name), self.assertRaises(RuntimeError):
                build_offline_view(changed, segments, online_windows, manifest)

        for bad_class in (0, 5):
            with self.subTest(class_id=bad_class):
                changed = events.copy()
                changed["class_id"][0] = bad_class
                with self.assertRaises(RuntimeError):
                    build_offline_view(changed, segments, online_windows, manifest)

        for name, mutate in (
                ("negative", lambda array: array.__setitem__(0, -1)),
                ("duplicate", lambda array: array.__setitem__(1, array[0]))):
            with self.subTest(trial_start=name):
                changed = events.copy()
                mutate(changed["trial_start"])
                changed["mi_start"] = changed["trial_start"] + 2 * FS
                changed["mi_stop"] = changed["trial_start"] + 6 * FS
                with self.assertRaises(RuntimeError):
                    build_offline_view(changed, segments, online_windows, manifest)

        subject0_events, subject0_segments = events.copy(), segments.copy()
        subject0_windows = online_windows.copy()
        subject0_events["subject"] = subject0_segments["subject"] = subject0_windows["subject"] = 0
        subject0_manifest = copy.deepcopy(manifest)
        subject0_manifest["subject"] = 0
        subject0_manifest["protocol_id"] = "bnci2014001_s00_native250_artifact_trial_v1"
        with self.assertRaises(ValueError):
            build_offline_view(subject0_events, subject0_segments, subject0_windows, subject0_manifest)

        extra_window = online_windows[[0]].copy()
        extra_window["session"] = 2
        with self.assertRaises(RuntimeError):
            build_offline_view(events, segments, np.concatenate((online_windows, extra_window)), manifest)

    def test_rejects_missing_or_internally_inconsistent_online_windows(self) -> None:
        """母索引必须包含能由segment逐行重建的在线窗口及一致summary。"""
        subject = 1
        base_id = f"bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
        source_index = self.index_dir / f"{base_id}.npz"
        source_manifest = self.index_dir / f"{base_id}_manifest.json"
        with np.load(source_index, allow_pickle=False) as data:
            events = data["events"].copy()
            segments = data["segments"].copy()
            online_windows = data["windows"].copy()
        original_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            changed_index = copied / source_index.name
            np.savez_compressed(changed_index, events=events, segments=segments)
            changed_manifest = copy.deepcopy(original_manifest)
            changed_manifest["index_sha256"] = file_hash(changed_index)
            (copied / source_manifest.name).write_text(
                json.dumps(changed_manifest, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_base(copied, subject)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            changed_segments = segments.copy()
            mask = ((changed_segments["session"] == 0) & (changed_segments["run"] == 0))
            last_segment = int(np.flatnonzero(mask)[-1])
            changed_segments["stop"][last_segment] += STEP
            changed_manifest = copy.deepcopy(original_manifest)
            run_summary = changed_manifest["summaries"]["0train"]["runs"][0]
            run_summary["original_samples"] += STEP
            run_summary["clean_samples"] += STEP
            changed_index = copied / source_index.name
            np.savez_compressed(changed_index, events=events, segments=changed_segments,
                                windows=online_windows)
            changed_manifest["index_sha256"] = file_hash(changed_index)
            (copied / source_manifest.name).write_text(
                json.dumps(changed_manifest, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_base(copied, subject)

    def test_rejects_inconsistent_first_freeze(self) -> None:
        """保存接口不能接受错被试、错Stage 2或伪造summary。"""
        subject = 1
        base_id = f"bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
        base_files = (self.index_dir / f"{base_id}.npz", self.index_dir / f"{base_id}_manifest.json")
        stage1, stage2, summary = self.offline[subject]
        cases = [
            (2, (stage1, stage2, summary), base_files),
            (subject, (stage1, stage2[:-1], summary), base_files),
            (subject, (stage1, stage2, {**summary, "0train": {}}), base_files),
        ]
        for index, arguments in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    save_offline_view(Path(directory), *arguments)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_manifest_freeze_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subject = 1
            base_id = f"bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
            base_files = (self.index_dir / f"{base_id}.npz", self.index_dir / f"{base_id}_manifest.json")
            first = save_offline_view(output, subject, self.offline[subject], base_files)
            second = save_offline_view(output, subject, self.offline[subject], base_files)
            self.assertEqual(first, second)
            frozen_index = output / first["index_file"]
            frozen_index.write_bytes(b"corrupted frozen output")
            with self.assertRaises(FileExistsError):
                save_offline_view(output, subject, self.offline[subject], base_files)
            self.assertEqual(frozen_index.read_bytes(), b"corrupted frozen output")

            command = [sys.executable, str(SOURCE_FILE), "--index-dir", str(self.index_dir),
                       "--subjects", "4", "--output-dir", str(output)]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            pattern = OFFLINE_ID.format(subject=1).replace("s01", "s??") + ".npz"
            self.assertEqual(len(list(output.glob(pattern))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
