"""协议索引构建器的结构测试与真实数据回归测试。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE_FILE = Path(__file__).resolve().parents[1] / "code" / "preprocessing" / "build_protocol_index.py"
sys.path.insert(0, str(SOURCE_FILE.parent))

from build_protocol_index import (  # noqa: E402
    FS, STEP, build_subject, clean_segments, file_hash, parse_run, save_subject,
)


def composite_keys(array: np.ndarray, fields: tuple[str, ...]) -> list[tuple[int, ...]]:
    return [tuple(int(row[field]) for field in fields) for row in array]


class SyntheticProtocolTests(unittest.TestCase):
    """不依赖真实数据的手算规则永远运行，不能被环境变量跳过。"""

    def test_segment_complement_on_hand_checkable_example(self) -> None:
        self.assertEqual(
            clean_segments(2000, [(250, 750), (1000, 1500)]),
            [(0, 250, "run_start"), (750, 1000, "after_artifact"),
             (1500, 2000, "after_artifact")],
        )

    def test_fractional_sampling_rate_is_rejected(self) -> None:
        record = SimpleNamespace(
            X=np.empty((9000, 25), dtype=np.int8), fs=250.9,
            trial=np.arange(48) * 150 + 1, y=np.tile([1, 2, 3, 4], 12), artifacts=np.zeros(48),
        )
        with self.assertRaises(RuntimeError):
            parse_run(record, Path("synthetic.mat"), 0)


class RealDataProtocolTests(unittest.TestCase):
    """缺少真实数据路径时直接失败，不把 skipped 当作通过。"""

    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.data_root = Path(value)
        cls.built = {subject: build_subject(cls.data_root, subject) for subject in range(1, 10)}

    def test_subject1_golden_counts_and_anchor(self) -> None:
        events, _, _, metadata = self.built[1]
        train, test = metadata["summaries"]["0train"], metadata["summaries"]["1test"]
        self.assertEqual((train["artifact_events"], train["segments"], train["windows"]), (15, 21, 4392))
        self.assertEqual((test["artifact_events"], test["segments"], test["windows"]), (7, 13, 4514))
        first = events[0]
        self.assertEqual(
            (int(first["class_id"]), int(first["trial_start"]), int(first["mi_start"]), int(first["mi_stop"])),
            (4, 250, 750, 1750),
        )

    def test_all_subjects_and_a04t_record_mapping(self) -> None:
        for subject, (events, segments, windows, metadata) in self.built.items():
            with self.subTest(subject=subject):
                self.assertEqual(len(events), 576)
                self.assertTrue(np.all(events["subject"] == subject))
                self.assertEqual(metadata["summaries"]["0train"]["events"], 288)
                self.assertEqual(metadata["summaries"]["1test"]["events"], 288)
                self.assertGreater(len(segments), 0)
                self.assertGreater(len(windows), 0)
        a04t_runs = self.built[4][3]["summaries"]["0train"]["runs"]
        self.assertEqual([run["source_record_index"] for run in a04t_runs], [1, 2, 3, 4, 5, 6])

    def test_event_semantics_keys_and_order(self) -> None:
        for events, segments, windows, _ in self.built.values():
            self.assertTrue(np.all(events["mi_start"] == events["trial_start"] + 2 * FS))
            self.assertTrue(np.all(events["mi_stop"] == events["trial_start"] + 6 * FS))
            for array, fields in (
                (events, ("subject", "session", "run", "trial")),
                (segments, ("subject", "session", "run", "segment")),
                (windows, ("subject", "session", "run", "segment", "window")),
            ):
                keys = composite_keys(array, fields)
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(keys, sorted(keys))

    def test_window_semantics_and_artifact_exclusion(self) -> None:
        for subject, (events, segments, windows, _) in self.built.items():
            self.assertTrue(np.array_equal(windows["decision_time"], windows["stop"] / FS))
            for segment in segments:
                mask = ((windows["session"] == segment["session"]) &
                        (windows["run"] == segment["run"]) &
                        (windows["segment"] == segment["segment"]))
                selected = windows[mask]
                self.assertTrue(np.all(selected["start"] >= segment["start"]))
                self.assertTrue(np.all(selected["stop"] <= segment["stop"]))
                self.assertTrue(np.all(selected["stop"] - selected["start"] == 2 * FS))
                self.assertTrue(np.all((selected["start"] - segment["start"]) % STEP == 0))
                self.assertTrue(np.array_equal(selected["window"], np.arange(len(selected))))
            for artifact in events[events["artifact"]]:
                mask = ((windows["session"] == artifact["session"]) &
                        (windows["run"] == artifact["run"]))
                selected = windows[mask]
                overlap = ((selected["start"] < artifact["trial_start"] + 6 * FS) &
                           (selected["stop"] > artifact["trial_start"]))
                self.assertFalse(np.any(overlap), f"Subject {subject} 存在跨伪迹窗口")

    def test_manifest_and_frozen_write_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            built = self.built[1]
            first = save_subject(output, 1, built)
            index = output / first["index_file"]
            first_hash = file_hash(index)
            second = save_subject(output, 1, built)
            self.assertEqual(first_hash, file_hash(index))
            self.assertEqual(first, second)
            self.assertEqual(first["artifact_policy"], "official_trial_exclusion")
            self.assertIn("source_record_index", first["summaries"]["0train"]["runs"][0])

            changed_events = built[0].copy()
            changed_events["mi_start"] += 1
            with self.assertRaises(FileExistsError):
                save_subject(output, 1, (changed_events, *built[1:]))

    def test_multi_subject_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(SOURCE_FILE), "--data-root", str(self.data_root),
                       "--subjects", "1", "4", "--output-dir", directory]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 2)
            self.assertEqual(len(list(Path(directory).glob("*_manifest.json"))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
