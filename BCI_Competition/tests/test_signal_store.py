"""Subject 1 原生 250 Hz 干净 segment 信号存储的真实数据测试。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_signal_store.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_offline_view import build_offline_view  # noqa: E402
from build_protocol_index import build_subject, save_subject, vector  # noqa: E402
from build_signal_store import (  # noqa: E402
    EEG_CHANNELS,
    SEGMENT_POLICY,
    SIGNAL_ID,
    SignalStore,
    build_signal_store,
    write_frozen_array,
)


class RealSignalStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.data_root = Path(value)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.index_dir = cls.root / "indices"
        cls.output_dir = cls.root / "portable_output"

        cls.base = build_subject(cls.data_root, 1)
        cls.base_manifest = save_subject(cls.index_dir, 1, cls.base)
        cls.offline = build_offline_view(
            cls.base[0], cls.base[1], cls.base[2], cls.base_manifest
        )
        cls.manifest, cls.manifest_path = build_signal_store(
            cls.data_root, cls.index_dir, cls.output_dir, 1
        )
        cls.store = SignalStore(cls.manifest_path, verify_hashes=True)

        # 测试独立保留原始record，用于逐segment和逐窗口数值对照。
        cls.records = {}
        for session, suffix in ((0, "T"), (1, "E")):
            path = cls.data_root / f"A01{suffix}.mat"
            cls.records[session] = vector(
                loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
            ).tolist()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.store._cache.clear()
        cls.temporary.cleanup()

    def expected_from_raw(self, session: int, run: int, start: int, stop: int) -> np.ndarray:
        summary = self.base_manifest["summaries"]["0train" if session == 0 else "1test"]["runs"][run]
        record = self.records[session][summary["source_record_index"]]
        return np.ascontiguousarray(record.X[start:stop, :22].T * 1e-6, dtype=np.float32)

    def test_manifest_layout_counts_and_relative_paths(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["protocol_id"], SIGNAL_ID.format(subject=1))
        self.assertEqual(manifest["sampling_rate"], 250)
        self.assertEqual(manifest["channels"], list(EEG_CHANNELS))
        self.assertEqual((manifest["source_unit"], manifest["stored_unit"]),
                         ("microvolts", "volts"))
        self.assertEqual(manifest["dtype"], "float32")
        self.assertEqual(manifest["artifact_policy"], "official_trial_exclusion")
        self.assertEqual(manifest["segment_policy"], SEGMENT_POLICY)
        self.assertEqual(len(manifest["segments"]), 34)
        self.assertEqual((manifest["summaries"]["0train"]["segments"],
                          manifest["summaries"]["1test"]["segments"]), (21, 13))
        for item in manifest["segments"]:
            self.assertFalse(Path(item["file"]).is_absolute())
            self.assertEqual(item["shape"], [22, item["n_samples"]])
        self.assertEqual(len(list(self.manifest_path.parent.glob("*.npy"))), 34)

    def test_every_saved_segment_matches_original_mat_exactly(self) -> None:
        for item in self.manifest["segments"]:
            actual = self.store.load_segment(item["session"], item["run"], item["segment"])
            expected = self.expected_from_raw(
                item["session"], item["run"], item["start_native"], item["stop_native"]
            )
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual, expected))

    def test_segments_equal_clean_time_complement_without_concatenation(self) -> None:
        events, segments, _, _ = self.base
        manifest_keys = [
            (item["session"], item["run"], item["segment"],
             item["start_native"], item["stop_native"])
            for item in self.manifest["segments"]
        ]
        base_keys = [
            (int(row["session"]), int(row["run"]), int(row["segment"]),
             int(row["start"]), int(row["stop"])) for row in segments
        ]
        self.assertEqual(manifest_keys, base_keys)
        for item in self.manifest["segments"]:
            run_events = events[(events["session"] == item["session"]) &
                                (events["run"] == item["run"]) & events["artifact"]]
            overlaps = ((item["start_native"] < run_events["trial_start"] + 1500) &
                        (item["stop_native"] > run_events["trial_start"]))
            self.assertFalse(np.any(overlaps))
        for session, name in ((0, "0train"), (1, "1test")):
            expected = sum(run["clean_samples"] for run in self.base_manifest["summaries"][name]["runs"])
            self.assertEqual(self.manifest["summaries"][name]["samples"], expected)

    def test_online_and_offline_windows_read_the_original_samples(self) -> None:
        online = self.base[2]
        stage1 = self.offline[0]
        selected_rows = []
        for item in self.manifest["segments"]:
            rows = online[(online["session"] == item["session"]) &
                          (online["run"] == item["run"]) &
                          (online["segment"] == item["segment"])]
            if len(rows):
                selected_rows.extend((rows[0], rows[-1]))
        selected_rows.extend((stage1[0], stage1[len(stage1) // 2], stage1[-1]))

        for row in selected_rows:
            actual = self.store.read_window(row)
            expected = self.expected_from_raw(
                int(row["session"]), int(row["run"]), int(row["start"]), int(row["stop"])
            )
            self.assertEqual(actual.shape, (22, 500))
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual, expected))

    def test_reader_rejects_wrong_subject_or_segment_bounds(self) -> None:
        row = self.base[2][0].copy()
        row["subject"] = 2
        with self.assertRaises(ValueError):
            self.store.read_window(row)
        row = self.base[2][0].copy()
        row["stop"] = self.manifest["segments"][0]["stop_native"] + 1
        with self.assertRaises(ValueError):
            self.store.read_window(row)

    def test_frozen_array_rejects_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tiny.npy"
            first = np.arange(12, dtype=np.float32).reshape(3, 4)
            self.assertEqual(write_frozen_array(target, first), write_frozen_array(target, first))
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_frozen_array(target, first + 1)
            self.assertEqual(target.read_bytes(), original)

    def test_repeat_build_and_cli_are_portable(self) -> None:
        first_bytes = self.manifest_path.read_bytes()
        second, second_path = build_signal_store(
            self.data_root, self.index_dir, self.output_dir, 1
        )
        self.assertEqual(second, self.manifest)
        self.assertEqual(second_path.read_bytes(), first_bytes)

        command = [sys.executable, str(SOURCE_FILE), "--data-root", str(self.data_root),
                   "--index-dir", str(self.index_dir), "--subjects", "1",
                   "--output-dir", str(self.output_dir)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        moved_reader = SignalStore(self.manifest_path.parent, verify_hashes=True)
        self.assertEqual(moved_reader.read_window(self.base[2][0]).shape, (22, 500))


if __name__ == "__main__":
    unittest.main(verbosity=2)
