"""正式250 Hz因果滤波存储的因果性、坐标约束和Subject 1真实数据测试。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_causal_filter_store.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_causal_filter_store import (  # noqa: E402
    FILTER_ID,
    FILTER_SOS,
    WARMUP_SAMPLES,
    CausalFilterStore,
    build_causal_filter_store,
    filter_chunk,
    filter_segment,
    validate_filter_manifest,
    validate_source_manifest,
)
from build_offline_view import build_offline_view  # noqa: E402
from build_protocol_index import build_subject, save_subject  # noqa: E402
from build_signal_store import SignalStore, build_signal_store  # noqa: E402


def nested_strings(value):
    """遍历清单全部字符串，检查冻结产物没有绑定本机绝对路径。"""
    if isinstance(value, dict):
        for item in value.values():
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)
    elif isinstance(value, str):
        yield value


class CausalFilterCoreTests(unittest.TestCase):
    """用合成信号验证数学因果性，不依赖作者生成的清单或计数。"""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260712)
        cls.signal = rng.normal(size=(22, 1500)).astype(np.float32) * 1e-5

    def test_future_change_cannot_alter_past_output(self) -> None:
        changed = self.signal.copy()
        changed[:, 700:] += 3e-5
        original_output = filter_segment(self.signal)
        changed_output = filter_segment(changed)
        self.assertTrue(np.array_equal(original_output[:, :700], changed_output[:, :700]))
        self.assertFalse(np.array_equal(original_output[:, 700:], changed_output[:, 700:]))

    def test_chunked_stream_equals_one_shot_segment(self) -> None:
        first, state = filter_chunk(self.signal[:, :317])
        second, state = filter_chunk(self.signal[:, 317:911], state)
        third, _ = filter_chunk(self.signal[:, 911:], state)
        chunked = np.ascontiguousarray(np.concatenate((first, second, third), axis=1),
                                       dtype=np.float32)
        self.assertTrue(np.array_equal(chunked, filter_segment(self.signal)))

    def test_one_second_warmup_covers_impulse_tail(self) -> None:
        impulse = np.zeros((1, 5000), dtype=np.float64)
        impulse[0, 0] = 1.0
        response, _ = filter_chunk(impulse)
        total_energy = float(np.square(response).sum())
        tail_energy = float(np.square(response[:, WARMUP_SAMPLES:]).sum())
        self.assertLess(tail_energy / total_energy, 1e-10)

    def test_segment_reset_is_independent_of_previous_signal(self) -> None:
        expected = filter_segment(self.signal[:, 800:])
        _, previous_state = filter_chunk(self.signal[:, :800])
        inherited, _ = filter_chunk(self.signal[:, 800:], previous_state)
        self.assertFalse(np.allclose(expected, inherited.astype(np.float32)))
        self.assertTrue(np.array_equal(expected, filter_segment(self.signal[:, 800:])))


class RealCausalFilterStoreTests(unittest.TestCase):
    """从真实MAT重新构建上游和滤波存储，核对正式S1产物及窗口规则。"""

    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.data_root = Path(value)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.index_dir = cls.root / "indices"
        cls.raw_output = cls.root / "raw_store"
        cls.filtered_output = cls.root / "filtered_store"

        cls.base = build_subject(cls.data_root, 1)
        cls.base_manifest = save_subject(cls.index_dir, 1, cls.base)
        cls.offline = build_offline_view(
            cls.base[0], cls.base[1], cls.base[2], cls.base_manifest
        )
        cls.raw_manifest, cls.raw_manifest_path = build_signal_store(
            cls.data_root, cls.index_dir, cls.raw_output, 1
        )
        cls.manifest, cls.manifest_path = build_causal_filter_store(
            cls.raw_output, cls.filtered_output, 1
        )
        cls.raw_store = SignalStore(cls.raw_manifest_path)
        cls.store = CausalFilterStore(cls.manifest_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.raw_store._cache.clear()
        cls.store._signal_store._cache.clear()
        cls.temporary.cleanup()

    def test_manifest_freezes_filter_and_startup_policy(self) -> None:
        self.assertEqual(self.manifest["protocol_id"], FILTER_ID.format(subject=1))
        self.assertEqual(self.manifest["sampling_rate"], 250)
        self.assertEqual(self.manifest["filter"]["butterworth_N"], 4)
        self.assertEqual(self.manifest["filter"]["realized_bandpass_order"], 8)
        self.assertEqual(self.manifest["warmup_policy"]["configured_samples"], 250)
        self.assertTrue(np.array_equal(
            np.asarray(self.manifest["filter"]["sos_coefficients_float64"]), FILTER_SOS
        ))
        self.assertEqual(len(self.manifest["segments"]), 34)
        self.assertEqual(self.manifest["summaries"]["0train"], {
            "segments": 21, "samples": 557910,
            "warmup_excluded_samples": 5250, "formal_samples": 552660,
        })
        self.assertEqual(self.manifest["summaries"]["1test"], {
            "segments": 13, "samples": 569910,
            "warmup_excluded_samples": 3250, "formal_samples": 566660,
        })
        for value in nested_strings(self.manifest):
            self.assertFalse(Path(value).is_absolute(), value)

    def test_every_segment_equals_independent_forward_filter(self) -> None:
        coefficients = np.asarray(
            self.manifest["filter"]["sos_coefficients_float64"], dtype=np.float64
        )
        for item in self.manifest["segments"]:
            key = (item["session"], item["run"], item["segment"])
            raw = np.asarray(self.raw_store.load_segment(*key), dtype=np.float64)
            expected = np.ascontiguousarray(sosfilt(coefficients, raw, axis=1),
                                             dtype=np.float32)
            actual = self.store.load_segment(*key)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(item["warmup_start_native"], item["start_native"])
            self.assertEqual(item["formal_start_native"], item["start_native"] + 250)

    def test_formal_window_masks_and_reader_enforcement(self) -> None:
        expected_counts = ((self.base[2], 8845), (self.offline[0], 5001),
                           (self.offline[1], 2770))
        for rows, expected in expected_counts:
            mask = np.asarray([self.store.window_is_formal(row) for row in rows])
            self.assertEqual(int(mask.sum()), expected)
            accepted = rows[np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2]]
            self.assertEqual(self.store.read_window(accepted).shape, (22, 500))

        first_online = self.base[2][0]
        self.assertFalse(self.store.window_is_formal(first_online))
        with self.assertRaises(ValueError):
            self.store.read_window(first_online)
        self.assertEqual(self.store.read_window(first_online, allow_warmup=True).shape,
                         (22, 500))

    def test_repeat_cli_and_moved_store_are_portable(self) -> None:
        original = self.manifest_path.read_bytes()
        repeated, repeated_path = build_causal_filter_store(
            self.raw_output, self.filtered_output, 1
        )
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(repeated_path.read_bytes(), original)

        command = [sys.executable, str(SOURCE_FILE), "--signal-dir", str(self.raw_output),
                   "--subjects", "1", "--output-dir", str(self.filtered_output)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        moved = self.root / "moved" / self.manifest["protocol_id"]
        shutil.copytree(self.manifest_path.parent, moved)
        moved_store = CausalFilterStore(moved, verify_hashes=True)
        formal_row = next(row for row in self.base[2] if moved_store.window_is_formal(row))
        self.assertEqual(moved_store.read_window(formal_row).shape, (22, 500))

    def test_source_manifest_validation_rejects_wrong_units(self) -> None:
        wrong = json.loads(json.dumps(self.raw_manifest))
        wrong["stored_unit"] = "microvolts"
        with self.assertRaises(RuntimeError):
            validate_source_manifest(wrong, 1)

        wrong_filter = json.loads(json.dumps(self.manifest))
        wrong_filter["filter"]["butterworth_N"] = 2
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(wrong_filter)

        wrong_warmup = json.loads(json.dumps(self.manifest))
        wrong_warmup["segments"][0]["formal_start_native"] -= 250
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(wrong_warmup)

        wrong_source = json.loads(json.dumps(self.manifest))
        wrong_source["source_signal_manifest_sha256"] = "not-a-sha256"
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(wrong_source)

        duplicate_file = json.loads(json.dumps(self.manifest))
        duplicate_file["segments"][1]["file"] = duplicate_file["segments"][0]["file"]
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(duplicate_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
