"""正式250 Hz零相位FIR存储的非因果性、双侧边缘和S1真实数据测试。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_zero_phase_filter_store.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_causal_filter_store import (  # noqa: E402
    CausalFilterStore,
    build_causal_filter_store,
)
from build_offline_view import build_offline_view  # noqa: E402
from build_protocol_index import build_subject, save_subject  # noqa: E402
from build_signal_store import SignalStore, build_signal_store  # noqa: E402
from build_zero_phase_filter_store import (  # noqa: E402
    FILTER_ID,
    FILTER_LENGTH,
    FIR_SHA256,
    HALF_SUPPORT,
    HIGH_TRANSITION_HZ,
    LOW_TRANSITION_HZ,
    ZeroPhaseFilterStore,
    build_zero_phase_filter_store,
    coefficient_hash,
    filter_segment,
    fir_coefficients,
    formal_bounds,
    validate_filter_manifest,
)


class ZeroPhaseCoreTests(unittest.TestCase):
    """合成信号测试明确证明未来依赖、有限支撑和短segment边界。"""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260713)
        cls.signal = rng.normal(size=(2, 1500)).astype(np.float32) * 1e-5

    def test_frozen_fir_identity_and_symmetry(self) -> None:
        coefficients = fir_coefficients()
        self.assertEqual(coefficients.shape, (413,))
        self.assertEqual(coefficient_hash(coefficients), FIR_SHA256)
        self.assertLess(float(np.max(np.abs(coefficients - coefficients[::-1]))), 2e-17)

    def test_wrong_mne_version_is_rejected_before_filter_design(self) -> None:
        fir_coefficients.cache_clear()
        try:
            with patch.object(mne, "__version__", "future-version"):
                with self.assertRaisesRegex(RuntimeError, "要求MNE 1.11.0"):
                    fir_coefficients()
        finally:
            fir_coefficients.cache_clear()

    def test_future_sample_changes_earlier_output_only_within_half_support(self) -> None:
        changed = self.signal.copy()
        changed[:, 700] += 1e-3
        original_output = filter_segment(self.signal)
        changed_output = filter_segment(changed)
        difference = np.abs(original_output - changed_output)

        # 700点的变化会向前影响至494点，证明该分支不能声明因果。
        self.assertGreater(float(difference[:, 600].max()), 1e-8)
        self.assertLess(float(difference[:, :494].max()), 1e-12)
        self.assertLess(float(difference[:, 907:].max()), 1e-12)

    def test_short_segment_formal_bounds(self) -> None:
        expected = {
            1: (1, 1), 206: (206, 206), 412: (206, 206),
            413: (206, 207), 911: (206, 705), 912: (206, 706),
        }
        for length, bounds in expected.items():
            with self.subTest(length=length):
                self.assertEqual(formal_bounds(0, length), bounds)
                self.assertEqual(bounds[1] - bounds[0] >= 500, length >= 912)


class RealZeroPhaseFilterStoreTests(unittest.TestCase):
    """从真实MAT重建两种滤波存储，独立核对零相位数值和共同窗口支持。"""

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
        cls.zero_output = cls.root / "zero_store"
        cls.causal_output = cls.root / "causal_store"

        cls.base = build_subject(cls.data_root, 1)
        cls.base_manifest = save_subject(cls.index_dir, 1, cls.base)
        cls.offline = build_offline_view(
            cls.base[0], cls.base[1], cls.base[2], cls.base_manifest
        )
        cls.raw_manifest, cls.raw_manifest_path = build_signal_store(
            cls.data_root, cls.index_dir, cls.raw_output, 1
        )
        cls.zero_manifest, cls.zero_manifest_path = build_zero_phase_filter_store(
            cls.raw_output, cls.zero_output, 1
        )
        _, cls.causal_manifest_path = build_causal_filter_store(
            cls.raw_output, cls.causal_output, 1
        )
        cls.raw_store = SignalStore(cls.raw_manifest_path)
        cls.zero_store = ZeroPhaseFilterStore(cls.zero_manifest_path)
        cls.causal_store = CausalFilterStore(cls.causal_manifest_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.raw_store._cache.clear()
        cls.zero_store._signal_store._cache.clear()
        cls.causal_store._signal_store._cache.clear()
        cls.temporary.cleanup()

    def independent_mne_filter(self, values: np.ndarray) -> np.ndarray:
        """通过高级Raw.filter接口独立复算，避免直接调用被测包装函数。"""
        info = mne.create_info(values.shape[0], 250.0, ch_types="eeg")
        raw = mne.io.RawArray(np.asarray(values, dtype=np.float64).copy(), info, verbose=False)
        raw.filter(
            8.0, 30.0, filter_length=FILTER_LENGTH,
            l_trans_bandwidth=LOW_TRANSITION_HZ,
            h_trans_bandwidth=HIGH_TRANSITION_HZ,
            n_jobs=1, method="fir", phase="zero", fir_window="hamming",
            fir_design="firwin", pad="reflect_limited", verbose=False,
        )
        return np.ascontiguousarray(raw.get_data(), dtype=np.float32)

    def test_manifest_freezes_filter_edges_and_counts(self) -> None:
        manifest = self.zero_manifest
        self.assertEqual(manifest["protocol_id"], FILTER_ID.format(subject=1))
        self.assertEqual(manifest["causality"],
                         "noncausal_uses_past_and_future_within_fir_half_support")
        self.assertEqual(manifest["filter"]["filter_length_samples"], 413)
        self.assertEqual(manifest["edge_policy"]["half_support_samples"], 206)
        self.assertEqual(len(manifest["segments"]), 34)
        self.assertEqual(manifest["summaries"]["0train"], {
            "segments": 21, "samples": 557910,
            "edge_excluded_samples": 8652, "formal_samples": 549258,
        })
        self.assertEqual(manifest["summaries"]["1test"], {
            "segments": 13, "samples": 569910,
            "edge_excluded_samples": 5356, "formal_samples": 564554,
        })
        self.assertTrue(all(not Path(item["file"]).is_absolute()
                            for item in manifest["segments"]))

    def test_every_segment_matches_independent_mne_filter(self) -> None:
        for item in self.zero_manifest["segments"]:
            key = (item["session"], item["run"], item["segment"])
            raw = self.raw_store.load_segment(*key)
            expected = self.independent_mne_filter(raw)
            actual = self.zero_store.load_segment(*key)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(item["formal_start_native"],
                             item["start_native"] + HALF_SUPPORT)
            self.assertEqual(item["formal_stop_native"],
                             item["stop_native"] - HALF_SUPPORT)

    def formal_mask(self, store, rows: np.ndarray) -> np.ndarray:
        return np.asarray([store.window_is_formal(row) for row in rows])

    def test_zero_phase_and_shared_window_counts(self) -> None:
        tables = ((self.base[2], 8792), (self.offline[0], 4999),
                  (self.offline[1], 2770))
        expected_shared = (8792, 4993, 2770)
        for index, (rows, expected_zero) in enumerate(tables):
            zero_mask = self.formal_mask(self.zero_store, rows)
            causal_mask = self.formal_mask(self.causal_store, rows)
            shared = zero_mask & causal_mask
            self.assertEqual(int(zero_mask.sum()), expected_zero)
            self.assertEqual(int(shared.sum()), expected_shared[index])

        stage1_shared = (self.formal_mask(self.zero_store, self.offline[0]) &
                         self.formal_mask(self.causal_store, self.offline[0]))
        stage2_shared = (self.formal_mask(self.zero_store, self.offline[1]) &
                         self.formal_mask(self.causal_store, self.offline[1]))
        self.assertEqual(int(stage1_shared[self.offline[0]["session"] == 0].sum()), 2454)
        self.assertEqual(int(stage2_shared[self.offline[1]["session"] == 0].sum()), 1365)

    def test_reader_rejects_both_edges_but_allows_audit(self) -> None:
        online = self.base[2]
        mask = self.formal_mask(self.zero_store, online)
        left = online[0]
        right = next(
            row for row, accepted in zip(online, mask)
            if not accepted and int(row["start"]) >= int(left["start"]) + HALF_SUPPORT
        )
        for row in (left, right):
            with self.assertRaises(ValueError):
                self.zero_store.read_window(row)
            self.assertEqual(self.zero_store.read_window(row, allow_edges=True).shape,
                             (22, 500))
        accepted = online[np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2]]
        self.assertEqual(self.zero_store.read_window(accepted).shape, (22, 500))

    def test_repeat_cli_move_and_manifest_tampering(self) -> None:
        original = self.zero_manifest_path.read_bytes()
        repeated, path = build_zero_phase_filter_store(self.raw_output, self.zero_output, 1)
        self.assertEqual(repeated, self.zero_manifest)
        self.assertEqual(path.read_bytes(), original)

        command = [sys.executable, str(SOURCE_FILE), "--signal-dir", str(self.raw_output),
                   "--subjects", "1", "--output-dir", str(self.zero_output)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        moved = self.root / "moved" / self.zero_manifest["protocol_id"]
        shutil.copytree(self.zero_manifest_path.parent, moved)
        moved_store = ZeroPhaseFilterStore(moved, verify_hashes=True)
        row = next(row for row in self.base[2] if moved_store.window_is_formal(row))
        self.assertEqual(moved_store.read_window(row).shape, (22, 500))

        wrong = json.loads(json.dumps(self.zero_manifest))
        wrong["segments"][0]["formal_start_native"] -= 1
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(wrong)
        wrong = json.loads(json.dumps(self.zero_manifest))
        wrong["filter"]["coefficients"][0] += 1e-4
        with self.assertRaises(RuntimeError):
            validate_filter_manifest(wrong)


if __name__ == "__main__":
    unittest.main(verbosity=2)
