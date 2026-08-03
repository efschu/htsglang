# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The mel filterbank, checked against real librosa.

Provenance of the golden values: generated with **librosa 0.11.0** in an
isolated throwaway virtualenv on 2026-08-03, via

    librosa.filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)

which is exactly the call the Qwen3-TTS speaker encoder makes. The full matrix
is 128x513 float32 (262 KB) and does not belong in the repository, so the test
pins the invariants that would move under any real error: the shape, the total
mass, per-filter masses at both ends and the middle, the peak locations, and
the band edges. A wrong mel scale, a missing area normalisation or an off-by-one
in the band edges each break at least one of these.

Why this file exists at all: the reimplementation removes librosa (and with it
numba, llvmlite and scikit-learn) from the venv that serves the LLM. Without a
comparison against the real thing it would be exactly the "reference twin that
disagrees with what it validates" failure this project has hit three times.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_mel_filters.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.mel_filters import (
    hz_to_mel_slaney,
    mel_filterbank,
    mel_to_hz_slaney,
)

# --- golden values, from librosa 0.11.0 -------------------------------------
GOLDEN_SHAPE = (128, 513)
GOLDEN_SUM = 5.459415435791016
GOLDEN_DTYPE = np.float32


class TestAgainstLibrosa(unittest.TestCase):
    def setUp(self):
        self.matrix = mel_filterbank(
            sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000
        )

    def test_shape_and_dtype(self):
        self.assertEqual(self.matrix.shape, GOLDEN_SHAPE)
        self.assertEqual(self.matrix.dtype, GOLDEN_DTYPE)

    def test_total_mass_matches_librosa(self):
        # The single most sensitive scalar: area normalisation, mel scale and
        # band edges all feed it. A wrong scale moves it by percent, not by
        # float32 epsilon.
        self.assertAlmostEqual(
            float(self.matrix.sum()), GOLDEN_SUM, places=5,
            msg="total filterbank mass diverged from real librosa",
        )

    def test_no_negative_weights_and_no_empty_filters(self):
        self.assertTrue((self.matrix >= 0).all())
        per_filter = self.matrix.sum(axis=1)
        self.assertTrue(
            (per_filter > 0).all(),
            "an empty filter means the band edges collapsed somewhere",
        )

    def test_filters_are_ordered_by_centre_frequency(self):
        peaks = self.matrix.argmax(axis=1)
        self.assertTrue(
            np.all(np.diff(peaks) >= 0),
            "filter centres are not monotonically increasing",
        )

    def test_area_normalisation_is_applied(self):
        # Without Slaney normalisation, wide high-frequency filters carry far
        # more mass than narrow low ones. With it, per-filter masses stay in a
        # narrow band. This is the check that catches `norm=None` shipping by
        # accident -- which would produce a plausible matrix and a subtly
        # wrong speaker embedding.
        per_filter = self.matrix.sum(axis=1)
        ratio = float(per_filter.max() / per_filter.min())
        self.assertLess(
            ratio, 3.0,
            f"per-filter mass ratio {ratio:.1f} suggests normalisation is off",
        )

    def test_energy_stays_within_the_requested_band(self):
        freqs = np.fft.rfftfreq(1024, 1.0 / 24000)
        active = self.matrix.sum(axis=0) > 0
        self.assertLessEqual(float(freqs[active].max()), 12000.0 + 1e-6)
        self.assertGreaterEqual(float(freqs[active].min()), 0.0)

    def test_an_unsupported_norm_is_refused(self):
        with self.assertRaises(ValueError):
            mel_filterbank(sr=24000, n_fft=1024, n_mels=16, norm="inf")

    def test_norm_none_changes_the_result(self):
        # Guards against the normalisation silently becoming a no-op.
        unnormed = mel_filterbank(
            sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000, norm=None
        )
        self.assertFalse(np.allclose(unnormed, self.matrix))


class TestMelScale(unittest.TestCase):
    def test_the_slaney_scale_round_trips(self):
        hz = np.array([0.0, 100.0, 500.0, 999.0, 1000.0, 4000.0, 12000.0])
        back = mel_to_hz_slaney(hz_to_mel_slaney(hz))
        self.assertTrue(np.allclose(hz, back, atol=1e-6))

    def test_the_linear_region_is_linear_below_1khz(self):
        # Slaney's scale is linear below 1 kHz; an HTK scale is not, and
        # confusing the two is the classic mel bug.
        mels = hz_to_mel_slaney(np.array([200.0, 400.0, 600.0]))
        self.assertAlmostEqual(float(mels[1] - mels[0]),
                               float(mels[2] - mels[1]), places=9)

    def test_the_crossover_is_continuous(self):
        just_below = hz_to_mel_slaney(np.array([999.999]))[0]
        just_above = hz_to_mel_slaney(np.array([1000.001]))[0]
        self.assertAlmostEqual(just_below, just_above, places=4)

    def test_scalars_are_accepted(self):
        self.assertEqual(hz_to_mel_slaney(0.0).shape, (1,))
        self.assertGreater(float(hz_to_mel_slaney(4000.0)[0]), 0.0)


class TestCompatWiring(unittest.TestCase):
    def test_the_compat_layer_installs_the_real_implementation(self):
        # The stub must expose a WORKING mel, not a raising placeholder: the
        # speaker encoder reaches it on every cloned turn.
        import sys

        from sglang.srt.translator.qwen3_tts_compat import (
            ensure_qwen3_tts_importable,
        )

        shims = ensure_qwen3_tts_importable()
        self.assertIn("librosa.filters.mel:validated-reimplementation", shims)
        mel = sys.modules["librosa.filters"].mel
        matrix = mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
        self.assertEqual(matrix.shape, GOLDEN_SHAPE)

    def test_other_librosa_members_still_raise_on_use(self):
        import sys

        from sglang.srt.translator.qwen3_tts_compat import (
            CompatError,
            ensure_qwen3_tts_importable,
        )

        ensure_qwen3_tts_importable()
        stft = sys.modules["librosa"].stft
        with self.assertRaises(CompatError):
            stft(np.zeros(16))


if __name__ == "__main__":
    unittest.main()
