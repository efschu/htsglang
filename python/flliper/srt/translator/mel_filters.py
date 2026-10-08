# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``librosa.filters.mel``, reimplemented in numpy so librosa is not a dependency.

The speaker encoder -- the module that turns reference audio into the x-vector
the whole voice-cloning path conditions on -- needs one mel filterbank matrix.
That is the ONLY thing the audio path wanted librosa for, and pulling librosa
into the venv that serves the LLM would drag numba, llvmlite and scikit-learn
along with it for a single deterministic 128x513 matrix.

So it is computed here instead. This is a reimplementation of a published,
deterministic function, not an approximation of one: :mod:`test_mel_filters`
compares it element-wise against a golden matrix produced by **real librosa
0.11.0** in an isolated environment, and the test fails on any difference above
float32 rounding. That comparison is the whole justification for the file --
without it this would be exactly the "reference twin that disagrees with the
thing it validates" family the project has been bitten by three times.

Defaults follow librosa: the **Slaney** mel scale (not HTK) and Slaney area
normalisation. Both matter. The Slaney scale is linear below 1 kHz and
logarithmic above, and getting that boundary wrong shifts every filter; the
area normalisation scales each filter by its bandwidth, and omitting it changes
the relative weight of high filters. Either mistake yields a plausible-looking
matrix and a subtly wrong speaker embedding -- which would degrade cloning in a
way no exception would ever report.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mel_filterbank", "hz_to_mel_slaney", "mel_to_hz_slaney"]

# Slaney's mel scale constants, as used by librosa with htk=False.
_F_MIN = 0.0
_F_SP = 200.0 / 3           # linear region: 66.667 Hz per mel step
_MIN_LOG_HZ = 1000.0        # the linear/log crossover
_MIN_LOG_MEL = (_MIN_LOG_HZ - _F_MIN) / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Hz to mel on the Slaney scale (``librosa.hz_to_mel(..., htk=False)``)."""
    # atleast_1d: a scalar input would otherwise produce a 0-d array that
    # cannot be assigned into by mask, and callers legitimately pass scalars.
    frequencies = np.atleast_1d(np.asanyarray(frequencies, dtype=np.float64))
    mels = (frequencies - _F_MIN) / _F_SP
    log_t = frequencies >= _MIN_LOG_HZ
    mels[log_t] = _MIN_LOG_MEL + np.log(frequencies[log_t] / _MIN_LOG_HZ) / _LOGSTEP
    return mels


def mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Mel to Hz on the Slaney scale (``librosa.mel_to_hz(..., htk=False)``)."""
    mels = np.atleast_1d(np.asanyarray(mels, dtype=np.float64))
    freqs = _F_MIN + _F_SP * mels
    log_t = mels >= _MIN_LOG_MEL
    freqs[log_t] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels[log_t] - _MIN_LOG_MEL))
    return freqs


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str | None = "slaney",
    dtype=np.float32,
) -> np.ndarray:
    """A ``(n_mels, 1 + n_fft // 2)`` mel filterbank matrix.

    Signature and semantics match ``librosa.filters.mel`` for the parameters
    this project uses. ``norm`` accepts ``"slaney"`` (librosa's default, area
    normalisation) or ``None``.
    """
    if fmax is None:
        fmax = float(sr) / 2

    weights = np.zeros((int(n_mels), int(1 + n_fft // 2)), dtype=np.float64)

    # Centre frequencies of each FFT bin.
    fftfreqs = np.fft.rfftfreq(n=int(n_fft), d=1.0 / sr)

    # n_mels + 2 band edges, evenly spaced in mel, mapped back to Hz.
    min_mel = float(hz_to_mel_slaney(np.asarray([fmin], dtype=np.float64))[0])
    max_mel = float(hz_to_mel_slaney(np.asarray([fmax], dtype=np.float64))[0])
    mel_f = mel_to_hz_slaney(np.linspace(min_mel, max_mel, int(n_mels) + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(int(n_mels)):
        # Rising edge of filter i, falling edge of filter i+2.
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    if norm == "slaney":
        # Area normalisation: each filter is scaled by 2 / its Hz bandwidth,
        # so wide high-frequency filters do not dominate narrow low ones.
        enorm = 2.0 / (mel_f[2 : int(n_mels) + 2] - mel_f[: int(n_mels)])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(
            f"unsupported mel norm {norm!r}; this reimplementation covers "
            "librosa's 'slaney' default and None"
        )

    return weights.astype(dtype)
