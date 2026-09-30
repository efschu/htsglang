# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Parity gate: every built engine is graded against an fp32 reference.

DESIGN #333 §8.7 fixes the *metrics* -- PSNR and SSIM against an fp32
end-to-end reference, on a fixed clip set containing both high-detail and
flat content -- but does not fix the thresholds. The thresholds below are
therefore this build's, not the design's, and are recorded as such so they
can be argued with:

``PSNR >= 40 dB`` and ``SSIM >= 0.995`` for an fp16 engine against the fp32
reference of the same graph. At 40 dB the mean squared error is 1e-4 of full
scale, which is below the 8-bit quantisation step the output is encoded at,
so a passing engine cannot differ from the reference by a representable
output value on average. The SSIM floor is the structural counterpart: PSNR
alone tolerates a small uniform bias that SSIM does not.

The gate runs once per engine build and its result is written into the
engine's provenance manifest, so an engine can never be in the cache without
a recorded verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: fp16-against-fp32 thresholds. See the module docstring for the derivation;
#: these are gate parameters of this build, not values taken from DESIGN #333.
DEFAULT_PSNR_DB = 40.0
DEFAULT_SSIM = 0.995


@dataclass(frozen=True)
class ParityResult:
    psnr_db: float
    ssim: float
    max_abs_error: float
    passed: bool
    psnr_threshold_db: float
    ssim_threshold: float
    samples: int
    note: str = ""

    def as_manifest(self) -> dict:
        return asdict(self)


def psnr(candidate, reference, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio over the whole tensor, in dB."""
    import torch

    a = candidate.to(torch.float32)
    b = reference.to(torch.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()


def ssim(candidate, reference, data_range: float = 1.0, window: int = 11) -> float:
    """Mean SSIM over an NCHW pair, Gaussian-windowed, per the original paper.

    Implemented here rather than pulled from a dependency so the gate has no
    optional-package failure mode on a host where the engine build succeeded.
    """
    import torch
    import torch.nn.functional as F

    a = candidate.to(torch.float32)
    b = reference.to(torch.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dim() != 4:
        raise ValueError("SSIM expects NCHW")

    sigma = 1.5
    coords = (
        torch.arange(window, dtype=torch.float32, device=a.device) - (window - 1) / 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (
        (g[:, None] @ g[None, :]).expand(a.shape[1], 1, window, window).contiguous()
    )

    pad = window // 2
    groups = a.shape[1]
    mu_a = F.conv2d(a, kernel, padding=pad, groups=groups)
    mu_b = F.conv2d(b, kernel, padding=pad, groups=groups)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = F.conv2d(a * a, kernel, padding=pad, groups=groups) - mu_a2
    sigma_b2 = F.conv2d(b * b, kernel, padding=pad, groups=groups) - mu_b2
    sigma_ab = F.conv2d(a * b, kernel, padding=pad, groups=groups) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return (numerator / denominator).mean().item()


def grade(
    candidate,
    reference,
    *,
    psnr_threshold_db: float = DEFAULT_PSNR_DB,
    ssim_threshold: float = DEFAULT_SSIM,
    data_range: float = 1.0,
    note: str = "",
) -> ParityResult:
    """Grade one candidate batch against its fp32 reference."""
    import torch

    value_psnr = psnr(candidate, reference, data_range)
    value_ssim = ssim(candidate, reference, data_range)
    max_abs = torch.max(
        torch.abs(candidate.to(torch.float32) - reference.to(torch.float32))
    ).item()
    return ParityResult(
        psnr_db=value_psnr,
        ssim=value_ssim,
        max_abs_error=max_abs,
        passed=value_psnr >= psnr_threshold_db and value_ssim >= ssim_threshold,
        psnr_threshold_db=psnr_threshold_db,
        ssim_threshold=ssim_threshold,
        samples=int(candidate.shape[0]),
        note=note,
    )


class ParityGateFailure(RuntimeError):
    """A built engine did not meet the parity thresholds."""

    def __init__(self, result: ParityResult, engine: str) -> None:
        super().__init__(
            f"parity gate failed for {engine}: PSNR {result.psnr_db:.2f} dB "
            f"(threshold {result.psnr_threshold_db}) SSIM {result.ssim:.5f} "
            f"(threshold {result.ssim_threshold}), max abs error "
            f"{result.max_abs_error:.5f}"
        )
        self.result = result
