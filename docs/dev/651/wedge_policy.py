#!/usr/bin/env python
"""#651: keep the gfx1103 MES-wedge regime unreachable.

THE DRIVER BUG (not ours to fix). On this laptop's Radeon 780M (gfx1103) the
amdgpu firmware scheduler hangs under prefill load:

    amdgpu 0000:c4:00.0: MES failed to respond to msg=REMOVE_QUEUE   (x3)
    amdgpu 0000:c4:00.0: GPU reset(N) succeeded!
    amdgpu 0000:c4:00.0: [drm] device wedged, but recovered through reset

Six such resets occurred in one session, each killing the server. Userspace
sees only `HIP error: unspecified launch failure`, and three different
subsystems each report it as their own fault (the MoE-align RuntimeCheck,
hipBLAS `INTERNAL_ERROR`, torch `AcceleratorError`). dmesg is the only place
that names it.

REPRODUCER (no server, no model, ~2 minutes):
    fill device memory to ~3% free, then run a bf16 GEMM.
    M=512  -> passes.
    M=1024 -> wedges the GPU.
See docs/dev/651/bf16_gemm_falsifier.py for the shape sweep and
FINAL_651.md section 3 for the full chain.

WHAT THIS MODULE DOES. It cannot fix a firmware hang, so it makes the
triggering regime unreachable instead, and says so loudly rather than
silently degrading:

  * the GGUF large-batch path runs ONE bf16 GEMM whose M is the prefill chunk,
    so `--chunked-prefill-size` is capped on affected hardware;
  * that GEMM only wedges under memory pressure, so a free-memory headroom
    floor is required as well -- a large GEMM must never run at ~3% free.

Both are POLICY, expressed as a pure function so it can be unit-tested without
a GPU and reused from a pre-flight guard. It is deliberately NOT a claim that
the wedge is fixed: `--chunked-prefill-size 256` was measured letting one full
prefill sweep through and then wedging the GPU on a later run.
"""

from typing import List, NamedTuple, Optional

#: Architectures observed to wedge. Kept as an explicit list rather than a
#: "is it an iGPU" heuristic: the wedge was MEASURED here, and quietly
#: applying a throughput cap to hardware that does not need it would be a
#: silent performance regression on every other ROCm box.
WEDGE_ARCHS = ("gfx1103",)

#: Largest prefill chunk that survived on gfx1103. M=512 passed and M=1024
#: wedged in the standalone reproducer, so the cap sits a full step below the
#: first observed failure rather than immediately beneath it.
MAX_CHUNKED_PREFILL = 256

#: Minimum free device memory before a large GEMM is considered safe. The
#: wedge was reproduced at ~3% free (736 MiB of 24 GiB); this floor is an
#: order of magnitude above that, in absolute terms so it does not shrink on
#: a small card.
MIN_FREE_MIB_FOR_LARGE_GEMM = 2048

#: A GEMM at or above this M is "large" for the purposes of the floor above.
LARGE_GEMM_M = 512


class WedgePolicyResult(NamedTuple):
    """Outcome of the policy check.

    `errors` block a boot; `warnings` are printed but allow it. Kept separate
    so a caller can decide, and so tests can assert on which bucket a case
    lands in rather than on message text.
    """

    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def check_wedge_policy(
    arch: Optional[str],
    chunked_prefill_size: Optional[int],
    free_mib: Optional[float] = None,
) -> WedgePolicyResult:
    """Decide whether this configuration may run on this architecture.

    `arch` is the gcnArchName as torch reports it (None when unknown, e.g. a
    CPU-only rank -- which is not affected and must not be blocked).

    Note that on this laptop `HSA_OVERRIDE_GFX_VERSION=11.0.0` makes torch
    report **gfx1100** for gfx1103 silicon, so callers that can only see the
    overridden name will not match WEDGE_ARCHS. That is why the override is
    treated as a separate warning case below rather than silently passing:
    the check is only as good as the arch string it is handed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if arch is None:
        return WedgePolicyResult(errors, warnings)

    affected = any(arch.startswith(a) for a in WEDGE_ARCHS)

    if not affected:
        if arch.startswith("gfx1100"):
            warnings.append(
                "arch reports gfx1100: if this is gfx1103 running under "
                "HSA_OVERRIDE_GFX_VERSION=11.0.0, the MES-wedge policy is NOT "
                "being applied. Pass the physical arch to apply it."
            )
        return WedgePolicyResult(errors, warnings)

    if chunked_prefill_size is not None and chunked_prefill_size > MAX_CHUNKED_PREFILL:
        errors.append(
            f"--chunked-prefill-size {chunked_prefill_size} exceeds "
            f"{MAX_CHUNKED_PREFILL} on {arch}. The GGUF large-batch path runs "
            f"one bf16 GEMM with M = the prefill chunk, and on this "
            f"architecture that wedges the GPU in firmware (amdgpu 'MES failed "
            f"to respond' -> 'GPU reset'), killing the server. Measured: "
            f"M=512 passes, M=1024 wedges. Use "
            f"--chunked-prefill-size {MAX_CHUNKED_PREFILL}."
        )

    if free_mib is not None and free_mib < MIN_FREE_MIB_FOR_LARGE_GEMM:
        errors.append(
            f"only {free_mib:.0f} MiB device memory free on {arch}; a large "
            f"GEMM needs at least {MIN_FREE_MIB_FOR_LARGE_GEMM} MiB of "
            f"headroom. The wedge was reproduced at ~3% free (736 MiB). "
            f"Lower --mem-fraction-static, or reduce the model/KV footprint."
        )

    if not errors:
        warnings.append(
            f"{arch} is subject to the amdgpu MES wedge under prefill load. "
            f"--chunked-prefill-size {chunked_prefill_size} and the memory "
            f"floor are MITIGATIONS, not a fix: a sweep that survived once has "
            f"wedged on a later run. Watch dmesg for 'GPU reset'."
        )

    return WedgePolicyResult(errors, warnings)


def physical_arch() -> Optional[str]:
    """The PHYSICAL architecture, seeing through HSA_OVERRIDE_GFX_VERSION.

    torch reports whatever the override says, so the override is undone here
    by asking the device its marketing name and mapping the one case this
    project actually runs on. Returns None when no accelerator is visible,
    which keeps a CPU-only rank out of the policy entirely.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        name = getattr(props, "name", "") or ""
        arch = getattr(props, "gcnArchName", "") or ""
        # Radeon 780M is the gfx1103 iGPU; under the gfx1100 override the
        # arch string lies but the device NAME does not.
        if "780M" in name:
            return "gfx1103"
        return arch or None
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    import sys

    arch = physical_arch()
    size = int(sys.argv[1]) if len(sys.argv) > 1 else None
    res = check_wedge_policy(arch, size)
    print(f"arch: {arch}")
    for w in res.warnings:
        print(f"WEDGE-POLICY WARNING: {w}")
    for e in res.errors:
        print(f"WEDGE-POLICY ERROR: {e}")
    print("WEDGE-POLICY:", "OK" if res.ok else "REFUSED")
    sys.exit(0 if res.ok else 1)
