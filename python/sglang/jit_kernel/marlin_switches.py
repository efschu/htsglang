"""Task #49 (2026-09-20): the Marlin MoE build- and grid-shape switches.

Pure string/env logic, deliberately free of ``torch`` and of any CUDA import, so
the desk tests drive every branch without a GPU and without the runtime.

Three of the four switches are consumed by the CUDA host code
(``csrc/gemm/marlin_moe/moe_wna16_marlin.cuh``), which reads the SAME
environment variables through ``std::getenv``. The parsers here mirror that C++
byte for byte -- first-character truthiness, ``atoi`` for the integer -- because
the only purpose of the Python copy is the ONE log line per rank that says which
switches a boot actually ran with. A log line whose parser disagrees with the
kernel's would be worse than no log line at all.

    SGLANG_MARLIN_EPILOGUE_SYNC   default ON  -- the barrier between
        ``cp_async_wait<0>()`` and the epilogue's first write into ``sh_red``
        (which aliases the B staging buffer ``sh_b``). '0' removes it again.
    SGLANG_MARLIN_NO_K_SPLIT      default off -- round the per-threadblock
        stripe up to whole k-columns so ``slice_count == 1`` and no
        cross-threadblock partial sum exists at all.
    SGLANG_MARLIN_SMS_OVERRIDE    default off -- pretend the device has N SMs
        when sizing the grid (downward only; the shared lock workspace was
        sized from the hardware SM count at load time).
    SGLANG_JIT_MARLIN_ARCH_OVERRIDE  default off -- build the Marlin MoE JIT
        module for a DIFFERENT virtual architecture and let the driver JIT the
        embedded PTX for the real one. This is the discriminator for "is the
        fault in the sm_120 cubin ptxas emits, or in the algorithm".
"""

from __future__ import annotations

import os
import re
from typing import List, NamedTuple, Optional

# Env names, in one place so the log line and the tests cannot drift from the
# strings the CUDA host code reads.
ENV_EPILOGUE_SYNC = "SGLANG_MARLIN_EPILOGUE_SYNC"
ENV_NO_K_SPLIT = "SGLANG_MARLIN_NO_K_SPLIT"
ENV_SMS_OVERRIDE = "SGLANG_MARLIN_SMS_OVERRIDE"
ENV_ARCH_OVERRIDE = "SGLANG_JIT_MARLIN_ARCH_OVERRIDE"


def _c_truthy_first_char(raw: Optional[str]) -> bool:
    """`raw[0] in '1tTyY'` -- exactly what marlin_moe_no_k_split() does."""
    if not raw:
        return False
    return raw[0] in ("1", "t", "T", "y", "Y")


def _c_falsy_first_char(raw: Optional[str]) -> bool:
    """`raw[0] in '0fFnN'` -- exactly what marlin_moe_epilogue_sync() negates."""
    if not raw:
        return False
    return raw[0] in ("0", "f", "F", "n", "N")


def _c_atoi(raw: Optional[str]) -> int:
    """C ``atoi``: leading whitespace, optional sign, leading digits, 0 on junk."""
    if not raw:
        return 0
    m = re.match(r"\s*([+-]?\d+)", raw)
    return int(m.group(1)) if m else 0


def epilogue_sync_on(env=None) -> bool:
    env = os.environ if env is None else env
    return not _c_falsy_first_char(env.get(ENV_EPILOGUE_SYNC))


def no_k_split_on(env=None) -> bool:
    env = os.environ if env is None else env
    return _c_truthy_first_char(env.get(ENV_NO_K_SPLIT))


def sms_override(env=None) -> int:
    """The raw request. 0 means 'unset / ignored'; negatives are ignored too."""
    env = os.environ if env is None else env
    n = _c_atoi(env.get(ENV_SMS_OVERRIDE))
    return n if n > 0 else 0


def effective_sms(hardware_sms: int, env=None) -> int:
    """What the kernel launch will actually use as the grid width.

    Mirrors the clamp in the .cuh: the override applies only DOWNWARD, because
    the shared lock workspace was sized ``hardware_sms * 4`` at weight-load time
    and a wider grid would fail the workspace check instead of running."""
    want = sms_override(env)
    if want > 0 and want < int(hardware_sms):
        return want
    return int(hardware_sms)


class ArchOverride(NamedTuple):
    major: int
    minor: int
    suffix: str
    ptx: bool

    @property
    def target_name(self) -> str:
        """The value handed to ``override_jit_cuda_arch`` / TVM_FFI_CUDA_ARCH_LIST."""
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def virtual_arch(self) -> str:
        return f"compute_{self.major}{self.minor}{self.suffix}"


# The only suffixes nvcc knows are 'a' (arch-specific, sm_90a/sm_100a/sm_120a)
# and 'f' (family-specific, CUDA 12.9+). Anything else is a typo, and a typo
# must raise rather than build a module for an architecture nobody meant.
_ARCH_RE = re.compile(
    r"^(?:sm_|compute_)?(\d{1,2})(?:\.)?(\d)([af]?)$",
)


def parse_arch_override(raw: Optional[str]) -> Optional[ArchOverride]:
    """Parse SGLANG_JIT_MARLIN_ARCH_OVERRIDE.

    Accepted: ``9.0``, ``90``, ``sm_90``, ``compute_90``, ``12.0a``, ``10.0``.
    An optional ``+ptx`` / ``+noptx`` suffix decides whether PTX for the virtual
    architecture is embedded alongside the cubin.

    PTX defaults to ON, and that default is the point: tvm-ffi emits only
    ``-gencode=arch=compute_XY,code=sm_XY``, i.e. a cubin and nothing else, and a
    cubin built for one compute-capability MAJOR does not load on another. An
    override without PTX would therefore produce a module the 5090 cannot load,
    which reads exactly like a broken switch. With PTX embedded the loader finds
    no matching cubin, falls back to the PTX and the DRIVER compiles it for
    sm_120 -- a different code generator than the one that produced the native
    sm_120 cubin, which is precisely the A/B this switch exists for.

    Returns None for unset/empty. Raises ValueError on junk, so a typo in a boot
    line fails at the first Marlin call instead of silently running the default.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    ptx = True
    body = raw
    low = raw.lower()
    if low.endswith("+noptx"):
        ptx, body = False, raw[: -len("+noptx")]
    elif low.endswith("+ptx"):
        ptx, body = True, raw[: -len("+ptx")]

    m = _ARCH_RE.match(body.strip().lower())
    if m is None:
        raise ValueError(
            f"{ENV_ARCH_OVERRIDE}={raw!r} is not a CUDA architecture. Expected "
            "e.g. '9.0', '90', 'sm_90', 'compute_90', '12.0a', optionally with "
            "'+ptx' (default) or '+noptx'."
        )
    major, minor, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    if major < 7:
        raise ValueError(
            f"{ENV_ARCH_OVERRIDE}={raw!r}: Marlin needs sm_80 or newer; the "
            "pre-Ampere stub compiles to an empty kernel."
        )
    return ArchOverride(major=major, minor=minor, suffix=suffix, ptx=ptx)


def arch_override_from_env(env=None) -> Optional[ArchOverride]:
    env = os.environ if env is None else env
    return parse_arch_override(env.get(ENV_ARCH_OVERRIDE))


def arch_override_cuda_cflags(override: Optional[ArchOverride]) -> List[str]:
    """Extra nvcc flags that embed PTX for the overridden virtual architecture.

    ``override_jit_cuda_arch`` already makes tvm-ffi emit
    ``-gencode=arch=compute_XY,code=sm_XY``; this adds the ``code=compute_XY``
    half so the fatbin also carries PTX. Both land in the JIT build hash, so the
    overridden module gets its OWN cache directory and can never be confused
    with the native one.
    """
    if override is None or not override.ptx:
        return []
    return [f"-gencode=arch={override.virtual_arch},code={override.virtual_arch}"]


#: What `cudaDevAttrMaxSharedMemoryPerBlockOptin` must report on every
#: architecture Marlin runs on here: 99 KiB on sm_86 (3080) AND on sm_120
#: (5090). Consumer Blackwell did NOT shrink the opt-in budget.
EXPECTED_SMEM_OPTIN = 101376


def smem_optin_verdict(value: Optional[int]) -> str:
    """Name a driver that mis-reports the opt-in shared-memory budget.

    Reported on RTX 5090 with early Blackwell drivers: the attribute comes back
    as 0x100000001 (4294967297) or 0 instead of 101376. That number is not
    cosmetic here -- the Marlin host code reads the SAME attribute into an
    ``int`` (truncating 0x100000001 to 1) and feeds it to `is_valid_config`,
    which is the check vLLM PR #11493 added to keep the shared-memory buffers
    from overlapping. A poisoned budget disables that check on ONE card while
    the two 3080s keep a correct one, which is a structural explanation for
    'sm_120 dirty, sm_86 clean' that needs no undiscovered hardware bug.

    Printing it costs nothing and turns a lead into a measurement."""
    if value is None:
        return "unknown"
    value = int(value)
    if value == EXPECTED_SMEM_OPTIN:
        return "ok"
    if value <= 0 or value > (1 << 31) - 1:
        return "DRIVER-POISONED"
    return "unexpected"


def switch_census(
    hardware_sms: Optional[int] = None,
    env=None,
    smem_optin: Optional[int] = None,
) -> dict:
    """Everything the one-shot log line needs, as plain data (testable)."""
    override = arch_override_from_env(env)
    return {
        "smem_optin": None if smem_optin is None else int(smem_optin),
        "smem_optin_verdict": smem_optin_verdict(smem_optin),
        "epilogue_sync": epilogue_sync_on(env),
        "no_k_split": no_k_split_on(env),
        "sms_requested": sms_override(env),
        "sms_hw": None if hardware_sms is None else int(hardware_sms),
        "sms_effective": (
            None if hardware_sms is None else effective_sms(hardware_sms, env)
        ),
        "arch_override": None if override is None else override.target_name,
        "ptx_jit": bool(override is not None and override.ptx),
    }


def format_switch_log(device_arch: str, census: dict) -> str:
    """The named log line, one per rank. Grep marker: ``[nan-49c] marlin``."""
    return (
        "[nan-49c] marlin switches: device_arch=%s arch_override=%s ptx_jit=%s "
        "epilogue_sync=%s no_k_split=%s sms_hw=%s sms_effective=%s "
        "sms_requested=%s smem_optin=%s(%s)"
        % (
            device_arch,
            census["arch_override"],
            census["ptx_jit"],
            census["epilogue_sync"],
            census["no_k_split"],
            census["sms_hw"],
            census["sms_effective"],
            census["sms_requested"],
            census["smem_optin"],
            census["smem_optin_verdict"],
        )
    )
