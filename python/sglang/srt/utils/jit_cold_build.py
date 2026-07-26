"""Cold-JIT-build window: keep a deadline-bearing collective off the critical
path while a peer rank is still compiling kernels.

THE DEFECT
----------
sglang's ``jit_kernel`` modules build on FIRST CALL, and for several of them
that first call lands in the pre-capture warmup forward of the CUDA-graph
runner -- not at import, not at model load. On a mixed-arch rig each rank
builds its own artefacts, so the ranks arrive at that forward minutes apart:
one rank sits in ``nvcc`` while another is already inside a collective.

For a host-staged collective with a *cycle-counted* GPU-side deadline
(``HTCCLDeviceTransport``, ``_TIMEOUT_CYCLES = 60e9``, i.e. ~23 s at 2.6 GHz)
this is fatal and, worse, misattributed: the waiting kernel executes
``HTCCL_TRAP()``, which poisons the CUDA context, and the resulting
``cudaErrorLaunchFailure`` surfaces at the NEXT launch -- some unrelated
norm/attention kernel. Measured on the r3 merge tip: 6/6 boots RED on a cold
JIT cache, 1/1 GREEN once the same tree found the cache warm, with a stall
of 23-30 s that matches the deadline to the digit.

THE SHAPE OF THE FIX
--------------------
The deadline is right for steady state -- it is what turns a diverged peer
into a loud trap instead of a wedged GPU. It is wrong for exactly one window:
the eager warmup forwards that precede graph capture, which is where the cold
builds are paid. So the window, not the constant, is what changes.

``run_capture_warmups()`` owns that window. Inside it,
``resolve_timeout_cycles()`` returns a multiplied deadline; outside it, it
returns the caller's constant unchanged. The RECORDED pass runs outside the
window, so the deadline baked into the captured graph -- the one that governs
every replay for the rest of the process's life -- is untouched.

RANK-UNIFORMITY
---------------
The window is opened unconditionally by every rank at the same code point,
before the barrier, and closed at the same code point. It is never opened on
a rank-local predicate (a cache hit, a device kind, an arch): a rank-local
condition in front of a group collective is the hang family this repository
has already been bitten by twice (pynccl, CustomAllreduce). Nothing inside
the window is a collective decision either -- each rank compiles whatever its
own model needs and then meets the others at the barrier that is already
there.

LOUD, NOT SILENT
----------------
If the relaxed deadline tears anyway, the failure is re-raised with the
cold-build hypothesis attached, so the next reader is not sent chasing the
unrelated kernel that happened to launch next.

Default path: with no HTCCL device transport in the process, nothing reads
the window. ``resolve_timeout_cycles`` is the only consumer, and it is called
only from the HTCCL device collectives.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# u64 ceiling for the cycle counter the wait kernels compare against.
_U64_MAX = (1 << 64) - 1

#: How much slack a cold build gets. 40 x 60e9 cycles is ~15 min at 2.6 GHz,
#: which covers an nvcc marlin build with room to spare while still being a
#: finite number. Set to 1 (or 0) to restore the previous behaviour exactly.
_DEFAULT_MULT = 40

_ENV_MULT = "SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT"

_lock = threading.Lock()
_depth = 0
_reason: Optional[str] = None


def cold_build_timeout_mult() -> int:
    """Read the multiplier from the environment at call time.

    Read per call, not at import: the warmup window is entered long after
    import, and tests (and operators) must be able to change it in place.
    """
    raw = os.environ.get(_ENV_MULT)
    if raw is None:
        return _DEFAULT_MULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using the default %d.",
            _ENV_MULT,
            raw,
            _DEFAULT_MULT,
        )
        return _DEFAULT_MULT
    return max(value, 1)


def in_cold_build_window() -> bool:
    """True while a cold-build window is open anywhere in this process."""
    return _depth > 0


def cold_build_window_reason() -> Optional[str]:
    return _reason


def resolve_timeout_cycles(base_cycles: int) -> int:
    """The deadline a device collective should launch with, right now.

    Outside the window this is the identity function -- that is what makes
    the default and the steady-state paths byte-identical.
    """
    if not in_cold_build_window():
        return base_cycles
    mult = cold_build_timeout_mult()
    if mult <= 1:
        return base_cycles
    return min(base_cycles * mult, _U64_MAX)


@contextmanager
def cold_build_window(reason: str) -> Iterator[None]:
    """Open the window for the enclosed block. Re-entrant, exception-safe."""
    global _depth, _reason
    with _lock:
        _depth += 1
        outermost = _depth == 1
        if outermost:
            _reason = reason
    if outermost:
        logger.info(
            "JIT cold-build window open (%s): deadline-bearing device "
            "collectives run with a %dx deadline until it closes. First boot "
            "on an empty kernel cache can spend minutes in nvcc here.",
            reason,
            cold_build_timeout_mult(),
        )
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                _reason = None


class ColdBuildWindowError(RuntimeError):
    """A failure raised from inside the cold-build window.

    Carries the original exception as ``__cause__``; exists so the cold-build
    collision is named at the point of failure instead of being inferred from
    an unrelated kernel's traceback three frames later.
    """


def _cold_build_hint(reason: str) -> str:
    return (
        f"Failure inside the JIT cold-build window ({reason}). On a first boot "
        "with an empty kernel cache the ranks reach this forward minutes "
        "apart, so a peer may still be in nvcc while this rank waits on a "
        "deadline-bearing collective. If the traceback below names an "
        "unrelated kernel, that is the SYMPTOM: a tripped wait kernel poisons "
        "the CUDA context and the error surfaces at the next launch. Warm the "
        "cache (boot once with --disable-cuda-graph), raise "
        f"{_ENV_MULT} (currently {cold_build_timeout_mult()}), or check the "
        "kernel cache for incomplete entries."
    )


def run_capture_warmups(
    forward_fn: Callable[[], Any],
    *,
    repeats: int = 2,
    device_module: Any = None,
    tp_group: Any = None,
    skip_barrier: bool = False,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    reason: str = "cuda-graph capture warmup",
) -> Any:
    """The pre-capture warmup forwards, inside the cold-build window.

    Exactly the loop the graph backends used to inline (synchronize, barrier,
    forward, hook), with two additions: the window, and the loud re-raise.
    Returns the last forward's result, which the breakable backend sizes its
    shared output buffer from. Extracted so it is importable by its test -- a
    copy in the test would let a revert of the fix stay green.
    """
    with cold_build_window(reason):
        try:
            out = None
            for _ in range(repeats):
                if device_module is not None:
                    device_module.synchronize()
                if tp_group is not None and not skip_barrier:
                    tp_group.barrier()
                out = forward_fn()
                if post_warmup_hook is not None:
                    post_warmup_hook()
            return out
        except BaseException as exc:
            hint = _cold_build_hint(reason)
            logger.error("%s", hint)
            if isinstance(exc, Exception):
                raise ColdBuildWindowError(hint) from exc
            raise
