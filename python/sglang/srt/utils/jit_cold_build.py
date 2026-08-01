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
(``BarlinkDeviceTransport``, ``_TIMEOUT_CYCLES = 60e9``, i.e. ~23 s at 2.6 GHz)
this is fatal and, worse, misattributed: the waiting kernel executes
``BARLINK_TRAP()``, which poisons the CUDA context, and the resulting
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

That hypothesis is about PEER ranks, so it is attached only when a peer
exists. On a single-rank boot the original exception is re-raised unchanged
(#257): an OOM in the warmup forward is an OOM, and dressing it as a
cold-build collision sends the reader looking for a second rank that was
never there.

ATTACHED, NEVER SUBSTITUTED (#386)
----------------------------------
The hypothesis is also only about a specific FAMILY of failures: the
allocator, and the launch failure a tripped wait kernel leaves behind. Any
other exception that happens to be in flight when the window is open is an
independent error, and replacing its type and message with the cold-build
text costs the reader the only thing that identified it. Three times on
record:

  #377 (twice)  FP8 layer construction failed; the operator got the window's
                text plus "set --mem-fraction-static to a smaller value" from
                the graph runner one frame further out. The real message was
                two ``__context__`` hops down.
  #384 (once)   an sgl-kernel wheel without ``int8_scaled_mm`` failed during
                layer construction and read as the same memory advice, with
                the sm75 gencode floor as a false secondary diagnosis.

So ``_is_cold_build_symptom`` decides, and it is deliberately conservative:
a symptom keeps today's message, EVERYTHING else is re-raised as itself with
the window named in a note. The asymmetry is the point -- a wrong "lower
your memory fraction" costs hours of misdiagnosis, an honest error carrying
one extra note costs nothing.

Default path: with no barlink device transport in the process, nothing reads
the window. ``resolve_timeout_cycles`` is the only consumer, and it is called
only from the barlink device collectives.
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


#: Lower-cased message fragments that mark a failure as a cold-build-window
#: SYMPTOM. Enumerated from this module's own history, nothing speculative:
#:
#:  * the allocator, in the shapes torch and the driver print it. The window
#:    is where a peer's nvcc/ptxas and its freshly loaded modules sit on the
#:    same card, so an allocation that fails here is plausibly the window's.
#:  * ``cudaErrorLaunchFailure`` -- the exact aftermath described at the top
#:    of this file: ``BARLINK_TRAP()`` poisons the context and the error
#:    surfaces at the NEXT launch, on some unrelated kernel.
#:
#: Substring match on the whole message, case-folded: torch prefixes these
#: with device ids, byte counts and "CUDA error:" in several arrangements.
_SYMPTOM_MESSAGE_FRAGMENTS = (
    "out of memory",  # covers "CUDA out of memory" and "CUDA error: out of memory"
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "unspecified launch failure",
    "cudaerrorlaunchfailure",
)


def _torch_oom_types() -> tuple:
    """``torch.cuda.OutOfMemoryError`` and its aliases, if torch is importable.

    A type check, not only a message check: the allocator's text has been
    reworded between torch releases, the class has not. Guarded so this
    module stays importable (and testable) without torch.
    """
    types: list = []
    try:
        import torch
    except Exception:  # pragma: no cover - torch is present in every real boot
        return ()
    for holder, name in ((torch, "OutOfMemoryError"), (torch.cuda, "OutOfMemoryError")):
        candidate = getattr(holder, name, None)
        if isinstance(candidate, type) and candidate not in types:
            types.append(candidate)
    return tuple(types)


def _is_cold_build_symptom(exc: BaseException) -> bool:
    """Is this failure one the cold-build window's diagnosis actually explains?

    True only for the enumerated families above. When in doubt the answer is
    False, and False is the cheap direction: the caller then re-raises the
    original exception with the window named in a note, which costs the
    reader one line. A wrong True replaces a real error's type and message
    with memory advice -- that is #377 and #384.
    """
    oom_types = _torch_oom_types()
    if oom_types and isinstance(exc, oom_types):
        return True
    # The peer-liveness deadline is the window's own instrument: it expires
    # here precisely when a peer is still in nvcc (see
    # barlink_bar1_ext.grouped_jit_build). PeerLostError is NOT included --
    # a dead peer is decided, self-describing, and not a cold build.
    try:
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            CollectiveTimeoutError,
        )

        if isinstance(exc, CollectiveTimeoutError):
            return True
    except Exception:  # pragma: no cover - barlink is optional in this process
        pass
    text = f"{exc}".casefold()
    return any(fragment in text for fragment in _SYMPTOM_MESSAGE_FRAGMENTS)


def _attach_note(exc: BaseException, note: str) -> None:
    """Record the window on the exception without touching type or message.

    ``BaseException.add_note`` is 3.11+; this package supports 3.10, where
    the note reaches the reader through the log line the caller emits next.
    """
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)


def _peer_ranks_possible(tp_group: Any) -> bool:
    """True only when this process provably has a peer rank.

    The cold-build hypothesis is a statement about OTHER ranks: one of them
    is still in nvcc while this one waits in a deadline-bearing collective.
    On a single-rank boot there is no such rank, so the hypothesis cannot
    hold and must not be attached to the failure (#257: a plain
    ``torch.OutOfMemoryError`` from the decode warmup surfaced as a
    ``ColdBuildWindowError`` advertising a peer, on a TP=1 boot, with the
    real cause reachable only through the chained traceback).

    Undeterminable counts as "no peer": the hint is a hypothesis, and
    asserting it without evidence is exactly the defect.
    """
    ws = getattr(tp_group, "world_size", None)
    if isinstance(ws, int) and ws > 0:
        return ws > 1
    try:
        from sglang.srt.distributed import get_world_group

        return int(get_world_group().world_size) > 1
    except Exception:
        return False


def _single_rank_note(reason: str) -> str:
    return (
        f"Failure inside the JIT cold-build window ({reason}) on a "
        "single-rank boot: there is no peer that could still be in nvcc, so "
        "the cold-build collision does not apply and the original exception "
        "is re-raised unchanged -- it IS the cause, not a symptom. (A cold "
        "kernel cache still costs minutes in this window on a first boot; "
        "it just cannot trip a collective deadline without a peer.)"
    )


def _passthrough_note(reason: str) -> str:
    """The one line a non-symptom failure carries out of the window (#386)."""
    return (
        f"Raised during the JIT cold-build window ({reason}); the exception "
        "above is its own cause and is passed through unchanged. If it does "
        "turn out to be an out-of-memory symptom, --mem-fraction-static and "
        f"{_ENV_MULT} are the knobs the window would have pointed at."
    )


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
            if not isinstance(exc, Exception):
                raise
            if not _peer_ranks_possible(tp_group):
                # Single rank: pass the cause through as itself. Wrapping it
                # would bury an OOM (or any other honest local failure) one
                # __cause__ level down behind a hypothesis that cannot apply.
                logger.error("%s", _single_rank_note(reason))
                raise
            if not _is_cold_build_symptom(exc):
                # #386: a peer exists, but this failure is not one the
                # cold-build hypothesis explains. Attach the window, keep the
                # exception -- its type is what the reader dispatches on (an
                # AttributeError from layer construction must arrive as an
                # AttributeError) and its message is the only text that says
                # what actually broke.
                note = _passthrough_note(reason)
                logger.error("%s: %s\n%s", type(exc).__name__, exc, note)
                _attach_note(exc, note)
                raise
            hint = _cold_build_hint(reason)
            logger.error("%s", hint)
            raise ColdBuildWindowError(hint) from exc
