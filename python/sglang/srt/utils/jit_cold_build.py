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
import time
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

#: #1158: the nominal device clock the cycle deadlines are documented
#: against ("60_000_000_000 cycles ~30 s at 2 GHz", barlink_device.py
#: _TIMEOUT_CYCLES and barlink_host.py). Used only to express the peers'
#: wall-clock build cap in the device reader's unit; a faster clock makes
#: the device bound slightly tighter than the host bound, never looser.
_NOMINAL_CYCLES_PER_S = 2_000_000_000

#: #1158: what the opener's deadline falls back to when the peers' cap
#: cannot be read at all -- the same 900 s published default the #1073
#: poller falls back to (barlink_liveness.py), never "uncapped".
_FALLBACK_CAP_S = 900.0

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


def build_cap_s() -> float:
    """#1158: the peers' absolute build cap, read from where they read it.

    ``barlink_build_window.build_cap_s`` (``SGLANG_BARLINK_BUILD_WINDOW_CAP_S``,
    default 900 s; this rig's launcher sets 60) is the deadline a PEER honours
    for a published window. The opener's own extension is bounded by the
    same number below, so the two sides of one collective agree on how long
    a build may hold a deadline open.
    """
    try:
        from sglang.srt.distributed.device_communicators import (
            barlink_build_window,
        )

        return float(barlink_build_window.build_cap_s())
    except Exception:  # pragma: no cover - the module is optional context
        return _FALLBACK_CAP_S


def capped_cold_build_deadline(base: float, cap: float) -> float:
    """#1158 THE OPENER HONOURS THE CAP TOO: ``min(base * mult, base + cap)``.

    ONE formula for both readers (host seconds in ``wait_timeout_s``, device
    cycles in ``resolve_timeout_cycles``), so they cannot drift apart. The
    rank that OPENS a cold-build window used to extend its own deadline by
    the bare multiplier (x40) while the peers reading its marker stopped at
    ``build_cap_s()`` -- so an opener waiting on a peer that never joins sat
    from the peers' 60 s cap to the 300 s scheduler watchdog (boot weg1b3,
    23:59:54 -> 00:06:47, '#1033c CUTOVER FORWARD WARMUP begin' with no
    done). Capped, it ends in the existing named abort within base + cap.
    ``mult <= 1`` and ``cap <= 0`` both mean "no extension", exactly as they
    do for the peers.
    """
    mult = cold_build_timeout_mult()
    if mult <= 1 or cap <= 0:
        return base
    return min(base * mult, base + cap)


def resolve_timeout_cycles(base_cycles: int) -> int:
    """The deadline a device collective should launch with, right now.

    Outside the window this is the identity function -- that is what makes
    the default and the steady-state paths byte-identical. Inside it the
    extension is capped at ``build_cap_s()`` worth of nominal device cycles
    (#1158), the same bound ``barlink_liveness.wait_timeout_s`` applies.
    """
    if not in_cold_build_window():
        return base_cycles
    cap_cycles = build_cap_s() * _NOMINAL_CYCLES_PER_S
    return int(min(capped_cold_build_deadline(base_cycles, cap_cycles), _U64_MAX))


def _publish_build_window(reason: str) -> None:
    """Make this window visible to the peers (#615). Never raises.

    Imported lazily and swallowed on failure for the same reason every other
    reference to the distributed package from ``utils`` is: this module is
    imported by processes that have no barlink transport, no peers and in
    some cases no torch.distributed, and a diagnostic must never be the thing
    that breaks them. Publication failing degrades to exactly the pre-#615
    behaviour -- a rank-local window -- which is why nothing here reports an
    error the caller could act on.
    """
    try:
        from sglang.srt.distributed.device_communicators.barlink_build_window import (
            publish_building,
        )

        publish_building(reason)
    except Exception:  # noqa: BLE001 - see the docstring
        pass


def _withdraw_build_window() -> None:
    """Symmetric partner of ``_publish_build_window``. Never raises."""
    try:
        from sglang.srt.distributed.device_communicators.barlink_build_window import (
            clear_building,
        )

        clear_building()
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def cold_build_window(reason: str) -> Iterator[None]:
    """Open the window for the enclosed block. Re-entrant, exception-safe.

    Open and close are logged SYMMETRICALLY, at the same level, with the same
    prefix. That is not cosmetic. The #431 window
    (``/spinning/gpu-battery-results/2026-08-02_431_repro/jit/``) collected
    ``grep -aE 'JIT cold-build window (open|close)'`` and read 6 OPEN / 0
    CLOSE over a 22-minute stall, which looks exactly like a window that
    leaked -- while in fact this function had never emitted a close line at
    all, so zero was the only number that reading could ever produce. An
    accounting instrument that can only count one direction reports a leak
    whether or not one exists; the close line is what makes the open count
    falsifiable.

    GROUP VISIBILITY (#615). The depth counter above is RANK-LOCAL, so every
    deadline it relaxes is one evaluated in the builder's own process -- and
    the rank stuck in nvcc is never the rank whose deadline matters. The
    peers are, and they are not in a window. So opening this window also
    PUBLISHES the fact, via ``barlink_build_window``: one marker file the
    peers can read without any cooperation from a process that is about to
    disappear into a compiler. Every existing call site therefore becomes
    group-visible without moving, which is the point of hooking it here
    rather than at each build.
    """
    global _depth, _reason
    opened_at = time.monotonic()
    with _lock:
        _depth += 1
        outermost = _depth == 1
        if outermost:
            _reason = reason
    _publish_build_window(reason)
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
            closed = _depth == 0
            if closed:
                _reason = None
        _withdraw_build_window()
        if closed:
            logger.info(
                "JIT cold-build window close (%s) after %.1fs: "
                "deadline-bearing device collectives are back on their "
                "steady-state deadline.",
                reason,
                time.monotonic() - opened_at,
            )


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
            # Trailing join: the loop's synchronize/barrier lead each forward,
            # so without this the LAST warmup's async work is still in flight
            # when the caller constructs torch.cuda.CUDAGraph() and enters
            # stream capture. JIT compilation triggered by that forward
            # (DeepGEMM/TVM, and in the DSpark compact ragged-verify arm every
            # new non-uniform verify_lens shape) then issues driver calls --
            # cuModuleLoadData and friends -- from inside the capture region,
            # which surfaces as CUDA_ERROR_ILLEGAL_ADDRESS or
            # cudaErrorStreamCaptureUnsupported at capture time.
            #
            # It stays INSIDE the cold-build window on purpose: this barrier is
            # the join that ENDS the cold-build phase, so a peer rank still
            # sitting in nvcc during its own warmups must be waited out under
            # the relaxed deadline. Closing the window first would give this
            # collective the steady-state deadline and turn a slow peer's cold
            # build into a false wedge -- the exact failure the window exists
            # to prevent. The recorded pass still runs outside the window.
            if device_module is not None:
                device_module.synchronize()
            if tp_group is not None and not skip_barrier:
                tp_group.barrier()
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
