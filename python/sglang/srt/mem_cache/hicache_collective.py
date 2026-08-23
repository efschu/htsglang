"""Bounded waits for the HiCache cross-rank control collectives (#259, #630).

The control collectives of the hierarchical caches run on the gloo ``cpu_group``s
built by ``GroupCoordinator.__init__``, whose ``gloo_timeout`` defaults to
``timedelta(seconds=120 * 60)`` -- two hours (``parallel_state.py:619``). Nothing
on the HiCache path shortens it. A peer that dies without closing its socket, or
a PP rank that never reaches the matching send, therefore parks the survivor for
two hours and then surfaces a generic ``RuntimeError`` from inside gloo that
names no call site.

Every helper here turns that into a named, bounded failure: poll the ``Work``
against a deadline and raise ``HiCacheCollectiveTimeoutError`` carrying the
collective's label, the rank that gave up, and how long it actually waited.

The healthy path is unchanged. ``COLLECTIVE_POLL_SPINS`` bare loop iterations run
before the first ``time.sleep``, which covers the latency of a CPU collective
between local ranks by a wide margin, so a completed ``isend`` or ``all_reduce``
costs a few hundred predicted-branch iterations and no syscall. The sleep path is
only ever reached once something is already wrong.

This module holds the mechanism so that ``unified_radix_cache`` and
``hiradix_cache`` share ONE implementation rather than two drifting copies --
the drift is exactly what #630 was: ``hiradix_cache`` never received the #259
work and kept raw blocking calls on the same collectives.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import torch


class HiCacheCollectiveError(RuntimeError):
    """A HiCache control collective cannot be issued rank-uniformly."""


class HiCacheCollectiveDesyncError(HiCacheCollectiveError):
    """A HiCache control collective reduced a payload it did not post.

    Raised when a self-identifying control vector comes back carrying a
    foreign tag, i.e. the ranks were not all inside the same collective.
    gloo only aborts the process when the mismatched buffers differ in size
    (``op.preamble.length <= op.nbytes``); equal-sized traffic from another
    site is accepted silently and would corrupt the reduced values instead.
    This turns that second case into a named error on every rank.
    """


class HiCacheCollectiveTimeoutError(HiCacheCollectiveError):
    """A HiCache control collective did not complete within its bound.

    Raised on the surviving rank when a peer is dead or wedged. The alternative
    is the failure this exists to prevent: the gloo ``cpu_group`` these control
    collectives run on carries a two-hour default timeout, so without a bound
    the survivor parks in ``all_reduce`` for hours after its peer dies of OOM,
    with nothing in the log naming the reason.
    """


# Poll schedule for bounded_wait. The spin window covers the latency of a
# healthy CPU collective between local ranks, so the sleep path is only ever
# reached once something is actually wrong.
COLLECTIVE_POLL_SPINS = 512
COLLECTIVE_POLL_MIN_S = 0.0005
COLLECTIVE_POLL_MAX_S = 0.05


def collective_rank_desc(holder: Any) -> str:
    """Describe the rank that is about to wait, for timeout messages.

    Defensive throughout: this runs on the failure path of a process that is
    already in trouble, and a missing attribute or an uninitialized default
    process group must never mask the timeout it is trying to report.
    """
    parts = []
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts.append(f"global_rank={torch.distributed.get_rank()}")
    except Exception:  # pragma: no cover - diagnostics must not raise
        pass
    pp_rank = getattr(holder, "pp_rank", None)
    pp_size = getattr(holder, "pp_size", None)
    if pp_rank is not None:
        parts.append(f"pp_rank={pp_rank}" + (f"/{pp_size}" if pp_size else ""))
    tp_world_size = getattr(holder, "tp_world_size", None)
    if tp_world_size is not None:
        parts.append(f"tp_world_size={tp_world_size}")
    return " ".join(parts) or "rank=unknown"


def _timeout_message(label: str, timeout_s: float, waited: float, rank_desc: str) -> str:
    return (
        f"HiCache control collective '{label}' did not complete "
        f"within {timeout_s:g}s (waited {waited:.1f}s) on "
        f"[{rank_desc or collective_rank_desc(None)}]. A peer rank is "
        "dead or wedged; this rank aborts instead of blocking in the "
        "collective until the process-group timeout (2h) expires. "
        "Adjust the bound with SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S."
    )


#: ``parallel_state.py:619`` builds the groups these collectives run on with
#: ``gloo_timeout=timedelta(seconds=120 * 60)``. Used only as the fallback when
#: the live value cannot be read off the group.
DEFAULT_PG_TIMEOUT_S = 120.0 * 60.0

#: Bounds already checked in this process, so the guard costs one set lookup
#: per wait rather than a torch introspection.
_DISCRIMINABLE_BOUNDS: set = set()


def process_group_timeout_s(group=None) -> Optional[float]:
    """The process group's OWN timeout in seconds, or ``None`` if unreadable.

    Read off the gloo backend's options, which is the only place the value
    actually configured for this group appears;
    ``distributed_c10d._get_default_timeout`` returns the LIBRARY default and
    was measured lying about a group built with an explicit timeout (reported
    30:00 for a group constructed with 20:34). Both are private APIs, so this
    never raises: an unreadable value is answered with ``None`` and the caller
    falls back to the documented constant.
    """
    try:
        import torch.distributed as dist

        grp = group if group is not None else dist.group.WORLD
        backend = grp._get_backend(torch.device("cpu"))
        return float(backend.options._timeout.total_seconds())
    except Exception:  # noqa: BLE001 - introspection must never break a wait
        return None


def assert_bound_is_discriminable(
    timeout_s: float, pg_timeout_s: Optional[float] = None
) -> None:
    """The bound must stay BELOW the process group's own timeout (#825).

    WHY THIS IS A NAMED CHECK AND NOT A TIME RATIO. ``bounded_wait`` tells a
    dead peer from an expired deadline by CONTROL FLOW: expiry returns from
    ``ParkedWait.join`` as ``False``, transport failure re-raises. That
    separation holds on exactly one condition -- that the parked, UNBOUNDED
    ``work.wait()`` cannot itself expire while we are waiting on it. It cannot,
    with room to spare, because the group's timeout is 7200 s against a default
    bound of 600 s (``SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S``, ``environ.py``), a
    12x margin.

    But that is a CONFIGURATION fact, not a law. Raise the bound above the
    group's timeout and the parked wait starts raising genuine expiries, which
    would then be reported as transport failures -- #734's error, inverted.
    The predecessor of this check was `waited < timeout_s * 0.95`, which tried
    to cover that case with arithmetic on every single failure and paid for it
    with a 5 %-wide band of mislabelled peer deaths. The condition is a
    property of the configuration, so it is checked once, by name, where the
    configuration is wrong -- not re-derived from a stopwatch at every raise.

    Raises ``HiCacheCollectiveError`` naming both numbers.
    """
    limit = DEFAULT_PG_TIMEOUT_S if pg_timeout_s is None else float(pg_timeout_s)
    if timeout_s < limit:
        return
    raise HiCacheCollectiveError(
        f"HiCache collective bound of {timeout_s:g}s is not below the process "
        f"group's own timeout of {limit:g}s, so an expiry inside the parked "
        f"wait would be reported as a dead peer. Lower "
        f"SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S below {limit:g}s, or raise the "
        f"group's gloo_timeout (parallel_state.py, gloo_timeout=)."
    )


def _check_bound_once(timeout_s: float) -> None:
    """Run :func:`assert_bound_is_discriminable` once per distinct bound."""
    if timeout_s in _DISCRIMINABLE_BOUNDS:
        return
    assert_bound_is_discriminable(timeout_s, process_group_timeout_s())
    _DISCRIMINABLE_BOUNDS.add(timeout_s)


def bounded_wait(
    work: Optional[torch.distributed.Work],
    label: str,
    timeout_s: float,
    rank_desc: str = "",
) -> None:
    """Wait for ``work`` with a deadline, or raise a named error.

    ``timeout_s <= 0`` restores the raw blocking ``work.wait()``, which is the
    documented escape hatch of SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.

    ``work`` may be ``None``: single-rank process groups make torch.distributed
    p2p helpers return ``None`` rather than a ``Work``, and that is a completed
    no-op, not something to wait on.
    """
    if work is None:
        return
    if not timeout_s or timeout_s <= 0:
        work.wait()
        return
    # #825: the control-flow discriminator below is only sound while the bound
    # sits under the group's own timeout. Checked once per distinct bound.
    _check_bound_once(timeout_s)

    # THE BOUND MUST NOT REPLACE THE PROGRESS. This loop used to be
    #
    #     while not work.is_completed():  ... sleep ...
    #     work.wait()
    #
    # which polls passively and only calls ``wait()`` once the poll has already
    # succeeded. For gloo that never happens: ``is_completed()`` REPORTS state,
    # ``wait()`` DRIVES the transfer. With both peers polling, neither side
    # advances the exchange and every rank sits until its own deadline -- so
    # the #630 bound, written to stop a hang, produced a livelock instead.
    #
    # PROVEN by a red/green pair on a 3-process gloo harness driving this exact
    # function (evidence-665-f1/repro_pp_sync_630.py), no CUDA and no serving:
    #
    #   timeout_s = 8.0  (this poll)          -> all three ranks time out with
    #                                            pp_sync/isend[0]->pp1,
    #                                            recv<-pp0, recv<-pp1
    #   timeout_s = 0.0  (the wait() escape)  -> rendezvous, rank 1 receives
    #                                            rank 0's values
    #
    # and matching the live wedge of 2026-08-17 label for label. Raw gloo
    # isend/irecv with the same tag and group rendezvous fine when waited, which
    # is what isolated the defect to this wrapper rather than to gloo.
    #
    # So the wait is BLOCKING again, with the deadline handed to the wait
    # itself. ``Work.wait`` takes a timeout and raises on expiry, which gives
    # both properties at once: the transfer progresses, and a dead peer still
    # cannot park this rank for the group's two-hour timeout.
    # #829: THE BOUND MUST NOT DESTROY THE PAIR EITHER.
    #
    # The line above used to be, verbatim:
    #
    #     completed = work.wait(timeout=datetime.timedelta(seconds=timeout_s))
    #
    # It was the ONLY timed ``Work.wait`` on a gloo ``Work`` anywhere in
    # ``sglang/srt`` (every other ``.wait(timeout=`` in the tree is a
    # ``threading.Event`` or a ``Popen``), and #630 introduced it. It fixed the
    # livelock and bought a second defect at group scope, measured hermetically
    # on 2-process gloo by #824 W4 and written up on :class:`ParkedWait`:
    # an expired ``wait(timeout=...)`` CLOSES THE GLOO PAIR. The waiter then
    # gets "Application timeout caused pair closure" from every later call, and
    # the PEER gets "Connection closed by peer" from its next send.
    #
    # THE CONTRACT THAT BROKE IS NOT THIS FUNCTION'S. ``bounded_recv``'s
    # docstring calls the raise terminal -- "the process is on its way down" --
    # and that is a RANK-LOCAL promise. It is not containable: the pair is
    # shared, so this rank's deadline kills collectives on peers that are
    # healthy and are not waiting on anything of ours. 34 of 262 boot logs
    # carry the peer's half of it, always at a BARE ``work.wait()`` in the PP
    # tensor-dict transport (``parallel_state.recv_object`` :2130-2132,
    # ``scheduler_pp_mixin._pp_commit_comm_work``), and those victims run on
    # the SAME gloo context this function does: ``kv_cache_builder.py:230``
    # passes ``pp_cache_group=pp_group.cpu_group``, which is the very group
    # ``recv_object`` posts its ``irecv`` on. Same ranks, same context, same
    # TCP pairs.
    #
    # So the deadline moves off the ``Work`` and onto the CALLER, which is
    # exactly what :class:`ParkedWait` is for: the unbounded ``wait()`` -- the
    # one call with positive evidence that it DRIVES the transfer, which is
    # #630's whole finding -- runs on its own thread, and we join that thread
    # with the deadline. Expiry returns control without touching the ``Work``,
    # so the pair is never poisoned.
    #
    # TWO OF THIS FUNCTION'S CONTRACTS ARE UNCHANGED: the terminal raise and the
    # ``_timeout_message`` wording. The third -- #734's numeric discriminator --
    # is RETIRED here, because this redesign is what made it both unnecessary
    # and harmful. See below.
    started = time.monotonic()
    parked = ParkedWait(work, label)
    try:
        completed = parked.join(timeout_s)
    except RuntimeError as exc:
        waited = time.monotonic() - started
        # #734 IS SATISFIED BY CONTROL FLOW HERE, NOT BY ARITHMETIC (#825).
        #
        # #734's finding stands and is not in question: a dead peer is not a
        # slow peer, and reporting transport failure as timeout produced a
        # self-contradicting line on 2026-08-17 --
        #
        #   'pp_sync/isend[2]->pp2' did not complete within 600s (waited 34.3s)
        #
        # -- 34.3 s against a 600 s bound, whose real cause was one frame
        # underneath: `gloo ... Connection closed by peer`.
        #
        # #734 separated the two causes with `waited < timeout_s * 0.95`,
        # because at that time BOTH arrived here: `Work.wait(timeout=...)`
        # raised for expiry AND for transport failure, so the only thing
        # telling them apart was how long the wait had lasted. That was correct
        # for that shape.
        #
        # THIS SHAPE IS DIFFERENT, AND THE COMPARISON BECAME A DEFECT. Since
        # #829 the deadline lives on ``ParkedWait.join``, and the two causes no
        # longer meet:
        #
        #   expiry            -> join() returns False, wait still parked
        #                        -> the `if not completed:` raise below
        #   transport failure -> the parked wait raised, join() re-raises it
        #                        -> HERE
        #
        # So every RuntimeError arriving in this block is ALREADY a transport
        # failure, whatever ``waited`` says. The comparison could no longer
        # improve a verdict -- it could only downgrade a correct "dead peer"
        # into a wrong "timeout", for any death landing in the last 5 % of the
        # bound. That is 30 seconds wide at the 600 s default.
        #
        # MEASURED 2026-08-23, hermetic 2-process gloo, CVD="", both causes
        # generated for real and both raise sites instrumented to name which
        # branch fired:
        #
        #   real expiry, 4/4 trials      -> `not completed` path.  NEVER here.
        #   real SIGKILL peer, 14 trials -> here, detection latency 15-21 ms.
        #   peer death at 0.98/0.97/0.96 of the bound -> reached this block and
        #       was relabelled HiCacheCollectiveTimeoutError by the comparison.
        #       Three specimens, the exact wrong-instrument reading #734 exists
        #       to prevent, still live in the band the comparison created.
        #
        # The replacement is not a string match either -- #734's objection to
        # those ("backend messages change between versions") is upheld. It is
        # control flow, which is stronger than both arithmetic and strings:
        # nothing that is not a transport failure can reach this line.
        #
        # ``waited`` stays in the message as evidence. It is no longer the
        # decision.
        raise HiCacheCollectiveError(
            f"HiCache control collective '{label}' FAILED after "
            f"{waited:.1f}s, inside its {timeout_s:g}s bound, on "
            f"[{rank_desc or collective_rank_desc(None)}]. This is NOT a "
            f"timeout -- the transport ended the wait early, which usually "
            f"means a peer process died or its connection dropped. Look for "
            f"a dead rank, not a slow one. Underlying error: {exc}"
        ) from exc
    # Expiry. ``ParkedWait.join`` returns False with the wait STILL PARKED and
    # the pair intact; the raise below keeps this function's terminal contract,
    # so callers are unaffected. (Before #829 this branch also caught backends
    # that reported expiry by returning False rather than raising.)
    if not completed:
        raise HiCacheCollectiveTimeoutError(
            _timeout_message(label, timeout_s, time.monotonic() - started, rank_desc)
        )


def bounded_recv(
    tensor: torch.Tensor,
    *,
    group,
    group_src: int,
    tag: int,
    label: str,
    timeout_s: float,
    rank_desc: str = "",
) -> None:
    """Blocking receive with a deadline.

    ``torch.distributed.recv`` has no async form and no timeout, so it cannot be
    bounded in place. ``irecv`` posts the identical operation and returns a
    ``Work`` that ``bounded_wait`` can poll; its signature carries the same
    ``group_src``/``tag``/``group`` arguments, and ``recv`` is itself implemented
    as a posted receive that is immediately waited on. Message matching is by
    (source, tag, group) in posting order, and this function posts exactly one
    receive and does not return until it has completed, so the substitution
    preserves the ordering semantics of the ``recv`` it replaces.

    On timeout the receive is still outstanding and ``tensor`` remains
    registered with the backend as its destination buffer. That is only
    acceptable because the raised error is fatal: the scheduler does not catch
    ``HiCacheCollectiveError`` and the process is on its way down. Do not
    "recover" from this exception and reuse ``tensor``.
    """
    work = torch.distributed.irecv(
        tensor,
        group_src=group_src,
        group=group,
        tag=tag,
    )
    bounded_wait(work, label, timeout_s, rank_desc)


class ParkedWait:
    """A RESUMABLE bound on one ``Work``: the caller gets a deadline, the
    transfer keeps its unbounded ``wait()``, and nothing is destroyed.

    WHY THIS EXISTS NEXT TO ``bounded_wait`` (#824 W4)
    --------------------------------------------------
    ``bounded_wait`` above is terminal by design and says so:
    ``bounded_recv``'s docstring ends "Do not 'recover' from this exception
    and reuse ``tensor``". That is the right contract for a HiCache control
    collective, where a missed deadline means the process is on its way
    down.

    It is the WRONG contract for the PP request chain. That stream is a
    two-step size-then-payload protocol whose receiver must stay framed
    across a stall: a rank that gives up mid-protocol and later re-posts
    would read a payload AS a size and misframe every later message
    (``managers/pp_chain_receiver.py``, module docstring). It has to be
    able to wait again on the SAME posted receive.

    ``bounded_wait`` cannot provide that, and the reason is measured, not
    argued (2026-08-23, hermetic 2-process gloo, CVD=""):

      * ``Work.wait(timeout=...)`` does fire on time -- and closes the gloo
        pair while doing it. The waiter then gets "Application timeout
        caused pair closure" from every later call, and the PEER gets
        "Connection closed by peer" from its next send. One expired wait
        takes the whole group down, on both sides.
      * ``Work.is_completed()`` is no way out either: it never reports True
        on this build even seconds after the payload has landed (the
        "corpse F" measurement pinned in
        ``test_pp_chain_receiver.test_measured_gloo_does_not_progress_a_posted_irecv_by_polling``).
        A poll loop absorbs nothing and bounds nothing.

    So the bound cannot live in the wait. It lives in the CALLER: the
    unbounded ``wait()`` -- the one call with positive evidence that it
    drives the transfer -- is parked on a dedicated thread, and the caller
    joins that thread with a deadline. Expiry returns control without
    touching the ``Work``, so:

      * the pair is never poisoned (the wait was never interrupted),
      * the posted receive stays posted and stays framed,
      * a peer that sends LATE still completes this very wait, and the
        payload arrives intact.

    All three are measured green in the same hermetic harness, and the last
    one is asserted on the wire by
    ``test_pp_ring_abort_recovery.test_aborted_flip_does_not_wedge_the_pp_ring``.

    ONE THREAD PER POSTED WORK, and the object owning the ``Work`` owns the
    ``ParkedWait`` too -- two threads waiting on one ``Work``, or a second
    receive posted on a stream whose first is still parked, is the
    misframing this class exists to avoid.
    """

    __slots__ = ("_work", "_label", "_done", "_error", "_thread", "_started_at")

    def __init__(self, work: Optional[torch.distributed.Work], label: str):
        self._work = work
        self._label = label
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        self._started_at = time.monotonic()
        if work is None:
            # Single-rank groups return None rather than a Work; that is a
            # completed no-op, exactly as in bounded_wait.
            self._thread = None
            self._done.set()
            return
        self._thread = threading.Thread(
            target=self._park,
            name=f"parked-wait-{label}",
            daemon=True,
        )
        self._thread.start()

    def _park(self) -> None:
        try:
            self._work.wait()
        except BaseException as exc:  # noqa: BLE001 - re-raised in join()
            self._error = exc
        finally:
            self._done.set()

    @property
    def label(self) -> str:
        return self._label

    @property
    def waited_s(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def completed(self) -> bool:
        return self._done.is_set()

    def join(self, timeout_s: Optional[float]) -> bool:
        """Wait up to ``timeout_s`` for the parked wait to finish.

        Returns True when it completed (the transfer is done and the
        buffer is filled), False when the deadline expired with the
        receive STILL POSTED -- call again to keep waiting on the same one.
        A transport error raised inside the parked wait is re-raised here,
        on the caller's thread, so it is not swallowed by the waiter.

        ``timeout_s`` of None or <= 0 restores the raw blocking wait.
        """
        if timeout_s is None or timeout_s <= 0:
            self._done.wait()
        else:
            self._done.wait(timeout=timeout_s)
        if not self._done.is_set():
            return False
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        return True
