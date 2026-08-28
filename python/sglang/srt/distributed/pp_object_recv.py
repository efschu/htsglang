# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#980: a RESUMABLE, observable bound on ``GroupCoordinator.recv_object``.

WHY THIS EXISTS
---------------
``recv_object`` is a TWO-STEP protocol -- an ``irecv`` of a size, then an
``irecv`` of exactly that many bytes -- and until #980 both steps ended in a
naked ``work.wait()`` (``parallel_state.py:2133`` and ``:2145``). A rank blocked
there is SILENT: nothing on that path recorded that it was blocked, in which
half of the frame, on which peer, or for how long, so the stall is
indistinguishable from a healthy idle wait from the outside.

THE EVIDENCE, AT ITS ACTUAL STRENGTH. The nearest VERBATIM on-disk stack for
this site is ``evidence-665-f1/SPECIMEN-2026-08-27T1641Z-944-LIVE-WEDGE-ROOTED
.txt``: PP1 blocked, stack identical over 20 s, ``recv_object``
(``parallel_state.py:2133``) under ``_pp_recv_typed_dict`` under
``_pp_drain_voided_proxy``, while PP0 kept computing. Boot 7 of the 0828 window
(pin 8be86f55fe) is filed in ``REGISTER_OPEN_876.txt`` as the same shape on PP1
and is what escalated this ticket, but that boot's OWN py-spy could not unwind
PP1 (``UNW_EBADREG``) and the quoted stack was never persisted -- so boot 7 is
register prose here, and the #944 specimen is the measurement. Both name the
same two lines, which is what this module is placed against.

That unattributable dump is itself the argument for this module: the boot that
mattered most produced NO frames, so the only durable record of a stall at this
site has to come from the site itself.

WHY NOT A PLAIN TIMEOUT
-----------------------
A terminal timeout in the MIDDLE of this protocol misframes the stream. Once
the size header has been received, the payload is already on the wire and MUST
be taken off it; a receiver that gives up mid-frame and later re-posts reads a
payload AS a size, and every later message on that stream is garbage. That is
the same law that makes ``managers/pp_chain_receiver.py`` a state machine
rather than a timeout; this module is its sibling for the generic object
channel, which that class deliberately does not own.

WHY THE BOUND SITS ON A JOIN AND NOT ON THE WAIT (#630, #829)
-------------------------------------------------------------
Measured, not argued (2026-08-23, hermetic 2-process gloo, CVD=""), and pinned
in ``mem_cache/hicache_collective.py``'s ``ParkedWait`` docstring:

  * ``Work.wait(timeout=...)`` does fire on time -- and CLOSES THE GLOO PAIR
    while doing it. The waiter then gets "Application timeout caused pair
    closure" from every later call and the PEER gets "Connection closed by
    peer" from its next send. One expired wait takes the whole group down, on
    both sides (#829).
  * ``Work.is_completed()`` never reports True on this build even seconds after
    the payload has landed, so a poll loop drives nothing and bounds nothing
    (#630, corpse F).

So the bound cannot live in the wait. ``ParkedWait`` runs the unbounded
``wait()`` -- the one call with positive evidence that it DRIVES the transfer --
on its own thread, and the deadline goes on the join. An expired step returns
control without touching the ``Work``: the receive stays POSTED, the pair stays
INTACT, the frame stays RESUMABLE, and a peer that sends late still completes
that very wait with the payload intact.

WHAT THE CALLER DECIDES
-----------------------
An expired step is an OBSERVATION, not a verdict. ``ObjectRecvFrame.advance``
returns False and leaves the frame resumable; ``receive`` then either keeps
waiting -- the default, emitting a #650-style statement that names the silent
hop and which half of the frame is outstanding -- or, once ``abort_after_s`` is
armed and exceeded, raises ``ObjectRecvStalled`` with a named site code.
Nothing is destroyed on either branch, so an abort that is caught and retried
resumes the SAME frame rather than posting a second receive on the stream.

DEFAULTS ARE BEHAVIOUR-PRESERVING BY INTENT
-------------------------------------------
``abort_after_s`` defaults to 0 == never abort. A rank that would have blocked
for ever still blocks for ever; what changed is that it now SAYS SO once per
step, with its frame state. Arming an abort by default would fire on any
channel whose upstream is legitimately slow, and the boot-first cycle wants the
wedge named before it wants it killed.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.hicache_collective import ParkedWait

logger = logging.getLogger(__name__)

#: Every state-transition line this module emits starts with this token, so a
#: boot log can be swept for the whole receiver lifecycle with one grep.
LOG_PREFIX = "PP-RECV-OBJ"

#: How long ONE poll step may take before the frame surfaces to the caller.
#: This is an OBSERVATION CADENCE, not a deadline on the transfer: expiry never
#: touches the posted receive, it only hands control back so the stall can be
#: named. Shorter values cost one Event wakeup per step and buy log resolution.
ENV_STEP_BUDGET = "SGLANG_PP_RECV_OBJECT_STEP_S"
DEFAULT_STEP_BUDGET_S = 30.0

#: Total wall-clock, measured from the moment the frame armed, after which
#: ``receive`` stops waiting and raises ``ObjectRecvStalled``.
#:
#: DEFAULT 0 == NEVER ABORT, which is byte-for-byte the pre-#980 outcome. This
#: deliberately does NOT follow ``ENV_RING_COMMIT_BUDGET``'s convention of a
#: positive default: the ring commit is provably mid-pass and a stall there is
#: provably a wedge, whereas ``recv_object`` also carries channels whose
#: upstream may simply have nothing to say yet. Arm it where something else
#: establishes that a stall here is a wedge (#824 W4 makes the same call for
#: ``SGLANG_PP_CHAIN_RECV_STALL_S`` and for the same reason).
ENV_ABORT_AFTER = "SGLANG_PP_RECV_OBJECT_ABORT_S"
DEFAULT_ABORT_AFTER_S = 0.0

#: State machine positions. ``_IDLE`` is the ONLY state in which this stream
#: may be abandoned safely: in the other two a message is half-received and
#: dropping it misframes every later message on the stream.
_IDLE = "idle"
_AWAITING_SIZE = "awaiting_size"
_AWAITING_DATA = "awaiting_data"
_COMPLETE = "complete"

#: Lifecycle counters, so a probe can tell NO OBSERVATION from ZERO EXPIRIES.
#: ``armed == 0`` means this module never ran on this boot (old binary, or the
#: path was never taken); ``armed > 0`` with ``step_expired == 0`` means it ran
#: and nothing ever stalled. The two read identically in a log that only
#: reports problems, which is exactly how boot 7 stayed unattributable.
RECV_OBJECT_STATS = {
    "armed": 0,
    "step_expired": 0,
    "resumed": 0,
    "completed": 0,
    "aborted": 0,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s ignoring unparsable %s=%r; using %g",
            LOG_PREFIX,
            name,
            raw,
            default,
        )
        return default


def recv_object_step_budget_s() -> float:
    return _env_float(ENV_STEP_BUDGET, DEFAULT_STEP_BUDGET_S)


def recv_object_abort_after_s() -> float:
    return _env_float(ENV_ABORT_AFTER, DEFAULT_ABORT_AFTER_S)


class ObjectRecvStalled(RuntimeError):
    """#980: a bounded ``recv_object`` gave up by a NAMED branch.

    THE STREAM IS STILL FRAMED when this is raised. The posted receive was
    never cancelled and its parked wait is still running, so calling
    ``recv_object`` again on the same ``(src, tag)`` resumes the SAME frame and
    a late message still arrives intact. Handle this by servicing whatever the
    upstream is actually waiting on and then coming back -- do NOT rebuild the
    coordinator, and do not post another receive on this stream.
    """


class ObjectRecvFrame:
    """One resumable size-then-payload receive on one ``(src, tag)`` stream.

    Mirrors the receive half of ``GroupCoordinator.recv_object`` exactly --
    same ``irecv`` calls, same dtypes, same ``pickle.loads`` -- and adds only
    the explicit protocol state that makes an expired step recoverable.

    ONE FRAME PER STREAM, and the frame owns both the ``Work`` and its
    ``ParkedWait``. Two frames on one stream, or a second receive posted while
    the first is still parked, is the misframing this class exists to avoid.
    """

    __slots__ = (
        "_group",
        "_src_global",
        "_tag",
        "_site",
        "_rank_desc",
        "_state",
        "_work",
        "_parked",
        "_size_tensor",
        "_object_tensor",
        "_expected_bytes",
        "_armed_at",
        "_expiries",
    )

    def __init__(
        self,
        group: Any,
        src_global: int,
        tag: int,
        site: str,
        rank_desc: str,
    ) -> None:
        self._group = group
        self._src_global = src_global
        self._tag = tag
        self._site = site
        self._rank_desc = rank_desc
        self._state = _IDLE
        self._work: Optional[Any] = None
        self._parked: Optional[ParkedWait] = None
        self._size_tensor: Optional[torch.Tensor] = None
        self._object_tensor: Optional[torch.Tensor] = None
        self._expected_bytes = -1
        self._armed_at = 0.0
        self._expiries = 0

    # -- observation ----------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def expiries(self) -> int:
        return self._expiries

    def _log(self, event: str, level: int, extra: str = "") -> None:
        """One line per state transition: event, site, rank, frame state.

        Every line carries all four so a single grep of ``LOG_PREFIX`` in a
        boot log reconstructs the whole lifecycle of every stream without
        cross-referencing anything else.
        """
        logger.log(
            level,
            "%s %s site=%s %s frame=%s src_global=%s waited=%.1fs expiries=%d%s",
            LOG_PREFIX,
            event,
            self._site,
            self._rank_desc,
            self._state,
            self._src_global,
            time.monotonic() - self._armed_at if self._armed_at else 0.0,
            self._expiries,
            f" {extra}" if extra else "",
        )

    def peer_statement(self) -> str:
        """#650-style: name the silent hop and which half of the frame it is.

        Deliberately speaks only what the TRANSPORT layer honestly knows. The
        per-peer phase-flip counters live on the Scheduler, which this layer
        cannot see, so this names the hop and the frame half and points at the
        peer to py-spy rather than inventing a peer-side claim.

        Defensive throughout, for the same reason ``collective_rank_desc`` is:
        this runs on the failure path of a process that is already in trouble.
        """
        try:
            if self._state == _AWAITING_SIZE:
                half = (
                    "no size header has arrived, so the upstream has not "
                    "committed its matching send at all -- the silent hop is "
                    f"global rank {self._src_global} BEFORE its send"
                )
            elif self._state == _AWAITING_DATA:
                half = (
                    f"the size header announced {self._expected_bytes} bytes and "
                    f"the payload has not landed -- global rank "
                    f"{self._src_global} DID commit its send, so the stall is on "
                    "the wire or between the two sends, not before them"
                )
            else:
                half = f"frame is {self._state}, nothing is outstanding"
            return (
                f"{half}; the receive is STILL POSTED and the gloo pair is "
                f"intact (the deadline was on the join, not on the Work), so a "
                f"late send still completes it. Read the peer side from a "
                f"py-spy of global rank {self._src_global}."
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
            return f"peer statement unavailable ({type(exc).__name__}: {exc})"

    # -- state machine --------------------------------------------------

    def _arm(self) -> None:
        self._size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        # We have to use irecv here to make it work for both isend and send.
        self._work = dist.irecv(
            self._size_tensor,
            src=self._src_global,
            group=self._group,
            tag=self._tag,
        )
        self._parked = ParkedWait(self._work, f"{self._site}/size")
        self._state = _AWAITING_SIZE
        self._armed_at = time.monotonic()
        self._expiries = 0
        RECV_OBJECT_STATS["armed"] += 1
        self._log("armed", logging.DEBUG)

    def _post_payload(self) -> None:
        assert self._size_tensor is not None
        self._expected_bytes = int(self._size_tensor.item())
        self._object_tensor = torch.empty(
            self._expected_bytes, dtype=torch.uint8, device="cpu"
        )
        self._work = dist.irecv(
            self._object_tensor,
            src=self._src_global,
            group=self._group,
            tag=self._tag,
        )
        self._parked = ParkedWait(self._work, f"{self._site}/payload")
        self._state = _AWAITING_DATA

    def _join(self, step_budget_s: float) -> bool:
        """Join the parked wait for one bounded step.

        Returns True when the step completed, False when the budget expired
        with the receive STILL POSTED. A transport error raised inside the
        parked wait is re-raised here rather than swallowed -- it is NOT
        converted into ``ObjectRecvStalled``, so a dead peer stays
        distinguishable from a slow one (#734).
        """
        assert self._parked is not None
        if self._expiries:
            RECV_OBJECT_STATS["resumed"] += 1
            self._log("resumed", logging.WARNING)
        if self._parked.join(step_budget_s):
            return True
        self._expiries += 1
        RECV_OBJECT_STATS["step_expired"] += 1
        self._log("step-expired", logging.WARNING, self.peer_statement())
        return False

    def advance(self, step_budget_s: float) -> bool:
        """Drive the frame as far as it goes within bounded steps.

        Returns True when a whole object is on the floor and ``take`` may be
        called, False when a step expired -- in which case the frame is
        RESUMABLE and calling ``advance`` again continues the same receive.
        """
        if self._state == _COMPLETE:
            return True
        if self._state == _IDLE:
            self._arm()
        if self._state == _AWAITING_SIZE:
            if not self._join(step_budget_s):
                return False
            self._post_payload()
        if self._state == _AWAITING_DATA:
            if not self._join(step_budget_s):
                return False
            self._state = _COMPLETE
        return True

    def take(self) -> Any:
        """Unpickle the completed object and reset the frame for the next one.

        Only legal in ``_COMPLETE``; the reset returns the frame to ``_IDLE``,
        which is the only state in which the stream may be left alone.
        """
        assert self._state == _COMPLETE, f"take() in state {self._state}"
        assert self._object_tensor is not None
        obj = pickle.loads(self._object_tensor.numpy())
        RECV_OBJECT_STATS["completed"] += 1
        # INFO, not DEBUG, only when this receive had to be resumed at least
        # once: a healthy stream stays silent, a recovered stall says so.
        self._log(
            "completed",
            logging.INFO if self._expiries else logging.DEBUG,
            f"bytes={self._expected_bytes}",
        )
        self._state = _IDLE
        self._work = None
        self._parked = None
        self._size_tensor = None
        self._object_tensor = None
        self._expected_bytes = -1
        self._armed_at = 0.0
        self._expiries = 0
        return obj

    def receive(self, step_budget_s: float, abort_after_s: float) -> Any:
        """Blocking receive with bounded, observable, resumable steps.

        The default (``abort_after_s <= 0``) keeps waiting for ever, exactly as
        the pre-#980 naked ``work.wait()`` pair did -- but names the stall once
        per expired step instead of going silent.
        """
        if step_budget_s <= 0:
            # Documented escape hatch: byte-for-byte the pre-#980 behaviour,
            # including the unbounded park and the silence.
            while not self.advance(0.0):  # pragma: no cover - escape hatch
                pass
            return self.take()
        while not self.advance(step_budget_s):
            if (
                abort_after_s > 0
                and (time.monotonic() - self._armed_at) > abort_after_s
            ):
                RECV_OBJECT_STATS["aborted"] += 1
                self._log("aborted", logging.ERROR, self.peer_statement())
                message = (
                    f"#980 RECV-OBJECT STALL at {self._site}: this rank waited "
                    f"{time.monotonic() - self._armed_at:.1f}s in frame state "
                    f"{self._state} for global rank {self._src_global} "
                    f"({self._rank_desc}). This is the boot-7 wedge shape -- "
                    f"parallel_state.recv_object blocked with nothing on the "
                    f"path recording it. Peer statement: {self.peer_statement()} "
                    f"The frame is RESUMABLE: call recv_object on this "
                    f"(src, tag) again to continue the SAME receive. Raise or "
                    f"disable the bound with {ENV_ABORT_AFTER} (<= 0 restores "
                    f"the unbounded wait). See #980."
                )
                raise ObjectRecvStalled(message)
        return self.take()


def get_or_create_frame(
    frames: dict,
    key: Tuple[int, int],
    group: Any,
    src_global: int,
    tag: int,
    site: str,
    rank_desc: str,
) -> ObjectRecvFrame:
    """One frame per ``(src, tag)`` stream, created once and kept.

    Kept across calls ON PURPOSE: that is what makes an expired or aborted step
    resumable. A per-call frame would post a second receive on a stream whose
    first is still parked, which is precisely the misframing this module
    exists to prevent.
    """
    frame = frames.get(key)
    if frame is None:
        frame = frames[key] = ObjectRecvFrame(
            group=group,
            src_global=src_global,
            tag=tag,
            site=site,
            rank_desc=rank_desc,
        )
    return frame


__all__ = [
    "ObjectRecvFrame",
    "ObjectRecvStalled",
    "LOG_PREFIX",
    "ENV_STEP_BUDGET",
    "ENV_ABORT_AFTER",
    "DEFAULT_STEP_BUDGET_S",
    "DEFAULT_ABORT_AFTER_S",
    "RECV_OBJECT_STATS",
    "recv_object_step_budget_s",
    "recv_object_abort_after_s",
    "get_or_create_frame",
]
