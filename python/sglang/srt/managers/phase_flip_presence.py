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
"""#631: epoch-stamped presence flags for the phase flip's entry gate.

THE DESIGN LAW THIS SERVES
--------------------------
    NO RANK MAY BLOCK ON ANY CHANNEL WHILE A PEER MAY BE IN A DIFFERENT
    BLOCKING CHANNEL.

The PP loop has at least two independent blocking channels -- the request
chain and the hidden-states exchange -- with no global order between them,
and the flip's consensus reduction is a third. Every design that let a
rank block anywhere while a peer might be elsewhere has deadlocked. Eight
measured corpses, 2026-08-08:

  A   arm same-pass, async forward       rank0 in reduction, peers in chain recv
  B   arm same-pass, sync forward        rank0 in send, peers in hidden-states recv
  B'  arm same-pass, targeted commit     IDENTICAL to B (boot 13) -- the
                                         "targeted vs blanket" distinction has
                                         no force; both block rank0 on the
                                         chain send
  D   defer arm by one pass              peers never reach the acting pass; the
                                         pass they need begins with a recv that
                                         blocks (boot 12)
  --  message-free local decision        rank0 alone in the reduction, peers
                                         never woken (boot 10)
  --  bounded chain recv                 breaks the 1:1 send/consume contract;
                                         unmatched sends pile up and the SENDERS
                                         block
  --  bounded join (abandon from inside) FATAL: a rank that has entered an
                                         all_reduce owes it; walking away closed
                                         the gloo pairs and aborted every rank
  E   presence announced while still
      owing a chain send             the gate ASSEMBLED and still wedged
                                         (boot 18). The decided fix rests on a
                                         false premise -- see corpse F -- and
                                         ships OFF behind
                                         SGLANG_PP_CHAIN_RECEIVER.
  F   the NON-BLOCKING PUMP          MEASURED DEAD, and it was dead all along.
                                         pp_pump_send_req_work reaps a chain
                                         send on work.is_completed(), and on
                                         this build that predicate NEVER fires
                                         for an isend -- not even after the peer
                                         has fully consumed the message. The
                                         pump has therefore never cleared
                                         send_req_work; the only thing that has
                                         ever reaped a chain send is the
                                         BLOCKING _pp_commit_comm_work.
                                         Arms reached downstream stages via
                                         those stages' OWN blocking recv all
                                         along -- the recv side's wait() is what
                                         progresses the transfer -- never
                                         because an armed rank "pumped the arm
                                         forward while it waited". Every design
                                         note that credited the pump was
                                         reasoning about a no-op. No one-line
                                         repair exists: only wait() progresses a
                                         send here, and blocking is precisely
                                         what the armed path may not do.
                                         Pinned:
                                         test_measured_the_send_side_pump_can_never_reap.

THE TRANSPORT PREMISE, falsified from three directions
------------------------------------------------------
This module already rejected a posted-and-polled ``all_reduce`` because
its progress-without-explicit-wait premise was unverified. That premise is
now MEASURED FALSE for point-to-point too, in both directions: a posted
``irecv`` never completes by polling (so a non-blocking drain absorbs
nothing), and an ``isend`` never completes by polling either (corpse F).
On this build, ONLY ``wait()`` progresses a transfer. Any future design
that needs an armed rank to make progress on a channel without blocking
must supply its own progress engine -- a thread, or a different transport
-- and may not assume the handle advances on its own.

A SECOND MEASUREMENT bears on boot 18's diagnosis and is not yet
explained: an upstream's commit of an UNCONSUMED forward returns in
0.00 s (8 B and 512 KiB). So "the downstream stopped consuming" does not
by itself block an upstream here, and what rank 1 was actually waiting on
at :705 -> :1109 is an OPEN QUESTION. Do not build on the boot-18 story
until a reproduction with all three stacks on disk says what it is.

WHAT BOOT 18 ACTUALLY SHOWED, and what it did not
-------------------------------------------------
OBSERVED (py-spy, tree cf478d1634):
  rank 0  inside the consensus reduction (_reduce -> bounded_collective)
  rank 1  blocked in p2p_work.work.wait() (scheduler_pp_mixin :1109) from
          the ORDINARY top-of-pass commit :705 of the previous pass's
          chain forward
NOT OBSERVED:
  rank 2's stack. It was never recorded, the serving log was truncated by
  the next boot, and no dump survives. Rank 2 is the LAST PP stage, so
  :705 is structurally unreachable for it -- but what it WAS doing is
  unknown and is not reconstructable.

The inference the fix rests on, stated as an inference: rank 1's forward
to rank 2 was not completing, so rank 2 had stopped consuming the chain;
and rank 0 was inside the reduction, so it had observed a full quorum,
which means rank 1's flag was already up while rank 1 still owed that
send. TWO lessons follow, both structural and both independently
sufficient reasons to change the code:

  * a flag must mean "I OWE NO SEND", not merely "I am armed" -- otherwise
    the quorum a peer enters on is a promise the flagged rank has not
    kept;
  * an armed rank must KEEP CONSUMING, because the moment it stops, its
    upstream blocks at a point that PRECEDES the gate, where no gate can
    reach it.

THE GATE IS NOT THE WHOLE OBLIGATION. That is the general form, and it is
why this entry is not merely a bug report: a gate can only make ENTRY
safe. It cannot help a rank that never reaches the entry because it is
blocked at an ordinary channel operation upstream of it. Every blocking
point between arming and the reduction has to be removed on its own
terms.

This module is the other half of the only shape left: make the ENTRY to
the blocking reduction conditional on knowing that every peer is already
at that entry. Then the reduction is safe -- not by argument, but because
no participant is anywhere else.

WHY /dev/shm FLAGS AND NOT A COLLECTIVE
---------------------------------------
The gate cannot itself be a collective: a collective is the thing being
gated. It must be POLLABLE -- readable without blocking and without any
peer's cooperation -- so an armed rank can spin without ever entering a
channel it cannot leave. Group-visible ``/dev/shm`` state is the in-fork
precedent for exactly this on a single-node topology (the #615 build-window
markers), and this deployment is single-node by V1 scope.

A posted-and-polled ``all_reduce`` Work was the alternative and is
REJECTED. Its load-bearing premise -- that such a Work progresses while
merely polled across passes, without explicit progress calls -- is an
unverified transport assumption of exactly the kind that has already
killed designs here: this fork's async SENDS demonstrably do not progress
without an explicit commit (that is corpse A). Betting the gate on the
reduction transport behaving differently, unverified, is the same bet.

EPOCHS, AND WHY FLAGS ARE NEVER CLEARED
---------------------------------------
Flags are monotone within an epoch: a rank sets its own and never unsets
it. Retraction -- a policy that changes its mind, a disarm on timeout --
mints a NEW epoch instead. That is what makes a poll safe against a stale
read: a flag observed for epoch E is a fact about E forever, so no reader
can be fooled by a racing writer, and no writer has to coordinate a clear.
Clearing would reintroduce exactly the ordering problem the flags exist to
remove.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-PRESENCE"

# Single-node by V1 scope, and /dev/shm is the fork's precedent for
# group-visible markers (#615). Overridable for tests.
DEFAULT_PRESENCE_DIR = "/dev/shm/sglang-phase-flip-presence"


class PhaseFlipPresence:
    """Epoch-stamped, monotone, pollable per-rank ready markers.

    One file per (epoch, rank). Presence is the file's EXISTENCE -- never
    its contents -- so a reader needs no parsing and a half-written file
    cannot be mistaken for a different state. The body carries diagnostics
    only, for a human reading a stuck gate.
    """

    def __init__(
        self,
        n_ranks: int,
        rank: int,
        directory: str = DEFAULT_PRESENCE_DIR,
        instance: str = "",
    ):
        self.n_ranks = int(n_ranks)
        self.rank = int(rank)
        # The instance tag keeps two servers on one box (a test boot beside
        # production) from reading each other's flags as their own quorum.
        # THE INSTANCE TAG MUST BE UNIQUE PER BOOT AND IDENTICAL ACROSS
        # RANKS. Both halves matter, and getting the first wrong is a
        # measured failure: it was os.getpid()//100000, which collided
        # across consecutive boots (PIDs 3163115 and 3180590 both give
        # 31). Boot 15 then read boot 14's leftover markers, the gate
        # opened "after 0.00s" on STALE evidence before its peers had
        # armed, and rank 0 entered the reduction alone -- the exact
        # failure the gate exists to prevent, caused by the gate.
        #
        # It must be identical across ranks because the flags are a
        # rendezvous: a per-process value would make every rank look at a
        # different quorum and none would ever assemble. So it comes from
        # the environment, which the boot script sets ONCE and every rank
        # inherits. The fallback is deliberately NOT process-derived --
        # it uses the boot's own start time, shared via the parent.
        self.instance = instance or os.environ.get("SGLANG_PHASE_FLIP_INSTANCE")
        if not self.instance:
            try:
                # Start time of the process group leader: the same value
                # on every rank of this boot, different on the next one.
                with open(f"/proc/{os.getpgrp()}/stat") as fh:
                    self.instance = "pg" + fh.read().split()[21]
            except (OSError, IndexError):
                self.instance = "default"
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        # Drop anything left by an earlier boot before the first poll can
        # read it as quorum. Best effort: a marker that survives is at
        # worst re-swept next time, but one that is READ is a false gate.
        self.sweep_foreign_instances()

    def sweep_foreign_instances(self) -> int:
        """Remove markers from other instances (i.e. earlier boots).

        Called at construction, before anything can poll. A stale marker
        that is merely present is harmless; one that is READ as quorum
        opens the gate on a peer that is not there.
        """
        removed = 0
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        mine = f"{self.instance}."
        for name in names:
            if name.startswith(mine):
                continue
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        if removed:
            logger.warning(
                "%s swept %d marker(s) from earlier boots", LOG_PREFIX, removed
            )
        return removed

    def _path(self, epoch: int, rank: int) -> str:
        return os.path.join(
            self.directory, f"{self.instance}.e{int(epoch)}.r{int(rank)}"
        )

    def announce(self, epoch: int, note: str = "") -> None:
        """Publish THIS rank's readiness for ``epoch``. Idempotent.

        Written via a temp file and an atomic rename, so a reader can never
        observe a partially created marker. Announcing twice is a no-op by
        construction, which matters because the armed poll loop calls this
        every iteration rather than tracking whether it already did.
        """
        path = self._path(epoch, self.rank)
        if os.path.exists(path):
            return
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                fh.write(f"rank={self.rank} epoch={epoch} pid={os.getpid()} {note}\n")
            os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.error("%s could not announce %s: %s", LOG_PREFIX, path, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def observe(self, epoch: int) -> Set[int]:
        """Which ranks have announced for ``epoch``. Never blocks."""
        present: Set[int] = set()
        for r in range(self.n_ranks):
            if os.path.exists(self._path(epoch, r)):
                present.add(r)
        return present

    def missing(self, epoch: int) -> List[int]:
        present = self.observe(epoch)
        return [r for r in range(self.n_ranks) if r not in present]

    def all_present(self, epoch: int) -> bool:
        return len(self.observe(epoch)) == self.n_ranks

    def sweep(self, keep_epoch: Optional[int] = None) -> int:
        """Drop markers from older epochs. Housekeeping only.

        Never removes ``keep_epoch``'s markers, and never removes another
        instance's. Losing an old marker is harmless -- epochs only ever
        move forward, so nothing consults them again -- which is why this
        can run best-effort and ignore errors.
        """
        removed = 0
        prefix = f"{self.instance}.e"
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(prefix):
                continue
            try:
                epoch_part = name.split(".e", 1)[1].split(".r", 1)[0]
                epoch = int(epoch_part)
            except (IndexError, ValueError):
                continue
            if keep_epoch is not None and epoch >= int(keep_epoch):
                continue
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        return removed


__all__ = ["PhaseFlipPresence", "DEFAULT_PRESENCE_DIR", "LOG_PREFIX"]
