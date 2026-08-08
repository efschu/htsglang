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

A SECOND MEASUREMENT looked like it undercut boot 18's diagnosis: an
upstream's commit of an UNCONSUMED forward returns in 0.00 s (8 B and
512 KiB). The reproduction below resolves it -- in that probe the receiver
had POSTED an irecv and merely never polled it to completion. When the
receiver has posted NO matching irecv at all, the sender's wait() does
block. Both facts are needed, and only together do they describe the wire.

THE REPRODUCTION, 2026-08-08 23:12:38Z (boot POLICY=auto, tree 526e53cffc)
-------------------------------------------------------------------------
All three stacks on disk this time, in
/spinning/evidence-631/wedge_20260808T231450Z_INSIDE_REDUCTION:

  rank 0  bounded_collective -> _reduce -> on_round  (in the reduction)
  rank 2  bounded_collective -> _reduce -> on_round  (in the reduction)
  rank 1  _pp_commit_comm_work <- _pp_forward_and_process_input_requests
          (blocked in work.wait() on ONE P2PWork, the top-of-pass commit)

Boot 18's inferred geometry is CONFIRMED, and rank 2 -- the datum that was
missing -- is in the reduction beside rank 0, exactly as was guessed.

BUT THE CAUSE IS NOT WHAT WAS INFERRED, and this is the finding. The log
shows all three ranks announcing and the gate opening on all three
("group present for epoch 0 after 0.00s"), so rank 1 DID pass the gate and
DID complete a consensus round. The deadlock is in the NEXT one:

  round N    all three enter, reduce, agree "armed but not ready", hold
  pass N+1   rank 1 must traverse its top-of-pass commit to get back to
             the hook. That commit blocks: rank 2 has posted no matching
             irecv, because rank 2 is inside round N+1's reduction
  round N+1  ranks 0 and 2 re-open the gate INSTANTLY on the epoch-0 flags
             that are still up from round N -- flags are never cleared --
             and enter. They wait for rank 1, which is blocked behind
             rank 2's missing recv. Cycle closed.

THE GATE'S GUARANTEE IS PER-ROUND; ITS EVIDENCE IS PER-EPOCH. That is the
defect, and it is structural. Monotone flags are what make a poll safe
against a racing writer -- and that same never-cleared property makes a
round-N quorum re-open the gate for round N+1 while proving nothing about
it. The gate is sound exactly once per epoch and is a rubber stamp after
that. Neither deadline saves it: both are evaluated BEFORE entry, and the
gate opens in 0.00 s on stale evidence.

Note for anyone re-reading the decided clause (i)/(ii) fix: it would not
have prevented this. Clause (i) gates the FIRST announce; rank 1's flag was
already up from round N, so the peers still enter round N+1 on it. And
clause (ii) cannot help, because the rank that must consume (rank 2) is
inside the blocking reduction, not in the poll loop where a drain runs.
The next design must address the inter-round interval, not the entry.

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

EVIDENCE MUST HAVE THE SAME SCOPE AS THE GUARANTEE IT LICENSES
--------------------------------------------------------------
THE RULE, and the one that cost the most to learn. A flag is evidence.
The gate's guarantee -- "every participant is at THIS reduction's entry"
-- is per ROUND. Stamping the evidence per EPOCH made round N's quorum a
standing authorisation for round N+1 and every round after it, because
flags are never cleared. The gate was then sound exactly once and a rubber
stamp thereafter, and the reproduction (corpse E) is what that buys: rank
1 blocked at its top-of-pass commit BETWEEN rounds while ranks 0 and 2
re-opened the gate in 0.00 s on stale flags and entered round N+1 without
it. Markers are therefore stamped ``(epoch, round)``.

THE INDUCTION, AND WHY IT NOW CLOSES
------------------------------------
Round-scoping is not merely a tighter check; it makes the announce mean
something it could not mean before. Announcing for round R is only
REACHABLE after this rank has completed its round-R top-of-pass commit --
that commit sits on the path to the hook -- so a round-stamped flag says
"my chain is settled for R". Hence:

  rank k announces R  =>  k's send to k+1 is settled for R
  all ranks announced R  =>  the whole chain 0->1->2 is settled for R
  => no rank inside round R's reduction is owed any chain operation
  => the blocking reduction is safe, per round, BY CONSTRUCTION

That is the same intent as the withdrawn "announce only once you owe no
send" clause, obtained at the correct granularity and with no new
machinery -- no drain, no progress engine, no predicate the transport
cannot honour (see corpse F).

The remaining failure mode is a rank that never REACHES round R because it
is busy with real work. The per-round pre-entry bound converts that into a
loud, unanimous abandonment and a later retry -- the designed outcome, and
free, because nothing has been entered.

MEASURED ON METAL, 2026-08-08 23:28Z (boot POLICY=auto, tree b51480f177).
The rubber stamp is gone: the markers left on disk were e0.n0.r0/r1/r2 and
NOTHING for round 1, so no rank re-entered a reduction on round 0's
evidence, and the ranks progressed PAST the flip reduction instead of
wedging inside it as they did at 23:12:38Z on the epoch-scoped build.

IT IS NOT SUFFICIENT, and the same boot proves that too. Round-scoping
closes the stale-evidence hole and nothing else; the flip still wedges,
for a reason that lives INSIDE a single round.

A NOTE ON A MISREADING, kept because it cost an hour and would cost the
next reader the same. The log reports this timeout as the collective
'kv_pressure_ladder.consensus', which reads like a different subsystem
diverging. It is not. The flip's consensus channel is built from
kv_pressure_runtime.default_collective_min (phase_flip_runtime, the
collective_min= argument), and that helper hardcodes its own module's
label. The timing-out collective IS the flip's own reduction, wearing
another feature's name. Do not go looking for a second bug.

THE REAL GAP: BETWEEN ANNOUNCE AND ENTRY, WITHIN ONE ROUND
----------------------------------------------------------
Measured 2026-08-08 23:39Z, all three stacks
(evidence-631/wedge_20260808T233910Z_KVPRESSURE_DIVERGENCE), markers
e0.n0.r0/r1/r2 -- one round, full quorum:

  rank 2  in the reduction for round 0
  rank 1  blocked at its top-of-pass commit (:724 -> :1187)
  rank 0  blocked at its top-of-pass commit (:724 -> :1187)

Announcing and ENTERING are not the same instant, and a whole pass can sit
between them. A rank announces at the hook; if the quorum is not yet
complete it returns and goes around the loop, and the NEXT thing it meets
is the top-of-pass commit. So the LAST rank to announce enters
immediately, while every rank that announced EARLIER must traverse a
blocking chain commit to get back to the entry -- a commit that blocks
precisely because the rank which already entered is no longer consuming.
Cycle: rank 2 in the reduction -> does not recv -> rank 1 stuck committing
to rank 2 -> does not recv -> rank 0 stuck committing to rank 1.

So the flag means "I was at the entry once", not "I am at the entry". The
gate's evidence is now correctly scoped in TIME (per round) and still
wrong in PLACE: nothing keeps a rank at the entry between announcing and
entering. Any fix must remove the blocking chain operation from that
interval -- not tighten the stamp again.

EPOCHS, ROUNDS, AND WHY FLAGS ARE NEVER CLEARED
-----------------------------------------------
Flags are monotone WITHIN AN (epoch, round) STAMP: a rank sets its own and
never unsets it. Retraction -- a policy that changes its mind, a disarm on
timeout -- mints a NEW epoch instead. That is what makes a poll safe
against a stale read: a flag observed for (E, R) is a fact about (E, R)
forever, so no reader can be fooled by a racing writer, and no writer has
to coordinate a clear. Clearing would reintroduce exactly the ordering
problem the flags exist to remove.

Round-scoping does NOT weaken that. Monotonicity was never about the flag
outliving its question -- it is about no reader ever seeing a flag go
backwards. Narrowing the stamp keeps every bit of that safety and removes
only the over-reach: the flag stops answering a question it was never
evidence for. A later round simply asks a NEW question, and gets no answer
until its own quorum forms.

WHERE THE ROUND NUMBER COMES FROM, and why the ranks agree on it
---------------------------------------------------------------
NOT from any rank's local loop counter. Under event_loop_pp those diverge
in absolute value (pipeline fill, conditional per-slot ops) -- that
divergence is the reason the armed/parked gate exists at all, so building
the stamp on it would be circular. The round index is instead the count of
CONSENSUS REDUCTIONS this arm has completed, and the ranks agree on it by
construction: a completed reduction is a synchronisation point that every
participant leaves together, so all of them enter the next one with the
same count. The very collective being gated is what numbers the gate.
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

    def _path(self, epoch: int, rank: int, round_: int = 0) -> str:
        return os.path.join(
            self.directory,
            f"{self.instance}.e{int(epoch)}.n{int(round_)}.r{int(rank)}",
        )

    def announce(self, epoch: int, note: str = "", round_: int = 0) -> None:
        """Publish THIS rank's readiness for ``(epoch, round_)``. Idempotent.

        Written via a temp file and an atomic rename, so a reader can never
        observe a partially created marker. Announcing twice is a no-op by
        construction, which matters because the armed poll loop calls this
        every iteration rather than tracking whether it already did.
        """
        path = self._path(epoch, self.rank, round_)
        if os.path.exists(path):
            return
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                fh.write(
                    f"rank={self.rank} epoch={epoch} round={round_} "
                    f"pid={os.getpid()} {note}\n"
                )
            os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.error("%s could not announce %s: %s", LOG_PREFIX, path, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def observe(self, epoch: int, round_: int = 0) -> Set[int]:
        """Which ranks have announced for ``(epoch, round_)``. Never blocks."""
        present: Set[int] = set()
        for r in range(self.n_ranks):
            if os.path.exists(self._path(epoch, r, round_)):
                present.add(r)
        return present

    def missing(self, epoch: int, round_: int = 0) -> List[int]:
        present = self.observe(epoch, round_)
        return [r for r in range(self.n_ranks) if r not in present]

    def all_present(self, epoch: int, round_: int = 0) -> bool:
        return len(self.observe(epoch, round_)) == self.n_ranks

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
                # {instance}.e{epoch}.n{round}.r{rank}
                epoch_part = name.split(".e", 1)[1].split(".n", 1)[0]
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
