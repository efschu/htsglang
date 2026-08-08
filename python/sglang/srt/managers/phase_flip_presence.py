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
rank block anywhere while a peer might be elsewhere has deadlocked. Seven
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
        self.instance = instance or str(os.environ.get("SGLANG_PHASE_FLIP_INSTANCE", os.getpid() // 100000))
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

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
