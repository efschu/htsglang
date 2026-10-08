# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E1W: the E1 gather -- the COLLECTIVE half of the wake check.

l15_restore already carries the pure half (check_vote, group_check,
refusal_message; L15-12c-E1).  This module adds the one collective that
makes it a *group* decision, kept here -- out of weight_updater -- so
that it is unit-testable with a fake group:

  * gather_votes: ONE torch.distributed.all_gather_object of this rank's
    check_vote tuple over the passed group.  ``world <= 1`` or
    ``group is None`` -> ``[vote]`` (no collective needed).
  * decide: gather_votes + group_check; if gc.refuse -> raise the named
    ``L15CheckRefused`` with ``refusal_message(gc, epoch)`` -- identical
    text on every rank (F11 "mismatch = stop"; the group stays DORMANT
    together).
  * sample_rows_equal: the byte-comparison primitive of the sample check
    (plan sec 4), pure over already-loaded rows; no loading here.

The weight_updater call site is a DESIGN NOTE (plan sec 9, this round
weight_updater belongs to the refill switch worker), not code yet:
decide() lands at the top of _pdflip_kv_clear_part, before
``scheduler.pdflip_dormant = False``, reached at the same list position by
every rank (xsn410: the clear half runs on all ranks or none).

A failed collective PROPAGATES: a rank whose all_gather_object fails must
not silently decide alone -- the xsn410 rule forbids any rank skipping the
collective, and a silent fallback would split the group at the tail.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from flliper.srt.pdflip import l15_restore


class L15CheckRefused(RuntimeError):
    """The named F11 refusal: raised by ``decide()`` on EVERY rank after
    the gather, with the identical refusal_message text (the message is a
    pure function of the gathered vote list, which is group-uniform)."""


def gather_votes(
    vote: Optional[Tuple], group: Optional[object], world: int
) -> List[Optional[Tuple]]:
    """One all_gather_object of this rank's check_vote tuple over ``group``.

    ``world`` is the group size; ``world <= 1`` or ``group is None`` means
    no collective exists to run (single-rank group, or the caller passed
    no group) and the vote list is ``[vote]``.  The gathered list is
    returned as given by the collective, in rank order.
    """
    if world <= 1 or group is None:
        return [vote]
    gathered: List[Optional[Tuple]] = [None] * world
    # NO try/except around the collective: a rank whose all_gather_object
    # fails must raise, not vote alone (xsn410: no rank may skip the
    # collective, and a silent fallback would split the group at the tail).
    torch.distributed.all_gather_object(gathered, vote, group=group)
    return gathered


def decide(
    vote: Optional[Tuple], group: Optional[object], world: int, epoch: int
) -> "l15_restore.GroupCheck":
    """gather_votes + l15_restore.group_check; refuse -> L15CheckRefused.

    Every rank passes its OWN ``vote`` but the collective hands every rank
    the SAME list, so the returned GroupCheck (and the raised message)
    is group-uniform.  ``vote is None`` is a legitimate vote ("I have no
    vote for this epoch").
    """
    votes = gather_votes(vote, group, world)
    gc = l15_restore.group_check(votes)
    if gc.refuse:
        raise L15CheckRefused(l15_restore.refusal_message(gc, epoch))
    return gc


def sample_rows_equal(device_rows: List[torch.Tensor], l2_rows: List[torch.Tensor]) -> Tuple[int, int]:
    """Byte-compare two equally-ordered lists of tensors.

    Pure: no loading, no device transfer, no side effects.  Returns
    ``(ok, bad)`` where *ok* is the number of positionwise-equal pairs
    (torch.equal) and *bad* is the number of positionwise-unequal pairs
    plus any unpaired rows (unequal list lengths).  Empty vs empty -> (0, 0).
    """
    n = min(len(device_rows), len(l2_rows))
    ok = sum(1 for i in range(n) if torch.equal(device_rows[i], l2_rows[i]))
    bad = (len(device_rows) - n) + (len(l2_rows) - n) + (n - ok)
    return ok, bad
