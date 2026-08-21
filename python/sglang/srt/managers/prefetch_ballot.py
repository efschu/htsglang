"""#791b: the rank-uniform ballot for HiCache storage-prefetch admission.

THE DEFECT THIS CLOSES, measured twice on metal (boots instr22 and instr23,
2026-08-21). The TP-phase scheduler is N replicas that agree only by
determinism, and the admission loop's prefetch veto reads
``check_prefetch_progress`` -- a RANK-LOCAL verdict, since the storage
backend loads each rank's KV shard at that rank's own speed. instr23's
named-reason trace caught the split verbatim:

    11:45:27 PP0/PP1/PP2  DECLINE reason=loop_skips(prefetch_pending=2(...))
    11:45:28 PP2          ADMIT rid=5e58e0b6... chunked=1
    11:45:28 PP0/PP1      (still declining on prefetch_pending)

One rank formed a batch its peers refused. The admitting rank entered the
batch's model all_reduce, one peer entered the NEXT pass's uniform-budget
all_reduce, the other the next pass's request broadcast -- three crossed
lockstep families, none completable, until the barlink spin deadline
aborted the group 150 s later. The killer is the divergent admission; the
abort is only the messenger.

THE FIX SHAPE, and why a ballot rather than a synced verdict: the verdict
CANNOT be made rank-uniform at its source (the loads genuinely finish at
different times), so the DECISION built from it must be. Every rank
contributes its local verdict for the replicated head of the waiting
queue; a MIN-reduce turns that into the group verdict (MIN == AND: done
only when EVERY rank is done); every rank then admits or skips from the
GROUP verdict. A rank is never asked to admit a request it has not
finished prefetching -- group-done implies locally done -- so the ballot
only ever DELAYS an admission, never forces one.

The ballot rides ``_update_uniform_pool_budget``'s existing packed reduce
(scheduler.py, #610/#616g/#639 precedent): that site runs exactly once per
TP-loop iteration, is proven alignment-safe, and adds no new collective.
The PP loop never passes that site, deliberately -- PP admission uniformity
is #791's forwarded-schedule contract, not this ballot.

SELECTION MUST BE REPLICATED, VERDICTS ARE LOCAL. The slots are the first
``PREFETCH_BALLOT_SLOTS`` rids of ``waiting_queue`` in queue order, which
the replication premise makes identical across ranks. The premise is
VERIFIED, not assumed: a CRC of the slot rids rides the same reduce as a
(x, -x) pair, so every rank sees the group's min and max digest and knows
whether the heads agreed. On mismatch the ballot is void for the pass and
the caller falls back to the rank-local verdict -- the status quo ante --
after saying so once, loudly: a divergent queue head is a deeper breakage
this module must surface, not paper over.

``hash()`` is process-salted (PYTHONHASHSEED) and MUST NOT touch this
payload; the digest is ``zlib.crc32``, deterministic across processes.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence

# Payload slots reserved for the ballot: [digest, -digest, v0..v{K-1}].
# 32 covers the realistic queue depth of this rig's shapes (max-running 8
# plus a burst of 8 plus headroom); a rid deeper than that is handled
# conservatively by the caller (skip until it reaches the head), which can
# only delay it, never split the group.
PREFETCH_BALLOT_SLOTS = 32

# Sentinel verdict for a slot with no rid behind it (queue shorter than the
# slot count). 1 is the MIN-neutral element for AND, so short queues never
# veto anything.
_EMPTY_SLOT = 1


def prefetch_ballot_digest(rids: Sequence[str]) -> int:
    """Deterministic cross-process digest of the slot rids, as a non-negative
    int64. Empty head digests to 0 on every rank, which agrees trivially."""
    if not rids:
        return 0
    return zlib.crc32("|".join(rids).encode("utf-8")) & 0x7FFFFFFF


def build_prefetch_ballot_payload(
    rids: Sequence[str], local_done: Dict[str, bool]
) -> List[int]:
    """This rank's ballot contribution: [digest, -digest, v0..v{K-1}].

    ``local_done`` maps rid -> this rank's OWN verdict. A rid the drain did
    not report (registered this very pass, or never registered because there
    is nothing to prefetch for it) contributes its absence as True: the
    drain only tracks ONGOING prefetches, so absence means "nothing pending
    on this rank" -- exactly what the local admission veto reads today. The
    fixed width is the point: every rank appends the same number of
    elements whatever its queue looks like.
    """
    rids = list(rids)[:PREFETCH_BALLOT_SLOTS]
    digest = prefetch_ballot_digest(rids)
    payload = [digest, -digest]
    for i in range(PREFETCH_BALLOT_SLOTS):
        if i < len(rids):
            payload.append(1 if local_done.get(rids[i], True) else 0)
        else:
            payload.append(_EMPTY_SLOT)
    return payload


def unpack_prefetch_ballot(
    reduced: Sequence[int], rids: Sequence[str]
) -> Optional[Dict[str, bool]]:
    """The GROUP verdict from the reduced ballot slice, or None on a digest
    mismatch (the ranks disagreed about the queue head -- void the ballot
    for this pass; the caller falls back to the local verdict and logs).

    ``reduced`` is the ballot's slice of the MIN-reduced payload, laid out
    exactly as ``build_prefetch_ballot_payload`` packed it. The (x, -x)
    digest pair yields min at [0] and (negated) max at [1]; equality of the
    two IS the uniformity check, on the same trick #616g uses for pool
    availability.
    """
    if len(reduced) != PREFETCH_BALLOT_SLOTS + 2:
        return None
    digest_min = int(reduced[0])
    digest_max = -int(reduced[1])
    if digest_min != digest_max:
        return None
    rids = list(rids)[:PREFETCH_BALLOT_SLOTS]
    return {rid: bool(int(reduced[2 + i])) for i, rid in enumerate(rids)}


def prefetch_done_under_ballot(
    local_done: bool, rid: str, ballot: Optional[Dict[str, bool]]
) -> bool:
    """The verdict the admission loop must consume.

    * ballot None (single rank, PP loop, digest mismatch): the local
      verdict, byte-identical to the pre-ballot path.
    * rid in the ballot: the GROUP verdict, whatever the local one says.
      MIN semantics guarantee group-done implies locally done, so this
      never admits an unfinished rank; a locally-done rank held back by a
      pending peer is the ballot doing its one job.
    * rid NOT in the ballot (deeper than the slot count): CONSERVATIVELY
      pending. Admitting from the local verdict here would reopen the
      exact split this module closes, one queue position deeper; skipping
      merely delays the rid until it enters the covered head.
    """
    if ballot is None:
        return local_done
    verdict = ballot.get(rid)
    if verdict is None:
        return False
    return verdict
