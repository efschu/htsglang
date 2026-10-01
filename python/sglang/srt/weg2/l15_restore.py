"""L15-12 wake side of the L1.5 hold: pure decision and check logic.

At D's wake every rank reads the manifest its sleep wrote
(``load_for_wake``). The caller collects the group's min/max fingerprint
(the collective is the caller's job) and every rank reaches the same
verdict via ``verdict``: "hold" keeps the hold, "fallback" drops it
group-wide (today's empty tree), "none" means nobody holds anything.

On "hold" the 3080 ranks kept their rows mapped (TMS keep spans); TP0
(cap 0, the 5090) kept nothing and refills its owned rows from L2 via
``refill_plan``. Ownership follows the l15_compact.owner_of rule: rank r
owns global slot L iff prefix[r] <= L % S < prefix[r+1], S = prefix[-1];
compact row = (L // S) * (prefix[r+1] - prefix[r]) + (L % S - prefix[r]).
Before the first admission, ``sample_rows`` picks a deterministic subset
to compare against L2; ``check_line`` and ``restore_line`` format it.
"""

import dataclasses
from typing import Callable, List, Optional, Sequence, Tuple

from sglang.srt.weg2.l15_manifest import Manifest, decide, read_and_clear


def load_for_wake(
    path: str, pid_alive: Optional[Callable[[int], bool]] = None
) -> Optional[Manifest]:
    """Manifest written by this rank's sleep, or None when nothing is
    held (absent file, or owning pid dead -> reaped by the read).

    READ-AND-CLEAR: the manifest's lifetime is one sleep-wake pair, so the
    file is unlinked as soon as it is read -- a stale record from an earlier
    sleep cannot re-vote at this wake or any later one, and that
    consumption is what makes the absent wake-epoch comparison moot (only
    this sleep's record can ever be what the wake reads)."""
    if pid_alive is None:
        return read_and_clear(path)
    return read_and_clear(path, pid_alive=pid_alive)


def verdict(
    local_fp: Optional[int], min_fp: Optional[int], max_fp: Optional[int]
) -> str:
    """"none" when nobody holds; "fallback" when the group is mixed (None
    mixed with ints) or decide() says so; else "hold"."""
    vals = (local_fp, min_fp, max_fp)
    if all(v is None for v in vals):
        return "none"
    if any(v is None for v in vals):
        return "fallback"
    return decide(min_fp, max_fp)


def _owns(prefix: Sequence[int], rank: int, slot: int) -> bool:
    lo = slot % prefix[-1]
    return prefix[rank] <= lo < prefix[rank + 1]


def _compact_row(prefix: Sequence[int], rank: int, slot: int) -> int:
    width = prefix[rank + 1] - prefix[rank]
    return (slot // prefix[-1]) * width + (slot % prefix[-1] - prefix[rank])


def _owned_tokens(m: Manifest, rank: int, prefix: Sequence[int]):
    for span in m.spans:
        for i, slot in enumerate(span.slots):
            if _owns(prefix, rank, slot):
                yield span, i, slot


def _l2_source(span, i: int) -> Optional[Tuple[int, int]]:
    """(l2_slot, l2_gen) for token i, or None when it has no L2 entry."""
    if i < len(span.l2_slots) and i < len(span.l2_gens):
        slot = span.l2_slots[i]
        if slot is not None and slot >= 0:
            return slot, span.l2_gens[i]
    return None


def refill_plan(
    m: Manifest, rank: int, prefix: Sequence[int],
    cap_rows_by_rank: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Rows rank must refill from L2 as (compact_row, l2_slot, l2_gen).

    Invariant (cap > 0 = resident): ``cap_rows_by_rank[rank] > 0`` means
    this rank KEPT its rows mapped on the TMS keep spans through the hold,
    so it owns no gap to refill and the plan is ``[]`` -- even a rank that
    owns slots. Only a cap-0 rank (TP0, the 5090: held nowhere, refilled
    from L2) gets a non-empty plan, naming exactly its L2-backed rows. A
    future PARTIAL-hold variant (a rank that keeps only some of its rows)
    must change this function: the ``cap > 0 -> []`` shortcut assumes
    "kept everything", not "kept some".

    Tokens without an L2 entry are skipped here and counted by
    count_missing()."""
    if cap_rows_by_rank[rank] > 0:
        return []
    plan = []
    for span, i, slot in _owned_tokens(m, rank, prefix):
        src = _l2_source(span, i)
        if src is not None:
            plan.append((_compact_row(prefix, rank, slot), src[0], src[1]))
    return plan


def count_missing(m: Manifest, rank: int, prefix: Sequence[int]) -> int:
    return sum(
        1 for span, i, _ in _owned_tokens(m, rank, prefix)
        if _l2_source(span, i) is None
    )


def sample_rows(plan_or_rows, k: int = 64) -> List[int]:
    """Deterministic, evenly spaced subset of row ids (stable across
    runs): accepts (compact_row, ...) tuples or plain row ids."""
    rows = sorted({x[0] if isinstance(x, tuple) else x for x in plan_or_rows})
    n = len(rows)
    if n <= k:
        return rows
    return [rows[(j * n) // k] for j in range(k)]


def check_line(rank: int, ok: int, bad: int, missing: int) -> str:
    return "L15-CHECK rank=%d ok=%d bad=%d missing=%d" % (
        rank, ok, bad, missing,
    )


def restore_line(
    epoch: int, verdict: str, rows_by_rank: Sequence[int],
    refill_rows: int, missing: int,
) -> str:
    # rows_by_rank is the manifest's per-rank KEEP capacity (not the
    # admitted rows of a HoldSet); printed as keep_rows_by_rank so the
    # two never clash under the same log key.
    rows = ",".join(str(x) for x in rows_by_rank)
    return (
        "L15-RESTORE epoch=%d verdict=%s keep_rows_by_rank=%s "
        "refill_rows=%d missing=%d"
        % (epoch, verdict, rows, refill_rows, missing)
    )


def l15_fp_reduce(
    votes: Sequence[Optional[int]],
) -> Tuple[Optional[Tuple[int, int]], bool]:
    """Reduce the group's per-rank manifest fingerprints (L15-12 part 2).

    ``votes`` is one entry per rank of the group: an int fingerprint when that
    rank read its sleep manifest, ``None`` when it held nothing (master off,
    or the manifest was absent / its owning process died).

    Returns ``(minmax, mixed)`` where:
      * ``minmax`` is ``(min, max)`` over the int votes, or ``None`` when no
        rank had a fingerprint (the group holds nothing).
      * ``mixed`` is True when some ranks had an int and at least one had
        ``None``: the group disagrees on whether a hold exists at all, which
        the wake site maps to "fallback" (a split hold cannot be kept whole).

    Pure over the vote list (no I/O, no process group) so it is unit-testable
    and its result is group-uniform: every rank passes the same ``gathered``
    list and therefore computes the same ``(minmax, mixed)``.
    """
    ints = [v for v in votes if isinstance(v, int) and not isinstance(v, bool)]
    if not ints:
        return None, False
    return (min(ints), max(ints)), (len(ints) != len(votes))


@dataclasses.dataclass(frozen=True)
class GroupCheck:
    """ONE group decision from the wake's sample-check votes (L15-12c-E1).

    Field values are order-independent aggregates of the gathered vote
    list, so every rank that passes the same list computes an equal
    GroupCheck (group-uniformity, same discipline as l15_fp_reduce)."""

    refuse: bool
    drop_rids: Tuple[str, ...]
    fp_mixed: bool
    verdict: str
    bad_ranks: Tuple[int, ...]


def check_vote(
    fp: Optional[int], ok: int, bad: int, missing: int,
    drop_rids: Sequence[str],
) -> Tuple[Optional[int], int, int, int, Tuple[str, ...]]:
    """Per-rank payload for the wake's check gather (plan section 4):
    plain ints plus a sorted tuple of str rids -- picklable and
    deterministic, so the gathered list replays identically everywhere."""
    return (
        None if fp is None else int(fp),
        int(ok), int(bad), int(missing),
        tuple(sorted({str(r) for r in drop_rids})),
    )


def group_check(votes: Sequence[Optional[tuple]]) -> GroupCheck:
    """Reduce one vote per rank (None = that rank had no hold; else a
    check_vote tuple) into the single group decision (plan section 4):

      * bad > 0 on ANY rank -> the whole group refuses (F11 "mismatch =
        stop"), the group stays DORMANT together;
      * drop_rids is the sorted UNION over all ranks -- dropped on all
        ranks together, never rank-local;
      * fp_mixed: the fingerprints disagree, or some ranks hold and others
        do not (l15_fp_reduce's split-hold rule) -> "fallback";
      * verdict order: refuse > fallback > hold (any rank held) > none.

    Pure over the vote list; sorting every aggregate makes the result
    identical for any order of the same votes."""
    bad_ranks = tuple(
        i for i, v in enumerate(votes) if v is not None and int(v[2]) > 0
    )
    drops = set()
    fps = []
    for v in votes:
        if v is None:
            fps.append(None)
        else:
            fps.append(v[0])
            drops.update(v[4])
    present = {f for f in fps if f is not None}
    fp_mixed = len(present) > 1 or (bool(present) and any(f is None for f in fps))
    held = any(f is not None for f in fps)
    if bad_ranks:
        verdict = "refuse"
    elif fp_mixed:
        verdict = "fallback"
    elif held:
        verdict = "hold"
    else:
        verdict = "none"
    return GroupCheck(
        refuse=bool(bad_ranks), drop_rids=tuple(sorted(drops)),
        fp_mixed=fp_mixed, verdict=verdict, bad_ranks=bad_ranks,
    )


def refusal_message(gc: GroupCheck, epoch: int) -> str:
    """The named refusal -- one string every rank raises identically after
    the gather (F11): identical inputs give identical text by construction."""
    return "L15-CHECK REFUSED epoch=%d bad_ranks=%s" % (
        epoch, ",".join(str(i) for i in gc.bad_ranks),
    )
