"""#825 -- prove the prefix trees are congruent at the one aligned point.

THE DEFECT THIS CLOSES, in the shape the tree itself states it:

  * ``scheduler.py:4684 _update_uniform_pool_budget`` MIN-reduces on
    ``tp_cpu_group``. Under ``--enable-phase-flip`` the PP-prefill phase has
    ``tp_size=1``, so ``scheduler.py:4770`` takes the world<=1 branch and
    switches all three uniformity floors OFF, logging
    "tp_cpu_group world=1 -> floors OFF (evict/host/mamba). pp_size=3".
  * ``_publish_uniform_evict_floor``'s own docstring names the consequence:
    the ranks evict independently "so the radix trees stop being replicas".
  * ``_cutover`` (``phase_flip_runtime.py``) touches ``tree_cache`` NOWHERE,
    while ``build_flip_live_slots_fn``'s docstring assumes "the tree and the
    batch state are rank-replicated between rounds".
  * The TP-decode phase then requires that identity, and does not have it.

WHY THE FIX IS HERE AND NOT AT THE REDUCE. Widening the per-iteration reduce
to a PP-spanning group is not merely risky, it is impossible: while a flip is
armed, ``_pp_flip_hold_slot`` continues the PP loop body WITHOUT advancing
``mb_id`` and the body free-runs at ~8 kHz per rank -- measured in-tree at
44477 / 33690 / 38069 passes in one window, a spread of 10787. The ranks
reach that reduce a different number of times, by thousands. The two recorded
deaths of this idea are ``scheduler.py:7121-7124`` (the HiCache ack-count
reduction, 2026-08-17: "PP0/PP1 in the drain, PP2 in the pipeline recv") and
``phase_flip_runtime.py:1463-1467`` ("one rank in the pool-budget all_reduce,
another in the relay's point_to_point recv, wedge") -- the latter about this
very reduce.

``scheduler.py:7100-7118`` states the requirement: "Anything added above that
needs GROUP agreement needs a pipeline-aligned point, not this one."
``phase_flip_runtime.py`` step 4c names the point: "the cutover is
group-aligned". This module supplies the arithmetic for a collective there.

DESIGN CONSTRAINTS, each of which is a way this could have been wrong:

  1. The digest is over KEYS, never over values. A node's ``value`` holds KV
     slot indices from THIS rank's allocator; two ranks caching identical
     text hold it at different slots by construction. Mixing values in would
     report divergence on every healthy boot -- a reconcile that fires always
     looks exactly like a working feature.
  2. The fold is COMMUTATIVE. Ranks reach identical content by different
     insertion orders, and Python dict iteration is insertion-ordered, so an
     order-dependent fold would disagree on trees that are in fact identical.
  3. The hash is STABLE ACROSS PROCESSES. Python's built-in ``hash()`` of
     strings and tuples is salted per process by PYTHONHASHSEED, and the
     ranks ARE separate processes. Using ``hash()`` here would have produced
     a digest that disagrees on identical trees, non-deterministically, and
     only on multi-process runs -- i.e. never in a unit test. blake2b instead.
  4. The payload width is FIXED at two and contributed UNCONDITIONALLY, on
     the argument ``_update_uniform_pool_budget`` already makes for its host
     and mamba pairs: "contributing it unconditionally is what makes that a
     property of the code rather than of the flagset". A rank with no tree
     contributes ``ABSENT_TREE_DIGEST``.
  5. The verdict is derived from the COLLECTIVE RESULT ONLY, never from the
     local value, so every rank takes the same branch. A reconcile that some
     ranks skip is the same class of defect as the divergence it repairs.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# A Mersenne prime, comfortably below 2**62 so that the (x, -x) pair rides an
# int64 reduce with no chance of the negation wrapping. If a digest could
# reach 2**63 the negated element would wrap to a positive value and MIN
# would return something that decodes as AGREEMENT -- a silent false-congruent
# on exactly the runs that need the check.
DIGEST_MODULUS = 2**61 - 1

# Distinct, non-zero, and deliberately not adjacent. Zero is excluded on
# purpose: a reduce that failed to write its output, or a payload that was
# never filled, reads as zero, and "everyone is zero" must not decode as
# "everyone agrees they are empty".
EMPTY_TREE_DIGEST = 0x5175_1EAF_0000_0001 % DIGEST_MODULUS
ABSENT_TREE_DIGEST = 0x4B5E_17ED_0000_0002 % DIGEST_MODULUS


def node_fingerprint(token_ids: Sequence[int], *, extra_key=None) -> int:
    """Stable fingerprint of ONE tree node's key.

    Takes token ids and nothing else -- there is deliberately no parameter for
    the node's ``value``, so a caller that tries to fold rank-local KV indices
    in gets a TypeError rather than a plausible number (constraint 1).

    Length is mixed in explicitly so that a key and its extension cannot
    collide, and so the empty key has a defined fingerprint.
    """
    ids = list(token_ids)
    payload = struct.pack("<Q", len(ids)) + b"".join(
        struct.pack("<q", int(t)) for t in ids
    )
    if extra_key is not None:
        # `RadixKey.extra_key` carries lora_id / cache_salt
        # (radix_cache.py:61). Two nodes with identical token ids but
        # different salts are DIFFERENT cache entries, so a digest that
        # ignored it would call two genuinely divergent trees congruent.
        payload += b"\x00" + str(extra_key).encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=8).digest()
    return struct.unpack("<Q", raw)[0] % DIGEST_MODULUS


def fold_digest(fingerprints: Iterable[int]) -> int:
    """Commutative fold of node fingerprints into one tree digest.

    Addition modulo a prime, not XOR: XOR cancels duplicates, so a tree
    holding the same key twice would fold identically to one holding it zero
    times. Commutativity is the requirement (constraint 2); multiplicity
    preservation is why it is addition.
    """
    total = EMPTY_TREE_DIGEST
    for fp in fingerprints:
        total = (total + int(fp)) % DIGEST_MODULUS
    return total


def digest_pair(digest: int) -> List[int]:
    """The (x, -x) pair, so ONE MIN all_reduce yields group min AND group max.

    This is the codebase's existing idiom, used three times over in
    ``_update_uniform_pool_budget`` for the device, host and mamba pools.
    Width is fixed at two and never depends on a per-rank capability
    (constraint 4).
    """
    d = int(digest)
    return [d, -d]


def reduce_pair_result(pairs: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Pure model of what ``all_reduce(..., op=MIN)`` does to those pairs.

    Exists so the arithmetic is testable without a process group. The real
    call site reduces the tensor; this reduces the list. Both must agree, and
    the wiring test is what pins that.
    """
    mins = min(int(p[0]) for p in pairs)
    min_neg = min(int(p[1]) for p in pairs)
    return mins, min_neg


def agreement(group_min: int, group_neg_min: int) -> bool:
    """True when every rank contributed the same digest.

    ``group_neg_min`` is ``-group_max``, so equality of min and max is
    exactly unanimity.
    """
    return int(group_min) == -int(group_neg_min)


@dataclass(frozen=True)
class CongruenceVerdict:
    congruent: bool
    must_reconcile: bool
    reason: str


def congruence_verdict(
    *, local_digest: int, group_min: int, group_neg_min: int
) -> CongruenceVerdict:
    """Decide from the COLLECTIVE RESULT ONLY (constraint 5).

    ``local_digest`` is accepted and reported so the log line can say what
    THIS rank held, but it must never enter the branch decision: a rank whose
    own digest happens to equal the group minimum still reconciles when the
    group disagreed. Every rank therefore takes the same branch, which is the
    property that keeps the repair from becoming a new divergence.
    """
    if agreement(group_min, group_neg_min):
        return CongruenceVerdict(congruent=True, must_reconcile=False, reason="")
    lo = int(group_min)
    hi = -int(group_neg_min)
    return CongruenceVerdict(
        congruent=False,
        must_reconcile=True,
        reason=(
            f"prefix trees diverged across ranks at the cutover: "
            f"group digest min={lo} max={hi}, this rank held {int(local_digest)}"
        ),
    )


def tree_digest_of(tree_cache) -> int:
    """Adapter: fold a live radix tree into one digest.

    Deliberately duck-typed over the three lineages (``radix_cache.py``,
    ``unified_radix_cache.py``, ``mamba_radix_cache.py``): all three expose a
    ``root_node`` whose nodes carry ``key`` (a ``RadixKey``) and ``children``
    (a dict). Anything missing yields ``ABSENT_TREE_DIGEST`` rather than an
    exception -- this runs on the flip's consensus path, where a raise takes
    the instance down, and a unit stub with no tree must behave like the
    census handle does (a no-op), not like a crash.

    ITERATIVE, not recursive: a deep radix tree would otherwise risk a
    RecursionError on the seam, which is the same "a raise here kills the
    instance" hazard.

    COST: O(nodes), and it runs only after ``on_round``'s early returns --
    i.e. on a round where the group is actually reducing, not on the ~8 kHz
    free-running loop body.
    """
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return ABSENT_TREE_DIGEST
    fingerprints: List[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        key = getattr(node, "key", None)
        ids = getattr(key, "token_ids", None)
        if ids is not None:
            fingerprints.append(
                node_fingerprint(ids, extra_key=getattr(key, "extra_key", None))
            )
        children = getattr(node, "children", None)
        if children:
            try:
                stack.extend(children.values())
            except AttributeError:
                stack.extend(children)
    return fold_digest(fingerprints)
