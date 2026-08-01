# Copyright 2026 SGLang Team
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
"""Reserving capacity on a tier, as a NAMED ledger post (#407 cut 1).

DESIGN_305 §6 states the constraint this module exists to satisfy, and states
it in capitals: *one ledger, no second accounting*. There are already two --
``registry/ledger.py`` (cross-process, one JSON file per card, keyed by NVML
UUID, enforcing ``sum(reserved) + corridor <= nvml_total``) and
``offload_movement.CapacityLedger`` (in-process, keyed by
``(target_name, cuda_ordinal)``). #407 makes the second a *view* of the first.
It does not write a third.

So this module is a **hook layer**, not a store. It contributes exactly two
things on top of what already exists:

*   **A reservation must be a named post.** The #260/#400 Posten convention:
    every byte a holder declares is booked under a name a human can read back
    in a refusal (``ReservationEntry.posts``, ``LaneCardDemand.posts``). An
    unnamed reservation is refused at the door -- a bucket called "other" is
    how a ledger stops being able to explain itself.
*   **A refusal names the tier, the holders and the arithmetic.** #400's rule,
    learned when arm L was accepted and then died at 31.14 GiB in use on a
    31.34 GiB card: the ledger is a FLOOR, an unpriceable item makes the
    caller refuse rather than guess, and a rejection prints an itemised table.

Two implementations:

``InMemoryTierLedger``
    process-local, for tiers that have no cross-process store yet (host RAM,
    filesystems) and for tests. It is the reference semantics.
``VramLedgerHook``
    device tiers, forwarding to ``registry.ledger.ReservationStore`` so a
    device reservation lands in the file the card already owns -- posts and
    all. Nothing in cut 1 calls it in production; cut 4 makes #286's
    ``CapacityLedger`` a view through it.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List, Mapping, Optional, Protocol, Tuple

import msgspec

from sglang.srt.memtier.tiers import TierDescriptor, TierId, TierKind
from sglang.srt.planner.cost_model import AbsentRate

__all__ = [
    "InMemoryTierLedger",
    "TierLedger",
    "TierPost",
    "TierReservation",
    "TierReservationRejected",
    "UnnamedPost",
    "VramLedgerHook",
    "ledger_post_name",
]

#: A post name is a lower-case identifier. Restrictive on purpose: it ends up
#: in a JSON key, a log line and a refusal table, and a name with a space or a
#: colon in it would collide with the ``TierId`` grammar.
_POST_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Namespace for every post this package writes into a shared ledger, so a
#: memtier post can never be confused with a tenant's own (``vram_dial``,
#: ``serving_boot``, ``training``) in the file they share.
POST_NAMESPACE = "memtier"


class UnnamedPost(ValueError):
    """A reservation with no name, or with a name a refusal cannot print."""


class TierReservationRejected(RuntimeError):
    """The tier cannot hold the request. Nothing was written.

    Carries the itemisation rather than only a sentence, because the failure
    this replaces is a runtime OOM three minutes later with no table.
    """

    def __init__(
        self,
        message: str,
        *,
        tier_id: TierId,
        requested_bytes: int,
        holders: Mapping[str, int],
    ) -> None:
        super().__init__(message)
        self.tier_id = tier_id
        self.requested_bytes = requested_bytes
        self.holders = dict(holders)


def ledger_post_name(tier: TierDescriptor, post: str) -> str:
    """``memtier:<ledger_key>:<post>`` -- the bucket a reservation books to.

    The tier's own ``caps.ledger_key`` is in the middle so that two tiers of
    the same kind on one card (a VRAM tier and, later, a peer window on the
    same card) stay separable in one ledger file.
    """
    if not _POST_NAME_RE.match(post):
        raise UnnamedPost(
            f"{post!r} is not a post name; a tier reservation must be booked "
            "under a lower-case identifier that a refusal table can print "
            "(#260/#400 Posten convention)"
        )
    return f"{POST_NAMESPACE}:{tier.caps.ledger_key}:{post}"


class TierPost(msgspec.Struct, frozen=True, kw_only=True):
    """One named claim on one tier by one holder."""

    name: str
    nbytes: int
    holder: str

    def __post_init__(self) -> None:
        if not _POST_NAME_RE.match(self.name):
            raise UnnamedPost(
                f"{self.name!r} is not a post name; see the #260/#400 Posten "
                "convention -- lower-case identifier, printable in a table"
            )
        if not self.holder:
            raise UnnamedPost("a post must name its holder")
        if self.nbytes < 0:
            raise ValueError("a post cannot claim a negative number of bytes")


class TierReservation(msgspec.Struct, frozen=True, kw_only=True):
    """A granted claim: the tier, the post, and the ledger key it landed in."""

    tier_id: TierId
    post: TierPost
    ledger_key: str

    @property
    def nbytes(self) -> int:
        return self.post.nbytes


class TierLedger(Protocol):
    """What a consumer needs from whichever store owns a tier's bytes."""

    def reserve(self, *, tier: TierDescriptor, post: TierPost) -> TierReservation: ...

    def release(self, *, tier: TierDescriptor, post_name: str, holder: str) -> bool: ...

    def posts(self, tier_id: TierId) -> Mapping[str, int]: ...

    def reserved_bytes(self, tier_id: TierId) -> int: ...


def _headroom_or_refuse(tier: TierDescriptor, post: TierPost, held: int) -> float:
    """The bytes actually available, or the refusal that says why nobody knows."""
    headroom = tier.capacity.headroom()
    try:
        available = headroom.require("headroom")
    except AbsentRate as exc:
        raise TierReservationRejected(
            f"cannot reserve {post.nbytes} bytes on {tier.id} for "
            f"{post.holder}: the headroom is unbounded ({exc.reason}). "
            "#400's rule is that the caller refuses rather than guesses -- a "
            "headroom nobody could compute is not a large headroom.",
            tier_id=tier.id,
            requested_bytes=post.nbytes,
            holders={},
        ) from exc
    return available - held


class InMemoryTierLedger:
    """Process-local posts for tiers with no cross-process store yet.

    Thread-safe: the insert/test/claim is one locked block, which is the shape
    #391 had to retrofit after a lost update landed on the very number a host
    model was judged by.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._posts: Dict[TierId, Dict[str, TierPost]] = {}

    def reserve(self, *, tier: TierDescriptor, post: TierPost) -> TierReservation:
        key = ledger_post_name(tier, post.name)
        with self._lock:
            existing = self._posts.setdefault(tier.id, {})
            held = sum(p.nbytes for name, p in existing.items() if name != key)
            available = _headroom_or_refuse(tier, post, held)
            if post.nbytes > available:
                raise TierReservationRejected(
                    self._rejection_text(tier, post, existing, held),
                    tier_id=tier.id,
                    requested_bytes=post.nbytes,
                    holders={n: p.nbytes for n, p in existing.items()},
                )
            existing[key] = post
        return TierReservation(tier_id=tier.id, post=post, ledger_key=key)

    def release(self, *, tier: TierDescriptor, post_name: str, holder: str) -> bool:
        key = ledger_post_name(tier, post_name)
        with self._lock:
            posts = self._posts.get(tier.id, {})
            current = posts.get(key)
            if current is None or current.holder != holder:
                return False
            del posts[key]
            return True

    def posts(self, tier_id: TierId) -> Mapping[str, int]:
        with self._lock:
            return {n: p.nbytes for n, p in self._posts.get(tier_id, {}).items()}

    def reserved_bytes(self, tier_id: TierId) -> int:
        return sum(self.posts(tier_id).values())

    @staticmethod
    def _rejection_text(
        tier: TierDescriptor,
        post: TierPost,
        existing: Mapping[str, TierPost],
        held: int,
    ) -> str:
        lines: List[str] = [
            f"reservation rejected on tier {tier.id} "
            f"({tier.kind.value}, host {tier.host}): "
            f"{post.holder} asked for {post.nbytes} bytes under post "
            f"{post.name!r}, {held} bytes are already posted and the tier's "
            f"headroom is {int(tier.capacity.headroom().require('headroom'))}"
        ]
        for name, other in sorted(existing.items()):
            lines.append(f"  {name:<40} {other.nbytes:>16} bytes  {other.holder}")
        return "\n".join(lines)


class VramLedgerHook:
    """Device tiers, booked in the card's own cross-process ledger.

    This is the "view, not a second accounting" half: the store, the
    invariant, the lease, the reaping and the corridor all stay in
    ``registry.ledger``; what this adds is the tier vocabulary on top and the
    namespaced post name underneath.
    """

    def __init__(self, store, *, tenant_id: str, klass: int = 1) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._klass = int(klass)

    def reserve(self, *, tier: TierDescriptor, post: TierPost) -> TierReservation:
        card_uuid = self._card_uuid(tier)
        key = ledger_post_name(tier, post.name)
        entry = self._store.read(card_uuid).tenant(self._tenant_id)
        posts: Dict[str, int] = dict(entry.posts) if entry is not None else {}
        posts[key] = post.nbytes
        total = sum(posts.values())
        try:
            self._store.acquire(
                card_uuid=card_uuid,
                tenant_id=self._tenant_id,
                klass=self._klass,
                reserved_bytes=total,
                nvml_total_bytes=self._nvml_total(tier),
                posts=posts,
            )
        except Exception as exc:  # the store's own rejection, re-labelled
            raise TierReservationRejected(
                f"reservation rejected on tier {tier.id}: {exc}",
                tier_id=tier.id,
                requested_bytes=post.nbytes,
                holders=dict(posts),
            ) from exc
        return TierReservation(tier_id=tier.id, post=post, ledger_key=key)

    def release(self, *, tier: TierDescriptor, post_name: str, holder: str) -> bool:
        del holder  # the store's holder is the tenant this hook was built for
        card_uuid = self._card_uuid(tier)
        key = ledger_post_name(tier, post_name)
        entry = self._store.read(card_uuid).tenant(self._tenant_id)
        if entry is None or key not in entry.posts:
            return False
        posts = {n: b for n, b in entry.posts.items() if n != key}
        self._store.acquire(
            card_uuid=card_uuid,
            tenant_id=self._tenant_id,
            klass=self._klass,
            reserved_bytes=sum(posts.values()),
            nvml_total_bytes=self._nvml_total(tier),
            posts=posts,
        )
        return True

    def posts(self, tier_id: TierId) -> Mapping[str, int]:
        entry = self._store.read(_card_uuid_of(tier_id)).tenant(self._tenant_id)
        return dict(entry.posts) if entry is not None else {}

    def reserved_bytes(self, tier_id: TierId) -> int:
        return sum(self.posts(tier_id).values())

    @staticmethod
    def _card_uuid(tier: TierDescriptor) -> str:
        if tier.kind is not TierKind.DEVICE:
            raise TypeError(
                f"tier {tier.id} is {tier.kind.value}; the card ledger books "
                "device tiers only"
            )
        if not tier.is_bound:
            raise TierReservationRejected(
                f"tier {tier.id} has never been enumerated from this host, so "
                "there is no card ledger to book it in",
                tier_id=tier.id,
                requested_bytes=0,
                holders={},
            )
        return _card_uuid_of(tier.id)

    @staticmethod
    def _nvml_total(tier: TierDescriptor) -> Optional[int]:
        total = tier.capacity.total
        return None if total.is_absent else int(total.require("total"))


def _card_uuid_of(tier_id: TierId) -> str:
    from sglang.srt.memtier.tiers import parse_tier_id

    parsed = parse_tier_id(tier_id)
    if parsed.card_key is None:
        raise TypeError(f"{tier_id} is not a device tier")
    return parsed.card_key


def summarise(ledger: TierLedger, tiers: Tuple[TierDescriptor, ...]) -> List[str]:
    """One line per tier with a post on it. The boot-log shape."""
    lines = []
    for tier in tiers:
        posts = ledger.posts(tier.id)
        if not posts:
            continue
        lines.append(
            f"{tier.id}: {sum(posts.values())} bytes over {len(posts)} post(s)"
        )
        for name, nbytes in sorted(posts.items()):
            lines.append(f"  {name:<40} {nbytes:>16} bytes")
    return lines
