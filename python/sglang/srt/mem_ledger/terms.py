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
"""The ledger primitive: one VRAM term, its number, and where the number came
from.

THE RULE THIS MODULE ENFORCES (user decree, 2026-08-05). A card's memory is

    card_total = user_reserve + internal_demand + kv_pool

with all three exact and none of them padding for another:

``user_reserve``
    Space the OPERATOR wants left free for things outside this engine. It is
    the only number a human chooses, it defaults to
    :data:`DEFAULT_USER_RESERVE_MIB`, and NOTHING internal is ever funded from
    it. Growing it must never be the remedy for an internal term that was
    modelled too small -- that is the guessing loop this ledger exists to end.

``internal_demand``
    The sum of the terms below, each computed. A term is either MODELED (it
    follows from configuration and geometry) or CALIBRATED (it depends on the
    card, driver or build, so it is measured ONCE per hardware fingerprint and
    cached). There is no third kind. A bare literal is not a term.

``kv_pool``
    The residual claimant. It takes exactly what the first two leave, which is
    why surplus never sits idle and why an underestimated internal term shows
    up as an OOM rather than as unused memory.

WHY PROVENANCE IS A FIELD AND NOT A COMMENT. #493 is the case: a transient the
reserve did not charge was chased with +500 MiB of reserve, the free-memory
floor did not move, and the KV pool lost 48 640 tokens for nothing. The reserve
shapes the BUDGET; it does not cap an allocation. A term that cannot be bounded
by a number must therefore be bounded by a MECHANISM (an allocator cap, staged
execution), and the ledger records which mechanism does it. A term that is
neither computed nor mechanism-bounded is :attr:`CardVramLedger.unbounded`, and
an unbounded term is a REFUSAL, never a warning -- "short by N MiB" is exactly
the class of message this design deletes.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_USER_RESERVE_MIB",
    "Provenance",
    "LedgerTerm",
    "CardVramLedger",
    "LedgerError",
    "LedgerOvercommit",
    "USER_RESERVE_TERM",
]


#: Default operator headroom per card, in MiB. This is the ONE number in the
#: ledger that is a policy choice rather than a derivation, and it is a choice
#: about the world OUTSIDE the engine: a desktop compositor, an nvidia-smi, a
#: short-lived CUDA tool. It is deliberately not zero -- a card driven to its
#: last megabyte leaves the operator no room to look at it -- and it is
#: deliberately not larger, because every MiB here is a MiB the KV pool does
#: not get. Internal demand is never charged against it.
DEFAULT_USER_RESERVE_MIB = 1024

#: The ledger line that carries the user reserve. Named so that a renderer, a
#: test and a refusal all point at the same row.
USER_RESERVE_TERM = "user reserve (external)"


class Provenance(str, enum.Enum):
    """Where a term's number comes from. There are exactly three kinds.

    ``MODELED``
        Derived from configuration and model geometry alone: shard bytes, KV
        entry bytes, pool floors, graph-capture token counts. Reproducible on
        any host without touching a card, and therefore checkable by a
        hermetic test.

    ``CALIBRATED``
        A HARDWARE RESIDUAL: it depends on the card, the driver, or the build
        (per-process CUDA context size, JIT/workspace footprints, allocator
        granularity). Measured once by a cheap probe and cached under the
        hardware fingerprint; invalidated when the fingerprint changes. Never
        a literal, never carried across a driver or wheel change.

    ``DECLARED``
        A co-resident tenant's own ledger, folded in whole. The tenant is
        responsible for the provenance of its lines; this ledger only records
        that the bytes were declared and by whom, so a coresident boot sums
        exactly instead of summing an engine against a guess.
    """

    MODELED = "modeled"
    CALIBRATED = "calibrated"
    DECLARED = "declared"


class LedgerError(RuntimeError):
    """A ledger that cannot be turned into a boot."""


class LedgerOvercommit(LedgerError):
    """The card cannot hold ``user_reserve + internal_demand``.

    Raised at parse/validate time, carrying the itemization, because the
    alternative is an OOM in the first real prefill with no statement of what
    asked for the bytes.
    """

    def __init__(self, ledgers: Sequence["CardVramLedger"]):
        self.ledgers = tuple(ledgers)
        super().__init__("\n\n".join(x.render() for x in self.ledgers))


@dataclasses.dataclass(frozen=True)
class LedgerTerm:
    """One item of a card's internal demand.

    ``derivation`` is not documentation, it is the term's justification, and
    the constructor refuses a term without one. The point is that a reader can
    check the number against the configuration without reading the code that
    produced it, and that a future edit which replaces a derivation with a
    constant is visible in a diff rather than invisible in a sum.

    ``inputs`` names the configuration fields the number actually reads. A
    hermetic test asserts that a MODELED term moves when one of its declared
    inputs moves, which is what makes "no bare literals" checkable rather than
    aspirational.

    ``bounded_by`` names the MECHANISM that caps this term when the term is a
    transient whose size is not fixed by configuration (an allocator cap, a
    chunk size, staged execution). #493: padding does not cap a transient, so
    a transient term without a mechanism is not a term -- it is an
    :attr:`CardVramLedger.unbounded` entry and refuses the boot.

    ``not_applicable`` is the THIRD state a quantity can be in, and it is the
    reason this flag exists rather than a bare 0 MiB row (#598). A term is:

      * a NUMBER, when it was modeled or measured. Zero is one of those
        numbers: "we looked at this allocation and it came out at 0 MiB".
      * :attr:`CardVramLedger.unbounded`, when the allocation exists and
        nobody could bound it. A refusal.
      * NOT APPLICABLE, when the allocation does not EXIST in this
        configuration -- a different transport owns the path, the feature is
        off, the group has one rank.

    Collapsing the third into the first loses the distinction that matters
    when the configuration changes: a measured zero is invalidated by a new
    measurement, a NOT_APPLICABLE is invalidated by a config change, and a
    reader of a 0 row cannot tell which of the two he is looking at unless the
    row says so. Charging is identical (0 MiB either way) -- the difference is
    entirely in what the ledger CLAIMS, and this ledger's whole premise is
    that the claim is checkable.
    """

    name: str
    mib: int
    provenance: Provenance
    derivation: str
    inputs: Tuple[str, ...] = ()
    bounded_by: Optional[str] = None
    #: True when the quantity does not exist in this configuration. Forces
    #: ``mib == 0``; see the class docstring for why this is not the same
    #: statement as a measured zero.
    not_applicable: bool = False
    #: Fingerprint the calibration was taken under. Set for CALIBRATED terms
    #: only; a mismatch against the live fingerprint invalidates the value.
    fingerprint: Optional[str] = None
    #: For DECLARED terms: which tenant declared these bytes.
    tenant: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise LedgerError("a ledger term must be named")
        if not str(self.derivation).strip():
            raise LedgerError(
                f"ledger term {self.name!r} carries no derivation. Every term "
                "is either computed from configuration or measured under a "
                "hardware fingerprint; a number without a derivation is the "
                "guess this ledger exists to remove."
            )
        if int(self.mib) < 0:
            raise LedgerError(
                f"ledger term {self.name!r} is negative ({self.mib} MiB); a "
                "term is a claim on memory, never a credit against another "
                "term"
            )
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        object.__setattr__(self, "mib", int(self.mib))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if self.provenance is Provenance.MODELED and not self.inputs:
            raise LedgerError(
                f"MODELED term {self.name!r} declares no config inputs. A "
                "modeled term that reads nothing is a literal wearing a "
                "derivation string."
            )
        if self.provenance is Provenance.CALIBRATED and not self.fingerprint:
            raise LedgerError(
                f"CALIBRATED term {self.name!r} carries no hardware "
                "fingerprint. A measured number without the hardware it was "
                "measured on cannot be invalidated, and an un-invalidatable "
                "measurement is a literal."
            )
        if self.provenance is Provenance.DECLARED and not self.tenant:
            raise LedgerError(
                f"DECLARED term {self.name!r} names no tenant; a coresident "
                "sum must be able to say who asked for the bytes"
            )
        if self.not_applicable and self.mib:
            raise LedgerError(
                f"term {self.name!r} is marked not applicable and still "
                f"charges {self.mib} MiB. 'This allocation does not exist in "
                "this configuration' and 'it exists and costs something' are "
                "not both true"
            )

    @property
    def mark(self) -> str:
        """The provenance column, carrying every qualifier a reader needs in
        order to tell two 0 MiB rows apart."""
        mark = self.provenance.value
        if self.not_applicable:
            mark = f"{mark}/n-a"
        if self.fingerprint:
            mark = f"{mark}@{self.fingerprint[:12]}"
        if self.tenant:
            mark = f"{mark}:{self.tenant}"
        return mark

    @property
    def row(self) -> str:
        cap = f" [capped by {self.bounded_by}]" if self.bounded_by else ""
        return f"{self.name}|{self.mib}|{self.mark}|{self.derivation}{cap}"

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "mib": self.mib,
            "provenance": self.provenance.value,
            "derivation": self.derivation,
            "inputs": list(self.inputs),
            "bounded_by": self.bounded_by,
            "fingerprint": self.fingerprint,
            "tenant": self.tenant,
            "not_applicable": self.not_applicable,
        }


@dataclasses.dataclass(frozen=True)
class CardVramLedger:
    """One physical card's complete accounting.

    The invariant, checked by :meth:`validate` and printed by :meth:`render`:

        total_mib == user_reserve_mib + demand_mib + kv_pool_mib

    It holds by construction, because :attr:`kv_pool_mib` is computed as the
    residual rather than chosen. What can fail is FEASIBILITY -- the reserve
    plus the demand exceeding the card -- and that is a refusal.
    """

    gpu_id: int
    card: str
    total_mib: int
    user_reserve_mib: int
    terms: Tuple[LedgerTerm, ...]
    #: Items that belong on this card, are known to exist, and could be
    #: neither computed nor mechanism-bounded. A non-empty tuple is a refusal:
    #: the ledger cannot answer, so it must not pretend to.
    unbounded: Tuple[str, ...] = ()
    #: Ranks placed on this card. Purely descriptive -- co-location is already
    #: priced inside the individual terms -- but a refusal has to name them.
    ranks: Tuple[int, ...] = ()
    #: What the residual is called. Normally the KV pool. At the point in the
    #: boot where the shard vector does not exist yet, the weights cannot be a
    #: term (they depend on a ratio that is derived FROM these budgets), so the
    #: residual is the RANK BUDGET, which funds weights and KV pool together.
    #: The label is a field rather than a constant because a ledger that
    #: printed "KV pool" for a number that also has to hold the weights would
    #: be lying in exactly the direction that OOMs a boot.
    residual_label: str = "KV pool (residual)"
    residual_note: str = (
        "total - user reserve - demand; surplus flows here rather than sitting idle"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "unbounded", tuple(self.unbounded))
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if self.user_reserve_mib < 0:
            raise LedgerError(
                f"user reserve on GPU {self.gpu_id} is negative "
                f"({self.user_reserve_mib} MiB)"
            )
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise LedgerError(
                f"GPU {self.gpu_id} charges {dupes} twice. A term is charged "
                "once; co-location and multi-tenancy are priced INSIDE a term "
                "or declared as separate named terms, never by repeating one."
            )

    @property
    def demand_mib(self) -> int:
        """Internal demand. Excludes the user reserve, by definition."""
        return sum(t.mib for t in self.terms)

    @property
    def committed_mib(self) -> int:
        return self.user_reserve_mib + self.demand_mib

    @property
    def kv_pool_mib(self) -> int:
        """The residual claimant. Never negative; when the card is
        overcommitted this is 0 and :meth:`fits` is False."""
        return max(self.total_mib - self.committed_mib, 0)

    @property
    def fits(self) -> bool:
        return not self.unbounded and self.committed_mib <= self.total_mib

    @property
    def overcommit_mib(self) -> int:
        return max(self.committed_mib - self.total_mib, 0)

    def term(self, name: str) -> Optional[LedgerTerm]:
        for t in self.terms:
            if t.name == name:
                return t
        return None

    def by_provenance(self, provenance: Provenance) -> Tuple[LedgerTerm, ...]:
        return tuple(t for t in self.terms if t.provenance is provenance)

    def validate(self) -> None:
        """Raise :class:`LedgerOvercommit` unless this card is fundable."""
        if not self.fits:
            raise LedgerOvercommit([self])
        assert (
            self.total_mib == self.user_reserve_mib + self.demand_mib + self.kv_pool_mib
        ), "ledger identity broken"

    def render(self) -> str:
        """The itemization the boot log prints and a refusal carries.

        Deliberately one line per term with the derivation on it: the reader
        of a refusal is trying to find WHICH term is too big, and a bare total
        cannot answer that.
        """
        rows = [
            (USER_RESERVE_TERM, self.user_reserve_mib, "operator", "external headroom")
        ]
        for t in self.terms:
            # One mark builder, so a qualifier added to the row (e.g. the
            # #598 not-applicable marker) cannot be visible in one renderer
            # and invisible in the other.
            mark = t.mark
            why = t.derivation
            if t.bounded_by:
                why = f"{why} [capped by {t.bounded_by}]"
            rows.append((t.name, t.mib, mark, why))

        name_w = max([len(r[0]) for r in rows] + [24])
        mark_w = max([len(r[2]) for r in rows] + [10])
        ranks = ", ".join(str(r) for r in self.ranks) or "none"
        if self.fits:
            verdict = f"FITS, KV pool {self.kv_pool_mib} MiB"
        elif self.unbounded:
            # Distinguished on purpose: "OVERCOMMITTED by 0 MiB" is what this
            # said before, and it sent a reader hunting for a size problem when
            # the actual fault is a term nobody could bound.
            verdict = f"REFUSED, {len(self.unbounded)} unbounded item(s)" + (
                f" (and overcommitted by {self.overcommit_mib} MiB)"
                if self.overcommit_mib
                else ""
            )
        else:
            verdict = f"OVERCOMMITTED by {self.overcommit_mib} MiB"
        lines = [
            f"VRAM ledger for {self.card} (ranks: {ranks}): "
            f"{self.total_mib} MiB total -- {verdict}",
        ]
        for name, mib, mark, why in rows:
            lines.append(f"  {name:<{name_w}}  {mib:>7} MiB  {mark:<{mark_w}}  {why}")
        for item in self.unbounded:
            lines.append(
                f"  {item:<{name_w}}  {'?':>7} MiB  {'UNBOUNDED':<{mark_w}}  "
                "neither computed nor mechanism-bounded"
            )
        lines.append(f"  {'-' * name_w}  {'-' * 7}      {'-' * mark_w}")
        lines.append(
            f"  {'user reserve + demand':<{name_w}}  {self.committed_mib:>7} MiB"
        )
        lines.append(
            f"  {self.residual_label:<{name_w}}  {self.kv_pool_mib:>7} MiB  "
            f"{'residual':<{mark_w}}  {self.residual_note}"
        )
        if self.unbounded:
            lines.append(
                "  REFUSED: an item above could not be bounded. Padding does "
                "not cap a transient (#493); bound it by mechanism (allocator "
                "cap / chunk size / staged execution) or model it."
            )
        elif not self.fits:
            lines.append(
                f"  REFUSED: this card is short {self.overcommit_mib} MiB "
                "BEFORE any KV pool. Lower a term's driver (context length, "
                "chunked prefill size, graph ladder, co-located ranks) or "
                "lower --rank-user-reserve-mib; raising the reserve cannot "
                "help, it is on the wrong side of the sum."
            )
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "card": self.card,
            "total_mib": self.total_mib,
            "user_reserve_mib": self.user_reserve_mib,
            "demand_mib": self.demand_mib,
            "kv_pool_mib": self.kv_pool_mib,
            "fits": self.fits,
            "ranks": list(self.ranks),
            "unbounded": list(self.unbounded),
            "terms": [t.to_json() for t in self.terms],
        }


def merge_terms(
    base: Iterable[LedgerTerm], extra: Iterable[LedgerTerm]
) -> Tuple[LedgerTerm, ...]:
    """Concatenate two term sets, refusing a name collision.

    Used where a tenant folds its declared lines into an engine's card ledger.
    A collision is an error rather than a silent overwrite or a silent sum:
    two sources charging the same name means one of them is describing the
    other's bytes, and summing them double-charges the card while overwriting
    them loses a claim.
    """
    out = list(base)
    seen = {t.name for t in out}
    for t in extra:
        if t.name in seen:
            raise LedgerError(
                f"two sources both charge {t.name!r} on this card. Rename one "
                "or fold them into a single term; a coresident sum may not "
                "guess which claim is real."
            )
        seen.add(t.name)
        out.append(t)
    return tuple(out)


def validate_all(ledgers: Sequence[CardVramLedger]) -> None:
    """Refuse the whole boot when ANY card is infeasible, printing every
    infeasible card at once.

    One card at a time would make an operator fix three cards in three boots.
    """
    bad = [x for x in ledgers if not x.fits]
    if bad:
        raise LedgerOvercommit(bad)
    for x in ledgers:
        x.validate()


def render_all(ledgers: Sequence[CardVramLedger]) -> str:
    return "\n\n".join(x.render() for x in ledgers)


def summarize(ledgers: Sequence[CardVramLedger]) -> Dict[str, int]:
    """Rig totals, for a dashboard row."""
    return {
        "total_mib": sum(x.total_mib for x in ledgers),
        "user_reserve_mib": sum(x.user_reserve_mib for x in ledgers),
        "demand_mib": sum(x.demand_mib for x in ledgers),
        "kv_pool_mib": sum(x.kv_pool_mib for x in ledgers),
    }


def terms_from_posts(
    posts: Mapping[str, int],
    *,
    tenant: str,
    derivation: str,
) -> Tuple[LedgerTerm, ...]:
    """Adapt a tenant's ``{post name: bytes}`` map to DECLARED ledger terms.

    The bridge from :class:`sglang.srt.registry.spec.ResourceProfile` (which
    speaks bytes per post) into this ledger (which speaks MiB per term), so the
    registry keeps its own vocabulary and the ledger keeps one row shape.
    """
    return tuple(
        LedgerTerm(
            name=f"{tenant}: {post}",
            mib=int(byte_count) // (1 << 20),
            provenance=Provenance.DECLARED,
            derivation=derivation,
            tenant=tenant,
        )
        for post, byte_count in sorted(posts.items())
    )
