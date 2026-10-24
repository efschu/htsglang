# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-BUDGET-REST: an awake group's MEASURED rest beyond its budget line, as ONE post.

THE DEFECT (27B inventory 29.09., ``LEISTUNGSSCHALTER-INVENTAR-0929.md`` R1-R3).
Group D's budget on the 27B booked, per card, three named constants above the
measured transient of its corridor floor (767 MiB on the 5090)::

    user reserve 1800 / 1400 / 1400      (--user-reserve-mib, 27b.env)
    awake_overshoot 404                  (launcher constant, boot weg2onebackup2, 07.09.)
    measured_awake_overshoot 489 / 0 / 0 (D_OVERSHOOT_MIB, boot weg2ls4b1, 07.09.)

The law (users 19.09. and 29.09.) is: no reserve, only measured transients.
The same boots that carried those constants MEASURED what an awake D actually
holds on each card beyond its budget line: ``PDFLIP-VRAM-PEAK`` (every D rank,
at every chunk / round / flip / idle event) prints ``card_free_mib`` and
``card_total_mib`` (the torch-visible card: NVML total minus the driver carve).
At the tightest measured instant of a boot::

    rest = card_total_mib - min(card_free_mib) - budget - dormant_other

where ``budget`` and ``dormant_other`` are the terms of that boot's own
``budget D group=D`` line (the real pass). ``rest`` is everything on the card
that the budget line did not name: the served transient, the allocator cache
above the allocations, the non-torch part of the D process (context, BAR1
windows), the other group's served dormant growth. It includes the corridor
floor's transient -- so the floor is NOT charged beside it, and neither is any
reserve, the builtin 404 or a budget-relative overshoot: one measured post
replaces all of them (one would count the other twice).

MEASURED 29.09. on the eight newest 27B row-authority boots (INT8, TP3, bs6
capture set): the D ranks left 273 / 4 / 14 MiB free at their tightest
instant (5090 / nvml0 / nvml2). The "reserve" was never free VRAM -- the
awake D consumed it. The rest is 3191 / 2079 / 2067 MiB, so the D budget
rises by only what was really left over (see the report of desk/27b-no-
reserve-0929), and the names on the budget line become the truth: one
measured rest instead of three constants.

FIXPOINT. The rest is measured against the budget line, not against the form:
it is budget-independent exactly while the budget BINDS the pool (the 27B D:
its KV pool is sized from the budget, 563744+ tokens, far above the 262144
context). A group whose pool is capped below its budget (27B group P: the
P-CUT cap ``--max-total-tokens``) shows a NEGATIVE rest on some card -- there
the measurement cannot price the rest (it would credit free VRAM the pool
never takes) and the card stays UNMEASURED, by name.

``python -m flliper.srt.pdflip.budget_rest --group D <front.log> ...`` prints the
record row for ``pdflip/profile_records_data/<profile>.json``: per card ordinal
the MAXIMUM over the boots given (the newest N of the form, like
``launcher.dormant_max_from_records``), ``null`` where no boot priced it.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MARKER = "PDFLIP-BUDGET-REST"
#: the record row per group (``<G>_AWAKE_REST_BOOKED_MIB``): per card ordinal
RECORD_FMT = "{group}_AWAKE_REST_BOOKED_MIB"
#: the explicit off switch (the registry row is the default)
ENV = "FLLIPER_PDFLIP_BUDGET_REST_RECORD"

SOURCE_RECORD = "RECORD"
SOURCE_UNMEASURED = "UNMEASURED"


def record_name(group: str) -> str:
    return RECORD_FMT.format(group=str(group).upper())


_BUDGET = re.compile(
    r"PDFLIP-LAUNCH budget (?P<label>P|D) group=(?P<group>P|D) ordinal=(?P<ord>\d+) "
    r"nvml_idx=(?P<nvml>\d+) [^:]*: (?P<budget>-?\d+) MiB = total (?P<total>\d+) "
    r".*?dormant_other (?P<dormant>\d+)"
)
_PEAK = re.compile(r"PDFLIP-VRAM-PEAK rank=(?P<rank>\d+) ")
_FIELD = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class CardRest:
    """One card of one boot: the budget line's terms and the tightest instant."""

    ordinal: int
    nvml: int
    budget: int
    dormant_other: int
    card_total: int
    min_free: int
    phase: str

    @property
    def rest(self) -> int:
        return self.card_total - self.min_free - self.budget - self.dormant_other


def budget_lines(front_text: str, group: str) -> Dict[int, Tuple[int, int, int]]:
    """ordinal -> (nvml, budget, dormant_other) of the REAL pass of ``group``
    (label == group, i.e. not ``D(dry, expectation)`` / ``D(Karte, ...)``).
    The LAST such line per ordinal wins (a re-planned pass replaces the first)."""
    out: Dict[int, Tuple[int, int, int]] = {}
    for line in front_text.splitlines():
        m = _BUDGET.search(line)
        if not m or m.group("group") != group or m.group("label") != group:
            continue
        out[int(m.group("ord"))] = (int(m.group("nvml")), int(m.group("budget")),
                                    int(m.group("dormant")))
    return out


def tightest(group_text: str) -> Dict[int, Tuple[int, int, str]]:
    """rank -> (min card_free_mib, card_total_mib, phase) over every
    ``PDFLIP-VRAM-PEAK`` line of the group's log. Lines without a numeric
    ``card_free_mib`` are skipped (``na``), never read as zero."""
    out: Dict[int, Tuple[int, int, str]] = {}
    for line in group_text.splitlines():
        m = _PEAK.search(line)
        if not m:
            continue
        f = dict(_FIELD.findall(line[m.start():]))
        try:
            free, total, rank = int(f["card_free_mib"]), int(f["card_total_mib"]), int(f["rank"])
        except (KeyError, ValueError):
            continue
        if rank not in out or free < out[rank][0]:
            out[rank] = (free, total, str(f.get("phase", "?")))
    return out


def boot_rests(front_text: str, group_text: str, group: str) -> Dict[int, CardRest]:
    """Per card ordinal the rest of ONE boot; a card whose budget line or
    whose rank's peak line is missing is left out (not priced, never zero).
    D ranks are card ordinals (TP r on ordinal r); a P stage is too (PP s)."""
    lines = budget_lines(front_text, group)
    peaks = tightest(group_text)
    out: Dict[int, CardRest] = {}
    for o, (nvml, budget, dormant) in lines.items():
        if o not in peaks:
            continue
        free, total, phase = peaks[o]
        out[o] = CardRest(ordinal=o, nvml=nvml, budget=budget, dormant_other=dormant,
                          card_total=total, min_free=free, phase=phase)
    return out


def record_from_boots(
    boots: Sequence[Tuple[str, str, str]], group: str, n_cards: int,
) -> Tuple[List[Optional[int]], List[str]]:
    """``boots`` = (tag, front text, group text), newest first. Per card ordinal
    the MAXIMUM rest over the boots that priced it; ``None`` where no boot did
    or where the maximum is NEGATIVE (the budget does not bind there -- a
    negative rest would credit VRAM the pool never takes: UNMEASURED)."""
    best: List[Optional[int]] = [None] * int(n_cards)
    lines: List[str] = []
    for tag, front, grp in boots:
        for o, c in sorted(boot_rests(front, grp, group).items()):
            if o >= n_cards:
                continue
            best[o] = c.rest if best[o] is None else max(best[o], c.rest)
            lines.append(
                f"{MARKER} {tag} group={group} ordinal={o} nvml{c.nvml}: rest {c.rest} = "
                f"card_total {c.card_total} - min card_free {c.min_free} ({c.phase}) - "
                f"budget {c.budget} - dormant_other {c.dormant_other} MiB")
    out = [None if v is None or v < 0 else int(v) for v in best]
    for o, v in enumerate(best):
        if v is not None and v < 0:
            lines.append(f"{MARKER} group={group} ordinal={o}: max rest {v} < 0 -- the budget "
                         f"does not bind this card (pool capped below it): UNMEASURED")
    return out, lines


@dataclass(frozen=True)
class Resolved:
    """The booked awake rest of one group, per card ordinal (``None`` =
    UNMEASURED: the caller books its legacy terms there, named)."""

    values: Tuple[Optional[int], ...]
    provenance: str

    def source(self, i: int) -> str:
        return SOURCE_RECORD if i < len(self.values) and self.values[i] is not None else SOURCE_UNMEASURED


def switch_on(row_default: bool, environ: Optional[Mapping[str, str]] = None) -> bool:
    """The registry row's ``budget_rest_from_records`` unless :data:`ENV` is set."""
    env = os.environ if environ is None else environ
    raw = str(env.get(ENV, "") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(row_default)


def resolve(values: Optional[Sequence[Optional[int]]], boots: str, n_cards: int,
            group: str, name: Optional[str] = None) -> Optional[Resolved]:
    """The record's values for ``n_cards`` cards, or ``None`` when the
    profile carries no record for ``group`` (every card UNMEASURED; the
    caller's budgets stay byte-identical). A record of another card count is
    REFUSED by raising -- a partial vector must not price a card it never saw."""
    if values is None:
        return None
    vals = list(values)
    if len(vals) != int(n_cards):
        raise ValueError(f"{name or record_name(group)} has {len(vals)} entries for {n_cards} "
                         f"cards; re-measure with python -m flliper.srt.pdflip.budget_rest")
    return Resolved(values=tuple(None if v is None else int(v) for v in vals),
                    provenance=f"{name or record_name(group)} {boots}".strip())


# --------------------------------------------------------------------------
# PDFLIP-VOTE-LINE-BOOK (desk/27b-int8-tp1corridor-1024): the rest is
# measured UNDER A VOTE LINE, and that line has moved since.
#
# The rest above is ``card_total - min card_free - budget - dormant_other`` at
# the tightest instant of its boots. That instant is the extend chunk the
# width vote (extend_trim.width_vote) funded down to ITS line: the eight
# boots of 28./29.09. ran rc12g, whose line was the 300 MiB halt line
# (extend_trim.CHUNKING_FLOOR_MIB) -- the 3080s ended at 4 / 14 MiB free. On
# 21.10. (786fe038dc) the vote's line became the corridor law and on 22.10.
# (NACHT2 port) the card's MEASURED floor (767 / 701 / 700 on the 27B D): the
# vote now keeps ``line - 300`` MiB more free at exactly that instant. The
# record does not contain that room (it was measured without it): the booked
# rest UNDER-books the current regime, the budget gave the room to the KV pool,
# and the vote takes it out of the chunk. (Review 24.10.: no double booking in
# the budget is resolved here -- the overlap of the line with rate x rows inside
# the vote is the root, ticket S4.) The delta is DERIVED, not measured.
#
# METAL (boot boot_weg2_dkr27bbar1drei10240803_b9cddabda1_1024_080355, rc39a,
# D.log 08:16:36): TP1 (3080, budget 17384 = 20480 - 425 - 592 - 2079) post
# 790 / 734 MiB after the trim, line 701, rate 0.3091 -> cap 287 / 105 rows
# (#794 GROUP-NARROWED 4096 -> 287 / 256); 741 GROUP-NARROWED lines, mode
# 256. The card rested at the record's served-transient room (~750 MiB) and
# the vote kept 701 of it as line.
#
# THE FIX books the moved line beside the rest, per card,
#
#     delta = ceil8(max(0, line_now - record_line))   line_now = extend_trim.line_for_floor_mib(floor)
#
# (ceil8: up to the budget's 8-MiB grain, so budget + rest -- the torch cap -- stays
# byte-identical, review A2). Only where the floor is MEASURED
# (corridor_guard.CorridorFloor.measured) AND its footprint is of THIS form (the
# store's card set and TP equal this pass's cards, review S3): an unmeasured
# card's vote line is the stated law 1024 and UNMEASURED-FALLBACK is verdict-only
# -- it never spends KV pool (corridor_guard #1257c); a hand-set law
# (ENV-OVERRIDE) is a moved TARGET with its +-20 % band, not a measured
# requirement; a footprint of another form (the launcher reads the LAST
# published D digest) is not the line this boot's ranks keep. The floor is read
# with reserve 0 (the booked path charges no user reserve; a reserve passed here
# would be a forbidden corridor reserve, keine-korridor-reserve-nie).
#
# No reserve: the delta is the vote's own line, which the vote enforces anyway;
# it moves those bytes from the chunk to the KV pool's account. The PREDICTED
# end state at the tightest instant is unchanged -- the vote still ends at its
# line: rows_cap(post + delta, rate, page, line_now) == rows_cap(post, rate,
# page, record_line) -- the metal check is A1. The delta is DERIVED (code state
# of the record's boots vs today's measured floor), never a measurement; a
# record re-measured with budget_rest under the current line switches it off
# (its boots differ).
#
# A record re-measured under another line (its boots differ from the ones
# named here) books no delta and says so: the regime of an unknown boot set
# is not guessed.
# --------------------------------------------------------------------------

VOTE_LINE_MARKER = "PDFLIP-VOTE-LINE-BOOK"
#: the D budget's rounding grain (launcher.budgets_from_dc: ``// 8 * 8``)
BUDGET_GRAIN_MIB = 8
#: review A3 (HOCHRECHNUNG != MESSUNG): the delta is DERIVED -- the code state of the record's
#: boots (line 300) against today's measured floor -- never a measurement of VRAM.
VOTE_LINE_BASIS = "DERIVED (record boots' code line vs measured floor; not a measurement)"
#: the KV-vs-width choice (user law 30.08.: a KV<->performance trade is selectable). Default ON
#: (user order 14.10.: new features on); =0 = the pre-fix budgets: the KV pool keeps the vote's
#: room and the vote narrows the chunk (rc39a: 4096 -> 256..691 rows on the 27B D).
VOTE_LINE_ENV = "FLLIPER_PDFLIP_VOTE_LINE_BOOK"


def vote_line_switch_on(environ: Optional[Mapping[str, str]] = None) -> bool:
    """:data:`VOTE_LINE_ENV` unless it says off (0/false/no/off); default on."""
    env = os.environ if environ is None else environ
    return str(env.get(VOTE_LINE_ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")

#: the eight row-authority boots of D_AWAKE_REST_BOOKED_MIB / D_AWAKE_REST_CAPPED_MIB (28./29.09.)
BOOTS_0929: Tuple[str, ...] = (
    "dkr27browauthoritybar1fs09291152",
    "dkr27browauthoritybar1fs09290956",
    "dkr27browauthoritybar1w109290020",
    "dkr27browauthoritybar1w109290002",
    "dkr27browauthoritybar1w109281851",
    "dkr27browauthoritybar1w109281715",
    "dkr27browauthoritybar1w109281556",
    "dkr27browauthoritybar1w109281300",
)

#: record name -> (the boots it was measured on, the extend width vote's line in force on them).
#: Not a VRAM measurement: the code state of those boots (rc12g ``CHUNKING_FLOOR_MIB``; the
#: line moved in 786fe038dc 21.10. and the NACHT2 port 22.10.).
RECORD_VOTE_LINE_MIB: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "D_AWAKE_REST_BOOKED_MIB": (BOOTS_0929, 300),
    "D_AWAKE_REST_CAPPED_MIB": (BOOTS_0929, 300),
}


def record_vote_line(name: str, boots: Sequence[str]) -> Optional[int]:
    """The vote line the record ``name`` was measured under, or ``None`` when its
    boots are not the ones named in :data:`RECORD_VOTE_LINE_MIB` (unknown regime)."""
    ent = RECORD_VOTE_LINE_MIB.get(str(name))
    if ent is None or tuple(boots or ()) != tuple(ent[0]):
        return None
    return int(ent[1])


def footprint_form_mismatch(store: Mapping, uuids: Sequence[str]) -> Optional[str]:
    """Review S3: ``None`` when the phase-footprint store behind a measured floor is of THIS
    form -- its card set equals ``uuids`` and its profile's TP x PP (fields 2, 3 of
    ``profile``) equals the card count -- else the reason. The launcher reads the LAST
    published D digest; a footprint of another form (a duo boot after a drei boot) is not
    the line this boot's ranks will keep. Pure."""
    cards = sorted(str(k) for k in (store.get("cards") or {}))
    want = sorted(str(u) for u in uuids)
    if cards != want:
        return f"footprint cards {cards} != this pass's cards {want}"
    prof = list(store.get("profile") or [])
    if len(prof) >= 4:
        try:
            tp_pp = int(prof[2]) * int(prof[3])
        except (TypeError, ValueError):
            return f"footprint profile {prof!r} unreadable"
        if tp_pp != len(want):
            return f"footprint profile tp x pp {tp_pp} != {len(want)} cards"
    return None


def vote_line_deltas(rests: Sequence[Optional[int]], lines_now: Sequence[float],
                     measured: Sequence[bool], record_line: Optional[int]) -> List[int]:
    """Per card ``max(0, ceil(line_now) - record_line)`` rounded UP to :data:`BUDGET_GRAIN_MIB`
    -- 0 where the card has no booked rest, where its floor is not measured (or not of
    this form), or where the record's line is unknown. Pure. DERIVED, not measured."""
    if not (len(rests) == len(lines_now) == len(measured)):
        raise ValueError(f"{VOTE_LINE_MARKER}: {len(rests)} rests, {len(lines_now)} lines, "
                         f"{len(measured)} floors -- a partial vector never prices a card it never saw")
    out: List[int] = []
    for r, ln, act in zip(rests, lines_now, measured):
        if r is None or record_line is None or not bool(act):
            out.append(0)
            continue
        raw = max(0, int(-(-float(ln) // 1)) - int(record_line))
        # review A2: UP to the budget's 8-MiB grain -- the budget is ``(... - rest) // 8 * 8``, so
        # with a multiple of 8 the budget falls by exactly the delta and budget + rest (the torch
        # cap, budget + rest - OTHER) stays byte-identical; an unrounded delta could raise it by
        # up to 7 MiB (the OOM direction).
        out.append(-(-raw // BUDGET_GRAIN_MIB) * BUDGET_GRAIN_MIB)
    return out


# --------------------------------------------------------------------------
# VOTE-LINE-SINGLE (desk/27b-vote-line-single-1024, root ticket S4 of the VOTE-LINE-BOOK review 24.10.): the line
# the delta above books is no longer the measured floor but ONE honest line,
#
#     line = corridor_guard.NEAR_OOM_MIB (400) + D_EXTEND_VOTE_FIXED_MIB (measured)      extend_trim.line_for_floor_mib
#
# The floor (767 / 701 / 700 on the 27B D) is the 4096-row activation peak, which ``rate x rows`` prices already --
# held as line too, it was counted twice and VOTE-LINE-BOOK made the KV pool pay for the second copy. What rate x rows
# does NOT pay is measured here from a boot's own D log (:func:`extend_passes`, :func:`vote_fixed_from_passes`):
# per extend, the forward phase from the vote's post down to the chunk's end free (reserved is monotone between
# trims; the end is a boundary read, not a continuous minimum, and the draft extend of the same pass falls into the
# next window -- named, A1 checks it on the 100 ms trace), minus rate x rows = the admission rest the forward trim
# could not return plus the forward's own fixed part.
#
# MEASURED on the six INT8-drei boots whose vote logged DRIVER free as post (041a871f4b, 0798c91fcd, a4f4b8fec1 x3,
# b9cddabda1; the 121e4bf3f2..a4f4b8fec1 boots logged driver + cache and are left out by name): 618 passes with a
# forward trim, max 238 MiB (boot _1022_003738 TP1 00:42:45, a 6-request post-wake handback, 61 rows, admission 508,
# trim back 278, end 828). The rest grows with the number of handback requests (bs <= 6, the capture set), _080355
# alone said 155. -> line 638, delta ceil8(638 - 300) = 344 per card (was 472 / 408 / 400): KV back 128 / 64 / 56 MiB.
# IN-SAMPLE: the record is the max over exactly these passes, a regression lock, not a bound -- the A1 boot is the
# first out-of-sample check (``--vote-fixed`` <= record, else the record is raised before the next boot).
#
# The admission PEAK before the forward trim is not a line post (it is released before the forward and no width
# removes it). Its condition ``post - card_free_before >= 400`` is NOT structurally secured: the admission runs before
# the forward-need trim, and before it only the vote's STATIC trim acts (TP2 800 -> structurally 800 - 468 = 332).
# Measured on INT8 drei min 542 (six boots); on int4ar already violated (376 / 390, pre-existing, own ticket).
# -> PDFLIP-ADMIT-BOOK below prices it (desk/27b-admit-trim-1024; the chunk-window census found int4ar down to 216).
#
# FORM GATE (review M1 24.10.): the record holds only for the forms of its boots (:data:`VOTE_FIXED_FORMS`: checkpoint
# and D card inventory). Other forms of the same profile measured 223-252 (int4ar, NVFP4 duo) -- there the floor line
# stays, byte for byte, and the launcher names it.
#
# The value travels as the profile record ``D_EXTEND_VOTE_FIXED_MIB`` (launcher -> rank env
# ``FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB``), the budget reads the same record: rank line and booked line are one
# formula. ``FLLIPER_PDFLIP_VOTE_LINE_SINGLE=0`` = the floor line again (both sides; default ON, user order 14.10.).
# --------------------------------------------------------------------------

#: the record's boots (D logs with driver-free post, INT8 drei) and the forms they ran -- (checkpoint key, D card
#: inventory). A record re-measured on other boots books nothing until its forms are named here (never guessed).
BOOTS_VOTE_FIXED: Tuple[str, ...] = (
    "dkr27bbar1drei10210018",
    "dkr27bbar1drei10210043",
    "dkr27bbar1drei10220037",
    "dkr27bbar1drei10220104",
    "dkr27bbar1drei10220153",
    "dkr27bbar1drei10240803",
)
VOTE_FIXED_FORMS: Dict[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    BOOTS_VOTE_FIXED: (("Qwen3.8-27B-INT8-gdncov-vocabembed", ("RTX5090", "RTX3080", "RTX3080")),),
}


def vote_fixed_form_mismatch(boots: Sequence[str], model_key: str, inventory: Sequence[str]) -> Optional[str]:
    """Review M1: ``None`` when (``model_key``, ``inventory``) is a form the record's ``boots`` ran, else why the
    measured fixed part does not hold here (the floor line then stays). Pure."""
    forms = VOTE_FIXED_FORMS.get(tuple(boots or ()))
    if forms is None:
        return f"the record's boots {','.join(boots or ()) or '-'} name no form"
    want = (str(model_key or ""), tuple(inventory or ()))
    if want in forms:
        return None
    return (f"F unmeasured for form {want[0] or '?'} on {','.join(want[1]) or '?'} (measured on "
            + "; ".join(f"{m} on {','.join(i)}" for m, i in forms) + ")")


VOTE_FIXED_RECORD = "D_EXTEND_VOTE_FIXED_MIB"
VOTE_FIXED_MARKER = "VOTE-LINE-SINGLE"
#: rank env the launcher writes from :data:`VOTE_FIXED_RECORD` (extend_trim.vote_fixed reads it)
VOTE_FIXED_ENV = "FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB"
#: the switch (launcher process): =0 writes no rank env and books the floor line (b130e9dab4 behaviour)
VOTE_LINE_SINGLE_ENV = "FLLIPER_PDFLIP_VOTE_LINE_SINGLE"
#: the measurement's denominator: passes whose forward ran after a trim (the regime the forward-need trim forces)
VOTE_FIXED_BASIS = ("MEASURED (forward phase: vote post - chunk-end free - rate x rows over the extends whose forward "
                    "ran after a trim; admission peak and the boot's first extend excluded by name)")


def vote_line_single_on(environ: Optional[Mapping[str, str]] = None) -> bool:
    """:data:`VOTE_LINE_SINGLE_ENV` unless it says off (0/false/no/off); default on."""
    env = os.environ if environ is None else environ
    return str(env.get(VOTE_LINE_SINGLE_ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class ExtendPass:
    """One width vote paired with the chunk it funded (one D rank).

    ``post`` the vote's driver free (``PDFLIP-EXTEND-STUECKELUNG post=``), ``rate`` its priced rate, ``line`` its line
    (``floor=``), ``cap`` its cap; ``rows`` the chunk's new rows and ``end`` its card free at the window's end
    (``PDFLIP-VRAM-PEAK phase=chunk``); ``fwd_before`` / ``released`` the LAST trim between them (the forward trim:
    driver free after the admission, bytes it returned), ``None`` / 0 when the forward ran without a trim."""

    rank: int
    vote_ts: str
    chunk_ts: str
    post: float
    rate: float
    line: float
    cap: int
    rows: int
    end: float
    fwd_before: Optional[float]
    released: float

    @property
    def trimmed(self) -> bool:
        return self.fwd_before is not None

    @property
    def admission(self) -> Optional[float]:
        """The admission peak ``post - card_free_before`` of the forward trim (``None`` without one)."""
        return None if self.fwd_before is None else self.post - float(self.fwd_before)

    @property
    def excess(self) -> float:
        """Forward phase above the priced rows: ``post - end - rate x rows`` (MiB)."""
        return self.post - self.end - self.rate * self.rows


_RX_RANK = re.compile(r"^\[(\S+ \S+) TP(\d+)\] ")
_RX_KV = re.compile(r"(\w+)=([-\w.]+)")


def extend_passes(group_text: str) -> Dict[int, List[ExtendPass]]:
    """Every logged width vote of a D log paired with the next chunk window of the same rank, in log order. A
    ``phase=round`` / ``phase=flip`` window between them unpairs the vote (another pass ran in between); a vote
    that returned no cap logs nothing and is never paired. Pure (text in, passes out)."""
    out: Dict[int, List[ExtendPass]] = {}
    pend: Dict[int, Tuple[str, Dict[str, str]]] = {}
    trims: Dict[int, List[Dict[str, str]]] = {}
    # review admit-trim M1 (24.10.): a round window closed by the extend itself at its forward's end (round
    # t_unix_ms == the next chunk's t0_unix_ms, nothing in between) does not unpair -- held here until it is known
    rounds: Dict[int, Optional[int]] = {}
    for ln in group_text.splitlines():
        if "PDFLIP-EXTEND-STUECKELUNG" in ln:
            kind = "vote"
        elif "PDFLIP-EXTEND-CACHE-TRIM" in ln:
            kind = "trim"
        elif "PDFLIP-VRAM-PEAK" in ln:
            kind = "chunk" if "phase=chunk" in ln else ("round" if "phase=round" in ln else
                                                       ("other" if "phase=flip" in ln else ""))
            if not kind:
                continue
        else:
            continue
        m = _RX_RANK.match(ln)
        if not m:
            continue
        ts, r = m.group(1), int(m.group(2))
        d = dict(_RX_KV.findall(ln))
        if kind == "round":
            if r in rounds:          # two rounds in a row: the first was self-closed
                pend.pop(r, None)
                trims.pop(r, None)
            rounds[r] = _window_t(d, "t_unix_ms")
            continue
        if r in rounds:
            rt = rounds.pop(r)
            if not (kind == "chunk" and rt is not None and _window_t(d, "t0_unix_ms") == rt):
                pend.pop(r, None)    # a self-closed round between the vote and this event: unpaired
                trims.pop(r, None)
        if kind == "vote":
            pend[r] = (ts, d)
            trims[r] = []
        elif kind == "trim":
            if r in pend:
                trims.setdefault(r, []).append(d)
        elif kind == "other":
            pend.pop(r, None)
            trims.pop(r, None)
        else:
            v = pend.pop(r, None)
            tr = trims.pop(r, [])
            if v is None:
                continue
            vts, vd = v
            try:
                last = tr[-1] if tr else None
                out.setdefault(r, []).append(ExtendPass(
                    rank=r, vote_ts=vts, chunk_ts=ts, post=float(vd["post"]), rate=float(vd["rate"]),
                    line=float(vd["floor"]), cap=int(vd["cap"]), rows=int(d["rows"]), end=float(d["card_free_mib"]),
                    fwd_before=None if last is None else float(last["card_free_before"]),
                    released=0.0 if last is None else float(last["released"])))
            except (KeyError, ValueError):
                continue
    return out


def vote_fixed_from_passes(passes: Mapping[int, Sequence[ExtendPass]]) -> Tuple[Optional[int], List[str]]:
    """``D_EXTEND_VOTE_FIXED_MIB`` from paired passes: ``ceil(max excess)`` over every pass whose forward ran after
    a trim, the boot's FIRST pass per rank excluded (its growth is resident -- first-touch workspaces, the next vote's
    post is lower by it -- and the booked rest carries it). One value for every rank (TP-symmetric shards; the max).
    ``None`` without a trimmed pass. Lines name the worst pass, the admission peak and the untrimmed passes."""
    lines: List[str] = []
    worst: Optional[ExtendPass] = None
    for r in sorted(passes):
        ps = list(passes[r])
        if not ps:
            continue
        first = ps[0]
        lines.append(f"{VOTE_FIXED_MARKER} rank={r} first pass {first.vote_ts} excluded: excess {first.excess:.0f} "
                     f"MiB, resident growth {first.post - (ps[1].post if len(ps) > 1 else first.post):.0f} MiB "
                     f"(next vote's post lower by it)")
        tr = [p for p in ps[1:] if p.trimmed]
        nt = [p for p in ps[1:] if not p.trimmed]
        if tr:
            w = max(tr, key=lambda p: p.excess)
            adm = min(p.fwd_before for p in tr)  # type: ignore[type-var]
            lines.append(f"{VOTE_FIXED_MARKER} rank={r} trimmed={len(tr)} max excess {w.excess:.1f} MiB at "
                         f"{w.vote_ts} rows={w.rows} post={w.post:.0f} end={w.end:.0f}; admission peak max "
                         f"{max(p.admission for p in tr):.0f} MiB, min post - admission {adm:.0f} MiB "  # type: ignore
                         f"(should be >= 400; not structurally secured)")
            if worst is None or w.excess > worst.excess:
                worst = w
        if nt:
            wn = max(nt, key=lambda p: p.excess)
            lines.append(f"{VOTE_FIXED_MARKER} rank={r} untrimmed={len(nt)} max excess {wn.excess:.1f} MiB at "
                         f"{wn.vote_ts} rows={wn.rows} (admission held in the cache over the forward: the "
                         f"forward-need trim releases it under line + rate x rows; not the line's post)")
    if worst is None:
        return None, lines
    val = int(math.ceil(max(0.0, worst.excess) - 1e-9))
    lines.append(f"{VOTE_FIXED_MARKER} {VOTE_FIXED_RECORD} = {val} MiB (rank {worst.rank} {worst.vote_ts}, "
                 f"{VOTE_FIXED_BASIS})")
    return val, lines


# --------------------------------------------------------------------------
# PDFLIP-ADMIT-BOOK (desk/27b-admit-trim-1024, Auftrag #212, review-vote-line-single M2 / SOLL S1; review admit-trim
# M1-M4 24.10.): the D extend's ADMISSION PEAK, priced as a budget post.
#
# Every D extend pass runs, on each rank,
#
#     width vote (trim under the static threshold, read post) -> admission (post-wake handbacks: Mamba restore and
#     KV load-back, PDFLIP-HANDBACK / #988 LOADBACK) -> forward trim (card_free_before = driver free AFTER the
#     admission, released = the admission's fresh segments) -> forward
#
# so the card's driver free at the admission instant is ``post - A`` and the near-OOM law asks ``post - A >= 400``.
# The admission peak A grows with the number of handback requests (capture set bs <= 6, DEFAULT_D_BS = max_running = 6)
# and with the summed load-back depth (correlation 0.74 on int4ar), and saturates at 504..522 MiB from 3-4 handbacks
# on int4ar, INT8 drei and NVFP4 duo alike (load-backs measured up to 136k tokens deep; deeper ones are unmeasured).
#
# MEASURED on six int4ar serving boots (chunk-window census :func:`admission_passes`, vote-log independent, with the
# pass boundary of review M1): 995 admission passes, 57 admission instants under 400 (TP1 45, TP0 12), minimum
# 32 MiB (``_1023_223241`` 23:19:34 TP1, TP0 222). Two classes:
#   - the vote trim RAN before the admission (47 of the 57, it returned <= 204 MiB): the remaining 650-730 MiB of
#     allocator cache sit in segments with live blocks, the admission takes fresh segments -- only the budget helps;
#   - the vote did NOT trim (148 passes, 10 of the 57, among them 32 and 222): ``armed`` was left over from passes
#     whose admission was refused (568 refusals in 174 s before the 32 MiB instant), decode rounds refilled the cache
#     and the forward trim then returned up to 630 MiB. ``extend_trim.before_admission`` trims there now.
# (The vote-log pairing of extend_passes saw 376 / 390, the first census of this branch 216: the vote line is logged
# only when the cap changes, and a round window closed by the extend itself was read as a pass boundary.)
#
# THE PRICE (no reserve -- a derived bound from two measured maxima):
#
#     need = near-OOM 400 + A_peak - post_floor
#
#     A_peak      max admission peak over the 27B handback path (all forms: same model geometry, D TP2/TP3 mamba +
#                 KV load-back), 522 MiB (NVFP4 duo, 5 handbacks)
#     post_floor  the lowest post (driver free after the vote trim) of ANY pass on the record's form -- an admission is
#                 not structurally tied to passes with a high post (review M2) -- under the budget its boots ran (rest
#                 record D_AWAKE_REST_BOOKED_MIB 2079 borrowed from the INT8 3080 twin, vote-line delta 0)
#
# booked beside the rest as ``max(vote-line delta, ceil8(need))`` -- the admission and the forward never coexist
# (the forward trim releases the admission's segments before the forward; an untrimmed forward runs at post - A
# >= threshold >= the vote line), so the larger of the two posts is the card's need, never their sum. One value for
# every rank (TP-symmetric shards, the max). FORM GATE as VOTE-LINE-SINGLE: the record holds only for its boots'
# form (:data:`ADMIT_NEED_FORMS`), and only beside the rest it was derived under (:data:`ADMIT_REST_BASIS`; another
# rest is named loudly and asks for a re-measurement). ``FLLIPER_PDFLIP_ADMIT_BOOK=0`` = the budgets of 9a25171a59
# (default ON).
# --------------------------------------------------------------------------

ADMIT_MARKER = "PDFLIP-ADMIT-BOOK"
ADMIT_RECORD = "D_ADMIT_NEED_MIB"
#: launcher switch: =0 books no admission post (the budgets of 9a25171a59); default on (user order 14.10.)
ADMIT_ENV = "FLLIPER_PDFLIP_ADMIT_BOOK"
ADMIT_BASIS = ("DERIVED from two measured maxima (near-OOM 400 + admission peak over the 27B handback path - lowest "
               "post of any pass on the record's form); not a reserve")
#: a pass "carried an admission" when the forward trim found the card this much below the vote's post (MiB): the
#: smallest post-wake handback measured 58 MiB; below it the difference is decode/round noise of the allocator
ADMIT_MIN_MIB = 50.0
#: the rest the record was derived beside (record name, value on every int4ar card): another rest = re-measure
ADMIT_REST_BASIS: Tuple[str, int] = ("D_AWAKE_REST_BOOKED_MIB", 2079)
#: the record's boots (int4ar serving boots, 23./24.10.) and their form
BOOTS_ADMIT_NEED: Tuple[str, ...] = (
    "dkr27bint4arbar1int4ar10231552",
    "dkr27bint4arbar1int4ar10231723",
    "dkr27bint4arbar1int4ar10232111",
    "dkr27bint4arbar1int4ar10232232",
    "dkr27bint4arbar1int4ar10240037",
    "dkr27bint4arrc4bar1int4ar10240914",
)
ADMIT_NEED_FORMS: Dict[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    BOOTS_ADMIT_NEED: (("Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct", ("RTX3080", "RTX3080")),),
}


def admit_book_on(environ: Optional[Mapping[str, str]] = None) -> bool:
    """:data:`ADMIT_ENV` unless it says off (0/false/no/off); default on."""
    env = os.environ if environ is None else environ
    return str(env.get(ADMIT_ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")


def admit_form_mismatch(boots: Sequence[str], model_key: str, inventory: Sequence[str]) -> Optional[str]:
    """``None`` when (``model_key``, ``inventory``) is a form the admission record's ``boots`` ran, else why the
    measured need does not hold here (nothing is booked). Pure."""
    forms = ADMIT_NEED_FORMS.get(tuple(boots or ()))
    if forms is None:
        return f"the record's boots {','.join(boots or ()) or '-'} name no form"
    want = (str(model_key or ""), tuple(inventory or ()))
    if want in forms:
        return None
    return (f"admission need unmeasured for form {want[0] or '?'} on {','.join(want[1]) or '?'} (measured on "
            + "; ".join(f"{m} on {','.join(i)}" for m, i in forms) + ")")


@dataclass(frozen=True)
class AdmitPass:
    """One D extend pass of one rank, read from its chunk window (``PDFLIP-VRAM-PEAK phase=chunk``) -- independent of
    the width-vote line, which is logged only when the cap CHANGES.

    ``post`` the driver free the admission started from: the vote line's ``post=`` when one was logged in the pass
    (``post_src`` "vote"), else the vote trim's ``card_free_after`` when the pass trimmed twice ("trim"), else
    ``None``. ``fwd_before`` / ``released`` the pass's forward trim (driver free after the admission, the segments it
    returned), ``None`` / 0 without one. ``rows`` / ``end`` the chunk's rows and end free. ``vote_released`` what the
    trim right before the admission returned -- the admission trim (``site=admit``) or the vote's own trim; ``None`` =
    nothing trimmed before the admission (the armed class, review M3)."""

    rank: int
    ts: str
    post: Optional[float]
    post_src: str
    fwd_before: Optional[float]
    released: float
    rows: int
    end: float
    vote_released: Optional[float] = None

    @property
    def admission(self) -> Optional[float]:
        """``post - fwd_before`` (MiB), ``None`` when either reading is missing."""
        if self.post is None or self.fwd_before is None:
            return None
        return float(self.post) - float(self.fwd_before)


def _window_t(d: Mapping[str, str], key: str) -> Optional[int]:
    try:
        return int(d[key])
    except (KeyError, ValueError, TypeError):
        return None


def admission_passes(group_text: str) -> Dict[int, List[AdmitPass]]:
    """Every chunk window of a D log with the pass that led to it, per rank in log order -- independent of the
    width-vote line, which is logged only when the cap CHANGES.

    PASS BOUNDARIES (review 24.10. M1): a ``phase=flip`` / ``phase=idle`` window and a SELF-closed ``phase=round``
    window (N rounds or 5 s) end a pass. A ``phase=round`` window closed by the extend itself does NOT:
    ``vram_peak_window.on_forward_end`` closes a pending round window at the END of the extend forward, right before the
    extend's own ``phase=chunk`` line (round ``t_unix_ms`` == chunk ``t0_unix_ms``), while that extend's vote trim and
    forward trim were logged at the forward's START, before it. Treating that round line as a boundary dropped the
    trims of every extend that followed decode rounds -- 1980 of 7604 int4ar chunk windows, the serving case, among
    them the 32 MiB admission instant (``_1023_223241`` 23:19:34 TP1).

    INSIDE A PASS: a trim logged with ``site=admit`` (``extend_trim.before_admission``) is the trim right before the
    admission; the last other trim after the last vote / admission trim is the forward trim. ``post`` is, in this
    order of preference, the admission trim's ``card_free_after`` ("admit"), the logged vote's ``post=`` ("vote"), the
    first of two or more trims' ``card_free_after`` ("trim", the vote trim -- the vote trims once per extend), else
    ``None`` (a single trim without a vote is read as the forward trim). Pure (text in, passes out)."""
    out: Dict[int, List[AdmitPass]] = {}
    ev: Dict[int, List[Tuple[str, Dict[str, str]]]] = {}
    for ln in group_text.splitlines():
        if "PDFLIP-EXTEND-STUECKELUNG" in ln:
            kind = "vote"
        elif "PDFLIP-EXTEND-CACHE-TRIM" in ln:
            kind = "trim"
        elif "PDFLIP-VRAM-PEAK" in ln:
            if "phase=chunk" in ln:
                kind = "chunk"
            elif "phase=round" in ln:
                kind = "round"
            elif "phase=flip" in ln or "phase=idle" in ln:
                kind = "other"
            else:
                continue
        else:
            continue
        m = _RX_RANK.match(ln)
        if not m:
            continue
        ts, r = m.group(1), int(m.group(2))
        d = dict(_RX_KV.findall(ln))
        if kind == "trim" and d.get("site") == "admit":
            kind = "atrim"
        if kind in ("vote", "trim", "atrim", "round"):
            ev.setdefault(r, []).append((kind, d))
            continue
        evs = ev.pop(r, [])
        if kind == "other":
            continue
        t0 = _window_t(d, "t0_unix_ms")
        cut = -1
        for i, (k, x) in enumerate(evs):
            if k != "round":
                continue
            if i == len(evs) - 1 and t0 is not None and _window_t(x, "t_unix_ms") == t0:
                continue  # closed by this extend at its forward's end: not a boundary
            cut = i
        evs = [e for e in evs[cut + 1:] if e[0] != "round"]
        try:
            ai = max((i for i, (k, _) in enumerate(evs) if k == "atrim"), default=None)
            vi = max((i for i, (k, _) in enumerate(evs) if k == "vote"), default=None)
            trims = [x for k, x in evs if k == "trim"]
            post: Optional[float] = None
            src = ""
            pre: Optional[Dict[str, str]] = None
            start = max(-1 if ai is None else ai, -1 if vi is None else vi)
            after = [x for k, x in evs[start + 1:] if k == "trim"]
            fwd: Optional[Dict[str, str]] = after[-1] if after else None
            if ai is not None:
                pre = evs[ai][1]
                post, src = float(pre["card_free_after"]), "admit"
            elif vi is not None:
                post, src = float(evs[vi][1]["post"]), "vote"
                before = [x for k, x in evs[:vi] if k == "trim"]
                pre = before[-1] if before else None
            elif len(trims) >= 2:
                post, src = float(trims[0]["card_free_after"]), "trim"
                pre = trims[0]
            out.setdefault(r, []).append(AdmitPass(
                rank=r, ts=ts, post=post, post_src=src,
                fwd_before=None if fwd is None else float(fwd["card_free_before"]),
                released=0.0 if fwd is None else float(fwd["released"]),
                rows=int(d.get("rows", 0) or 0), end=float(d["card_free_mib"]),
                vote_released=None if pre is None else float(pre["released"])))
        except (KeyError, ValueError):
            continue
    return out


def admission_peak(boots: Sequence[Mapping[int, Sequence[AdmitPass]]]) -> Optional[float]:
    """The largest admission peak over every pass of every boot given (MiB), ``None`` without one."""
    vals = [p.admission for b in boots for v in b.values() for p in v if p.admission is not None]
    return max(vals) if vals else None


def admit_need_from_passes(boots: Sequence[Mapping[int, Sequence[AdmitPass]]], near_oom_mib: float,
                           peak_mib: Optional[float] = None) -> Tuple[Optional[int], List[str]]:
    """``D_ADMIT_NEED_MIB`` of the record's form: ``ceil(max over ranks (near_oom + A_peak - post_floor[rank]))``.

    ``post_floor`` is the lowest ``post`` over EVERY pass of ``boots`` whose post is known (review M2 24.10.): an
    admission (handback, #988 load-back) is not structurally tied to passes with a high post -- the lowest posts lie in
    the same minutes as admissions -- so no pass is excluded without a structural reason. ``A_peak`` = ``peak_mib``
    (the 27B handback path over all forms) or, without it, the largest admission of these boots. 0 when the floor
    already carries it; ``None`` without a pass with a known post or without an admission. One value for every rank
    (TP-symmetric shards, the max). Lines name per rank the floor, the lowest admission-pass post (>= ADMIT_MIN_MIB),
    the instants under ``near_oom`` at the boots' budget and the need. Pure; ``near_oom_mib`` is
    ``corridor_guard.NEAR_OOM_MIB`` (the caller passes it: one source)."""
    lines: List[str] = []
    a_peak = admission_peak(boots) if peak_mib is None else float(peak_mib)
    floors: Dict[int, AdmitPass] = {}
    adm_floors: Dict[int, AdmitPass] = {}
    low: Dict[int, List[AdmitPass]] = {}
    for b in boots:
        for r, v in b.items():
            for p in v:
                if p.fwd_before is not None and p.fwd_before < near_oom_mib:
                    low.setdefault(r, []).append(p)
                if p.post is None:
                    continue
                if r not in floors or float(p.post) < float(floors[r].post):  # type: ignore[arg-type]
                    floors[r] = p
                a = p.admission
                if a is not None and a >= ADMIT_MIN_MIB and (
                        r not in adm_floors or float(p.post) < float(adm_floors[r].post)):  # type: ignore[arg-type]
                    adm_floors[r] = p
    if a_peak is None or not floors:
        lines.append(f"{ADMIT_MARKER} no admission pass -- {ADMIT_RECORD} not measured")
        return None, lines
    need = 0.0
    for r in sorted(floors):
        f = floors[r]
        nr = float(near_oom_mib) + a_peak - float(f.post)  # type: ignore[arg-type]
        af = adm_floors.get(r)
        lo = low.get(r, [])
        w = min(lo, key=lambda p: float(p.fwd_before)) if lo else None  # type: ignore[arg-type]
        lines.append(f"{ADMIT_MARKER} rank={r} post_floor {f.post:.0f} MiB ({f.ts}, any pass)"
                     + ("" if af is None else f", lowest admission-pass post {af.post:.0f} ({af.ts})")
                     + f" + peak {a_peak:.0f} -> need {nr:.0f} MiB; {len(lo)} admission instant(s) under "
                     f"{near_oom_mib:.0f} at the boots' budget"
                     + ("" if w is None else f", min {w.fwd_before:.0f} at {w.ts}"))
        need = max(need, nr)
    val = int(math.ceil(max(0.0, need) - 1e-9))
    lines.append(f"{ADMIT_MARKER} {ADMIT_RECORD} = {val} MiB (near-OOM {near_oom_mib:.0f} + peak {a_peak:.0f} - "
                 f"lowest post of any pass; {ADMIT_BASIS})")
    return val, lines


def admit_deltas(rests: Sequence[Optional[int]], vote_deltas: Sequence[int],
                 needs: Optional[Sequence[Optional[float]]]) -> List[int]:
    """Per card ``max(vote-line delta, ceil8(need))`` where the card has a booked rest -- the admission and the
    forward never coexist, so the larger post is the card's, never the sum. ``needs`` ``None`` (no record, foreign
    form, switch off) returns the vote-line deltas unchanged. UP to :data:`BUDGET_GRAIN_MIB` (a multiple of 8 keeps
    budget + rest, the torch cap, byte-identical). Pure."""
    if not (len(rests) == len(vote_deltas)) or (needs is not None and len(needs) != len(rests)):
        raise ValueError(f"{ADMIT_MARKER}: {len(rests)} rests, {len(vote_deltas)} vote deltas, "
                         f"{'-' if needs is None else len(needs)} needs -- a partial vector never prices a card it "
                         f"never saw")
    out: List[int] = []
    for i, (r, d) in enumerate(zip(rests, vote_deltas)):
        n = None if needs is None else needs[i]
        if r is None or n is None:
            out.append(int(d))
            continue
        raw = max(0, int(math.ceil(float(n) - 1e-9)))
        out.append(max(int(d), -(-raw // BUDGET_GRAIN_MIB) * BUDGET_GRAIN_MIB))
    return out


# --------------------------------------------------------------------------
# PDFLIP-ALLOC-OVERHANG (desk/27b-d-alloc-overhang-0929): the rest SPLIT into its
# posts, and the part the torch cache cap (pdflip/torch_cache_cap.py, P0
# 5f33ec18836a) takes away.
#
# At the tightest instant the measured rest is (w109290020 D-TP0, 5090)::
#
#     3191 = over 435 (peak_allocated 27955 - budget 27520: the extend
#                      transient of a 3971-row chunk above the budget)
#          + cache 1325 (peak_reserved 29280 - peak_allocated 27955:
#                        private_free 428 of the graph/tag pools + 897 general)
#          + other 1431 (card_total - card_free - reserved - dormant_other:
#                        D's non-torch 1102 = context/driver/NCCL 846 + lmem
#                        256, PDFLIP-DC-BREAKDOWN; plus P's served dormant growth)
#
# With the cap armed, torch's caching allocator releases its free segments
# before it grows past the cap (set_per_process_memory_fraction), so the
# general cache stops being a post of its own. The capped rest books
#
#     OTHER  max over awake (chunk/round) lines of
#            card_total - card_free - reserved - dormant_other
#     OVER   max over awake lines of peak_allocated - budget (signed: a card
#            whose allocations stay under its budget books that slack)
#     KEEP   private_free + general_cache of the first post-capture
#            PDFLIP-GRAPH-POOL line: what the allocator holds right after the
#            graphs are captured (the private pools' free part, which no
#            empty_cache reaches, and the default pool's residue)
#
# each the maximum over the boots, and the cap per card is the physical line
# ``card_total - driver_carve - dormant_other - growth - OTHER`` (=
# ``budget + OVER + KEEP``): torch may hold its budget, its measured
# overhang and the kept cache, and nothing that belongs to the non-torch part.
# KEEP at post-capture is a LOWER bound of the cache the allocator cannot
# return under load -- the measurement cell prices the real one.
# --------------------------------------------------------------------------

CAPPED_RECORD_FMT = "{group}_AWAKE_REST_CAPPED_MIB"
OTHER_RECORD_FMT = "{group}_TORCH_CAP_OTHER_MIB"
OVERHANG_MARKER = "PDFLIP-ALLOC-OVERHANG"
_GRAPH_POOL = re.compile(r"PDFLIP-GRAPH-POOL rank=(?P<rank>\d+) ")
_AWAKE_PHASES = ("chunk", "round")


def capped_record_name(group: str) -> str:
    return CAPPED_RECORD_FMT.format(group=str(group).upper())


def other_record_name(group: str) -> str:
    return OTHER_RECORD_FMT.format(group=str(group).upper())


@dataclass(frozen=True)
class AwakePosts:
    """One rank of one boot: the maxima over its awake lines and the kept cache."""

    other_total: Optional[int]   # max card_total - card_free - reserved (incl. dormant_other)
    peak_allocated: Optional[int]
    keep: Optional[int]          # post-capture private_free + general_cache


def awake_posts(group_text: str) -> Dict[int, AwakePosts]:
    """rank -> :class:`AwakePosts` from ``PDFLIP-VRAM-PEAK`` lines of the awake
    phases (chunk / round: D serves, every tag is mapped) and the FIRST
    ``PDFLIP-GRAPH-POOL phase=post-capture`` line. ``na`` fields are skipped."""
    other: Dict[int, int] = {}
    peak: Dict[int, int] = {}
    keep: Dict[int, int] = {}
    for line in group_text.splitlines():
        m = _GRAPH_POOL.search(line)
        if m:
            f = dict(_FIELD.findall(line[m.start():]))
            if f.get("phase") != "post-capture":
                continue
            try:
                r = int(f["rank"])
                k = int(f["private_free_mib"]) + int(f["general_cache_mib"])
            except (KeyError, ValueError):
                continue
            keep.setdefault(r, k)
            continue
        m = _PEAK.search(line)
        if not m:
            continue
        f = dict(_FIELD.findall(line[m.start():]))
        if f.get("phase") not in _AWAKE_PHASES:
            continue
        try:
            r = int(f["rank"])
            o = int(f["card_total_mib"]) - int(f["card_free_mib"]) - int(f["reserved_mib"])
            pa = int(f["peak_allocated_mib"])
        except (KeyError, ValueError):
            continue
        other[r] = o if r not in other else max(other[r], o)
        peak[r] = pa if r not in peak else max(peak[r], pa)
    return {r: AwakePosts(other.get(r), peak.get(r), keep.get(r))
            for r in set(other) | set(peak) | set(keep)}


def capped_record_from_boots(
    boots: Sequence[Tuple[str, str, str]], group: str, n_cards: int,
) -> Tuple[List[Optional[int]], List[Optional[int]], List[str]]:
    """``(OTHER, CAPPED_REST, lines)`` per card ordinal: each post the MAXIMUM
    over the boots that priced it (a boot's budget line and its awake lines),
    the capped rest = OTHER + OVER + KEEP. ``None`` where a post is missing on
    every boot (UNMEASURED, never zero)."""
    other: List[Optional[int]] = [None] * int(n_cards)
    over: List[Optional[int]] = [None] * int(n_cards)
    keep: List[Optional[int]] = [None] * int(n_cards)
    lines: List[str] = []

    def _mx(vec, i, v):
        if v is not None:
            vec[i] = v if vec[i] is None else max(vec[i], v)

    for tag, front, grp in boots:
        posts = awake_posts(grp)
        for o, (nvml, budget, dormant) in sorted(budget_lines(front, group).items()):
            if o >= n_cards or o not in posts:
                continue
            p = posts[o]
            ot = None if p.other_total is None else p.other_total - dormant
            ov = None if p.peak_allocated is None else p.peak_allocated - budget
            _mx(other, o, ot)
            _mx(over, o, ov)
            _mx(keep, o, p.keep)
            lines.append(f"{OVERHANG_MARKER} {tag} group={group} ordinal={o} nvml{nvml}: "
                         f"other {ot} (max card_total - card_free - reserved - dormant_other "
                         f"{dormant}) over {ov} (max peak_allocated {p.peak_allocated} - budget "
                         f"{budget}) keep {p.keep} (post-capture private_free + general_cache) MiB")
    capped: List[Optional[int]] = []
    for o in range(int(n_cards)):
        if other[o] is None or over[o] is None or keep[o] is None:
            capped.append(None)
            lines.append(f"{OVERHANG_MARKER} group={group} ordinal={o}: other {other[o]} over "
                         f"{over[o]} keep {keep[o]} -- a post is missing: UNMEASURED")
            continue
        capped.append(int(other[o] + over[o] + keep[o]))
        lines.append(f"{OVERHANG_MARKER} group={group} ordinal={o}: capped rest {capped[-1]} = "
                     f"other {other[o]} + over {over[o]} + keep {keep[o]} MiB")
    return other, capped, lines


def torch_caps(budgets: Sequence[int], rests: Sequence[Optional[int]],
               others: Sequence[Optional[int]]) -> Optional[List[int]]:
    """Per card the torch cache cap on the physical line: the budget priced
    ``total - carve - dormant_other - growth - rest``, so
    ``total - carve - dormant_other - growth - OTHER = budget + rest - OTHER``.
    ``None`` when a card is not priced by both records (the caller keeps the
    verdict's cap, named) -- a partial vector never caps a card it never saw."""
    if len(budgets) != len(rests) or len(budgets) != len(others):
        return None
    if any(r is None for r in rests) or any(o is None for o in others):
        return None
    return [int(b) + int(r) - int(o) for b, r, o in zip(budgets, rests, others)]


# --------------------------------------------------------------------------
# PDFLIP-ALLOC-CACHE-BOOK (desk/27b-q-170-nvml1-alloccache-1003, item 170): D's
# torch allocator cache growth in the DUAL layout, booked into the card ledger
# of the card where it measurably binds.
#
# THE GAP (boot dkr27bnvfp4dual1mpsleepbar1fs10020527, D.log): the card ledger
# promised bytes the 5090 did not have -- LEDGER-PHYS OVER-PROMISE up to 343 MiB,
# CORRIDOR LAW BREACHED min 56 MiB, one cuMemCreate OOM rolled back by
# GROW-SHORT -- while D's own torch allocator cache swung 270 MiB in the same
# window (``#1028c BOUND alloc_cache ... instant=1648.1 delta=269.7``).
#
# WHAT THE SERIES SAYS (``#1028c BOUND alloc_cache``, ``_allocator_cache_bytes``
# = reserved - allocated, one line per second per D rank): every dual boot
# starts at the same sample (TP0 348.4 / TP1 266.5 / TP2 258.7 MiB, the cache
# when the pools were sized) and GROWS under traffic -- TP0 to 999 / 1003 / 991
# / 1666 MiB in the four newest serving boots of ``27b-nvfp4-dual1m-psleep``.
# The ledger budget (the SUM of the two groups' boot KV, sized when the cache
# was that first sample) knows nothing of the growth. On the 5090 the growth
# ends up in the ledger anyway, one OVER-PROMISE check at a time (3 checks =
# 30 s before ``LEDGER-PHYS BOOKED``): the budget of the three newest boots
# fell 535 / 583 / 1223 MiB from its post-join maximum for a cache growth of
# 654 / 642 / 1317 MiB (ratio 0.82 / 0.91 / 0.93). On the 3080s the same growth
# (1100-1500 MiB) is absorbed by the free memory beside the ledger (budget fell
# 0-162 MiB): booking it there would be a reserve, so those cards are not priced.
#
# THE BOOKING (accounting, not a reserve): per card ordinal::
#
#     book = HW - FIRST    if the card booked unbooked card use (LEDGER-PHYS
#                          BOOKED) in a strict majority of the boots given,
#            0             otherwise (absorbed -- measured, named)
#     HW    max ``instant`` of the series over the boots given
#     FIRST the series' first sample (the cache at the sizing)
#
# the boots being the newest serving boots of ONE checkpoint and form (27B
# NVFP4, dual layout). RECORD > BUILTIN > UNMEASURED (H94): a profile without
# the record books nothing and the launcher line says UNMEASURED. The ledger
# lowers its budget by ``book`` when P joins the card pool
# (``CardKvLedger.book_unpriced``), i.e. what the reconcile would have booked
# after the fact is priced at the join. No constant.
# --------------------------------------------------------------------------

ALLOC_CACHE_RECORD_FMT = "{group}_DUAL_ALLOC_CACHE_BOOK_MIB"
ALLOC_CACHE_MARKER = "PDFLIP-ALLOC-CACHE-BOOK"

_BOUND_CACHE = re.compile(
    r"TP(?P<rank>\d+)\] #1028c BOUND alloc_cache: .*?instant=(?P<inst>[0-9.]+) MiB "
    r"delta=(?P<delta>[0-9.]+) MiB")
_PHYS_BOOKED = re.compile(r"TP(?P<rank>\d+)\] \S+ P-KV LEDGER-PHYS BOOKED (?P<n>\d+) B")


def alloc_cache_record_name(group: str) -> str:
    return ALLOC_CACHE_RECORD_FMT.format(group=str(group).upper())


@dataclass(frozen=True)
class CacheSeries:
    """One D rank of one boot."""

    first: float      # first '#1028c BOUND alloc_cache' instant (MiB): the cache at the sizing
    highwater: float  # max instant (MiB)
    delta_max: float  # max window delta (MiB)
    n: int
    booked_b: int     # sum of 'LEDGER-PHYS BOOKED' bytes of this rank

    @property
    def growth(self) -> float:
        return self.highwater - self.first


def cache_series(group_text: str) -> Dict[int, CacheSeries]:
    """rank -> :class:`CacheSeries` from one D log (only the ``#1028c BOUND
    alloc_cache`` and ``LEDGER-PHYS BOOKED`` lines are read)."""
    first: Dict[int, float] = {}
    high: Dict[int, float] = {}
    dmax: Dict[int, float] = {}
    cnt: Dict[int, int] = {}
    booked: Dict[int, int] = {}
    for line in group_text.splitlines():
        m = _BOUND_CACHE.search(line)
        if m:
            r, inst, delta = int(m.group("rank")), float(m.group("inst")), float(m.group("delta"))
            first.setdefault(r, inst)
            high[r] = max(high.get(r, inst), inst)
            dmax[r] = max(dmax.get(r, delta), delta)
            cnt[r] = cnt.get(r, 0) + 1
            continue
        m = _PHYS_BOOKED.search(line)
        if m:
            r = int(m.group("rank"))
            booked[r] = booked.get(r, 0) + int(m.group("n"))
    return {r: CacheSeries(first[r], high[r], dmax[r], cnt[r], booked.get(r, 0)) for r in first}


def alloc_cache_book_from_boots(
    boots: Sequence[Tuple[str, str]], n_cards: int,
) -> Tuple[List[Optional[int]], List[str]]:
    """``boots`` = (tag, D-log text) of the SAME checkpoint and form, newest
    first. Per card ordinal ``max over boots (highwater - first)``, rounded UP
    to the MiB, where the card booked unbooked card use in a strict majority of
    the boots; 0 where it did not (absorbed -- a measured statement, named);
    ``None`` where no boot sampled the rank (UNMEASURED, never zero)."""
    import math

    per_card: List[List[Tuple[str, CacheSeries]]] = [[] for _ in range(int(n_cards))]
    lines: List[str] = []
    for tag, txt in boots:
        for r, cs in sorted(cache_series(txt).items()):
            if r < n_cards:
                per_card[r].append((tag, cs))
                lines.append(
                    f"{ALLOC_CACHE_MARKER} {tag} rank={r}: alloc_cache first {cs.first:.1f} -> "
                    f"highwater {cs.highwater:.1f} MiB (growth {cs.growth:.1f}, window delta max "
                    f"{cs.delta_max:.1f}, n={cs.n}); ledger booked {cs.booked_b / (1 << 20):.1f} MiB")
    out: List[Optional[int]] = []
    for o, rows in enumerate(per_card):
        if not rows:
            out.append(None)
            lines.append(f"{ALLOC_CACHE_MARKER} group=D ordinal={o}: no boot sampled it: UNMEASURED")
            continue
        binding = sum(1 for _, cs in rows if cs.booked_b > 0)
        if binding * 2 > len(rows):
            out.append(int(math.ceil(max(cs.growth for _, cs in rows))))
            lines.append(f"{ALLOC_CACHE_MARKER} group=D ordinal={o}: BOOK {out[-1]} MiB = max growth over "
                         f"{len(rows)} boots; the ledger had to book card use in {binding}/{len(rows)}")
        else:
            out.append(0)
            lines.append(f"{ALLOC_CACHE_MARKER} group=D ordinal={o}: 0 MiB -- the ledger booked card use in "
                         f"only {binding}/{len(rows)} boots, the card's free memory absorbs the growth "
                         f"(booking it would be a reserve)")
    return out, lines


def _boot_tag(path: str) -> str:
    m = re.search(r"boot_weg2_([A-Za-z0-9]+?)_[0-9a-f]{10}_", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m flliper.srt.pdflip.budget_rest")
    ap.add_argument("--group", choices=("P", "D"), default="D")
    ap.add_argument("--cards", type=int, default=3)
    ap.add_argument("--capped", action="store_true",
                    help="also print the torch-cache-cap records (<G>_TORCH_CAP_OTHER_MIB, "
                         "<G>_AWAKE_REST_CAPPED_MIB)")
    ap.add_argument("--dual-d-log", action="append", default=[], metavar="D_LOG",
                    help="PDFLIP-ALLOC-CACHE-BOOK: a D log of the dual form (newest first, repeat); prints "
                         "<G>_DUAL_ALLOC_CACHE_BOOK_MIB (see PDFLIP-ALLOC-CACHE-BOOK)")
    ap.add_argument("--vote-fixed", action="store_true",
                    help="VOTE-LINE-SINGLE: also print D_EXTEND_VOTE_FIXED_MIB measured from the D logs "
                         "(vote post - chunk-end free - rate x rows, trimmed forwards, first pass excluded)")
    ap.add_argument("--admit-need", action="store_true",
                    help="PDFLIP-ADMIT-BOOK: also print D_ADMIT_NEED_MIB from the D logs of the front logs given (the "
                         "record's form): near-OOM 400 + admission peak - lowest vote post of an admission pass")
    ap.add_argument("--admit-peak-d-log", action="append", default=[], metavar="D_LOG",
                    help="PDFLIP-ADMIT-BOOK: a D log of ANOTHER form of the same handback path whose admission peak "
                         "joins A_peak (repeat); without any, A_peak is the record boots' own")
    ap.add_argument("front_logs", nargs="+", help="front logs, newest first; the group log "
                    "is the same path with .front.log -> .<group>.log")
    a = ap.parse_args(argv)
    boots = []
    for p in a.front_logs:
        gp = re.sub(r"\.front\.log$", f".{a.group}.log", p)
        with open(p, errors="replace") as fh:
            front = fh.read()
        grp = ""
        if os.path.exists(gp):
            with open(gp, errors="replace") as fh:
                grp = "\n".join(ln for ln in fh
                                if "PDFLIP-VRAM-PEAK" in ln or "PDFLIP-GRAPH-POOL" in ln)
        boots.append((_boot_tag(p), front, grp))
    if a.vote_fixed:
        merged: Dict[int, List[ExtendPass]] = {}
        for p in a.front_logs:
            gp = re.sub(r"\.front\.log$", ".D.log", p)
            if not os.path.exists(gp):
                continue
            with open(gp, errors="replace") as fh:
                text = "".join(ln for ln in fh if "PDFLIP-EXTEND-STUECKELUNG" in ln or "PDFLIP-EXTEND-CACHE-TRIM" in ln
                               or "PDFLIP-VRAM-PEAK" in ln)
            for r, ps in extend_passes(text).items():
                # each boot's own first pass is resident growth: the first boot's is excluded by
                # vote_fixed_from_passes (it skips ps[0]), every later boot's is dropped here
                merged.setdefault(r, [])
                merged[r].extend(ps if not merged[r] else ps[1:])
        val, vlines = vote_fixed_from_passes(merged)
        for ln in vlines:
            print(ln)
        print(json.dumps({"name": VOTE_FIXED_RECORD, "value": None if val is None else [val] * int(a.cards),
                          "boots": [b[0] for b in boots],
                          "kind": "memory"}))
    if a.admit_need:
        from flliper.srt.managers.corridor_guard import NEAR_OOM_MIB

        def _admit_text(path: str) -> str:
            with open(path, errors="replace") as fh:
                return "".join(ln for ln in fh if "PDFLIP-EXTEND-STUECKELUNG" in ln or "PDFLIP-EXTEND-CACHE-TRIM" in ln
                               or "PDFLIP-VRAM-PEAK" in ln)

        own = [admission_passes(_admit_text(re.sub(r"\.front\.log$", ".D.log", p))) for p in a.front_logs
               if os.path.exists(re.sub(r"\.front\.log$", ".D.log", p))]
        other = [admission_passes(_admit_text(p)) for p in a.admit_peak_d_log]
        need, alines = admit_need_from_passes(own, float(NEAR_OOM_MIB), admission_peak(own + other))
        for ln in alines:
            print(ln)
        print(json.dumps({"name": ADMIT_RECORD, "value": None if need is None else [need] * int(a.cards),
                          "boots": [b[0] for b in boots], "kind": "memory"}))
    vals, lines = record_from_boots(boots, a.group, a.cards)
    for ln in lines:
        print(ln)
    print(json.dumps({"name": record_name(a.group), "value": vals,
                      "boots": [b[0] for b in boots], "kind": "memory"}))
    if a.capped:
        others, capped, clines = capped_record_from_boots(boots, a.group, a.cards)
        for ln in clines:
            print(ln)
        for name, v in ((other_record_name(a.group), others),
                        (capped_record_name(a.group), capped)):
            print(json.dumps({"name": name, "value": v, "boots": [b[0] for b in boots],
                              "kind": "memory"}))
    if a.dual_d_log:
        dual = []
        for p in a.dual_d_log:
            with open(p, errors="replace") as fh:
                dual.append((_boot_tag(p), "".join(
                    ln for ln in fh if "#1028c BOUND alloc_cache" in ln or "LEDGER-PHYS BOOKED" in ln)))
        book, blines = alloc_cache_book_from_boots(dual, int(a.cards))
        for ln in blines:
            print(ln)
        print(json.dumps({"name": alloc_cache_record_name(a.group), "value": book,
                          "boots": [d[0] for d in dual], "kind": "memory", "fmt": "nvfp4"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
