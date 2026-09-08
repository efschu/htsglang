"""WEG2 flip-cost C19/C20: the per-card ring table, SOLVED from a boot's own lines.

Nothing in this module is a constant that a human chose.  Every quantity it
returns is read out of the PREVIOUS boot's logs, carries the boot tag and the
line count it was read from, and is printed beside the arithmetic it feeds
(spec section 10.3: no hand-sized ring, no ``skew`` constant, no ``--ring-mib``
flag, no env knob a human sets).  When the lines are absent or short this
module returns ``None`` with a REASON -- it never guesses, and the launcher then
runs the OLD flip form and says so (R22).

The four quantities, and the exact line each comes from:

``image_g(c)``
    The host bytes group ``g`` parks for card ``c`` in ONE complete sleep pass.
    Read from ``WEG2-CHUNK-BYTES sleep tags=[...] host_image_delta=N MiB``
    (weight_updater.py:738), grouped into passes by tag repetition and taken as
    the PEAK pass, never the mean: a ring sized to a mean is a ring that blocks
    on the pass that was larger.

``max_tag_g(c)``
    The largest single tag of that same population -- the step size of R5's
    corridor walk.

``device_credit(c, S->W)``
    ``NVML free while S is awake`` + ``the kv_cache device bytes S releases``.
    The first from the front's ``WEG2-CORRIDOR phase=S(awake) ... nvmlN:free=``
    samples, taken as the MINIMUM over the phase (the conservative end: a
    smaller credit makes the requirement larger, never smaller).  The second
    from that group's ``KV Cache is allocated ... K size: X GB, V size: Y GB``
    lines, summed over every pool the rank allocates.

Then R5, per card per direction, with ``S`` the sleeping group and ``W`` the
waking one::

    H(c)    = max_g image_g(c)
    span1(c)= image_P(c)                       # R7: registered at P's first pause
    need    = image_W(c) - credit(c) + max_tag_S(c) + max_tag_W(c)
    slack   = H(c) - need                      # negative => W32, before either
                                               #             group starts

INSTRUMENT NOTE (spec R8), stated here because this module is the last consumer
that may legitimately read it: ``host_image_delta`` is an RssShmem delta and it
DIES the moment the ring lands -- ring granules are tmpfs pages mapped by both
co-located processes, so the delta collapses to ~0.  A table solved from a
RING boot must therefore come from ``tms_tag_bytes`` (C7/C16, next slice);
this module reads whichever of the two instruments the source boot carries and
NAMES it in the provenance string, so a number can never be quoted without its
instrument.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 1024 * 1024
GB = 1e9

#: ``[2026-09-07 21:10:23 TP2] WEG2-CHUNK-BYTES sleep tags=['weights_0'] host_image_delta=982 MiB``
_CHUNK_RE = re.compile(
    r"\b(?:TP|PP)(\d+)\]\s+WEG2-CHUNK-BYTES\s+sleep\s+tags=\[([^\]]*)\]\s+"
    r"host_image_delta=(-?\d+)\s+MiB"
)
#: The RING-era instrument (C16, next slice); same shape, honest name.
_TAG_RE = re.compile(
    r"WEG2-FLIP-TAG\s+group=\S+\s+rank=(\d+)\s+card=(\S+)\s+dir=\S+\s+tag=(\S+)\s+"
    r"bytes=(\d+)\s+MiB"
)
#: ``KV Cache is allocated. ... K size: 5.63 GB, V size: 5.63 GB``
_KV_RE = re.compile(
    r"\b(?:TP|PP)(\d+)\]\s+KV Cache is allocated\..*?K size:\s*([\d.]+)\s*GB,\s*"
    r"V size:\s*([\d.]+)\s*GB"
)
#: ``WEG2-CORRIDOR phase=D(awake) ... nvml0:free=2474MiB nvml1:free=1987MiB ...``
_CORRIDOR_PHASE_RE = re.compile(r"WEG2-CORRIDOR\s+phase=([A-Z])\(awake\)")
_CORRIDOR_FREE_RE = re.compile(r"nvml(\d+):free=(\d+)MiB")


class Weg2RingCreditRefused(RuntimeError):
    """W32: the R5 inequality fails on some card/direction at the launch check."""


class Weg2RingNeedsInterleave(RuntimeError):
    """W34: the ring was asked to arm under a SERIAL flip form it cannot fund.

    R5's corridor inequality is a statement about two legs CONCURRENTLY in
    flight (spec C9): W's per-tag releases are what fund S's acquires, so the
    walk of ``u`` never has to hold both whole images at once.  Under the serial
    per-tag order the tree actually ships (``front.FLIP_LEG_FORM == "serial"``)
    S must acquire before W's RPC is even sent, so the requirement is the much
    larger ``H(c) >= image_W(c) + max_tag_S(c)``.  A blocking ring sized to R5
    and armed under the serial order does not run slowly -- it wedges the FIRST
    flip and the acquire budget kills the rank (W31 -> group-fatal W4).  Raised
    at LAUNCH, before either group starts.
    """


@dataclass
class CardRing:
    uuid: str
    nvml_index: int
    name: str
    image_p_mib: int = 0
    image_d_mib: int = 0
    max_tag_p_mib: int = 0
    max_tag_d_mib: int = 0
    credit_d2p_mib: int = 0
    credit_p2d_mib: int = 0

    @property
    def h_mib(self) -> int:
        return max(self.image_p_mib, self.image_d_mib)

    @property
    def span1_mib(self) -> int:
        return self.image_p_mib

    @property
    def need_d2p_mib(self) -> int:
        # S = D (sleeping), W = P (waking).
        return self.image_p_mib - self.credit_d2p_mib + self.max_tag_d_mib + self.max_tag_p_mib

    @property
    def need_p2d_mib(self) -> int:
        return self.image_d_mib - self.credit_p2d_mib + self.max_tag_p_mib + self.max_tag_d_mib

    @property
    def slack_d2p_mib(self) -> int:
        return self.h_mib - self.need_d2p_mib

    @property
    def slack_p2d_mib(self) -> int:
        return self.h_mib - self.need_p2d_mib

    # -- the SERIAL form's requirement (front.FLIP_LEG_FORM == "serial") -----
    # ``need`` above is the corridor walk of C9's gathered legs, where W's
    # releases fund S's acquires.  With the legs serialised per tag, S acquires
    # while W still holds its WHOLE parked image and no device credit can be
    # spent on the host side: what has to fit is the parked image plus S's
    # first step.  A strictly larger requirement, and the one this tree runs.

    @property
    def need_serial_d2p_mib(self) -> int:
        # S = D (sleeping), W = P (waking, whole image parked on the host).
        return self.image_p_mib + self.max_tag_d_mib

    @property
    def need_serial_p2d_mib(self) -> int:
        return self.image_d_mib + self.max_tag_p_mib

    @property
    def slack_serial_d2p_mib(self) -> int:
        return self.h_mib - self.need_serial_d2p_mib

    @property
    def slack_serial_p2d_mib(self) -> int:
        return self.h_mib - self.need_serial_p2d_mib

    @property
    def complete(self) -> bool:
        return self.image_p_mib > 0 and self.image_d_mib > 0 and self.max_tag_p_mib > 0 and self.max_tag_d_mib > 0


@dataclass
class RingTable:
    boot: str
    instrument: str
    lines_read: int
    cards: List[CardRing] = field(default_factory=list)
    #: The largest SINGLE STEP of the flip, summed over cards: ``max_k Sigma_c
    #: bytes(k, c)`` over the tags the front's interleave actually walks.  This
    #: is the OLD form's in-flight term -- one tag is one RPC and its shards
    #: land on every card at once, so the sum is over cards and the max is over
    #: tags.  ``Sigma_c max_k`` (each card's own largest tag, summed) is a
    #: different and larger quantity: the maxima need not be the same tag.
    max_step_total_mib: int = 0

    @property
    def total_h_bytes(self) -> int:
        return sum(c.h_mib for c in self.cards) * MIB

    @property
    def total_span1_bytes(self) -> int:
        return sum(c.span1_mib for c in self.cards) * MIB

    def provenance(self) -> str:
        return (
            f"boot {self.boot}, {self.lines_read} {self.instrument} lines "
            f"(instrument: {self.instrument}; every MiB below is that boot's own, "
            "none is a constant in this tree)"
        )

    def env_map(self) -> str:
        """``TMS_HOST_RING_MAP`` -- ``<uuid>=<bytes>:<span1>,...``

        A PATH-addressed map only.  There is no ``:fd=<n>`` field: an fd number
        is meaningless in the process that has to open the region, because the
        ranks are ``spawn``-started scheduler processes that inherit no
        descriptor from the launcher (``entrypoints/engine.py`` sets
        ``mp.set_start_method("spawn", force=True)``).
        """
        return ",".join(f"{c.uuid}={c.h_mib * MIB}:{c.span1_mib * MIB}" for c in self.cards)

    def format_l6(self) -> List[str]:
        """L6, one line per card, plus the refusal lines (spec section 5)."""
        out = []
        prov = self.provenance()
        for c in self.cards:
            out.append(
                f"WEG2-HOST-LEDGER RING card={c.uuid} nvml{c.nvml_index} {c.name} "
                f"image_D={c.image_d_mib} image_P={c.image_p_mib} H={c.h_mib} "
                f"span1={c.span1_mib} credit_d2p={c.credit_d2p_mib} "
                f"credit_p2d={c.credit_p2d_mib} need_d2p={c.need_d2p_mib} "
                f"need_p2d={c.need_p2d_mib} slack={c.slack_d2p_mib}/{c.slack_p2d_mib} MiB "
                f"| SERIAL FORM need_d2p={c.need_serial_d2p_mib} "
                f"need_p2d={c.need_serial_p2d_mib} "
                f"slack={c.slack_serial_d2p_mib}/{c.slack_serial_p2d_mib} MiB "
                f"-- provenance: {prov}"
            )
        return out

    def refusals(self) -> List[str]:
        """Every violated R5 case, with its arithmetic.  Empty = the launch check passes."""
        bad = []
        for c in self.cards:
            for direction, need, s_tag, w_tag, image_w, credit in (
                ("d2p", c.need_d2p_mib, "D", "P", c.image_p_mib, c.credit_d2p_mib),
                ("p2d", c.need_p2d_mib, "P", "D", c.image_d_mib, c.credit_p2d_mib),
            ):
                if need > c.h_mib:
                    bad.append(
                        f"RING REFUSED: need {need} > H {c.h_mib} on card {c.uuid} "
                        f"(nvml{c.nvml_index} {c.name}, {direction}): "
                        f"image_{w_tag} {image_w} - credit {credit} + max_tag_{s_tag} "
                        f"{c.max_tag_d_mib if s_tag == 'D' else c.max_tag_p_mib} + max_tag_{w_tag} "
                        f"{c.max_tag_p_mib if w_tag == 'P' else c.max_tag_d_mib} = {need} MiB"
                    )
        return bad

    def serial_refusals(self) -> List[str]:
        """Every card/direction the SERIAL flip form cannot walk.  Empty = fundable.

        Read this beside :meth:`refusals`: R5's cases can all pass while every
        one of these fails, because they price two different flip forms.  A
        blocking ring armed on the R5 numbers under the serial order wedges at
        the first acquire -- the ARMED line would then certify an inequality for
        a form this tree does not contain.
        """
        bad = []
        for c in self.cards:
            for direction, need, image_w, w_tag, s_tag, step in (
                ("d2p", c.need_serial_d2p_mib, c.image_p_mib, "P", "D", c.max_tag_d_mib),
                ("p2d", c.need_serial_p2d_mib, c.image_d_mib, "D", "P", c.max_tag_p_mib),
            ):
                if need > c.h_mib:
                    bad.append(
                        f"RING NEEDS INTERLEAVE: on card {c.uuid} (nvml{c.nvml_index} "
                        f"{c.name}, {direction}) the serial form needs image_{w_tag} "
                        f"{image_w} + max_tag_{s_tag} {step} = {need} MiB but H is "
                        f"{c.h_mib} MiB -- at flip start the host holds {w_tag}'s whole "
                        f"parked image and only {c.h_mib - image_w} MiB is free, so "
                        f"{s_tag}'s first acquire of {step} MiB blocks and no "
                        f"{w_tag}-release can be issued until it returns"
                    )
        return bad


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _passes(records: Sequence[Tuple[Tuple[str, ...], int]]) -> List[int]:
    """Total host MiB of each complete sleep pass, in order.

    Two record shapes reach this, and conflating them is a measured 2x error:

    * a SINGLE-tag record -- one step of the front's interleave.  These group
      into passes by tag repetition: the sleep loop walks the weights family
      once per flip leg, so the second sighting of a tag is the next leg.
    * a MULTI-tag record -- the launcher's bulk ``/release_memory_occupation``
      with the whole family in one RPC (``sleep_group``).  That single line is
      already a COMPLETE pass; treating its tag tuple as one more member of the
      surrounding pass double-counts the image (measured on boot weg2dk5:
      image_P read 27 720 instead of 13 860 MiB).
    """
    out: List[int] = []
    cur = 0
    seen: set = set()
    for tags, mib in records:
        if len(tags) > 1:
            if seen:
                out.append(cur)
                cur, seen = 0, set()
            out.append(mib)
            continue
        tag = tags[0]
        if tag in seen:
            out.append(cur)
            cur, seen = 0, set()
        cur += mib
        seen.add(tag)
    if seen:
        out.append(cur)
    return out


def _family_roots(tags: Sequence[str]) -> set:
    """The tag names that name a whole FAMILY rather than one step of it.

    ``weights`` is a bulk record exactly as ``['weights_0', ..., 'weights_11']``
    in one line is: it names the family the chunked tags partition, so its delta
    is a whole pass and not a corridor step.  The multi-tag guard in
    :func:`_passes` does not catch it -- the record carries ONE tag name -- and
    counting it as a step read the 5090's D step as 2 856 instead of 1 608 MiB
    on boot weg2zr2, which is the "bulk record counted as a max_tag" shape.
    """
    names = set(tags)
    return {t for t in names if any(o != t and o.startswith(t + "_") for o in names)}


def parse_group_log(
    path: str,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], int, str, Dict[str, int]]:
    """(image_mib, max_tag_mib, kv_mib) per RANK INDEX, plus lines read,
    instrument, and the per-tag totals summed over ranks (the flip's step size).

    The rank index is the index within its group, which is also its CVD ordinal
    -- the launcher pins ``--rank-gpu-id 0,1,2`` against a CVD ordered by UUID,
    so rank ``n`` runs on ``cards[n]``.  The caller owns that mapping; this
    function never guesses a card.
    """
    per_rank: Dict[int, List[Tuple[Tuple[str, ...], int]]] = {}
    kv_gb: Dict[int, float] = {}
    lines_read = 0
    instrument = "WEG2-CHUNK-BYTES sleep host_image_delta (RssShmem; DEAD once the ring lands, R8)"
    with open(path, errors="replace") as f:
        for line in f:
            if "WEG2-FLIP-TAG" in line:
                m = _TAG_RE.search(line)
                if m:
                    rank = int(m.group(1))
                    per_rank.setdefault(rank, []).append(((m.group(3),), int(m.group(4))))
                    lines_read += 1
                    instrument = "WEG2-FLIP-TAG bytes (tms_tag_bytes, the saver's own accounting)"
                    continue
            if "WEG2-CHUNK-BYTES sleep" in line:
                m = _CHUNK_RE.search(line)
                if m:
                    tags = tuple(
                        t.strip().strip("'\"") for t in m.group(2).split(",") if t.strip()
                    )
                    per_rank.setdefault(int(m.group(1)), []).append((tags, int(m.group(3))))
                    lines_read += 1
                continue
            if "KV Cache is allocated" in line:
                m = _KV_RE.search(line)
                if m:
                    rank = int(m.group(1))
                    kv_gb[rank] = kv_gb.get(rank, 0.0) + float(m.group(2)) + float(m.group(3))
    image: Dict[int, int] = {}
    max_tag: Dict[int, int] = {}
    tag_peak: Dict[str, Dict[int, int]] = {}
    for rank, records in per_rank.items():
        # PEAK pass, never the mean (a ring sized to a mean blocks on the
        # larger pass); population = the passes actually logged.
        image[rank] = max(_passes(records), default=0)
        # max_tag is the corridor STEP SIZE, so only single-tag records may feed
        # it: a bulk record's delta is a whole family, not one tag, and would
        # read as a step nothing ever takes.  A family ROOT name is bulk too.
        singles = [(tags[0], v) for tags, v in records if len(tags) == 1]
        roots = _family_roots([t for t, _ in singles])
        steps = [(t, v) for t, v in singles if t not in roots]
        max_tag[rank] = max((v for _, v in steps), default=0)
        for tag, mib in steps:
            slot = tag_peak.setdefault(tag, {})
            if mib > slot.get(rank, 0):
                slot[rank] = mib
    kv_mib = {r: int(round(gb * GB / MIB)) for r, gb in kv_gb.items()}
    # One tag is ONE RPC of the front's interleave and its shards land on every
    # card at once, so the step the host must hold in flight is the sum over
    # ranks of that tag, maximised over tags.
    tag_totals = {tag: sum(per.values()) for tag, per in tag_peak.items()}
    return image, max_tag, kv_mib, lines_read, instrument, tag_totals


def parse_front_corridor(path: str) -> Dict[str, Dict[int, int]]:
    """``{phase letter: {nvml index: MINIMUM free MiB observed in that phase}}``.

    The minimum, not the first sample: it is the conservative end of the credit
    (a smaller credit only ever makes R5's requirement larger), and the first
    sample of a phase is taken before the awake group has any load on it.
    """
    out: Dict[str, Dict[int, int]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if "WEG2-CORRIDOR" not in line:
                continue
            m = _CORRIDOR_PHASE_RE.search(line)
            if not m:
                continue
            phase = out.setdefault(m.group(1), {})
            for idx, free in _CORRIDOR_FREE_RE.findall(line):
                i, v = int(idx), int(free)
                if i not in phase or v < phase[i]:
                    phase[i] = v
    return out


def _boot_stems(evidence_dir: str) -> List[str]:
    try:
        names = os.listdir(evidence_dir)
    except OSError:
        return []
    stems = {n[: -len(".front.log")] for n in names if n.endswith(".front.log")}
    have = [s for s in stems if f"{s}.P.log" in names and f"{s}.D.log" in names]
    have.sort(key=lambda s: os.path.getmtime(os.path.join(evidence_dir, f"{s}.front.log")), reverse=True)
    return have


def solve(
    cards: Sequence,
    evidence_dir: str,
    boot_stem: Optional[str] = None,
) -> Tuple[Optional[RingTable], str]:
    """Solve the table from the newest usable boot, or return (None, reason).

    ``cards`` is the launcher's ORDERED card list (ordinal 0 first): rank ``n``
    of either group runs on ``cards[n]``.
    """
    stems = _boot_stems(evidence_dir)
    if not stems:
        return None, f"no boot in {evidence_dir} carries all three of .front/.P/.D log"
    if boot_stem:
        # A pin is a SUBSTRING of a stem, not the stem itself: the operator
        # types the boot TAG (``weg2zr2``) while the files are named
        # ``boot_weg2_weg2zr2_<sha>_<date>_<time>.{front,P,D}.log``.  An
        # unmatched or ambiguous pin returns a REASON so the caller prints R22
        # and runs the OLD form; it never escapes as an OSError out of an
        # unconditional open (every refusal on this path is named).
        exact = [s for s in stems if s == boot_stem]
        hits = exact or [s for s in stems if boot_stem in s]
        if len(hits) != 1:
            return None, (
                f"--ring-table-boot {boot_stem!r} matches {len(hits)} of the "
                f"{len(stems)} complete boots in {evidence_dir}"
                + (f" ({', '.join(sorted(hits)[:4])})" if hits else "")
                + " -- a pin must name exactly one"
            )
        stems = hits
    reasons = []
    for stem in stems:
        p_log = os.path.join(evidence_dir, f"{stem}.P.log")
        d_log = os.path.join(evidence_dir, f"{stem}.D.log")
        f_log = os.path.join(evidence_dir, f"{stem}.front.log")
        image_p, maxtag_p, kv_p, n_p, inst_p, steps_p = parse_group_log(p_log)
        image_d, maxtag_d, kv_d, n_d, inst_d, steps_d = parse_group_log(d_log)
        if not image_p or not image_d:
            reasons.append(f"{stem}: no sleep-pass lines for {'P' if not image_p else 'D'}")
            continue
        corridor = parse_front_corridor(f_log)
        if "P" not in corridor or "D" not in corridor:
            reasons.append(f"{stem}: front carries no WEG2-CORRIDOR phase=P(awake)/D(awake) samples")
            continue
        table = RingTable(
            boot=stem,
            instrument=inst_d or inst_p,
            lines_read=n_p + n_d,
            max_step_total_mib=max(
                max(steps_p.values(), default=0), max(steps_d.values(), default=0)
            ),
        )
        for ordinal, card in enumerate(cards):
            cr = CardRing(uuid=card.uuid, nvml_index=card.nvml_index, name=card.name)
            cr.image_p_mib = int(image_p.get(ordinal, 0))
            cr.image_d_mib = int(image_d.get(ordinal, 0))
            cr.max_tag_p_mib = int(maxtag_p.get(ordinal, 0))
            cr.max_tag_d_mib = int(maxtag_d.get(ordinal, 0))
            # credit(S->W) = NVML free while S is awake + the kv bytes S releases
            cr.credit_d2p_mib = int(corridor["D"].get(card.nvml_index, 0)) + int(kv_d.get(ordinal, 0))
            cr.credit_p2d_mib = int(corridor["P"].get(card.nvml_index, 0)) + int(kv_p.get(ordinal, 0))
            table.cards.append(cr)
        missing = [c.uuid for c in table.cards if not c.complete]
        if missing:
            reasons.append(f"{stem}: incomplete rows for {missing}")
            continue
        return table, f"solved from {stem}"
    return None, "; ".join(reasons[:4]) or "no usable boot"
