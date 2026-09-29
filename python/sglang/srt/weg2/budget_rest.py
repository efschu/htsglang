# SPDX-License-Identifier: Apache-2.0
"""WEG2-BUDGET-REST: an awake group's MEASURED rest beyond its budget line, as ONE post.

THE DEFECT (27B inventory 29.09., ``LEISTUNGSSCHALTER-INVENTAR-0929.md`` R1-R3).
Group D's budget on the 27B booked, per card, three named constants above the
measured transient of its corridor floor (767 MiB on the 5090)::

    user reserve 1800 / 1400 / 1400      (--user-reserve-mib, 27b.env)
    awake_overshoot 404                  (launcher constant, boot weg2onebackup2, 07.09.)
    measured_awake_overshoot 489 / 0 / 0 (D_OVERSHOOT_MIB, boot weg2ls4b1, 07.09.)

The law (users 19.09. and 29.09.) is: no reserve, only measured transients.
The same boots that carried those constants MEASURED what an awake D actually
holds on each card beyond its budget line: ``WEG2-VRAM-PEAK`` (every D rank,
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

``python -m sglang.srt.weg2.budget_rest --group D <front.log> ...`` prints the
record row for ``weg2/profile_records_data/<profile>.json``: per card ordinal
the MAXIMUM over the boots given (the newest N of the form, like
``launcher.dormant_max_from_records``), ``null`` where no boot priced it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MARKER = "WEG2-BUDGET-REST"
#: the record row per group (``<G>_AWAKE_REST_BOOKED_MIB``): per card ordinal
RECORD_FMT = "{group}_AWAKE_REST_BOOKED_MIB"
#: the explicit off switch (the registry row is the default)
ENV = "SGLANG_WEG2_BUDGET_REST_RECORD"

SOURCE_RECORD = "RECORD"
SOURCE_UNMEASURED = "UNMEASURED"


def record_name(group: str) -> str:
    return RECORD_FMT.format(group=str(group).upper())


_BUDGET = re.compile(
    r"WEG2-LAUNCH budget (?P<label>P|D) group=(?P<group>P|D) ordinal=(?P<ord>\d+) "
    r"nvml_idx=(?P<nvml>\d+) [^:]*: (?P<budget>-?\d+) MiB = total (?P<total>\d+) "
    r".*?dormant_other (?P<dormant>\d+)"
)
_PEAK = re.compile(r"WEG2-VRAM-PEAK rank=(?P<rank>\d+) ")
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
    ``WEG2-VRAM-PEAK`` line of the group's log. Lines without a numeric
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
            group: str) -> Optional[Resolved]:
    """The record's values for ``n_cards`` cards, or ``None`` when the
    profile carries no record for ``group`` (every card UNMEASURED; the
    caller's budgets stay byte-identical). A record of another card count is
    REFUSED by raising -- a partial vector must not price a card it never saw."""
    if values is None:
        return None
    vals = list(values)
    if len(vals) != int(n_cards):
        raise ValueError(f"{record_name(group)} has {len(vals)} entries for {n_cards} cards; "
                         f"re-measure with python -m sglang.srt.weg2.budget_rest")
    return Resolved(values=tuple(None if v is None else int(v) for v in vals),
                    provenance=f"{record_name(group)} {boots}".strip())


def _boot_tag(path: str) -> str:
    m = re.search(r"boot_weg2_([A-Za-z0-9]+?)_[0-9a-f]{10}_", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sglang.srt.weg2.budget_rest")
    ap.add_argument("--group", choices=("P", "D"), default="D")
    ap.add_argument("--cards", type=int, default=3)
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
                grp = "\n".join(ln for ln in fh if "WEG2-VRAM-PEAK" in ln)
        boots.append((_boot_tag(p), front, grp))
    vals, lines = record_from_boots(boots, a.group, a.cards)
    for ln in lines:
        print(ln)
    print(json.dumps({"name": record_name(a.group), "value": vals,
                      "boots": [b[0] for b in boots], "kind": "memory"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
