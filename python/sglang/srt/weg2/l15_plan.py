# SPDX-License-Identifier: Apache-2.0
"""L1.5 planner post -- the pure decision layer (L15-PLAN-0930 2.4, AP L15-01).

L1.5 holds D's sleep residue in the brach VRAM of the P layout: the planner
carves a post ``l15`` out of every P budget, sized from the MEASURED P awake
peak (record ``P_AWAKE_PEAK_MIB``), never from a reserve -- keep spans are
fixed per sleep, so the planner sizes them against what P actually needed
(L15-PLAN-0930 2.4, "Elastic").  This module answers, from numbers handed in,
how much each card holds, under which provenance, and what an NF hold costs in
resident expert rows.  It decides; it never launches: no launcher import, no
torch, no record I/O.  The wiring into ``launcher.budgets_from_dc`` and
``vram_plan_view._p_group`` is the separate L15-01b.

Switches (code default off; off means the launch stays byte for byte today's):
  SGLANG_WEG2_L15          master
  SGLANG_WEG2_HOT_HANDOVER hot handover (2.0); refused on --dual-layout like L1.5
  SGLANG_WEG2_L15_MIB      ``auto`` | ``c<card>=<mib>,...`` operator override,
                           shown as OVERRIDE in the vram_plan

NF law (free VRAM = experts): on the ``nextflash`` line the post defaults to
0 no matter what the record would say.  An override > 0 is the user addendum:
an L1.5 hold PAID with resident MoE experts -- rows = floor(l15 / row_mib),
the row size from the P card fit, shown next to the post (boot line
``L15-POST card= mib= src= experts_rows_traded=``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

L15_MASTER_ENV = "SGLANG_WEG2_L15"
L15_MIB_ENV = "SGLANG_WEG2_L15_MIB"
HOT_HANDOVER_ENV = "SGLANG_WEG2_HOT_HANDOVER"
#: the accepted spellings of "on" (case and blanks tolerated; anything else is off)
_ON_VALUES = ("1", "true", "on")

LINE_QWEN27B = "qwen27b"
LINE_NEXTFLASH = "nextflash"

#: refusal code: L1.5 and the hot handover are refused on --dual-layout in V1
DUAL_REFUSAL_CODE = "W-L15-DUAL"
NOCAP_REFUSAL_CODE = "W-L15-NOCAP"


def _switch(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key, "") or "").strip().lower() in _ON_VALUES


def master_on(env: Mapping[str, str]) -> bool:
    """The L1.5 master switch; default off (0 = today byte for byte)."""
    return _switch(env, L15_MASTER_ENV)


def handover_on(env: Mapping[str, str]) -> bool:
    """The hot-handover switch (2.0, both lines); same parsing, default off."""
    return _switch(env, HOT_HANDOVER_ENV)


def parse_l15_mib(value: Optional[str]) -> Tuple[str, Dict[int, int]]:
    """``SGLANG_WEG2_L15_MIB`` -> ``("auto", {})`` or ``("override", {card: mib})``.

    Absent, empty or ``auto`` mean the record-derived post; ``c0=100,c2=3000``
    overrides cards 0 and 2 (a card NOT named is held 0, named
    OVERRIDE-UNNAMED -- an override never silently leaves a card on auto).
    Anything else raises by naming the value: a boot must not misread a typo
    in a MiB figure as the auto mode.
    """
    text = ("" if value is None else str(value)).strip()
    if text == "" or text.lower() == "auto":
        return ("auto", {})
    out: Dict[int, int] = {}
    for part in text.split(","):
        key, sep, val = part.strip().partition("=")
        if not sep or not key.startswith("c") or not key[1:].isdigit() or not val.isdigit():
            raise ValueError(
                f"{L15_MIB_ENV}={value!r} is malformed: expected 'auto' or "
                f"'c<card>=<mib>' pairs (got {part!r}) -- refusing to guess "
                "whether a typo meant a hold or the auto mode")
        out[int(key[1:])] = int(val)
    return ("override", out)


def l15_post_mib(p_budget_mib: int, p_awake_peak_mib: Optional[int]) -> Tuple[int, str]:
    """The post from the records: what the P budget holds beyond the measured P
    awake peak, clamped at 0 (a P that exceeds its record (F5) holds less next
    sleep, never negative).  A missing peak is UNMEASURED zero -- the post is
    priced from records only, never estimated."""
    if p_awake_peak_mib is None:
        return (0, "UNMEASURED")
    return (max(0, int(p_budget_mib) - int(p_awake_peak_mib)), "RECORD(P_AWAKE_PEAK_MIB)")


def nf_expert_trade(l15_mib: int, row_mib: float) -> int:
    """Expert rows an NF hold of ``l15_mib`` costs at ``row_mib`` per row."""
    if row_mib <= 0:
        raise ValueError(f"nf_expert_trade: row_mib must be > 0, got {row_mib!r} "
                         "-- an expert row of 0 or less MiB prices nothing")
    return int(math.floor(int(l15_mib) / float(row_mib)))


@dataclass(frozen=True)
class L15Post:
    """One card's L1.5 hold: the bytes, where they come from, their expert cost."""

    card: int
    mib: int
    src: str
    experts_rows_traded: int


def resolve_posts(line: str, p_budget_mib: Sequence[int],
                  p_awake_peak_mib: Sequence[Optional[int]],
                  env: Mapping[str, str],
                  row_mib: Optional[Sequence[float]] = None) -> List[L15Post]:
    """Per card what L1.5 holds on the given launch line, in order.

    Precedence (fail-fast, no hidden mode mixing): master off -> every card 0
    under OFF; else an override decides the cards it names (OVERRIDE) and
    holds 0 on the rest (OVERRIDE-UNNAMED); else the NF line defaults to 0
    (NF-DEFAULT-0, the free-VRAM-is-the-experts' law); else the post comes
    from the records.  ``experts_rows_traded`` is priced only where the hold
    is actually paid with experts: the NF line with a row size handed in.
    """
    n = len(p_budget_mib)
    if len(p_awake_peak_mib) != n:
        raise ValueError(f"resolve_posts: {len(p_awake_peak_mib)} awake peaks for "
                         f"{n} budgets -- a partial vector never prices a card")
    if row_mib is not None and len(row_mib) != n:
        raise ValueError(f"resolve_posts: {len(row_mib)} row sizes for {n} cards")
    norm = str(line).strip().lower()
    if norm not in (LINE_QWEN27B, LINE_NEXTFLASH):
        raise ValueError(f"resolve_posts: unknown launch line {line!r} "
                         f"(known: {LINE_QWEN27B!r}, {LINE_NEXTFLASH!r})")
    if not master_on(env):
        return [L15Post(card=i, mib=0, src="OFF", experts_rows_traded=0) for i in range(n)]
    mode, override = parse_l15_mib(env.get(L15_MIB_ENV))
    nf = norm == LINE_NEXTFLASH
    posts: List[L15Post] = []
    for i in range(n):
        if mode == "override":
            if i in override:
                mib, src = override[i], "OVERRIDE"
            else:
                mib, src = 0, "OVERRIDE-UNNAMED"
        elif nf:
            mib, src = 0, "NF-DEFAULT-0"
        else:
            mib, src = l15_post_mib(int(p_budget_mib[i]), p_awake_peak_mib[i])
        rows = (nf_expert_trade(mib, row_mib[i])
                if nf and mib > 0 and row_mib is not None else 0)
        posts.append(L15Post(card=i, mib=mib, src=src, experts_rows_traded=rows))
    return posts


def refuse_dual(argv: Sequence[str], env: Mapping[str, str]) -> Optional[str]:
    """The dual-layout refusal (V1): on ``--dual-layout`` P never sleeps, so L1.5
    has nothing to hold (the card KV ledger shares that room) and the hot
    handover has no flip to hand over across.  Refused BY NAME when either
    switch is on -- never silently off, because a silent off would let a boot
    run believing L1.5 holds what it does not.  Returns the message to abort
    with, or None when there is nothing to refuse."""
    if "--dual-layout" not in argv:
        return None
    armed = [name for name, on in ((L15_MASTER_ENV, master_on(env)),
                                   (HOT_HANDOVER_ENV, handover_on(env))) if on]
    if not armed:
        return None
    return (f"{DUAL_REFUSAL_CODE}: --dual-layout with {', '.join(armed)} is refused in V1: "
            "the dual layout never sleeps P, so L1.5 has no sleep to hold into (the card "
            "KV ledger already shares that room) and the hot handover has no flip to hand "
            "over across. Turn the switch off or run without --dual-layout; refusing to "
            "boot a launch whose L1.5 would silently do nothing.")


def refuse_no_caps(posts: Sequence[L15Post], env: Mapping[str, str]) -> Optional[str]:
    """The no-cap refusal (N3f): with the master on, a launch whose every card
    holds 0 MiB can never retain a single row -- N3c (10012013) and N3e
    (10012112) ran exactly that (SGLANG_WEG2_L15=1 without
    SGLANG_WEG2_L15_MIB: every post UNMEASURED 0, every D cap 0) and spent a
    whole boot window. Refused BY NAME with the fix in the text; master off
    or any card > 0 -> None."""
    if not master_on(env):
        return None
    if any(int(p.mib) > 0 for p in posts):
        return None
    srcs = ",".join(sorted({p.src for p in posts})) or "none"
    return (f"{NOCAP_REFUSAL_CODE}: {L15_MASTER_ENV}=1 but every card's L1.5 post is 0 MiB "
            f"(src={srcs}; {L15_MIB_ENV}={env.get(L15_MIB_ENV)!r}) -- this boot could never hold a "
            "single row. Set the per-card override, card = budget ordinal (c0 = the 5090), e.g. "
            f"{L15_MIB_ENV}=\"c1=7616,c2=1792\", or turn {L15_MASTER_ENV} off.")


def post_line(p: L15Post) -> str:
    """The boot line, one per card: what is held there and what it cost."""
    return (f"L15-POST card={p.card} mib={p.mib} src={p.src} "
            f"experts_rows_traded={p.experts_rows_traded}")


def residue_without_hold(residue_mib: int, held_mib: Optional[int]) -> int:
    """L15-13c: strip the L1.5 hold out of a D dormant-residue measurement.

    With the L1.5 hold on, D's kv_cache keeps its held rows mapped while
    asleep, so the residue NVML reads at D's sleep already contains the hold.
    The planner ALSO subtracts the hold as its own ``l15`` post (L15-01b,
    ``budgets_from_dc``); a record that keeps the hold would be charged twice
    on the next launch.  ``held_mib`` is the per-card kv_cache bytes still
    mapped at sleep (``tms_tag_mapped_bytes`` via the adapter's
    ``tag_mapped_bytes``), converted to MiB.  ``None`` (master off, no
    adapter, or no entry for the card) leaves the residue untouched.
    """
    if held_mib is None:
        return residue_mib
    return max(0, residue_mib - int(held_mib))
