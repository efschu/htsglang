# SPDX-License-Identifier: Apache-2.0
"""L1.5 planner post -- the pure decision layer (L15-PLAN-0930 2.4, AP L15-01).

L1.5 holds D's sleep residue in the brach VRAM of the P layout: the planner
carves a post ``l15`` out of every P budget, sized from the MEASURED P awake
peak (record ``P_AWAKE_PEAK_MIB``), never from a reserve -- keep spans are
fixed per sleep, so the planner sizes them against what P actually needed
(L15-PLAN-0930 2.4, "Elastic").  This module answers, from numbers handed in,
how much each card holds and under which provenance.  It decides; it never launches: no launcher import, no
torch, no record I/O.  The wiring into ``launcher.budgets_from_dc`` and
``vram_plan_view._p_group`` is the separate L15-01b.

Switches (code default off; off means the launch stays byte for byte today's):
  SGLANG_WEG2_L15          master
  SGLANG_WEG2_HOT_HANDOVER hot handover (2.0); refused on --dual-layout like L1.5
  SGLANG_WEG2_L15_MIB      ``auto`` | ``<card key>=<mib>,...`` operator override,
                           shown as OVERRIDE in the vram_plan; card keys see
                           ``resolve_l15_mib`` (identity or budget ordinal)

L1.5 is a 27B-ONLY feature (user correction 2026-10-02): the 27B fits the
VRAM whole, so its free VRAM is context/cache; on NF free VRAM belongs to the
MoE experts and there is NO L1.5 variant, not even as an option. A launch on
any other line with the master (or the hot handover) on is refused by name
(``W-L15-27B-ONLY``, :func:`refuse_not_27b`), never silently off -- the ranks
read the switch from the environment and would arm the hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import card_identity
from sglang.srt.weg2.l15_pool import POOL_ENV, POOL_SHADOW_ENV, pool_on, pool_shadow_on

L15_MASTER_ENV = "SGLANG_WEG2_L15"
L15_MIB_ENV = "SGLANG_WEG2_L15_MIB"
HOT_HANDOVER_ENV = "SGLANG_WEG2_HOT_HANDOVER"
#: the accepted spellings of "on" (case and blanks tolerated; anything else is off)
_ON_VALUES = ("1", "true", "on")

LINE_QWEN27B = "qwen27b"

#: refusal code: L1.5 and the hot handover are refused on --dual-layout in V1
DUAL_REFUSAL_CODE = "W-L15-DUAL"
NOCAP_REFUSAL_CODE = "W-L15-NOCAP"
#: refusal code: L1.5 / the hot handover on any line but the 27B
NOT27B_REFUSAL_CODE = "W-L15-27B-ONLY"
#: refusal code (HW-GENERIC 1002): a card key of SGLANG_WEG2_L15_MIB that
#: names no card, names one card twice, or mixes ordinal and identity keys
CARDKEY_REFUSAL_CODE = "W-L15-CARDKEY"


def _switch(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key, "") or "").strip().lower() in _ON_VALUES


def master_on(env: Mapping[str, str]) -> bool:
    """The L1.5 master switch; default off (0 = today byte for byte)."""
    return _switch(env, L15_MASTER_ENV)


def handover_on(env: Mapping[str, str]) -> bool:
    """The hot-handover switch (2.0, 27B only); same parsing, default off."""
    return _switch(env, HOT_HANDOVER_ENV)


def parse_l15_mib(value: Optional[str]) -> Tuple[str, Dict[int, int]]:
    """``SGLANG_WEG2_L15_MIB`` -> ``("auto", {})`` or ``("override", {card: mib})``.

    THE RANK-SIDE PARSER: ordinal keys only (identity keys are resolved into
    this form by the launcher, ``resolve_l15_mib``).
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
                "whether a typo meant a hold or the auto mode. Card-identity "
                "keys (GPU-<uuid>=, pci:<bus id>=, nvml<i>=) are resolved by the "
                "launcher (resolve_l15_mib) into this ordinal form before any "
                "rank reads the value")
        out[int(key[1:])] = int(val)
    return ("override", out)


# ---------------------------------------------------------------------------
# HW-GENERIC 1002: card-identity keys
#
# The caps are a per-CARD figure, and ``c<i>`` names the card by its budget
# ordinal (card order, ``card_identity.order_cards``) -- a position that moves
# with the inventory. The launcher therefore also accepts the card by what it
# IS, resolves that ONCE against the ordered cards and hands the ranks the
# canonical ordinal form (``parse_l15_mib`` above stays the only parser the
# ranks run). Key syntax, one per comma-separated pair, ``<key>=<mib>``:
#
#   c<i>            budget ordinal (legacy; D rank i sits on ordinal i)
#   GPU-<uuid>      NVML UUID as ``nvidia-smi -L`` prints it (case-insensitive)
#   pci:<bus id>    PCI bus id, ``00000000:02:00.0``, ``0000:02:00.0`` or
#                   ``02:00.0`` (domain 0); hex digits case-insensitive
#   nvml<i>         NVML index (host enumeration; changes with a container mask)
#
# Refused BY NAME (CARDKEY_REFUSAL_CODE): ordinal and identity keys mixed in
# one value, a key that names no card of this launch, two keys naming one
# card, a malformed key or MiB figure. Identity kinds may be mixed with each
# other (two of them naming one card is the duplicate refusal).

_KIND_ORDINAL = "ordinal"
_KIND_UUID = "uuid"
_KIND_PCI = "pci"
_KIND_NVML = "nvml"


def _cardkey_refusal(value: Optional[str], why: str) -> ValueError:
    return ValueError(
        f"{CARDKEY_REFUSAL_CODE}: {L15_MIB_ENV}={value!r}: {why}. Keys: c<ordinal>= "
        "(budget ordinal) OR card identity GPU-<uuid>= / pci:<bus id>= / nvml<index>= "
        "(not both kinds in one value); refusing rather than holding a card 0 MiB "
        "silently")


def _parse_bdf(text: str) -> Optional[Tuple[int, int, int, int]]:
    """``[domain:]bus:device.function`` (hex) -> ints; None when malformed."""
    parts = str(text).strip().split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    if len(parts) != 3:
        return None
    dom, bus, devfn = parts
    dev, dot, fn = devfn.partition(".")
    if not dot:
        return None
    try:
        return (int(dom, 16), int(bus, 16), int(dev, 16), int(fn, 16))
    except ValueError:
        return None


def _key_kind(key: str) -> Optional[str]:
    low = key.lower()
    if key.startswith("c") and key[1:].isdigit():
        return _KIND_ORDINAL
    if low.startswith("gpu-") and len(key) > 4:
        return _KIND_UUID
    if low.startswith("pci:") and _parse_bdf(key[4:]) is not None:
        return _KIND_PCI
    if low.startswith("nvml") and key[4:].isdigit():
        return _KIND_NVML
    return None


def parse_l15_mib_keys(value: Optional[str]) -> Tuple[str, List[Tuple[str, str, int]]]:
    """``SGLANG_WEG2_L15_MIB`` -> ``("auto", [])`` or ``("override",
    [(kind, key, mib), ...])`` in the written order; ``kind`` is one of
    ordinal / uuid / pci / nvml. Raises ``ValueError`` (W-L15-CARDKEY) on a
    malformed pair and on ordinal and identity keys mixed in one value."""
    text = ("" if value is None else str(value)).strip()
    if text == "" or text.lower() == "auto":
        return ("auto", [])
    out: List[Tuple[str, str, int]] = []
    for part in text.split(","):
        key, sep, val = part.strip().partition("=")
        key, val = key.strip(), val.strip()
        kind = _key_kind(key) if sep else None
        if kind is None or not val.isdigit():
            raise _cardkey_refusal(value, f"malformed pair {part.strip()!r}")
        out.append((kind, key, int(val)))
    kinds = {k for k, _, _ in out}
    if _KIND_ORDINAL in kinds and len(kinds) > 1:
        raise _cardkey_refusal(
            value, "ordinal keys ("
            + ", ".join(k for kd, k, _ in out if kd == _KIND_ORDINAL)
            + ") mixed with card-identity keys ("
            + ", ".join(k for kd, k, _ in out if kd != _KIND_ORDINAL) + ")")
    return ("override", out)


def _card_attr(card: Any, name: str) -> Any:
    if isinstance(card, Mapping):
        return card.get(name)
    return getattr(card, name, None)


def _card_label(card: Any) -> str:
    """One card by identity: ``card_identity.describe`` plus the UUID."""
    try:
        text = card_identity.describe(card)
    except Exception:  # noqa: BLE001 -- a hand-built card: the uuid still names it
        text = f"nvml{_card_attr(card, 'nvml_index')}"
    return f"{text} uuid={_card_attr(card, 'uuid') or '?'}"


@dataclass(frozen=True)
class L15MibResolution:
    """What the launcher hands the ranks, and how each key found its card.

    ``value`` is the SGLANG_WEG2_L15_MIB the ranks read: the input unchanged
    when it is auto/absent or already the ordinal form (byte-identical); the
    canonical ``c<i>=<mib>`` form (ordinal ascending) when identity keys were
    resolved. ``rows`` = (key as written, ordinal, mib) in the written order."""

    value: Optional[str]
    mode: str
    rewritten: bool
    rows: Tuple[Tuple[str, int, int], ...]


def resolve_l15_mib(value: Optional[str], ordered_cards: Sequence[Any],
                    pci_by_uuid: Optional[Mapping[str, str]] = None) -> L15MibResolution:
    """Resolve every card key of ``value`` against the cards in card order
    (``ordered_cards[i]`` = budget ordinal i, as ``order_cards`` returns
    them). A card is read duck-typed: ``uuid``, ``nvml_index`` and, for a
    ``pci:`` key, ``pci_bus_id`` (or ``pci_by_uuid[uuid]``). Raises
    ``ValueError`` (W-L15-CARDKEY) naming the bad token -- a key meant for a
    card never turns into a silent 0."""
    mode, pairs = parse_l15_mib_keys(value)
    if mode == "auto":
        return L15MibResolution(value=value, mode=mode, rewritten=False, rows=())
    cards = list(ordered_cards)
    pci_by_uuid = dict(pci_by_uuid or {})
    rows: List[Tuple[str, int, int]] = []
    named: Dict[int, str] = {}
    identity = False
    for kind, key, mib in pairs:
        if kind == _KIND_ORDINAL:
            ordinal = int(key[1:])
            if ordinal >= len(cards):
                raise _cardkey_refusal(
                    value, f"{key!r} names budget ordinal {ordinal}, this launch has "
                    f"{len(cards)} card(s) (ordinals 0..{len(cards) - 1})")
        else:
            identity = True
            hits = []
            for i, c in enumerate(cards):
                uuid = str(_card_attr(c, "uuid") or "")
                if kind == _KIND_UUID:
                    ok = uuid.lower() == key.lower()
                elif kind == _KIND_NVML:
                    idx = _card_attr(c, "nvml_index")
                    ok = idx is not None and int(idx) == int(key[4:])
                else:
                    bus = _card_attr(c, "pci_bus_id") or pci_by_uuid.get(uuid)
                    if not bus:
                        raise _cardkey_refusal(
                            value, f"{key!r}: no PCI bus id is known for card ordinal {i} "
                            f"({uuid or '?'}) -- name it by GPU-<uuid> instead")
                    ok = _parse_bdf(str(bus)) == _parse_bdf(key[4:])
                if ok:
                    hits.append(i)
            if len(hits) != 1:
                raise _cardkey_refusal(
                    value, f"{key!r} matches {len(hits) or 'no'} card(s) of this launch "
                    "(exactly one needed); the cards are: "
                    + "; ".join(f"c{i} = {_card_label(c)}" for i, c in enumerate(cards)))
            ordinal = hits[0]
        if ordinal in named:
            raise _cardkey_refusal(
                value, f"{named[ordinal]!r} and {key!r} both name card ordinal {ordinal}")
        named[ordinal] = key
        rows.append((key, ordinal, int(mib)))
    if not identity:
        return L15MibResolution(value=value, mode=mode, rewritten=False, rows=tuple(rows))
    canon = ",".join(f"c{o}={m}" for _, o, m in sorted(rows, key=lambda r: r[1]))
    return L15MibResolution(value=canon, mode=mode, rewritten=True, rows=tuple(rows))


def mib_cards_line(res: L15MibResolution, ordered_cards: Sequence[Any],
                   source: str = "env") -> str:
    """The launcher line naming the mapping key -> ordinal -> card, and the
    value the ranks read."""
    cards = list(ordered_cards)
    body = "; ".join(
        f"{key} -> c{o} -> {_card_label(cards[o]) if o < len(cards) else '?'} mib={m}"
        for key, o, m in res.rows) or "(none)"
    return (f"L15-MIB-CARDS source={source} mode={res.mode} {body} -- ranks read "
            f"{L15_MIB_ENV}={res.value!r}"
            + (" (rewritten from card identity)" if res.rewritten else ""))


def l15_post_mib(p_budget_mib: int, p_awake_peak_mib: Optional[int]) -> Tuple[int, str]:
    """The post from the records: what the P budget holds beyond the measured P
    awake peak, clamped at 0 (a P that exceeds its record (F5) holds less next
    sleep, never negative).  A missing peak is UNMEASURED zero -- the post is
    priced from records only, never estimated."""
    if p_awake_peak_mib is None:
        return (0, "UNMEASURED")
    return (max(0, int(p_budget_mib) - int(p_awake_peak_mib)), "RECORD(P_AWAKE_PEAK_MIB)")


@dataclass(frozen=True)
class L15Post:
    """One card's L1.5 hold: the bytes and where they come from."""

    card: int
    mib: int
    src: str


def resolve_posts(line: str, p_budget_mib: Sequence[int],
                  p_awake_peak_mib: Sequence[Optional[int]],
                  env: Mapping[str, str]) -> List[L15Post]:
    """Per card what L1.5 holds on the given launch line, in order.

    Precedence (fail-fast, no hidden mode mixing): master off -> every card 0
    under OFF; else an override decides the cards it names (OVERRIDE) and
    holds 0 on the rest (OVERRIDE-UNNAMED); else the post comes from the
    records.  Only the 27B line exists (any other line raises: L1.5 is
    27B-only, see the module docstring).
    """
    n = len(p_budget_mib)
    if len(p_awake_peak_mib) != n:
        raise ValueError(f"resolve_posts: {len(p_awake_peak_mib)} awake peaks for "
                         f"{n} budgets -- a partial vector never prices a card")
    norm = str(line).strip().lower()
    if norm != LINE_QWEN27B:
        raise ValueError(f"resolve_posts: launch line {line!r} has no L1.5 "
                         f"(27B-only: {LINE_QWEN27B!r})")
    if not master_on(env):
        return [L15Post(card=i, mib=0, src="OFF") for i in range(n)]
    mode, override = parse_l15_mib(env.get(L15_MIB_ENV))
    posts: List[L15Post] = []
    for i in range(n):
        if mode == "override":
            if i in override:
                mib, src = override[i], "OVERRIDE"
            else:
                mib, src = 0, "OVERRIDE-UNNAMED"
        else:
            mib, src = l15_post_mib(int(p_budget_mib[i]), p_awake_peak_mib[i])
        posts.append(L15Post(card=i, mib=mib, src=src))
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
                                   (HOT_HANDOVER_ENV, handover_on(env)),
                                   (POOL_SHADOW_ENV, pool_shadow_on(env)),
                                   (POOL_ENV, pool_on(env))) if on]
    if not armed:
        return None
    return (f"{DUAL_REFUSAL_CODE}: --dual-layout with {', '.join(armed)} is refused in V1: "
            "the dual layout never sleeps P, so L1.5 has no sleep to hold into (the card "
            "KV ledger already shares that room) and the hot handover has no flip to hand "
            "over across. Turn the switch off or run without --dual-layout; refusing to "
            "boot a launch whose L1.5 would silently do nothing.")


def refuse_not_27b(profile: str, env: Mapping[str, str]) -> Optional[str]:
    """L1.5 is 27B-only (user correction 2026-10-02): on any other launch
    line (NF: free VRAM = experts) the master or the hot handover switch is a
    launch error, refused BY NAME -- a silent off would still let every rank
    read SGLANG_WEG2_L15=1 from its environment and arm the hold."""
    if str(profile).strip().lower() == LINE_QWEN27B:
        return None
    armed = [name for name, on in ((L15_MASTER_ENV, master_on(env)),
                                   (HOT_HANDOVER_ENV, handover_on(env)),
                                   (POOL_SHADOW_ENV, pool_shadow_on(env)),
                                   (POOL_ENV, pool_on(env))) if on]
    if not armed:
        return None
    return (f"{NOT27B_REFUSAL_CODE}: {', '.join(armed)} on profile {profile!r} is refused: "
            "L1.5 is a 27B-only feature (the 27B fits the VRAM whole, its free VRAM is "
            "context/cache; on NF free VRAM belongs to the MoE experts, there is no L1.5 "
            "variant). Turn the switch off for this profile.")


POOLMASTER_REFUSAL_CODE = "W-L15-POOL-MASTER"


def refuse_pool_without_master(env: Mapping[str, str]) -> Optional[str]:
    """``SGLANG_WEG2_L15_POOL=1`` without the L1.5 master: every pool hook sits
    behind the master gate, so the switch would silently do nothing -- refused
    BY NAME (a silent off would let a boot run believing the pool holds)."""
    if not pool_on(env) or master_on(env):
        return None
    return (f"{POOLMASTER_REFUSAL_CODE}: {POOL_ENV}=1 without {L15_MASTER_ENV}=1 is "
            "refused: the pooled hold is a part of L1.5 and every one of its hooks "
            f"sits behind the master -- set {L15_MASTER_ENV}=1 or turn {POOL_ENV} off.")


def refuse_no_caps(posts: Sequence[L15Post], env: Mapping[str, str],
                   cards: Optional[Sequence[Any]] = None) -> Optional[str]:
    """The no-cap refusal (N3f): with the master on, a launch whose every card
    holds 0 MiB can never retain a single row -- N3c (10012013) and N3e
    (10012112) ran exactly that (SGLANG_WEG2_L15=1 without
    SGLANG_WEG2_L15_MIB: every post UNMEASURED 0, every D cap 0) and spent a
    whole boot window. Refused BY NAME with the fix in the text; master off
    or any card > 0 -> None. ``cards`` (card order, optional) names every
    budget ordinal by its card identity in the text."""
    if not master_on(env):
        return None
    if any(int(p.mib) > 0 for p in posts):
        return None
    srcs = ",".join(sorted({p.src for p in posts})) or "none"
    which = ("; ".join(f"c{i} = {_card_label(c)}" for i, c in enumerate(cards))
             if cards else "c<i> = the i-th card in card order, the card D rank i sits on")
    return (f"{NOCAP_REFUSAL_CODE}: {L15_MASTER_ENV}=1 but every card's L1.5 post is 0 MiB "
            f"(src={srcs}; {L15_MIB_ENV}={env.get(L15_MIB_ENV)!r}) -- this boot could never hold a "
            "single row. Set the per-card override keyed by card identity, e.g. "
            f"{L15_MIB_ENV}=\"GPU-<uuid>=<mib>,...\" (or pci:<bus id>=<mib>, nvml<index>=<mib>), "
            f"or by budget ordinal, e.g. {L15_MIB_ENV}=\"c1=<mib>,c2=<mib>\" ({which}); "
            f"or turn {L15_MASTER_ENV} off.")


def post_line(p: L15Post) -> str:
    """The boot line, one per card: what is held there."""
    return f"L15-POST card={p.card} mib={p.mib} src={p.src}"


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
