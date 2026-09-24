# SPDX-License-Identifier: Apache-2.0
"""Der Kreditbedarf des Wakes, als EINE Rechnung: Planer-Riegel und Ordnung.

fnFL2 H14, 24.09. Die bewiesene Flip-Form (FR_P 0.26/0.45/0.39, FR_D
0.06/0.44/0.365) flippt seit x104 stabil. Sobald die Residenz steigt, stirbt
der ERSTE Wake (D schlaeft, P wacht) nach 4 s an W109 (x114c, x114d). Der
Planer preiste die Residenz (H8, W122) und P's Puffer (H5), aber keinen
Kreditbedarf des Wakes -- der Dry-Run liess Formen durch, die am ersten Wake
deterministisch im Zyklus enden.

DIE MECHANIK (am Code, nicht am Log geraten):

* Schlaefer ``s`` (Karte ``c``) laeuft je Tag ``t`` der gemeinsamen Ordnung
  ``deposit(t) -> pause(t) -> credit.publish(t)``
  (``weight_updater.py``, Schleife in ``release_memory_occupation``).
  ``deposit(t)`` zu einem Waker auf EINER ANDEREN Karte geht ueber einen
  BAR1-Ring (4 x 11/32 MiB) und endet erst, wenn der Waker ihn abholt
  (``bar1_lanes``: "no 'free' for batch 0" = der Kollektor hat nicht
  angefangen). Zum Waker auf DERSELBEN Karte ist es ein D2D-Staging
  (``WEG2-SEQ ... on-card D2D staging``), das OHNE Kollektor endet, aber bis
  zum Collect Karten-VRAM haelt.
* Waker ``w`` (Karte ``c``) laeuft dieselbe Ordnung (``resume_order`` =
  ``pause_order``, weg2xsn84): je eigenem Tag ``VramCredit.wait_for`` --
  Kredit kommt NUR aus den Pausen des co-lokierten Schlaefers -- dann
  ``resume(t)``, dann ``collect(t)``. Fremde Tags sind No-ops.
* Also: der Schlaefer auf Karte A kann seinen Tag ``u`` (Ziel Karte B) erst
  pausieren, wenn Waker B den Kredit fuer ``u`` hat; Waker B bekommt ihn nur
  aus den Pausen des Schlaefers B; der steht an einem Tag, dessen Ziel Waker A
  ist, und Waker A wartet auf Schlaefer A. Das ist W109, und es ist eine
  Eigenschaft von ORDNUNG x TAG-GROESSEN x FREIEM VRAM je Karte -- also vor
  dem Laden rechenbar.

``simulate`` spielt genau diese Ordnung als Fixpunkt durch (Schlaefer und
Waker rechnen je Karte gegen dieselben Gates wie ``VramCredit.wait_for``);
``credit_order`` sucht eine Ordnung, in der immer ein Waker Kredit hat, und
laesst eine zyklusfreie Ordnung UNVERAENDERT; ``verdict_lines`` ist der
Riegel-Satz (W126) mit der Arithmetik je Karte.

Rein (kein torch, kein NVML), damit Launcher und Front es ohne CUDA rechnen.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Die Verweigerung (Dry-Run und Front). W122 = Residenz ueber Budget (H8),
#: W123..W125 belegt auf dem Layout-Switch-Zweig (4e11cd11ad).
REFUSAL_CODE = "W126 Weg2WakeCreditCycle"

#: Praefix der Planer-/Front-Zeilen, EIN Marker fuer ``grep -c``.
MARKER = "WAKE-CREDIT (#H14)"


@dataclass(frozen=True)
class WakeCard:
    """Eine Karte im Flip: der co-lokierte Schlaefer und Waker.

    ``free_mib``: Treiber-frei beim Flip-Start (nach der KV-Pause des
    Schlaefers) -- der ``driver_free`` der Front-Zeile ``WEG2-FLIP-ORDER``.
    ``floor_mib``: der Korridor-Floor des WAKERS auf dieser Karte
    (``WEG2-CREDIT-FLOOR group=<Waker>``), gegen den ``wait_for`` gradet.
    ``release_mib``: je Tag, was die Pause des Schlaefers freigibt
    (``tms_tag_bytes``, dieselbe Zahl, die ``credit.publish`` publiziert).
    ``demand_mib``: je Tag, was ``resume`` des Wakers braucht (0/fehlt =
    No-op fuer diesen Waker). ``oncard_mib``: je Tag, was der Schlaefer fuer
    den co-lokierten Waker auf DIESER Karte staged (``WEG2-SEQ lane=cN
    phase=deposit ... slot_bytes``; der Ring bleibt im Leg allokiert).
    """

    card: int
    free_mib: float
    floor_mib: float
    release_mib: Mapping[str, float]
    demand_mib: Mapping[str, float]
    oncard_mib: Mapping[str, float] = field(default_factory=dict)
    sleeper: str = ""
    waker: str = ""


@dataclass(frozen=True)
class CardState:
    """Die Arithmetik einer Karte am Ende (oder im Stillstand) des Laufs."""

    card: int
    freed_mib: float
    consumed_mib: float
    staged_mib: float
    #: der Tag, an dem der Waker steht (None = fertig), sein Bedarf und was
    #: das Gate bietet (``max`` der beiden Wege von ``wait_for``)
    waker_tag: Optional[str]
    need_mib: float
    offer_mib: float
    #: der Tag, an dem der Schlaefer steht, und die Zielkarten, deren Waker
    #: den Kredit dafuer noch nicht hat
    sleeper_tag: Optional[str]
    sleeper_waits_on: Tuple[int, ...]
    #: kleinste Luft ueber alle Grants dieser Karte (offer - need)
    min_headroom_mib: Optional[float]
    min_headroom_tag: Optional[str]

    @property
    def deficit_mib(self) -> float:
        return max(0.0, self.need_mib - self.offer_mib) if self.waker_tag else 0.0


@dataclass(frozen=True)
class WakeRun:
    complete: bool
    order: Tuple[str, ...]
    cards: Tuple[CardState, ...]
    #: der geschlossene Zyklus (Waker-Karte -> Schlaefer-Tag -> Ziel-Karte ...)
    #: als ``(schlaefer_karte, ziel_karte, tag)``; leer, wenn fertig oder wenn
    #: der Stillstand kein Zyklus ist (dann ist es reiner Kreditmangel)
    chain: Tuple[Tuple[int, int, str], ...]

    def card(self, c: int) -> CardState:
        for cs in self.cards:
            if cs.card == int(c):
                return cs
        raise KeyError(c)


def _owners(cards: Sequence[WakeCard], tag: str) -> List[int]:
    return [wc.card for wc in cards if float(wc.demand_mib.get(tag, 0.0) or 0.0) > 0.0]


#: Die On-card-Lane haelt so viele Staging-Puffer allokiert (``WEG2-SEQ
#: prime lane=cN drained=1 depth=2``; persist/reuse, im Leg nie frei).
STAGING_DEPTH = 2
#: ``SGLANG_WEG2_WAKE_COLLECT_WORKERS`` (Default 2): der Waker darf den Kredit
#: fuer Position k erst holen, wenn die Collects bis k-1-n fertig sind
#: (weg2xsn110-Schranke, ``_wake_futs[-(n+1)].result()``).
COLLECT_RUNAHEAD = 2


def simulate(order: Sequence[str], cards: Sequence[WakeCard], *,
             depth: int = STAGING_DEPTH,
             collect_runahead: Optional[int] = COLLECT_RUNAHEAD,
             double_staging: bool = False) -> WakeRun:
    """Die Ordnung als Fixpunkt: wie weit kommen Schlaefer und Waker?

    Waker je Karte, der Reihe nach: No-op-Tags laufen durch; ein eigener Tag
    wird gewaehrt, wenn EINER der beiden Wege von ``VramCredit.wait_for``
    traegt -- (a) Zaehler ``freed - consumed - staged >= need`` (das Staging
    ist auf dem Zaehler gebucht, xsn269) UND allozierbar ``phys - floor >=
    need``, oder (b) die Karte selbst ``phys - floor >= need`` (xsn290/x37;
    H11: das live Staging fehlt schon in der Free-Lesung) -- mit
    ``phys = free + freed - consumed - staged``.

    Schlaefer je Karte, der Reihe nach: staged den On-card-Anteil fuer den
    co-lokierten Eigentuemer (sofort, ohne Kollektor), wartet dann, bis JEDER
    Eigentuemer auf einer ANDEREN Karte den Tag gewaehrt bekommen hat (BAR1:
    der Deposit endet erst mit dem Collect), pausiert (``freed += release``).
    Das On-card-Staging ist ein Ring aus ``depth`` Puffern, die im Leg
    allokiert BLEIBEN (``WEG2-SEQ persist ... reuse``): live sind die
    ``depth`` groessten bisher gestagten Tags dieser Karte.

    ``collect_runahead`` (n): der Waker holt den Kredit fuer Position k erst,
    wenn die Collects seiner Tags bis Position k-1-n fertig sind; ein Collect
    ist fertig, wenn jeder Schlaefer mit Bytes dieses Tags ihn erreicht hat.
    ``None`` = unbeschraenkt.

    ``double_staging``: der Weg (b) zieht das Staging ein ZWEITES Mal von der
    Free-Lesung ab -- so rechnete ``wait_for`` VOR H11 (2f4dcf6893,
    SGLANG_WEG2_CREDIT_LIVE_STAGING); die Boots x100..x114d laufen so.

    Gegen das Metall gehalten (erster Wake D->P, jeder Boot unter SEINER
    Kredit-Regel): x101..x113 und x115/x116 laufen durch, x100/x114c/x114d
    enden im Zyklus -- 17 von 17 wie gemessen (H14-Batch ueber die Logs; im
    Test die sechs Fixture-Boots x104/x113/x114c/x114d/x115/x116), x114c mit
    genau der W109-Kette seines Logs (``test_weg2_wake_credit_h14``).
    """
    order = list(order)
    n = len(order)
    by = {wc.card: wc for wc in cards}
    owners = {t: _owners(cards, t) for t in order}
    s_idx = {c: 0 for c in by}
    w_idx = {c: 0 for c in by}
    freed = {c: 0.0 for c in by}
    consumed = {c: 0.0 for c in by}
    staged: Dict[int, float] = {c: 0.0 for c in by}
    ring: Dict[int, List[float]] = {c: [] for c in by}
    staged_done: Dict[int, set] = {c: set() for c in by}
    granted: Dict[int, set] = {c: set() for c in by}
    headroom: Dict[int, Tuple[Optional[float], Optional[str]]] = {c: (None, None) for c in by}

    def offer(c: int) -> float:
        wc = by[c]
        st = staged[c]
        phys = wc.free_mib + freed[c] - consumed[c] - st
        counter = freed[c] - consumed[c] - st
        alloc = phys - wc.floor_mib
        path_a = min(counter, alloc)
        path_b = alloc - (st if double_staging else 0.0)
        return max(path_a, path_b)

    holders = {t: [wc.card for wc in cards if float(wc.release_mib.get(t, 0.0) or 0.0) > 0.0]
               for t in order}

    def collected(c: int, j: int) -> bool:
        t = order[j]
        if float(by[c].demand_mib.get(t, 0.0) or 0.0) <= 0.0:
            return True
        return t in granted[c] and all(s_idx[h] >= j for h in holders[t])

    progress = True
    while progress:
        progress = False
        for c, wc in by.items():
            while w_idx[c] < n:
                t = order[w_idx[c]]
                need = float(wc.demand_mib.get(t, 0.0) or 0.0)
                if need <= 0.0:
                    w_idx[c] += 1
                    progress = True
                    continue
                if collect_runahead is not None and not all(
                        collected(c, j) for j in range(0, w_idx[c] - int(collect_runahead))):
                    break
                o = offer(c)
                if o + 1e-9 < need:
                    break
                h, _ = headroom[c]
                if h is None or o - need < h:
                    headroom[c] = (o - need, t)
                consumed[c] += need
                granted[c].add(t)
                w_idx[c] += 1
                progress = True
        for c, wc in by.items():
            while s_idx[c] < n:
                t = order[s_idx[c]]
                rel = float(wc.release_mib.get(t, 0.0) or 0.0)
                own = owners[t]
                if rel <= 0.0 and not own:
                    s_idx[c] += 1
                    progress = True
                    continue
                if c in own and t not in staged_done[c]:
                    staged_done[c].add(t)
                    b = float(wc.oncard_mib.get(t, 0.0) or 0.0)
                    if b > 0.0:
                        ring[c].append(b)
                        # the on-card lane keeps its last ``depth`` staging
                        # buffers ALLOCATED (persist/reuse, never freed inside
                        # the leg): its live bytes are the largest ``depth``
                        # tags it has staged so far
                        staged[c] = sum(sorted(ring[c], reverse=True)[:depth])
                        progress = True
                cross = [o for o in own if o != c]
                if rel > 0.0 and any(t not in granted[o] for o in cross):
                    break
                freed[c] += rel
                s_idx[c] += 1
                progress = True
    complete = all(s_idx[c] >= n and w_idx[c] >= n for c in by)
    states = []
    for c, wc in by.items():
        wt = order[w_idx[c]] if w_idx[c] < n else None
        st_tag = order[s_idx[c]] if s_idx[c] < n else None
        waits = tuple(o for o in owners.get(st_tag, []) if o != c and st_tag not in granted[o]) if st_tag else ()
        h, ht = headroom[c]
        states.append(CardState(
            card=c,
            freed_mib=freed[c],
            consumed_mib=consumed[c],
            staged_mib=staged[c],
            waker_tag=wt,
            need_mib=float(wc.demand_mib.get(wt, 0.0) or 0.0) if wt else 0.0,
            offer_mib=offer(c) if wt else 0.0,
            sleeper_tag=st_tag,
            sleeper_waits_on=waits,
            min_headroom_mib=h,
            min_headroom_tag=ht,
        ))
    chain: Tuple[Tuple[int, int, str], ...] = ()
    if not complete:
        chain = _cycle(states)
    return WakeRun(complete=complete, order=tuple(order), cards=tuple(states), chain=chain)


def _cycle(states: Sequence[CardState]) -> Tuple[Tuple[int, int, str], ...]:
    """Waker-Karte c wartet auf Schlaefer c; Schlaefer c wartet auf den Grant
    des Wakers auf Karte d fuer seinen Tag -- der erste geschlossene Weg."""
    st = {s.card: s for s in states}
    for start in st:
        if not st[start].waker_tag:
            continue
        path: List[Tuple[int, int, str]] = []
        seen: List[int] = []
        cur = start
        while cur not in seen:
            seen.append(cur)
            s = st[cur]
            if not s.waker_tag or not s.sleeper_waits_on:
                break
            nxt = s.sleeper_waits_on[0]
            path.append((cur, nxt, str(s.sleeper_tag)))
            cur = nxt
        else:
            i = seen.index(cur)
            return tuple(path[i:])
    return ()


#: Tags, die ihren Platz am ENDE behalten: der Basis-Tag schliesst den Schlaf
#: (``weights_family_tags``), die Front haengt ``rest`` hinten an.
PINNED_TAIL = ("weights",)


def credit_order(order: Sequence[str], cards: Sequence[WakeCard], *,
                 pinned_tail: Sequence[str] = PINNED_TAIL,
                 **sim) -> Tuple[List[str], WakeRun, str]:
    """Eine Ordnung, in der immer ein Waker Kredit hat -- oder die gegebene.

    ZUERST die gegebene Ordnung simulieren: laeuft sie durch, kommt sie
    UNVERAENDERT zurueck (die bewiesene Form behaelt ihren Pfad Byte fuer
    Byte). Nur wenn sie im Zyklus endet, gierig neu: je Schritt der erste Tag
    in der gegebenen Prioritaet, dessen Praefix-Simulation durchlaeuft (kein
    Vorlauf der Schlaefer ueber das Praefix hinaus -- das ist die konservative
    Seite: Vorlauf gibt nur Pausen dazu). Die fertige Ordnung wird noch
    einmal VOLL simuliert (mit Vorlauf und Staging); nur wenn DIE durchlaeuft,
    wird sie genommen, sonst bleibt die gegebene stehen und der Verdikt nennt
    den Zyklus. Rueckgabe ``(ordnung, lauf, warum)``.
    """
    order = list(order)
    base = simulate(order, cards, **sim)
    if base.complete:
        return order, base, "credit order: the given order funds every wake step (unchanged)"
    tail = [t for t in order if t in set(pinned_tail)]
    prefix: List[str] = []
    rest = [t for t in order if t not in set(pinned_tail)]
    while rest:
        pick = None
        for cand in rest:
            if simulate(prefix + [cand], cards, **sim).complete:
                pick = cand
                break
        if pick is None:
            return order, base, (
                "credit order: NO order funds the wake (stuck after %s of %d tags) -- "
                "given order kept" % (len(prefix), len(order)))
        prefix.append(pick)
        rest.remove(pick)
    prefix += tail
    full = simulate(prefix, cards, **sim)
    if not full.complete:
        return order, base, (
            "credit order: the greedy order %s still ends in a cycle under run-ahead "
            "staging -- given order kept" % prefix)
    return prefix, full, (
        "credit order: the given order ends in a credit cycle %s; reordered so every "
        "wake step is funded" % _chain_text(base.chain))


def _chain_text(chain: Sequence[Tuple[int, int, str]]) -> str:
    if not chain:
        return "(no cycle: plain credit shortage)"
    return " -> ".join(
        "sleeper@card%d blocked depositing %s to waker@card%d" % (s, t, d) for s, d, t in chain)


def card_line(wc: WakeCard, cs: CardState) -> str:
    """Die Arithmetik EINER Karte, woertlich."""
    head = "card%d %s/%s: free %.0f - floor %.0f" % (
        cs.card, wc.sleeper or "sleeper", wc.waker or "waker", wc.free_mib, wc.floor_mib)
    if cs.waker_tag is None:
        h = ("" if cs.min_headroom_mib is None else
             ", engste Luft %.0f MiB bei %s" % (cs.min_headroom_mib, cs.min_headroom_tag))
        return head + (" + freigegeben %.0f - verbraucht %.0f -> FERTIG%s"
                       % (cs.freed_mib, cs.consumed_mib, h))
    txt = head + (" + freigegeben %.0f - verbraucht %.0f - staged %.0f -> Angebot %.0f "
                  "fuer %s (Bedarf %.0f) = FEHLT %.0f MiB"
                  % (cs.freed_mib, cs.consumed_mib, cs.staged_mib, cs.offer_mib,
                     cs.waker_tag, cs.need_mib, cs.deficit_mib))
    if cs.sleeper_tag and cs.sleeper_waits_on:
        txt += "; Schlaefer steht vor der Pause von %s (Deposit zu card%s ohne Kollektor)" % (
            cs.sleeper_tag, ",".join(str(x) for x in cs.sleeper_waits_on))
    return txt


def verdict_lines(order: Sequence[str], cards: Sequence[WakeCard], *, label: str,
                  reorder: bool, **sim) -> Tuple[List[str], Optional[str], List[str]]:
    """``(zeilen, verweigerung_oder_None, gewaehlte_ordnung)`` fuer den Riegel.

    ``reorder``: ob die Runtime die Kredit-Ordnung faehrt
    (``SGLANG_WEG2_FLIP_ORDER_CREDIT``). Dann verweigert der Riegel nur, wenn
    auch die Kredit-Ordnung keinen Weg findet; sonst schon, wenn die gegebene
    Ordnung im Zyklus endet.
    """
    by = {wc.card: wc for wc in cards}
    base = simulate(order, cards, **sim)
    chosen, run, why = (credit_order(order, cards, **sim) if reorder
                        else (list(order), base, "credit order OFF (SGLANG_WEG2_FLIP_ORDER_CREDIT=0)"))
    lines = ["%s %s: order %s -- %s" % (MARKER, label, list(chosen), why)]
    for cs in run.cards:
        lines.append("%s %s %s" % (MARKER, label, card_line(by[cs.card], cs)))
    if run.complete:
        return lines, None, chosen
    stuck = [cs for cs in run.cards if cs.waker_tag is not None and cs.deficit_mib > 0]
    refusal = (
        "%s: %s -- der erste Wake endet deterministisch %s: %s. Je Karte: %s. Die "
        "Form ist so nicht flipbar (keine Ordnung gefunden, in der ein co-lokierter "
        "Schlaefer den Tag seines Wakers vor dem ersten Deposit ohne Kollektor "
        "freigibt); Residenz senken oder die Tags kleiner schneiden."
        % (REFUSAL_CODE, label,
           "im Kreditzyklus" if run.chain else "im Kreditmangel",
           _chain_text(run.chain),
           " | ".join(card_line(by[cs.card], cs) for cs in stuck) or "-")
    )
    return lines, refusal, chosen


# ---------------------------------------------------------------------------
# Messung: eine Referenz aus den Logs EINES Boots (der erste Wake, D -> P)
# ---------------------------------------------------------------------------

_RX_ORDER = re.compile(
    r"WEG2-FLIP-ORDER epoch=0 src=D driver_free=(\{[^}]*\}) pause_order=(\[[^\]]*\])")
_RANK = r"\[(?:[0-9-]+ [0-9:]+ )?"
_RX_P_RELEASE = re.compile(
    _RANK + r"PP(\d+)\] WEG2-DC-BREAKDOWN stage=release tags=\['kv_cache', 'weights_0'"
    r".*? tms_paused \d+ (\{[^}]*\})")
_RX_D_RELEASE = re.compile(
    _RANK + r"TP(\d+)\] WEG2-DC-BREAKDOWN stage=release tags=\['kv_cache', 'cuda_graph'\]"
    r".*? tms_resident \d+ (\{[^}]*\})")
_RX_FLOOR = re.compile(_RANK + r"PP(\d+)\] WEG2-CREDIT-FLOOR card=\S+ group=P floor_mib=(\d+)")
_RX_ROWS = re.compile(
    _RANK + r"(PP|TP)(\d+)\] MoE expert-offload active on layer \d+: .*?\(buffer=(\d+),")
_RX_ONCARD = re.compile(
    _RANK + r"TP(\d+)\] WEG2-SEQ lane=c\d+ phase=deposit handshake=\S+ descs=\d+ slot=\d+ "
    r"slot_bytes=(\d+)")
_RX_TAGTIME = re.compile(_RANK + r"TP(\d+)\] WEG2-SLEEP-TAG-TIME tag=(\w+)")


def _family(d: Mapping[str, float]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in d.items() if str(k).startswith("weights")}


@dataclass(frozen=True)
class WakeReference:
    """Der erste Wake (D schlaeft, P wacht) EINES gemessenen Boots.

    Alles Messung aus den eigenen Zeilen des Boots: ``free`` und ``order``
    (Front ``WEG2-FLIP-ORDER epoch=0``), P-Tags (``tms_paused`` des ersten
    P-Schlafs), D-Tags (``tms_resident`` am Flip-Start), P-Floors
    (``WEG2-CREDIT-FLOOR group=P``), Pufferzeilen je Rang (``MoE
    expert-offload active ... buffer=``), On-card-Staging je D-Rang und Tag
    (``WEG2-SEQ lane=cN phase=deposit ... slot_bytes``). Karten: PP-Stufe s
    und D-Rang s teilen sich eine Karte (Form A: ``p_card[s]``).
    """

    source: str
    free: Mapping[int, float]
    order: Tuple[str, ...]
    p_card: Tuple[int, ...]
    p_tags: Tuple[Mapping[str, float], ...]
    p_rows: Tuple[int, ...]
    p_floor: Tuple[float, ...]
    d_tags: Tuple[Mapping[str, float], ...]
    d_rows: Tuple[int, ...]
    d_oncard: Tuple[Mapping[str, float], ...]


def reference_from_logs(p_text: str, d_text: str, front_text: str, *, source: str,
                        p_card: Sequence[int]) -> WakeReference:
    """Die Referenz aus den drei Logs; fehlt eine Zeile, ``ValueError``."""
    import ast

    m = _RX_ORDER.search(front_text)
    if m is None:
        raise ValueError("%s: keine WEG2-FLIP-ORDER epoch=0 src=D-Zeile" % source)
    free = {int(k): float(v) for k, v in ast.literal_eval(m.group(1)).items()}
    order = tuple(ast.literal_eval(m.group(2)))
    n = len(p_card)
    p_tags: Dict[int, Dict[str, float]] = {}
    for mm in _RX_P_RELEASE.finditer(p_text):
        p_tags.setdefault(int(mm.group(1)), _family(ast.literal_eval(mm.group(2))))
    d_tags: Dict[int, Dict[str, float]] = {}
    for mm in _RX_D_RELEASE.finditer(d_text):
        d_tags.setdefault(int(mm.group(1)), _family(ast.literal_eval(mm.group(2))))
    floors: Dict[int, float] = {}
    for mm in _RX_FLOOR.finditer(p_text):
        floors.setdefault(int(mm.group(1)), float(mm.group(2)))
    rows: Dict[Tuple[str, int], int] = {}
    for text in (p_text, d_text):
        for mm in _RX_ROWS.finditer(text):
            rows.setdefault((mm.group(1), int(mm.group(2))), int(mm.group(3)))
    # the first flip's on-card staging per D rank and tag: the slot_bytes of
    # the c-lane handshakes since that rank's previous SLEEP-TAG-TIME line
    oncard: Dict[int, Dict[str, float]] = {}
    pending: Dict[int, float] = {}
    for line in d_text.splitlines():
        mm = _RX_ONCARD.search(line)
        if mm:
            r = int(mm.group(1))
            pending[r] = pending.get(r, 0.0) + int(mm.group(2)) / float(1 << 20)
            continue
        mm = _RX_TAGTIME.search(line)
        if mm:
            r = int(mm.group(1))
            if pending.get(r):
                oncard.setdefault(r, {}).setdefault(mm.group(2), pending[r])
            pending[r] = 0.0
    missing = [("P", s) for s in range(n) if s not in p_tags] + [
        ("D", r) for r in range(n) if r not in d_tags] + [
        ("floor", s) for s in range(n) if s not in floors] + [
        (g, i) for g in ("PP", "TP") for i in range(n) if (g, i) not in rows]
    if missing:
        raise ValueError("%s: Referenzzeilen fehlen: %s" % (source, missing))
    return WakeReference(
        source=source, free=free, order=order, p_card=tuple(int(c) for c in p_card),
        p_tags=tuple(p_tags[s] for s in range(n)),
        p_rows=tuple(rows[("PP", s)] for s in range(n)),
        p_floor=tuple(floors[s] for s in range(n)),
        d_tags=tuple(d_tags[r] for r in range(n)),
        d_rows=tuple(rows[("TP", r)] for r in range(n)),
        d_oncard=tuple(dict(oncard.get(r, {})) for r in range(n)),
    )


def wake_cards(ref: WakeReference) -> List[WakeCard]:
    """Die Karten des Referenz-Wakes (D schlaeft, P wacht), wie gemessen.

    Tags, deren On-card-Staging der Referenz-Boot nicht mehr erreicht hat (er
    starb vorher), bekommen den Median-Anteil ``staged/tag`` DIESES Rangs aus
    seinen gemessenen Einzel-Eigentuemer-Tags, anteilig am Bedarf der
    co-lokierten Stufe (geteilte Tags wie ``weights_9`` gehen an zwei Stufen)."""
    out: List[WakeCard] = []
    for s, card in enumerate(ref.p_card):
        dem = dict(ref.p_tags[s])
        rel = dict(ref.d_tags[s])
        meas = dict(ref.d_oncard[s])
        ratios = sorted(
            meas[t] / rel[t] for t in meas
            if rel.get(t, 0) > 0 and sum(1 for q in ref.p_tags if q.get(t, 0) > 0) == 1)
        r = ratios[len(ratios) // 2] if ratios else 0.0
        onc: Dict[str, float] = {}
        for t, need in dem.items():
            if t in meas:
                onc[t] = meas[t]
            else:
                tot = sum(q.get(t, 0.0) for q in ref.p_tags) or need
                onc[t] = rel.get(t, 0.0) * r * need / tot
        out.append(WakeCard(card=int(card), free_mib=float(ref.free[int(card)]),
                            floor_mib=float(ref.p_floor[s]), release_mib=rel, demand_mib=dem,
                            oncard_mib=onc, sleeper="D TP%d" % s, waker="P PP%d" % s))
    return out


# ---------------------------------------------------------------------------
# Planer: die geplante Residenz als Delta auf die gemessene Referenz
# ---------------------------------------------------------------------------


def tag_layers_by_stage(p_split: Sequence[int], chunk_layers: int, n_tags: int
                        ) -> List[Dict[str, int]]:
    """Je PP-Stufe: wie viele Layer jedes ``weights_<k>`` auf ihr liegen."""
    out: List[Dict[str, int]] = [{} for _ in p_split]
    bounds = []
    start = 0
    for L in p_split:
        bounds.append((start, start + int(L)))
        start += int(L)
    for k in range(int(n_tags)):
        lo, hi = k * int(chunk_layers), (k + 1) * int(chunk_layers)
        for s, (a, b) in enumerate(bounds):
            nl = max(0, min(hi, b) - max(lo, a))
            if nl:
                out[s]["weights_%d" % k] = nl
    return out


def planned_cards(ref: WakeReference, *, p_rows: Sequence[int], d_rows: Sequence[int],
                  slot_mib: float, p_split: Sequence[int], chunk_layers: int,
                  n_layers: int) -> List[WakeCard]:
    """Die Karten des ersten Wakes der GEPLANTEN Form.

    Kein zweites Modell: die Referenz ist gemessen, die Aenderung ist genau
    die Pufferregel (H8), Zeilen x Layer x ``slot_mib``:

    * P-Tag (Stufe s, Tag t) += (Zeilen_plan - Zeilen_ref) x Layer von t auf s x slot
    * D-Tag (Rang r, Chunk-Tag t) += (Zeilen_plan - Zeilen_ref) x chunk_layers x slot
    * frei beim Flip-Start (Karte von Rang r) -= (Zeilen_plan - Zeilen_ref) x
      n_layers x slot (D haelt beim Flip-Start seinen ganzen Puffer; P
      schlaeft, sein Rest ist der der Referenz)
    * On-card-Staging skaliert mit dem D-Tag.
    """
    base = wake_cards(ref)
    n_tags = int(math.ceil(int(n_layers) / int(chunk_layers))) if chunk_layers else 0
    layers = tag_layers_by_stage(p_split, chunk_layers, n_tags)
    out: List[WakeCard] = []
    for s, wc in enumerate(base):
        dp = int(p_rows[s]) - int(ref.p_rows[s])
        dd = int(d_rows[s]) - int(ref.d_rows[s])
        dem = {t: v + dp * layers[s].get(t, 0) * float(slot_mib)
               for t, v in wc.demand_mib.items()}
        rel = {}
        for t, v in wc.release_mib.items():
            k = t.split("_", 1)[1] if "_" in t else ""
            rel[t] = v + dd * (int(chunk_layers) if k.isdigit() else 0) * float(slot_mib)
        onc = {t: v * (rel[t] / wc.release_mib[t] if wc.release_mib.get(t) else 1.0)
               for t, v in wc.oncard_mib.items()}
        out.append(WakeCard(card=wc.card,
                            free_mib=wc.free_mib - dd * int(n_layers) * float(slot_mib),
                            floor_mib=wc.floor_mib, release_mib=rel, demand_mib=dem,
                            oncard_mib=onc, sleeper=wc.sleeper, waker=wc.waker))
    return out


#: Die eingebaute Referenz: fnFL2x114d (d7e3ece8bb, 24.09. 01:55Z), der erste
#: Wake D->P bis zu seinem Zyklus -- gemessen ist alles, was der Planer
#: braucht (free/Ordnung/Tags/Floors/Zeilen/On-card), und die D-Residenz
#: 0.207/0.55/0.45 ist genau die der Zielformen, also traegt die D-Seite kein
#: Delta. Per Test an ``fixtures/wake_credit_h14/fnFL2x114d.*.lines``
#: gebunden (``reference_from_logs`` == diese Konstante), auffrischbar mit
#: ``--wake-credit-reference-logs P.log,D.log,front.log``.
REFERENCE_FNFL2X114D = WakeReference(
    source='fnFL2x114d',
    free={0: 2745, 1: 7308, 2: 2747},
    order=(
        'weights_0', 'weights_9', 'weights_14', 'weights_1', 'weights_10', 'weights_15',
        'weights_2', 'weights_11', 'weights_3', 'weights_12', 'weights_4', 'weights_13',
        'weights_5', 'weights_6', 'weights_7', 'weights_8', 'weights_draft', 'weights',
    ),
    p_card=(1, 0, 2),
    p_tags=(
        {'weights': 630, 'weights_0': 2398, 'weights_1': 2484, 'weights_2': 2354,
         'weights_3': 2354, 'weights_4': 2364, 'weights_5': 2422, 'weights_6': 2354,
         'weights_7': 2354, 'weights_8': 2364, 'weights_9': 1664},
        {'weights_9': 1152, 'weights_10': 3338, 'weights_11': 3274, 'weights_12': 3284,
         'weights_13': 1142},
        {'weights': 1866, 'weights_13': 2222, 'weights_14': 3338, 'weights_15': 3274,
         'weights_draft': 2770},
    ),
    p_rows=(263, 391, 391),
    p_floor=(1055, 1095, 858),
    d_tags=(
        {'weights': 2494, 'weights_0': 1092, 'weights_1': 1146, 'weights_2': 1088,
         'weights_3': 1026, 'weights_4': 1038, 'weights_5': 1082, 'weights_6': 1088,
         'weights_7': 1026, 'weights_8': 1038, 'weights_9': 1082, 'weights_10': 1088,
         'weights_11': 1026, 'weights_12': 1038, 'weights_13': 1082, 'weights_14': 1088,
         'weights_15': 1026, 'weights_draft': 1556},
        {'weights': 2, 'weights_0': 964, 'weights_1': 1064, 'weights_2': 964,
         'weights_3': 964, 'weights_4': 964, 'weights_5': 964, 'weights_6': 964,
         'weights_7': 964, 'weights_8': 964, 'weights_9': 964, 'weights_10': 964,
         'weights_11': 964, 'weights_12': 964, 'weights_13': 964, 'weights_14': 964,
         'weights_15': 964},
        {'weights': 2, 'weights_0': 964, 'weights_1': 1064, 'weights_2': 964,
         'weights_3': 964, 'weights_4': 964, 'weights_5': 964, 'weights_6': 964,
         'weights_7': 964, 'weights_8': 964, 'weights_9': 964, 'weights_10': 964,
         'weights_11': 964, 'weights_12': 964, 'weights_13': 964, 'weights_14': 964,
         'weights_15': 964},
    ),
    d_rows=(84, 128, 128),
    d_oncard=(
        {'weights_0': 547.736, 'weights_1': 509.168, 'weights_2': 509.168,
         'weights_3': 509.168, 'weights_4': 515.861, 'weights_9': 337.214},
        {'weights_9': 193.442, 'weights_10': 580.327, 'weights_11': 580.327,
         'weights_12': 580.327, 'weights_13': 193.442},
        {'weights_14': 580.327, 'weights_15': 580.327},
    ),
)

#: Fuer welche Form die Referenz gilt: Modell, P-Schnitt, Chunk-Laenge, Karten
#: in Stufenfolge, D-Experten-Ratio. Weicht eine davon ab, ENTFAELLT der
#: Riegel mit Namen (eine fremde Geometrie ist kein Delta der Pufferregel).
REFERENCE_KEY_FNFL2X114D = {
    "model": "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    "p_split": (29, 11, 8),
    "chunk_layers": 3,
    "p_card": (1, 0, 2),
    "d_ratio": "183,137,168",
}


@dataclass(frozen=True)
class WakeCreditPlan:
    """Was der Launcher druckt, ob er verweigert, und was die Front bekommt."""

    lines: Tuple[str, ...]
    refusal: Optional[str]
    #: ``{"D->P": [{"card", "release", "demand", "oncard"}, ...]}`` fuer
    #: ``front --wake-credit-plan`` (free/floor liest die Front live), oder
    #: ``None``, wenn der Riegel entfiel.
    front_plan: Optional[Dict[str, List[Dict[str, object]]]] = None


def plan_wake_credit(*, model: str, p_split: Sequence[int], chunk_layers: int,
                     n_layers: int, p_card: Sequence[int], d_ratio: str,
                     p_rows: Sequence[int], d_rows: Sequence[int], slot_mib: float,
                     label: str, reorder: bool, double_staging: bool,
                     reference: Optional[WakeReference] = None,
                     reference_key: Optional[Mapping[str, object]] = None) -> WakeCreditPlan:
    """Der Planer-Riegel: der erste Wake der geplanten Form, je Karte gerechnet.

    ``p_rows``/``d_rows``: die Pufferzeilen je Stufe/Rang aus der Pufferregel
    (H8, ``expert_residency.buffer_rows``); ``slot_mib``: MiB je Zeile und Layer
    (``FRACTION-SOLVE``). ``reorder``: die Front faehrt die Kredit-Ordnung
    (SGLANG_WEG2_FLIP_ORDER_CREDIT); ``double_staging``: die Waker ziehen das
    Staging doppelt ab (SGLANG_WEG2_CREDIT_LIVE_STAGING=0).
    """
    import os

    ref = reference if reference is not None else REFERENCE_FNFL2X114D
    key = dict(reference_key if reference_key is not None else REFERENCE_KEY_FNFL2X114D)
    have = {
        "model": os.path.basename(os.path.normpath(str(model))),
        "p_split": tuple(int(x) for x in p_split),
        "chunk_layers": int(chunk_layers),
        "p_card": tuple(int(x) for x in p_card),
        "d_ratio": ",".join("%g" % float(x) for x in str(d_ratio).split(",") if x.strip()),
    }
    diff = [k for k in key if key[k] != have.get(k)]
    if diff:
        return WakeCreditPlan(lines=(
            "%s %s ENTFAELLT: die Referenz %s gilt fuer %s, diese Form hat %s -- eine "
            "fremde Geometrie ist kein Delta der Pufferregel; Referenz DIESER Form per "
            "--wake-credit-reference-logs P.log,D.log,front.log messen"
            % (MARKER, label, ref.source, {k: key[k] for k in diff},
               {k: have.get(k) for k in diff}),), refusal=None)
    cards = planned_cards(ref, p_rows=p_rows, d_rows=d_rows, slot_mib=slot_mib,
                          p_split=p_split, chunk_layers=chunk_layers, n_layers=n_layers)
    head = (
        "%s %s: erster Wake D->P gegen die Referenz %s (gemessen: free %s, P-Zeilen %s, "
        "D-Zeilen %s), Delta = Pufferregel (Zeilen x Layer x %.3f MiB): P-Zeilen %s, "
        "D-Zeilen %s; Waker-Gates wie VramCredit.wait_for (%s), Staging-Ring %d, "
        "Collect-Vorlauf %d"
        % (MARKER, label, ref.source, dict(ref.free), list(ref.p_rows), list(ref.d_rows),
           float(slot_mib), list(p_rows), list(d_rows),
           "Staging doppelt abgezogen" if double_staging else "H11: Staging nur gebucht-nicht-live",
           STAGING_DEPTH, COLLECT_RUNAHEAD))
    lines, refusal, _chosen = verdict_lines(ref.order, cards, label=label, reorder=reorder,
                                            double_staging=double_staging)
    front_plan = {"D->P": [
        {"card": wc.card, "release": dict(wc.release_mib), "demand": dict(wc.demand_mib),
         "oncard": dict(wc.oncard_mib), "sleeper": wc.sleeper, "waker": wc.waker}
        for wc in cards]}
    return WakeCreditPlan(lines=(head,) + tuple(lines), refusal=refusal,
                          front_plan=front_plan)


def front_order(pause_order: Sequence[str], plan_cards: Sequence[Mapping[str, object]], *,
                free_mib: Mapping[int, float], floor_mib: Mapping[int, float],
                double_staging: bool) -> Tuple[List[str], str]:
    """Die Front: die Kredit-Ordnung fuer DIESEN Flip, gegen die live
    gemessenen ``free``/Floors und die Tag-Tabelle des Planers.

    Rueckgabe ``(ordnung, warum)``; eine unvollstaendige Tabelle (Karte ohne
    free-Lesung, Tag der Ordnung ohne Eintrag) laesst die Ordnung stehen und
    sagt warum -- nie eine halbe Rechnung."""
    cards: List[WakeCard] = []
    for pc in plan_cards:
        c = int(pc["card"])  # type: ignore[arg-type]
        if c not in free_mib or c not in floor_mib:
            return list(pause_order), "credit order SKIPPED: card %d has no free/floor reading" % c
        cards.append(WakeCard(card=c, free_mib=float(free_mib[c]), floor_mib=float(floor_mib[c]),
                              release_mib=dict(pc.get("release") or {}),  # type: ignore[arg-type]
                              demand_mib=dict(pc.get("demand") or {}),  # type: ignore[arg-type]
                              oncard_mib=dict(pc.get("oncard") or {}),  # type: ignore[arg-type]
                              sleeper=str(pc.get("sleeper", "")), waker=str(pc.get("waker", ""))))
    known = set()
    for wc in cards:
        known |= set(wc.release_mib) | set(wc.demand_mib)
    unknown = [t for t in pause_order if t not in known]
    if unknown:
        return list(pause_order), "credit order SKIPPED: tags %s not in the plan table" % unknown
    order, run, why = credit_order(pause_order, cards, double_staging=double_staging)
    by = {wc.card: wc for wc in cards}
    detail = " | ".join(card_line(by[cs.card], cs) for cs in run.cards)
    if not run.complete:
        why = "%s (model: %s) -- %s" % (REFUSAL_CODE, why, detail)
    else:
        why = "%s -- %s" % (why, detail)
    return order, why
