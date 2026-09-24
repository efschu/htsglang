# SPDX-License-Identifier: Apache-2.0
"""Der Kredit des Wakes P->D, mit der Zeit gerechnet (fnFL2 H34).

H14 (``wake_credit.py``) rechnet den ersten Wake D->P als Fixpunkt: laeuft die
Ordnung durch oder endet sie im Kreditzyklus (W109)? Den Wake P->D rechnete
niemand -- und genau dort ist nvml2 knapp: im P->D-Wake von fnFL2x141 wartet
D TP2 (3080, nvml2) 73/83/94 ms an ``weights_4`` und 472/427/455 ms an
``weights_6`` auf VRAM-Kredit (die drei kurzen P->D-Flips), im 97k-Flip 125 ms
an ``weights_6``.

DIE MECHANIK (am Code, ``weight_updater.py``, beide Legs gleichzeitig, C9):

* Schlaefer P, Stufe ``s`` auf Karte ``p_card[s]``, laeuft die Ordnung SERIELL:
  je eigenem Tag ``deposit(t) -> pause(t) -> credit.publish(t)``. ``deposit``
  schiebt die Bytes je D-Rang ueber eine Lane: zur ANDEREN Karte ein BAR1-Ring
  (4 x 3 MiB), der erst leerlaeuft, wenn der Abholer (D-Rang) ``t`` gewaehrt
  und gemappt hat; zum co-lokierten D-Rang ein D2D-Staging (``WEG2-SEQ
  lane=cN``), das ohne Abholer endet, dessen zwei groesste Puffer aber bis zum
  Leg-Ende VRAM halten. ``pause(t)`` gibt die Bytes waehrend ``pause_ms``
  frei (Treiber-frei steigt dabei, der Zaehler springt am Ende).
* Waker D, Rang ``r`` auf Karte ``d_card[r]``, laeuft DIESELBE Ordnung (C9:
  die Ordnung gilt fuer das gesammelte Leg-Paar). Vor dem Kredit fuer Position
  ``k`` wartet er auf die Collects bis ``k-3`` (``_wake_futs[-(n+1)]``, n=2),
  dann ``VramCredit.wait_for``: seit H11 gewaehrt der Weg (b), sobald die
  Karte ``free - floor >= need`` haelt, also ``free0 + freigegeben(t) -
  verbraucht - staging(t) - floor >= need``. Dann ``resume(t)``, dann der
  Collect im Hintergrund.
* Ein Abholer, der spaet kommt, haelt die Lane des Schlaefers: die Pause des
  Schlaefers fuer ``t`` -- und damit seine ganze Kette -- wartet auf ihn.
  So wird Kreditwarten auf nvml2 zur Flipzeit: x141 D TP2 ``weights_6`` bekam
  Kredit bei +1221 ms, PP0 hatte ``weights_6`` bei +1086 ms begonnen; seine
  Lane p1 (PP0->TP2) wartete 161 ms und die Kette von PP0, der kritische
  Pfad, kam 88 ms spaeter an (deposit 203 statt ~115 ms).

``simulate_pd`` spielt das in Millisekunden durch (ein Takt je ms, keine
Parallelitaet ausser der gemessenen) und liefert je D-Rang Beginn, Kredit,
Warten, Luft je Tag und das Leg-Ende; ``plan_wake_credit_pd`` rechnet die
GEPLANTE Form als Delta der Pufferregel auf eine gemessene Referenz (x141 mit
Draft auf P, x144 in der H25-Form) und druckt die Dry-Run-Zeilen
``WAKE-CREDIT (#H14) P->D card<N> ...``; ``best_order_pd`` sucht unter den
Umstellungen der Baender EINER Quelle die Ordnung mit dem kuerzesten Leg und
nimmt sie nur bei messbarem Gewinn (sonst bleibt die gegebene Byte fuer Byte).

Rein (kein torch, kein NVML): Launcher, Front und Tests rechnen es ohne CUDA.
"""

from __future__ import annotations

import ast
import datetime
import itertools
import math
import re
import statistics
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2.wake_credit import MARKER, REFUSAL_CODE, tag_layers_by_stage

#: Richtung in jeder Zeile, damit ``grep 'P->D'`` die neue Rechnung trennt.
DIRECTION = "P->D"

#: ``SGLANG_WEG2_WAKE_COLLECT_WORKERS`` (Default 2): Kredit fuer Position k erst,
#: wenn die Collects bis k-1-n fertig sind (weg2xsn110).
COLLECT_RUNAHEAD = 2
#: Die On-card-Lane haelt ihre zwei groessten Staging-Puffer bis zum Leg-Ende
#: (``WEG2-SEQ persist ... reuse``; x141 TP2: 314 + 472 -> 472 + 472 MiB).
STAGING_DEPTH = 2
#: Collect-Threads je D-Rang: ``SGLANG_WEG2_WAKE_COLLECT_WORKERS`` (2) plus
#: ``SGLANG_WEG2_WAKE_COLLECT_SPARE`` (1). Ein Collect belegt seinen Thread, bis
#: alle Lanes seines Tags leer sind; eine BAR1-Lane laeuft erst leer, wenn ihr
#: Collect laeuft (x141 seq 3: PP0 weights_3 Lane p0 wartete 108 ms auf TP1,
#: dessen Threads PP2s langsame Lane p5 und PP1s Tags hielten).
COLLECT_WORKERS = 3
#: Ein SPAETER Abholer: seine Lane laeuft erst ab seinem Collect leer, und
#: zwar in diesem Bruchteil ihrer ungestoerten Dauer -- ALLEIN (die anderen
#: Lanes desselben Deposits sind durch: die Kopien teilen sich die Quelle nicht
#: mehr) halb so lang, GETEILT (eine andere Lane laeuft zugleich) ganz.
#: Gemessen: x141 seq 3 PP0 weights_6 Lane p1 allein 197-161 = 36 ms bei
#: ungestoert 103; x145 seq 3 PP0 weights_2 p1 allein 144-84 = 60 bei ~110,
#: weights_3 p1 geteilt (p0 wartete selbst 77 ms) 187-94 = 93 bei ~105.
LATE_ALONE_FRACTION = 0.5
LATE_SHARED_FRACTION = 1.0
#: Eine Lane gilt als ungestoert, wenn ihr eigenes ``wait_ms`` hoechstens das
#: ist (die Ring-Kredite eines rechtzeitigen Abholers kosten 0-23 ms und
#: gehoeren zur Dauer); sonst wird ihre Dauer aus der Rate DERSELBEN Lane im
#: selben Flip geschaetzt (``total_ms`` je MiB, Median der ungestoerten Tags).
LANE_UNBLOCKED_WAIT_MS = 30
#: Eine umgestellte Ordnung wird nur genommen, wenn sie das Leg um mindestens
#: so viel verkuerzt (Modellgenauigkeit, siehe Test gegen x141: +-10 ms).
MIN_GAIN_MS = 20.0
#: Harte Obergrenze des Takts (ms); danach gilt der Lauf als stehend.
HORIZON_MS = 30000


@dataclass(frozen=True)
class PDLane:
    """Eine Lane eines Tags: Stufe ``stage`` schiebt ``mib`` an Rang ``rank``.

    ``ms``: die ungestoerte Dauer (Abholer rechtzeitig). ``oncard``: D2D auf
    derselben Karte (endet ohne Abholer, haelt Staging); dann ist
    ``collect_ms`` die Kopie des Abholers aus dem Staging."""

    stage: int
    rank: int
    tag: str
    mib: float
    ms: float
    oncard: bool = False
    collect_ms: float = 0.0


@dataclass(frozen=True)
class PDMetal:
    """Was der Referenz-Boot an einem D-Tag gemessen hat (ms ab Order-Zeile)."""

    begin_ms: float
    grant_ms: float
    waited_ms: float
    free_mib: Optional[float]


@dataclass(frozen=True)
class PDReference:
    """Ein gemessener Wake P->D (P schlaeft, D wacht), alles aus den eigenen
    Zeilen des Boots (``pd_reference_from_logs``)."""

    source: str
    order: Tuple[str, ...]
    free: Mapping[int, float]
    p_card: Tuple[int, ...]
    d_floor: Tuple[float, ...]
    p_release: Tuple[Mapping[str, float], ...]
    p_pause_ms: Tuple[Mapping[str, float], ...]
    d_demand: Tuple[Mapping[str, float], ...]
    d_resume_ms: Tuple[Mapping[str, float], ...]
    lanes: Tuple[PDLane, ...]
    p_rows: Tuple[int, ...]
    d_rows: Tuple[int, ...]
    #: residente Zeilen (Puffer ohne Scratch) je Stufe/Rang, aus derselben
    #: ``MoE expert-offload active``-Zeile; sie bestimmen, was der Austausch
    #: traegt (``shared_rows``)
    p_resident: Tuple[int, ...] = ()
    d_resident: Tuple[int, ...] = ()
    #: ungestoerte Deposit-Dauer je (Stufe, Tag): die gemessene ``deposit_ms``,
    #: wenn keine Lane auf ihren Abholer wartete, sonst die laengste
    #: ungestoerte Lane plus der Median-Aufschlag der Stufe
    p_deposit_ms: Tuple[Mapping[str, float], ...] = ()
    #: wann die Legs anfangen (ms ab Order-Zeile): erste Tag-Zeile je Stufe/Rang
    p_start_ms: Tuple[float, ...] = ()
    d_start_ms: Tuple[float, ...] = ()
    #: gemessen je (Rang, Tag); leer fuer eine geplante Form
    metal: Mapping[Tuple[int, str], PDMetal] = field(default_factory=dict)
    #: gemessene Veroeffentlichung je (Stufe, Tag), ms ab Order-Zeile
    metal_publish: Mapping[Tuple[int, str], float] = field(default_factory=dict)

    @property
    def d_card(self) -> Tuple[int, ...]:
        # Form A: D-Rang r teilt sich die Karte mit P-Stufe r
        return tuple(self.p_card)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDTag:
    """Ein D-Tag im Lauf: Beginn (Collect-Schranke und Vorgaenger erfuellt),
    Kredit, Luft beim Kredit (``free - floor - need``)."""

    rank: int
    tag: str
    need_mib: float
    begin_ms: float
    grant_ms: Optional[float]
    headroom_mib: Optional[float]
    free_mib: Optional[float]

    @property
    def waited_ms(self) -> Optional[float]:
        return None if self.grant_ms is None else self.grant_ms - self.begin_ms


@dataclass(frozen=True)
class PDRun:
    complete: bool
    order: Tuple[str, ...]
    tags: Tuple[PDTag, ...]
    #: (Stufe, Tag) -> ms, zu dem der Zaehler springt (Pause fertig)
    publish_ms: Mapping[Tuple[int, str], float]
    #: (Stufe, Tag) -> ms, zu dem die Pause beginnt (Deposit fertig)
    pause_start_ms: Mapping[Tuple[int, str], float]
    #: Leg-Ende: letzter Collect eines D-Rangs / letzte Pause einer P-Stufe
    leg_ms: Optional[float]
    #: stehend: je Rang der Tag, an dem er steht, und was fehlt
    stuck: Tuple[Tuple[int, str, float], ...] = ()

    def tag(self, rank: int, tag: str) -> PDTag:
        for t in self.tags:
            if t.rank == int(rank) and t.tag == tag:
                return t
        raise KeyError((rank, tag))

    def rank_tags(self, rank: int) -> List[PDTag]:
        return [t for t in self.tags if t.rank == int(rank)]

    def wait_sum(self, rank: int) -> float:
        return sum(float(t.waited_ms or 0.0) for t in self.rank_tags(rank) if t.need_mib > 0)

    def worst(self, rank: int) -> Optional[PDTag]:
        ts = [t for t in self.rank_tags(rank) if t.need_mib > 0 and t.grant_ms is not None]
        return max(ts, key=lambda t: (t.waited_ms or 0.0), default=None)

    def tightest(self, rank: int) -> Optional[PDTag]:
        ts = [t for t in self.rank_tags(rank) if t.need_mib > 0 and t.headroom_mib is not None]
        return min(ts, key=lambda t: t.headroom_mib, default=None)


def simulate_pd(ref: PDReference, order: Optional[Sequence[str]] = None, *,
                runahead: int = COLLECT_RUNAHEAD, depth: int = STAGING_DEPTH,
                workers: int = COLLECT_WORKERS,
                alone_fraction: float = LATE_ALONE_FRACTION,
                shared_fraction: float = LATE_SHARED_FRACTION,
                unbounded_cards: Sequence[int] = ()) -> PDRun:
    """Der Wake P->D im Millisekundentakt.

    ``unbounded_cards``: Karten, deren Kredit NIE knapp ist (``floor = -inf``)
    -- der Vergleichslauf, der zeigt, was das Kreditwarten auf dieser Karte
    das Leg kostet."""
    order = list(ref.order if order is None else order)
    n = len(order)
    n_st = len(ref.p_card)
    n_rk = len(ref.d_floor)
    d_card = ref.d_card
    free0 = {int(c): float(v) for c, v in ref.free.items()}
    unb = {int(c) for c in unbounded_cards}
    lanes_of: Dict[Tuple[int, str], List[PDLane]] = {}
    into: Dict[Tuple[int, str], List[PDLane]] = {}
    for ln in ref.lanes:
        lanes_of.setdefault((ln.stage, ln.tag), []).append(ln)
        into.setdefault((ln.rank, ln.tag), []).append(ln)

    # --- P state
    p_k = [0] * n_st
    p_phase: List[Optional[str]] = [None] * n_st
    p_dep0 = [0.0] * n_st
    p_pause0 = [0.0] * n_st
    released_done: Dict[int, float] = {c: 0.0 for c in free0}
    lane_done: Dict[Tuple[int, int, str], float] = {}
    dep_start: Dict[Tuple[int, str], float] = {}
    publish: Dict[Tuple[int, str], float] = {}
    pause_start: Dict[Tuple[int, str], float] = {}
    ring: Dict[int, List[float]] = {c: [] for c in free0}
    # --- D state
    d_k = [0] * n_rk
    d_phase = ["gate"] * n_rk
    # previous resume end; the D leg starts when its first tag line says so
    d_ready = [float(ref.d_start_ms[r]) if ref.d_start_ms else 0.0 for r in range(n_rk)]
    d_begin: Dict[Tuple[int, str], float] = {}
    d_grant: Dict[Tuple[int, str], float] = {}
    d_resume_end: Dict[Tuple[int, str], float] = {}
    d_head: Dict[Tuple[int, str], float] = {}
    d_free: Dict[Tuple[int, str], float] = {}
    collect_end: Dict[Tuple[int, str], float] = {}
    job_start: Dict[Tuple[int, str], float] = {}
    d_queue: List[List[str]] = [[] for _ in range(n_rk)]
    d_running: List[List[str]] = [[] for _ in range(n_rk)]
    consumed: Dict[int, float] = {c: 0.0 for c in free0}

    def rel(s: int, t: str) -> float:
        return float(ref.p_release[s].get(t, 0.0) or 0.0)

    def released(c: int, now: float) -> float:
        v = released_done.get(c, 0.0)
        for s in range(n_st):
            if ref.p_card[s] == c and p_phase[s] == "pause":
                t = order[p_k[s]]
                pz = float(ref.p_pause_ms[s].get(t, 0.0) or 0.0)
                frac = 1.0 if pz <= 0 else min(1.0, max(0.0, (now - p_pause0[s]) / pz))
                v += rel(s, t) * frac
        return v

    def staged(c: int) -> float:
        return sum(sorted(ring.get(c, []), reverse=True)[:depth])

    def phys(c: int, now: float) -> float:
        return free0[c] + released(c, now) - consumed[c] - staged(c)

    def try_collect(r: int, t: str) -> None:
        if (r, t) in collect_end or (r, t) not in job_start:
            return
        js = job_start[(r, t)]
        ends = [js]
        for ln in into.get((r, t), []):
            key = (ln.stage, r, t)
            if ln.oncard:
                if (ln.stage, t) not in dep_start:
                    return
                ends.append(max(dep_start[(ln.stage, t)] + ln.ms, js) + ln.collect_ms)
            else:
                if key not in lane_done:
                    return
                ends.append(lane_done[key])
        collect_end[(r, t)] = max(ends)

    now = 0.0
    last_change = 0.0
    while True:
        changed = True
        while changed:
            changed = False
            # ---------------- P stages
            for s in range(n_st):
                while p_k[s] < n:
                    t = order[p_k[s]]
                    lanes = lanes_of.get((s, t), [])
                    if p_phase[s] is None:
                        if not lanes and rel(s, t) <= 0.0:
                            p_k[s] += 1
                            changed = True
                            continue
                        if ref.p_start_ms and now + 1e-9 < float(ref.p_start_ms[s]):
                            break
                        p_phase[s] = "dep"
                        p_dep0[s] = now
                        dep_start[(s, t)] = now
                        c = ref.p_card[s]
                        for ln in lanes:
                            if ln.oncard and ln.mib > 0:
                                ring.setdefault(c, []).append(float(ln.mib))
                        changed = True
                    if p_phase[s] == "dep":
                        done = True
                        for ln in lanes:
                            key = (s, ln.rank, t)
                            if key in lane_done:
                                continue
                            if ln.oncard:
                                end = p_dep0[s] + ln.ms
                            else:
                                if (ln.rank, t) not in job_start:
                                    done = False
                                    continue
                                j = job_start[(ln.rank, t)]
                                f = alone_fraction
                                for o in lanes:
                                    if o is ln or o.oncard or (o.rank, t) not in job_start:
                                        continue
                                    jo = job_start[(o.rank, t)]
                                    eo = lane_done.get((s, o.rank, t),
                                                       max(p_dep0[s] + o.ms, jo + o.ms))
                                    if jo <= j < eo:
                                        f = shared_fraction
                                        break
                                end = max(p_dep0[s] + ln.ms, j + f * ln.ms)
                            if now + 1e-9 >= end:
                                lane_done[key] = end
                                changed = True
                            else:
                                done = False
                        dep = (float(ref.p_deposit_ms[s].get(t, 0.0) or 0.0)
                               if ref.p_deposit_ms else 0.0)
                        if not done or now + 1e-9 < p_dep0[s] + dep:
                            break
                        p_phase[s] = "pause"
                        p_pause0[s] = now
                        pause_start[(s, t)] = now
                        changed = True
                    if p_phase[s] == "pause":
                        pz = float(ref.p_pause_ms[s].get(t, 0.0) or 0.0)
                        if now + 1e-9 < p_pause0[s] + pz:
                            break
                        released_done[ref.p_card[s]] = released_done.get(ref.p_card[s], 0.0) + rel(s, t)
                        publish[(s, t)] = p_pause0[s] + pz
                        p_phase[s] = None
                        p_k[s] += 1
                        changed = True
            # ---------------- D ranks
            for r in range(n_rk):
                c = d_card[r]
                while d_k[r] < n:
                    t = order[d_k[r]]
                    if d_phase[r] == "gate":
                        if now + 1e-9 < d_ready[r]:
                            break
                        k = d_k[r]
                        if k - 1 - runahead >= 0:
                            gate_tag = order[k - 1 - runahead]
                            try_collect(r, gate_tag)
                            ce = collect_end.get((r, gate_tag))
                            if ce is None or now + 1e-9 < ce:
                                break
                        d_begin[(r, t)] = now
                        d_phase[r] = "credit"
                        changed = True
                    if d_phase[r] == "credit":
                        need = float(ref.d_demand[r].get(t, 0.0) or 0.0)
                        if need > 0.0:
                            floor = -math.inf if c in unb else float(ref.d_floor[r])
                            avail = phys(c, now) - floor
                            if avail + 1e-9 < need:
                                break
                            d_head[(r, t)] = avail - need
                            d_free[(r, t)] = phys(c, now)
                            consumed[c] += need
                        d_grant[(r, t)] = now
                        rs = float(ref.d_resume_ms[r].get(t, 1.0) or 1.0) if need > 0 else 1.0
                        d_resume_end[(r, t)] = now + rs
                        d_ready[r] = now + rs
                        d_queue[r].append(t)
                        d_phase[r] = "gate"
                        d_k[r] += 1
                        changed = True
                        continue
            for r in range(n_rk):
                for t in order:
                    try_collect(r, t)
                # collect threads: FIFO in submission order, a thread is free
                # once its tag's collect has ended
                d_running[r] = [t for t in d_running[r]
                                if (r, t) not in collect_end or collect_end[(r, t)] > now + 1e-9]
                while (d_queue[r] and len(d_running[r]) < max(1, int(workers))
                       and d_resume_end[(r, d_queue[r][0])] <= now + 1e-9):
                    t = d_queue[r].pop(0)
                    job_start[(r, t)] = now
                    d_running[r].append(t)
                    try_collect(r, t)
                    changed = True
            if changed:
                last_change = now
        p_done = all(p_k[s] >= n for s in range(n_st))
        d_done = all(d_k[r] >= n for r in range(n_rk)) and all(
            (r, t) in collect_end for r in range(n_rk) for t in order)
        if p_done and d_done:
            break
        if now - last_change > 5000 or now > HORIZON_MS:
            break
        now += 1.0
    complete = all(p_k[s] >= n for s in range(n_st)) and all(d_k[r] >= n for r in range(n_rk))
    tags: List[PDTag] = []
    for r in range(n_rk):
        for t in order:
            if (r, t) not in d_begin:
                continue
            tags.append(PDTag(rank=r, tag=t, need_mib=float(ref.d_demand[r].get(t, 0.0) or 0.0),
                              begin_ms=d_begin[(r, t)], grant_ms=d_grant.get((r, t)),
                              headroom_mib=d_head.get((r, t)), free_mib=d_free.get((r, t))))
    stuck = []
    if not complete:
        for r in range(n_rk):
            # only a rank standing at its credit gate is SHORT; the others
            # wait on a collect of a rank that is
            if d_k[r] < n and d_phase[r] == "credit":
                t = order[d_k[r]]
                need = float(ref.d_demand[r].get(t, 0.0) or 0.0)
                c = d_card[r]
                short = need - (phys(c, now) - float(ref.d_floor[r]))
                if short > 0:
                    stuck.append((r, t, short))
    leg = None
    if complete:
        leg = max([collect_end.get((r, t), 0.0) for r in range(n_rk) for t in order]
                  + list(publish.values()) + [0.0])
    return PDRun(complete=complete, order=tuple(order), tags=tuple(tags), publish_ms=dict(publish),
                 pause_start_ms=dict(pause_start), leg_ms=leg, stuck=tuple(stuck))


def credit_cost_ms(ref: PDReference, card: int, order: Optional[Sequence[str]] = None) -> Optional[float]:
    """Was das Kreditwarten auf ``card`` das Leg kostet: Leg mit Kredit minus
    Leg mit unbegrenztem Kredit auf dieser Karte (dieselbe Ordnung)."""
    a = simulate_pd(ref, order)
    b = simulate_pd(ref, order, unbounded_cards=(card,))
    if a.leg_ms is None or b.leg_ms is None:
        return None
    return a.leg_ms - b.leg_ms


# ---------------------------------------------------------------------------
# Ordnung
# ---------------------------------------------------------------------------


def _stage_bands(ref: PDReference, order: Sequence[str], stage: int) -> List[int]:
    """Positionen der Ordnung, an denen NUR Stufe ``stage`` Bytes freigibt (ein
    Band, das zwei Stufen teilen -- x141 ``weights_13`` auf PP1 und PP2 --,
    bleibt stehen: es zu schieben verschoebe die Kette der anderen Stufe mit)."""
    out = []
    for i, t in enumerate(order):
        if t == "weights" or float(ref.p_release[stage].get(t, 0.0) or 0.0) <= 0.0:
            continue
        if any(float(ref.p_release[s].get(t, 0.0) or 0.0) > 0.0
               for s in range(len(ref.p_card)) if s != stage):
            continue
        out.append(i)
    return out


def candidate_orders_pd(ref: PDReference, order: Sequence[str], stage: int) -> List[List[str]]:
    """Die Umstellungen EINER Quelle: ihre eigenen Baender tauschen die Plaetze,
    die sie in der gegebenen Ordnung belegen (alle Permutationen), und je ein
    Band wandert an jede andere Position; alle anderen Tags behalten ihre
    Reihenfolge, ``weights`` schliesst den Schlaf und bleibt hinten."""
    order = list(order)
    pos = _stage_bands(ref, order, stage)
    out: List[List[str]] = []
    seen = {tuple(order)}
    bands = [order[i] for i in pos]
    if len(bands) <= 6:
        for perm in itertools.permutations(bands):
            cand = list(order)
            for i, t in zip(pos, perm):
                cand[i] = t
            if tuple(cand) not in seen:
                seen.add(tuple(cand))
                out.append(cand)
    tail = [t for t in order if t == "weights"]
    body = [t for t in order if t != "weights"]
    for b in bands:
        rest = [t for t in body if t != b]
        for j in range(len(rest) + 1):
            cand = rest[:j] + [b] + rest[j:] + tail
            if tuple(cand) not in seen:
                seen.add(tuple(cand))
                out.append(cand)
    return out


def _moved(a: Sequence[str], b: Sequence[str]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def best_order_pd(ref: PDReference, order: Optional[Sequence[str]] = None, *,
                  card: Optional[int] = None, min_gain_ms: float = MIN_GAIN_MS,
                  also: Sequence[PDReference] = ()) -> Tuple[List[str], PDRun, PDRun, str]:
    """Eine Ordnung, deren Leg der Kredit auf der knappsten Karte weniger kostet.

    Rueckgabe ``(ordnung, lauf_neu, lauf_gegeben, warum)`` (Laeufe auf ``ref``).
    Gesucht wird NUR, wenn das Kreditwarten auf ``card`` (Default: die Karte
    mit dem groessten Warten) das Leg der gegebenen Ordnung um mindestens
    ``min_gain_ms`` verlaengert -- die Umstellung soll Kreditwarten wegnehmen,
    nicht die Pipeline neu erfinden. Unter den Umstellungen der Quelle auf
    dieser Karte (``candidate_orders_pd``) gewinnt das kuerzeste Leg; ein
    Gewinn zaehlt hoechstens so viel, wie der Kredit die gegebene Ordnung
    kostet (der Rest waere Modellpipeline, nicht Kredit), und unter Kandidaten
    innerhalb von ``min_gain_ms / 2`` des besten nimmt er den, der am wenigsten
    Plaetze aendert.

    ``also``: weitere Referenzen DERSELBEN Form (z. B. der 97k-Flip neben dem
    kurzen): ein Kandidat, der dort das Leg um mehr als ``min_gain_ms / 4``
    verlaengert oder nicht durchlaeuft, faellt -- eine Ordnung gilt fuer jeden
    Flip, die Front kennt die Flip-Art nicht.

    Die gegebene Ordnung bleibt, wenn sie nicht durchlaeuft (Kreditmangel
    aendert keine Ordnung), wenn der Kredit nichts kostet oder wenn der
    gezaehlte Gewinn unter ``min_gain_ms`` liegt."""
    order = list(ref.order if order is None else order)
    base = simulate_pd(ref, order)
    if not base.complete or base.leg_ms is None:
        return order, base, base, "timed order: the given order does not complete (credit shortage) -- kept"
    if card is None:
        waits = {ref.d_card[r]: base.wait_sum(r) for r in range(len(ref.d_floor))}
        card = max(waits, key=lambda c: waits[c])
    free_run = simulate_pd(ref, order, unbounded_cards=(card,))
    cost = base.leg_ms - (free_run.leg_ms if free_run.leg_ms is not None else base.leg_ms)
    stage = list(ref.p_card).index(int(card))
    if cost < float(min_gain_ms):
        return order, base, base, (
            "timed order: credit on card %d costs the given order's leg %.0f ms (< %.0f) -- "
            "given order kept" % (card, cost, min_gain_ms))
    others = []
    for o in also:
        ob = simulate_pd(o, order)
        others.append((o, ob.leg_ms if ob.complete else None))
    runs: List[Tuple[float, int, List[str], PDRun]] = []
    vetoed = 0
    for cand in candidate_orders_pd(ref, order, stage):
        run = simulate_pd(ref, cand)
        if not run.complete or run.leg_ms is None:
            continue
        bad = False
        for o, leg0 in others:
            if leg0 is None:
                continue
            orun = simulate_pd(o, cand)
            if not orun.complete or orun.leg_ms is None or orun.leg_ms > leg0 + float(min_gain_ms) / 4.0:
                bad = True
                break
        if bad:
            vetoed += 1
            continue
        runs.append((run.leg_ms, _moved(order, cand), cand, run))
    veto_txt = (", %d candidate(s) vetoed by %d other reference(s)" % (vetoed, len(others))
                if others else "")
    if not runs:
        return order, base, base, (
            "timed order: no rearrangement of stage %d completes%s -- given order kept" % (stage, veto_txt))
    best_leg = min(x[0] for x in runs)
    near = [x for x in runs if x[0] <= best_leg + float(min_gain_ms) / 2.0]
    leg, _mv, best, best_run = min(near, key=lambda x: (x[1], x[0]))
    gain = min(base.leg_ms - leg, cost)
    if gain < float(min_gain_ms):
        return order, base, base, (
            "timed order: no rearrangement of stage %d's own bands (card %d) takes >= %.0f ms of "
            "the %.0f ms credit cost off the leg (best %.0f ms%s) -- given order kept"
            % (stage, card, min_gain_ms, cost, base.leg_ms - best_leg, veto_txt))
    return best, best_run, base, (
        "timed order: stage %d's own bands (card %d) rearranged, credit cost %.0f ms, leg %.0f -> "
        "%.0f ms (-%.0f ms, counted -%.0f%s)" % (stage, card, cost, base.leg_ms, leg,
                                                 base.leg_ms - leg, gain, veto_txt))


# ---------------------------------------------------------------------------
# Zeilen
# ---------------------------------------------------------------------------


def card_lines_pd(ref: PDReference, run: PDRun, *, label: str) -> List[str]:
    """Je Karte: free/floor, wann P freigibt, welcher D-Tag wann claimt, engste
    Luft, Warten (Summe, groesstes) und was es das Leg kostet."""
    out: List[str] = []
    for r, c in enumerate(ref.d_card):
        s = list(ref.p_card).index(c)
        rel_total = sum(float(v) for v in ref.p_release[s].values())
        need_total = sum(float(v) for v in ref.d_demand[r].values())
        pubs = sorted((ms, t) for (st, t), ms in run.publish_ms.items() if st == s)
        pub_txt = ", ".join("%s %.0f@%.0f" % (t, float(ref.p_release[s].get(t, 0.0)), ms)
                            for ms, t in pubs)
        head = "%s %s %s card%d D TP%d/P PP%d: free %.0f - floor %.0f + freigegeben %.0f - verbraucht %.0f" % (
            MARKER, DIRECTION, label, c, r, s, float(ref.free[c]), float(ref.d_floor[r]),
            rel_total, need_total)
        tight = run.tightest(r)
        worst = run.worst(r)
        if run.complete:
            txt = head + " -> FERTIG"
            if tight is not None:
                txt += ", engste Luft %.0f MiB bei %s (@%.0f ms)" % (
                    tight.headroom_mib, tight.tag, tight.grant_ms)
            txt += ", Kreditwarten %.0f ms" % run.wait_sum(r)
            if worst is not None and (worst.waited_ms or 0) > 0:
                txt += " (groesstes %.0f ms bei %s: Beginn %.0f, Kredit %.0f)" % (
                    worst.waited_ms, worst.tag, worst.begin_ms, worst.grant_ms)
            cost = credit_cost_ms(ref, c, run.order)
            if cost is not None:
                txt += ", kostet das Leg %.0f ms" % cost
        else:
            st = [x for x in run.stuck if x[0] == r]
            if st:
                txt = head + " -> STEHT bei %s: FEHLT %.0f MiB (Kredit kommt nicht mehr)" % (
                    st[0][1], st[0][2])
            else:
                txt = head + " -> fertig, wartet aber auf eine stehende Karte"
        txt += "; P-Freigaben (MiB@ms) " + (pub_txt or "-")
        out.append(txt)
    return out


def verdict_lines_pd(ref: PDReference, *, label: str, apply: bool,
                     also: Sequence[PDReference] = ()) -> Tuple[List[str], Optional[str], List[str]]:
    """``(zeilen, verweigerung_oder_None, empfohlene_ordnung)`` fuer den Riegel P->D.

    Die Empfehlung (``best_order_pd``) wird IMMER gerechnet und gedruckt;
    ``apply`` sagt nur, ob die Front sie faehrt
    (``SGLANG_WEG2_ENABLE_PD_TIMED_ORDER``). Die Kartenzeilen gelten der
    Ordnung, die die Front tatsaechlich faehrt."""
    chosen, run_new, base, why = best_order_pd(ref, also=also)
    run = run_new if apply else base
    lines = ["%s %s %s: gegebene Ordnung %s -- Leg %s; %s; %s" % (
        MARKER, DIRECTION, label, list(ref.order),
        "%.0f ms" % base.leg_ms if base.leg_ms is not None else "STEHT", why,
        ("Front faehrt die Empfehlung (SGLANG_WEG2_ENABLE_PD_TIMED_ORDER=1)" if apply
         else "Front faehrt die gegebene (SGLANG_WEG2_ENABLE_PD_TIMED_ORDER=0)"))]
    if list(chosen) != list(ref.order):
        lines.append("%s %s %s: empfohlene Ordnung %s -- Leg %.0f ms" % (
            MARKER, DIRECTION, label, list(chosen), run_new.leg_ms))
    lines += card_lines_pd(ref, run, label=label)
    for o in also:
        for tag, order in (("gegeben", list(ref.order)), ("empfohlen", list(chosen))):
            if tag == "empfohlen" and list(chosen) == list(ref.order):
                continue
            r2 = simulate_pd(o, order)
            costs = []
            for r, c in enumerate(o.d_card):
                cc = credit_cost_ms(o, c, order) if r2.wait_sum(r) > 0 else 0.0
                costs.append("card%d %.0f ms (Leg +%s)" % (
                    c, r2.wait_sum(r), "?" if cc is None else "%.0f" % cc))
            leg_txt = ("%.0f ms" % r2.leg_ms if r2.leg_ms is not None else "STEHT (%s)" % (
                "; ".join("D TP%d bei %s FEHLT %.0f MiB" % x for x in r2.stuck) or "Zyklus"))
            lines.append("%s %s %s: Vergleichsflip %s, Ordnung %s -- Leg %s, Kreditwarten je Karte %s" % (
                MARKER, DIRECTION, label, o.source, tag, leg_txt, ", ".join(costs)))
    if run.complete:
        return lines, None, list(chosen)
    detail = (" | ".join("D TP%d steht bei %s, FEHLT %.0f MiB" % x for x in run.stuck)
              or "kein Rang knapp -- die Legs warten im Kreis aufeinander")
    refusal = (
        "%s: %s %s -- der Wake P->D endet deterministisch im Kreditmangel: %s. Die "
        "Karte haelt nach allen Pausen ihrer P-Stufe (minus Staging-Ring und "
        "Korridor-Floor) die D-Tags nicht; keine Ordnung aendert eine Summe. "
        "D-Residenz auf dieser Karte senken oder P-Residenz dort senken."
        % (REFUSAL_CODE, DIRECTION, label, detail))
    return lines, refusal, list(chosen)


@dataclass(frozen=True)
class WakeCreditPlanPD:
    """Was der Launcher druckt, ob er verweigert, und was die Front bekommt:
    ``front_plan`` ``{"P->D": <Fixpunkt-Tabelle>, "P->D-order": {"given",
    "timed"}}`` (``None``, wenn der Riegel entfiel)."""

    lines: Tuple[str, ...]
    refusal: Optional[str]
    front_plan: Optional[Dict[str, object]] = None


def plan_wake_credit_pd(*, model: str, p_split: Sequence[int], chunk_layers: int,
                        n_layers: int, p_card: Sequence[int], d_ratio: str, draft_on_p: bool,
                        p_rows: Sequence[int], d_rows: Sequence[int], slot_mib: float,
                        label: str, apply: bool,
                        p_resident: Optional[Sequence[int]] = None,
                        d_resident: Optional[Sequence[int]] = None,
                        references: Optional[Mapping[str, Mapping[str, object]]] = None,
                        form_keys: Optional[Mapping[str, Mapping[str, object]]] = None,
                        dense_repack: Optional[bool] = None,
                        ) -> WakeCreditPlanPD:
    """Der Planer-Riegel P->D: der Wake der GEPLANTEN Form gegen die gemessene
    Referenz derselben Form (Draft auf P: fnFL2x141, H25: fnFL2x144), Delta =
    Pufferregel. ``p_rows``/``d_rows``: Pufferzeilen je Stufe/Rang (H8);
    ``p_resident``/``d_resident``: deren residente Zeilen (ohne Scratch), aus
    denen die Lane-Bytes folgen (``shared_rows``); ``slot_mib``: MiB je Zeile
    und Layer. ``dense_repack`` (H50): der Baum-Zustand H39 der Gruppen
    (SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL); er waehlt die Referenz mit, weil
    D-Bedarf und P-Freigabe derselben Form sich um die toten Tag-Pool-Bloecke
    unterscheiden (x144 gegen x158: D TP0 20720 gegen 16366 MiB). ``None`` =
    der Zustand vor H39 (die Referenzen, die es vor H50 gab)."""
    import os

    from sglang.srt.weg2 import wake_credit_pd_refs as _refs

    refs = dict(references if references is not None else _refs.REFERENCES)
    keys = dict(form_keys if form_keys is not None else _refs.FORM_KEYS)
    want_repack = bool(dense_repack) if dense_repack is not None else False
    boot = "fnFL2x141" if draft_on_p else "fnFL2x144"
    for name in sorted(keys):
        k = keys[name]
        if (bool(k.get("draft_on_p")) == bool(draft_on_p)
                and bool(k.get("dense_repack", False)) == want_repack):
            boot = name
            break
    have = {
        "model": os.path.basename(os.path.normpath(str(model))),
        "p_split": tuple(int(x) for x in p_split),
        "chunk_layers": int(chunk_layers),
        "p_card": tuple(int(x) for x in p_card),
        "d_ratio": ",".join("%g" % float(x) for x in str(d_ratio).split(",") if x.strip()),
        "draft_on_p": bool(draft_on_p),
        "dense_repack": want_repack,
    }
    key = dict(keys.get(boot) or {})
    diff = [k for k in key if key[k] != have.get(k)]
    main, first = refs.get(boot + "/1"), refs.get(boot + "/0")
    if not key or diff or main is None:
        return WakeCreditPlanPD(lines=(
            "%s %s %s ENTFAELLT: die Referenz %s gilt fuer %s, diese Form hat %s -- eine "
            "fremde Geometrie ist kein Delta der Pufferregel"
            % (MARKER, DIRECTION, label, boot, {k: key.get(k) for k in diff} or "-",
               {k: have.get(k) for k in diff} or "-"),), refusal=None)
    kw = dict(p_rows=p_rows, d_rows=d_rows, slot_mib=slot_mib, p_split=p_split,
              chunk_layers=chunk_layers, n_layers=n_layers, p_resident=p_resident,
              d_resident=d_resident)
    ref = planned_reference_pd(reference_from_dict(main), **kw)
    also = [planned_reference_pd(reference_from_dict(first), **kw)] if first is not None else []
    head = (
        "%s %s %s: Wake P->D (P schlaeft, D wacht, beide Legs zugleich) gegen die Referenz %s "
        "(gemessen: free %s, P-Zeilen %s, D-Zeilen %s), Delta = Pufferregel (Zeilen x Layer x "
        "%.3f MiB): P-Zeilen %s, D-Zeilen %s; Takt 1 ms, Kredit wie VramCredit.wait_for Weg (b) "
        "(free - floor >= need, Staging-Ring %d), Collect-Vorlauf %d, %d Collect-Threads je Rang"
        % (MARKER, DIRECTION, label, boot + "/1", dict(reference_from_dict(main).free),
           list(main["p_rows"]), list(main["d_rows"]), float(slot_mib),  # type: ignore[index]
           list(p_rows), list(d_rows), STAGING_DEPTH, COLLECT_RUNAHEAD, COLLECT_WORKERS))
    lines, refusal, chosen = verdict_lines_pd(ref, label=label, apply=apply, also=also)
    front: Dict[str, object] = {DIRECTION: untimed_table_pd(ref)}
    if list(chosen) != list(ref.order):
        front[DIRECTION + "-order"] = {"given": list(ref.order), "timed": list(chosen)}
    return WakeCreditPlanPD(lines=(head,) + tuple(lines), refusal=refusal, front_plan=front)


def front_timed_order(pause_order: Sequence[str], plan: Optional[Mapping[str, object]]
                      ) -> Tuple[List[str], str]:
    """Die Front: die Empfehlung des Planers fuer DIESEN Wake P->D, nur wenn die
    live gebaute Ordnung genau die ist, die der Planer gerechnet hat (sonst
    gilt seine Rechnung nicht) und die Empfehlung eine Permutation davon ist,
    die den Basis-Tag hinten laesst."""
    rec = (plan or {}).get(DIRECTION + "-order") if plan else None
    if not rec:
        return list(pause_order), "timed order: no planner recommendation for this form"
    given = list(rec.get("given") or [])  # type: ignore[union-attr]
    timed = list(rec.get("timed") or [])  # type: ignore[union-attr]
    if list(pause_order) != given:
        return list(pause_order), (
            "timed order SKIPPED: the live order %s is not the one the planner timed %s"
            % (list(pause_order), given))
    if sorted(timed) != sorted(given) or (given and given[-1] == "weights" and timed[-1] != "weights"):
        return list(pause_order), "timed order SKIPPED: the recommendation is not a permutation"
    return timed, "timed order: planner recommendation applied (fnFL2 H34)"


# ---------------------------------------------------------------------------
# Messung: eine Referenz aus den Logs EINES Boots (ein Wake P->D)
# ---------------------------------------------------------------------------

_TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:,\d+)?)"
_RX_ORDER = re.compile(
    _TS + r"\] INFO weg2\.front: WEG2-FLIP-ORDER epoch=(\d+) src=P driver_free=(\{[^}]*\}) "
    r"pause_order=(\[[^\]]*\])")
_RX_P_RES = re.compile(
    r"PP(\d+)\] WEG2-DC-BREAKDOWN stage=release tags=\['kv_cache', 'cuda_graph'\].*? "
    r"tms_resident \d+ (\{[^}]*\})")
_RX_ROWS = re.compile(r"(PP|TP)(\d+)\] MoE expert-offload active on layer \d+: (\d+)/\d+ experts "
                      r"resident \+ \d+ scratch \(buffer=(\d+),")
_RX_STT = re.compile(
    r"PP(\d+)\] WEG2-SLEEP-TAG-TIME tag=(\w+) deposit_ms=(\d+) sync_ms=\d+ pause_ms=(\d+) "
    r".*?t0=([\d.]+) t=([\d.]+)")
_RX_BAR1 = re.compile(
    r"(PP|TP)(\d+)\] WEG2-BAR1 lane-time lane=p(\d+) phase=(deposit|collect) seq=(\d+)-(\w+) "
    r".*?bytes=(\d+) total_ms=(\d+) wait_ms=(\d+)")
_RX_SEQ = re.compile(
    r"(PP|TP)(\d+)\] WEG2-SEQ lane-time lane=c(\d+) phase=(deposit|collect) units=\d+ "
    r"bytes=(\d+) total_ms=(\d+) wait_ms=(\d+) .*?t=([\d.]+) t0=([\d.]+)")
_RX_BEGIN = re.compile(r"TP(\d+)\] WEG2-RESUME begin tag=(\w+) need_mib=(\d+) ")
_RX_CREDIT = re.compile(
    r"TP(\d+)\] WEG2-VRAM-CREDIT card=\S+ tag=(\w+) waited=(\d+) ms .*?free_mib=(\d+) "
    r".*?corridor_floor_mib=(\d+)")
_RX_WTT = re.compile(r"TP(\d+)\] WEG2-WAKE-TAG-TIME tag=(\w+) resume_ms=(\d+) t0=([\d.]+) t=([\d.]+)")

_MIB = float(1 << 20)


def _epoch_s(ts: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S,%f" if "," in ts else "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.strptime(ts, fmt).replace(tzinfo=datetime.timezone.utc).timestamp()


def pd_reference_from_logs(p_text: str, d_text: str, front_text: str, *, source: str,
                           p_card: Sequence[int], flip: int = 1,
                           window_s: float = 8.0) -> PDReference:
    """Der ``flip``-te Wake P->D (0 = der erste) eines Boots, aus drei Logs.

    Fehlt eine Zeile, ``ValueError`` -- nie eine halbe Referenz."""
    orders = list(_RX_ORDER.finditer(front_text))
    if len(orders) <= flip:
        raise ValueError("%s: nur %d WEG2-FLIP-ORDER src=P-Zeilen, Flip %d fehlt"
                         % (source, len(orders), flip))
    m = orders[flip]
    t_order = _epoch_s(m.group(1))
    free = {int(k): float(v) for k, v in ast.literal_eval(m.group(3)).items()}
    order = tuple(ast.literal_eval(m.group(4)))
    seq = 2 * flip + 1
    n = len(p_card)
    p_rel: Dict[int, Dict[str, float]] = {}
    for mm in _RX_P_RES.finditer(p_text):
        p_rel.setdefault(int(mm.group(1)), {
            str(k): float(v) for k, v in ast.literal_eval(mm.group(2)).items()
            if str(k).startswith("weights")})
    rows: Dict[Tuple[str, int], int] = {}
    resid: Dict[Tuple[str, int], int] = {}
    for text in (p_text, d_text):
        for mm in _RX_ROWS.finditer(text):
            rows.setdefault((mm.group(1), int(mm.group(2))), int(mm.group(4)))
            resid.setdefault((mm.group(1), int(mm.group(2))), int(mm.group(3)))
    lo, hi = t_order - 0.5, t_order + window_s
    # P per tag: start, deposit, pause, publish
    stt: Dict[Tuple[int, str], Tuple[float, float, float, float]] = {}
    for mm in _RX_STT.finditer(p_text):
        t0 = float(mm.group(5))
        if lo <= t0 <= hi:
            stt[(int(mm.group(1)), mm.group(2))] = (
                t0, float(mm.group(3)), float(mm.group(4)), float(mm.group(6)))
    # BAR1 lanes of THIS flip: lane id -> (stage, rank)
    lane_src: Dict[int, int] = {}
    lane_dst: Dict[int, int] = {}
    bar1: Dict[Tuple[int, str], Tuple[float, float, float]] = {}
    for mm in _RX_BAR1.finditer(p_text + "\n" + d_text):
        if int(mm.group(5)) != seq:
            continue
        grp, rk, lane, phase = mm.group(1), int(mm.group(2)), int(mm.group(3)), mm.group(4)
        if grp == "PP" and phase == "deposit":
            lane_src[lane] = rk
            bar1[(lane, mm.group(6))] = (int(mm.group(7)) / _MIB, float(mm.group(8)), float(mm.group(9)))
        elif grp == "TP" and phase == "collect":
            lane_dst[lane] = rk
    # on-card lanes: deposit (P) and collect (D) by time window of the P tag
    seq_dep: Dict[Tuple[int, str], Tuple[float, float]] = {}
    seq_col: Dict[int, List[Tuple[float, float, float]]] = {}
    cl_rank: Dict[int, int] = {}
    for mm in _RX_SEQ.finditer(p_text + "\n" + d_text):
        grp, rk, lane, phase = mm.group(1), int(mm.group(2)), int(mm.group(3)), mm.group(4)
        t0 = float(mm.group(9))
        if not (lo <= t0 <= hi):
            continue
        if grp == "PP" and phase == "deposit":
            for (s, tag), (st0, _dep, _pz, st1) in stt.items():
                if s == rk and st0 - 0.002 <= t0 <= st1:
                    seq_dep[(s, tag)] = (int(mm.group(5)) / _MIB, float(mm.group(6)))
                    break
        elif grp == "TP" and phase == "collect":
            cl_rank[lane] = rk
            seq_col.setdefault(rk, []).append((int(mm.group(5)) / _MIB,
                                               float(mm.group(6)) - float(mm.group(7)), t0))
    # D per tag: demand, waited, free, floor, resume, grant
    demand: Dict[int, Dict[str, float]] = {}
    resume: Dict[int, Dict[str, float]] = {}
    metal: Dict[Tuple[int, str], PDMetal] = {}
    floors: Dict[int, float] = {}
    for mm in _RX_CREDIT.finditer(d_text):
        floors.setdefault(int(mm.group(1)), float(mm.group(5)))
    pend_need: Dict[int, Tuple[str, float]] = {}
    pend_cred: Dict[int, Tuple[str, float, float]] = {}
    for line in d_text.splitlines():
        mb = _RX_BEGIN.search(line)
        if mb:
            pend_need[int(mb.group(1))] = (mb.group(2), float(mb.group(3)))
            pend_cred.pop(int(mb.group(1)), None)
            continue
        mc = _RX_CREDIT.search(line)
        if mc:
            pend_cred[int(mc.group(1))] = (mc.group(2), float(mc.group(3)), float(mc.group(4)))
            continue
        mw = _RX_WTT.search(line)
        if mw:
            r, tag, t0 = int(mw.group(1)), mw.group(2), float(mw.group(4))
            if not (lo <= t0 <= hi):
                continue
            pn = pend_need.get(r)
            if pn is None or pn[0] != tag:
                continue
            demand.setdefault(r, {})[tag] = pn[1]
            resume.setdefault(r, {})[tag] = float(mw.group(3))
            pc = pend_cred.get(r)
            waited, fr = (pc[1], pc[2]) if pc and pc[0] == tag else (0.0, None)
            g = (t0 - t_order) * 1000.0
            metal[(r, tag)] = PDMetal(begin_ms=g - waited, grant_ms=g, waited_ms=waited, free_mib=fr)
    missing = ([("P-tms", s) for s in range(n) if s not in p_rel]
               + [("D-demand", r) for r in range(n) if r not in demand]
               + [("floor", r) for r in range(n) if r not in floors]
               + [(g, i) for g in ("PP", "TP") for i in range(n) if (g, i) not in rows])
    if missing:
        raise ValueError("%s: Referenzzeilen fehlen: %s" % (source, missing))
    # lanes with unblocked duration (rate of the same lane when it had to wait)
    lanes: List[PDLane] = []
    by_lane: Dict[int, List[Tuple[str, float, float, float]]] = {}
    for (lane, tag), (mib, tot, wt) in bar1.items():
        by_lane.setdefault(lane, []).append((tag, mib, tot, wt))
    for lane, items in sorted(by_lane.items()):
        if lane not in lane_src or lane not in lane_dst:
            raise ValueError("%s: Lane p%d ohne Quelle/Ziel im Flip seq %d" % (source, lane, seq))
        rates = [tot / mib for _t, mib, tot, wt in items
                 if wt <= LANE_UNBLOCKED_WAIT_MS and mib > 0]
        rate = statistics.median(rates) if rates else None
        for tag, mib, tot, wt in items:
            if wt <= LANE_UNBLOCKED_WAIT_MS or rate is None:
                ms = tot
            else:
                ms = rate * mib
            lanes.append(PDLane(stage=lane_src[lane], rank=lane_dst[lane], tag=tag,
                                mib=round(mib, 3), ms=round(ms, 3)))
    for (s, tag), (mib, ms) in sorted(seq_dep.items()):
        r = s  # Form A: the on-card collector is the D rank sharing stage s's card
        cols = [x for x in seq_col.get(r, []) if abs(x[0] - mib) < 0.5]
        cms = statistics.median([x[1] for x in cols]) if cols else 0.0
        lanes.append(PDLane(stage=s, rank=r, tag=tag, mib=round(mib, 3), ms=ms, oncard=True,
                            collect_ms=float(cms)))
    # deposit durations: measured where no lane waited on its collector,
    # else the longest unblocked lane plus the stage's median overhead
    blocked = {(lane_src[lane], tag) for (lane, tag), (_m, _tot, wt) in bar1.items()
               if wt > LANE_UNBLOCKED_WAIT_MS and lane in lane_src}
    lane_max: Dict[Tuple[int, str], float] = {}
    for ln in lanes:
        k = (ln.stage, ln.tag)
        lane_max[k] = max(lane_max.get(k, 0.0), ln.ms)
    over: Dict[int, List[float]] = {}
    for (s, tag), (_t0, dep, _pz, _t1) in stt.items():
        if (s, tag) in lane_max and (s, tag) not in blocked:
            over.setdefault(s, []).append(dep - lane_max[(s, tag)])
    p_dep: Dict[int, Dict[str, float]] = {s: {} for s in range(n)}
    for (s, tag), (_t0, dep, _pz, _t1) in stt.items():
        if p_rel.get(s, {}).get(tag, 0.0) <= 0 and (s, tag) not in lane_max:
            continue
        if (s, tag) in blocked:
            dep = lane_max.get((s, tag), 0.0) + max(0.0, statistics.median(over.get(s, [0.0])))
        p_dep[s][tag] = round(float(dep), 3)
    p_start = []
    for s in range(n):
        t0s = [v[0] for (st, _t), v in stt.items() if st == s]
        if not t0s:
            raise ValueError("%s: Stufe %d ohne WEG2-SLEEP-TAG-TIME im Flip" % (source, s))
        p_start.append(round((min(t0s) - t_order) * 1000.0, 3))
    d_start = []
    for r in range(n):
        b = [v.begin_ms for (rk, _t), v in metal.items() if rk == r]
        d_start.append(round(min(b), 3) if b else 0.0)
    pause = {s: {} for s in range(n)}
    publish: Dict[Tuple[int, str], float] = {}
    for (s, tag), (t0, dep, pz, t1) in stt.items():
        if p_rel.get(s, {}).get(tag, 0.0) > 0:
            pause[s][tag] = pz
            publish[(s, tag)] = (t1 - t_order) * 1000.0
    return PDReference(
        source=source, order=order, free=free, p_card=tuple(int(c) for c in p_card),
        d_floor=tuple(floors[r] for r in range(n)),
        p_release=tuple(p_rel[s] for s in range(n)),
        p_pause_ms=tuple(pause[s] for s in range(n)),
        d_demand=tuple(demand[r] for r in range(n)),
        d_resume_ms=tuple(resume[r] for r in range(n)),
        lanes=tuple(sorted(lanes, key=lambda x: (x.stage, x.rank, order.index(x.tag)
                                                  if x.tag in order else 99))),
        p_rows=tuple(rows[("PP", s)] for s in range(n)),
        d_rows=tuple(rows[("TP", r)] for r in range(n)),
        p_resident=tuple(resid[("PP", s)] for s in range(n)),
        d_resident=tuple(resid[("TP", r)] for r in range(n)),
        p_deposit_ms=tuple(p_dep[s] for s in range(n)),
        p_start_ms=tuple(p_start), d_start_ms=tuple(d_start),
        metal=metal, metal_publish=publish,
    )


# ---------------------------------------------------------------------------
# Planer: die geplante Residenz als Delta auf die gemessene Referenz
# ---------------------------------------------------------------------------


def shared_rows(p_resident: Sequence[int], d_resident: Sequence[int]) -> List[List[float]]:
    """Je (Stufe s, Rang r): wie viele residente Zeilen von Rang r der
    Austausch aus Stufe s traegt. Haelt die Stufe mindestens so viele Zeilen
    wie ALLE D-Raenge zusammen, alle; sonst ihre eigenen, anteilig verteilt
    (``expert_map._proportional_subset``) -- der Rest kommt im rearm aus dem
    Store (H30). Gemessen x144 -> x145 (D 64/65 -> 74/85 Zeilen): Stufe 0
    (134 < 141/171) Lane-Bytes x0.952 / x1.079 wie gerechnet 0.953 / 1.078;
    Stufen 1/2 (231/376 >= 171) x1.156 / x1.308 = 74/64, 85/65."""
    tot = float(sum(int(x) for x in d_resident))
    out: List[List[float]] = []
    for ps in p_resident:
        if tot <= 0:
            out.append([0.0 for _ in d_resident])
        elif float(ps) >= tot:
            out.append([float(x) for x in d_resident])
        else:
            out.append([float(ps) * float(x) / tot for x in d_resident])
    return out


def planned_reference_pd(ref: PDReference, *, p_rows: Sequence[int], d_rows: Sequence[int],
                         slot_mib: float, p_split: Sequence[int], chunk_layers: int,
                         n_layers: int, p_resident: Optional[Sequence[int]] = None,
                         d_resident: Optional[Sequence[int]] = None) -> PDReference:
    """Die Referenz der GEPLANTEN Form, Delta = Pufferregel (H8), wie H14:

    * P-Tag (Stufe s, Tag t) += (Zeilen_plan - Zeilen_ref) x Layer von t auf s x slot;
      die Pause skaliert mit den Bytes
    * frei beim Flip-Start (Karte von Stufe s) -= (Zeilen_plan - Zeilen_ref) x
      Layer der Stufe x slot (P haelt beim Flip-Start ihren ganzen Puffer; D
      schlaeft, seine Gewichte sind ungemappt)
    * D-Tag (Rang r, Chunk-Tag t) += (Zeilen_plan - Zeilen_ref) x chunk_layers x slot
    * Lane (s -> r, t): der Experten-Anteil folgt ``shared_rows`` (Zeilen x
      Layer von t auf s x slot), der Rest (Nicht-Experten) bleibt; Dauer,
      On-card-Staging und Deposit skalieren mit den Bytes. Ohne residente
      Zeilen (``p_resident``/``d_resident`` None) bleiben die Lanes.
    """
    n_tags = int(math.ceil(int(n_layers) / int(chunk_layers))) if chunk_layers else 0
    layers = tag_layers_by_stage(p_split, chunk_layers, n_tags)
    free = dict(ref.free)
    rel_out, pz_out, dem_out = [], [], []
    for s, card in enumerate(ref.p_card):
        dp = int(p_rows[s]) - int(ref.p_rows[s])
        rel = {}
        pz = {}
        for t, v in ref.p_release[s].items():
            nv = float(v) + dp * layers[s].get(t, 0) * float(slot_mib)
            rel[t] = nv
            if t in ref.p_pause_ms[s]:
                pz[t] = float(ref.p_pause_ms[s][t]) * (nv / float(v) if v else 1.0)
        rel_out.append(rel)
        pz_out.append(pz)
        free[card] = float(free[card]) - dp * int(p_split[s]) * float(slot_mib)
    for r in range(len(ref.d_floor)):
        dd = int(d_rows[r]) - int(ref.d_rows[r])
        dem = {}
        for t, v in ref.d_demand[r].items():
            k = t.split("_", 1)[1] if "_" in t else ""
            grow = k.isdigit() and float(v) > 0.0
            dem[t] = float(v) + (dd * int(chunk_layers) * float(slot_mib) if grow else 0.0)
        dem_out.append(dem)
    lanes = list(ref.lanes)
    dep = [dict(x) for x in ref.p_deposit_ms]
    if p_resident is not None and d_resident is not None and ref.p_resident and ref.d_resident:
        sh_ref = shared_rows(ref.p_resident, ref.d_resident)
        sh_new = shared_rows(p_resident, d_resident)
        lanes = []
        old_max: Dict[Tuple[int, str], float] = {}
        new_max: Dict[Tuple[int, str], float] = {}
        for ln in ref.lanes:
            nl = layers[ln.stage].get(ln.tag, 0)
            exp_ref = sh_ref[ln.stage][ln.rank] * nl * float(slot_mib)
            exp_new = sh_new[ln.stage][ln.rank] * nl * float(slot_mib)
            mib = max(0.0, float(ln.mib) - exp_ref) + exp_new
            k = mib / float(ln.mib) if ln.mib else 1.0
            nln = replace(ln, mib=round(mib, 3), ms=round(float(ln.ms) * k, 3),
                          collect_ms=round(float(ln.collect_ms) * k, 3))
            lanes.append(nln)
            key = (ln.stage, ln.tag)
            old_max[key] = max(old_max.get(key, 0.0), float(ln.ms))
            new_max[key] = max(new_max.get(key, 0.0), float(nln.ms))
        # the deposit keeps its own overhead; only its longest lane changes
        for (s, t), om in old_max.items():
            if s < len(dep) and t in dep[s]:
                dep[s][t] = round(float(dep[s][t]) + new_max[(s, t)] - om, 3)
    return replace(ref, source="%s+plan" % ref.source, free=free, p_release=tuple(rel_out),
                   p_pause_ms=tuple(pz_out), d_demand=tuple(dem_out), lanes=tuple(lanes),
                   p_deposit_ms=tuple(dep),
                   p_rows=tuple(int(x) for x in p_rows), d_rows=tuple(int(x) for x in d_rows),
                   p_resident=tuple(int(x) for x in (p_resident or ref.p_resident)),
                   d_resident=tuple(int(x) for x in (d_resident or ref.d_resident)),
                   metal={}, metal_publish={})


def untimed_table_pd(ref: PDReference) -> List[Dict[str, object]]:
    """Die Tag-Tabelle fuer ``wake_credit.front_order`` (Fixpunkt, W109-Pruefung
    in der Front) in der Richtung P->D: Karte c, Schlaefer = P-Stufe auf c,
    Waker = D-Rang auf c."""
    out = []
    for r, c in enumerate(ref.d_card):
        s = list(ref.p_card).index(c)
        onc = {ln.tag: ln.mib for ln in ref.lanes if ln.oncard and ln.stage == s}
        out.append({"card": int(c), "release": dict(ref.p_release[s]),
                    "demand": dict(ref.d_demand[r]), "oncard": onc,
                    "sleeper": "P PP%d" % s, "waker": "D TP%d" % r})
    return out


# ---------------------------------------------------------------------------
# Eingebaute Referenzen (gemessen, per Test an die Logzeilen gebunden)
# ---------------------------------------------------------------------------


def reference_to_dict(ref: PDReference) -> Dict[str, object]:
    """Die Referenz ohne Metall-Messwerte als reine Literale (fuer
    ``wake_credit_pd_refs.py`` und den Test, der sie an die Logs bindet)."""
    return {
        "source": ref.source, "order": list(ref.order),
        "free": {int(k): float(v) for k, v in ref.free.items()},
        "p_card": list(ref.p_card), "d_floor": [float(x) for x in ref.d_floor],
        "p_release": [dict(x) for x in ref.p_release],
        "p_pause_ms": [dict(x) for x in ref.p_pause_ms],
        "d_demand": [dict(x) for x in ref.d_demand],
        "d_resume_ms": [dict(x) for x in ref.d_resume_ms],
        "lanes": [[ln.stage, ln.rank, ln.tag, ln.mib, ln.ms, int(ln.oncard), ln.collect_ms]
                  for ln in ref.lanes],
        "p_rows": list(ref.p_rows), "d_rows": list(ref.d_rows),
        "p_resident": list(ref.p_resident), "d_resident": list(ref.d_resident),
        "p_deposit_ms": [dict(x) for x in ref.p_deposit_ms],
        "p_start_ms": list(ref.p_start_ms), "d_start_ms": list(ref.d_start_ms),
    }


def reference_from_dict(d: Mapping[str, object]) -> PDReference:
    lanes = tuple(PDLane(stage=int(x[0]), rank=int(x[1]), tag=str(x[2]), mib=float(x[3]),
                         ms=float(x[4]), oncard=bool(x[5]), collect_ms=float(x[6]))
                  for x in d["lanes"])  # type: ignore[union-attr]
    return PDReference(
        source=str(d["source"]), order=tuple(d["order"]),  # type: ignore[arg-type]
        free={int(k): float(v) for k, v in dict(d["free"]).items()},  # type: ignore[arg-type]
        p_card=tuple(int(x) for x in d["p_card"]),  # type: ignore[union-attr]
        d_floor=tuple(float(x) for x in d["d_floor"]),  # type: ignore[union-attr]
        p_release=tuple(dict(x) for x in d["p_release"]),  # type: ignore[union-attr]
        p_pause_ms=tuple(dict(x) for x in d["p_pause_ms"]),  # type: ignore[union-attr]
        d_demand=tuple(dict(x) for x in d["d_demand"]),  # type: ignore[union-attr]
        d_resume_ms=tuple(dict(x) for x in d["d_resume_ms"]),  # type: ignore[union-attr]
        lanes=lanes,
        p_rows=tuple(int(x) for x in d["p_rows"]),  # type: ignore[union-attr]
        d_rows=tuple(int(x) for x in d["d_rows"]),  # type: ignore[union-attr]
        p_resident=tuple(int(x) for x in d["p_resident"]),  # type: ignore[union-attr]
        d_resident=tuple(int(x) for x in d["d_resident"]),  # type: ignore[union-attr]
        p_deposit_ms=tuple(dict(x) for x in d["p_deposit_ms"]),  # type: ignore[union-attr]
        p_start_ms=tuple(float(x) for x in d["p_start_ms"]),  # type: ignore[union-attr]
        d_start_ms=tuple(float(x) for x in d["d_start_ms"]),  # type: ignore[union-attr]
    )
