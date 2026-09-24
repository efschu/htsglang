# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H33 -- der Posten ausserhalb des D-Budgets, als GEMESSENER Term.

DER BEFUND (H30 R1, 24.09.). Der Dry-Run liess auf der 5090 (D-TP0, Host der
Form A) FR_D[0] 0.10/0.15 und SCRATCH_D[0] 86 durch (W122 PASST, DECKE
0.150), am Metall starb fnFL2x128 mit SCRATCH 86 an einem OOM im Extend
("74.81 MiB is free ... 4.75 GiB allocated in private pools (e.g., CUDA
Graphs)"), und fnFL2x141 hat auf TP0 beim Decode nur 1,05 GiB frei. Der W122-
Ledger prueft das BUDGET (``--rank-gpu-memory-mib``), und dort passt alles:
das KV ist mit ``--max-total-tokens 262144`` gedeckelt, der Rest des Budgets
bleibt liegen. Was die KARTE fuellt, steht in keinem Budget-Posten:

* der Inhalt der PRIVATEN Pools (Graph-Pools, Tag-Pools des Memory-Savers).
  Freie Bloecke darin gibt ``empty_cache`` NIE an den Treiber zurueck
  (``phase_flip_runtime.graph_pool_free_bytes_from_segments``, #852 R3);
  gemessen auf TP0: 5235679744 B = 4993 MiB (x141/x144, ``#1027 ...
  trapped=``), 5097435648 B = 4861 MiB (x128) -- dieselbe Zahl, die torch im
  OOM-Text von x128 "allocated in private pools" nennt (4.75 GiB);
* die Transiente des schwersten Forwards (``allocator peak since pools``).

DIE BILANZ, je Rang und Messpunkt (alles torch-Sicht, MiB)::

    cap      = card_free + reserved          was dieser Prozess ueberhaupt
                                             bekommen kann (Karte minus
                                             fremde Prozesse minus Nicht-Torch)
    headroom = cap - peak - private_free     was nach dem schwersten Forward
                                             bleibt, WENN der Allokator seinen
                                             ganzen allgemeinen Cache abgibt

Der allgemeine Cache (``reserved - allocated - private_free``) wird NICHT
berechnet: ihn gibt der Allokator unter Druck frei (``alloc retries`` in den
``[vram-peak]``-Zeilen), er ist Luft, kein Posten. Die privaten freien Bloecke
dagegen sind belegt, solange ihr Pool lebt.

Gemessen (``[vram-peak]`` + ``#1027``, TP0): x141 decode ``card free 1.05,
reserved 27.66, peak 23.23`` -> 28.71 - 23.23 - 4.876 = 0.60 GiB; x144
dasselbe (0.59 GiB); x128 (Scratch 86, 4 Zeilen x 48 Layer mehr)
28.71 - 23.68 - 4.747 = 0.28 GiB -- und x128 ist gestorben. Die Grenze ist
``corridor_guard.NEAR_OOM_MIB`` (400): "one allocation from death, a stopper
in any phase".

WAS HIER STEHT (rein, kein torch -- der Launcher rechnet es ohne CUDA):

* :class:`GraphPoolSample` -- ein Messpunkt, mit ``cap``/``headroom``;
* :func:`sample_from_stats` -- aus den Rohzahlen des Allokators (Bytes) und
  dem Segment-Snapshot; die Runtime (``vram_family_census``) ruft sie;
* :func:`format_line` / :func:`samples_from_log` -- die ``WEG2-GRAPH-POOL``-
  Zeile und ihr Leser. Fuer Boots VOR diesem Instrument liest der Leser
  dieselben Terme aus ``[vram-peak]`` (card free/reserved/peak) und dem
  ``#1027``-Probe (``trapped=`` = private_free); die Aufloesung ist dort die
  der ``[vram-peak]``-Zeile (0.01 GiB je Term).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgspec

MIB = float(1 << 20)
GIB_IN_MIB = 1024.0

#: Das Praefix der Instrumentzeile. EIN Marker, damit ``grep -c`` eine Zahl
#: liefert.
MARKER = "WEG2-GRAPH-POOL"

#: Woher ein Messpunkt stammt.
SOURCE_INSTRUMENT = "instrument"
SOURCE_LEGACY = "vram-peak+#1027"

#: Aufloesung je Term in MiB: die Instrumentzeile druckt ganze MiB, die
#: ``[vram-peak]``-Zeile GiB mit zwei Stellen (+-0.005 GiB je Term).
PRECISION_MIB = {SOURCE_INSTRUMENT: 1.0, SOURCE_LEGACY: 0.005 * GIB_IN_MIB}


class GraphPoolSample(msgspec.Struct, frozen=True, kw_only=True):
    """Ein Messpunkt eines Rangs, alle Groessen in MiB (torch-Sicht)."""

    rank: int
    phase: str
    card_free_mib: float
    card_total_mib: float
    reserved_mib: float
    allocated_mib: float
    #: ``max_memory_allocated`` seit den Pools (dieselbe Zahl wie
    #: ``[vram-peak] allocator peak since pools``).
    peak_mib: float
    #: Summe ``total_size`` aller Segmente in privaten Pools (Graph-Pools,
    #: Tag-Pools); ``-1`` = der Messpunkt kennt sie nicht (Altlog).
    private_total_mib: float
    #: Freie Bytes in privaten Pools -- das, was ``empty_cache`` nicht
    #: erreicht (``graph_pool_free_bytes_from_segments``).
    private_free_mib: float
    source: str = SOURCE_INSTRUMENT
    #: Die groessten privaten Pools, ``((pool_id, total_mib, allocated_mib),
    #: ...)``; nur fuer den Druck, keine Rechengroesse.
    pools: Tuple[Tuple[str, float, float], ...] = ()

    @property
    def cap_mib(self) -> float:
        return self.card_free_mib + self.reserved_mib

    @property
    def allocator_cache_mib(self) -> float:
        return self.reserved_mib - self.allocated_mib

    @property
    def general_cache_mib(self) -> float:
        return self.allocator_cache_mib - self.private_free_mib

    @property
    def headroom_mib(self) -> float:
        return self.cap_mib - self.peak_mib - self.private_free_mib

    @property
    def precision_mib(self) -> float:
        """Messfehler der Bilanz ``headroom`` (vier Terme, je einer halben
        Druckstelle; ``private_free`` ist in beiden Quellen bytegenau)."""
        return 3.0 * PRECISION_MIB.get(self.source, 1.0)


# ---------------------------------------------------------------------------
# die Runtime-Seite: Rohzahlen -> Messpunkt
# ---------------------------------------------------------------------------


def segment_pool_id(seg: Mapping) -> Tuple[int, ...]:
    """``phase_flip_runtime._segment_pool_id``, Zeichen fuer Zeichen (per
    Test gebunden): ``(0, 0)`` ist der allgemeine Pool."""
    raw = seg.get("segment_pool_id", seg.get("owner_private_pool_id", (0, 0)))
    try:
        return tuple(int(x) for x in raw)
    except (TypeError, ValueError):
        return (0, 0)


def private_pools_from_segments(
    segments: Iterable,
) -> Dict[Tuple[int, ...], Tuple[int, int]]:
    """``{pool_id: (total_bytes, allocated_bytes)}`` der PRIVATEN Pools eines
    ``torch.cuda.memory_snapshot()``; der allgemeine Pool fehlt darin."""
    out: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        pid = segment_pool_id(seg)
        if pid == (0, 0):
            continue
        tot, used = out.get(pid, (0, 0))
        out[pid] = (
            tot + int(seg.get("total_size", 0)),
            used + int(seg.get("allocated_size", 0)),
        )
    return out


def sample_from_stats(
    *,
    rank: int,
    phase: str,
    free_bytes: int,
    total_bytes: int,
    reserved_bytes: int,
    allocated_bytes: int,
    peak_bytes: int,
    segments: Optional[Iterable],
    top_pools: int = 4,
) -> GraphPoolSample:
    """Der Messpunkt aus ``mem_get_info``, ``memory_reserved/allocated``,
    ``max_memory_allocated`` und ``memory_snapshot()``. Ohne Snapshot
    (``segments=None``) sind die privaten Terme ``-1`` und der Punkt ist KEIN
    Ledger-Eingang (:func:`usable`)."""
    if segments is None:
        priv_total = priv_free = -1.0
        pools: Tuple[Tuple[str, float, float], ...] = ()
    else:
        per_pool = private_pools_from_segments(segments)
        priv_total = sum(t for t, _ in per_pool.values()) / MIB
        priv_free = sum(max(0, t - u) for t, u in per_pool.values()) / MIB
        ranked = sorted(per_pool.items(), key=lambda kv: kv[1][0], reverse=True)
        pools = tuple(
            (
                ".".join(str(x) for x in pid),
                round(t / MIB, 1),
                round(u / MIB, 1),
            )
            for pid, (t, u) in ranked[: max(0, int(top_pools))]
        )
    return GraphPoolSample(
        rank=int(rank),
        phase=str(phase),
        card_free_mib=free_bytes / MIB,
        card_total_mib=total_bytes / MIB,
        reserved_mib=reserved_bytes / MIB,
        allocated_mib=allocated_bytes / MIB,
        peak_mib=peak_bytes / MIB,
        private_total_mib=priv_total,
        private_free_mib=priv_free,
        source=SOURCE_INSTRUMENT,
        pools=pools,
    )


def usable(sample: GraphPoolSample) -> bool:
    """Nur ein Punkt mit gemessenem privaten Term ist ein Ledger-Eingang --
    ohne ihn waere ``headroom`` um den ganzen Posten zu hoch."""
    return sample.private_free_mib >= 0.0


def format_line(sample: GraphPoolSample) -> str:
    """Die Instrumentzeile. Feldnamen sind Vertrag (:func:`samples_from_log`)."""
    pools = ",".join("%s:%.0f/%.0f" % p for p in sample.pools) or "-"
    return (
        "%s rank=%d phase=%s captured_mib=%.0f private_free_mib=%.0f "
        "reserved_after_mib=%.0f allocated_mib=%.0f peak_mib=%.0f "
        "allocator_cache_mib=%.0f general_cache_mib=%.0f card_free_mib=%.0f "
        "card_total_mib=%.0f cap_mib=%.0f headroom_mib=%.0f pools=%s -- "
        "captured = Segmente privater Pools (Graph/Tag), private_free = davon "
        "frei und fuer empty_cache unerreichbar; headroom = cap - peak - "
        "private_free ist, was der schwerste Forward auf dieser Karte uebrig "
        "laesst (Ledger-Eingang des D-KARTE-Riegels, H33)"
        % (
            MARKER,
            sample.rank,
            sample.phase,
            sample.private_total_mib,
            sample.private_free_mib,
            sample.reserved_mib,
            sample.allocated_mib,
            sample.peak_mib,
            sample.allocator_cache_mib,
            sample.general_cache_mib,
            sample.card_free_mib,
            sample.card_total_mib,
            sample.cap_mib,
            sample.headroom_mib,
            pools,
        )
    )


# ---------------------------------------------------------------------------
# der Leser
# ---------------------------------------------------------------------------

_TP = r"\[(?:[0-9-]+ [0-9:]+ )?TP(\d+)\]"
_RX_LINE = re.compile(re.escape(MARKER) + r" rank=(\d+) phase=(\S+) (.*)$")
_RX_KV = re.compile(r"(\w+)=(-?[0-9.]+)")
_RX_PEAK = re.compile(
    _TP + r" \[vram-peak\] (\S+) \((-?\d+) rows\): allocator peak since pools "
    r"([0-9.]+) GiB, allocated now ([0-9.]+), reserved ([0-9.]+), card free "
    r"([0-9.]+) of ([0-9.]+) GiB"
)
_RX_TRAPPED = re.compile(_TP + r" #1027 graph-pool-free probe: .*?trapped=(\d+)")

_NEED = (
    "captured_mib",
    "private_free_mib",
    "reserved_after_mib",
    "allocated_mib",
    "peak_mib",
    "card_free_mib",
    "card_total_mib",
)


def samples_from_log(text: str) -> Dict[int, List[GraphPoolSample]]:
    """Alle Messpunkte eines D-Logs je Rang, in Log-Reihenfolge.

    Hat ein Rang ``WEG2-GRAPH-POOL``-Zeilen, gelten NUR sie (das Instrument
    misst den privaten Term an jedem Punkt). Sonst die Altform: jede
    ``[vram-peak]``-Zeile NACH dem ersten ``#1027``-Probe des Rangs, mit dessen
    ``trapped=`` als privatem Term (der Probe druckt nur seinen ersten Wert;
    der Term aendert sich nur bei Capture und Flip). ``[vram-peak]``-Zeilen
    VOR dem Probe liegen im Capture und werden nicht gewertet.
    """
    inst: Dict[int, List[GraphPoolSample]] = {}
    legacy: Dict[int, List[GraphPoolSample]] = {}
    trapped: Dict[int, float] = {}
    for line in text.splitlines():
        m = _RX_LINE.search(line)
        if m:
            kv = {k: float(v) for k, v in _RX_KV.findall(m.group(3))}
            if all(k in kv for k in _NEED):
                r = int(m.group(1))
                inst.setdefault(r, []).append(
                    GraphPoolSample(
                        rank=r,
                        phase=m.group(2),
                        card_free_mib=kv["card_free_mib"],
                        card_total_mib=kv["card_total_mib"],
                        reserved_mib=kv["reserved_after_mib"],
                        allocated_mib=kv["allocated_mib"],
                        peak_mib=kv["peak_mib"],
                        private_total_mib=kv["captured_mib"],
                        private_free_mib=kv["private_free_mib"],
                        source=SOURCE_INSTRUMENT,
                    )
                )
            continue
        m = _RX_TRAPPED.search(line)
        if m:
            trapped[int(m.group(1))] = int(m.group(2)) / MIB
            continue
        m = _RX_PEAK.search(line)
        if m:
            r = int(m.group(1))
            if r not in trapped:
                continue
            legacy.setdefault(r, []).append(
                GraphPoolSample(
                    rank=r,
                    phase=m.group(2),
                    card_free_mib=float(m.group(7)) * GIB_IN_MIB,
                    card_total_mib=float(m.group(8)) * GIB_IN_MIB,
                    reserved_mib=float(m.group(6)) * GIB_IN_MIB,
                    allocated_mib=float(m.group(5)) * GIB_IN_MIB,
                    peak_mib=float(m.group(4)) * GIB_IN_MIB,
                    private_total_mib=-1.0,
                    private_free_mib=trapped[r],
                    source=SOURCE_LEGACY,
                )
            )
    out: Dict[int, List[GraphPoolSample]] = {}
    for r in set(inst) | set(legacy):
        out[r] = [s for s in inst[r] if usable(s)] if r in inst else legacy[r]
    return out


def binding_sample(samples: Sequence[GraphPoolSample]) -> Optional[GraphPoolSample]:
    """Der Punkt mit dem KLEINSTEN ``headroom`` -- er bindet."""
    good = [s for s in samples if usable(s)]
    if not good:
        return None
    return min(good, key=lambda s: s.headroom_mib)


def decode_sample(samples: Sequence[GraphPoolSample]) -> Optional[GraphPoolSample]:
    """Der erste Decode-Punkt (``[vram-peak] decode`` bzw. ``phase=decode``):
    die Karte im eingeschwungenen Decode."""
    for s in samples:
        if s.phase == "decode" and usable(s):
            return s
    return None
