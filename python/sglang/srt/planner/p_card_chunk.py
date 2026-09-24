# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H41 -- die P-Karte je Stufe als Funktion des Chunks (Task #114).

DER BEFUND. fnFL2x121/x122 (24.09.) bootete die 5090-Stufe (PP0) mit Residenz
0.45 bzw. 0.40 bei Chunk 16384, der Dry-Run sagte PASST, und beide starben im
ERSTEN Chunk-Forward mit CUDA-OOM ("Tried to allocate 320.00 MiB ... 28.81 MiB
is free", x122 384/148.81). Der Loeser kannte den Chunk an keiner Stelle, an
der die KARTE entschieden wird:

* ``pp_cut.solve_expert_fraction_per_stage`` (die #140-Decke) rechnet
  ``layers x (dense + row x Puffer) + reserve <= budget`` mit ``reserve`` =
  KV allein (#156); Aktivierung und Chunk fehlen ("Die Obergrenze laesst
  NICHTS fuer KV, Draft und Aktivierungen");
* ``PhasePoolModel.activation_reserve_mib`` (seit #114 der Chunk-Posten) ist
  EIN Skalar fuer alle Stufen und entscheidet nur die KV-Tokenzahl -- bei
  ``--max-total-tokens 262144`` nimmt der KV-Pool 262144 x Zelle, der Rest des
  Budgets bleibt liegen, und nichts verweigert eine Residenz;
* das Budget selbst ist nicht die Karte: ``measured free`` liegt 4-7 GiB ueber
  ``rest`` (``unaccounted=+4.150`` x121, ``+7.38`` x146), und was die Karte
  fuellt, steht in keinem Budget-Posten (private Graph-Pools, der allgemeine
  Cache, die mit dem Kontext wachsende Belegung).

DIE BILANZ, je P-Stufe s (torch-Sicht, MiB; dieselbe wie H33 fuer D)::

    Kopfraum_s = cap - peak - privat_frei             (WEG2-GRAPH-POOL)
    Kopfraum_s(f, c) = K0_s - Puffer_s(f) x L_s x Zeile - KV_s - T_s(c) - Draft_s
                       >= near-OOM (corridor_guard.NEAR_OOM_MIB = 400)

mit ``K0_s`` = der gemessene Kopfraum des bindenden Punkts eines Referenz-
Boots, normiert auf LEEREN Puffer, KEIN KV, KEINE Chunk-Transiente und keinen
Draft auf P (das Minimum ueber die Referenz-Boots). ``T_s(c)`` ist die
Transiente des Chunk-Forwards (``[vram-peak]``: allocator peak minus
allocated), GEMESSEN an Stuetzpunkten und dazwischen linear interpoliert;
ausserhalb der Stuetzpunkte wird verweigert (W131), nie extrapoliert. Keine
Reserve, kein Hand-Pin: jeder Term ist eine Messung oder die Geometrie des
Checkpoints.

Rein (kein torch): der Launcher rechnet es ohne CUDA.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import msgspec

MIB = float(1 << 20)
GIB_IN_MIB = 1024.0

#: Die Dry-Run-Zeilen. EIN Praefix je Zeilenart, damit ``grep -c`` zaehlt.
ACTIVATION_MARKER = "PP-CUT ACTIVATION"
CARD_MARKER = "PP-CUT P-KARTE"

#: Chunk ausserhalb der gemessenen Stuetzpunkte: keine Zahl, sondern Verweigerung.
UNMEASURED_CODE = "W131 Weg2PChunkTransientUnmeasured"
#: Die Karte einer P-Stufe stirbt im Chunk-Forward (x121/x122).
CARD_REFUSAL_CODE = "W132 Weg2PCardChunkOom"

#: ``corridor_guard.NEAR_OOM_MIB`` (per Test gebunden): "one allocation from
#: death, a stopper in any phase". Hier gespiegelt, weil corridor_guard torch
#: zieht und der Loeser ohne CUDA laufen muss.
NEAR_OOM_MIB = 400.0

#: Aufloesung der ``[vram-peak]``-Zeile (GiB mit zwei Stellen).
VRAM_PEAK_PRECISION_MIB = 0.005 * GIB_IN_MIB

#: Die Stufe druckt eine neue Hochwassermarke erst, wenn sie die letzte
#: GEDRUCKTE um diesen Schritt uebertrifft (``vram_family_census.
#: PEAK_HIGHWATER_STEP_GIB``): der gemessene bindende Punkt kann bis zu diesem
#: Betrag UNTER dem wahren liegen. Gedruckt, nicht abgezogen (keine Reserve).
HIGHWATER_STEP_MIB = 0.25 * GIB_IN_MIB


class PChunkUnmeasured(ValueError):
    """W131: ein Chunk ausserhalb der gemessenen Stuetzpunkte."""


# ---------------------------------------------------------------------------
# 1. die Transiente als Funktion des Chunks
# ---------------------------------------------------------------------------


class TransientPoint(msgspec.Struct, frozen=True, kw_only=True):
    """Ein gemessener Stuetzpunkt: die Transiente je Stufe bei ``chunk``."""

    chunk: int
    #: ``max(peak - allocated)`` ueber die ``[vram-peak]``-Zeilen des vollen
    #: Chunks, MiB je Stufe; das Maximum ueber die Boots dieses Punkts.
    mib: Tuple[float, ...]
    source: str


class TransientSupport(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    points: Tuple[TransientPoint, ...]

    @property
    def chunks(self) -> Tuple[int, ...]:
        return tuple(p.chunk for p in self.points)

    @property
    def n_stages(self) -> int:
        return len(self.points[0].mib) if self.points else 0


_STAGE = r"\[(?:[0-9-]+ [0-9:]+ )?PP(\d+)\]"
_RX_PEAK = re.compile(
    _STAGE + r" \[vram-peak\] (\S+) \((-?\d+) rows\): allocator peak since pools "
    r"([0-9.]+) GiB, allocated now ([0-9.]+), reserved ([0-9.]+), card free "
    r"([0-9.]+) of ([0-9.]+) GiB"
)
_RX_CHUNK = re.compile(
    _STAGE + r" max_total_num_tokens=(\d+), chunked_prefill_size=(-?\d+)"
)
_RX_BUFFER = re.compile(
    _STAGE + r" MoE expert-offload active on layer \d+: (\d+)/(\d+) experts "
    r"resident \+ (\d+) scratch \(buffer=(\d+), fraction=([0-9.]+)\)"
)
_RX_CELL = re.compile(_STAGE + r" KV pool sizing: available_bytes=\d+ .*?cell_size=(\d+),")
_DRAFT_ON_P = "draft pp group built"


def observe_p_log(text: str) -> Dict[str, Dict[int, object]]:
    """Was ein P-Log je Stufe ueber Chunk, Puffer, KV und Transiente sagt."""
    chunk: Dict[int, int] = {}
    tokens: Dict[int, int] = {}
    buffer: Dict[int, int] = {}
    experts: Dict[int, int] = {}
    fraction: Dict[int, float] = {}
    cell: Dict[int, int] = {}
    transient: Dict[int, float] = {}
    for line in text.splitlines():
        m = _RX_CHUNK.search(line)
        if m:
            s = int(m.group(1))
            tokens[s] = int(m.group(2))
            chunk[s] = int(m.group(3))
            continue
        m = _RX_BUFFER.search(line)
        if m:
            s = int(m.group(1))
            if s not in buffer:
                buffer[s] = int(m.group(5))
                experts[s] = int(m.group(3))
                fraction[s] = float(m.group(6))
            continue
        m = _RX_CELL.search(line)
        if m:
            cell.setdefault(int(m.group(1)), int(m.group(2)))
            continue
        m = _RX_PEAK.search(line)
        if m:
            s = int(m.group(1))
            rows = int(m.group(3))
            # Nur der VOLLE Chunk: ein Rest-Chunk oder ein Decode-Punkt ist
            # nicht die Transiente dieser Breite.
            if s in chunk and rows != chunk[s]:
                continue
            t = (float(m.group(4)) - float(m.group(5))) * GIB_IN_MIB
            transient[s] = max(transient.get(s, 0.0), t)
    return {
        "chunk": chunk,
        "tokens": tokens,
        "buffer": buffer,
        "experts": experts,
        "fraction": fraction,
        "cell": cell,
        "transient": transient,
        "draft_on_p": {0: _DRAFT_ON_P in text},
    }


def transient_support_from_logs(
    boots: Sequence[Tuple[str, str]], *, n_stages: int, model: str
) -> TransientSupport:
    """Die Stuetzpunkte aus P-Logs MESSEN (``boots`` = (Name, Text)).

    Je Boot: der Chunk (alle Stufen gleich, sonst kein Punkt) und je Stufe das
    Maximum von ``peak - allocated`` ueber die ``[vram-peak]``-Zeilen des
    vollen Chunks. Je Chunk das Maximum ueber die Boots (konservativ, wie H8
    das Maximum der Posten). Ein Boot, dem eine Stufe fehlt, liefert keinen
    Punkt -- eine fehlende Stufe ist nicht null.
    """
    acc: Dict[int, Tuple[List[float], List[str]]] = {}
    for name, text in boots:
        obs = observe_p_log(text)
        chunks = {obs["chunk"].get(s) for s in range(n_stages)}
        if len(chunks) != 1 or None in chunks:
            continue
        c = int(next(iter(chunks)))
        tr = obs["transient"]
        if any(s not in tr for s in range(n_stages)):
            continue
        vals, names = acc.setdefault(c, ([0.0] * n_stages, []))
        for s in range(n_stages):
            vals[s] = max(vals[s], float(tr[s]))
        names.append(name)
    if not acc:
        raise ValueError(
            "keine P-Transiente in %s: kein Boot mit [vram-peak] auf allen %d Stufen"
            % ([b[0] for b in boots], n_stages)
        )
    return TransientSupport(
        model=model,
        points=tuple(
            TransientPoint(
                chunk=c,
                mib=tuple(round(v, 1) for v in acc[c][0]),
                source="+".join(acc[c][1]),
            )
            for c in sorted(acc)
        ),
    )


#: Die gemessenen Stuetzpunkte der Next-Flash-P-Gruppe (Schnitt 29,11,8),
#: hergeleitet von :func:`transient_support_from_logs` aus den P-Logs unter
#: /spinning/evidence-665-f1 (dieselben Zeilen liegen als Fixture unter
#: test/registered/unit/weg2/fixtures/p_card_h41/; der Test
#: ``test_the_shipped_support_is_the_logs_own_measurement`` bindet sie).
#: GiB aus den Zeilen, je Punkt das Maximum ueber die Boots:
#:
#:   chunk    PP0 (5090)  PP1 (3080)  PP2 (3080)   Boots
#:     512      0.13        0.13        0.13       x130, x134
#:    4096      0.92        0.82        0.85       x12, x14, x101, x107
#:    8192      1.82        1.63        1.60       x113, x116
#:   16384      3.61        3.21        3.19       x118, x141, x145, x146
#:
#: KORREKTUR an #114 (add357d7c2): PP2 bei 16384 ist 3.19 GiB, nicht 2.43.
#: Auf PP2 druckt ein Chunk mit Draft auf P ZWEI extend-Zeilen (Ziel-Forward
#: 13.09 - 9.91 = 3.18 GiB, danach der Draft-Forward nach einem Peak-Reset
#: 12.34 - 9.91 = 2.43 GiB, fnFL2x118 02:36:24/25); #114 las die zweite. In
#: der H25-Form (kein Draft auf P) misst PP2 weiter 3.18 (x145/x146).
P_TRANSIENT_SUPPORT_FNFL2 = TransientSupport(
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    points=(
        TransientPoint(chunk=512, mib=(133.1, 133.1, 133.1),
                       source="fnFL2x130+fnFL2x134"),
        TransientPoint(chunk=4096, mib=(942.1, 839.7, 870.4),
                       source="fnFL2x12+fnFL2x14+fnFL2x101+fnFL2x107"),
        TransientPoint(chunk=8192, mib=(1863.7, 1669.1, 1638.4),
                       source="fnFL2x113+fnFL2x116"),
        TransientPoint(chunk=16384, mib=(3696.6, 3287.0, 3266.6),
                       source="fnFL2x118+fnFL2x141+fnFL2x145+fnFL2x146"),
    ),
)


def unmeasured_text(support: TransientSupport, chunk: int) -> str:
    lo, hi = support.chunks[0], support.chunks[-1]
    return (
        "%s: group P bootet mit --chunked-prefill-size %d, die Chunk-Transiente "
        "ist nur zwischen %d und %d Token gemessen (Stuetzpunkte %s, %s). "
        "Ausserhalb wird nicht extrapoliert -- die Aktivierungsspitze waechst "
        "mit dem Chunk (Attention, QSA-Sidecar, MoE-Dispatch, Marlin-Workspaces) "
        "und genau diese Luecke hat x121/x122 getoetet. Einen Boot dieser "
        "Chunkbreite mit [vram-peak] messen und die Stuetzpunkte auffrischen "
        "(p_card_chunk.transient_support_from_logs)."
        % (UNMEASURED_CODE, int(chunk), lo, hi, list(support.chunks), support.model)
    )


def transient_mib(support: TransientSupport, stage: int, chunk: int) -> Tuple[float, str]:
    """``(T_stage(chunk) MiB, Herkunft)`` -- linear zwischen den Stuetzpunkten.

    Ausserhalb ``[min, max]`` der gemessenen Chunks: :class:`PChunkUnmeasured`.
    """
    c = int(chunk)
    pts = support.points
    if not pts or c < pts[0].chunk or c > pts[-1].chunk:
        raise PChunkUnmeasured(unmeasured_text(support, c))
    for p in pts:
        if p.chunk == c:
            return float(p.mib[stage]), "gemessen %s" % p.source
    for a, b in zip(pts, pts[1:]):
        if a.chunk < c < b.chunk:
            w = (c - a.chunk) / float(b.chunk - a.chunk)
            v = float(a.mib[stage]) + w * (float(b.mib[stage]) - float(a.mib[stage]))
            return v, "interpoliert %d[%s]..%d[%s]" % (a.chunk, a.source, b.chunk, b.source)
    raise PChunkUnmeasured(unmeasured_text(support, c))  # pragma: no cover


def transient_vector_mib(support: TransientSupport, chunk: int) -> Tuple[float, ...]:
    return tuple(
        round(transient_mib(support, s, chunk)[0], 1) for s in range(support.n_stages)
    )


def activation_lines(support: TransientSupport, chunk: int) -> Tuple[str, ...]:
    """Die Dry-Run-Zeilen ``PP-CUT ACTIVATION chunk=... stageN transient_mib=...``."""
    out = []
    for s in range(support.n_stages):
        v, src = transient_mib(support, s, chunk)
        out.append(
            "%s chunk=%d stage%d transient_mib=%.0f source=%s (Stuetzpunkte %s, "
            "[vram-peak] peak - allocated des vollen Chunks, Maximum ueber die "
            "Boots; ausserhalb verweigert %s)"
            % (ACTIVATION_MARKER, int(chunk), s, v, src.replace(" ", "="),
               list(support.chunks), UNMEASURED_CODE.split()[0])
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# 2. die P-Karte: gemessener Kopfraum, um Puffer, KV, Chunk, Draft und LMEM
#    verschoben (H41, H41c)
# ---------------------------------------------------------------------------
#
# H41c (x149 aufgeklaert von H47, da80a26a59). Die Bilanz je Stufe am Chunk-
# Index i eines Prompts::
#
#   Kopfraum_s(i) = K0_s - Zeilen x L x Zeile - Ueber x (Preis - Bild)
#                   - KV_s - T_s(chunk) - Draft_s - g_s x (Token vor Chunk i)
#                   - LMEM_s
#
# * K0_s: der Kopfraum am Chunk 0 der Referenz (cap - peak - privat_frei),
#   normiert auf leeren Puffer, kein KV, keine Transiente, keinen Draft;
# * Ueber x (Preis - Bild): eine Zeile ueber der Referenz kostet den am Chunk 0
#   eines Boots mit MEHR Zeilen gemessenen Preis (x149 PP0: (4594 - 1300) / 46
#   = 71.6 MiB, Bild 70.1 -- eine OBERE Schranke, die +73 MiB privat_frei von
#   x149 stecken darin), nie weniger als das Bild;
# * g_s: das Wachstum der Spitze, aus den PEAK-Werten der Referenz, und es
#   SAETTIGT (H41d, gemessen fnFL2x160: 259441-Token-Prompt, 16 Chunks; PP0
#   24490 -> 24839 -> 25160 -> 25480 und danach bis Chunk 15 keine neue
#   Hochwassermarke, PP1 ebenso, PP2 nach Chunk 1): der Term ist
#   g_s x min(Chunk-Index, letzter wachsender Chunk) -- PP0 990, PP1 974,
#   PP2 329 MiB, unabhaengig von der Promptlaenge. Nur ein Prompt, der LAENGER
#   ist als der laengste der Referenz, ist eine HOCHRECHNUNG (so benannt);
# * LMEM_s: der lokale Treiberspeicher des Arena-Write-Kernels im Run-Modus
#   (H47): stack = run/32 - 64 B je residentem Thread, einmal je Karte,
#   sichtbar als cap-Abfall zwischen Chunk 0 und Chunk 1 (x149 PP0 -1457,
#   x146 PP1 -150, PP2 -48). Mit dem H47-Fix (``arena_write.
#   RUN_ELEMENT_BYTES``) ist er 0; ohne Fix gilt der gemessene Hochstand.
#
# Der TOD x149 ist KEIN Zeilenpreis: im Chunk 0 lag x149 auf dem Zeilenbild
# (1300 gemessen), gestorben ist er am LMEM (cap 28951 -> 27494 nach
# 'WEG2-ARENA-WRITE n=1 ... mode=run'). :func:`death_from_log` erkennt diesen
# Fall und bucht ihn nicht.


class PCardReference(msgspec.Struct, frozen=True, kw_only=True):
    """Der Kopfraum am Chunk 0 eines Referenz-Boots je P-Stufe, das Wachstum
    je Chunk und der LMEM-Posten dieser Referenz."""

    source: str
    model: str
    stage_layers: Tuple[int, ...]
    chunk: int
    #: ``Kopfraum(Chunk 0) + Puffer x L x Zeile + KV + T(chunk)``, MiB;
    #: Minimum ueber die Boots.
    headroom0_mib: Tuple[float, ...]
    #: Die Terme des Chunk-0-Punkts, fuer den Druck.
    headroom_mib: Tuple[float, ...]
    buffer_rows: Tuple[int, ...]
    kv_mib: Tuple[float, ...]
    cap_mib: Tuple[float, ...]
    peak_mib: Tuple[float, ...]
    private_free_mib: Tuple[float, ...]
    draft_on_p: bool
    #: Wachstum der Spitze je Token (MiB), aus den Peak-Werten ueber die
    #: gemessenen Chunk-Indizes; ``growth_measured_chunks`` = bis wohin.
    growth_mib_per_token: Tuple[float, ...]
    growth_measured_chunks: Tuple[int, ...]
    #: Der laengste Prompt der Referenz (Token), Tiefe des Riegels per Default.
    longest_prompt_tokens: int
    #: cap-Abfall nach dem ersten Run-Write der Referenz, MiB je Stufe (H47).
    lmem_mib: Tuple[float, ...]
    #: Preis einer Zeile ueber der Referenz, MiB (leer = das Zeilenbild).
    row_card_mib: Tuple[float, ...] = ()
    row_card_source: str = ""


_RX_EXTENT = re.compile(_STAGE + r" #969 EXTENT n=\d+ fwd=\d+ reqs=\[\('[^']*', (\d+), (\d+)")
_RX_POOL = re.compile(
    _STAGE + r" WEG2-GRAPH-POOL rank=\d+ phase=(\S+) .*?private_free_mib=(-?\d+) "
    r".*?peak_mib=(\d+) .*?cap_mib=(\d+) "
)
_RX_EXC = re.compile(_STAGE + r" Scheduler hit an exception")
_RX_RUNWRITE = re.compile(_STAGE + r" WEG2-ARENA-WRITE n=\d+ .*mode=run ")
_RX_SLEEP_LMEM = re.compile(_STAGE + r" WEG2-SLEEP-LMEM lmem \d+->\d+ MiB released \(stack (\d+)->")
_RX_OOM = re.compile(
    r"OutOfMemoryError: CUDA out of memory\. Tried to allocate ([0-9.]+) (MiB|GiB)\. "
    r"GPU \d+ has a total capacity of [0-9.]+ GiB of which ([0-9.]+) (MiB|GiB) is free\."
    r".*?Of the allocated memory ([0-9.]+) GiB is allocated by PyTorch.*?"
    r"([0-9.]+) (MiB|GiB) is reserved by PyTorch but unallocated"
)

#: Ab diesem cap-Abfall zwischen zwei Punkten derselben Stufe ist es der
#: Run-Write-LMEM (x149 1457 MiB), keine Zeile und kein Chunk.
LMEM_CAP_DROP_MIB = 1024.0
#: Ab diesem Stack je Thread ist ein WEG2-SLEEP-LMEM der Run-Write-Hochstand
#: (Basis 1248-1504 B, Run-Write PP0 7104 B, x151).
LMEM_STACK_BYTES = 2000


def _mib(v: str, unit: str) -> float:
    return float(v) * (GIB_IN_MIB if unit == "GiB" else 1.0)


class PoolSample(msgspec.Struct, frozen=True, kw_only=True):
    stage: int
    chunk_index: int
    phase: str
    private_free_mib: float
    peak_mib: float
    cap_mib: float

    @property
    def headroom_mib(self) -> float:
        return self.cap_mib - self.peak_mib - self.private_free_mib


def pool_samples(text: str) -> Dict[int, List[PoolSample]]:
    """Die Forward-Punkte (WEG2-GRAPH-POOL, ohne ``post-capture``) je Stufe mit
    ihrem Chunk-Index ``start // chunk`` der letzten ``#969 EXTENT``-Zeile."""
    chunk: Dict[int, int] = {}
    start: Dict[int, int] = {}
    out: Dict[int, List[PoolSample]] = {}
    for line in text.splitlines():
        m = _RX_CHUNK.search(line)
        if m:
            chunk[int(m.group(1))] = int(m.group(3))
            continue
        m = _RX_EXTENT.search(line)
        if m:
            start[int(m.group(1))] = int(m.group(2))
            continue
        m = _RX_POOL.search(line)
        if m and m.group(2) != "post-capture":
            s = int(m.group(1))
            if s not in chunk or s not in start or int(m.group(3)) < 0:
                continue
            out.setdefault(s, []).append(
                PoolSample(
                    stage=s,
                    chunk_index=start[s] // max(1, chunk[s]),
                    phase=m.group(2),
                    private_free_mib=float(m.group(3)),
                    peak_mib=float(m.group(4)),
                    cap_mib=float(m.group(5)),
                )
            )
    return out


def headroom_by_chunk_index(text: str) -> Dict[int, Dict[int, float]]:
    """``{stage: {chunk_index: min headroom}}`` (siehe :func:`pool_samples`)."""
    out: Dict[int, Dict[int, float]] = {}
    for s, ss in pool_samples(text).items():
        cur = out.setdefault(s, {})
        for p in ss:
            cur[p.chunk_index] = min(cur.get(p.chunk_index, p.headroom_mib), p.headroom_mib)
    return out


def longest_prompt_tokens(text: str) -> int:
    """Das groesste Extent-Ende eines Logs (der laengste Prompt, Token)."""
    best = 0
    for line in text.splitlines():
        m = _RX_EXTENT.search(line)
        if m:
            best = max(best, int(m.group(3)))
    return best


class PCardDeath(msgspec.Struct, frozen=True, kw_only=True):
    """Ein P-Rang, der an der Karte starb, torch-Sicht, MiB.

    ``allocated by PyTorch`` im OOM-Text enthaelt die freien Bloecke der
    privaten Pools (H47): echt belegt = belegt - privat_frei. Damit::

        Kopfraum_Tod = cap_Tod - (belegt - privat_frei + Anforderung) - privat_frei
                     = cap_Tod - belegt - Anforderung

    ``cause`` = ``"lmem"``, wenn zwischen dem letzten Punkt der Stufe und dem
    Tod ein ``WEG2-ARENA-WRITE ... mode=run`` lag und cap um mehr als
    :data:`LMEM_CAP_DROP_MIB` fiel (oder ein WEG2-SLEEP-LMEM-Stack ueber
    :data:`LMEM_STACK_BYTES` steht) -- dann ist der Tod der LMEM-Posten und
    kein Zeilenpreis; sonst ``"card"``."""

    source: str
    stage: int
    chunk: int
    chunk_index: int
    buffer_rows: int
    cause: str
    headroom_mib: float
    cap_mib: float
    cap_before_mib: float
    used_mib: float
    request_mib: float
    private_free_mib: float
    #: Kopfraum am Tod, wenn cap nicht gefallen waere (LMEM-Fall).
    headroom_without_lmem_mib: float


def death_from_log(name: str, text: str) -> Optional[PCardDeath]:
    """Der erste OOM-Tod eines P-Logs, oder ``None``. Die Stufe ist die der
    letzten ``Scheduler hit an exception``-Zeile vor dem OOM-Text."""
    chunk: Dict[int, int] = {}
    start: Dict[int, int] = {}
    pfree: Dict[int, float] = {}
    cap_last: Dict[int, float] = {}
    runwrite_after: Dict[int, bool] = {}
    big_stack: Dict[int, bool] = {}
    exc_stage: Optional[int] = None
    for line in text.splitlines():
        m = _RX_CHUNK.search(line)
        if m:
            chunk[int(m.group(1))] = int(m.group(3))
            continue
        m = _RX_EXTENT.search(line)
        if m:
            start[int(m.group(1))] = int(m.group(2))
            continue
        m = _RX_POOL.search(line)
        if m:
            s = int(m.group(1))
            pfree[s] = float(m.group(3))
            cap_last[s] = float(m.group(5))
            runwrite_after[s] = False
            continue
        m = _RX_RUNWRITE.search(line)
        if m:
            runwrite_after[int(m.group(1))] = True
            continue
        m = _RX_SLEEP_LMEM.search(line)
        if m and int(m.group(2)) > LMEM_STACK_BYTES:
            big_stack[int(m.group(1))] = True
            continue
        m = _RX_EXC.search(line)
        if m:
            exc_stage = int(m.group(1))
            continue
        m = _RX_OOM.search(line)
        if m and exc_stage is not None:
            s = exc_stage
            if s not in chunk or s not in start or s not in pfree:
                return None
            obs = observe_p_log(text)
            if s not in obs["buffer"]:
                return None
            req = _mib(m.group(1), m.group(2))
            free = _mib(m.group(3), m.group(4))
            alloc = float(m.group(5)) * GIB_IN_MIB
            unalloc = _mib(m.group(6), m.group(7))
            cap = free + alloc + unalloc
            used = alloc - pfree[s]
            head = cap - used - req - pfree[s]
            drop = cap_last[s] - cap
            lmem = (runwrite_after.get(s, False) and drop > LMEM_CAP_DROP_MIB) or big_stack.get(s, False)
            return PCardDeath(
                source=name,
                stage=s,
                chunk=int(chunk[s]),
                chunk_index=int(start[s]) // max(1, int(chunk[s])),
                buffer_rows=int(obs["buffer"][s]),
                cause="lmem" if lmem else "card",
                headroom_mib=round(head, 1),
                cap_mib=round(cap, 1),
                cap_before_mib=cap_last[s],
                used_mib=round(used, 1),
                request_mib=req,
                private_free_mib=pfree[s],
                headroom_without_lmem_mib=round(head + max(0.0, drop), 1),
            )
    return None


def row_card_cost_from_boots(
    reference_boots: Sequence[Tuple[str, str]],
    over_boots: Sequence[Tuple[str, str]],
    *,
    reference: "PCardReference",
    row_mib: float,
) -> Tuple[Tuple[float, ...], str]:
    """Preis einer Zeile ueber der Referenz, je Stufe, aus dem CHUNK-0-Punkt.

    ``k = (Kopfraum_ref(Chunk 0) - Kopfraum_over(Chunk 0) - dKV) / dZeilen``;
    am Chunk 0 liegt noch kein Run-Write und kein Chunk-Wachstum dazwischen.
    Eine Differenz im privat_frei steckt darin, also ist ``k`` eine OBERE
    Schranke. Nie unter dem Zeilenbild ``L x Zeile``.
    """
    n = len(reference.stage_layers)
    img = [int(reference.stage_layers[s]) * float(row_mib) for s in range(n)]
    ref0: Dict[int, float] = {}
    for _name, text in reference_boots:
        for s, d in headroom_by_chunk_index(text).items():
            if 0 in d:
                ref0[s] = min(ref0.get(s, d[0]), d[0])
    k = list(img)
    notes: List[str] = []
    for name, text in over_boots:
        obs = observe_p_log(text)
        h = headroom_by_chunk_index(text)
        if obs["chunk"].get(0) != reference.chunk or bool(obs["draft_on_p"][0]) != reference.draft_on_p:
            raise ValueError(
                "Ueber-Boot %s: Chunk %s / Draft %s, die Referenz %s hat %d / %s"
                % (name, obs["chunk"].get(0), obs["draft_on_p"][0], reference.source,
                   reference.chunk, reference.draft_on_p)
            )
        for s in range(n):
            if s not in obs["buffer"] or 0 not in h.get(s, {}) or s not in ref0:
                continue
            drows = int(obs["buffer"][s]) - int(reference.buffer_rows[s])
            if drows <= 0:
                continue
            kv = float(obs["tokens"].get(s, 0)) * float(obs["cell"].get(s, 0)) / MIB
            val = (ref0[s] - h[s][0] - (kv - float(reference.kv_mib[s]))) / drows
            notes.append(
                "stage%d %.1f MiB/Zeile = (Referenz %.0f - %s %.0f am Chunk 0) / %d Zeilen "
                "(Bild %.1f%s)"
                % (s, val, ref0[s], name, h[s][0], drows, img[s],
                   "" if val >= img[s] else "; darunter gilt das Bild")
            )
            k[s] = max(k[s], val)
    return tuple(round(x, 1) for x in k), "; ".join(notes)


def p_card_reference_from_logs(
    boots: Sequence[Tuple[str, str]],
    *,
    stage_layers: Sequence[int],
    row_mib: float,
    support: TransientSupport,
    model: str,
    over_boots: Sequence[Tuple[str, str]] = (),
) -> PCardReference:
    """Die P-Karten-Referenz aus P-Logs MESSEN (``boots`` = (Name, Text)).

    Je Stufe: der Chunk-0-Punkt (Minimum ueber die Boots), normiert mit
    Puffer, KV-Zelle x Token und Stuetzpunkt-Transiente; das Wachstum aus den
    Peak-Werten der gemessenen Chunk-Indizes (je Token); der LMEM-Posten als
    cap-Abfall zwischen Chunk 0 und dem letzten Punkt; der laengste Prompt.
    ``over_boots`` (mehr Zeilen, lebend oder tot) messen den Zeilenpreis.
    """
    n = len(stage_layers)
    best: Dict[int, Tuple[float, PoolSample, int, float]] = {}
    growth: Dict[int, float] = {}
    gidx: Dict[int, int] = {}
    lmem: Dict[int, float] = {}
    chunks = set()
    draft = set()
    longest = 0
    for _name, text in boots:
        obs = observe_p_log(text)
        samples = pool_samples(text)
        draft.add(bool(obs["draft_on_p"][0]))
        longest = max(longest, longest_prompt_tokens(text))
        for s in range(n):
            ss = samples.get(s, [])
            first = [p for p in ss if p.chunk_index == 0]
            if not first or s not in obs["buffer"] or s not in obs["cell"] or s not in obs["tokens"]:
                continue
            c = int(obs["chunk"][s])
            chunks.add(c)
            p0 = min(first, key=lambda p: p.headroom_mib)
            rows = int(obs["buffer"][s])
            kv = float(obs["tokens"][s]) * float(obs["cell"][s]) / MIB
            t, _src = transient_mib(support, s, c)
            h0 = p0.headroom_mib + rows * int(stage_layers[s]) * float(row_mib) + kv + t
            if s not in best or h0 < best[s][0]:
                best[s] = (h0, p0, rows, kv)
            last = max(ss, key=lambda p: (p.chunk_index, p.peak_mib))
            if last.chunk_index > 0:
                g = (last.peak_mib - p0.peak_mib) / (last.chunk_index * c)
                growth[s] = max(growth.get(s, 0.0), g)
                gidx[s] = max(gidx.get(s, 0), last.chunk_index)
            lmem[s] = max(lmem.get(s, 0.0), max(0.0, p0.cap_mib - min(p.cap_mib for p in ss)))
    missing = [s for s in range(n) if s not in best or s not in growth]
    if missing:
        raise ValueError(
            "P-Karten-Referenz unvollstaendig in %s: Stufe %s ohne Chunk-0-Punkt, ohne "
            "spaeteren Punkt (Wachstum), Pufferzeile, KV-Zelle oder Chunk"
            % ([b[0] for b in boots], missing)
        )
    if len(chunks) != 1 or len(draft) != 1:
        raise ValueError(
            "P-Karten-Referenz %s mischt Chunks %s bzw. Draft-auf-P %s; eine Referenz "
            "ist EINE Form" % ([b[0] for b in boots], sorted(chunks), sorted(draft))
        )
    ref = PCardReference(
        source=" + ".join(b[0] for b in boots),
        model=model,
        stage_layers=tuple(int(x) for x in stage_layers),
        chunk=int(next(iter(chunks))),
        headroom0_mib=tuple(round(best[s][0], 1) for s in range(n)),
        headroom_mib=tuple(round(best[s][1].headroom_mib, 1) for s in range(n)),
        buffer_rows=tuple(best[s][2] for s in range(n)),
        kv_mib=tuple(round(best[s][3], 1) for s in range(n)),
        cap_mib=tuple(best[s][1].cap_mib for s in range(n)),
        peak_mib=tuple(best[s][1].peak_mib for s in range(n)),
        private_free_mib=tuple(best[s][1].private_free_mib for s in range(n)),
        draft_on_p=bool(next(iter(draft))),
        growth_mib_per_token=tuple(round(growth[s], 7) for s in range(n)),
        growth_measured_chunks=tuple(gidx[s] for s in range(n)),
        longest_prompt_tokens=int(longest),
        lmem_mib=tuple(round(lmem[s], 1) for s in range(n)),
    )
    if not over_boots:
        return ref
    cost, src = row_card_cost_from_boots(boots, over_boots, reference=ref, row_mib=row_mib)
    return msgspec.structs.replace(ref, row_card_mib=cost, row_card_source=src)


#: HISTORISCH (Baum bis 78878cdd58, vor H39/H44/H46/H47): die P-Karten-Referenz
#: aus fnFL2x146 (012db511b8) und fnFL2x150 (78878cdd58), beide Schnitt 29,11,8,
#: Chunk 16384, FR_P 0.26,0.45,0.39 -> H25 0.733887 (Puffer 166/263/408
#: Zeilen), KV 262144 x 7616/3264/2176 B, 97k-Prompt (97841 Token). Chunk 0
#: PP0: cap 28953 - peak 21965 - privat_frei 2394 = 4594 MiB; Wachstum der
#: Spitze 320.7 MiB je 16k-Chunk (gemessen bis Chunk 3); LMEM-Posten der
#: Referenz 0 / 150 / 48 MiB (PP0: Run-Write scheiterte, Fallback paged-copy).
#: Zeilenpreis ueber der Referenz aus fnFL2x149 Chunk 0: PP0 71.6 MiB (Bild
#: 70.1); PP1 26.4 < Bild -> Bild. Zeile 2.4170 MiB (512 Experten, 1237.5 MiB
#: je Layer). Auffrischen: ``--p-card-reference-logs`` / ``--p-card-over-logs``.
P_CARD_REFERENCE_FNFL2_X150 = PCardReference(
    source="fnFL2x146 + fnFL2x150",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    stage_layers=(29, 11, 8),
    chunk=16384,
    headroom0_mib=(21820.2, 15225.5, 13990.8),
    headroom_mib=(4584.0, 4130.0, 2291.0),
    buffer_rows=(166, 263, 408),
    kv_mib=(1904.0, 816.0, 544.0),
    cap_mib=(28943.0, 18614.0, 18622.0),
    peak_mib=(21965.0, 13490.0, 14430.0),
    private_free_mib=(2394.0, 994.0, 1901.0),
    draft_on_p=False,
    growth_mib_per_token=(0.0195719, 0.0198161, 0.0200806),
    growth_measured_chunks=(3, 3, 1),
    longest_prompt_tokens=97841,
    lmem_mib=(0.0, 150.0, 48.0),
    row_card_mib=(71.4, 26.6, 19.3),
    row_card_source=(
        "stage0 71.4 MiB/Zeile = (Referenz 4584 - fnFL2x149 1300 am Chunk 0) / 46 Zeilen "
        "(Bild 70.1); stage1 26.4 MiB/Zeile = (Referenz 4130 - fnFL2x149 1545 am Chunk 0) "
        "/ 98 Zeilen (Bild 26.6; darunter gilt das Bild)"
    ),
)


#: Die gemessene P-Karten-Referenz des HEUTIGEN Baums (76ce5580d4 = H39 Repack
#: ausserhalb der Tag-Pools, H44/H46 Lanes, H47 Run-Write-Fix + LMEM-Bedarf),
#: hergeleitet aus fnFL2x160 (FR_P 0.332,0.605,0.39 -> H25 0.733887, Puffer
#: 202/342/408 Zeilen, Chunk 16384, KV 262144 x 7616/3264/2176 B), der den
#: 97k- UND den 259441-Token-Prompt (16 Chunks) ohne OOM lief. Gegen die
#: historische Referenz: privat_frei 2394 -> 225 MiB auf PP0 (173 PP1, 142
#: PP2), daher liegt der Kopfraum bei MEHR Zeilen hoeher -- eine Referenz ist
#: ein Baumstand und wird mit ihm aufgefrischt. Chunk 0 PP0: cap 28967 -
#: peak 24490 - privat_frei 225 = 4252 MiB; Wachstum 990 / 974 / 329 MiB,
#: saettigt nach Chunk 3 / 3 / 1 (gemessen bis Chunk 15); kein LMEM-Abfall
#: (0 / 2 / 2 MiB). Zeilenpreis: das Bild (kein Boot mit mehr Zeilen auf
#: diesem Baum). Auffrischen: ``--p-card-reference-logs``.
P_CARD_REFERENCE_FNFL2 = PCardReference(
    source="fnFL2x160",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    stage_layers=(29, 11, 8),
    chunk=16384,
    headroom0_mib=(24011.6, 16054.9, 15756.8),
    headroom_mib=(4252.0, 2859.0, 4057.0),
    buffer_rows=(202, 342, 408),
    kv_mib=(1904.0, 816.0, 544.0),
    cap_mib=(28967.0, 18626.0, 18634.0),
    peak_mib=(24490.0, 15594.0, 14435.0),
    private_free_mib=(225.0, 173.0, 142.0),
    draft_on_p=False,
    growth_mib_per_token=(0.0201416, 0.0198161, 0.0200806),
    growth_measured_chunks=(3, 3, 1),
    longest_prompt_tokens=259441,
    lmem_mib=(0.0, 2.0, 2.0),
)


class PCardFit(msgspec.Struct, frozen=True, kw_only=True):
    """Eine P-Stufe auf ihrer Karte fuer (Fraction, Chunk, Prompt)."""

    stage: int
    card: str
    fraction: float
    buffer_rows: int
    layer_row_mib: float
    #: Preis einer Zeile ueber der Referenz (>= layer_row_mib).
    row_card_mib: float
    expert_mib: float
    kv_mib: float
    transient_mib: float
    transient_ref_mib: float
    transient_source: str
    draft_mib: float
    prompt_tokens: int
    last_chunk_index: int
    growth_mib: float
    growth_extrapolated: bool
    lmem_mib: float
    lmem_source: str
    headroom_mib: float
    near_oom_mib: float
    ceiling_fraction: Optional[float]
    ceiling_max_rows: int

    @property
    def refused(self) -> bool:
        return self.headroom_mib < self.near_oom_mib

    @property
    def verdict(self) -> str:
        return "STIRBT IM CHUNK-FORWARD" if self.refused else "PASST"


def solve_p_card(
    *,
    reference: PCardReference,
    support: TransientSupport,
    fractions: Sequence[float],
    lru_rows: Sequence[int],
    stage_layers: Sequence[int],
    chunk: int,
    kv_mib: Sequence[float],
    num_experts: int,
    row_mib: float,
    draft_on_p: bool = False,
    draft_mib_last_stage: float = 0.0,
    cards: Sequence[str] = (),
    near_oom_mib: float = NEAR_OOM_MIB,
    prompt_tokens: int = 0,
    lmem_fixed: bool = True,
    use_growth: bool = True,
) -> Tuple[PCardFit, ...]:
    """Je P-Stufe der Kopfraum am LETZTEN Chunk eines Prompts von
    ``prompt_tokens`` Token (0 = der laengste Prompt der Referenz)::

        K0 - Zeilen x Bild - Ueber x (Preis - Bild) - KV - T(chunk) - Draft
           - g x (Token vor dem letzten Chunk) - LMEM   >= near-OOM

    ``lmem_fixed``: der Baum traegt den H47-Fix (Run-Write in 1-KiB-Elementen)
    -> LMEM 0; sonst der gemessene Hochstand je Stufe (PP0 aus x149).
    ``use_growth=False`` ist nur fuer den Mutanten-Test.
    """
    from sglang.srt.planner import expert_residency as _er

    n = len(reference.headroom0_mib)
    if tuple(int(x) for x in stage_layers) != tuple(reference.stage_layers):
        raise ValueError(
            "P-Karte: Schnitt %s, die Referenz (%s) ist bei %s gemessen; der Kopfraum "
            "einer Stufe haengt an ihrer Layerzahl. Boots DIESES Schnitts per "
            "--p-card-reference-logs geben"
            % (list(stage_layers), reference.source, list(reference.stage_layers))
        )
    if len(fractions) != n or len(lru_rows) != n or len(kv_mib) != n:
        raise ValueError(
            "P-Karte: %d Stufen, aber %d Fractions / %d LRU / %d KV"
            % (n, len(fractions), len(lru_rows), len(kv_mib))
        )
    prompt = int(prompt_tokens) if int(prompt_tokens) > 0 else int(reference.longest_prompt_tokens)
    c = int(chunk)
    n_chunks = max(1, -(-prompt // c))
    last_idx = n_chunks - 1
    out: List[PCardFit] = []
    for s in range(n):
        layer_mib = int(stage_layers[s]) * float(row_mib)
        rows = _er.buffer_rows(
            local_experts=int(num_experts),
            fraction=float(fractions[s]),
            scratch_rows=int(lru_rows[s]),
        )
        t, src = transient_mib(support, s, c)
        draft = 0.0
        if s == n - 1:
            draft = (int(bool(draft_on_p)) - int(bool(reference.draft_on_p))) * float(
                draft_mib_last_stage
            )
        # H41d: das Wachstum saettigt am letzten wachsenden Chunk der Referenz
        # (x160: gemessen ueber 16 Chunks); darueber hinaus kein Term.
        sat_idx = min(last_idx, int(reference.growth_measured_chunks[s]))
        growth = (
            float(reference.growth_mib_per_token[s]) * sat_idx * int(reference.chunk)
            if use_growth else 0.0
        )
        ref_last = max(0, -(-int(reference.longest_prompt_tokens) // c) - 1)
        extrapolated = last_idx > ref_last
        if lmem_fixed:
            lmem, lsrc = 0.0, "0 (H47-Fix: Run-Write in 1-KiB-Elementen)"
        else:
            lmem = max(float(reference.lmem_mib[s]), float(P_LMEM_RUN_WRITE_MIB[s]))
            lsrc = "%.0f (ohne H47-Fix: gemessener Run-Write-Hochstand)" % lmem
        ref_rows = int(reference.buffer_rows[s])
        row_card = (
            max(layer_mib, float(reference.row_card_mib[s]))
            if reference.row_card_mib
            else layer_mib
        )
        rest = float(reference.headroom0_mib[s]) - float(kv_mib[s]) - t - draft - growth - lmem
        over = max(0, int(rows) - ref_rows)
        head = rest - rows * layer_mib - over * (row_card - layer_mib)
        at_ref = rest - ref_rows * layer_mib - float(near_oom_mib)
        if at_ref >= 0.0:
            max_rows = ref_rows + int(math.floor(at_ref / row_card))
        else:
            max_rows = int(math.floor((rest - float(near_oom_mib)) / layer_mib))
        out.append(
            PCardFit(
                stage=s,
                card=str(cards[s]) if s < len(cards) else "stage%d" % s,
                fraction=float(fractions[s]),
                buffer_rows=int(rows),
                layer_row_mib=layer_mib,
                row_card_mib=row_card,
                expert_mib=rows * layer_mib + over * (row_card - layer_mib),
                kv_mib=float(kv_mib[s]),
                transient_mib=t,
                transient_ref_mib=transient_mib(support, s, reference.chunk)[0],
                transient_source=src,
                draft_mib=draft,
                prompt_tokens=prompt,
                last_chunk_index=last_idx,
                growth_mib=growth,
                growth_extrapolated=bool(extrapolated and use_growth),
                lmem_mib=lmem,
                lmem_source=lsrc,
                headroom_mib=head,
                near_oom_mib=float(near_oom_mib),
                ceiling_fraction=_er.largest_fraction_for_rows(
                    local_experts=int(num_experts),
                    scratch_rows=int(lru_rows[s]),
                    max_rows=max_rows,
                ),
                ceiling_max_rows=max_rows,
            )
        )
    return tuple(out)


def describe_p_card(fit: PCardFit, reference: PCardReference) -> str:
    s = fit.stage
    ceiling = "KEINE" if fit.ceiling_fraction is None else "%.3f" % fit.ceiling_fraction
    return (
        "%s stage%d (%s) prompt=%d: Referenz %s Kopfraum Chunk 0 %.0f = cap %.0f - peak "
        "%.0f - privat_frei %.0f MiB bei %d Zeilen, KV %.0f, Chunk %d; hier f %.4f -> %d "
        "Zeilen (%+.0f MiB; je Zeile ueber der Referenz %.1f, Bild %.1f), KV %.0f (%+.0f), "
        "Transiente %.0f (%+.0f, %s)%s, Chunk-Wachstum am Chunk %d %.0f MiB (%.1f je "
        "16k, saettigt nach Chunk %d, gemessen ueber %d Token%s), LMEM %s -> Kopfraum %.0f "
        "MiB (near-OOM %.0f) -> "
        "%s | KARTEN-DECKE f %s (<= %d Zeilen)"
        % (
            CARD_MARKER,
            s,
            fit.card,
            fit.prompt_tokens,
            reference.source,
            reference.headroom_mib[s],
            reference.cap_mib[s],
            reference.peak_mib[s],
            reference.private_free_mib[s],
            reference.buffer_rows[s],
            reference.kv_mib[s],
            reference.chunk,
            fit.fraction,
            fit.buffer_rows,
            fit.expert_mib - reference.buffer_rows[s] * fit.layer_row_mib,
            fit.row_card_mib,
            fit.layer_row_mib,
            fit.kv_mib,
            fit.kv_mib - reference.kv_mib[s],
            fit.transient_mib,
            fit.transient_mib - fit.transient_ref_mib,
            fit.transient_source,
            (" Draft %+.0f" % fit.draft_mib) if fit.draft_mib else "",
            fit.last_chunk_index,
            fit.growth_mib,
            reference.growth_mib_per_token[s] * 16384,
            reference.growth_measured_chunks[s],
            reference.longest_prompt_tokens,
            ", Prompt laenger als gemessen: HOCHRECHNUNG" if fit.growth_extrapolated else "",
            fit.lmem_source,
            fit.headroom_mib,
            fit.near_oom_mib,
            fit.verdict,
            ceiling,
            fit.ceiling_max_rows,
        )
    )


def p_card_refusal_text(
    fits: Sequence[PCardFit], reference: PCardReference, *, chunk: int
) -> Optional[str]:
    bad = [f for f in fits if f.refused]
    if not bad:
        return None
    return (
        "%s (P, chunk %d, prompt %d): die Experten-Residenz passt ins Budget, aber nicht "
        "auf die KARTE am letzten Chunk -- %s. Gemessen: Kopfraum am Chunk 0 von %s, "
        "verschoben um Puffer (je Zeile ueber der Referenz zum gemessenen Preis), KV, "
        "Chunk-Transiente, Draft, Chunk-Wachstum der Spitze und den Run-Write-LMEM; "
        "fnFL2x121 (FR_P[0] 0.45) und x122 (0.40) starben im ersten 16k-Chunk, "
        "fnFL2x149 (0.351) im zweiten am Run-Write-LMEM (H47). "
        "Groesste tragbare Fraction je Stufe: %s."
        % (
            CARD_REFUSAL_CODE,
            int(chunk),
            fits[0].prompt_tokens if fits else 0,
            "; ".join(
                "stage%d (%s) f %.4f -> %d Zeilen, Kopfraum %.0f MiB < near-OOM %.0f"
                % (f.stage, f.card, f.fraction, f.buffer_rows, f.headroom_mib, f.near_oom_mib)
                for f in bad
            ),
            reference.source,
            ",".join(
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ),
        )
    )


#: H47: der Run-Write-LMEM je P-Stufe OHNE Fix, MiB -- der gemessene
#: Hochstand (x149/x151 PP0 cap -1457/-1456; x146 PP1 -150, PP2 -48). Gesetz
#: stack = run/32 - 64 B je residentem Thread (SM x 1536), einmal je Karte.
P_LMEM_RUN_WRITE_MIB = (1457.0, 150.0, 48.0)


def arena_write_lmem_fixed() -> bool:
    """Traegt DIESER Baum den H47-Fix (``arena_write.RUN_ELEMENT_BYTES``)?"""
    try:
        from sglang.srt.weg2 import arena_write as _aw
    except Exception:  # noqa: BLE001 -- ohne Modul kein Run-Write, also kein LMEM
        return True
    return hasattr(_aw, "RUN_ELEMENT_BYTES")
