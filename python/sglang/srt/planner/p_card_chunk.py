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
# 2. die P-Karte: gemessener Kopfraum, um Puffer, KV, Chunk und Draft verschoben
# ---------------------------------------------------------------------------


class PCardReference(msgspec.Struct, frozen=True, kw_only=True):
    """Der Kopfraum am bindenden Punkt eines Referenz-Boots, je P-Stufe."""

    source: str
    model: str
    stage_layers: Tuple[int, ...]
    chunk: int
    #: ``Kopfraum + Puffer x L x Zeile + KV + T(chunk)`` -- die Karte ohne
    #: Experten-Puffer, KV und Chunk-Transiente, MiB; Minimum ueber die Boots.
    headroom0_mib: Tuple[float, ...]
    #: Die Terme des bindenden Punkts, fuer den Druck.
    headroom_mib: Tuple[float, ...]
    buffer_rows: Tuple[int, ...]
    kv_mib: Tuple[float, ...]
    cap_mib: Tuple[float, ...]
    peak_mib: Tuple[float, ...]
    private_free_mib: Tuple[float, ...]
    phase: Tuple[str, ...]
    draft_on_p: bool


def p_card_reference_from_logs(
    boots: Sequence[Tuple[str, str]],
    *,
    stage_layers: Sequence[int],
    row_mib: float,
    support: TransientSupport,
    model: str,
) -> PCardReference:
    """Die P-Karten-Referenz aus P-Logs MESSEN (``boots`` = (Name, Text)).

    Je Boot und Stufe der ``WEG2-GRAPH-POOL``-Punkt mit dem kleinsten
    Kopfraum (``graph_pool_ledger.binding_sample``), normiert mit dem Puffer,
    der KV-Zelle x Token und der Stuetzpunkt-Transiente DESSELBEN Logs. Fehlt
    einer Stufe einer der Terme in ALLEN Boots, wird verweigert, statt eine
    Null einzusetzen.
    """
    from sglang.srt.planner import graph_pool_ledger as gpl

    n = len(stage_layers)
    best: Dict[int, Tuple[float, object, int, float]] = {}
    chunks = set()
    draft = set()
    for _name, text in boots:
        obs = observe_p_log(text)
        samples = gpl.samples_from_log(text)
        draft.add(bool(obs["draft_on_p"][0]))
        for s in range(n):
            if s not in samples or s not in obs["buffer"] or s not in obs["cell"]:
                continue
            if s not in obs["chunk"] or s not in obs["tokens"]:
                continue
            smp = gpl.binding_sample(samples[s])
            if smp is None:
                continue
            c = int(obs["chunk"][s])
            chunks.add(c)
            rows = int(obs["buffer"][s])
            kv = float(obs["tokens"][s]) * float(obs["cell"][s]) / MIB
            t, _src = transient_mib(support, s, c)
            h0 = smp.headroom_mib + rows * int(stage_layers[s]) * float(row_mib) + kv + t
            if s not in best or h0 < best[s][0]:
                best[s] = (h0, smp, rows, kv)
    missing = [s for s in range(n) if s not in best]
    if missing:
        raise ValueError(
            "P-Karten-Referenz unvollstaendig in %s: Stufe %s ohne WEG2-GRAPH-POOL-"
            "Punkt, Pufferzeile, KV-Zelle oder Chunk" % ([b[0] for b in boots], missing)
        )
    if len(chunks) != 1 or len(draft) != 1:
        raise ValueError(
            "P-Karten-Referenz %s mischt Chunks %s bzw. Draft-auf-P %s; eine Referenz "
            "ist EINE Form" % ([b[0] for b in boots], sorted(chunks), sorted(draft))
        )
    return PCardReference(
        source=" + ".join(b[0] for b in boots),
        model=model,
        stage_layers=tuple(int(x) for x in stage_layers),
        chunk=int(next(iter(chunks))),
        headroom0_mib=tuple(round(best[s][0], 1) for s in range(n)),
        headroom_mib=tuple(round(best[s][1].headroom_mib, 1) for s in range(n)),
        buffer_rows=tuple(best[s][2] for s in range(n)),
        kv_mib=tuple(round(best[s][3], 1) for s in range(n)),
        cap_mib=tuple(round(best[s][1].cap_mib, 1) for s in range(n)),
        peak_mib=tuple(round(best[s][1].peak_mib, 1) for s in range(n)),
        private_free_mib=tuple(round(best[s][1].private_free_mib, 1) for s in range(n)),
        phase=tuple(best[s][1].phase for s in range(n)),
        draft_on_p=bool(next(iter(draft))),
    )


#: Die gemessene P-Karten-Referenz der Next-Flash-Bestform (H25-Form, kein
#: Draft auf P), hergeleitet von :func:`p_card_reference_from_logs` aus
#: fnFL2x145 (9310d2893a) und fnFL2x146 (012db511b8), beide Schnitt 29,11,8,
#: Chunk 16384, FR_P 0.26,0.45,0.39 -> H25 0.26,0.45,0.733887 (Puffer
#: 166/263/408 Zeilen), KV 262144 x 7616/3264/2176 B. Bindend ist auf jeder
#: Stufe der letzte ``high-water``-Punkt des 97k-Prompts (PP0 x146: cap 28953
#: - peak 22927 - privat_frei 2394 = 3632 MiB). Zeile 2.4170 MiB (512 Experten,
#: 1237.5 MiB je Layer, aus den Safetensors-Headern). Auffrischen:
#: ``--p-card-reference-logs``.
P_CARD_REFERENCE_FNFL2 = PCardReference(
    source="fnFL2x145 + fnFL2x146",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    stage_layers=(29, 11, 8),
    chunk=16384,
    headroom0_mib=(20868.2, 14101.5, 13613.8),
    headroom_mib=(3632.0, 3006.0, 1914.0),
    buffer_rows=(166, 263, 408),
    kv_mib=(1904.0, 816.0, 544.0),
    cap_mib=(28953.0, 18464.0, 18574.0),
    peak_mib=(22927.0, 14464.0, 14759.0),
    private_free_mib=(2394.0, 994.0, 1901.0),
    phase=("high-water", "high-water", "high-water"),
    draft_on_p=False,
)


class PCardFit(msgspec.Struct, frozen=True, kw_only=True):
    """Eine P-Stufe auf ihrer Karte fuer (Fraction, Chunk)."""

    stage: int
    card: str
    fraction: float
    buffer_rows: int
    layer_row_mib: float
    expert_mib: float
    kv_mib: float
    transient_mib: float
    #: T_s bei der Chunkbreite der Referenz (die Verschiebung ist die Differenz).
    transient_ref_mib: float
    transient_source: str
    draft_mib: float
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
) -> Tuple[PCardFit, ...]:
    """Je P-Stufe::

        Kopfraum_s = K0_s - Puffer_s x L_s x Zeile - KV_s - T_s(chunk) - Draft_s
                     >= near-OOM

    ``Draft_s`` gilt auf der LETZTEN Stufe und ist die Differenz der Draft-
    Form gegen die Referenz: ``(draft_on_p - reference.draft_on_p) x
    draft_mib_last_stage`` (``draft_post.p_draft_post_mib``: Gewichte + Puffer
    + Produzenten-Transiente, derselbe Posten, den H25 in Zeilen umrechnet). ``PChunkUnmeasured`` fuer einen Chunk
    ausserhalb der Stuetzpunkte; ``ValueError`` fuer einen anderen Schnitt als
    den der Referenz (der Kopfraum ist je Layerzahl gemessen).
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
    out: List[PCardFit] = []
    for s in range(n):
        layer_mib = int(stage_layers[s]) * float(row_mib)
        rows = _er.buffer_rows(
            local_experts=int(num_experts),
            fraction=float(fractions[s]),
            scratch_rows=int(lru_rows[s]),
        )
        t, src = transient_mib(support, s, chunk)
        draft = 0.0
        if s == n - 1:
            draft = (int(bool(draft_on_p)) - int(bool(reference.draft_on_p))) * float(
                draft_mib_last_stage
            )
        rest = float(reference.headroom0_mib[s]) - float(kv_mib[s]) - t - draft
        head = rest - rows * layer_mib
        max_rows = int(math.floor((rest - float(near_oom_mib)) / layer_mib))
        out.append(
            PCardFit(
                stage=s,
                card=str(cards[s]) if s < len(cards) else "stage%d" % s,
                fraction=float(fractions[s]),
                buffer_rows=int(rows),
                layer_row_mib=layer_mib,
                expert_mib=rows * layer_mib,
                kv_mib=float(kv_mib[s]),
                transient_mib=t,
                transient_ref_mib=transient_mib(support, s, reference.chunk)[0],
                transient_source=src,
                draft_mib=draft,
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
        "%s stage%d (%s): Referenz %s Kopfraum %.0f = cap %.0f - peak %.0f - "
        "privat_frei %.0f MiB am Punkt '%s' bei %d Zeilen, KV %.0f, Chunk %d; hier "
        "f %.4f -> %d Zeilen (%+.0f MiB), KV %.0f (%+.0f), Transiente %.0f (%+.0f, %s)%s "
        "-> Kopfraum %.0f MiB (near-OOM %.0f) -> %s | KARTEN-DECKE f %s (<= %d Zeilen) "
        "-- Messaufloesung: der bindende Punkt kann bis %.0f MiB unter dem wahren "
        "liegen (Hochwasser-Schritt), gedruckt, nicht abgezogen"
        % (
            CARD_MARKER,
            s,
            fit.card,
            reference.source,
            reference.headroom_mib[s],
            reference.cap_mib[s],
            reference.peak_mib[s],
            reference.private_free_mib[s],
            reference.phase[s],
            reference.buffer_rows[s],
            reference.kv_mib[s],
            reference.chunk,
            fit.fraction,
            fit.buffer_rows,
            (fit.buffer_rows - reference.buffer_rows[s]) * fit.layer_row_mib,
            fit.kv_mib,
            fit.kv_mib - reference.kv_mib[s],
            fit.transient_mib,
            fit.transient_mib - fit.transient_ref_mib,
            fit.transient_source,
            (" Draft %+.0f" % fit.draft_mib) if fit.draft_mib else "",
            fit.headroom_mib,
            fit.near_oom_mib,
            fit.verdict,
            ceiling,
            fit.ceiling_max_rows,
            HIGHWATER_STEP_MIB,
        )
    )


def p_card_refusal_text(
    fits: Sequence[PCardFit], reference: PCardReference, *, chunk: int
) -> Optional[str]:
    bad = [f for f in fits if f.refused]
    if not bad:
        return None
    return (
        "%s (P, chunk %d): die Experten-Residenz passt ins Budget, aber nicht auf die "
        "KARTE im Chunk-Forward -- %s. Gemessen: Kopfraum = cap - peak - privat_frei "
        "am bindenden Punkt von %s, verschoben um Puffer, KV, die Chunk-Transiente "
        "(Stuetzpunkte %s) und den Draft; fnFL2x121 (FR_P[0] 0.45) und x122 (0.40) "
        "starben genau hier (CUDA-OOM im ersten 16k-Chunk, 320/384 MiB). Groesste "
        "tragbare Fraction je Stufe bei diesem Chunk: %s."
        % (
            CARD_REFUSAL_CODE,
            int(chunk),
            "; ".join(
                "stage%d (%s) f %.4f -> %d Zeilen, Kopfraum %.0f MiB < near-OOM %.0f"
                % (f.stage, f.card, f.fraction, f.buffer_rows, f.headroom_mib, f.near_oom_mib)
                for f in bad
            ),
            reference.source,
            "chunk %d" % reference.chunk,
            ",".join(
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ),
        )
    )
