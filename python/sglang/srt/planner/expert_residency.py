# SPDX-License-Identifier: Apache-2.0
"""Die Metallregeln der Experten-Residenz, als EINE Rechnung fuer den Planer.

fnFL2 H8, 23.09. Der D-FRACTION-SOLVE des Launchers nannte fuer D-TP1 eine
Decke 0.807 und fuer D-TP2 0.661 -- beide toeten den Boot (x98/x99: KV-Pool-
ValueError, 288 MiB zuviel, "before a single KV token"). Drei Terme fehlten
ihm, und alle drei sind am Metall gemessen, nicht geraten:

1. DIE PUFFERREGEL. Der GPU-Puffer je Layer ist NICHT ``fraction x Spanne +
   Scratch``, sondern ``min(R + Scratch, E)`` mit ``R = ceil(fraction x E)``
   (``expert_offload.resident_slot_count``) und ``E`` = die LOKALEN Zeilen
   des Rangs: die ``--rank-moe-ratio``-Spanne (Largest-Remainder ueber die
   Checkpoint-Experten, ``distributed.utils.partition_units``) PLUS die
   Pad-Zeile, die der generische uneven Experten-Schnitt an lokalem Index 0
   anlegt (``fused_moe_triton/layer.py``: ``num_local_experts = n_local +
   1``, SGLANG_UNEVEN_MOE_EXPERT_SHARD=1). ``plan_load_time_staging`` baut
   ``buffer_slots = min(R + C, E)`` mit ``C = min(Scratch, E - R)``
   (``scratch_slot_count``). SGLANG_MOE_POOL_STAGING ist KEIN eigener Posten:
   die Staging-Zeilen liegen INNERHALB des Scratch (``min(staging, C-1)``).
   Gemessen (fnFL2x97..x100, ``MoE expert-offload active on layer 47``):
   TP1 E=145 (144+1): 0.75 -> 109+36=145, 0.72 -> 105+40=145, 0.62 -> 90+48=138;
   TP2 E=177 (176+1): 0.54 -> 96+48=144, 0.52 -> 93+48=141; TP0 E=193: 0.006
   -> 2+70=72.

2. DER KV-POSTEN UND DIE AKTIVIERUNG gegen das BUDGET (``--rank-gpu-memory-
   mib``), nicht gegen die Karte: die Runtime rechnet ``rest = budget -
   'weights + runtime state' - mamba - speculative - activation`` und
   verweigert bei ``rest <= 0`` (``model_runner_kv_cache_mixin``), und der
   Pool traegt ``rest / cell`` Token -- 262144 sind Pflicht.

3. DER FESTE RANG-POSTEN: ``'weights + runtime state'`` minus Puffer-Bytes.
   Er ist je Rang am Metall GEMESSEN (KV-budget-posts-Zeile bzw. ValueError
   desselben Boots) und traegt Gate/Other/Draft plus Allokator-Luecke. Der
   Draft-Anteil darin haengt an SGLANG_WEG2_DRAFT_SHARE_EMBED (H1b): mit
   geteiltem Vokabular faellt die eigene BF16-embed_tokens/lm_head-Tabelle des
   Drafts weg (2 x vocab x hidden x 2 B = 2425 MiB auf dem Draft-Host).

4. (H33) DIE KARTE: das Budget passt, die Karte nicht. Mit gedeckeltem KV
   (262144 Token) bleibt Budget liegen, und was die 5090 fuellt -- freie
   Bloecke privater Graph-/Tag-Pools (4993 MiB auf D-TP0, fuer empty_cache
   unerreichbar) plus die Transiente des schwersten Forwards -- steht in
   keinem Budget-Posten. Abschnitt 4b rechnet den Kopfraum gegen die near-
   OOM-Grenze aus gemessenen Punkten (``graph_pool_ledger``, W130).

5. (H50) DER BAUM-ZUSTAND: Posten 3 und 4 sind Messungen EINES Baums. H39
   (SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL, d6b7d4a1d3) nahm auf D-TP0 4354 MiB
   tote Checkpoint-/Repack-Bloecke aus den Tag-Pools (fest 11618 -> 7264 MiB,
   privat_frei 4993 -> 625 MiB, fnFL2x150 gegen x151/x158). Die Referenzen
   gibt es deshalb je Zustand; gewaehlt wird nach dem Schalter der D-Gruppe,
   und der Zustand eines Logs ist die WIRKUNG ("checkpoint-format pool
   RELEASED"), nie "der neueste Boot".

Alles hier ist rein (kein torch), damit der Launcher es ohne CUDA-Import
rechnen kann; die zwei gespiegelten Runtime-Formeln (``resident_rows``,
``expert_span_by_rank``) sind per Test an ihre Runtime-Quellen gebunden.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

MIB = float(1 << 20)
GIB_IN_MIB = 1024.0

#: Der W-Code der Verweigerung. W120 (H5, Platztausch-Puffer) und W121 (Flip-
#: Peer-Leg) sind vergeben.
REFUSAL_CODE = "W122 Weg2ExpertResidencyOverBudget"

#: SGLANG_WEG2_DRAFT_SHARE_EMBED ist ein EnvBool mit Default AN (H1b,
#: environ.py). Die Spiegelung hier ist noetig, weil der Launcher die Env der
#: GRUPPE D liest, nicht seine eigene.
DRAFT_SHARE_EMBED_ENV = "SGLANG_WEG2_DRAFT_SHARE_EMBED"
DRAFT_SHARE_EMBED_DEFAULT = True

#: Die Runtime-Defaults der zwei Pool-Envs (``expert_offload``).
POOL_STAGING_ENV = "SGLANG_MOE_POOL_STAGING"
POOL_STAGING_DEFAULT = 12

#: fnFL2 H50: SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL (H39, d6b7d4a1d3) ist ein
#: EnvBool mit Default AN (environ.py). Er entscheidet, ob die Checkpoint-
#: Tensoren der Dense-Marlin-Linears und der Repack als tote Bloecke in den
#: Tag-Pools liegen (AUS: D-TP0 der Next-Flash-Form 4354 MiB mehr 'weights +
#: runtime state', 4993 statt 625 MiB privat_frei) oder im eigenen Lade-Pool
#: und nach dem Laden zurueckgegeben werden (AN). Die gemessenen Referenzen
#: unten gelten deshalb je Zustand dieses Schalters -- gespiegelt wie
#: DRAFT_SHARE_EMBED, weil der Launcher die Env der GRUPPE D liest.
DENSE_REPACK_OUTSIDE_POOL_ENV = "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL"
DENSE_REPACK_OUTSIDE_POOL_DEFAULT = True

_TRUE = ("true", "1", "yes", "y")
_FALSE = ("false", "0", "no", "n")


# ---------------------------------------------------------------------------
# 1. die Pufferregel
# ---------------------------------------------------------------------------


def resident_rows(local_experts: int, fraction: float) -> int:
    """``expert_offload.resident_slot_count``, Zeichen fuer Zeichen."""
    n = int(math.ceil(float(fraction) * int(local_experts)))
    return max(1, min(int(local_experts), n))


def buffer_rows(*, local_experts: int, fraction: float, scratch_rows: int) -> int:
    """GPU-Zeilen je Layer, die ein Rang bei ``fraction`` wirklich haelt.

    ``fraction >= 1`` oder ``R >= E``: kein Offload, der volle lokale Stapel
    (``plan_load_time_staging`` gibt ``None``). Sonst ``min(R + Scratch, E)``;
    weniger als zwei freie Zeilen verweigert die Runtime
    (``scratch_slot_count``), und genau so verweigert es diese Rechnung.
    """
    E = int(local_experts)
    if float(fraction) >= 1.0:
        return E
    R = resident_rows(E, fraction)
    if R >= E:
        return E
    room = E - R
    if room < 2:
        raise ValueError(
            f"expert pool: {E} lokale Zeilen mit {R} resident lassen {room} "
            f"Scratch-Zeile(n); die Runtime verlangt mindestens 2 "
            f"(scratch_slot_count) -- fraction {fraction} ist zu hoch"
        )
    return min(R + min(int(scratch_rows), room), E)


def expert_span_by_rank(
    *, num_experts: int, ratios: Sequence[float]
) -> Tuple[int, ...]:
    """``distributed.utils._partition_units_raw``: Largest-Remainder, jeder
    Rang >= 1, Gleichstand zum kleineren Rang. Die Summe ist ``num_experts``;
    die Ratios sind ein VERHAELTNIS, nicht die Zeilenzahl."""
    n = len(ratios)
    units = int(num_experts)
    if units < n:
        raise ValueError(f"{units} Experten reichen nicht fuer {n} Raenge")
    total_w = float(sum(float(w) for w in ratios))
    if total_w <= 0.0:
        raise ValueError(f"--rank-moe-ratio {list(ratios)} hat Summe <= 0")
    quotas = [units * float(w) / total_w for w in ratios]
    sizes = [max(int(q), 1) for q in quotas]
    remaining = units - sum(sizes)
    for _ in range(max(0, -remaining)):
        i = max(range(n), key=lambda r: (sizes[r], -r))
        sizes[i] -= 1
    remaining = max(0, remaining)
    order = sorted(
        range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True
    )
    for k in range(remaining):
        sizes[order[k % n]] += 1
    return tuple(sizes)


def largest_fraction_for_rows(
    *, local_experts: int, scratch_rows: int, max_rows: int
) -> Optional[float]:
    """Die GROESSTE Fraction (3 Stellen, wie sie im argv steht), deren Puffer
    in ``max_rows`` Zeilen passt, oder ``None``, wenn nicht einmal ein
    residenter Experte plus Scratch passt.

    Nach oben begrenzt auf ``R = E - 2``: das ist die groesste Fraction, die
    noch einen Offload-Puffer baut (darueber verweigert ``scratch_slot_count``;
    bei ``fraction >= 1`` baut die Runtime keinen Pool und der Platztausch
    hat keinen Puffer -- H5, W120).
    """
    E = int(local_experts)
    S = int(scratch_rows)
    M = int(max_rows)
    R = E - 2 if M >= E else min(M - S, E - 2)
    if R < 1:
        return None
    f = math.floor(R * 1000 / E) / 1000.0
    while f > 0.0 and resident_rows(E, f) > R:
        f = round(f - 0.001, 3)
    if f <= 0.0:
        return None
    return f


def scratch_edge(
    *, local_experts: int, fraction: float, max_rows: int
) -> Tuple[int, Optional[int]]:
    """H50: die KANTE bei GEGEBENER Fraction -- ``(Zeilen, Scratch)``: wie viele
    Pufferzeilen je Layer hoechstens tragen (``max_rows``, gedeckelt auf E) und
    welches SGLANG_MOE_SCRATCH_SLOTS sie ergibt (``Zeilen - R``); ``None``,
    wenn weniger als die zwei Scratch-Zeilen bleiben, die die Runtime verlangt
    (``scratch_slot_count``). Keine Reserve: die Grenze ist die der Bilanz."""
    E = int(local_experts)
    R = resident_rows(E, fraction) if float(fraction) < 1.0 else E
    rows = min(int(max_rows), E)
    s = rows - R
    return rows, (s if s >= 2 else None)


# ---------------------------------------------------------------------------
# 2. der Draft mit geteiltem Vokabular
# ---------------------------------------------------------------------------


def draft_share_embed(env: Mapping[str, str]) -> bool:
    """SGLANG_WEG2_DRAFT_SHARE_EMBED aus der Gruppen-Env, EnvBool-Semantik."""
    raw = str(env.get(DRAFT_SHARE_EMBED_ENV, "")).strip().lower()
    if not raw:
        return DRAFT_SHARE_EMBED_DEFAULT
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f'{DRAFT_SHARE_EMBED_ENV}="{raw}" ist kein Boolean')


def dense_repack_outside_pool(env: Mapping[str, str]) -> bool:
    """SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL aus der Gruppen-Env, EnvBool-Semantik
    (H50: welcher Baum-Zustand die gemessene Referenz waehlt)."""
    raw = str(env.get(DENSE_REPACK_OUTSIDE_POOL_ENV, "")).strip().lower()
    if not raw:
        return DENSE_REPACK_OUTSIDE_POOL_DEFAULT
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f'{DENSE_REPACK_OUTSIDE_POOL_ENV}="{raw}" ist kein Boolean')


#: Die Zeile, die ein Rang NUR druckt, wenn H39 am Metall gewirkt hat: der
#: Checkpoint-Format-Pool existierte und wurde nach dem Laden zurueckgegeben
#: (``weg2_memory_saver._release_one_load_pool("ckpt")``; ein noch lebender
#: Block haelt den Pool und die Zeile bleibt aus). Der Zustand einer Referenz
#: ist damit die gemessene WIRKUNG, nicht der Schalter.
_RX_H39_RELEASED = re.compile(
    r"\[(?:[0-9-]+ [0-9:]+ )?TP\d+\] WEG2-TAG-POOL checkpoint-format pool RELEASED"
)


def boot_dense_repack_outside_pool(text: str) -> bool:
    """Hat dieser D-Log den H39-Zustand (Checkpoint-Format-Pool freigegeben)?"""
    return _RX_H39_RELEASED.search(text) is not None


def _boots_dense_repack_state(boots: Sequence[Tuple[str, str]]) -> bool:
    """Der H39-Zustand EINER Referenz; Boots aus beiden Zustaenden mischen hiesse
    ueber 4,3 GiB auf der 5090 das Maximum bzw. Minimum zweier Baeume nehmen --
    das wird verweigert, nicht gemittelt."""
    states = {name: boot_dense_repack_outside_pool(text) for name, text in boots}
    if len(set(states.values())) > 1:
        raise ValueError(
            "Referenz mischt Baeume vor und nach H39 (checkpoint-format pool "
            "RELEASED): %s" % states
        )
    return bool(next(iter(states.values()), False))


def draft_vocab_mib(
    *, vocab_size: int, hidden_size: int, dtype_bytes: int = 2
) -> float:
    """Die EIGENEN Vokabular-Tabellen eines NEXTN-Drafts (embed_tokens +
    lm_head, BF16), die ohne H1b neben denen des Ziels liegen. Gemessen
    fnFL2x98..x100 ``pp0tp0-draft after load``: embed_tokens 1.18 + lm_head
    1.18 GiB = 2 x 248320 x 2560 x 2 B."""
    return 2.0 * int(vocab_size) * int(hidden_size) * int(dtype_bytes) / MIB


# ---------------------------------------------------------------------------
# 3. der gemessene feste Rang-Posten
# ---------------------------------------------------------------------------


class DRankReference(msgspec.Struct, frozen=True, kw_only=True):
    """Was ein D-Boot je Rang NEBEN dem Experten-Puffer ans Budget gebucht
    hat, aus seinem eigenen Log. Je Term das Maximum ueber die Boots."""

    source: str
    #: Modell-Verzeichnisname und --rank-tp-ratio der Referenz-Boots: nur fuer
    #: diese Form gilt ``fixed_mib``.
    model: str
    rank_tp_ratio: str
    #: 'weights + runtime state' minus Puffer x Layer x Zeile, MiB.
    fixed_mib: Tuple[float, ...]
    mamba_mib: Tuple[float, ...]
    spec_mib: Tuple[float, ...]
    activation_mib: Tuple[float, ...]
    kv_cell_bytes: Tuple[int, ...]
    #: Rang, dessen Draft Experten traegt (Form A: der Host); -1 = keiner.
    draft_host_rank: int
    #: Hielt der Draft-Host in der Referenz eigene embed_tokens/lm_head?
    draft_vocab_held: bool
    #: H50: lief die Referenz im H39-Zustand (Dense-Repack ausserhalb der
    #: Tag-Pools, :func:`boot_dense_repack_outside_pool`)?
    dense_repack_outside_pool: bool = False
    #: H64: die Verify-Form der Referenz-Boots (``None`` = rekurrent, sonst
    #: die Ringlaenge L des ReplaySSM-Spec-Rings), aus der Wirkung im Log.
    replayssm_spec_ring_len: Optional[int] = None
    #: H91b: die SITZE (--max-running-requests) der Referenz-Boots. Die
    #: Posten 'mamba state pool' und 'speculative intermediate state' sind
    #: sitz-proportional; ein Boot mit anderer Sitzzahl bucht sie um
    #: (:func:`seat_rebook`), statt die bs1-Zahl stumm weiterzutragen.
    max_running: int = 1


_TP = r"\[(?:[0-9-]+ [0-9:]+ )?TP(\d+)\]"
_RX_BUFFER = re.compile(
    _TP + r" MoE expert-offload active on layer (\d+): \d+/\d+ experts resident "
    r"\+ \d+ scratch \(buffer=(\d+),"
)
_RX_POSTS = re.compile(
    _TP + r" \[world_rank \d+\] KV budget posts \(GiB\): weights \+ runtime "
    r"state=([0-9.]+), mamba state pool=([0-9.]+), speculative intermediate "
    r"state=([0-9.]+), prefill activation reserve=([0-9.]+)"
)
_RX_REFUSED = re.compile(
    r"KV cache under --rank-gpu-memory-mib on rank (\d+): .*? spent on weights "
    r"\+ runtime state ([0-9.]+) GiB; prefill activation reserve ([0-9.]+) GiB"
)
_RX_CELL = re.compile(_TP + r" KV pool sizing: available_bytes=\d+ .*?cell_size=(\d+),")
_RX_DRAFT = re.compile(
    _TP + r" \[vram-census\] pp0tp\d+-draft after load: model tensors on device "
    r"[0-9.]+ GiB = \{([^}]*)\}"
)


def _observe_boot(text: str, *, n_layers: int) -> Dict[str, Dict[int, float]]:
    obs: Dict[str, Dict[int, float]] = {
        k: {}
        for k in (
            "buffer",
            "wr",
            "mamba",
            "spec",
            "act",
            "cell",
            "draft_experts",
            "draft_vocab",
        )
    }
    layer_of: Dict[int, int] = {}
    for line in text.splitlines():
        m = _RX_BUFFER.search(line)
        if m:
            r, layer, buf = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if layer < n_layers and layer >= layer_of.get(r, -1):
                layer_of[r] = layer
                obs["buffer"][r] = float(buf)
            continue
        m = _RX_POSTS.search(line)
        if m:
            r = int(m.group(1))
            obs["wr"][r] = float(m.group(2)) * GIB_IN_MIB
            obs["mamba"][r] = float(m.group(3)) * GIB_IN_MIB
            obs["spec"][r] = float(m.group(4)) * GIB_IN_MIB
            obs["act"][r] = float(m.group(5)) * GIB_IN_MIB
            continue
        m = _RX_REFUSED.search(line)
        if m:
            r = int(m.group(1))
            obs["wr"][r] = float(m.group(2)) * GIB_IN_MIB
            obs["act"][r] = float(m.group(3)) * GIB_IN_MIB
            continue
        m = _RX_CELL.search(line)
        if m:
            obs["cell"][int(m.group(1))] = float(m.group(2))
            continue
        m = _RX_DRAFT.search(line)
        if m:
            r, body = int(m.group(1)), m.group(2)
            obs["draft_experts"][r] = 1.0 if "experts" in body else 0.0
            obs["draft_vocab"][r] = (
                1.0 if ("embed_tokens" in body or "lm_head" in body) else 0.0
            )
    return obs


def d_rank_reference_from_logs(
    boots: Sequence[Tuple[str, str]],
    *,
    n_ranks: int,
    n_layers: int,
    slot_bytes: float,
    model: str,
    rank_tp_ratio: str,
    max_running: int = 1,
) -> DRankReference:
    """Den festen Rang-Posten aus D-Logs MESSEN (``boots`` = (Name, Text)).

    Je Boot und Rang: ``fixed = 'weights + runtime state' - buffer x
    n_layers x slot`` -- beide Zahlen aus DEMSELBEN Log, sonst kein Wert. Ueber
    die Boots gilt je Term das Maximum (der Posten streut gemessen um bis zu
    130 MiB je Rang: Allokator-Luecke des Ladens). Fehlt einem Rang ein Term
    in ALLEN Boots, wird verweigert, statt eine Null einzusetzen.
    """
    layer_mib = float(n_layers) * float(slot_bytes) / MIB
    h39 = _boots_dense_repack_state(boots)
    ring = _boots_replayssm_spec_state(boots)
    fixed: Dict[int, float] = {}
    best: Dict[str, Dict[int, float]] = {
        k: {} for k in ("mamba", "spec", "act", "cell")
    }
    host = -1
    vocab_held = False
    for _name, text in boots:
        obs = _observe_boot(text, n_layers=n_layers)
        for r, wr in obs["wr"].items():
            if r in obs["buffer"]:
                fixed[r] = max(
                    fixed.get(r, float("-inf")), wr - obs["buffer"][r] * layer_mib
                )
        for key in best:
            for r, v in obs[key].items():
                best[key][r] = max(best[key].get(r, float("-inf")), v)
        for r, has in obs["draft_experts"].items():
            if has:
                host = r
                vocab_held = vocab_held or bool(obs["draft_vocab"].get(r, 0.0))
    missing = [
        f"{name} rang {r}"
        for name, table in (
            ("fixed(W+R & buffer)", fixed),
            ("activation", best["act"]),
            ("cell", best["cell"]),
        )
        for r in range(n_ranks)
        if r not in table
    ]
    if missing:
        raise ValueError(
            "D-Referenz unvollstaendig in %s: %s"
            % ([b[0] for b in boots], ", ".join(missing))
        )
    return DRankReference(
        source=" + ".join(b[0] for b in boots),
        model=model,
        rank_tp_ratio=rank_tp_ratio,
        fixed_mib=tuple(round(fixed[r], 1) for r in range(n_ranks)),
        mamba_mib=tuple(round(best["mamba"].get(r, 0.0), 1) for r in range(n_ranks)),
        spec_mib=tuple(round(best["spec"].get(r, 0.0), 1) for r in range(n_ranks)),
        activation_mib=tuple(round(best["act"][r], 1) for r in range(n_ranks)),
        kv_cell_bytes=tuple(int(best["cell"][r]) for r in range(n_ranks)),
        draft_host_rank=host,
        draft_vocab_held=vocab_held,
        dense_repack_outside_pool=h39,
        replayssm_spec_ring_len=ring,
        max_running=max(1, int(max_running)),
    )


#: Die gemessene Referenz der Next-Flash-Form-A-D-Gruppe (Stand 5a96de48be).
#: Sitze: beide eingebauten NF-Referenzen liefen mit --max-running-requests 1
#: (Form A, bs1). Beleg aus ihren eigenen Zahlen: 'mamba state pool' 393.2 MiB
#: = 7 Slots x 56.17 MiB, und 7 = ceil(1 x (3 + 2) x 1.25) ist genau die
#: Slotformel der Runtime (``_auto_mamba_demand_size``, Overlap an) fuer EINEN
#: Sitz; zwei Sitze waeren 13 Slots = 730 MiB. Der Spec-Posten 224.3 MiB = 1
#: Sitz x 4 Draft-Zeilen x 56.07 MiB je Request-Zustand bestaetigt es.
#: Hergeleitet von :func:`d_rank_reference_from_logs` aus den Boots
#: fnFL2x98/x99/x100 (/spinning/evidence-665-f1/boot_weg2_fnFL2x{98,99,100}_*.D.log,
#: dieselben Zeilen liegen als Test-Fixture unter
#: test/registered/unit/weg2/fixtures/d_residency_h8/), der Test
#: ``test_the_shipped_reference_is_the_logs_own_measurement`` bindet sie an
#: diese Logs. Auffrischen: ``--d-residency-reference-logs``.
D_RESIDENCY_REFERENCE_FNFL2 = DRankReference(
    source="fnFL2x98 + fnFL2x99 + fnFL2x100",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    rank_tp_ratio="1,0,0",
    fixed_mib=(13732.3, 954.1, 1035.2),
    mamba_mib=(393.2, 0.0, 0.0),
    spec_mib=(224.3, 0.0, 0.0),
    activation_mib=(1024.0, 1024.0, 1024.0),
    kv_cell_bytes=(14143, 768, 768),
    draft_host_rank=0,
    draft_vocab_held=True,
    dense_repack_outside_pool=False,
)

#: fnFL2 H50: dieselbe Messung im H39-Zustand (Baum ab d6b7d4a1d3), von
#: :func:`d_rank_reference_from_logs` aus fnFL2x151 (d6b7d4a1d3) und fnFL2x158
#: (c01951e3e1), beide FR_D 0.06,0.51,0.48 / SCRATCH_D 82,48,48
#: (/spinning/evidence-665-f1/boot_weg2_fnFL2x1{51,58}_*.D.log, Fixture unter
#: test/registered/unit/weg2/fixtures/d_h39_h50/). D-TP0: 'weights + runtime
#: state' 17.744 GiB (x150 vor H39: 21.996) bei 94 Pufferzeilen -> fest 7264
#: statt 11618 MiB; die 4354 MiB sind dieselben, um die die Gewichts-Tags im
#: WEG2-DC-BREAKDOWN schrumpfen (20720 -> 16366 MiB ohne weights_draft).
#: TP1/TP2 tragen keine Dense-Marlin-Linears, ihr Posten ist x150 gleich.
D_RESIDENCY_REFERENCE_FNFL2_H39 = DRankReference(
    source="fnFL2x151 + fnFL2x158",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    rank_tp_ratio="1,0,0",
    fixed_mib=(7264.2, 1044.0, 921.9),
    mamba_mib=(393.2, 0.0, 0.0),
    spec_mib=(224.3, 0.0, 0.0),
    activation_mib=(1024.0, 1024.0, 1024.0),
    kv_cell_bytes=(14143, 768, 768),
    draft_host_rank=0,
    draft_vocab_held=False,
    dense_repack_outside_pool=True,
)


# ---------------------------------------------------------------------------
# 4. die Decke je D-Rang
# ---------------------------------------------------------------------------


class DRankResidency(msgspec.Struct, frozen=True, kw_only=True):
    """Ein D-Rang: jeder Term der Budget-Bilanz, die gefahrene Fraction und
    die groesste, die das Budget mit 262k-KV traegt."""

    rank: int
    budget_mib: float
    ratio: float
    span: int
    pad_rows: int
    local_experts: int
    scratch_rows: int
    staging_rows: int
    fraction: float
    resident_rows: int
    #: -1 = die Runtime baut bei dieser Fraction gar keinen Puffer (Fehler
    #: in ``buffer_error``).
    buffer_rows: int
    buffer_error: str
    n_layers: int
    slot_mib: float
    expert_mib: float
    fixed_mib: float
    draft_vocab_delta_mib: float
    mamba_mib: float
    spec_mib: float
    activation_mib: float
    kv_tokens: int
    kv_cell_bytes: int
    kv_mib: float
    ceiling_fraction: Optional[float]
    ceiling_max_rows: int

    @property
    def pre_kv_rest_mib(self) -> float:
        """``rest`` der Runtime VOR dem KV-Pool; <= 0 ist ihr ValueError."""
        return self.budget_mib - (
            self.expert_mib
            + self.fixed_mib
            + self.draft_vocab_delta_mib
            + self.mamba_mib
            + self.spec_mib
            + self.activation_mib
        )

    @property
    def kv_rest_mib(self) -> float:
        return self.pre_kv_rest_mib - self.kv_mib

    def kv_tokens_reachable(self, page_size: int = 64) -> int:
        if self.pre_kv_rest_mib <= 0.0 or self.kv_cell_bytes <= 0:
            return 0
        tokens = int(self.pre_kv_rest_mib * MIB) // int(self.kv_cell_bytes)
        return (tokens // int(page_size)) * int(page_size)

    @property
    def verdict(self) -> str:
        if self.buffer_rows < 0:
            return "RUNTIME VERWEIGERT DEN PUFFER"
        if self.pre_kv_rest_mib <= 0.0:
            return "STIRBT AM KV-POOL"
        if self.kv_rest_mib < 0.0:
            return "262K VERFEHLT"
        return "PASST"

    @property
    def refused(self) -> bool:
        return self.verdict != "PASST"


def solve_d_rank_residency(
    *,
    budgets_mib: Sequence[float],
    fractions: Sequence[float],
    ratios: Sequence[float],
    scratch_rows: Sequence[int],
    staging_rows: int,
    num_experts: int,
    pad_rows: int,
    n_layers: int,
    slot_bytes: float,
    reference: DRankReference,
    vocab_mib: float,
    share_embed: bool,
    kv_tokens: int,
) -> Tuple[DRankResidency, ...]:
    """Je D-Rang die Bilanz gegen SEIN Budget und die Decke.

    Bedingung je Rang r (MiB)::

        L x slot x buffer_rows(E_r, f_r, S_r) + fixed_r + vocab_delta_r
            + mamba_r + spec_r + activation_r + kv_tokens x cell_r
            <= budget_r

    ``vocab_delta_r`` ist auf dem Draft-Host ``-vocab`` wenn die Referenz
    die eigene Tabelle hielt und dieser Boot sie teilt, ``+vocab`` im
    umgekehrten Fall, sonst 0.
    """
    n = len(budgets_mib)
    for name, vec in (
        ("fractions", fractions),
        ("ratios", ratios),
        ("scratch", scratch_rows),
        ("reference.fixed_mib", reference.fixed_mib),
    ):
        if len(vec) != n:
            raise ValueError(
                f"solve_d_rank_residency: {n} Budgets, aber {len(vec)} {name}"
            )
    spans = expert_span_by_rank(num_experts=num_experts, ratios=ratios)
    slot_mib = float(slot_bytes) / MIB
    layer_row_mib = float(n_layers) * slot_mib
    out: List[DRankResidency] = []
    for r in range(n):
        E = spans[r] + int(pad_rows)
        f = float(fractions[r])
        S = int(scratch_rows[r])
        error = ""
        try:
            rows = buffer_rows(local_experts=E, fraction=f, scratch_rows=S)
        except ValueError as exc:
            rows, error = -1, str(exc)
        delta = 0.0
        if r == reference.draft_host_rank and reference.draft_vocab_held != (
            not share_embed
        ):
            delta = (
                -float(vocab_mib) if reference.draft_vocab_held else float(vocab_mib)
            )
        kv_mib = float(kv_tokens) * float(reference.kv_cell_bytes[r]) / MIB
        posts = (
            reference.fixed_mib[r]
            + delta
            + reference.mamba_mib[r]
            + reference.spec_mib[r]
            + reference.activation_mib[r]
            + kv_mib
        )
        max_rows = int(math.floor((float(budgets_mib[r]) - posts) / layer_row_mib))
        out.append(
            DRankResidency(
                rank=r,
                budget_mib=float(budgets_mib[r]),
                ratio=float(ratios[r]),
                span=spans[r],
                pad_rows=int(pad_rows),
                local_experts=E,
                scratch_rows=S,
                staging_rows=int(staging_rows),
                fraction=f,
                resident_rows=resident_rows(E, f) if f < 1.0 else E,
                buffer_rows=rows,
                buffer_error=error,
                n_layers=int(n_layers),
                slot_mib=slot_mib,
                expert_mib=max(rows, 0) * layer_row_mib,
                fixed_mib=float(reference.fixed_mib[r]),
                draft_vocab_delta_mib=delta,
                mamba_mib=float(reference.mamba_mib[r]),
                spec_mib=float(reference.spec_mib[r]),
                activation_mib=float(reference.activation_mib[r]),
                kv_tokens=int(kv_tokens),
                kv_cell_bytes=int(reference.kv_cell_bytes[r]),
                kv_mib=kv_mib,
                ceiling_fraction=largest_fraction_for_rows(
                    local_experts=E, scratch_rows=S, max_rows=max_rows
                ),
                ceiling_max_rows=max_rows,
            )
        )
    return tuple(out)


def describe_rank(fit: DRankResidency) -> str:
    """Eine Logzeile je Rang mit ALLEN Termen."""
    ceiling = "KEINE" if fit.ceiling_fraction is None else "%.3f" % fit.ceiling_fraction
    return (
        "rang%d: Ratio %g -> Spanne %d + Pad %d = E %d, Scratch %d (Staging %d liegt "
        "darin, kein eigener Posten), f %.3f -> R %d, Puffer min(R+S,E) = %s Zeilen x %d Layer x "
        "%.3f MiB = %.0f MiB | fest %.0f%s + mamba %.0f + spec %.0f + Aktivierung "
        "%.0f + KV %d Token x %d B = %.0f MiB | Budget %.0f -> Rest vor KV %.0f, nach "
        "KV %.0f MiB (%d Token erreichbar) -> %s | DECKE f %s (<= %d Zeilen)"
        % (
            fit.rank,
            fit.ratio,
            fit.span,
            fit.pad_rows,
            fit.local_experts,
            fit.scratch_rows,
            fit.staging_rows,
            fit.fraction,
            fit.resident_rows,
            fit.buffer_rows if fit.buffer_rows >= 0 else "KEIN(%s)" % fit.buffer_error,
            fit.n_layers,
            fit.slot_mib,
            fit.expert_mib,
            fit.fixed_mib,
            (
                (" %+.0f Draft-Vokabular" % fit.draft_vocab_delta_mib)
                if fit.draft_vocab_delta_mib
                else ""
            ),
            fit.mamba_mib,
            fit.spec_mib,
            fit.activation_mib,
            fit.kv_tokens,
            fit.kv_cell_bytes,
            fit.kv_mib,
            fit.budget_mib,
            fit.pre_kv_rest_mib,
            fit.kv_rest_mib,
            fit.kv_tokens_reachable(),
            fit.verdict,
            ceiling,
            fit.ceiling_max_rows,
        )
    )


def refusal_text(fits: Sequence[DRankResidency], *, label: str) -> Optional[str]:
    """Der W122-Satz, wenn mindestens ein Rang nicht passt; sonst ``None``."""
    bad = [f for f in fits if f.refused]
    if not bad:
        return None
    return (
        "%s (%s): die gefahrene Experten-Fraction passt auf %s nicht ins Budget "
        "mit %d Token KV -- %s. Die Runtime stirbt daran (KV-Pool-ValueError "
        "'budget leaves no GPU memory for the KV cache', fnFL2x98/x99) bzw. "
        "serviert weniger Kontext als Pflicht (fnFL2x100: 117376 Token). "
        "Groesste tragbare Fraction je Rang: %s."
        % (
            REFUSAL_CODE,
            label,
            ["rang%d" % f.rank for f in bad],
            bad[0].kv_tokens,
            "; ".join(
                "rang%d f %.3f -> %s, %.0f MiB zuviel"
                % (f.rank, f.fraction, f.verdict, max(0.0, -f.kv_rest_mib))
                for f in bad
            ),
            ",".join(
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ),
        )
    )


def solve_stage_fraction_by_buffer_rule(
    *,
    budgets_mib: Sequence[float],
    stage_layers: Sequence[int],
    dense_layer_mib: float,
    slot_mib: float,
    num_experts: int,
    scratch_rows: Sequence[float],
    reserve_mib_by_stage: Optional[Sequence[float]] = None,
) -> List[float]:
    """P-Seite, dieselbe Regel: je PP-Stufe haelt jeder Layer ALLE Experten
    (E = num_experts, kein Pad), Puffer ``min(R + S, E)``. Rueckgabe: je Stufe
    die groesste Fraction; 0.0 heisst "nicht einmal ein Experte plus Scratch"."""
    n = len(stage_layers)
    res = list(reserve_mib_by_stage) if reserve_mib_by_stage is not None else [0.0] * n
    if len(budgets_mib) != n or len(scratch_rows) != n or len(res) != n:
        raise ValueError(
            f"solve_stage_fraction_by_buffer_rule: {n} Stufen, aber {len(budgets_mib)} "
            f"Budgets / {len(scratch_rows)} Scratch / {len(res)} Reserven"
        )
    out: List[float] = []
    for b, L, s, rsv in zip(budgets_mib, stage_layers, scratch_rows, res):
        L = max(1, int(L))
        frei = float(b) - float(rsv) - L * float(dense_layer_mib)
        max_rows = int(math.floor(frei / (L * float(slot_mib)))) if slot_mib > 0 else 0
        f = largest_fraction_for_rows(
            local_experts=int(num_experts), scratch_rows=int(s), max_rows=max_rows
        )
        out.append(0.0 if f is None else f)
    return out


# ---------------------------------------------------------------------------
# 4b. die KARTE: der Posten ausserhalb des Budgets (fnFL2 H33)
# ---------------------------------------------------------------------------
#
# Die Bilanz oben prueft das BUDGET. Mit ``--max-total-tokens 262144`` nimmt
# der KV-Pool nur 262144 x Zelle, der Rest des Budgets bleibt liegen -- und die
# KARTE fuellt, was in keinem Budget-Posten steht: die freien Bloecke der
# privaten Pools (Graph-/Tag-Pools, fuer ``empty_cache`` unerreichbar) und die
# Transiente des schwersten Forwards. Auf der 5090 (D-TP0) hat der Dry-Run
# deshalb FR_D[0] 0.10/0.15 und SCRATCH_D[0] 86 durchgelassen, und x128 starb
# mit SCRATCH 86 an OOM (H30 R1). Hier steht dieser Posten GEMESSEN im Ledger
# (``graph_pool_ledger``): je Rang der Kopfraum am bindenden Messpunkt eines
# Referenz-Boots, ``cap - peak - privat_frei``. Ein Boot mit einem anderen
# Puffer verschiebt ihn um genau die Pufferbytes (x128 gegen x141: peak
# +460 MiB bei +4 Zeilen x 48 Layer = 464 MiB); darunter liegt die near-OOM-
# Grenze des Korridorgesetzes. Keine Reserve: jeder Term ist eine Messung.

#: Der W-Code. W123..W125 liegen auf dem Layout-Switch-Zweig, W126..W128 sind
#: auf dieser Linie vergeben (H14, Draft-auf-P, Draft-Park); W129 bleibt frei
#: fuer die parallel laufenden Riegel.
CARD_REFUSAL_CODE = "W130 Weg2DCardNearOom"


class DCardReference(msgspec.Struct, frozen=True, kw_only=True):
    """Was die Karte eines D-Rangs am bindenden Messpunkt eines Referenz-Boots
    uebrig liess, auf einen LEEREN Experten-Puffer normiert. Je Rang das
    Minimum ueber die Boots (konservativ, wie H8 das Maximum der Posten)."""

    source: str
    model: str
    rank_tp_ratio: str
    #: ``headroom + Puffer x Layer x Zeile`` des bindenden Messpunkts, MiB.
    headroom0_mib: Tuple[float, ...]
    #: ``card_free + Pufferbytes`` im Decode, MiB (``None`` = kein Decode-Punkt).
    free_decode0_mib: Tuple[Optional[float], ...]
    #: Die Terme des bindenden Punkts, nur fuer den Druck.
    buffer_rows: Tuple[int, ...]
    cap_mib: Tuple[float, ...]
    peak_mib: Tuple[float, ...]
    private_free_mib: Tuple[float, ...]
    phase: Tuple[str, ...]
    precision_mib: Tuple[float, ...]
    draft_host_rank: int
    draft_vocab_held: bool
    #: H50: lief die Referenz im H39-Zustand (siehe DRankReference)?
    dense_repack_outside_pool: bool = False
    #: H64: die Verify-Form der Referenz-Boots (``None`` = rekurrent, sonst
    #: die Ringlaenge L des ReplaySSM-Spec-Rings), aus der Wirkung im Log.
    replayssm_spec_ring_len: Optional[int] = None


def d_card_reference_from_logs(
    boots: Sequence[Tuple[str, str]],
    *,
    n_ranks: int,
    n_layers: int,
    slot_bytes: float,
    model: str,
    rank_tp_ratio: str,
) -> DCardReference:
    """Die Karten-Referenz aus D-Logs MESSEN (``boots`` = (Name, Text)).

    Je Boot und Rang: der Messpunkt mit dem kleinsten Kopfraum
    (``graph_pool_ledger.binding_sample``) und der Puffer desselben Logs
    (``MoE expert-offload active on layer 47``). Fehlt einem Rang einer der
    beiden in ALLEN Boots, wird verweigert, statt eine Null einzusetzen -- ein
    fehlender privater Term ist der ganze Posten.
    """
    from sglang.srt.planner import graph_pool_ledger as gpl

    layer_mib = float(n_layers) * float(slot_bytes) / MIB
    h39 = _boots_dense_repack_state(boots)
    ring = _boots_replayssm_spec_state(boots)
    best: Dict[int, Tuple[float, object, int]] = {}
    dec: Dict[int, float] = {}
    host = -1
    vocab_held = False
    for _name, text in boots:
        obs = _observe_boot(text, n_layers=n_layers)
        samples = gpl.samples_from_log(text)
        for r, ss in samples.items():
            if r not in obs["buffer"]:
                continue
            rows = int(obs["buffer"][r])
            s = gpl.binding_sample(ss)
            if s is None:
                continue
            h0 = s.headroom_mib + rows * layer_mib
            if r not in best or h0 < best[r][0]:
                best[r] = (h0, s, rows)
            d = gpl.decode_sample(ss)
            if d is not None:
                f0 = d.card_free_mib + rows * layer_mib
                dec[r] = min(dec.get(r, f0), f0)
        for r, has in obs["draft_experts"].items():
            if has:
                host = r
                vocab_held = vocab_held or bool(obs["draft_vocab"].get(r, 0.0))
    missing = [r for r in range(n_ranks) if r not in best]
    if missing:
        raise ValueError(
            "D-Karten-Referenz unvollstaendig in %s: Rang %s ohne gemessenen "
            "privaten Term (WEG2-GRAPH-POOL bzw. [vram-peak] + #1027) oder ohne "
            "Pufferzeile" % ([b[0] for b in boots], missing)
        )
    return DCardReference(
        source=" + ".join(b[0] for b in boots),
        model=model,
        rank_tp_ratio=rank_tp_ratio,
        headroom0_mib=tuple(round(best[r][0], 1) for r in range(n_ranks)),
        free_decode0_mib=tuple(
            (round(dec[r], 1) if r in dec else None) for r in range(n_ranks)
        ),
        buffer_rows=tuple(best[r][2] for r in range(n_ranks)),
        cap_mib=tuple(round(best[r][1].cap_mib, 1) for r in range(n_ranks)),
        peak_mib=tuple(round(best[r][1].peak_mib, 1) for r in range(n_ranks)),
        private_free_mib=tuple(
            round(best[r][1].private_free_mib, 1) for r in range(n_ranks)
        ),
        phase=tuple(best[r][1].phase for r in range(n_ranks)),
        precision_mib=tuple(
            round(best[r][1].precision_mib, 1) for r in range(n_ranks)
        ),
        draft_host_rank=host,
        draft_vocab_held=vocab_held,
        dense_repack_outside_pool=h39,
        replayssm_spec_ring_len=ring,
    )


#: Die gemessene Karten-Referenz der Next-Flash-Form-A-D-Gruppe, hergeleitet
#: von :func:`d_card_reference_from_logs` aus fnFL2x141 (c9ea5d5d24) und
#: fnFL2x144 (cc660d8786), beide FR_D 0.06,0.44,0.365 / SCRATCH_D 82,48,48,
#: FR_P 0.26,0.45,0.39 (/spinning/evidence-665-f1/boot_weg2_fnFL2x14{1,4}_*.D.log;
#: dieselben Zeilen liegen als Fixture unter
#: test/registered/unit/weg2/fixtures/d_card_h33/). Bindend ist auf jedem Rang
#: der ``[vram-peak] decode``-Punkt; TP0 aus x144 (608 MiB Kopfraum, x141 618).
#: Der Test ``test_the_shipped_card_reference_is_the_logs_own_measurement``
#: bindet sie an die Logs. Auffrischen: ``--d-card-reference-logs``.
D_CARD_REFERENCE_FNFL2 = DCardReference(
    source="fnFL2x141 + fnFL2x144",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    rank_tp_ratio="1,0,0",
    headroom0_mib=(11513.8, 16705.9, 16661.4),
    free_decode0_mib=(11970.6, 17417.7, 17339.1),
    buffer_rows=(94, 112, 113),
    cap_mib=(29388.8, 18780.2, 18780.2),
    peak_mib=(23787.5, 14428.2, 14438.4),
    private_free_mib=(4993.1, 640.1, 790.3),
    phase=("decode", "decode", "decode"),
    precision_mib=(15.4, 15.4, 15.4),
    draft_host_rank=0,
    draft_vocab_held=False,
    dense_repack_outside_pool=False,
)

#: fnFL2 H50: die Karten-Referenz im H39-Zustand, von
#: :func:`d_card_reference_from_logs` aus fnFL2x151 + fnFL2x158 (Fixture
#: ``d_h39_h50``). Bindend ist auf jedem Rang ``WEG2-GRAPH-POOL phase=decode``;
#: D-TP0 aus x158: cap 29369 - peak 23800 - privat_frei 625 = 4944 MiB
#: Kopfraum bei 94 Zeilen (x151 4969; x150 vor H39: privat_frei 4993,
#: Kopfraum 618). Die 4368 MiB weniger privat_frei sind die toten Bloecke,
#: die H39 aus den Tag-Pools genommen hat (Pool 0.2 'weights' 2494 -> 1258).
#: Decode-frei (Befund, kein Stopper) ist das Minimum: x151 hielt im Decode
#: 3485 MiB allgemeinen Allokator-Cache (card_free 2429), x158 488 (5399).
D_CARD_REFERENCE_FNFL2_H39 = DCardReference(
    source="fnFL2x151 + fnFL2x158",
    model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    rank_tp_ratio="1,0,0",
    headroom0_mib=(15849.7, 16701.2, 16873.4),
    free_decode0_mib=(13334.7, 16114.2, 15926.4),
    buffer_rows=(94, 122, 133),
    cap_mib=(29369.0, 18782.0, 18802.0),
    peak_mib=(23800.0, 15551.0, 16788.0),
    private_free_mib=(625.0, 684.0, 571.0),
    phase=("decode", "decode", "decode"),
    precision_mib=(3.0, 3.0, 3.0),
    draft_host_rank=0,
    draft_vocab_held=False,
    dense_repack_outside_pool=True,
)

#: Die eingebauten Referenzen je Baum-Zustand (H50): gewaehlt wird nach
#: SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL der D-Gruppe, nie nach "neuester".
D_RESIDENCY_REFERENCES: Tuple[DRankReference, ...] = (
    D_RESIDENCY_REFERENCE_FNFL2,
    D_RESIDENCY_REFERENCE_FNFL2_H39,
)
D_CARD_REFERENCES: Tuple[DCardReference, ...] = (
    D_CARD_REFERENCE_FNFL2,
    D_CARD_REFERENCE_FNFL2_H39,
)


def h39_state_text(state: bool) -> str:
    return (
        "H39 an: Dense-Repack ausserhalb der Tag-Pools"
        if state
        else "H39 aus: Checkpoint-Tensoren und Repack in den Tag-Pools"
    )


class DCardFit(msgspec.Struct, frozen=True, kw_only=True):
    """Ein D-Rang auf seiner KARTE: Kopfraum nach dem schwersten Forward und
    Decode-frei, beide aus der Referenz um die Pufferbytes verschoben."""

    rank: int
    fraction: float
    scratch_rows: int
    local_experts: int
    buffer_rows: int
    ref_buffer_rows: int
    layer_row_mib: float
    expert_delta_mib: float
    vocab_delta_mib: float
    headroom_mib: float
    free_decode_mib: Optional[float]
    near_oom_mib: float
    band_floor_mib: float
    ceiling_fraction: Optional[float]
    ceiling_max_rows: int

    @property
    def verdict(self) -> str:
        if self.buffer_rows < 0:
            return "KEIN PUFFER (W122)"
        if self.headroom_mib < self.near_oom_mib:
            return "STIRBT AN DER KARTE"
        if self.free_decode_mib is not None and self.free_decode_mib < self.band_floor_mib:
            return "PASST, KORRIDOR GERISSEN (Befund, kein Stopper)"
        return "PASST"

    @property
    def refused(self) -> bool:
        return self.verdict == "STIRBT AN DER KARTE"


def solve_d_card(
    *,
    fits: Sequence[DRankResidency],
    reference: DCardReference,
    vocab_mib: float,
    share_embed: bool,
    near_oom_mib: float,
    band_floor_mib: float,
) -> Tuple[DCardFit, ...]:
    """Je D-Rang die Karten-Bilanz::

        headroom_r = headroom0_r - L x slot x buffer_rows_r - vocab_delta_r
                     >= near_oom        (corridor_guard.NEAR_OOM_MIB)

    ``vocab_delta_r`` gilt auf dem Draft-Host: ``+vocab`` wenn dieser Boot
    die eigene Draft-Tabelle haelt und die Referenz nicht, ``-vocab`` im
    umgekehrten Fall. ``free_decode`` ist die Karte im Decode (Befund gegen
    den Band-Floor, kein Stopper).
    """
    n = len(fits)
    if len(reference.headroom0_mib) != n:
        raise ValueError(
            f"solve_d_card: {n} Raenge, aber die Karten-Referenz "
            f"({reference.source}) hat {len(reference.headroom0_mib)}"
        )
    out: List[DCardFit] = []
    for fit in fits:
        r = fit.rank
        layer_mib = float(fit.n_layers) * float(fit.slot_mib)
        delta = 0.0
        if r == reference.draft_host_rank:
            holds = not share_embed
            if holds and not reference.draft_vocab_held:
                delta = float(vocab_mib)
            elif not holds and reference.draft_vocab_held:
                delta = -float(vocab_mib)
        rows = int(fit.buffer_rows)
        expert = max(rows, 0) * layer_mib
        head = float(reference.headroom0_mib[r]) - expert - delta
        f0 = reference.free_decode0_mib[r]
        free_dec = None if f0 is None else float(f0) - expert - delta
        max_rows = int(
            math.floor(
                (float(reference.headroom0_mib[r]) - delta - float(near_oom_mib))
                / layer_mib
            )
        )
        out.append(
            DCardFit(
                rank=r,
                fraction=float(fit.fraction),
                scratch_rows=int(fit.scratch_rows),
                local_experts=int(fit.local_experts),
                buffer_rows=rows,
                ref_buffer_rows=int(reference.buffer_rows[r]),
                layer_row_mib=layer_mib,
                expert_delta_mib=(rows - int(reference.buffer_rows[r])) * layer_mib,
                vocab_delta_mib=delta,
                headroom_mib=head,
                free_decode_mib=free_dec,
                near_oom_mib=float(near_oom_mib),
                band_floor_mib=float(band_floor_mib),
                ceiling_fraction=largest_fraction_for_rows(
                    local_experts=int(fit.local_experts),
                    scratch_rows=int(fit.scratch_rows),
                    max_rows=max_rows,
                ),
                ceiling_max_rows=max_rows,
            )
        )
    return tuple(out)


def describe_card(card: DCardFit, reference: DCardReference) -> str:
    r = card.rank
    edge = scratch_edge(
        local_experts=card.local_experts,
        fraction=card.fraction,
        max_rows=card.ceiling_max_rows,
    )
    ceiling = "KEINE" if card.ceiling_fraction is None else "%.3f" % card.ceiling_fraction
    free_dec = "n/a" if card.free_decode_mib is None else "%.0f" % card.free_decode_mib
    return (
        "rang%d: Referenz %d Zeilen, Kopfraum %.0f = cap %.0f - peak %.0f - "
        "privat_frei %.0f MiB am Punkt '%s' (+-%.0f); hier f %.3f S %d -> %s "
        "Zeilen, Puffer %+.0f MiB%s -> Kopfraum %.0f MiB (near-OOM %.0f), Decode "
        "frei %s MiB (Band-Floor %.0f) -> %s | KARTEN-DECKE f %s (<= %d Zeilen) | "
        "KARTEN-KANTE bei f %.3f: <= %d Zeilen = SCRATCH <= %s"
        % (
            r,
            reference.buffer_rows[r],
            reference.headroom0_mib[r] - reference.buffer_rows[r] * card.layer_row_mib,
            reference.cap_mib[r],
            reference.peak_mib[r],
            reference.private_free_mib[r],
            reference.phase[r],
            reference.precision_mib[r],
            card.fraction,
            card.scratch_rows,
            card.buffer_rows if card.buffer_rows >= 0 else "KEIN",
            card.expert_delta_mib,
            (" %+.0f Draft-Vokabular" % card.vocab_delta_mib)
            if card.vocab_delta_mib
            else "",
            card.headroom_mib,
            card.near_oom_mib,
            free_dec,
            card.band_floor_mib,
            card.verdict,
            ceiling,
            card.ceiling_max_rows,
            card.fraction,
            edge[0],
            "KEINE" if edge[1] is None else edge[1],
        )
    )


def card_refusal_text(
    cards: Sequence[DCardFit], reference: DCardReference, *, label: str
) -> Optional[str]:
    """Der W130-Satz, wenn mindestens ein Rang auf seiner Karte stirbt."""
    bad = [c for c in cards if c.refused]
    if not bad:
        return None
    return (
        "%s (%s): der Experten-Puffer passt ins Budget, aber nicht auf die Karte "
        "von %s -- %s. Der Posten ausserhalb des Budgets (freie Bloecke privater "
        "Graph-/Tag-Pools, fuer empty_cache unerreichbar, plus die Transiente des "
        "schwersten Forwards) ist gemessen in %s; fnFL2x128 (SCRATCH_D 82->86 auf "
        "der 5090) starb genau daran (OOM im Extend, 74.81 MiB frei, 4.75 GiB in "
        "privaten Pools). Groesste tragbare Fraction je Rang auf der Karte: %s."
        % (
            CARD_REFUSAL_CODE,
            label,
            ["rang%d" % c.rank for c in bad],
            "; ".join(
                "rang%d f %.3f S %d -> %d Zeilen, Kopfraum %.0f MiB < near-OOM %.0f"
                % (
                    c.rank,
                    c.fraction,
                    c.scratch_rows,
                    c.buffer_rows,
                    c.headroom_mib,
                    c.near_oom_mib,
                )
                for c in bad
            ),
            reference.source,
            ",".join(
                "KEINE" if c.ceiling_fraction is None else "%.3f" % c.ceiling_fraction
                for c in cards
            ),
        )
    )


# ---------------------------------------------------------------------------
# 5. die Launcher-Naht: alles, was der D-FRACTION-SOLVE liest, an EINER Stelle
# ---------------------------------------------------------------------------


class DResidencyPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Was der Launcher druckt (``lines``) und ob er verweigert (``refusal``,
    der W122-Satz und/oder der W130-Satz; ``None`` = der Boot passt oder die
    Rechnung entfaellt)."""

    lines: Tuple[str, ...]
    refusal: Optional[str]
    fits: Tuple[DRankResidency, ...] = ()
    #: H33: die Karten-Bilanz je Rang (leer = sie entfiel, Grund in ``lines``).
    card_fits: Tuple["DCardFit", ...] = ()


# ---------------------------------------------------------------------------
# 6. (H64) die Verify-Form der D-Gruppe: rekurrent oder ReplaySSM-Spec-Ring
# ---------------------------------------------------------------------------

#: H64: die Zeile, die ein D-Rang mit dem ReplaySSM-Spec-Ring beim Bau des
#: Mamba-Pools einmal druckt (``MambaPool``, 27B ReplaySSM S2). Wie bei H39 ist
#: die Form eines Referenz-Logs die WIRKUNG, nie ein Flag der Argumentzeile.
REPLAYSSM_SPEC_RING_MARK = "GDN ReplaySSM SPEC ring allocated"
_RX_SPEC_RING = re.compile(re.escape(REPLAYSSM_SPEC_RING_MARK) + r" \(L=(\d+),")

#: dtype-Namen der Konfiguration/Argumentzeile -> Bytes je Element.
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def boot_replayssm_spec_ring_len(text: str) -> Optional[int]:
    """Die Ringlaenge L eines D-Logs mit Spec-Ring, ``None`` = rekurrenter
    Verify (per-Draft-Zwischenzustaende)."""
    m = _RX_SPEC_RING.search(text)
    return int(m.group(1)) if m else None


def _boots_replayssm_spec_state(boots: Sequence[Tuple[str, str]]) -> Optional[int]:
    """Die Verify-Form EINER Referenz. Boots beider Formen (oder zweier
    Ringlaengen) zu mischen hiesse, das Maximum zweier Allokationen zu nehmen,
    die sich auf D-TP0 der Next-Flash-Form um ~0,4 GiB unterscheiden -- das
    wird verweigert, nicht gemittelt (wie H39)."""
    states = {name: boot_replayssm_spec_ring_len(text) for name, text in boots}
    if len(set(states.values())) > 1:
        raise ValueError(
            "Referenz mischt D-Boots verschiedener Verify-Form ('%s', L): %s"
            % (REPLAYSSM_SPEC_RING_MARK, states)
        )
    return next(iter(states.values()), None)


def spec_form_text(ring_len: Optional[int]) -> str:
    return (
        "rekurrentem Verify (Zwischenzustand je Draft-Schritt)"
        if ring_len is None
        else "ReplaySSM-Spec-Ring L=%d" % int(ring_len)
    )


class ReplaySSMSpecForm(msgspec.Struct, frozen=True, kw_only=True):
    """H64: was an der Verify-Form der D-Gruppe den Posten 'speculative
    intermediate state' und die Verify-Allokation bestimmt; gebaut vom
    Launcher (``d_replayssm_spec_plan_form``), nur unter --d-replayssm-spec on.
    """

    #: --linear-replayssm-cache-len unter --enable-linear-replayssm-spec;
    #: ``None`` = rekurrenter Verify.
    ring_len: Optional[int]
    #: Das breiteste Verify-Fenster D (NEXTN: --speculative-num-draft-tokens,
    #: wie D es faehrt; DFLASH: der Block).
    draft_tokens: int
    #: D's --max-running-requests. Der Posten zaehlt so viele Zeilen
    #: (``capped_reqs``), die Allokation eine Padding-Zeile mehr
    #: (``spec_state_size + 1``).
    max_running: int
    #: D's --mamba-ssm-dtype; ``None`` = das des Checkpoints.
    ssm_dtype: Optional[str] = None


class GdnSpecUnitBytes(msgspec.Struct, frozen=True, kw_only=True):
    """Bytes je GDN-Einheit (1 k-Kopf + r v-Koepfe), Schicht und Request-Zeile:
    die Laufzeitformeln aus ``configs/mamba_utils.py`` (``mamba_cache_per_req``,
    ``spec_ring_workspace_bytes_per_req``, ``replayssm_ring_bytes_per_req``) je
    Einheit, per Test an sie gebunden. Alles skaliert linear mit den Einheiten
    eines Rangs (uneven GDN-TP teilt in ganzen Einheiten), deshalb sind die
    Verhaeltnisse unten rang- und schichtfrei."""

    per_req: float
    #: EIN SSM-Zustand (= ein per-Draft-Zwischenzustand des rekurrenten Verify).
    ssm: float
    #: Die Conv-Verify-Fenster; sie bleiben in BEIDEN Formen allokiert.
    conv_window: float
    #: Der Ring der Laenge L (d, k, ihre Low-Parts bei 16 Bit, g in fp32);
    #: 0 fuer den rekurrenten Verify.
    ring: float


def gdn_spec_unit_bytes(
    text_cfg: Mapping[str, object],
    *,
    draft_tokens: int,
    ring_len: Optional[int],
    ssm_dtype: Optional[str] = None,
    act_dtype: Optional[str] = None,
) -> GdnSpecUnitBytes:
    """Die Einheits-Bytes aus der Checkpoint-Geometrie (``text_config``).

    Aktivierungs-dtype (Conv-Zustand, Ring-Records) = das Modell-dtype wie in
    ``mamba2_state_dtype`` (ohne SGLANG_MAMBA_CONV_DTYPE); SSM-dtype = D's
    --mamba-ssm-dtype, sonst das des Checkpoints, sonst float32.
    """
    k = int(text_cfg["linear_key_head_dim"])
    v = int(text_cfg["linear_value_head_dim"])
    hv = int(text_cfg["linear_num_value_heads"])
    h = int(text_cfg["linear_num_key_heads"])
    if h <= 0 or hv % h:
        raise ValueError(
            "GDN-Geometrie: %d v-Koepfe sind kein Vielfaches von %d k-Koepfen" % (hv, h)
        )
    r = hv // h
    kc = int(text_cfg["linear_conv_kernel_dim"])
    act_name = str(
        act_dtype or text_cfg.get("dtype") or text_cfg.get("torch_dtype") or "bfloat16"
    )
    ssm_name = str(ssm_dtype or text_cfg.get("mamba_ssm_dtype") or "float32")
    act = _DTYPE_BYTES[act_name]
    ssm_b = _DTYPE_BYTES[ssm_name]
    conv_dim = 2 * k + r * v
    ssm = float(r * v * k * ssm_b)
    ring = 0.0
    if ring_len is not None:
        L = int(ring_len)
        ring = float(r * L * v * act + L * k * act + r * L * 4)
        if act != 4:
            ring += float(r * L * v * act + L * k * act)
    return GdnSpecUnitBytes(
        per_req=float(conv_dim * (kc - 1) * act) + ssm,
        ssm=ssm,
        conv_window=float(conv_dim * (int(draft_tokens) + kc - 2) * act),
        ring=ring,
    )


def _spec_post_and_alloc(
    per_req_mib: float,
    *,
    ring_len: Optional[int],
    form: ReplaySSMSpecForm,
    unit_of,
) -> Tuple[float, float]:
    """(Posten, Verify-Allokation ohne Conv-Fenster) eines Rangs in MiB, fuer
    die Form ``ring_len``: der Posten wie ``handle_max_mamba_cache`` ihn bucht
    (``per_req x capped x D`` bzw. ``capped x Ring-Werkraum``), die Allokation
    wie ``MambaPool`` sie baut (``spec_state_size + 1`` Zeilen)."""
    u = unit_of(ring_len)
    cap = int(form.max_running)
    rows = cap + 1
    d = int(form.draft_tokens)
    if ring_len is None:
        return (
            per_req_mib * cap * d,
            per_req_mib * rows * d * u.ssm / u.per_req,
        )
    return (
        per_req_mib * cap * (u.conv_window + u.ring) / u.per_req,
        per_req_mib * rows * u.ring / u.per_req,
    )


class ReplaySSMSpecRebook(msgspec.Struct, frozen=True, kw_only=True):
    """H64: je Rang der gemessene Posten der Referenz und was die gefahrene
    Form bucht (Budget) bzw. allokiert (Karte), MiB -- GERECHNET aus dem
    gemessenen Referenz-Posten, nicht gemessen."""

    ref_ring_len: Optional[int]
    ring_len: Optional[int]
    per_req_mib: Tuple[float, ...]
    spec_ref_mib: Tuple[float, ...]
    spec_mib: Tuple[float, ...]
    alloc_ref_mib: Tuple[float, ...]
    alloc_mib: Tuple[float, ...]

    @property
    def freed_mib(self) -> Tuple[float, ...]:
        """Was die Karte je Rang gegenueber der Referenz-Form gewinnt (MiB)."""
        return tuple(
            round(a - b, 1) for a, b in zip(self.alloc_ref_mib, self.alloc_mib)
        )


def replayssm_spec_rebook(
    *,
    spec_mib_ref: Sequence[float],
    ref_ring_len: Optional[int],
    form: ReplaySSMSpecForm,
    text_cfg: Mapping[str, object],
    act_dtype: Optional[str] = None,
) -> ReplaySSMSpecRebook:
    """Den Spec-Posten einer Referenz auf die gefahrene Verify-Form umbuchen.

    Der gemessene Posten traegt je Rang ``per_req x capped x D`` (rekurrent)
    bzw. ``capped x Ring-Werkraum`` (Ring) -- daraus folgt der Zustand je
    Request ``per_req`` des Rangs (0 auf einem Rang ohne GDN-Koepfe, Form-A-
    Worker), und aus ihm beide Formen. Vorausgesetzt wie fuer jeden anderen
    Referenz-Posten: dieselbe D-Form (Fenster, --max-running-requests,
    SSM-dtype) wie der Boot, den die Referenz vertritt.
    """

    def unit_of(ring_len):
        return gdn_spec_unit_bytes(
            text_cfg,
            draft_tokens=form.draft_tokens,
            ring_len=ring_len,
            ssm_dtype=form.ssm_dtype,
            act_dtype=act_dtype,
        )

    cap = int(form.max_running)
    d = int(form.draft_tokens)
    if cap <= 0 or d <= 0:
        raise ValueError(
            "ReplaySSMSpecForm: --max-running-requests %d, Fenster %d" % (cap, d)
        )
    per_req: List[float] = []
    for s in spec_mib_ref:
        if ref_ring_len is None:
            per_req.append(float(s) / (cap * d))
        else:
            u = unit_of(ref_ring_len)
            per_req.append(float(s) / cap * u.per_req / (u.conv_window + u.ring))
    ref = [
        _spec_post_and_alloc(p, ring_len=ref_ring_len, form=form, unit_of=unit_of)
        for p in per_req
    ]
    run = [
        _spec_post_and_alloc(p, ring_len=form.ring_len, form=form, unit_of=unit_of)
        for p in per_req
    ]
    return ReplaySSMSpecRebook(
        ref_ring_len=ref_ring_len,
        ring_len=form.ring_len,
        per_req_mib=tuple(per_req),
        spec_ref_mib=tuple(float(s) for s in spec_mib_ref),
        spec_mib=tuple(round(r[0], 1) for r in run),
        alloc_ref_mib=tuple(round(r[1], 1) for r in ref),
        alloc_mib=tuple(round(r[1], 1) for r in run),
    )


def replayssm_spec_alloc_mib(
    *,
    per_req_mib: Sequence[float],
    ring_len: Optional[int],
    form: ReplaySSMSpecForm,
    text_cfg: Mapping[str, object],
    act_dtype: Optional[str] = None,
) -> Tuple[float, ...]:
    """H64: die Verify-Allokation ohne Conv-Fenster je Rang (MiB) in der Form
    ``ring_len`` -- fuer die Karte, deren Referenz eine andere Form haben kann
    als die Budget-Referenz."""

    def unit_of(rl):
        return gdn_spec_unit_bytes(
            text_cfg,
            draft_tokens=form.draft_tokens,
            ring_len=rl,
            ssm_dtype=form.ssm_dtype,
            act_dtype=act_dtype,
        )

    return tuple(
        round(
            _spec_post_and_alloc(
                float(p), ring_len=ring_len, form=form, unit_of=unit_of
            )[1],
            1,
        )
        for p in per_req_mib
    )


def describe_spec_rebook(
    rb: ReplaySSMSpecRebook, *, source: str, form: ReplaySSMSpecForm
) -> str:
    return (
        "REPLAYSSM-SPEC (H64): D faehrt mit %s (Fenster %d, --max-running-requests "
        "%d, SSM %s), die Referenz %s ist mit %s gemessen -> 'speculative "
        "intermediate state' je Rang [%s] MiB (Budget), Verify-Allokation ohne "
        "Conv-Fenster [%s] MiB (Karte, Kopfraum %s MiB) -- GERECHNET aus dem "
        "gemessenen Referenz-Posten (per_req je Rang %s MiB), nicht gemessen"
        % (
            spec_form_text(rb.ring_len),
            int(form.draft_tokens),
            int(form.max_running),
            form.ssm_dtype or "des Checkpoints",
            source,
            spec_form_text(rb.ref_ring_len),
            ", ".join(
                "%.1f -> %.1f" % (a, b) for a, b in zip(rb.spec_ref_mib, rb.spec_mib)
            ),
            ", ".join(
                "%.1f -> %.1f" % (a, b) for a, b in zip(rb.alloc_ref_mib, rb.alloc_mib)
            ),
            ["%+.1f" % x for x in rb.freed_mib],
            ["%.2f" % p for p in rb.per_req_mib],
        )
    )


# ---------------------------------------------------------------------------
# 7. (H91b) die Sitze der D-Gruppe: bs1 -> bs2 (Stufe 1), spaeter 1..6
# ---------------------------------------------------------------------------


class SeatRebook(msgspec.Struct, frozen=True, kw_only=True):
    """H91b: die sitz-proportionalen Posten einer Referenz, umgebucht auf die
    Sitzzahl des Boots -- GERECHNET aus den gemessenen Posten, nicht gemessen.

    ``mamba``: die Runtime baut ``ceil(Sitze x ratio x 1.25)`` Zustands-Slots
    (``_auto_mamba_demand_size``), der Slot kostet ``mamba_ref / slots_ref``.
    ``spec``: 'speculative intermediate state' ist ``per_req x Sitze x D``
    (rekurrent) bzw. ``Sitze x Ring-Werkraum`` -- in beiden Formen linear in
    den Sitzen."""

    ref_seats: int
    seats: int
    slots_ref: int
    slots: int
    mamba_ref_mib: Tuple[float, ...]
    mamba_mib: Tuple[float, ...]
    spec_ref_mib: Tuple[float, ...]
    spec_mib: Tuple[float, ...]

    @property
    def budget_delta_mib(self) -> Tuple[float, ...]:
        return tuple(
            round((m - m0) + (s - s0), 1)
            for m, m0, s, s0 in zip(
                self.mamba_mib, self.mamba_ref_mib, self.spec_mib, self.spec_ref_mib
            )
        )


def seat_rebook(reference: DRankReference, *, seats: int) -> SeatRebook:
    from sglang.srt.weg2.d_seats import mamba_slots_for_seats

    ref_seats = max(1, int(reference.max_running))
    s = max(1, int(seats))
    slots_ref = mamba_slots_for_seats(ref_seats)
    slots = mamba_slots_for_seats(s)
    return SeatRebook(
        ref_seats=ref_seats,
        seats=s,
        slots_ref=slots_ref,
        slots=slots,
        mamba_ref_mib=tuple(float(m) for m in reference.mamba_mib),
        mamba_mib=tuple(round(float(m) / slots_ref * slots, 1) for m in reference.mamba_mib),
        spec_ref_mib=tuple(float(x) for x in reference.spec_mib),
        spec_mib=tuple(round(float(x) / ref_seats * s, 1) for x in reference.spec_mib),
    )


def describe_seat_rebook(rb: SeatRebook, *, source: str) -> str:
    return (
        "D-SITZE (H91b): D faehrt %d Sitz(e), die Referenz %s ist mit %d gemessen -> "
        "'mamba state pool' %d -> %d Slots, je Rang [%s] MiB; 'speculative intermediate "
        "state' je Rang [%s] MiB; Budget-Delta je Rang %s MiB -- GERECHNET aus den "
        "gemessenen Posten (Slotformel ceil(Sitze x 5 x 1.25), Spec linear in den "
        "Sitzen), nicht gemessen"
        % (
            rb.seats,
            source,
            rb.ref_seats,
            rb.slots_ref,
            rb.slots,
            ", ".join("%.1f -> %.1f" % (a, b) for a, b in zip(rb.mamba_ref_mib, rb.mamba_mib)),
            ", ".join("%.1f -> %.1f" % (a, b) for a, b in zip(rb.spec_ref_mib, rb.spec_mib)),
            ["%+.1f" % d for d in rb.budget_delta_mib],
        )
    )


#: The D group's MoE graph mode; the per-step row bound below holds only for
#: the device-planned pool (``expert_offload.prepare_pool``).
POOL_GRAPH_MODE_ENV = "SGLANG_MOE_OFFLOAD_GRAPH_MODE"

#: H95: SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES (environ.py, EnvInt default 0) --
#: mirrored like DRAFT_SHARE_EMBED because the launcher reads GROUP D's env.
POOL_OVERFLOW_WAVES_ENV = "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES"


def pool_overflow_waves(env: Mapping[str, str]) -> int:
    """H95: the wave cap group D runs with; 1 when off (0, 1 or unset)."""
    raw = str(env.get(POOL_OVERFLOW_WAVES_ENV, "")).strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError('%s="%s" ist keine Zahl' % (POOL_OVERFLOW_WAVES_ENV, raw)) from exc
    return max(1, n)


def pool_step_rows_needed(
    *, seats: int, verify_tokens: int, top_k: int, local_experts: int, resident_rows: int
) -> int:
    """Rows (LRU + staging = scratch) one captured decode step can demand on a
    rank: the Task #40 overflow-impossible bound of ``expert_pool_device.step``
    -- every distinct non-resident expert of the step needs a victim or a
    staging row -- over ``seats x verify_tokens x top_k`` routed ids, capped by
    how many non-resident experts the rank has at all."""
    ids = int(seats) * int(verify_tokens) * int(top_k)
    return int(min(ids, max(int(local_experts) - int(resident_rows), 0)))


def pool_step_waves_needed(*, need_rows: int, scratch_rows: int) -> int:
    """H95: ``ceil(min(ids, E-R) / Scratch)`` -- the overflow waves a captured
    step needs when Scratch (LRU + staging) rows serve one wave."""
    return max(1, -(-int(need_rows) // max(1, int(scratch_rows))))


def pool_step_rows_check(
    fits: Sequence[DRankResidency],
    *,
    seats: Optional[int],
    verify_tokens: Optional[int],
    top_k: Optional[object],
    pool_mode: bool,
    marker: str,
    label: str,
    waves: int = 1,
) -> Tuple[Tuple[str, ...], Optional[str]]:
    """H91b: the per-step row bound at D's seat count, per rank.

    Measured by the Form-A bs2 audit: at bs1 a verify step routes 4 x 10 = 40
    ids, at bs2 80 -- and a worker with Scratch 48 then raises 'Step ids exceed
    the LRU rows plus the staging rows' while the bs2 verify graph is CAPTURED
    (loud, every boot). Refused here with the numbers instead.

    H95: with ``waves`` = SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES >= 2 the bound is
    ``min(ids, E-R) <= waves x Scratch`` -- the scratch no longer grows with
    the seats, the number of waves a graph captures does (per rank, named in
    the line)."""
    if seats is None or not pool_mode:
        return (), None
    if verify_tokens is None or top_k is None:
        return (
            (
                "%s FRACTION-SOLVE %s POOL-SCHRITT (H91b) ENTFAELLT: Verify-Fenster "
                "oder top_k unbekannt (keine Verify-Form uebergeben) -- die "
                "Zeilenschranke je Schritt bei %d Sitz(en) ist NICHT geprueft"
                % (marker, label, int(seats)),
            ),
            None,
        )
    need = [
        pool_step_rows_needed(
            seats=int(seats),
            verify_tokens=int(verify_tokens),
            top_k=int(top_k),
            local_experts=f.local_experts,
            resident_rows=f.resident_rows,
        )
        for f in fits
    ]
    w = max(1, int(waves))
    short = [f.rank for f, n in zip(fits, need) if w * int(f.scratch_rows) < n]
    wave_text = ""
    if w > 1:
        wave_text = (
            " -- H95 Ueberlaufwellen (%s=%d): Schranke min(Ids, E-R) <= %d x Scratch, "
            "Wellen je Rang %s"
            % (
                POOL_OVERFLOW_WAVES_ENV,
                w,
                w,
                [
                    pool_step_waves_needed(need_rows=n, scratch_rows=int(f.scratch_rows))
                    for f, n in zip(fits, need)
                ],
            )
        )
    line = (
        "%s FRACTION-SOLVE %s POOL-SCHRITT (H91b): %d Sitz(e) x %d Verify-Zeilen x "
        "top_k %d = %d Ids je Schritt -> Zeilen (LRU+Staging) Pflicht je Rang %s = "
        "min(Ids, E-R), Scratch gegeben %s%s%s"
        % (
            marker,
            label,
            int(seats),
            int(verify_tokens),
            int(top_k),
            int(seats) * int(verify_tokens) * int(top_k),
            need,
            [int(f.scratch_rows) for f in fits],
            wave_text,
            (" -- ZU KLEIN auf Rang %s" % short) if short else " -- passt",
        )
    )
    if not short:
        return (line,), None
    if w > 1:
        refusal = (
            "W-SITZE D-Pool-Schritt: bei %d Sitz(en) braucht ein Decode-Schritt je Rang "
            "%s Zeilen (LRU+Staging), %d Ueberlaufwellen x SGLANG_MOE_SCRATCH_SLOTS %s "
            "tragen %s -- Rang %s wirft beim Capture des bs%d-Graphen 'Step ids exceed "
            "the LRU rows plus the staging rows'; %s anheben oder Scratch dort anheben"
            % (
                int(seats),
                need,
                w,
                [int(f.scratch_rows) for f in fits],
                [w * int(f.scratch_rows) for f in fits],
                short,
                int(seats),
                POOL_OVERFLOW_WAVES_ENV,
            )
        )
        return (line,), refusal
    refusal = (
        "W-SITZE D-Pool-Schritt: bei %d Sitz(en) braucht ein Decode-Schritt je Rang "
        "%s Zeilen (LRU+Staging), SGLANG_MOE_SCRATCH_SLOTS gibt %s -- Rang %s "
        "wirft beim Capture des bs%d-Graphen 'Step ids exceed the LRU rows plus the "
        "staging rows'; Scratch dort anheben (Fraction senken haelt das Budget) "
        "oder --max-running-requests/--d-bs senken"
        % (
            int(seats),
            need,
            [int(f.scratch_rows) for f in fits],
            short,
            int(seats),
        )
    )
    return (line,), refusal


def _env_true(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in _TRUE


def _reference_for(
    *,
    model_path: str,
    rank_tp_ratio: str,
    reference_logs: str,
    n_ranks: int,
    n_layers: int,
    slot_bytes: float,
    dense_repack: bool = DENSE_REPACK_OUTSIDE_POOL_DEFAULT,
    reference_seats: int = 1,
) -> Tuple[Optional[DRankReference], str]:
    """Die Referenz fuer DIESE Form und DIESEN Baum-Zustand (H50: H39 an/aus),
    oder ``(None, warum nicht)``. ``reference_seats`` gilt nur fuer gegebene
    Logs; die eingebauten Referenzen tragen ihre Sitze selbst (H91b)."""
    import os

    model = os.path.basename(os.path.normpath(model_path))
    paths = [p.strip() for p in str(reference_logs or "").split(",") if p.strip()]
    if paths:
        boots = []
        for p in paths:
            with open(p, errors="replace") as fh:
                boots.append((os.path.basename(p), fh.read()))
        ref = d_rank_reference_from_logs(
            boots,
            n_ranks=n_ranks,
            n_layers=n_layers,
            slot_bytes=slot_bytes,
            model=model,
            rank_tp_ratio=rank_tp_ratio,
            max_running=reference_seats,
        )
        if ref.dense_repack_outside_pool != bool(dense_repack):
            return None, _h39_mismatch_text(
                ref.source, ref.dense_repack_outside_pool, dense_repack,
                "--d-residency-reference-logs",
            )
        return ref, ""
    ref = D_RESIDENCY_REFERENCE_FNFL2
    for cand in D_RESIDENCY_REFERENCES:
        if cand.dense_repack_outside_pool == bool(dense_repack):
            ref = cand
            break
    if (
        ref.model != model
        or ref.rank_tp_ratio != rank_tp_ratio
        or len(ref.fixed_mib) != n_ranks
        or ref.dense_repack_outside_pool != bool(dense_repack)
    ):
        return None, (
            "die eingebaute Referenz (%s) gilt fuer %s mit --rank-tp-ratio %s auf %d "
            "Raengen (%s), dieser Boot faehrt %s mit %s auf %d (%s); den festen "
            "Rang-Posten per --d-residency-reference-logs <D.log,...> aus Boots "
            "DIESER Form messen"
            % (
                ref.source,
                ref.model,
                ref.rank_tp_ratio,
                len(ref.fixed_mib),
                h39_state_text(ref.dense_repack_outside_pool),
                model,
                rank_tp_ratio,
                n_ranks,
                h39_state_text(bool(dense_repack)),
            )
        )
    return ref, ""


def _h39_mismatch_text(source: str, have: bool, want: bool, flag: str) -> str:
    return (
        "die Referenz-Logs %s sind im Zustand '%s' gemessen, die D-Gruppe faehrt "
        "'%s' (%s) -- auf D-TP0 der Next-Flash-Form liegen dazwischen 4354 MiB "
        "'weights + runtime state' und 4368 MiB privat_frei (fnFL2x150 gegen "
        "x158); %s aus Boots DESSELBEN Zustands angeben"
        % (source, h39_state_text(have), h39_state_text(want),
           DENSE_REPACK_OUTSIDE_POOL_ENV, flag)
    )


def plan_d_residency(
    *,
    model_path: str,
    budgets_mib: Sequence[float],
    ratios: Sequence[float],
    fractions: Sequence[float],
    scratch_rows: Sequence[int],
    rank_tp_ratio: str,
    env_d: Mapping[str, str],
    reference_logs: str,
    kv_tokens: int,
    label: str,
    marker: str,
    card_reference_logs: str = "",
    replayssm_spec: Optional[ReplaySSMSpecForm] = None,
    seats: Optional[int] = None,
    seat_graph_mib: Optional[Sequence[float]] = None,
    reference_seats: int = 1,
) -> DResidencyPlan:
    """Der D-FRACTION-SOLVE mit den Metallregeln, fuer ``launcher``.

    H91b: ``seats`` = D's wirksame --max-running-requests. Weicht sie von den
    Sitzen der Referenz ab, werden die sitz-proportionalen Posten umgebucht
    (Budget) und die Karte um die Mehr-Allokation verschoben; ``None`` laesst
    die Rechnung byte-gleich. ``seat_graph_mib`` = gemessene Mehrkosten des
    Decode-CUDA-Graphen je zusaetzlichem Sitz und Rang (Karte, ausserhalb des
    Budgets); fehlt sie, sagt die KARTE-Zeile das mit Namen.

    Liest die Checkpoint-Geometrie (Header, keine Tensoren), die Gruppen-Env
    von D und die gemessene Referenz; rechnet je Rang die Bilanz und gibt die
    Zeilen und, wenn ein Rang nicht passt, den W122-Satz zurueck. Seit H33
    daneben die KARTEN-Bilanz (Posten ausserhalb des Budgets, gemessen;
    ``card_reference_logs`` leer = :data:`D_CARD_REFERENCE_FNFL2`) mit W130.
    """
    import json
    import os

    from sglang.srt.planner import pp_cut as _pp_cut

    n = len(budgets_mib)
    if not _env_true(env_d, "SGLANG_UNEVEN_MOE_EXPERT_SHARD"):
        return DResidencyPlan(
            lines=(
                "%s FRACTION-SOLVE %s ENTFAELLT: SGLANG_UNEVEN_MOE_EXPERT_SHARD ist in der "
                "D-Env nicht an -- ohne den Experten-Schnitt ist die lokale Zeilenzahl "
                "nicht Spanne+Pad und die Pufferregel dieser Rechnung gilt nicht."
                % (marker, label),
            ),
            refusal=None,
        )
    terms = _pp_cut.checkpoint_weight_terms(model_path)
    with open(os.path.join(model_path, "config.json")) as fh:
        cfg = json.load(fh)
    text_cfg = cfg.get("text_config") or cfg
    slot_bytes = float(terms.expert_layer_weight_bytes) / int(terms.num_experts)
    dense_repack = dense_repack_outside_pool(env_d)
    ref, why = _reference_for(
        model_path=model_path,
        rank_tp_ratio=rank_tp_ratio,
        reference_logs=reference_logs,
        n_ranks=n,
        n_layers=int(terms.n_layers),
        slot_bytes=slot_bytes,
        dense_repack=dense_repack,
        reference_seats=reference_seats,
    )
    if ref is None:
        return DResidencyPlan(
            lines=("%s FRACTION-SOLVE %s ENTFAELLT: %s." % (marker, label, why),),
            refusal=None,
        )
    # H91b: die Sitze. Die Referenz-Posten gelten fuer IHRE Sitzzahl; ein
    # bs2-Boot mit der bs1-Zahl waere um einen ganzen Sitz unterbepreist und
    # stuerbe am KV-Pool bzw. an der Karte. Umbuchen VOR H64, damit dessen
    # per_req (Posten / (Sitze x D)) aus dem Posten DERSELBEN Sitzzahl folgt.
    seat_rb: Optional[SeatRebook] = None
    seat_lines: Tuple[str, ...] = ()
    ref_seats = int(ref.max_running)
    if seats is not None and int(seats) != ref_seats:
        seat_rb = seat_rebook(ref, seats=int(seats))
        ref = msgspec.structs.replace(
            ref,
            mamba_mib=seat_rb.mamba_mib,
            spec_mib=seat_rb.spec_mib,
            max_running=int(seats),
        )
        seat_lines = (
            "%s FRACTION-SOLVE %s %s"
            % (marker, label, describe_seat_rebook(seat_rb, source=ref.source)),
        )
    elif seats is not None:
        seat_lines = (
            "%s FRACTION-SOLVE %s D-SITZE (H91b): D faehrt %d Sitz(e) wie die Referenz "
            "%s -- ihre sitz-proportionalen Posten gelten unveraendert"
            % (marker, label, int(seats), ref.source),
        )
    # H64: der Spec-Posten der Referenz gilt fuer IHRE Verify-Form. Nur wenn
    # der Launcher eine Form uebergibt (--d-replayssm-spec on), wird er auf die
    # gefahrene umgebucht; ohne sie (Schalter aus) bleibt die Rechnung
    # byte-gleich.
    spec_rebook: Optional[ReplaySSMSpecRebook] = None
    spec_lines: Tuple[str, ...] = ()
    act_dtype = str(
        text_cfg.get("dtype")
        or text_cfg.get("torch_dtype")
        or cfg.get("torch_dtype")
        or cfg.get("dtype")
        or "bfloat16"
    )
    if replayssm_spec is not None:
        spec_rebook = replayssm_spec_rebook(
            spec_mib_ref=ref.spec_mib,
            ref_ring_len=ref.replayssm_spec_ring_len,
            form=replayssm_spec,
            text_cfg=text_cfg,
            act_dtype=act_dtype,
        )
        if replayssm_spec.ring_len != ref.replayssm_spec_ring_len:
            ref = msgspec.structs.replace(ref, spec_mib=spec_rebook.spec_mib)
            spec_lines = (
                "%s FRACTION-SOLVE %s %s"
                % (
                    marker,
                    label,
                    describe_spec_rebook(
                        spec_rebook, source=ref.source, form=replayssm_spec
                    ),
                ),
            )
        else:
            spec_lines = (
                "%s FRACTION-SOLVE %s REPLAYSSM-SPEC (H64): D faehrt mit %s, die "
                "Referenz %s ist in derselben Form gemessen -- ihr Posten gilt "
                "unveraendert"
                % (marker, label, spec_form_text(replayssm_spec.ring_len), ref.source),
            )
    share = draft_share_embed(env_d)
    staging = int(str(env_d.get(POOL_STAGING_ENV, "")).strip() or POOL_STAGING_DEFAULT)
    vocab = draft_vocab_mib(
        vocab_size=int(text_cfg["vocab_size"]), hidden_size=int(text_cfg["hidden_size"])
    )
    fits = solve_d_rank_residency(
        budgets_mib=budgets_mib,
        fractions=fractions,
        ratios=ratios,
        scratch_rows=scratch_rows,
        staging_rows=staging,
        num_experts=int(terms.num_experts),
        pad_rows=1,
        n_layers=int(terms.n_layers),
        slot_bytes=slot_bytes,
        reference=ref,
        vocab_mib=vocab,
        share_embed=share,
        kv_tokens=int(kv_tokens),
    )
    dcp_note = (
        " KV-Anteil: SGLANG_UNEVEN_DCP ist an, der Token-Schnitt je Rang ist hier NICHT "
        "modelliert -- jeder Rang ist mit dem vollen Kontext bepreist (Obergrenze des "
        "KV-Postens)."
        if _env_true(env_d, "SGLANG_UNEVEN_DCP")
        else " KV-Anteil: kein uneven DCP, jeder Rang haelt den vollen Kontext."
    )
    head = (
        "%s FRACTION-SOLVE %s (Pufferregel, H8): Budget je Rang %s MiB, %d Layer x "
        "%.3f MiB/Zeile, %d Experten nach --rank-moe-ratio %s + 1 Pad-Zeile, Scratch "
        "%s, Staging %d, Draft-Vokabular %s (%s=%s, %.0f MiB), %d Token KV Pflicht, "
        "fester Rang-Posten gemessen in %s (%s, %s=%s) -> DECKE je Rang %s "
        "(gegeben: %s)%s"
        % (
            marker,
            label,
            [int(b) for b in budgets_mib],
            int(terms.n_layers),
            slot_bytes / MIB,
            int(terms.num_experts),
            ",".join("%g" % r for r in ratios),
            [int(s) for s in scratch_rows],
            staging,
            "GETEILT" if share else "EIGEN",
            DRAFT_SHARE_EMBED_ENV,
            "1" if share else "0",
            vocab,
            int(kv_tokens),
            ref.source,
            h39_state_text(ref.dense_repack_outside_pool),
            DENSE_REPACK_OUTSIDE_POOL_ENV,
            "1" if dense_repack else "0",
            [
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ],
            ["%.3f" % f.fraction for f in fits],
            dcp_note,
        )
    )
    lines = (head,) + seat_lines + spec_lines + tuple(
        "%s FRACTION-SOLVE %s %s" % (marker, label, describe_rank(f)) for f in fits
    )
    step_lines, step_refusal = pool_step_rows_check(
        fits,
        seats=seats,
        verify_tokens=(replayssm_spec.draft_tokens if replayssm_spec is not None else None),
        top_k=text_cfg.get("num_experts_per_tok"),
        pool_mode=str(env_d.get(POOL_GRAPH_MODE_ENV, "")).strip().lower() == "pool",
        marker=marker,
        label=label,
        waves=pool_overflow_waves(env_d),
    )
    lines = lines + step_lines
    card_lines, cards, card_refusal = _plan_d_card(
        fits=fits,
        model_path=model_path,
        rank_tp_ratio=rank_tp_ratio,
        card_reference_logs=card_reference_logs,
        n_layers=int(terms.n_layers),
        slot_bytes=slot_bytes,
        vocab_mib=vocab,
        share_embed=share,
        label=label,
        marker=marker,
        dense_repack=dense_repack,
        replayssm_spec=replayssm_spec,
        spec_per_req_mib=(
            spec_rebook.per_req_mib if spec_rebook is not None else None
        ),
        text_cfg=text_cfg,
        act_dtype=act_dtype,
        seat_rb=seat_rb,
        seat_graph_mib=seat_graph_mib,
    )
    refusals = [
        t for t in (refusal_text(fits, label=label), card_refusal, step_refusal) if t
    ]
    return DResidencyPlan(
        lines=lines + card_lines,
        refusal="; ".join(refusals) if refusals else None,
        fits=fits,
        card_fits=cards,
    )


def _seat_card_shift(
    rb: SeatRebook,
    *,
    card_ring_len: Optional[int],
    replayssm_spec: Optional[ReplaySSMSpecForm],
    spec_per_req_mib: Optional[Sequence[float]],
    text_cfg: Optional[Mapping[str, object]],
    act_dtype: Optional[str],
    seat_graph_mib: Optional[Sequence[float]],
    source: str,
    marker: str,
    label: str,
) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
    """H91b: um wie viel die Karte je Rang ENGER wird (negativ), weil D mehr
    Sitze faehrt als die Karten-Referenz: Mamba-Slots (Allokation = Posten),
    Verify-Allokation in der Form der Karten-Referenz (``(Sitze+1)`` Zeilen,
    wie ``MambaPool`` sie baut) und -- nur wenn gemessen uebergeben -- der
    Decode-Graph je zusaetzlichem Sitz."""
    n = len(rb.mamba_mib)
    extra = rb.seats - rb.ref_seats
    mamba = [rb.mamba_mib[r] - rb.mamba_ref_mib[r] for r in range(n)]
    how = ""
    if replayssm_spec is not None and spec_per_req_mib is not None and text_cfg is not None:
        a_seats = replayssm_spec_alloc_mib(
            per_req_mib=spec_per_req_mib,
            ring_len=card_ring_len,
            form=msgspec.structs.replace(replayssm_spec, max_running=rb.seats),
            text_cfg=text_cfg,
            act_dtype=act_dtype,
        )
        a_ref = replayssm_spec_alloc_mib(
            per_req_mib=spec_per_req_mib,
            ring_len=card_ring_len,
            form=msgspec.structs.replace(replayssm_spec, max_running=rb.ref_seats),
            text_cfg=text_cfg,
            act_dtype=act_dtype,
        )
        spec = [a - b for a, b in zip(a_seats, a_ref)]
        how = "Verify-Allokation in der Form der Karten-Referenz (%s)" % spec_form_text(card_ring_len)
    else:
        spec = [rb.spec_mib[r] - rb.spec_ref_mib[r] for r in range(n)]
        how = "Verify-Allokation ~ Spec-Posten (keine Verify-Form uebergeben; Obergrenze)"
    graph = [0.0] * n
    if seat_graph_mib is not None and len(seat_graph_mib) in (1, n):
        vals = list(seat_graph_mib) * (n if len(seat_graph_mib) == 1 else 1)
        graph = [float(v) * extra for v in vals]
        graph_text = "Decode-Graph je Sitz %s MiB GEMESSEN uebergeben" % [float(v) for v in vals]
    else:
        graph_text = (
            "Decode-Graph des zusaetzlichen Sitzes NICHT GEBUCHT (--d-seat-graph-mib "
            "fehlt: die Graph-Pool-Mehrkosten von bs%d sind nur am Metall messbar) -- "
            "diese KARTE-Zeile ist damit eine OBERGRENZE" % rb.seats
        )
    shift = tuple(round(-(mamba[r] + spec[r] + graph[r]), 1) for r in range(n))
    line = (
        "%s KARTE %s D-SITZE (H91b): D faehrt %d Sitz(e), Karten-Referenz %s mit %d -> "
        "Mamba-Slots %s MiB, %s %s MiB, Graph %s MiB => Kopfraum/Decode-frei je Rang "
        "%s MiB -- GERECHNET; %s"
        % (
            marker,
            label,
            rb.seats,
            source,
            rb.ref_seats,
            ["%+.1f" % x for x in mamba],
            how,
            ["%+.1f" % x for x in spec],
            ["%+.1f" % x for x in graph],
            ["%+.1f" % x for x in shift],
            graph_text,
        ),
    )
    return shift, line


def _card_reference_for(
    *,
    model_path: str,
    rank_tp_ratio: str,
    card_reference_logs: str,
    n_ranks: int,
    n_layers: int,
    slot_bytes: float,
    dense_repack: bool = DENSE_REPACK_OUTSIDE_POOL_DEFAULT,
) -> Tuple[Optional[DCardReference], str]:
    """Die Karten-Referenz fuer DIESE Form und DIESEN Baum-Zustand (H50), oder
    ``(None, warum nicht)``."""
    import os

    model = os.path.basename(os.path.normpath(model_path))
    paths = [p.strip() for p in str(card_reference_logs or "").split(",") if p.strip()]
    if paths:
        boots = []
        for p in paths:
            with open(p, errors="replace") as fh:
                boots.append((os.path.basename(p), fh.read()))
        ref = d_card_reference_from_logs(
            boots,
            n_ranks=n_ranks,
            n_layers=n_layers,
            slot_bytes=slot_bytes,
            model=model,
            rank_tp_ratio=rank_tp_ratio,
        )
        if ref.dense_repack_outside_pool != bool(dense_repack):
            return None, _h39_mismatch_text(
                ref.source, ref.dense_repack_outside_pool, dense_repack,
                "--d-card-reference-logs",
            )
        return ref, ""
    ref = D_CARD_REFERENCE_FNFL2
    for cand in D_CARD_REFERENCES:
        if cand.dense_repack_outside_pool == bool(dense_repack):
            ref = cand
            break
    if (
        ref.model != model
        or ref.rank_tp_ratio != rank_tp_ratio
        or len(ref.headroom0_mib) != n_ranks
        or ref.dense_repack_outside_pool != bool(dense_repack)
    ):
        return None, (
            "die eingebaute Karten-Referenz (%s) gilt fuer %s mit --rank-tp-ratio %s "
            "auf %d Raengen (%s), dieser Boot faehrt %s mit %s auf %d (%s); den "
            "Posten ausserhalb des Budgets per --d-card-reference-logs <D.log,...> "
            "aus Boots DIESER Form messen (WEG2-GRAPH-POOL bzw. [vram-peak] + #1027)"
            % (
                ref.source,
                ref.model,
                ref.rank_tp_ratio,
                len(ref.headroom0_mib),
                h39_state_text(ref.dense_repack_outside_pool),
                model,
                rank_tp_ratio,
                n_ranks,
                h39_state_text(bool(dense_repack)),
            )
        )
    return ref, ""


def _plan_d_card(
    *,
    fits: Sequence[DRankResidency],
    model_path: str,
    rank_tp_ratio: str,
    card_reference_logs: str,
    n_layers: int,
    slot_bytes: float,
    vocab_mib: float,
    share_embed: bool,
    label: str,
    marker: str,
    dense_repack: bool = DENSE_REPACK_OUTSIDE_POOL_DEFAULT,
    replayssm_spec: Optional[ReplaySSMSpecForm] = None,
    spec_per_req_mib: Optional[Sequence[float]] = None,
    text_cfg: Optional[Mapping[str, object]] = None,
    act_dtype: Optional[str] = None,
    seat_rb: Optional[SeatRebook] = None,
    seat_graph_mib: Optional[Sequence[float]] = None,
) -> Tuple[Tuple[str, ...], Tuple[DCardFit, ...], Optional[str]]:
    """H33: die Karten-Bilanz neben der Budget-Bilanz. Eine unlesbare Referenz
    verweigert nicht, sie wird benannt (wie H8); verweigert wird nur aus einer
    GERECHNETEN Bilanz."""
    from sglang.srt.managers.corridor_guard import (
        NEAR_OOM_MIB,
        corridor_band_floor_mib,
    )

    try:
        ref, why = _card_reference_for(
            model_path=model_path,
            rank_tp_ratio=rank_tp_ratio,
            card_reference_logs=card_reference_logs,
            n_ranks=len(fits),
            n_layers=n_layers,
            slot_bytes=slot_bytes,
            dense_repack=dense_repack,
        )
    except (OSError, ValueError) as exc:
        ref, why = None, "%s: %s" % (type(exc).__name__, exc)
    if ref is None:
        return (
            ("%s KARTE %s ENTFAELLT: %s." % (marker, label, why),),
            (),
            None,
        )
    # H64: die Karten-Referenz misst die Verify-Allokation IHRER Form im Peak
    # mit. Faehrt D eine andere (--d-replayssm-spec), wird der Kopfraum um den
    # Unterschied verschoben -- GERECHNET, aus dem Zustand je Request des Rangs
    # (spec_per_req_mib, hergeleitet aus dem gemessenen Referenz-Posten).
    shift_line: Tuple[str, ...] = ()
    if (
        replayssm_spec is not None
        and spec_per_req_mib is not None
        and text_cfg is not None
        and ref.replayssm_spec_ring_len != replayssm_spec.ring_len
    ):
        a_ref = replayssm_spec_alloc_mib(
            per_req_mib=spec_per_req_mib,
            ring_len=ref.replayssm_spec_ring_len,
            form=replayssm_spec,
            text_cfg=text_cfg,
            act_dtype=act_dtype,
        )
        a_run = replayssm_spec_alloc_mib(
            per_req_mib=spec_per_req_mib,
            ring_len=replayssm_spec.ring_len,
            form=replayssm_spec,
            text_cfg=text_cfg,
            act_dtype=act_dtype,
        )
        shift = tuple(round(a - b, 1) for a, b in zip(a_ref, a_run))
        ref = msgspec.structs.replace(
            ref,
            headroom0_mib=tuple(
                round(h + s, 1) for h, s in zip(ref.headroom0_mib, shift)
            ),
            # der Peak der Referenz traegt die Verify-Allokation ihrer Form;
            # gesenkt um den Unterschied, damit 'Kopfraum = cap - peak -
            # privat_frei' im Druck (describe_card) wahr bleibt.
            peak_mib=tuple(
                round(pk - s, 1) for pk, s in zip(ref.peak_mib, shift)
            ),
            free_decode0_mib=tuple(
                None if f is None else round(f + s, 1)
                for f, s in zip(ref.free_decode0_mib, shift)
            ),
        )
        shift_line = (
            "%s KARTE %s REPLAYSSM-SPEC (H64): Referenz %s mit %s gemessen, D "
            "faehrt mit %s -> Verify-Allokation je Rang %s MiB, Kopfraum und "
            "Decode-frei um %s MiB verschoben -- GERECHNET, nicht gemessen"
            % (
                marker,
                label,
                ref.source,
                spec_form_text(ref.replayssm_spec_ring_len),
                spec_form_text(replayssm_spec.ring_len),
                ["%.1f -> %.1f" % (a, b) for a, b in zip(a_ref, a_run)],
                ["%+.1f" % s for s in shift],
            ),
        )
    seat_line: Tuple[str, ...] = ()
    if seat_rb is not None:
        seat_shift, seat_line = _seat_card_shift(
            seat_rb,
            card_ring_len=ref.replayssm_spec_ring_len,
            replayssm_spec=replayssm_spec,
            spec_per_req_mib=spec_per_req_mib,
            text_cfg=text_cfg,
            act_dtype=act_dtype,
            seat_graph_mib=seat_graph_mib,
            source=ref.source,
            marker=marker,
            label=label,
        )
        ref = msgspec.structs.replace(
            ref,
            headroom0_mib=tuple(
                round(h + s, 1) for h, s in zip(ref.headroom0_mib, seat_shift)
            ),
            peak_mib=tuple(round(pk - s, 1) for pk, s in zip(ref.peak_mib, seat_shift)),
            free_decode0_mib=tuple(
                None if f is None else round(f + s, 1)
                for f, s in zip(ref.free_decode0_mib, seat_shift)
            ),
        )
        shift_line = shift_line + seat_line
    floor = float(corridor_band_floor_mib())
    cards = solve_d_card(
        fits=fits,
        reference=ref,
        vocab_mib=vocab_mib,
        share_embed=share_embed,
        near_oom_mib=float(NEAR_OOM_MIB),
        band_floor_mib=floor,
    )
    edges = [
        scratch_edge(
            local_experts=f.local_experts,
            fraction=f.fraction,
            max_rows=min(f.ceiling_max_rows, c.ceiling_max_rows),
        )
        for f, c in zip(fits, cards)
    ]
    head = (
        "%s KARTE %s (H33, Posten ausserhalb des Budgets GEMESSEN): Kopfraum je Rang "
        "= cap - peak - privat_frei am bindenden Messpunkt von %s (cap = card free + "
        "reserved, peak = allocator peak since pools, privat_frei = freie Bloecke "
        "privater Graph-/Tag-Pools, fuer empty_cache unerreichbar; %s), verschoben "
        "um die Pufferbytes dieses Boots; Grenze near-OOM %d MiB "
        "(corridor_guard.NEAR_OOM_MIB, Stopper in jeder Phase), Decode gegen den "
        "Band-Floor %d MiB (Befund) -> KARTEN-DECKE je Rang %s, BUDGET-DECKE %s "
        "(gegeben: %s) | KANTE bei gegebener f (min Budget, Karte): Zeilen %s = "
        "SCRATCH <= %s (gegeben: %s)"
        % (
            marker,
            label,
            ref.source,
            h39_state_text(ref.dense_repack_outside_pool),
            NEAR_OOM_MIB,
            floor,
            [
                "KEINE" if c.ceiling_fraction is None else "%.3f" % c.ceiling_fraction
                for c in cards
            ],
            [
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ],
            ["%.3f" % f.fraction for f in fits],
            [e[0] for e in edges],
            ["KEINE" if e[1] is None else e[1] for e in edges],
            [int(f.scratch_rows) for f in fits],
        )
    )
    lines = (head,) + tuple(
        "%s KARTE %s %s" % (marker, label, describe_card(c, ref)) for c in cards
    ) + shift_line
    return lines, cards, card_refusal_text(cards, ref, label=label)
