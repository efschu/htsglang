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
) -> DRankReference:
    """Den festen Rang-Posten aus D-Logs MESSEN (``boots`` = (Name, Text)).

    Je Boot und Rang: ``fixed = 'weights + runtime state' - buffer x
    n_layers x slot`` -- beide Zahlen aus DEMSELBEN Log, sonst kein Wert. Ueber
    die Boots gilt je Term das Maximum (der Posten streut gemessen um bis zu
    130 MiB je Rang: Allokator-Luecke des Ladens). Fehlt einem Rang ein Term
    in ALLEN Boots, wird verweigert, statt eine Null einzusetzen.
    """
    layer_mib = float(n_layers) * float(slot_bytes) / MIB
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
    )


#: Die gemessene Referenz der Next-Flash-Form-A-D-Gruppe (Stand 5a96de48be).
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
# 5. die Launcher-Naht: alles, was der D-FRACTION-SOLVE liest, an EINER Stelle
# ---------------------------------------------------------------------------


class DResidencyPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Was der Launcher druckt (``lines``) und ob er verweigert (``refusal``,
    der W122-Satz; ``None`` = der Boot passt oder die Rechnung entfaellt)."""

    lines: Tuple[str, ...]
    refusal: Optional[str]
    fits: Tuple[DRankResidency, ...] = ()


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
) -> Tuple[Optional[DRankReference], str]:
    """Die Referenz fuer DIESE Form, oder ``(None, warum nicht)``."""
    import os

    model = os.path.basename(os.path.normpath(model_path))
    paths = [p.strip() for p in str(reference_logs or "").split(",") if p.strip()]
    if paths:
        boots = []
        for p in paths:
            with open(p, errors="replace") as fh:
                boots.append((os.path.basename(p), fh.read()))
        return (
            d_rank_reference_from_logs(
                boots,
                n_ranks=n_ranks,
                n_layers=n_layers,
                slot_bytes=slot_bytes,
                model=model,
                rank_tp_ratio=rank_tp_ratio,
            ),
            "",
        )
    ref = D_RESIDENCY_REFERENCE_FNFL2
    if (
        ref.model != model
        or ref.rank_tp_ratio != rank_tp_ratio
        or len(ref.fixed_mib) != n_ranks
    ):
        return None, (
            "die eingebaute Referenz (%s) gilt fuer %s mit --rank-tp-ratio %s auf %d "
            "Raengen, dieser Boot faehrt %s mit %s auf %d; den festen Rang-Posten "
            "per --d-residency-reference-logs <D.log,...> aus Boots DIESER Form messen"
            % (
                ref.source,
                ref.model,
                ref.rank_tp_ratio,
                len(ref.fixed_mib),
                model,
                rank_tp_ratio,
                n_ranks,
            )
        )
    return ref, ""


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
) -> DResidencyPlan:
    """Der D-FRACTION-SOLVE mit den Metallregeln, fuer ``launcher``.

    Liest die Checkpoint-Geometrie (Header, keine Tensoren), die Gruppen-Env
    von D und die gemessene Referenz; rechnet je Rang die Bilanz und gibt die
    Zeilen und, wenn ein Rang nicht passt, den W122-Satz zurueck.
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
    ref, why = _reference_for(
        model_path=model_path,
        rank_tp_ratio=rank_tp_ratio,
        reference_logs=reference_logs,
        n_ranks=n,
        n_layers=int(terms.n_layers),
        slot_bytes=slot_bytes,
    )
    if ref is None:
        return DResidencyPlan(
            lines=("%s FRACTION-SOLVE %s ENTFAELLT: %s." % (marker, label, why),),
            refusal=None,
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
        "fester Rang-Posten gemessen in %s -> DECKE je Rang %s (gegeben: %s)%s"
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
            [
                "KEINE" if f.ceiling_fraction is None else "%.3f" % f.ceiling_fraction
                for f in fits
            ],
            ["%.3f" % f.fraction for f in fits],
            dcp_note,
        )
    )
    lines = (head,) + tuple(
        "%s FRACTION-SOLVE %s %s" % (marker, label, describe_rank(f)) for f in fits
    )
    return DResidencyPlan(
        lines=lines, refusal=refusal_text(fits, label=label), fits=fits
    )
