# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H57 -- das Power-Limit je Karte: Startzeile, Datierung der Zeit-/Raten-
Referenzen, Schnitt aus gemessenen Stufenraten.

DIE ORDER (Nutzer 24.09. 18:53Z, woertlich): "die geschwindigkeiten der karten
ist noch im powerlimit bei 400 und 230 deswegen muss der schnitt aufjedenfall
anpassbar sein, da ich spaeter das powerlimit ggf. erhoehen werde". Am Metall
(nvidia-smi 18:55Z): nvml0 3080 230 W (max 320), nvml1 5090 400 W (max 600),
nvml2 3080 230 W (max 320).

WAS AM LIMIT HAENGT, GEMESSEN. fnFL2x148 fuhr die x146-Form (Schnitt 29,11,8,
Chunk 16384, FR_P 0.26/0.45/0.39) unter 525/320 W statt 400/230 W: der volle
Chunk je Stufe wurde 8-11 % schneller (compute PP0 3890.6 -> 3476.3 ms, PP1
4139.8 -> 3792.9, PP2 2298.5 -> 2072.9; P 97k 32,61 -> 30,29 s), Decode blieb
in der Streuung, der VRAM gleich. Bytes haengen nicht am Limit, Rechenzeit
schon -- dieselbe Trennung, die #584 fuer die Kartenraten fand (``rate_env``:
GEMM -12/-22 % nach der Limit-Senkung vom 05.08., Bandbreite unveraendert).

DREI TEILE:

1. :func:`read_card_power` / :func:`launch_line` / :func:`state_dict` -- die
   Startzeile ``POWER-LIMIT nvml0=230/320W nvml1=400/600W nvml2=230/320W
   sm_clock_max=...`` (NVML, nur lesend) und derselbe Inhalt im
   Boot-Zustands-JSON. Ab H57 traegt jeder Boot sein Limit; vorher keiner --
   die Stempel unten stammen darum aus nvidia-smi-Notizen, nicht aus Logs.
2. :data:`TIMED_REFERENCES` + :func:`card_library_reference` -- jede Zeit-/
   Raten-Referenz, die der NF-Planer liest, mit dem Limit je Karte
   (:class:`LimitStamp`, nach UUID), unter dem sie gemessen wurde. Weicht das
   aktuelle ab, druckt :func:`reference_lines` ``REFERENZ VERALTET
   (Power-Limit ...)`` mit ihrem Namen: eine Zeile, KEINE Verweigerung.
   :data:`LIMIT_NEUTRAL_REFERENCES` (VRAM, PCIe) tragen keinen Stempel; die
   VRAM-Riegel W122/W130/W132/W126 lesen nur sie und bleiben unberuehrt.
3. :data:`STAGE_RATES` + :func:`rate_cut_line` -- ms je Schicht und Token je
   Karte aus den P-Logs (``Prefill rank batch ... gpu-ms (compute ..)``), je
   Power-Limit ein Satz. Passt ein Satz zum AKTUELLEN Limit, druckt der Planer
   den Schnitt, der die Stufen balanciert, sonst ``keine Raten fuer <Limit>,
   Schnitt bleibt gepinnt``. Nie angewendet: der Schnitt bleibt EINE
   Arm-Variable, ``PP_RATIO`` in ``/spinning/gpu-arb/weg2/arm_fnFL2_long.sh``
   (Default 29,11,8) -> ``--pp-stage-ratio`` -> ``solve_p_cut`` (PINNED) ->
   group P's argv; ``PP_ATTN_RATIO`` ist keine zweite Wahl, sondern die
   Attention-Zahl, auf der dieser Schnitt landet (die Zeile druckt beide).

Rein (kein torch). NVML nur in :func:`read_card_power`, injizierbar.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.planner.rate_env import POWER_LIMIT_TOLERANCE_MW, RateEnv

#: Je Zeilenart EIN Praefix, damit ``grep -c`` zaehlt.
LAUNCH_MARKER = "POWER-LIMIT"
STALE_MARKER = "REFERENZ VERALTET"
UNDATED_MARKER = "REFERENZ UNDATIERT"
SUMMARY_MARKER = "POWER-LIMIT REFERENZEN"
CUT_MARKER = "PP-CUT RATEN-SCHNITT"

#: Limits werden in ganzen Watt gesetzt, der Treiber rundet um Milliwatt --
#: dieselbe Toleranz wie ``rate_env`` (dort in mW), nicht eine zweite.
POWER_TOLERANCE_W = POWER_LIMIT_TOLERANCE_MW / 1000.0


# ---------------------------------------------------------------------------
# 1. die Lesung
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardPower:
    """Eine Karte, wie NVML sie jetzt meldet (Watt/MHz; ``None`` = nicht lesbar)."""

    nvml_index: int
    uuid: str
    name: str
    limit_w: Optional[float]
    max_w: Optional[float] = None
    default_w: Optional[float] = None
    sm_clock_max_mhz: Optional[int] = None

    @property
    def label(self) -> str:
        return "nvml%d" % int(self.nvml_index)


@dataclasses.dataclass(frozen=True)
class PowerReading:
    """Alle Karten einer Lesung. ``error`` sagt, warum etwas fehlt."""

    cards: Tuple[CardPower, ...] = ()
    source: str = "nvml"
    error: str = ""

    def by_uuid(self) -> Dict[str, CardPower]:
        return {c.uuid: c for c in self.cards}

    @property
    def known(self) -> bool:
        return bool(self.cards) and all(c.limit_w is not None for c in self.cards)


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _call(obj, attr: str, *args):
    """Eine NVML-Abfrage, die fehlen oder scheitern darf (aeltere Bindung,
    Karte ohne Power-Management): ``None`` statt eines erfundenen Werts."""
    try:
        return getattr(obj, attr)(*args)
    except Exception:  # noqa: BLE001 -- ein fehlendes Feld ist kein Fehler des Boots
        return None


def _watt(mw) -> Optional[float]:
    return None if mw is None else float(mw) / 1000.0


def _float_or_none(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _read_nvml(pynvml, source: str) -> PowerReading:
    try:
        count = int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:  # noqa: BLE001
        return PowerReading(
            (), source, "nvmlDeviceGetCount: %s: %s" % (type(exc).__name__, exc)
        )
    clock_sm = getattr(pynvml, "NVML_CLOCK_SM", 1)
    cards: List[CardPower] = []
    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            name = _decode(pynvml.nvmlDeviceGetName(handle))
        except (
            Exception
        ):  # noqa: BLE001 -- eine unlesbare Karte fehlt, sie wird nicht erfunden
            continue
        constraints = _call(
            pynvml, "nvmlDeviceGetPowerManagementLimitConstraints", handle
        )
        clock = _call(pynvml, "nvmlDeviceGetMaxClockInfo", handle, clock_sm)
        cards.append(
            CardPower(
                nvml_index=index,
                uuid=uuid,
                name=name,
                # power.limit = was ``nvidia-smi -pl`` setzt; derselbe Wert, mit dem
                # rate_env die Kartenraten stempelt -- beide Stempel vergleichen
                # dieselbe Groesse.
                limit_w=_watt(
                    _call(pynvml, "nvmlDeviceGetPowerManagementLimit", handle)
                ),
                max_w=_watt(constraints[1]) if constraints else None,
                default_w=_watt(
                    _call(pynvml, "nvmlDeviceGetPowerManagementDefaultLimit", handle)
                ),
                sm_clock_max_mhz=None if clock is None else int(clock),
            )
        )
    missing = [c.label for c in cards if c.limit_w is None]
    error = (
        "NVML meldet keine Karte"
        if not cards
        else "Limit unlesbar auf %s" % ",".join(missing) if missing else ""
    )
    return PowerReading(tuple(cards), source, error)


def _read_replay(path: str) -> PowerReading:
    try:
        with open(path) as fh:
            rows = json.load(fh)
        cards = tuple(
            CardPower(
                nvml_index=int(r["index"]),
                uuid=str(r["uuid"]),
                name=str(r["name"]),
                limit_w=_float_or_none(r.get("power_limit_w")),
                max_w=_float_or_none(r.get("power_max_w")),
                default_w=_float_or_none(r.get("power_default_w")),
                sm_clock_max_mhz=(
                    None
                    if r.get("sm_clock_max_mhz") is None
                    else int(r["sm_clock_max_mhz"])
                ),
            )
            for r in rows
        )
    except Exception as exc:  # noqa: BLE001
        return PowerReading(
            (), "replay", "Replay %s unlesbar: %s: %s" % (path, type(exc).__name__, exc)
        )
    known = cards and all(c.limit_w is not None for c in cards)
    return PowerReading(
        cards,
        "replay",
        (
            ""
            if known
            else "Replay ohne Power-Felder (power_limit_w) -- das laufende Rig wird nicht gelesen"
        ),
    )


def read_card_power(nvml=None) -> PowerReading:
    """Jede Karte, die NVML sieht: Limit, Maximum, Default, SM-Maximaltakt.

    Nur lesend -- dieselben Abfragen wie ``nvidia-smi --query-gpu=power.limit,
    power.max_limit,power.default_limit,clocks.max.sm``. Wirft NIE: ein
    unlesbares NVML ergibt eine Lesung ohne Limits, und jede Referenz liest
    sich dann ``undatiert`` statt ``aktuell`` -- ein erfundener Wert laese sich
    als passend.

    Unter ``SGLANG_NVML_REPLAY_JSON`` (Desk-Replay, #1377) kommen Karten UND
    Limits aus den aufgezeichneten Zeilen (optionale Felder ``power_limit_w``,
    ``power_max_w``, ``power_default_w``, ``sm_clock_max_mhz``), nie vom
    laufenden Rig: ein Replay mit Live-Limits waere ein Boot auf zwei Maschinen.
    ``nvml`` (ein pynvml-artiges Objekt) ersetzt die Bindung fuer Tests.
    """
    if nvml is not None:
        return _read_nvml(nvml, "nvml")
    try:
        from sglang.srt.registry import nvml as _registry
    except Exception as exc:  # noqa: BLE001
        return PowerReading(
            (), "nvml", "registry.nvml: %s: %s" % (type(exc).__name__, exc)
        )
    replay = os.environ.get(_registry.ENV_NVML_REPLAY, "")
    if replay:
        return _read_replay(replay)
    try:
        with _registry.nvml_session() as pynvml:
            return _read_nvml(pynvml, "nvml")
    except Exception as exc:  # noqa: BLE001
        return PowerReading((), "nvml", "%s: %s" % (type(exc).__name__, exc))


def _fmt_w(w: Optional[float]) -> str:
    if w is None:
        return "?"
    return "%d" % round(w) if abs(w - round(w)) < 0.05 else "%.1f" % w


def launch_line(reading: PowerReading) -> str:
    """``POWER-LIMIT nvml0=230/320W nvml1=400/600W nvml2=230/320W
    sm_clock_max=nvml0:2100,nvml1:3090,nvml2:2100MHz source=nvml`` -- Limit/
    Maximum je Karte in NVML-Reihenfolge."""
    cards = sorted(reading.cards, key=lambda c: c.nvml_index)
    if not cards:
        return "%s unbekannt (%s) -- keine Zeit-/Raten-Referenz ist datierbar" % (
            LAUNCH_MARKER,
            reading.error or "keine Karte",
        )
    limits = " ".join(
        "%s=%s/%sW" % (c.label, _fmt_w(c.limit_w), _fmt_w(c.max_w)) for c in cards
    )
    clocks = ",".join(
        "%s:%s" % (c.label, "?" if c.sm_clock_max_mhz is None else c.sm_clock_max_mhz)
        for c in cards
    )
    tail = " (%s)" % reading.error if reading.error else ""
    return "%s %s sm_clock_max=%sMHz source=%s%s" % (
        LAUNCH_MARKER,
        limits,
        clocks,
        reading.source,
        tail,
    )


def state_dict(reading: PowerReading) -> Dict[str, object]:
    """Derselbe Inhalt fuer ``boot_<TAG>.json`` (``BootState.power_limits``)."""
    return {
        "line": launch_line(reading),
        "source": reading.source,
        "error": reading.error,
        "cards": {
            c.label: {
                "uuid": c.uuid,
                "name": c.name,
                "limit_w": c.limit_w,
                "max_w": c.max_w,
                "default_w": c.default_w,
                "sm_clock_max_mhz": c.sm_clock_max_mhz,
            }
            for c in sorted(reading.cards, key=lambda c: c.nvml_index)
        },
    }


_RX_LAUNCH = re.compile(r"POWER-LIMIT ((?:nvml\d+=[0-9.?]+/[0-9.?]+W\s*)+)")
_RX_LAUNCH_CARD = re.compile(r"nvml(\d+)=([0-9.?]+)/([0-9.?]+)W")
_RX_ORDINAL = re.compile(r"ordinal (\d+) = nvml (\d+) (.+?) (GPU-[0-9a-fA-F-]+) total")


def limits_from_launch_log(text: str) -> Dict[int, float]:
    """``{nvml_index: W}`` aus der ersten ``POWER-LIMIT``-Zeile eines
    Launcher-Logs (leer: der Boot ist aelter als H57 oder las NVML nicht)."""
    m = _RX_LAUNCH.search(text)
    if m is None:
        return {}
    return {
        int(i): float(w)
        for i, w, _mx in _RX_LAUNCH_CARD.findall(m.group(1))
        if w != "?"
    }


def ordinal_cards_from_launch_log(text: str) -> List[Tuple[str, int, str]]:
    """``[(uuid, nvml_index, name)]`` in CUDA-Ordinal- = P-Stufen-Reihenfolge,
    aus der Zeile ``NVML -> CUDA ordinal map`` desselben Launcher-Logs."""
    line = next(
        (ln for ln in text.splitlines() if "NVML -> CUDA ordinal map" in ln), ""
    )
    rows = sorted(
        (int(o), int(n), name.strip(), uuid)
        for o, n, name, uuid in _RX_ORDINAL.findall(line)
    )
    return [(uuid, n, name) for _o, n, name, uuid in rows]


# ---------------------------------------------------------------------------
# 2. die Stempel und die Referenzen
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LimitStamp:
    """Das Power-Limit EINER Karte, unter dem eine Referenz gemessen wurde.

    Nach UUID verglichen -- die NVML-Nummer kann zwischen Treiber-Sitzungen
    wandern; ``nvml_index`` steht nur fuer den Druck. ``limit_w`` ``None`` =
    die Referenz selbst weiss es nicht (Kartenraten ohne ``rate_env``)."""

    uuid: str
    nvml_index: int
    name: str
    limit_w: Optional[float]

    @property
    def label(self) -> str:
        return "nvml%d" % int(self.nvml_index)


#: Die drei Karten dieses Rigs, wie jeder NF-Boot sie druckt (``NVML -> CUDA
#: ordinal map``, fnFL2x148/x160/x162/x163 identisch): Ordinal 0 = P-Stufe 0 =
#: die 5090, dann die 3080 nach NVML-Nummer.
RIG_5090 = ("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", 1, "NVIDIA GeForce RTX 5090")
RIG_3080_NVML0 = (
    "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
    0,
    "NVIDIA GeForce RTX 3080",
)
RIG_3080_NVML2 = (
    "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",
    2,
    "NVIDIA GeForce RTX 3080",
)


def rig_stamps(w_5090: float, w_3080: float) -> Tuple[LimitStamp, ...]:
    """Die drei Stempel in P-Stufen-Reihenfolge (5090, nvml0, nvml2)."""
    return (
        LimitStamp(*RIG_5090, float(w_5090)),
        LimitStamp(*RIG_3080_NVML0, float(w_3080)),
        LimitStamp(*RIG_3080_NVML2, float(w_3080)),
    )


#: Woher die Stempel wissen, was galt. Vor H57 schrieb kein Boot sein Limit.
STAMP_SOURCE_400_230 = (
    "nvidia-smi 2026-09-06 22:17Z (Memory power-targets-reduziert: 3080 230 W, 5090 400 W) "
    "und 2026-09-24 18:55Z (dasselbe); dazwischen einzig fnFL2x148 (24.09. 11:50-12:10Z) "
    "unter 525/320 W"
)
STAMP_SOURCE_525_320 = (
    "Operator-Notiz 24.09. 11:50Z (Memory decode-probe-kalt-gegen-warm-ple-seiten-0924): "
    "Nutzer hob die 3080 230->320 W und die 5090 400->525 W fuer fnFL2x148 und senkte "
    "danach wieder -- nicht NVML-gestempelt"
)


@dataclasses.dataclass(frozen=True)
class TimedReference:
    """Eine Zeit-/Raten-Referenz des Planers mit ihrem Power-Limit je Karte."""

    name: str
    source: str
    quantity: str
    consumer: str
    stamps: Tuple[LimitStamp, ...]
    stamp_source: str
    #: Wie sie unter dem neuen Limit neu entsteht -- die Zeile sagt es mit.
    remeasure: str = ""


#: Jede Zeit-/Raten-Referenz, die der NF-Launcher im Dry-Run liest. Der Test
#: ``test_every_stamped_name_is_a_live_constant`` bindet jeden Namen an seine
#: Konstante; die Kartenraten-Bibliothek kommt dynamisch dazu
#: (:func:`card_library_reference`), weil sie ihren Stempel selbst traegt.
TIMED_REFERENCES: Tuple[TimedReference, ...] = (
    TimedReference(
        name="launcher.MEASURED_MS_PER_LAYER",
        source="boot bsscale 2026-09-07 19:0xZ (Qwen3.8-27B, Schnitt 32,18,14, Chunk 4096, "
        "BSSCALE_0907.md)",
        quantity="ms je Schicht und voller Chunk je Stufe",
        consumer="PP-CUT solver (Zeitachse: Anker der Kartenraten, Familien-Split) und "
        "PP-CUT depth axis",
        stamps=rig_stamps(400.0, 230.0),
        stamp_source=STAMP_SOURCE_400_230,
        remeasure="ein P-Messboot unter dem neuen Limit, ms je Schicht aus 'Prefill rank batch' "
        "(--pp-cut-measured-ms-per-layer)",
    ),
    TimedReference(
        name="launcher.ATTN_ANCHOR_MS",
        source="Nutzer-Physiknotiz 2026-09-07 (0,4 s je Attention-Schicht und Chunk auf "
        "einer 3080 bei 262144 Token Praefix)",
        quantity="ms je Attention-Schicht und Chunk (tiefer Anker)",
        consumer="PP-CUT depth axis (Attention/Linear-Split)",
        stamps=rig_stamps(400.0, 230.0)[1:],
        stamp_source=STAMP_SOURCE_400_230,
        remeasure="ein 262k-Prompt unter dem neuen Limit, tiefster Chunk auf einer 3080-Stufe "
        "(--pp-cut-attn-anchor-ms)",
    ),
    TimedReference(
        name="wake_credit_pd_refs.REFERENCES",
        source="fnFL2x141 + fnFL2x144 + fnFL2x158 (24.09. 09:32Z / 10:25Z / 14:02Z)",
        quantity="Flip-Leg-Zeiten P->D (deposit/pause/resume/Lane-ms)",
        consumer="WAKE-CREDIT (#H14) P->D: Leg-ms, Kreditwarten, getimte Ordnung (die "
        "W126-Verweigerung ist eine VRAM-Summe und bleibt)",
        stamps=rig_stamps(400.0, 230.0),
        stamp_source=STAMP_SOURCE_400_230,
        remeasure="ein Flip-Boot unter dem neuen Limit, wake_credit_pd.pd_reference_from_logs",
    ),
)


@dataclasses.dataclass(frozen=True)
class NeutralReference:
    """Eine Referenz, deren Werte NICHT am Power-Limit haengen -- kein Stempel."""

    name: str
    source: str
    why: str
    gate: str


#: Was der Planer ausserdem liest und warum es keinen Stempel traegt. Diese
#: Liste ist die Grenze der Order: 'VRAM-Riegel bleiben unberuehrt'.
LIMIT_NEUTRAL_REFERENCES: Tuple[NeutralReference, ...] = (
    NeutralReference(
        "p_card_chunk.P_CARD_REFERENCE_FNFL2",
        "fnFL2x160",
        "VRAM: Kopfraum je P-Stufe (MiB)",
        "W132",
    ),
    NeutralReference(
        "p_card_chunk.P_CARD_REFERENCE_FNFL2_X150",
        "fnFL2x146 + fnFL2x150",
        "VRAM: Kopfraum je P-Stufe (MiB), historisch",
        "W132",
    ),
    NeutralReference(
        "launcher.P_PREFILL_TRANSIENT_SUPPORT",
        "fnFL2x118 + fnFL2x141 + fnFL2x145 + fnFL2x146 (Chunk 16384; 512/4096/8192 "
        "aus weiteren Boots)",
        "VRAM: Chunk-Transiente je Stufe (MiB)",
        "PP-CUT ACTIVATION, W131, W132",
    ),
    NeutralReference(
        "expert_residency.D_RESIDENCY_REFERENCE_FNFL2_H39",
        "fnFL2x151 + fnFL2x158",
        "VRAM: D-Posten je Rang (MiB)",
        "W122",
    ),
    NeutralReference(
        "expert_residency.D_RESIDENCY_REFERENCE_FNFL2",
        "fnFL2x98 + fnFL2x99 + fnFL2x100",
        "VRAM: D-Posten je Rang (MiB)",
        "W122",
    ),
    NeutralReference(
        "expert_residency.D_CARD_REFERENCE_FNFL2_H39",
        "fnFL2x151 + fnFL2x158",
        "VRAM: D-Kopfraum je Rang (MiB)",
        "W130",
    ),
    NeutralReference(
        "expert_residency.D_CARD_REFERENCE_FNFL2",
        "fnFL2x141 + fnFL2x144",
        "VRAM: D-Kopfraum je Rang (MiB)",
        "W130",
    ),
    NeutralReference(
        "wake_credit.REFERENCE_FNFL2X114D",
        "fnFL2x114d",
        "VRAM-Fixpunkt des ersten Wakes D->P, ohne Zeit",
        "W126:D->P",
    ),
    NeutralReference(
        "wake_credit_pd_refs.REFERENCES (MiB-Spalten)",
        "fnFL2x141 + fnFL2x144 + fnFL2x158",
        "VRAM: free/floor/Bedarf/Freigabe je Tag",
        "W126:P->D",
    ),
    NeutralReference(
        "pp_crossing_transport.MEASURED_GBPS_BY_LANES",
        "PCIe-Messtabelle",
        "PCIe-Kopie (Copy-Engine, kein SM-Takt)",
        "PP-CUT crossing prices",
    ),
)


def stamp_verdict(
    stamps: Sequence[LimitStamp], reading: Optional[PowerReading]
) -> Tuple[str, List[str]]:
    """``("aktuell"|"veraltet"|"unbekannt", Gruende)``. Veraltet gewinnt: EINE
    Karte unter anderem Limit genuegt, und ein unlesbarer Nachbar macht eine
    gemessene Abweichung nicht ungeschehen."""
    live = reading.by_uuid() if reading is not None else {}
    moved: List[str] = []
    unknown: List[str] = []
    for st in stamps:
        card = live.get(st.uuid)
        label = card.label if card is not None else st.label
        if st.limit_w is None:
            unknown.append("%s: Referenz ohne Stempel" % label)
        elif card is None:
            unknown.append("%s: Karte %s nicht in der Lesung" % (label, st.uuid))
        elif card.limit_w is None:
            unknown.append("%s: Limit unlesbar" % label)
        elif abs(float(card.limit_w) - float(st.limit_w)) > POWER_TOLERANCE_W:
            moved.append(
                "%s %s->%s W" % (label, _fmt_w(st.limit_w), _fmt_w(card.limit_w))
            )
    if moved:
        return "veraltet", moved
    if unknown:
        return "unbekannt", unknown
    return "aktuell", []


def _stamps_text(stamps: Sequence[LimitStamp]) -> str:
    return " ".join("%s=%sW" % (st.label, _fmt_w(st.limit_w)) for st in stamps)


def _now_text(stamps: Sequence[LimitStamp], reading: Optional[PowerReading]) -> str:
    live = reading.by_uuid() if reading is not None else {}
    out = []
    for st in stamps:
        card = live.get(st.uuid)
        out.append(
            "%s=%sW"
            % (card.label if card else st.label, _fmt_w(card.limit_w) if card else "?")
        )
    return " ".join(out)


def card_library_reference(
    reading: Optional[PowerReading],
    stage_cards: Sequence[object],
    *,
    library=None,
    path: str = "",
) -> Optional[TimedReference]:
    """Die gemessene Kartenraten-Bibliothek, SO WIE DER SOLVER SIE LIEST.

    ``pp_cut_launch.ms_per_layer_from_card_library`` nimmt je Kartenname die
    erste Variante mit ``gemm_tflops``; ihr eigener Stempel ist ``rate_env``
    (#584, ``plimit_mw``). Hier dieselbe Variante, ihr Limit als
    :class:`LimitStamp` der Karte, auf der die Stufe laeuft. ``None``, wenn es
    keine Bibliothek gibt -- dann liest der Solver sie auch nicht.

    ``stage_cards``: Objekte mit ``uuid``/``nvml_index``/``name`` (der Launcher
    reicht seine ``Card``-Liste in Ordinal-Reihenfolge)."""
    if library is None:
        try:
            from sglang.srt.planner.card_rate_pass import (
                card_library_path,
                load_measured_library,
            )

            library = load_measured_library()
            path = path or card_library_path()
        except Exception:  # noqa: BLE001 -- ohne Bibliothek kein Leser, kein Stempel
            return None
    if library is None:
        return None
    stamps: List[LimitStamp] = []
    for card in stage_cards:
        variant = None
        try:
            for cand in library.variants(str(card.name)) or ():
                if getattr(cand, "gemm_tflops", None):
                    variant = cand
                    break
        except Exception:  # noqa: BLE001
            variant = None
        if variant is None:
            # Der Solver faellt fuer diese Karte auf die Liste zurueck und sagt es.
            return None
        env = RateEnv.parse(getattr(variant, "rate_env", None))
        stamps.append(
            LimitStamp(
                str(card.uuid),
                int(card.nvml_index),
                str(card.name),
                None if env is None else env.power_limit_w,
            )
        )
    return TimedReference(
        name="card_library.json (gemm_tflops, card_rate_pass)",
        source=path or "card_rate_pass.card_library_path()",
        quantity="GEMM-Rate je Kartenname (TFLOP/s)",
        consumer="PP-CUT solver (cost=MEASURED card-rate library) und WEG2 D-OPERATING-POINTS "
        "decode-bs6",
        stamps=tuple(stamps),
        stamp_source="rate_env der Bibliothek (#584, beim Messen von NVML gestempelt)",
        remeasure="`python -m sglang.srt.planner.card_rate_pass --run` (GPU, ~30 s fuer drei "
        "Karten, Fenster buchen)",
    )


def reference_lines(
    reading: Optional[PowerReading],
    references: Sequence[TimedReference] = TIMED_REFERENCES,
    neutral: Sequence[NeutralReference] = LIMIT_NEUTRAL_REFERENCES,
) -> List[str]:
    """Je abweichender Referenz eine Zeile ``REFERENZ VERALTET (Power-Limit
    <Karte alt->neu W>): <Name> ...``, je undatierbarer ``REFERENZ UNDATIERT``,
    zuletzt EINE Summenzeile. Nie eine Verweigerung."""
    lines: List[str] = []
    current: List[str] = []
    for ref in references:
        state, why = stamp_verdict(ref.stamps, reading)
        if state == "aktuell":
            current.append(ref.name)
            continue
        marker = STALE_MARKER if state == "veraltet" else UNDATED_MARKER
        head = (
            "Power-Limit " + ", ".join(why)
            if state == "veraltet"
            else ("Power-Limit unbekannt: " + "; ".join(why))
        )
        lines.append(
            "%s (%s): %s [%s; %s] gemessen bei %s, jetzt %s (Stempel: %s) -- %s; wirkt auf %s. "
            "Keine Verweigerung, VRAM-Riegel unberuehrt."
            % (
                marker,
                head,
                ref.name,
                ref.source,
                ref.quantity,
                _stamps_text(ref.stamps),
                _now_text(ref.stamps, reading),
                ref.stamp_source,
                (
                    (
                        "ihre Zeitwerte beschreiben diese Karten nicht mehr, neu messen"
                        + (": " + ref.remeasure if ref.remeasure else "")
                    )
                    if state == "veraltet"
                    else "ob sie noch gilt, ist nicht pruefbar"
                ),
                ref.consumer,
            )
        )
    lines.append(
        "%s: %d von %d Zeit-/Raten-Referenzen passen zum Limit (%s); limit-neutral, ohne Stempel "
        "(Bytes/PCIe, ihre Riegel unberuehrt): %s"
        % (
            SUMMARY_MARKER,
            len(current),
            len(references),
            ", ".join(current) or "keine",
            ", ".join("%s [%s]" % (n.name, n.gate) for n in neutral),
        )
    )
    return lines


# ---------------------------------------------------------------------------
# 3. Stufenraten und der Schnitt, den sie empfehlen
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageRates:
    """Gemessene Rechenzeit je P-Stufe fuer EIN Power-Limit.

    ``ms_per_chunk[s]``: Median der compute-ms (``gpu-ms`` minus Warten) ueber
    alle VOLLEN Chunks (``#new-token == chunk_tokens``, ``#chunks: 1``) der
    Stufe s in den Referenz-Logs, bei ``stage_layers`` Schichten. Geteilt durch
    die Schichtzahl ist das die Rate je Schicht; noch einmal durch den Chunk die
    Rate je Schicht und Token. ``stamps[s]``: die Karte der Stufe s mit ihrem
    Limit."""

    name: str
    source: str
    model: str
    trees: str
    fr_p: str
    stage_layers: Tuple[int, ...]
    chunk_tokens: int
    ms_per_chunk: Tuple[float, ...]
    n_chunks: Tuple[int, ...]
    stamps: Tuple[LimitStamp, ...]
    stamp_source: str

    def ms_per_layer(self) -> Tuple[float, ...]:
        return tuple(
            float(m) / float(n) for m, n in zip(self.ms_per_chunk, self.stage_layers)
        )

    def us_per_layer_token(self) -> Tuple[float, ...]:
        return tuple(1000.0 * m / float(self.chunk_tokens) for m in self.ms_per_layer())


_RX_RANK_BATCH = re.compile(
    r"\[[^\]]*?PP(\d+)\] Prefill rank batch, #new-token: (\d+), #cached-token: \d+, "
    r"#chunks: (\d+), gpu-ms: [0-9.]+ \(compute ([0-9.]+), wait [0-9.]+\)"
)


def full_chunk_compute_ms(
    texts: Iterable[str], chunk_tokens: int
) -> Dict[int, List[float]]:
    """``{PP-Stufe: [compute-ms je vollem Chunk]}`` aus P-Logs. Teil-Chunks
    (letzter Chunk eines Prompts, Burst-Pakete), zusammengefaltete Zeilen
    (``#chunks`` > 1) und Zeilen ohne compute/wait-Split zaehlen nicht."""
    out: Dict[int, List[float]] = {}
    for text in texts:
        for m in _RX_RANK_BATCH.finditer(text):
            if int(m.group(2)) != int(chunk_tokens) or int(m.group(3)) != 1:
                continue
            out.setdefault(int(m.group(1)), []).append(float(m.group(4)))
    return out


def stage_rates_from_logs(
    p_logs: Sequence[str],
    *,
    name: str,
    source: str,
    model: str,
    trees: str,
    fr_p: str,
    stage_layers: Sequence[int],
    chunk_tokens: int,
    stamps: Sequence[LimitStamp],
    stamp_source: str,
) -> StageRates:
    """Der Satz aus den P-Logs der Referenz-Boots; ``ValueError``, wenn eine
    Stufe keinen vollen Chunk hat (dann gibt es fuer sie keine Rate)."""
    per = full_chunk_compute_ms(p_logs, chunk_tokens)
    n = len(stage_layers)
    if len(stamps) != n:
        raise ValueError("%s: %d Stempel fuer %d Stufen" % (name, len(stamps), n))
    empty = [s for s in range(n) if not per.get(s)]
    if empty:
        raise ValueError(
            "%s: kein voller %d-Token-Chunk auf Stufe(n) %s"
            % (name, chunk_tokens, empty)
        )
    return StageRates(
        name=name,
        source=source,
        model=model,
        trees=trees,
        fr_p=fr_p,
        stage_layers=tuple(int(x) for x in stage_layers),
        chunk_tokens=int(chunk_tokens),
        ms_per_chunk=tuple(round(statistics.median(per[s]), 1) for s in range(n)),
        n_chunks=tuple(len(per[s]) for s in range(n)),
        stamps=tuple(stamps),
        stamp_source=stamp_source,
    )


NF_MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"

#: 400/230 W -- das Limit seit spaetestens 06.09. 22:17Z. Hergeleitet von
#: :func:`stage_rates_from_logs` aus den P-Logs von fnFL2x160 (76ce5580d4,
#: 97k + 259441 Token), fnFL2x162 (7517ff598d) und fnFL2x163 (ce1dac1984, 97k +
#: Burst), alle Schnitt 29,11,8, Chunk 16384; Fixture
#: ``test/registered/unit/weg2/fixtures/power_limit_h57/`` (woertliche
#: 'Prefill rank batch'-Zeilen), der Test bindet die Zahlen daran.
STAGE_RATES_FNFL2 = StageRates(
    name="P_STAGE_RATES_FNFL2",
    source="fnFL2x160 + fnFL2x162 + fnFL2x163",
    model=NF_MODEL,
    trees="76ce5580d4 / 7517ff598d / ce1dac1984",
    fr_p="0.332,0.605,0.39 (x160) / 0.36,0.64,0.39 (x162, x163); PP2 nach dem H25-Draft-Posten 0.734",
    stage_layers=(29, 11, 8),
    chunk_tokens=16384,
    ms_per_chunk=(3868.1, 3810.3, 2303.6),
    n_chunks=(30, 30, 30),
    stamps=rig_stamps(400.0, 230.0),
    stamp_source=STAMP_SOURCE_400_230,
)

#: 525/320 W -- fnFL2x148 (1d95257feb, x146-Form, 97k-Prompt), dieselbe
#: Herleitung. Andere Residenz als der 400/230-Satz (FR_P 0.26/0.45), darum
#: nie gegen ihn verrechnet: jeder Satz empfiehlt nur fuer sein eigenes Limit.
#: ``fr_p`` nennt, wie bei jedem Satz, den ARM-Wert (--pp-cut-expert-device-
#: fraction), damit die Zeile ihn neben den des laufenden Boots stellen kann.
STAGE_RATES_FNFL2_X148 = StageRates(
    name="P_STAGE_RATES_FNFL2_X148",
    source="fnFL2x148",
    model=NF_MODEL,
    trees="1d95257feb",
    fr_p="0.26,0.45,0.39; PP2 nach dem H25-Draft-Posten 0.734",
    stage_layers=(29, 11, 8),
    chunk_tokens=16384,
    ms_per_chunk=(3476.3, 3792.9, 2072.9),
    n_chunks=(5, 5, 5),
    stamps=rig_stamps(525.0, 320.0),
    stamp_source=STAMP_SOURCE_525_320,
)

STAGE_RATES: Tuple[StageRates, ...] = (STAGE_RATES_FNFL2, STAGE_RATES_FNFL2_X148)


def _csv(values: Iterable) -> str:
    return ",".join(str(int(v)) for v in values)


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def attention_per_stage(
    is_full_attention: Sequence[bool], cut: Sequence[int]
) -> Tuple[int, ...]:
    """Full-Attention-Schichten je Stufe eines zusammenhaengenden Schnitts --
    die Zahl, die ``--pp-attn-stage-ratio`` fuer diesen Schnitt nennen muss."""
    out, start = [], 0
    for n in cut:
        out.append(sum(1 for f in is_full_attention[start : start + int(n)] if f))
        start += int(n)
    return tuple(out)


def balanced_cut(
    ms_per_layer: Sequence[float],
    is_full_attention: Sequence[bool],
    pinned: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[float, ...]]:
    """Der zusammenhaengende Schnitt mit dem kleinsten Takt ``max_s(L_s x
    ms_s)``, jede Stufe mit mindestens einer Full-Attention-Schicht (sonst ist
    ihr KV-Pool leer, ``derive_pp_layer_split`` verweigert ihn). Gleichstand:
    der Schnitt, der am wenigsten Schichten vom gepinnten wegbewegt."""
    n_layers, n_stages = len(is_full_attention), len(ms_per_layer)
    best = None
    for cut in _compositions(n_layers, n_stages):
        attn = attention_per_stage(is_full_attention, cut)
        if min(attn) < 1:
            continue
        stage_ms = tuple(float(n) * float(m) for n, m in zip(cut, ms_per_layer))
        moved = (
            sum(abs(a - b) for a, b in zip(cut, pinned))
            if pinned is not None and len(pinned) == n_stages
            else 0
        )
        key = (round(max(stage_ms), 6), moved, cut)
        if best is None or key < best[0]:
            best = (key, cut, attn, stage_ms)
    if best is None:
        raise ValueError(
            "kein Schnitt von %d Schichten auf %d Stufen gibt jeder Stufe "
            "eine Attention-Schicht" % (n_layers, n_stages)
        )
    return best[1], best[2], best[3]


def _stage_limits_text(
    stage_cards: Sequence[object], reading: Optional[PowerReading]
) -> str:
    live = reading.by_uuid() if reading is not None else {}
    out = []
    for card in stage_cards:
        cp = live.get(str(card.uuid))
        out.append(
            "nvml%d=%sW"
            % (int(card.nvml_index), _fmt_w(cp.limit_w) if cp is not None else "?")
        )
    return " ".join(out)


def rates_for(
    reading: Optional[PowerReading],
    stage_cards: Sequence[object],
    model_name: str,
    rates: Sequence[StageRates] = STAGE_RATES,
) -> Optional[StageRates]:
    """Der Satz dieses Modells, dessen Stufe s auf DERSELBEN Karte (UUID) unter
    DEMSELBEN Limit lief wie hier; sonst ``None``."""
    uuids = [str(c.uuid) for c in stage_cards]
    for r in rates:
        if r.model != model_name or len(r.stamps) != len(uuids):
            continue
        if [st.uuid for st in r.stamps] != uuids:
            continue
        if stamp_verdict(r.stamps, reading)[0] == "aktuell":
            return r
    return None


def rate_cut_line(
    reading: Optional[PowerReading],
    *,
    stage_cards: Sequence[object],
    model_name: str,
    is_full_attention: Sequence[bool],
    pinned: Optional[Sequence[int]],
    chunk_tokens: int,
    fr_p: str = "",
    rates: Sequence[StageRates] = STAGE_RATES,
) -> str:
    """Die EINE Zeile ``PP-CUT RATEN-SCHNITT``: der Schnitt aus gemessenen
    Stufenraten fuer das AKTUELLE Power-Limit, oder ``keine Raten fuer
    <Limit>, Schnitt bleibt gepinnt``. Rechnet nur; wer die Zeile liest,
    entscheidet -- nichts hier aendert den Schnitt, den der Boot faehrt."""
    head = "%s (H57)" % CUT_MARKER
    pin_txt = _csv(pinned) if pinned else "(keiner, der Solver waehlt)"
    stay = (
        "Schnitt bleibt gepinnt (%s)" % pin_txt
        if pinned
        else "Schnitt bleibt beim Solver"
    )
    own = [r for r in rates if r.model == model_name]
    if not own:
        return (
            "%s ENTFAELLT: die Stufenraten sind auf %s gemessen, dieser Boot faehrt %s; %s."
            % (
                head,
                ", ".join(sorted({r.model for r in rates})) or "nichts",
                model_name,
                stay,
            )
        )
    now = _stage_limits_text(stage_cards, reading)
    match = rates_for(reading, stage_cards, model_name, own)
    if match is None:
        why = (
            " (Power-Limit unbekannt: %s)"
            % ((reading.error if reading is not None else "") or "keine Lesung")
            if reading is None or not reading.known
            else ""
        )
        return (
            "%s: keine Raten fuer %s%s, %s. Gemessen sind: %s. Raten fuer ein neues Limit: "
            "einen Boot mit gepinntem Schnitt fahren, dann `python -m "
            "sglang.srt.planner.power_limit --p-log <P.log> --launch-log <front.log>` und den "
            "Satz in power_limit.STAGE_RATES eintragen."
            % (
                head,
                now,
                why,
                stay,
                "; ".join(
                    "%s bei %s (%s)" % (r.name, _stamps_text(r.stamps), r.source)
                    for r in own
                ),
            )
        )
    per_layer = match.ms_per_layer()
    per_token = match.us_per_layer_token()
    try:
        cut, attn, stage_ms = balanced_cut(per_layer, is_full_attention, pinned)
    except ValueError as exc:
        # Eine Zeile rechnet, sie verweigert nie: ohne realisierbaren Schnitt
        # entfaellt die Empfehlung mit Grund.
        return "%s ENTFAELLT: %s; %s." % (head, exc, stay)
    takt = max(stage_ms)
    if pinned and len(pinned) == len(per_layer):
        pin_ms = [float(n) * m for n, m in zip(pinned, per_layer)]
        pin_attn = attention_per_stage(is_full_attention, pinned)
        pin_txt_full = "gepinnt %s attn %s -> %s ms, Takt %.0f ms" % (
            _csv(pinned),
            _csv(pin_attn),
            "/".join("%.0f" % x for x in pin_ms),
            max(pin_ms),
        )
        delta = (
            " (%+.1f %%)" % (100.0 * (takt - max(pin_ms)) / max(pin_ms))
            if max(pin_ms)
            else ""
        )
    else:
        pin_txt_full, delta = "kein gepinnter Schnitt", ""
    verdict = (
        "empfohlen = gepinnt: der Schnitt ist fuer dieses Limit balanciert"
        if pinned and tuple(int(x) for x in pinned) == tuple(cut)
        else "NICHT angewendet (kein Umschneiden): der Schnitt bleibt die Arm-Variable PP_RATIO "
        "(--pp-stage-ratio); uebernehmen mit PP_RATIO=%s PP_ATTN_RATIO=%s (PP_ATTN_RATIO "
        "ist keine zweite Wahl, sondern die Attention-Zahl dieses Schnitts -- ohne sie "
        "folgt die W40-Verweigerung des Paars)" % (_csv(cut), _csv(attn))
    )
    chunk_note = (
        ""
        if int(chunk_tokens) == int(match.chunk_tokens)
        else " Dieser Boot chunkt %d statt %d: das Stufenverhaeltnis gilt naeherungsweise, der "
        "Sockel je Forward nicht." % (int(chunk_tokens), int(match.chunk_tokens))
    )
    return (
        "%s Power-Limit %s = Raten %s (%s, Schnitt %s, Chunk %d, FR_P %s): voller Chunk compute "
        "%s ms (Median ueber %s) = %s ms je Schicht = %s us je Schicht und Token; %s; empfohlen "
        "%s attn %s -> %s ms, Takt %.0f ms%s. %s. Grenze: die Raten gelten bei der Residenz der "
        "Referenz (dieser Boot FR_P %s); mehr Schichten auf einer Stufe senken dort die "
        "Experten-Residenz -- ob die Karten den Schnitt tragen, sagt der Dry-Run mit dem neuen "
        "PP_RATIO (#140, W132, W126), nicht diese Zeile.%s"
        % (
            head,
            now,
            match.name,
            match.source,
            _csv(match.stage_layers),
            match.chunk_tokens,
            match.fr_p,
            "/".join("%.1f" % x for x in match.ms_per_chunk),
            "/".join(str(n) for n in match.n_chunks),
            "/".join("%.1f" % x for x in per_layer),
            "/".join("%.2f" % x for x in per_token),
            pin_txt_full,
            _csv(cut),
            _csv(attn),
            "/".join("%.0f" % x for x in stage_ms),
            takt,
            delta,
            verdict,
            fr_p or "?",
            chunk_note,
        )
    )


# ---------------------------------------------------------------------------
# 4. neue Raten aus einem Boot (CLI)
# ---------------------------------------------------------------------------


def stage_rates_from_boot(
    p_log_text: str,
    launch_log_text: str,
    *,
    name: str,
    source: str,
    model: str = NF_MODEL,
    trees: str = "?",
    fr_p: str = "?",
    stage_layers: Optional[Sequence[int]] = None,
    chunk_tokens: int = 16384,
) -> StageRates:
    """Ein Satz aus EINEM Boot ab H57: Raten aus dem P-Log, Stempel aus der
    ``POWER-LIMIT``- und der ``NVML -> CUDA ordinal map``-Zeile seines
    Launcher-Logs (``<base>.front.log``). ``ValueError`` ohne POWER-LIMIT-Zeile:
    ein Boot, der sein Limit nicht schrieb, stempelt keine Rate."""
    limits = limits_from_launch_log(launch_log_text)
    cards = ordinal_cards_from_launch_log(launch_log_text)
    if not limits or not cards:
        raise ValueError(
            "%s: Launcher-Log ohne POWER-LIMIT- oder NVML-Ordinal-Zeile -- der Boot ist "
            "aelter als H57, sein Limit ist nicht belegt" % name
        )
    if stage_layers is None:
        m = re.search(r"REALIZED layer split \[([0-9, ]+)\]", launch_log_text)
        if m is None:
            raise ValueError(
                "%s: kein 'REALIZED layer split' im Launcher-Log; --layers angeben"
                % name
            )
        stage_layers = [int(x) for x in m.group(1).split(",")]
    missing = [n for _u, n, _name in cards if n not in limits]
    if missing:
        raise ValueError(
            "%s: POWER-LIMIT-Zeile ohne Limit fuer nvml%s" % (name, missing)
        )
    stamps = [LimitStamp(u, n, nm, limits[n]) for u, n, nm in cards]
    return stage_rates_from_logs(
        [p_log_text],
        name=name,
        source=source,
        model=model,
        trees=trees,
        fr_p=fr_p,
        stage_layers=stage_layers,
        chunk_tokens=chunk_tokens,
        stamps=stamps,
        stamp_source="POWER-LIMIT-Zeile des Boots (NVML, H57)",
    )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m sglang.srt.planner.power_limit",
        description="Stufenraten eines Boots (ab H57) als StageRates-Satz drucken.",
    )
    ap.add_argument("--p-log", required=True, help="P-Log des Boots (.P.log)")
    ap.add_argument(
        "--launch-log", required=True, help="Launcher-Log desselben Boots (.front.log)"
    )
    ap.add_argument("--name", default="P_STAGE_RATES_NEU")
    ap.add_argument("--source", default="")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument(
        "--layers", default="", help="Schnitt, sonst aus 'REALIZED layer split'"
    )
    ap.add_argument("--fr-p", default="?")
    ap.add_argument("--trees", default="?")
    args = ap.parse_args(list(argv) if argv is not None else None)
    with open(args.p_log, errors="replace") as fh:
        p_text = fh.read()
    with open(args.launch_log, errors="replace") as fh:
        l_text = fh.read()
    rates = stage_rates_from_boot(
        p_text,
        l_text,
        name=args.name,
        source=args.source or os.path.basename(args.p_log),
        trees=args.trees,
        fr_p=args.fr_p,
        chunk_tokens=args.chunk,
        stage_layers=[int(x) for x in args.layers.split(",")] if args.layers else None,
    )
    print(repr(rates))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
