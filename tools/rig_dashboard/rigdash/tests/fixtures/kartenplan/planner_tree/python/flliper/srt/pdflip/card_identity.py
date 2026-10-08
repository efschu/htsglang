"""HW-GENERIC 1002: card identity, the arch gate and the card order -- from
NVML PROPERTIES, never from a name substring.

User order 02.10. (verbatim): "die software soll fuer jede hardware laufen ...
aktuell halt 'nur' sm86 und sm120 - bevor du sm89 hinzufuegst, mach erstmal
dass sm86 und sm120 generell unterstuetzt wird".

What this module replaces (f7099c0cbd / 3fe878018d):

* ``launcher.order_cards`` sorted by ``"5090" in name`` / ``"3080" in name``
  and refused every other inventory -- a 10 GB RTX 3080 passed (and borrowed
  the 20 GB card's measurements), a 3090 or an A6000 was refused for its NAME.
* ``"5090" in c.name`` / ``"3080" in c.name`` selectors for measured numbers
  (W19 ``dc_measured_d_mib``, ``DC_EXPECT_*``, the xchg census constants, the
  P-cut attention anchor stage).

THE THREE NOTIONS, kept apart on purpose:

1. :func:`card_key` -- what the card IS: model + NVML total + compute
   capability (``"RTX3090/24576MiB/sm86"``). Two boards with one key are
   the same calibration subject. Model is the NVML name with the vendor
   words dropped, compared EXACTLY (no substring: "A10" is not "A100").
2. :func:`calibration_class` -- which MEASURED class the card belongs to, or
   None. Today two classes exist, both measured on the reference rig, under
   the labels every record file already uses (``RTX5090``, ``RTX3080``).
   Membership needs the model, the arch AND the VRAM tier (5 %, the same
   band as ``planner.flags._CALIBRATED_RIG_TOTAL_TOLERANCE``): a stock 10 GB
   RTX 3080 is NOT the 20 GB class and gets no 20 GB number.
3. :func:`class_label` -- the label a record lookup uses: the calibration
   class when there is one, else the card key (which then matches no record:
   UNCALIBRATED by name, never a borrow).

THE ORDER (:func:`order_cards`): biggest NVML total first, then the higher
nameplate DRAM bandwidth (NVML bus width x max memory clock), then NVML
index. On the reference rig (nvml0 RTX 3080 20480, nvml1 RTX 5090 32607,
nvml2 RTX 3080 20480) that is nvml1, nvml0, nvml2 -- exactly the old
``order_cards`` ("the 5090 first, then the 3080s by NVML index").

THE ARCH GATE (:func:`arch_gate`): sm_86 and sm_120 only (the wheel arch
list ``86;120a`` and the JIT prebuild ``8.6,12.0``); every other compute
capability -- sm_89 explicitly (user: "bevor du sm89 hinzufuegst") -- and an
UNREPORTED one are refused BY NAME, per card.

PURE: stdlib only (launcher, entrypoint CLI and desk tests import it).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

#: The compute capabilities the release image carries code for. sm_89 is OUT
#: of scope (user order 02.10.); sm_90/sm_100 have no cubins in the wheel.
SUPPORTED_ARCHS: Tuple[Tuple[int, int], ...] = ((8, 6), (12, 0))

#: VRAM-tier band for class membership (same 5 % as planner.flags).
TOTAL_TOLERANCE = 0.05

#: Refusal codes, printed at the head of every message.
CODE_ARCH = "HW-ARCH"
CODE_COUNT = "HW-COUNT"
CODE_UNCALIBRATED = "HW-UNCALIBRATED"


class CardInventoryRefused(RuntimeError):
    """The inventory cannot run this release at all (arch, count)."""


class CardUncalibrated(RuntimeError):
    """The inventory could run, but a value the plan needs was measured on
    other cards. Named, with what to measure -- never a borrowed number."""


@dataclass(frozen=True)
class CalibratedClass:
    """A card class with measured records: model + arch + VRAM tier."""

    label: str
    model: str
    cc: Tuple[int, int]
    total_mib: int


#: The classes the release records were measured on (reference rig, NVML
#: totals measured there). The labels are the keys every record file, power
#: record and p_stage_model JSON already uses -- kept, so the reference rig's
#: lookups are byte-identical.
CALIBRATED_CLASSES: Tuple[CalibratedClass, ...] = (
    CalibratedClass("RTX5090", "RTX 5090", (12, 0), 32607),
    CalibratedClass("RTX3080", "RTX 3080", (8, 6), 20480),
)

#: The reference rig's inventory in card order (the records' positional
#: vectors are measured in this order).
REFERENCE_INVENTORY: Tuple[str, ...] = ("RTX5090", "RTX3080", "RTX3080")


@dataclass(frozen=True)
class CardProps:
    """The properties identity and order are decided on (one card)."""

    nvml_index: int
    uuid: str
    name: str
    total_mib: int
    cc: Optional[Tuple[int, int]] = None
    bar1_total_mib: Optional[int] = None
    pcie_max_gen: Optional[int] = None
    pcie_max_width: Optional[int] = None
    mem_bus_width_bits: Optional[int] = None
    mem_clock_max_mhz: Optional[int] = None

    @property
    def peak_membw_gbps(self) -> Optional[float]:
        """Nameplate DRAM bandwidth (GB/s) from NVML bus width x max memory
        clock x 2 (DDR); None when NVML did not report both. Only a sort
        tie-breaker -- never a rate the planner prices with."""
        if not self.mem_bus_width_bits or not self.mem_clock_max_mhz:
            return None
        return self.mem_bus_width_bits / 8.0 * self.mem_clock_max_mhz * 2.0 / 1000.0


def _get(obj, key, default=None):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _opt_int(v) -> Optional[int]:
    return None if v is None else int(v)


def props_of(card) -> CardProps:
    """:class:`CardProps` of a launcher ``Card``, a registry ``DeviceInfo``,
    a planner descriptor or a plain dict (duck-typed: the fields exist under
    the same names on all of them)."""
    if isinstance(card, CardProps):
        return card
    cc = _get(card, "cc")
    if cc is None:
        maj, mnr = _get(card, "cc_major"), _get(card, "cc_minor")
        cc = None if maj is None or mnr is None else (int(maj), int(mnr))
    else:
        cc = (int(cc[0]), int(cc[1]))
    total_mib = _get(card, "total_mib")
    if total_mib is None and _get(card, "total_bytes") is not None:
        total_mib = int(_get(card, "total_bytes")) // (1024 * 1024)
    bar1 = _get(card, "bar1_total_mib")
    if bar1 is None and _get(card, "bar1_total_bytes") is not None:
        bar1 = int(_get(card, "bar1_total_bytes")) // (1024 * 1024)
    idx = _get(card, "nvml_index")
    if idx is None:
        idx = _get(card, "index", -1)
    return CardProps(
        nvml_index=int(idx),
        uuid=str(_get(card, "uuid", "") or ""),
        name=str(_get(card, "name", "") or ""),
        total_mib=int(total_mib or 0),
        cc=cc,
        bar1_total_mib=_opt_int(bar1),
        pcie_max_gen=_opt_int(_get(card, "pcie_max_gen")),
        pcie_max_width=_opt_int(_get(card, "pcie_max_width")),
        mem_bus_width_bits=_opt_int(_get(card, "mem_bus_width_bits")),
        mem_clock_max_mhz=_opt_int(_get(card, "mem_clock_max_mhz")),
    )


_VENDOR_WORDS = re.compile(r"\b(NVIDIA|GeForce)\b", re.IGNORECASE)


def model_name(name: str) -> str:
    """'NVIDIA GeForce RTX 5090' -> 'RTX 5090'; 'NVIDIA RTX A6000' ->
    'RTX A6000'. Vendor words dropped, whitespace collapsed -- the rest is
    compared EXACTLY (case-insensitive), never as a substring."""
    return " ".join(_VENDOR_WORDS.sub(" ", str(name or "")).split())


def _sm(cc: Optional[Tuple[int, int]]) -> str:
    return "sm?" if cc is None else f"sm{cc[0]}{cc[1]}"


def card_key(card) -> str:
    """What the card IS: ``"<model>/<total>MiB/sm<cc>"``, spaces dropped from
    the model so the key is one log token (``card=RTX3090/24576MiB/sm86``)."""
    p = props_of(card)
    return f"{model_name(p.name).replace(' ', '')}/{p.total_mib}MiB/{_sm(p.cc)}"


def calibration_class(card) -> Optional[str]:
    """The measured class label of this card, or None (uncalibrated).

    Model (exact), compute capability (exact) and NVML total (within
    :data:`TOTAL_TOLERANCE`) must all agree. A card WITHOUT a reported cc
    (or total) never reached a boot (:func:`arch_gate` refuses it at
    ``resolve_cards``; NVML always states the total); it is a hand-built
    object (desk test, offline tool) and is matched on what it states --
    the model always exactly, never as a substring."""
    p = props_of(card)
    model = model_name(p.name).lower()
    for cls in CALIBRATED_CLASSES:
        if (model == cls.model.lower()
                and (p.cc is None or tuple(p.cc) == tuple(cls.cc))
                # total 0 = not stated (a hand-built stand-in; NVML always states it)
                and (not p.total_mib
                     or abs(p.total_mib - cls.total_mib) <= cls.total_mib * TOTAL_TOLERANCE)):
            return cls.label
    return None


def class_label(card) -> str:
    """The record-lookup label: the calibration class, else :func:`card_key`."""
    return calibration_class(card) or card_key(card)


def describe(card) -> str:
    p = props_of(card)
    bw = p.peak_membw_gbps
    return (f"nvml{p.nvml_index} {p.name} {p.total_mib} MiB {_sm(p.cc)}"
            + (f" {bw:.0f} GB/s" if bw else "")
            + (f" PCIe gen{p.pcie_max_gen} x{p.pcie_max_width}" if p.pcie_max_gen else "")
            + (f" BAR1 {p.bar1_total_mib} MiB" if p.bar1_total_mib else "")
            + f" class={class_label(card)}")


# ---------------------------------------------------------------------------
# the gate and the order


def arch_gate(cards: Iterable) -> None:
    """Refuse BY NAME every card whose compute capability is not in
    :data:`SUPPORTED_ARCHS` (or was not reported)."""
    bad = []
    for c in cards:
        p = props_of(c)
        if p.cc is None:
            bad.append(f"nvml{p.nvml_index} {p.name!r}: compute capability not reported by NVML "
                       "(cannot verify the arch -- refusing rather than guessing from the name)")
        elif tuple(p.cc) not in SUPPORTED_ARCHS:
            bad.append(f"nvml{p.nvml_index} {p.name!r}: compute capability "
                       f"{p.cc[0]}.{p.cc[1]} ({_sm(p.cc)})")
    if bad:
        raise CardInventoryRefused(
            f"{CODE_ARCH}: this release carries kernels for "
            + " and ".join(f"sm_{a}{b}" for a, b in SUPPORTED_ARCHS)
            + " only (sgl-kernel wheel 86;120a, JIT prebuild 8.6,12.0); refused: "
            + "; ".join(bad)
            + ". sm_89 is out of scope until sm_86/sm_120 are generic (user order 02.10.).")


def order_key(card) -> Tuple[int, float, int]:
    """Biggest NVML total first, then higher nameplate DRAM bandwidth, then
    NVML index (stable, so identical boards keep NVML order)."""
    p = props_of(card)
    return (-int(p.total_mib), -float(p.peak_membw_gbps or 0.0), int(p.nvml_index))


def order_cards(cards: Sequence, expect_count: Optional[int] = None,
                gate: bool = True) -> List:
    """The cards in CUDA-ordinal order (rank 0 / PP0 / TP0 first), after the
    arch gate (``gate``; the launcher gates once at ``resolve_cards``) and --
    when ``expect_count`` is given -- the count check. Returns the SAME
    objects it was given (sorted), never copies."""
    cards = list(cards)
    if gate:
        arch_gate(cards)
    if expect_count is not None and len(cards) != int(expect_count):
        raise CardInventoryRefused(
            f"{CODE_COUNT}: {len(cards)} card(s) visible, this launch needs exactly "
            f"{int(expect_count)} (P = PP{int(expect_count)}, D = TP{int(expect_count)}); visible: "
            + "; ".join(describe(c) for c in cards)
            + ". Restrict the container to the cards it should use (--gpus / NVIDIA_VISIBLE_DEVICES).")
    return sorted(cards, key=order_key)


def inventory_signature(ordered_cards: Sequence) -> Tuple[str, ...]:
    """The class labels of the ordered cards -- what a positional vector or a
    positional record is keyed by."""
    return tuple(class_label(c) for c in ordered_cards)


def parse_inventory(text: Optional[str]) -> Optional[Tuple[str, ...]]:
    """``"RTX5090,RTX3080,RTX3080"`` -> tuple; empty/None -> None."""
    t = str(text or "").strip()
    if not t:
        return None
    return tuple(x.strip() for x in t.split(",") if x.strip())


def uncalibrated_message(ordered_cards: Sequence, calibrated: Sequence[str],
                         what: Sequence[str], source: str) -> Optional[str]:
    """None when the live ordered inventory IS the calibrated one; else the
    named HW-UNCALIBRATED message: which ordinal differs, which values are
    positional measurements of the other inventory, and how to measure them."""
    live = inventory_signature(ordered_cards)
    want = tuple(calibrated)
    if live == want:
        return None
    diff = []
    for i in range(max(len(live), len(want))):
        a = live[i] if i < len(live) else "-"
        b = want[i] if i < len(want) else "-"
        if a != b:
            diff.append(f"ordinal {i}: live {a} vs calibrated {b}")
    return (f"{CODE_UNCALIBRATED}: {source} was measured on the inventory "
            f"[{', '.join(want)}] (card order), this rig is [{', '.join(live)}] ("
            + "; ".join(diff) + "). Positional measurements of the other inventory are NOT "
            "borrowed: " + (", ".join(what) if what else "(none named)")
            + ". Measure them on this rig: card_rate_pass --run (GEMM/membw/link rates), then "
            "one calibration boot per profile that writes the per-rank records "
            "(weg2_measured_record.json, profile_records_data/<profile>.json with this "
            "inventory), and a profile whose vectors name this inventory "
            "(--profile-inventory).")


def check_calibrated(ordered_cards: Sequence, calibrated: Sequence[str],
                     what: Sequence[str], source: str) -> None:
    """Raise :class:`CardUncalibrated` with :func:`uncalibrated_message`."""
    msg = uncalibrated_message(ordered_cards, calibrated, what, source)
    if msg is not None:
        raise CardUncalibrated(msg)


# ---------------------------------------------------------------------------
# CLI (the entrypoint's gate)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m flliper.srt.pdflip.card_identity",
        description="Arch gate + card count + card order from NVML properties "
                    "(entrypoint preflight; honours FLLIPER_NVML_REPLAY_JSON).")
    ap.add_argument("--expect-count", type=int, default=None)
    ap.add_argument("--inventory", default="",
                    help="the profile's calibrated inventory (class labels in card order); "
                         "a mismatch prints HW-UNCALIBRATED and exits 4")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    from flliper.srt.registry import nvml as _nvml

    devs = _nvml.list_devices()
    try:
        ordered = order_cards(devs, ns.expect_count)
    except CardInventoryRefused as exc:
        print(f"refuse {exc}", flush=True)
        return 3
    rows = [{"ordinal": i, "nvml_index": props_of(c).nvml_index, "key": card_key(c),
             "class": class_label(c), "describe": describe(c)} for i, c in enumerate(ordered)]
    if ns.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"ordinal {r['ordinal']}: {r['describe']}")
    want = parse_inventory(ns.inventory)
    if want is not None:
        msg = uncalibrated_message(ordered, want, (), "the profile")
        if msg is not None:
            print(msg, flush=True)
            return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
