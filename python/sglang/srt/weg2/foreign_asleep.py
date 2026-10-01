# SPDX-License-Identifier: Apache-2.0
"""Dual-model: ``foreign_asleep`` -- the other model's sleeping groups per card.

Each stack's D budget already charges its own sleeping group
(``budget D = total - corridor - dormant_other - awake_overshoot``,
``dormant_residue.py``). With two models booted, every card also carries the
OTHER model's two groups, both asleep (DUAL-MODEL-FLIP-KONZEPT-1001.md section
8.4: +1.4-2.8 GiB per 3080, +2.9-3.9 GiB on the 5090 at today's residues).

The planners stay separate per model (user 24.09.: NF values do not fit the
27B and vice versa). Exactly one number crosses: this term, read from the
other stack's ``vram_plan.json`` (its ``asleep.P`` / ``asleep.D`` per NVML
UUID -- the one card identity both containers share; ordinals may differ).

Rules, all refusing by name rather than charging zero:

* a card of ours the other plan does not list -> refuse (it would be
  charged nothing, the unsafe direction for a capacity term);
* a foreign group missing a card -> refuse;
* a foreign schema -> refuse.

``growth_mib`` adds the served growth per group (the residue grows after the
first sleep, ``dormant_residue.served_growth``). ``deep_floor_mib`` -- the
measured per-process floor of a model in deep sleep -- replaces both groups'
residue (2 x floor per card) once the deep sleep exists and is measured.
Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from sglang.srt.weg2.vram_plan import SCHEMA as VRAM_PLAN_SCHEMA

GROUPS = ("P", "D")


class ForeignAsleepRefused(ValueError):
    """The foreign term cannot be priced; never silently zero."""


@dataclass(frozen=True)
class ForeignAsleep:
    mib: Dict[str, int]
    provenance: str


def foreign_asleep_mib(
    other_plan: Optional[Mapping],
    *,
    own_uuids: Sequence[str],
    growth_mib: Optional[Mapping[str, Mapping[str, int]]] = None,
    deep_floor_mib: Optional[Mapping[str, int]] = None,
) -> ForeignAsleep:
    if other_plan is None:
        return ForeignAsleep({u: 0 for u in own_uuids}, "NONE single-model")
    if other_plan.get("schema") != VRAM_PLAN_SCHEMA:
        raise ForeignAsleepRefused(
            f"foreign vram plan schema {other_plan.get('schema')!r} != {VRAM_PLAN_SCHEMA!r}")
    boot = str(other_plan.get("boot_id") or "?")
    known = {str(c.get("uuid")) for c in other_plan.get("cards", [])}
    missing = [u for u in own_uuids if u not in known]
    if missing:
        raise ForeignAsleepRefused(
            f"foreign plan {boot} does not list card(s) {missing}: they would be charged nothing")
    if deep_floor_mib is not None:
        lack = [u for u in own_uuids if u not in deep_floor_mib]
        if lack:
            raise ForeignAsleepRefused(f"deep-sleep floor has no value for card(s) {lack}")
        return ForeignAsleep({u: 2 * int(deep_floor_mib[u]) for u in own_uuids},
                             f"MEASURED deep-sleep floor x 2 groups (foreign boot {boot})")
    asleep = other_plan.get("asleep") or {}
    out: Dict[str, int] = {}
    for u in own_uuids:
        total = 0
        for g in GROUPS:
            per = asleep.get(g) or {}
            if u not in per:
                raise ForeignAsleepRefused(
                    f"foreign plan {boot}: group {g} has no asleep residue for card {u}")
            total += int(per[u])
            if growth_mib is not None:
                total += int((growth_mib.get(g) or {}).get(u, 0))
        out[u] = total
    prov = f"RECORD foreign asleep P+D (boot {boot})" + (" + served growth" if growth_mib else "")
    return ForeignAsleep(out, prov)
