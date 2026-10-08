# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Manual drafter selection and per-request task routing (#309).

Two things the #156 machinery does not give you today, both pure:

* **Manual selection.** #156's controller picks the arm; #309 makes that ONE
  source among several. An operator can pin an arm, a request can carry a tag
  that routes to one, and the controller keeps the rest. The precedence is
  explicit (:func:`resolve_selection`) rather than emergent, because "who won"
  is the first question anyone asks of a run.
* **Task routing.** A request tag (``code`` / ``prose`` / ``multiturn`` / any
  string a deployment invents) maps to a rung. The canonical case is the #156
  measurement that DFLASH is the WORST arm on multiturn -- a fact about a
  workload, not about the code, so the mapping is CONFIG and this module has no
  built-in tag names.

RUNG VOCABULARY IS REUSED, NOT REDEFINED. A rung is the existing
``(family, value)`` pair from ``cross_algo_utils`` -- ``("nextn", k)`` or
``("dflash", block)`` -- and a routed rung is validated against the SAME
configured arm set that ``resolve_drafter_policy_table`` validates against. A
second spelling of "which drafter" is how the two tables drift apart and a
routing entry silently selects an arm that was never loaded.

THE AXES ARE DIFFERENT AND BOTH ARE KEPT. The existing policy table is keyed by
CONTEXT LENGTH (``start_ctx -> rung``); routing is keyed by TAG. They compose:
a tag selects a policy, the policy still resolves by context. Collapsing them
would force a deployment to enumerate tag x context, which is the table nobody
maintains.

NO SILENT FALLBACK, anywhere. An unknown tag, an unconfigured arm, a manual pin
to an algorithm this server did not load -- each is a named error. A routing
miss that quietly runs the default arm is indistinguishable from routing that
works, which is the whole failure class this fork keeps paying for.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict, List, Optional, Tuple

#: A rung, exactly as ``cross_algo_utils`` spells it.
Rung = Tuple[str, int]

#: Families a rung may name. Kept in step with resolve_drafter_policy_table's
#: check; a family that is legal there and unknown here would let a routing
#: table name an arm the policy table would have refused.
KNOWN_FAMILIES = ("nextn", "dflash")


class SelectionSource(str, enum.Enum):
    """Who chose the active rung. Reported with every selection.

    The point of naming the source is that #156's controller is no longer
    privileged: an operator reading a log line needs to know whether the arm
    they see is theirs, the router's, or the bandit's.
    """

    #: The boot configuration; nothing overrode it.
    BOOT = "boot"
    #: An operator pinned it through the selection endpoint.
    MANUAL = "manual"
    #: A request tag routed to it.
    ROUTED = "routed"
    #: The #156 adaptive controller chose it.
    CONTROLLER = "controller"


class SelectionError(ValueError):
    """A selection that cannot be honoured. Always names what is available."""


@dataclasses.dataclass(frozen=True)
class Selection:
    """The resolved answer: which rung, and who chose it."""

    rung: Rung
    source: SelectionSource
    detail: str = ""

    @property
    def family(self) -> str:
        return self.rung[0]

    @property
    def value(self) -> int:
        return self.rung[1]

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "value": self.value,
            "source": self.source.value,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class ArmSet:
    """What this server actually loaded and can therefore switch to.

    A selection is validated against THIS, not against what is spellable. The
    difference matters: ``nextn:7`` is a perfectly well-formed rung and a
    server that never loaded a k=7 arm cannot serve it, and saying so is the
    difference between a named refusal and a silent downgrade.
    """

    #: NEXTN k values with a loaded arm.
    nextn_ks: Tuple[int, ...] = ()
    #: The resident DFLASH block size, or None when no DFLASH arm is loaded.
    dflash_block: Optional[int] = None

    def describe(self) -> str:
        parts = [f"nextn:{k}" for k in sorted(self.nextn_ks)]
        if self.dflash_block is not None:
            parts.append(f"dflash:{self.dflash_block}")
        return ", ".join(parts) if parts else "(no arms loaded)"

    def validate(self, rung: Rung, *, what: str) -> None:
        family, value = rung
        if family not in KNOWN_FAMILIES:
            raise SelectionError(
                f"{what}: unknown drafter family {family!r}; known families "
                f"are {', '.join(KNOWN_FAMILIES)}"
            )
        if family == "nextn":
            if value not in self.nextn_ks:
                raise SelectionError(
                    f"{what}: nextn k={value} is not a loaded arm on this "
                    f"server. Available: {self.describe()}. Widen the arm set "
                    "at boot (--speculative-adaptive-config) or attach a "
                    "drafter that provides it."
                )
        elif family == "dflash":
            if self.dflash_block is None:
                raise SelectionError(
                    f"{what}: no DFLASH arm is loaded on this server. "
                    f"Available: {self.describe()}."
                )
            if value != self.dflash_block:
                raise SelectionError(
                    f"{what}: the resident DFLASH rung has block size "
                    f"{self.dflash_block}; dflash:{value} is not a loaded arm."
                )


def parse_rung(raw: str, *, what: str = "selection") -> Rung:
    """``'nextn:3'`` / ``'dflash:16'`` -> a rung. Loud on anything else."""
    text = str(raw).strip().lower()
    if ":" not in text:
        raise SelectionError(
            f"{what}: expected 'family:value' (e.g. 'nextn:3', 'dflash:16'), "
            f"got {raw!r}"
        )
    family, _, value = text.partition(":")
    family = family.strip()
    if family not in KNOWN_FAMILIES:
        raise SelectionError(
            f"{what}: unknown drafter family {family!r}; known families are "
            f"{', '.join(KNOWN_FAMILIES)}"
        )
    try:
        return (family, int(value.strip()))
    except ValueError:
        raise SelectionError(
            f"{what}: value must be an integer, got {value.strip()!r}"
        ) from None


#: tag -> rung.
RoutingTable = Dict[str, Rung]


def parse_routing_table(raw: Optional[str], *, arms: ArmSet) -> RoutingTable:
    """``'code=nextn:3,multiturn=nextn:5'`` -> a validated table.

    No built-in tags: which tags a deployment uses is a fact about its traffic.
    The canonical entry the #156 work paid for is keeping multiturn OFF DFLASH,
    and that belongs in a config line an operator can read and change, not in
    an ``if tag == "multiturn"`` somebody has to find.

    Every rung is validated against the loaded arm set at PARSE time, so a
    routing miss cannot first appear as a request that quietly ran the wrong
    arm.
    """
    table: RoutingTable = {}
    if raw is None or not str(raw).strip():
        return table
    for entry in str(raw).split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SelectionError(
                f"routing entry {entry!r}: expected 'tag=family:value', "
                "e.g. 'multiturn=nextn:5'"
            )
        tag, _, rung_raw = entry.partition("=")
        tag = tag.strip().lower()
        if not tag:
            raise SelectionError(f"routing entry {entry!r}: empty tag")
        if tag in table:
            raise SelectionError(
                f"routing entry {entry!r}: tag {tag!r} is listed twice; one "
                "of the two would silently win"
            )
        rung = parse_rung(rung_raw, what=f"routing entry {entry!r}")
        arms.validate(rung, what=f"routing entry {entry!r}")
        table[tag] = rung
    return table


def resolve_selection(
    *,
    arms: ArmSet,
    boot_rung: Rung,
    manual_rung: Optional[Rung] = None,
    request_tag: Optional[str] = None,
    routing: Optional[RoutingTable] = None,
    controller_rung: Optional[Rung] = None,
    strict_tags: bool = False,
) -> Selection:
    """Resolve the active rung for one request. THE precedence, in one place.

    Order, highest first, and the reason for each rank:

    1. **MANUAL** -- an operator pinned it. A pin that a controller could
       override is not a pin; the whole point of the endpoint is to take the
       server off automatic.
    2. **ROUTED** -- the request carries a tag with an entry. Per-request
       beats per-server automatic because the tag is a statement about THIS
       request's workload, which the controller cannot see.
    3. **CONTROLLER** -- #156's adaptive choice, now one source among several.
    4. **BOOT** -- the configured default.

    ``strict_tags``: when True an unknown tag is an error rather than a
    fall-through to the controller. Off by default, because a deployment that
    tags some traffic and not the rest is normal; on when a deployment wants
    to know that its tagger and its table have drifted apart.
    """
    if manual_rung is not None:
        arms.validate(manual_rung, what="manual selection")
        return Selection(manual_rung, SelectionSource.MANUAL, "operator pin")

    if request_tag:
        tag = str(request_tag).strip().lower()
        table = routing or {}
        if tag in table:
            rung = table[tag]
            # Re-validated here as well as at parse time: an arm can go away
            # under a runtime detach, and a table validated at boot would then
            # be routing to something that is no longer loaded.
            arms.validate(rung, what=f"routing for tag {tag!r}")
            return Selection(rung, SelectionSource.ROUTED, f"tag {tag!r}")
        if strict_tags:
            known = ", ".join(sorted(table)) or "(empty table)"
            raise SelectionError(
                f"request tag {tag!r} has no routing entry and strict tag "
                f"routing is on; configured tags: {known}"
            )

    if controller_rung is not None:
        arms.validate(controller_rung, what="controller selection")
        return Selection(controller_rung, SelectionSource.CONTROLLER, "#156 controller")

    arms.validate(boot_rung, what="boot selection")
    return Selection(boot_rung, SelectionSource.BOOT, "boot default")


def arms_from_server_args(server_args) -> ArmSet:
    """Build the loaded-arm set from resolved server args.

    Mirrors what ``resolve_drafter_policy_table`` validates against, so the
    routing table and the policy table cannot disagree about which arms exist.
    """
    ks: List[int] = []
    boot_k = getattr(server_args, "speculative_num_steps", None)
    if boot_k:
        ks.append(int(boot_k))
    adaptive = getattr(server_args, "speculative_adaptive_config", None) or ""
    for token in str(adaptive).replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            ks.append(int(token))
    block = getattr(server_args, "speculative_dflash_block_size", None)
    return ArmSet(
        nextn_ks=tuple(sorted(set(ks))),
        dflash_block=int(block) if block else None,
    )


def selection_is_noop(current: Optional[Selection], new: Selection) -> bool:
    """Whether applying ``new`` would change anything.

    Used so the endpoint can report "already active" instead of logging a
    switch that did not happen -- a switch count that includes no-ops is the
    kind of metric that makes a controller look busier than it is.
    """
    return current is not None and current.rung == new.rung
