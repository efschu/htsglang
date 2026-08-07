"""#396(b): a declarative regex -> placement surface over the residency plan.

``llama.cpp``'s ``-ot`` / ``--override-tensor`` lets an operator say "these
tensors go there" by matching tensor names with a regex. This fork's planner
already places more precisely than that -- it SOLVES the split rather than
accepting one -- but it had no surface through which an operator could express
an intent the solver has no way to know: a tensor family that must stay on the
fast card, a family that is fine in host RAM, a family that should be reached
from a disk tier.

This module is that surface, and the shape of it is the whole point:

**Overrides are CONSTRAINTS handed to the solve, not a bypass around it.** The
planner still computes the placement, still checks it, and still refuses when
the constraints cannot be satisfied. An operator can say what must be true; it
cannot say what is true. That distinction is why
:class:`PlacementOverrideConflict` exists and why the refusals below name the
rank, the expert and the arithmetic rather than silently dropping an override
the solve could not honour -- a dropped constraint is a plan that looks solved
and is not the one that was asked for.

Grammar (one ``--expert-placement-override`` per rule, repeatable)::

    <python-regex>=<target>

    target := cpu | host                 host RAM (the #77 cold tier)
            | gpu:<cuda-ordinal>         a card by CUDA-order index
            | gpu:GPU-<uuid>             a card by NVML UUID (#397 IdentityMap)
            | disk:<tier-id>             a #407 filesystem/blob tier id

The spec is split on the LAST ``=``, so a regex may contain ``=`` freely. The
first override whose regex matches a tensor name wins, which is ``-ot``'s own
rule and the one an operator writing a specific rule before a general one
already expects.

Device names resolve through the #397 :class:`IdentityMap`, never through
torch's enumeration order: on the reference rig CUDA 0 and NVML 0 are
different physical cards, and an override that named the wrong one would move
a tensor family to the wrong card while reporting success.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PlacementOverride",
    "PlacementOverrideConflict",
    "PlacementOverrideError",
    "PlacementTarget",
    "expert_tensor_names",
    "parse_placement_overrides",
    "resolve_expert_constraints",
]


class PlacementOverrideError(ValueError):
    """A placement override could not be parsed or validated."""


class PlacementOverrideConflict(PlacementOverrideError):
    """Overrides are self-contradictory, or cannot be met by this plan.

    Raised rather than reported so an unmeetable constraint cannot reach a
    boot as a silently relaxed one. This is the ``planner/rejected.py``
    posture applied to operator input: settle it loudly at parse/solve time,
    at the point where the arithmetic that refutes it is still in hand.
    """


@dataclasses.dataclass(frozen=True)
class PlacementTarget:
    """Where a matched tensor is required to live.

    ``kind`` is one of ``gpu`` / ``cpu`` / ``disk``. For ``gpu`` exactly one of
    ``cuda_ordinal`` / ``uuid`` is what the operator wrote and the other is
    filled in when an identity map is available -- so a plan can be checked
    against an index even when the operator named a UUID, and the record still
    remembers which one was authoritative.
    """

    kind: str
    raw: str
    cuda_ordinal: Optional[int] = None
    uuid: Optional[str] = None
    tier_id: Optional[str] = None

    def describe(self) -> str:
        if self.kind == "gpu":
            if self.uuid and self.cuda_ordinal is not None:
                return f"gpu:{self.cuda_ordinal} ({self.uuid})"
            if self.uuid:
                return f"gpu:{self.uuid}"
            return f"gpu:{self.cuda_ordinal}"
        if self.kind == "disk":
            return f"disk:{self.tier_id}"
        return "cpu"


@dataclasses.dataclass(frozen=True)
class PlacementOverride:
    """One ``regex=target`` rule, in the order the operator gave it."""

    raw_spec: str
    pattern: str
    target: PlacementTarget
    #: Position on the command line. First match wins, so this is the
    #: precedence and it is worth carrying into every refusal message.
    order: int

    @property
    def regex(self) -> "re.Pattern[str]":
        return re.compile(self.pattern)

    def matches(self, tensor_name: str) -> bool:
        return self.regex.search(tensor_name) is not None


# ---------------------------------------------------------------------------
# parse + validate
# ---------------------------------------------------------------------------


def _parse_target(raw: str, identity, spec: str) -> PlacementTarget:
    token = raw.strip()
    if not token:
        raise PlacementOverrideError(
            f"placement override {spec!r} has an empty target; expected one of "
            f"cpu, gpu:<index-or-uuid>, disk:<tier-id>"
        )
    low = token.lower()
    if low in ("cpu", "host"):
        return PlacementTarget(kind="cpu", raw=token)
    if low.startswith("gpu:"):
        return _parse_gpu_target(token[4:].strip(), identity, spec, token)
    if low.startswith("disk:"):
        return _parse_disk_target(token[5:].strip(), spec, token)
    raise PlacementOverrideError(
        f"placement override {spec!r} names an unknown target {token!r}; "
        f"expected cpu, gpu:<index-or-uuid> or disk:<tier-id>"
    )


def _parse_gpu_target(body: str, identity, spec: str, raw: str) -> PlacementTarget:
    if not body:
        raise PlacementOverrideError(
            f"placement override {spec!r} has a gpu target with no card; "
            f"write gpu:<cuda-index> or gpu:GPU-<uuid>"
        )
    if body.isdigit():
        ordinal = int(body)
        uuid = None
        if identity is not None:
            known = _cuda_ordinals(identity)
            # ABSENCE OF INFORMATION IS NOT CONTRADICTION. The CUDA-ordinal
            # side of the identity map is filled in only when a CUDA context
            # exists, and a planner deliberately does not create one
            # (registry/rank_cards.py:43 -- it costs a few hundred MiB). With
            # no ordinals resolved at all the map simply cannot answer, and
            # refusing then would make gpu:<index> unusable on every host that
            # has not paid for a context. The check runs when the map CAN
            # answer and disagrees, which is the case worth catching.
            if known:
                card = _card_by_cuda_ordinal(identity, ordinal)
                if card is None:
                    raise PlacementOverrideError(
                        f"placement override {spec!r} names cuda ordinal "
                        f"{ordinal}, which no card on this host occupies; "
                        f"visible cuda ordinals: {sorted(known)}"
                    )
                uuid = card.uuid
        return PlacementTarget(kind="gpu", raw=raw, cuda_ordinal=ordinal, uuid=uuid)
    # A UUID. Resolve it through the identity map when one is available; a
    # UUID that is not present is a pulled or renamed card, and IdentityMap
    # deliberately raises rather than returning a nearest match (#331).
    ordinal = None
    if identity is not None:
        try:
            card = identity.require(body)
        except Exception as exc:  # noqa: BLE001 - re-raised with our context
            raise PlacementOverrideError(
                f"placement override {spec!r} names card {body}, which this "
                f"host cannot resolve: {exc}"
            ) from exc
        ordinal = card.cuda_ordinal
    return PlacementTarget(kind="gpu", raw=raw, cuda_ordinal=ordinal, uuid=body)


def _parse_disk_target(body: str, spec: str, raw: str) -> PlacementTarget:
    if not body:
        raise PlacementOverrideError(
            f"placement override {spec!r} has a disk target with no tier; "
            f"write disk:fs:<host>:<mount> or disk:blob:<backend>:<scope>"
        )
    try:
        from sglang.srt.memtier.tiers import parse_tier_id

        parsed = parse_tier_id(body)
    except ImportError:  # pragma: no cover - memtier is in-tree
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with our context
        raise PlacementOverrideError(
            f"placement override {spec!r} names disk tier {body!r}, which is "
            f"not a valid #407 tier id: {exc}"
        ) from exc
    from sglang.srt.memtier.tiers import TierKind

    if parsed.kind not in (TierKind.FILESYSTEM, TierKind.BLOB):
        raise PlacementOverrideError(
            f"placement override {spec!r} names tier {body!r} of kind "
            f"{parsed.kind.value!r}; a disk: target must be an fs: or blob: "
            f"tier (use cpu for host RAM and gpu:<card> for device memory)"
        )
    return PlacementTarget(kind="disk", raw=raw, tier_id=body)


def _cuda_ordinals(identity) -> List[int]:
    return [c.cuda_ordinal for c in identity if c.cuda_ordinal is not None]


def _card_by_cuda_ordinal(identity, ordinal: int):
    for card in identity:
        if card.cuda_ordinal == ordinal:
            return card
    return None


def parse_placement_overrides(
    specs: Optional[Sequence[str]], identity=None
) -> Tuple[PlacementOverride, ...]:
    """Parse ``regex=target`` specs into ordered, validated overrides.

    ``identity`` is a #397 :class:`~sglang.srt.registry.nvml.IdentityMap` (or
    anything iterating :class:`CardIdentity`-shaped records) used to check that
    a named card exists. ``None`` skips the card check -- which is what a desk
    plan for a rig that is not this one needs, and is the only reason it is
    optional; a boot passes the live map.
    """
    if not specs:
        return ()
    out: List[PlacementOverride] = []
    for order, spec in enumerate(specs):
        if spec is None:
            continue
        spec = str(spec).strip()
        if not spec:
            continue
        if "=" not in spec:
            raise PlacementOverrideError(
                f"placement override {spec!r} is not 'regex=target'; the "
                f"target follows the LAST '=' so the regex may contain one"
            )
        pattern, _, target_raw = spec.rpartition("=")
        pattern = pattern.strip()
        if not pattern:
            raise PlacementOverrideError(
                f"placement override {spec!r} has an empty regex; a rule that "
                f"matches everything must say so explicitly (e.g. '.*=cpu')"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise PlacementOverrideError(
                f"placement override {spec!r} has an invalid regex: {exc}"
            ) from exc
        out.append(
            PlacementOverride(
                raw_spec=spec,
                pattern=pattern,
                target=_parse_target(target_raw, identity, spec),
                order=order,
            )
        )
    return tuple(out)


def first_match(
    overrides: Sequence[PlacementOverride], tensor_name: str
) -> Optional[PlacementOverride]:
    """The winning rule for one tensor name, or ``None``.

    FIRST match, not most-specific: the operator's order IS the precedence,
    the same rule ``-ot`` uses. A most-specific-wins rule would need a
    specificity metric over regexes, which does not exist.
    """
    for override in overrides:
        if override.matches(tensor_name):
            return override
    return None


# ---------------------------------------------------------------------------
# tensor names -> expert constraints
# ---------------------------------------------------------------------------

#: The names one expert of one layer answers to. The first three are REAL
#: checkpoint tensor names -- HF/safetensors stores each expert as its own
#: tensor, so a regex against them matches something that exists on disk.
#:
#: The ``blk.`` forms are SYNTHETIC and deliberately so. GGUF does not store a
#: tensor per expert: it stores one expert-major ``blk.N.ffn_gate_exps.weight``
#: holding every expert, so there is no real per-expert name for an operator to
#: match on. Since the placement unit here IS the individual expert, the fork
#: supplies an addressing for it by splicing the expert id into the GGUF tensor
#: name. Nobody should read these back as checkpoint keys; they exist so the
#: same rule can be written once and mean the same expert on either format.
_EXPERT_NAME_TEMPLATES = (
    # Real HF/safetensors tensor names.
    "model.layers.{layer}.mlp.experts.{expert}.w1.weight",
    "model.layers.{layer}.mlp.experts.{expert}.w2.weight",
    "model.layers.{layer}.mlp.experts.{expert}.w3.weight",
    # Synthetic per-expert addressing over GGUF's expert-major tensors.
    "blk.{layer}.ffn_gate_exps.{expert}.weight",
    "blk.{layer}.ffn_up_exps.{expert}.weight",
    "blk.{layer}.ffn_down_exps.{expert}.weight",
)


def expert_tensor_names(layer: int, expert: int) -> Tuple[str, ...]:
    """Every name one expert of one layer answers to.

    Real checkpoint keys for HF/safetensors; a synthetic per-expert addressing
    for GGUF, which has no per-expert tensor. See the templates above.
    """
    return tuple(
        t.format(layer=int(layer), expert=int(expert)) for t in _EXPERT_NAME_TEMPLATES
    )


@dataclasses.dataclass(frozen=True)
class ExpertConstraints:
    """The constraint set one rank's expert range must satisfy.

    Expert ids are GLOBAL (the same space as ``ExpertPlacement.expert_start``),
    because that is the space an operator's regex is written against -- a
    checkpoint's tensor names do not know about this rig's rank split.
    """

    rank: int
    gpu_index: Optional[int]
    #: Must stay GPU-resident on this rank's card.
    resident: Tuple[int, ...] = ()
    #: Must live in the host cold tier.
    host: Tuple[int, ...] = ()
    #: Must be reached from a disk tier, keyed by tier id.
    disk: Tuple[Tuple[int, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.resident or self.host or self.disk)


def resolve_expert_constraints(
    overrides: Sequence[PlacementOverride],
    rank: int,
    gpu_index: Optional[int],
    expert_start: int,
    expert_end: int,
    num_layers: int,
) -> ExpertConstraints:
    """Turn matching rules into this rank's constraint set, or refuse.

    Two refusals are decided here, both of them cases where continuing would
    mean quietly answering a different question than the one asked:

    * an expert whose tensor names are matched by rules with DIFFERENT targets
      -- e.g. a rule sending ``w1`` to the GPU and a later one sending ``w2``
      to the host. The offload's unit is the whole expert (its rows are copied
      whole, on the one axis with no block structure), so splitting one expert
      across tiers is not a thing this plan can express;
    * an expert pinned to a card that is not the card its owning rank runs on.
      Moving the expert would mean moving the rank, which is a different plan
      entirely -- ``--rank-moe-ratio`` is the surface for that.
    """
    resident: List[int] = []
    host: List[int] = []
    disk: List[Tuple[int, str]] = []
    if not overrides:
        return ExpertConstraints(rank=rank, gpu_index=gpu_index)

    for expert in range(int(expert_start), int(expert_end)):
        winners: Dict[str, PlacementOverride] = {}
        for layer in range(int(num_layers)):
            for name in expert_tensor_names(layer, expert):
                hit = first_match(overrides, name)
                if hit is not None:
                    winners.setdefault(hit.target.describe(), hit)
        if not winners:
            continue
        if len(winners) > 1:
            rules = ", ".join(
                f"{o.raw_spec!r} -> {o.target.describe()}"
                for o in sorted(winners.values(), key=lambda o: o.order)
            )
            raise PlacementOverrideConflict(
                f"expert {expert} (rank {rank}) is matched by placement "
                f"overrides with different targets: {rules}. An expert is "
                f"placed whole -- its rows are copied on the expert axis, the "
                f"one axis with no quantization-block structure -- so it "
                f"cannot be split across tiers. Narrow the regexes so at most "
                f"one target matches each expert."
            )
        winner = next(iter(winners.values()))
        target = winner.target
        if target.kind == "cpu":
            host.append(expert)
        elif target.kind == "disk":
            disk.append((expert, str(target.tier_id)))
        else:
            if (
                target.cuda_ordinal is not None
                and gpu_index is not None
                and int(target.cuda_ordinal) != int(gpu_index)
            ):
                raise PlacementOverrideConflict(
                    f"placement override {winner.raw_spec!r} pins expert "
                    f"{expert} to {target.describe()}, but expert {expert} "
                    f"belongs to rank {rank}, which runs on gpu {gpu_index}. "
                    f"An expert cannot be moved to another card without "
                    f"moving the rank; use --rank-moe-ratio to change which "
                    f"rank owns which experts."
                )
            resident.append(expert)

    return ExpertConstraints(
        rank=rank,
        gpu_index=gpu_index,
        resident=tuple(resident),
        host=tuple(host),
        disk=tuple(disk),
    )


def apply_expert_constraints(
    constraints: ExpertConstraints,
    expert_start: int,
    expert_end: int,
    resident_slots: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(resident_ids, host_ids)`` for one rank under its constraints.

    Feasibility is checked against ``resident_slots`` -- the number of experts
    the solved resident fraction leaves room for on this rank's card. Forcing
    more experts resident than there are slots is the one arithmetic an
    operator can write that the plan cannot absorb, and it is refused with
    both numbers in hand rather than by silently dropping the tail.

    Unconstrained experts fill the remaining resident slots in ascending id
    order, which is the SAME rule the unconstrained plan uses
    (``plan_load_time_staging``: pinned ids take the lowest slots, the rest
    ascend). So an empty constraint set reproduces today's split exactly.
    """
    if constraints.disk:
        # The residency solve models exactly two places an expert can be: this
        # rank's card, or the host cold tier. A disk: target is grammatically
        # valid and its tier id is real (#407), but folding it into "off-card"
        # would answer a different question than the operator asked -- the
        # plan would report a host-RAM figure for mass the operator asked to
        # keep on an NVMe tier. Refuse until there is a disk residency class
        # to place into; that is a rung on the ladder, not a parser change.
        experts = sorted({e for e, _ in constraints.disk})
        tiers = sorted({t for _, t in constraints.disk})
        raise PlacementOverrideConflict(
            f"placement overrides send experts {experts} on rank "
            f"{constraints.rank} to disk tier(s) {tiers}, but the residency "
            f"solve has only two classes today -- GPU-resident and the host "
            f"cold tier -- so there is no disk rung to place them on. "
            f"Accepting this would report host RAM for mass you asked to keep "
            f"on disk. Use cpu for the host tier; a disk residency class is "
            f"the #396(a) on-demand tier's follow-up, not a parse change."
        )
    forced_resident = list(constraints.resident)
    forced_host = set(constraints.host)
    slots = int(resident_slots)
    if len(forced_resident) > slots:
        raise PlacementOverrideConflict(
            f"placement overrides pin {len(forced_resident)} experts resident "
            f"on rank {constraints.rank} (gpu {constraints.gpu_index}), but "
            f"the solved resident fraction leaves only {slots} resident slot"
            f"{'' if slots == 1 else 's'} there. Raise the resident fraction, "
            f"or pin fewer experts."
        )
    overlap = sorted(set(forced_resident) & forced_host)
    if overlap:
        raise PlacementOverrideConflict(
            f"placement overrides pin experts {overlap} both resident and "
            f"off-card on rank {constraints.rank}"
        )
    rest = [
        e
        for e in range(int(expert_start), int(expert_end))
        if e not in set(forced_resident) and e not in forced_host
    ]
    resident = sorted(forced_resident) + rest[: slots - len(forced_resident)]
    resident_set = set(resident)
    host = [
        e for e in range(int(expert_start), int(expert_end)) if e not in resident_set
    ]
    return tuple(sorted(resident)), tuple(host)


def describe_overrides(overrides: Iterable[PlacementOverride]) -> str:
    """One human-readable line per rule, for the plan's notes."""
    return "; ".join(
        f"#{o.order} {o.pattern!r} -> {o.target.describe()}" for o in overrides
    )
