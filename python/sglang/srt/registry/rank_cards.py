# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The rank -> card-UUID vector, published WITHOUT a collective (#407 cut 2).

Several placement decisions want to know which physical card every rank of the
group is running on -- not just their own. #394's link-proportional cold-expert
sharding is the first: it weights a rank's share of the host expert tier by that
rank's measured PCIe bandwidth, and a rank cannot weight itself against peers it
cannot name.

**Why not a collective.** The obvious answer is an ``all_gather`` of
``current_device_uuid()``. It is the wrong answer here for two independent
reasons:

* The consumer runs inside the *weight-load* loop, before the model is built.
  A group collective there is the rank-local-before-group hazard: any rank that
  reaches the load path on a different schedule (a different quant, a different
  layer count, an early return) hangs the whole group with no diagnosis.
* The datum is not a runtime measurement at all. It is a *launch decision* the
  parent process already made when it computed ``gpu_id_for_rank`` for every
  scheduler it spawned. Asking the ranks to rediscover by collective what the
  launcher wrote down is a round trip through the network to read a local
  variable.

So the launcher publishes and the workers read. The channel is the environment,
which ``mp.Process`` carries into every spawned scheduler; the payload is
UUIDs, which are immune to the CUDA-ordinal / NVML-index divergence that
CUDA_VISIBLE_DEVICES narrowing introduces in each child (#392, #397).

**Why the launcher, and what it costs.** Resolving a CUDA ordinal to a physical
card needs the CUDA side of the ``#331`` :class:`~sglang.srt.registry.nvml.
IdentityMap`, and building that side creates a CUDA context -- a few hundred MiB
on every visible card, in the process that is about to fork workers onto those
same cards. That cost is refused by default. It is paid only when it has
*already* been paid:

* ``--rank-gpu-id`` resolves every ordinal through the same identity map during
  argument validation, so the context exists before this module runs;
* a context created by anything else in the launcher is likewise reused;
* an operator who wants the vector on the plain ``base_gpu_id`` path can ask
  for it with ``SGLANG_RANK_CARD_PROBE_CUDA=1`` and knows what it costs.

Otherwise nothing is published, and every consumer sees an honest absence with
its reason attached rather than a vector assembled from a guess about
enumeration order.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "RANK_CARD_UUIDS_ENV",
    "RankCardVector",
    "publish_rank_card_uuids",
    "rank_card_vector",
    "rank_card_uuids",
    "clear_published_rank_cards",
]

#: Comma-separated NVML UUIDs, one per WORLD rank, in ``world_rank`` order
#: (``pp_rank * tp_size + tp_rank``). Written by the launcher, read by every
#: worker. An operator may also set it by hand, which is how a launch that
#: does not go through ``_launch_subprocesses`` (a bench harness, a test rig)
#: supplies the same datum.
RANK_CARD_UUIDS_ENV = "SGLANG_RANK_CARD_UUIDS"


@dataclass(frozen=True)
class RankCardVector:
    """Which physical card each rank runs on, or a named absence.

    ``uuids`` is empty exactly when ``reason`` is set, so a caller can test
    either one and never disagree with itself. ``bdfs`` carries the PCI slot
    beside the UUID because the slot is what a human reads off ``lspci`` when
    they want to know which rank is behind the x4 link -- it is documentation,
    never a key.
    """

    uuids: Tuple[str, ...] = ()
    bdfs: Tuple[str, ...] = ()
    source: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if bool(self.uuids) == bool(self.reason):
            raise ValueError(
                "a rank-card vector carries either UUIDs or the reason they "
                f"are absent, never both and never neither (uuids={self.uuids}, "
                f"reason={self.reason!r})"
            )

    @property
    def present(self) -> bool:
        return bool(self.uuids)

    def describe(self) -> str:
        if not self.present:
            return f"rank->card vector absent: {self.reason}"
        cards = ", ".join(
            f"rank{i}={u}{f' [{b}]' if b else ''}"
            for i, (u, b) in enumerate(zip(self.uuids, self.bdfs or self.uuids))
        )
        return f"rank->card vector ({self.source}): {cards}"

    @classmethod
    def absent(cls, reason: str) -> "RankCardVector":
        return cls(reason=reason)


def _world_ordinals(server_args) -> Tuple[int, ...]:
    """Every rank's CUDA device index, in ``world_rank`` order.

    ``gpu_id_for_rank`` is the one placement formula the launcher itself uses
    (``--rank-gpu-id`` when given, the ``base_gpu_id``/``gpu_id_step`` formula
    otherwise), so enumerating it here reproduces the placement exactly rather
    than modelling it. The loop nesting is pp-outer / tp-inner because that is
    what ``world_rank = pp_rank * tp_size + tp_rank`` means.
    """
    pp = int(getattr(server_args, "pp_size", 1) or 1)
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    return tuple(
        int(server_args.gpu_id_for_rank(pp_rank, tp_rank, pp, tp))
        for pp_rank in range(pp)
        for tp_rank in range(tp)
    )


def _cuda_side_already_paid(server_args) -> bool:
    """True when resolving CUDA ordinals costs this process nothing extra.

    Three ways that happens, in the order they occur at launch: the operator
    set ``--rank-gpu-id`` (whose validation resolved every ordinal through the
    identity map already), a CUDA context exists here for some other reason, or
    the operator asked for the probe explicitly.
    """
    if getattr(server_args, "rank_gpu_id", None) is not None:
        return True
    try:
        from sglang.srt.environ import envs

        if envs.SGLANG_RANK_CARD_PROBE_CUDA.get():
            return True
    except Exception:  # noqa: BLE001 - env module absent in a bare desk import
        pass
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - no torch is simply "not paid"
        return False


def resolve_rank_card_vector(server_args, identity_map=None) -> RankCardVector:
    """Compute the vector in the launcher. Pure apart from reading NVML.

    ``identity_map`` is injectable so the hermetic tests can build a rig whose
    CUDA and NVML orders deliberately disagree without a driver.
    """
    nnodes = int(getattr(server_args, "nnodes", 1) or 1)
    if nnodes > 1:
        # Deliberate scope wall, not an oversight: this node's launcher knows
        # only its own cards, so the vector it could build would be a prefix
        # wearing a world-length label. A multi-node group needs a real
        # exchange, and #394 is single-node by construction.
        return RankCardVector.absent(
            f"the group spans {nnodes} nodes; this launcher can only see its "
            "own cards, and a per-node prefix is not a world-length vector"
        )

    if identity_map is None:
        if not _cuda_side_already_paid(server_args):
            return RankCardVector.absent(
                "the launcher has no CUDA context, so the CUDA enumeration "
                "order that --rank-gpu-id / base_gpu_id index into is unknown "
                "here. Creating one costs VRAM on every visible card in the "
                "process that is about to spawn workers onto them. Set "
                "--rank-gpu-id (its validation resolves the cards already) or "
                "SGLANG_RANK_CARD_PROBE_CUDA=1 to pay it deliberately"
            )
        try:
            from sglang.srt.registry import nvml as registry_nvml

            identity_map = registry_nvml.identity_map(allow_cuda_init=True)
        except Exception as exc:  # noqa: BLE001 - absent driver is not a crash
            return RankCardVector.absent(
                f"the NVML/CUDA identity map could not be built ({exc})"
            )

    try:
        ordinals = _world_ordinals(server_args)
    except Exception as exc:  # noqa: BLE001 - a malformed placement is absent
        return RankCardVector.absent(f"the rank placement could not be read ({exc})")
    if not ordinals:
        return RankCardVector.absent("the group has no ranks")

    uuids, bdfs = [], []
    for rank, ordinal in enumerate(ordinals):
        card = identity_map.by_cuda_ordinal(ordinal)
        if card is None:
            # All-or-nothing, the #397 stance: a partial vector would be
            # completed by the consumer with the index it already has, which
            # is the exact substitution this module exists to remove.
            return RankCardVector.absent(
                f"rank {rank} is placed on CUDA device {ordinal}, which this "
                "host cannot resolve to a physical card"
            )
        uuids.append(card.uuid)
        bdfs.append(card.pci_bus_id)

    return RankCardVector(
        uuids=tuple(uuids),
        bdfs=tuple(bdfs),
        source="launcher placement (gpu_id_for_rank -> #331 IdentityMap)",
    )


def publish_rank_card_uuids(server_args, identity_map=None) -> RankCardVector:
    """Resolve the vector and put it in the environment for the workers.

    Never raises and never overwrites an operator-set value: someone who typed
    the vector by hand meant it, and silently replacing it with a derived one
    is how a measurement arm turns into a lie (same contract as
    ``SGLANG_MOE_HOST_SHARD_RATIO``).
    """
    existing = os.environ.get(RANK_CARD_UUIDS_ENV, "").strip()
    if existing:
        vector = _parse_env_vector(existing)
        logger.info("%s (kept, set before launch)", vector.describe())
        return vector

    try:
        vector = resolve_rank_card_vector(server_args, identity_map=identity_map)
    except Exception as exc:  # noqa: BLE001 - publication is best-effort
        vector = RankCardVector.absent(f"resolution raised {type(exc).__name__}: {exc}")

    if vector.present:
        os.environ[RANK_CARD_UUIDS_ENV] = ",".join(vector.uuids)
        logger.info("%s", vector.describe())
    else:
        logger.debug("%s", vector.describe())
    return vector


def _parse_env_vector(raw: str) -> RankCardVector:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return RankCardVector.absent(f"{RANK_CARD_UUIDS_ENV} is set but empty")
    return RankCardVector(uuids=tuple(parts), bdfs=(), source=RANK_CARD_UUIDS_ENV)


def rank_card_vector(world_size: Optional[int] = None) -> RankCardVector:
    """Read the published vector in a worker. Never talks to any peer.

    ``world_size`` is checked rather than trusted: a vector whose length does
    not match the group asking for it describes a DIFFERENT group (a MoE-TP
    subgroup, a pipeline stage, a lane), and stretching or truncating it would
    silently attribute one rank's card to another.
    """
    raw = os.environ.get(RANK_CARD_UUIDS_ENV, "").strip()
    if not raw:
        return RankCardVector.absent(f"{RANK_CARD_UUIDS_ENV} is not set")
    vector = _parse_env_vector(raw)
    if not vector.present:
        return vector
    if world_size is not None and len(vector.uuids) != int(world_size):
        return RankCardVector.absent(
            f"{RANK_CARD_UUIDS_ENV} names {len(vector.uuids)} ranks but the "
            f"asking group has {int(world_size)}; that vector describes a "
            "different group"
        )
    return vector


def rank_card_uuids(world_size: Optional[int] = None) -> Optional[Tuple[str, ...]]:
    """The UUID per rank, or ``None`` when it is not known here.

    The ``None`` shape exists because every consumer's fallback is "behave
    exactly as before"; a caller that wants the reason asks
    :func:`rank_card_vector` instead.
    """
    vector = rank_card_vector(world_size)
    return vector.uuids if vector.present else None


def clear_published_rank_cards() -> None:
    """Test hook: forget the published vector."""
    os.environ.pop(RANK_CARD_UUIDS_ENV, None)


def format_rank_card_table(uuids: Sequence[str]) -> str:
    """One line per rank, for a log or a runbook. Resolves names via NVML."""
    try:
        from sglang.srt.registry import nvml as registry_nvml

        imap = registry_nvml.identity_map()
    except Exception:  # noqa: BLE001 - names are decoration, absence is fine
        imap = None
    lines = []
    for rank, uuid in enumerate(uuids):
        card = imap.get(uuid) if imap is not None else None
        detail = card.describe() if card is not None else uuid
        lines.append(f"  rank {rank}: {detail}")
    return "\n".join(lines)
