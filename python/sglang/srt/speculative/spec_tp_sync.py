from __future__ import annotations

import logging
from enum import IntEnum

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SpecTpSyncSite(IntEnum):
    """Every place a speculative step broadcasts a decision from rank 0.

    Number and slug are the stable handles ``SGLANG_SPEC_TP_SYNC`` selects by.
    """

    # -- DSpark --
    DSPARK_MEM = 1  # gates both graph capture and folded sampling
    DSPARK_DRAFT_GREEDY = 2
    DSPARK_DRAFT_SAMPLE = 3
    DSPARK_DRAFT_MULTINOMIAL = 4
    DSPARK_GRAPH_SAMPLE = 5  # in-graph philox, redrawn per replay
    DSPARK_GRAPH_GREEDY = 6
    DSPARK_PLAN = 7
    DSPARK_ACCEPT_GREEDY = 8
    DSPARK_ACCEPT_SAMPLE = 9
    DSPARK_ACCEPT_GRAPH = 10
    DSPARK_TARGET = 11

    # -- DFlash --
    DFLASH_MEM = 12
    DFLASH_SELECTOR = 13
    DFLASH_ACCEPT_SAMPLE = 14
    DFLASH_ACCEPT_GREEDY = 15
    DFLASH_TARGET = 16

    @property
    def slug(self) -> str:
        return self.name.lower().replace("_", "-")


_ALL = frozenset(SpecTpSyncSite)
_INIT = frozenset({SpecTpSyncSite.DSPARK_MEM, SpecTpSyncSite.DFLASH_MEM})
# Sites that draw from the RNG, so they can differ under identical logits.
_RNG = frozenset(
    {
        SpecTpSyncSite.DSPARK_DRAFT_SAMPLE,
        SpecTpSyncSite.DSPARK_DRAFT_MULTINOMIAL,
        SpecTpSyncSite.DSPARK_GRAPH_SAMPLE,
        SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        SpecTpSyncSite.DSPARK_TARGET,
        SpecTpSyncSite.DFLASH_SELECTOR,
        SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE,
        SpecTpSyncSite.DFLASH_TARGET,
    }
)

_PRESETS = {
    "all": _ALL,
    "off": frozenset(),
    "none": frozenset(),
    # The one input measured to differ across ranks.
    "init": _INIT,
    "rng": _INIT | _RNG,
}

_BY_SLUG = {site.slug: site for site in SpecTpSyncSite}
_BY_NUMBER = {str(int(site)): site for site in SpecTpSyncSite}


def _resolve(name: str) -> frozenset[SpecTpSyncSite]:
    if name in _PRESETS:
        return _PRESETS[name]
    site = _BY_SLUG.get(name) or _BY_NUMBER.get(name)
    if site is None:
        raise ValueError(
            f"SGLANG_SPEC_TP_SYNC: unknown token {name!r}. "
            f"Presets: {sorted(_PRESETS)}. "
            f"Sites: {[(int(s), s.slug) for s in SpecTpSyncSite]}."
        )
    return frozenset({site})


def parse_spec_tp_sync(spec: str) -> frozenset[SpecTpSyncSite]:
    """Parse ``SGLANG_SPEC_TP_SYNC``; see its comment in environ.py for the syntax."""
    sites: frozenset[SpecTpSyncSite] = frozenset()
    for token in spec.replace(" ", "").replace("_", "-").lower().split(","):
        if not token:
            continue
        negate = token.startswith("-")
        value = _resolve(token[1:] if negate else token)
        sites = sites - value if negate else sites | value
    return sites


class SpecTpSync:
    """Broadcasts a speculative decision from rank 0 to its TP group."""

    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        # Parsed even on a single rank so a typo fails on every deployment.
        sites = parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.get())
        self._sites = sites if tp_group.world_size > 1 else frozenset()
        if sites != _ALL and tp_group.world_size > 1 and tp_group.rank_in_group == 0:
            logger.warning(
                "Speculative TP sync limited to %s.",
                [f"{int(s)}:{s.slug}" for s in sorted(sites)] or "no site",
            )

    def enabled(self, site: SpecTpSyncSite) -> bool:
        return site in self._sites

    def sync(self, site: SpecTpSyncSite, values: torch.Tensor) -> torch.Tensor:
        if site in self._sites:
            # #1485 instrument (df2l6, 17.09.): the broadcast ENFORCES rank 0's
            # decision and thereby hides where the ranks diverged (TP0 finished
            # a 1024-token decode one round before TP1/TP2, which then hung in
            # the graph's collective).  Keep the rank-local value and report a
            # mismatch, bounded.
            _local = None
            if self._tp_group.world_size > 1 and getattr(self, "_1485_n", 0) < 40:
                try:
                    _local = values.detach().clone()
                except Exception:  # noqa: BLE001
                    _local = None
            self._tp_group.broadcast(values, src=0)
            if _local is not None:
                try:
                    if not torch.equal(_local, values):
                        self._1485_n = getattr(self, "_1485_n", 0) + 1
                        _d = int((_local != values).sum().item())
                        logger.warning("#1485 SPEC-TP-DIVERGE site=%s rank=%d differing=%d/%d local=%s synced=%s",
                                       site.name, int(self._tp_group.rank_in_group), _d, int(values.numel()),
                                       _local.flatten()[:8].tolist(), values.flatten()[:8].tolist())
                except Exception:  # noqa: BLE001
                    pass
        return values

    def available_memory_gb(self, site: SpecTpSyncSite, device, gpu_id, *, group):
        """Free GPU memory, reduced to the group minimum when ``site`` is on."""
        distributed = self.enabled(site) and group.world_size > 1
        return get_available_gpu_memory(
            device,
            gpu_id,
            distributed=distributed,
            cpu_group=group.cpu_group if distributed else None,
        )
