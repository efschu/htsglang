# SPDX-License-Identifier: Apache-2.0
"""--d-token-placement: WHERE new D KV tokens land, chosen by fill (27B, 26.09.).

User 26.09. ~09:40Z/09:45Z: "es muss dynamisch sein ... den kv so zu verteilen
wie er optimal schnell ist zur jeweiligen 'fülle'".  Design and evidence:
``/spinning/gpu-arb/docs/DYN_D_RESHARD.md`` sec. 12-13.

WHY NO SECOND LEDGER.  Under weighted uneven DCP a token's owner is a pure
function of its GLOBAL allocator slot id: rank r owns slot L iff
``L % S in [lo_r, hi_r)`` (``layers/dcp/owner.py``), and attention / the DCP
LSE merge derive each rank's token set from ``out_cache_loc`` itself
(``flashinfer_backend.py`` weighted write path).  The allocator therefore
chooses the card when it chooses the id.  The installed token vector (pool
shape, compact slots, HiCache owner bounds) stays exactly as it is; only the
ORDER of the replicated free list changes, so the next ids the allocator hands
out come from each rank's class in the wanted proportion.  Nothing already
stored moves.  This is the weighted generalisation of the #656/#657 owner bias
(``mem_cache/allocator/base.py`` ``set_owner_bias``, ``managers/corridor_steering``).

THE POLICY.  Decode reads weights AND KV every step, both bandwidth-bound, so
while the pool is far from full new tokens go to the ranks in proportion to
their EFFECTIVE bandwidth (the d_reshard fit, 937/604/604 GB/s -> 0.437/0.281/
0.281; today's capacity vector puts 0.407 on the 5090, after the static MLP
move only 0.33).  New tokens also repay an existing imbalance (weights =
the per-rank deficit against the bandwidth target at the current fill plus a
look-ahead).  When the bandwidth target would overrun a rank's class
(``fill_switch`` of it), the policy falls to capacity: weights proportional to
each class's free slots, i.e. every card fills up -- the ordinary behaviour.

DETERMINISM.  The free list is replicated scheduler state; every input here is
either a boot constant (shares, fill switch, class bounds) or a count derived
from that replicated list, and the interleave is a stable sort.  Every rank
therefore produces the same order without a collective.

Pure torch + arithmetic; the allocator hook lives in ``mem_cache/allocator``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

POLICY_CAPACITY = "capacity"
POLICY_BANDWIDTH = "bandwidth"
POLICIES = (POLICY_CAPACITY, POLICY_BANDWIDTH)

ENV = "SGLANG_WEG2_D_TOKEN_PLACEMENT"          # JSON spec; absent = capacity (off)
LOG_PREFIX = "D-TOKEN-PLACEMENT"

#: d_reshard two-point fit (rank order of the rc9 D group: 5090, 3080, 3080)
RC9_EFF_BW_GBS = (937.0, 604.0, 604.0)
DEFAULT_FILL_SWITCH = 0.85
DEFAULT_LOOKAHEAD = 0.02
#: re-apply the interleave at most every N allocation calls when the list was
#: touched (frees land at the head of the paged free list and wash the order)
DEFAULT_REAPPLY_EVERY = 32


class PlacementError(ValueError):
    pass


@dataclass(frozen=True)
class PlacementSpec:
    policy: str
    shares: Tuple[float, ...]
    fill_switch: float = DEFAULT_FILL_SWITCH
    lookahead: float = DEFAULT_LOOKAHEAD
    reapply_every: int = DEFAULT_REAPPLY_EVERY

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise PlacementError(f"policy {self.policy!r} not in {POLICIES}")
        if len(self.shares) < 2 or any(s <= 0 for s in self.shares):
            raise PlacementError(f"shares {self.shares} must be >= 2 positive entries")
        if not (0.0 < self.fill_switch <= 1.0):
            raise PlacementError(f"fill_switch {self.fill_switch} not in (0, 1]")
        if not (0.0 <= self.lookahead < 1.0):
            raise PlacementError(f"lookahead {self.lookahead} not in [0, 1)")
        if int(self.reapply_every) < 1:
            raise PlacementError("reapply_every must be >= 1")

    def normalized_shares(self) -> Tuple[float, ...]:
        s = float(sum(self.shares))
        return tuple(x / s for x in self.shares)

    def to_json(self) -> str:
        return json.dumps({"policy": self.policy, "shares": list(self.shares),
                           "fill_switch": self.fill_switch, "lookahead": self.lookahead,
                           "reapply_every": int(self.reapply_every)}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "PlacementSpec":
        d = json.loads(raw)
        spec = cls(str(d["policy"]), tuple(float(x) for x in d["shares"]),
                   float(d.get("fill_switch", DEFAULT_FILL_SWITCH)),
                   float(d.get("lookahead", DEFAULT_LOOKAHEAD)),
                   int(d.get("reapply_every", DEFAULT_REAPPLY_EVERY)))
        spec.validate()
        return spec


def spec_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[PlacementSpec]:
    raw = (os.environ if env is None else env).get(ENV)
    if not raw:
        return None
    spec = PlacementSpec.from_json(raw)
    return None if spec.policy == POLICY_CAPACITY else spec


# ---------------------------------------------------------------------------
# Policy: weights for the NEXT allocations from the class fill
# ---------------------------------------------------------------------------


def placement_weights(class_total: Sequence[int], class_free: Sequence[int],
                      spec: PlacementSpec) -> Tuple[Tuple[float, ...], str]:
    """(weights, regime).  'bandwidth': new tokens go to the per-rank deficit
    against ``share_r * (used + look-ahead)`` (or plain shares when balanced);
    'capacity': weights proportional to each class's free slots."""
    n = len(class_total)
    if len(class_free) != n or len(spec.shares) != n:
        raise PlacementError(f"class counts {class_total}/{class_free} vs shares {spec.shares}")
    share = spec.normalized_shares()
    total = [max(0, int(t)) for t in class_total]
    free = [max(0, min(int(f), t)) for f, t in zip(class_free, total)]
    used = [t - f for t, f in zip(total, free)]
    T, U = sum(total), sum(used)
    if T <= 0 or sum(free) <= 0:
        return tuple(0.0 for _ in range(n)), "full"
    target_all = U + spec.lookahead * T
    target = [share[r] * target_all for r in range(n)]
    if spec.policy == POLICY_BANDWIDTH and all(target[r] <= spec.fill_switch * total[r] for r in range(n)):
        deficit = [max(0.0, target[r] - used[r]) if free[r] > 0 else 0.0 for r in range(n)]
        if sum(deficit) > 0:
            s = sum(deficit)
            return tuple(d / s for d in deficit), "bandwidth"
        live = [share[r] if free[r] > 0 else 0.0 for r in range(n)]
        s = sum(live)
        return tuple(x / s for x in live), "bandwidth"
    s = float(sum(free))
    return tuple(f / s for f in free), "capacity"


# ---------------------------------------------------------------------------
# Mechanism: a stable weighted interleave of the replicated free list
# ---------------------------------------------------------------------------


def class_of(pages: torch.Tensor, mod: int, prefix: Sequence[int]) -> torch.Tensor:
    """Owner class (DCP rank) of every slot id under the weighted owner rule."""
    res = pages % int(mod)
    bounds = torch.tensor(list(prefix[1:-1]), dtype=res.dtype, device=res.device)
    return torch.bucketize(res, bounds, right=True)


def class_counts(pages: torch.Tensor, mod: int, prefix: Sequence[int]) -> List[int]:
    n = len(prefix) - 1
    if pages.numel() == 0:
        return [0] * n
    return [int(x) for x in torch.bincount(class_of(pages, mod, prefix), minlength=n).tolist()]


def slot_class_totals(size: int, mod: int, prefix: Sequence[int]) -> List[int]:
    """How many of the allocator's slot ids 1..size fall in each class."""
    q, rr = divmod(int(size) + 1, int(mod))  # ids 0..size: q full blocks + rr residues 0..rr-1
    out = []
    for r in range(len(prefix) - 1):
        lo, hi = int(prefix[r]), int(prefix[r + 1])
        cnt = q * (hi - lo) + max(0, min(hi, rr) - lo)
        if lo == 0:
            cnt -= 1  # id 0 is the padded dummy slot, never handed out
        out.append(cnt)
    return out


def weighted_interleave(pages: torch.Tensor, mod: int, prefix: Sequence[int],
                        weights: Sequence[float]) -> torch.Tensor:
    """Reorder ``pages`` so every prefix of the result draws from class r in
    proportion ``weights[r]`` while the class has pages left (then the others
    continue: capacity fallback by construction).  Stable within a class,
    deterministic across ranks (a stable sort on a pure key)."""
    if pages.numel() == 0:
        return pages
    n = len(prefix) - 1
    if len(weights) != n:
        raise PlacementError(f"{len(weights)} weights for {n} classes")
    cls = class_of(pages, mod, prefix)
    onehot = torch.nn.functional.one_hot(cls, n).to(torch.int64)
    rank_in_class = (onehot.cumsum(0) * onehot).sum(1) - 1
    w = torch.tensor([float(x) for x in weights], dtype=torch.float64, device=pages.device)
    wc = w[cls]
    key = torch.where(wc > 0, (rank_in_class.to(torch.float64) + 0.5) / wc.clamp_min(1e-300),
                      torch.full_like(wc, float("inf")))
    # zero-weight classes keep their relative order after all weighted ones
    order = torch.sort(key, stable=True).indices
    return pages[order]


def installed_prefix() -> Optional[Tuple[int, ...]]:
    """Prefix sums of the INSTALLED token vector (read at every apply: the
    measured install and the cutover replace the boot seed after arming)."""
    from sglang.srt.distributed.utils import get_cp_token_ratios

    ratios = get_cp_token_ratios()
    if not ratios:
        return None
    out = [0]
    for r in ratios:
        out.append(out[-1] + int(r))
    return tuple(out)


class OwnerPlacement:
    """What the allocator holds: the spec, the counters and where the class
    bounds come from.  ``apply(pages, size)`` -> reordered pages, log line."""

    def __init__(self, spec: PlacementSpec, prefix_fn=installed_prefix):
        spec.validate()
        self.spec = spec
        self.prefix_fn = prefix_fn
        self.calls = 0
        self.touched = True
        self.last_regime = ""
        self.last_weights: Tuple[float, ...] = ()

    def due(self) -> bool:
        self.calls += 1
        return self.touched and self.calls >= int(self.spec.reapply_every)

    def apply(self, pages: torch.Tensor, size: int) -> Tuple[torch.Tensor, str]:
        self.calls, self.touched = 0, False
        prefix = self.prefix_fn()
        if prefix is None or len(prefix) - 1 != len(self.spec.shares):
            regime, line = "inactive", ""
            if self.last_regime != regime:
                line = f"{LOG_PREFIX} regime=inactive token-vector-prefix={prefix} shares={self.spec.shares}"
            self.last_regime = regime
            return pages, line
        mod = int(prefix[-1])
        totals = slot_class_totals(size, mod, prefix)
        free = class_counts(pages, mod, prefix)
        weights, regime = placement_weights(totals, free, self.spec)
        line = ""
        if regime != self.last_regime:
            line = (f"{LOG_PREFIX} regime={regime} weights={[round(x, 3) for x in weights]} "
                    f"free={free} total={totals} token_vector_prefix={list(prefix)} "
                    f"shares={[round(x, 3) for x in self.spec.normalized_shares()]}")
        self.last_regime, self.last_weights = regime, weights
        if regime == "full":
            return pages, line
        return weighted_interleave(pages, mod, prefix, weights), line


def arm_on_allocator(allocator, spec: Optional[PlacementSpec], logger=None) -> bool:
    """Install the placement on a page_size-1 uneven-DCP allocator; False (and
    nothing installed) for None / capacity.  Refuses next to the corridor owner
    bias (two authors of one free-list order)."""
    if spec is None or spec.policy == POLICY_CAPACITY:
        return False
    if getattr(allocator, "page_size", 0) != 1:
        raise PlacementError(f"{LOG_PREFIX} needs page_size 1 (slot id == owner), got {allocator.page_size}")
    if getattr(allocator, "_owner_bias", None) is not None or os.environ.get("SGLANG_CORRIDOR_STEERING", "0") == "1":
        raise PlacementError(f"{LOG_PREFIX} refuses next to SGLANG_CORRIDOR_STEERING (one free-list order author)")
    allocator.set_owner_placement(OwnerPlacement(spec))
    if logger is not None:
        logger.info("%s armed: %s (class bounds read from the installed token vector at every apply)",
                    LOG_PREFIX, spec.to_json())
    return True
