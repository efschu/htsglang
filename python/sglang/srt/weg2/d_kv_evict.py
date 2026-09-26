# SPDX-License-Identifier: Apache-2.0
"""--d-kv-evict-for-placement: cold radix KV makes room for the bandwidth placement (27B D, 26.09., row 24c).

User 26.09.: "fertige requests oder vorherige requests muessen dann natuerlich
den vram kvcache frei machen um die bestverteilung zu ermoeglichen".  Design:
``/spinning/gpu-arb/docs/DYN_D_RESHARD.md`` sec. 14 (builds sec. 13.4).

WHY.  ``--d-token-placement bandwidth`` (``d_token_placement.py``) keeps new D
tokens on the cards by effective bandwidth only while the TOTAL fill U stays
under ``U_switch = min_r(f * T_r / s_r) - la * T``.  The radix tree keeps every
finished request on the device until the whole free list is exhausted, so
within a D epoch U grows with COLD prefix and the placement falls to capacity
-- where the fast card, filled first, gets less than even its capacity share.

WHAT.  Every ``every`` scheduler ticks, at the one group-uniform point after
``check_hicache_events`` (the admission pass), this module:

  1. picks cold structural device leaves (last access before the agreed cold
     floor), coldest first, sorted by a replicated key;
  2. agrees with the group in ONE fixed-shape MIN all_reduce: the tree digest
     (divergence = crash-stop, never continue), the fill, a monotonic "now"
     and two per-candidate masks of the RANK-LOCAL state (unlocked here -- no
     active reader, no load-back or write pin -- and backed up in L2 on this
     rank / publishable); acks are rank-local, so both masks are;
  3. selects victims by a pure function of agreed inputs (identical list on
     every rank) and demotes them with ``_evict_to_host`` -- a device free,
     the node stays in the tree with its L2 (arena) rows: NO copy;
  4. hands cold UN-backed leaves (finished decode tails) to the existing
     ``publish_unbacked_sweep`` (write stream, small staging ring / arena
     direct claim), so a later pass can demote them copy-free.

NEVER ``_evict_device_leaf``: under D's ``write_back`` it writes an un-backed
node back SYNCHRONOUSLY -- the brake the user forbade (hicache bremst nie).

OFF (default) = no env = the scheduler hook returns at its first lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_D_KV_EVICT"            # JSON spec; absent = off
LOG_PREFIX = "D-KV-EVICT"
MAX_CLASSES = 8                            # fixed header width (D group is TP3)

# header layout of the one MIN all_reduce (int64); negated twins give the MAX
H_FLAG, H_NFLAG, H_TMS, H_DIG, H_NDIG, H_N, H_NN, H_U, H_NU = range(9)
H_UR = 9                                   # U_r[MAX_CLASSES] then -U_r[MAX_CLASSES]
HEADER_LEN = H_UR + 2 * MAX_CLASSES


class EvictError(ValueError):
    pass


class RankDivergence(RuntimeError):
    """raenge-nie-uneins: the group disagrees on replicated state -> crash-stop."""


@dataclass(frozen=True)
class EvictSpec:
    every: int = 16                 # scheduler ticks between passes (each pass = one small collective)
    min_idle_s: float = 20.0        # cold = no access for this long (group-agreed clock)
    hysteresis: float = 0.02        # trigger at U_switch - h*T, evict down to U_switch - 2h*T
    max_candidates: int = 128       # K: fixed mask width of the collective
    max_evict_nodes: int = 64
    max_evict_tokens: int = 131072
    publish_max: int = 8            # cold un-backed leaves handed to the publish sweep per pass (0 = never)

    def validate(self) -> None:
        if int(self.every) < 1:
            raise EvictError("every must be >= 1")
        if not (self.min_idle_s >= 0.0):
            raise EvictError(f"min_idle_s {self.min_idle_s} must be >= 0")
        if not (0.0 <= self.hysteresis < 0.5):
            raise EvictError(f"hysteresis {self.hysteresis} not in [0, 0.5)")
        if not (1 <= int(self.max_candidates) <= 4096):
            raise EvictError("max_candidates not in [1, 4096]")
        if int(self.max_evict_nodes) < 1 or int(self.max_evict_tokens) < 1:
            raise EvictError("max_evict_nodes / max_evict_tokens must be >= 1")
        if int(self.publish_max) < 0:
            raise EvictError("publish_max must be >= 0")

    def to_json(self) -> str:
        return json.dumps({"every": int(self.every), "min_idle_s": float(self.min_idle_s),
                           "hysteresis": float(self.hysteresis),
                           "max_candidates": int(self.max_candidates),
                           "max_evict_nodes": int(self.max_evict_nodes),
                           "max_evict_tokens": int(self.max_evict_tokens),
                           "publish_max": int(self.publish_max)}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "EvictSpec":
        d = json.loads(raw)
        base = cls()
        spec = cls(int(d.get("every", base.every)), float(d.get("min_idle_s", base.min_idle_s)),
                   float(d.get("hysteresis", base.hysteresis)),
                   int(d.get("max_candidates", base.max_candidates)),
                   int(d.get("max_evict_nodes", base.max_evict_nodes)),
                   int(d.get("max_evict_tokens", base.max_evict_tokens)),
                   int(d.get("publish_max", base.publish_max)))
        spec.validate()
        return spec


def spec_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[EvictSpec]:
    raw = (os.environ if env is None else env).get(ENV)
    if not raw:
        return None
    return EvictSpec.from_json(raw)


# ---------------------------------------------------------------------------
# Pure policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pressure:
    total: int
    used: int
    switch: float
    goal: float
    triggered: bool
    need: int
    excess: Tuple[int, ...]


def switch_level(totals: Sequence[int], shares: Sequence[float], fill_switch: float, lookahead: float) -> float:
    """The total fill above which ``placement_weights`` leaves 'bandwidth':
    s_r * (U + la*T) <= f * T_r for all r  <=>  U <= min_r(f*T_r/s_r) - la*T."""
    ssum = float(sum(shares))
    if ssum <= 0 or len(shares) != len(totals):
        raise EvictError(f"shares {shares} vs totals {totals}")
    T = float(sum(max(0, int(t)) for t in totals))
    return min(fill_switch * max(0, int(t)) / (s / ssum) for t, s in zip(totals, shares)) - lookahead * T


def pressure(totals: Sequence[int], used_per_class: Sequence[int], shares: Sequence[float],
             fill_switch: float, lookahead: float, hysteresis: float) -> Pressure:
    n = len(totals)
    if len(used_per_class) != n or len(shares) != n:
        raise EvictError(f"class counts {totals}/{used_per_class} vs shares {shares}")
    T = sum(max(0, int(t)) for t in totals)
    used = [max(0, min(int(u), int(t))) for u, t in zip(used_per_class, totals)]
    U = sum(used)
    sw = switch_level(totals, shares, fill_switch, lookahead)
    goal = max(0.0, sw - 2.0 * hysteresis * T)
    triggered = T > 0 and U > sw - hysteresis * T
    ssum = float(sum(shares))
    if not triggered:
        return Pressure(T, U, sw, goal, False, 0, tuple(0 for _ in range(n)))
    need = max(0, int(U - goal + 0.999999))
    excess = tuple(max(0, int(used[r] - (shares[r] / ssum) * goal + 0.999999)) for r in range(n))
    return Pressure(T, U, sw, goal, True, need, excess)


def triggered_by_total(totals: Sequence[int], used_total: int, shares: Sequence[float],
                       fill_switch: float, lookahead: float, hysteresis: float) -> bool:
    """The cheap pre-check (total fill only; the regime switch depends on the total)."""
    T = sum(max(0, int(t)) for t in totals)
    if T <= 0:
        return False
    return int(used_total) > switch_level(totals, shares, fill_switch, lookahead) - hysteresis * T


def select_victims(lengths: Sequence[int], class_counts: Sequence[Sequence[int]],
                   evict_ok: Sequence[int], publish_ok: Sequence[int],
                   p: Pressure, spec: EvictSpec) -> Tuple[List[int], List[int]]:
    """(evict indices, publish indices) -- a pure function of agreed inputs.

    Candidates arrive coldest first.  Greedy: the candidate whose tokens lie
    most in classes still over their share of the goal (useful fraction) wins,
    ties go to the colder one; stop when the need is covered or a cap hits.
    Only agreed-backed candidates are evicted; if the backed ones cannot cover
    the need, the coldest useful un-backed ones are published instead."""
    k = len(lengths)
    if not p.triggered or p.need <= 0 or k == 0:
        return [], []
    X = list(p.excess)
    need = int(p.need)
    chosen: List[int] = []
    taken = [False] * k
    tokens = 0

    def useful(i: int) -> int:
        return sum(min(int(c), x) for c, x in zip(class_counts[i], X))

    while need > 0 and len(chosen) < int(spec.max_evict_nodes):
        best, best_score = -1, 0.0
        for i in range(k):
            if taken[i] or not evict_ok[i] or lengths[i] <= 0:
                continue
            if tokens + int(lengths[i]) > int(spec.max_evict_tokens):
                continue
            score = useful(i) / float(lengths[i])
            if score > best_score + 1e-12:
                best, best_score = i, score
        if best < 0:
            break
        taken[best] = True
        chosen.append(best)
        for r, c in enumerate(class_counts[best]):
            X[r] = max(0, X[r] - int(c))
        need -= int(lengths[best])
        tokens += int(lengths[best])
    publish: List[int] = []
    if need > 0 and int(spec.publish_max) > 0:
        for i in range(k):
            if len(publish) >= int(spec.publish_max):
                break
            if taken[i] or not publish_ok[i] or lengths[i] <= 0 or useful(i) <= 0:
                continue
            publish.append(i)
    return chosen, publish


def candidate_digest(keys: Sequence[Tuple[int, ...]]) -> int:
    """63-bit digest of the ORDERED candidate keys (length, first slot, last slot)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(len(keys)).encode())
    for key in keys:
        h.update((",".join(str(int(x)) for x in key) + ";").encode())
    return int.from_bytes(h.digest(), "big") >> 1


class ColdClock:
    """(agreed time, local access-counter) pairs, one per pass.  ``floor()`` is
    the counter of the newest pass at least ``min_idle`` before the newest
    agreed time: a node whose last access is older was untouched that long.
    Agreed times are identical on every rank and each rank reads its own
    counter at the same pass, so the cold set is group-uniform."""

    def __init__(self, min_idle_ms: float):
        self.min_idle_ms = float(min_idle_ms)
        self.ring: deque = deque()

    def record(self, t_ms: int, counter: float) -> None:
        self.ring.append((int(t_ms), float(counter)))
        # keep one entry older than the idle window, drop the rest
        while len(self.ring) >= 2 and self.ring[1][0] <= int(t_ms) - self.min_idle_ms:
            self.ring.popleft()

    def floor(self) -> Optional[float]:
        if not self.ring:
            return None
        now = self.ring[-1][0]
        best = None
        for t, c in self.ring:
            if t <= now - self.min_idle_ms:
                best = c
            else:
                break
        return best


def pack(flag: int, t_ms: int, digest: int, n: int, used_total: int, used_per_class: Sequence[int],
         evict_ok: Sequence[int], publish_ok: Sequence[int], k: int) -> torch.Tensor:
    if len(used_per_class) > MAX_CLASSES:
        raise EvictError(f"{len(used_per_class)} classes > {MAX_CLASSES}")
    v = torch.zeros(HEADER_LEN + 2 * k, dtype=torch.int64)
    v[H_FLAG], v[H_NFLAG] = int(flag), -int(flag)
    v[H_TMS] = int(t_ms)
    v[H_DIG], v[H_NDIG] = int(digest), -int(digest)
    v[H_N], v[H_NN] = int(n), -int(n)
    v[H_U], v[H_NU] = int(used_total), -int(used_total)
    for r, u in enumerate(used_per_class):
        v[H_UR + r] = int(u)
        v[H_UR + MAX_CLASSES + r] = -int(u)
    for i, ok in enumerate(evict_ok):
        v[HEADER_LEN + i] = 1 if ok else 0
    for i, ok in enumerate(publish_ok):
        v[HEADER_LEN + k + i] = 1 if ok else 0
    return v


@dataclass(frozen=True)
class Agreed:
    flag: int
    t_ms: int
    digest: int
    n: int
    used_total_max: int
    used_total_min: int
    used_per_class_max: Tuple[int, ...]
    evict_ok: Tuple[int, ...]
    publish_ok: Tuple[int, ...]


def unpack(v: torch.Tensor, n_classes: int, k: int) -> Agreed:
    """Read the MIN-reduced vector; any disagreement on replicated state
    (flag, digest, candidate count) is a RankDivergence."""
    x = [int(a) for a in v.tolist()]
    fmin, fmax = x[H_FLAG], -x[H_NFLAG]
    dmin, dmax = x[H_DIG], -x[H_NDIG]
    nmin, nmax = x[H_N], -x[H_NN]
    if fmin != fmax or dmin != dmax or nmin != nmax:
        raise RankDivergence(
            f"{LOG_PREFIX} RANK DIVERGENCE: flag {fmin}/{fmax} digest {dmin}/{dmax} candidates {nmin}/{nmax} "
            "(min/max over the D group). The radix replicas disagree -- crash-stop per raenge-nie-uneins; "
            "nothing was evicted.")
    return Agreed(fmin, x[H_TMS], dmin, nmin, -x[H_NU], x[H_U],
                  tuple(-x[H_UR + MAX_CLASSES + r] for r in range(n_classes)),
                  tuple(x[HEADER_LEN + i] for i in range(nmin)),
                  tuple(x[HEADER_LEN + k + i] for i in range(nmin)))


# ---------------------------------------------------------------------------
# Tree adapter (duck-typed on UnifiedRadixCache)
# ---------------------------------------------------------------------------


def _base_ct():
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    return BASE_COMPONENT_TYPE


def access_counter_now() -> float:
    """The tree's logical access clock (read, never advanced)."""
    from sglang.srt.mem_cache.unified_cache_components import tree_component as _tc

    return float(_tc._LAST_ACCESS_TIME_COUNTER_FLOAT)


def cold_leaves(tree, floor: Optional[float], k: int) -> list:
    """STRUCTURAL device leaves (Full device rows present, no child with Full
    device rows) last touched before ``floor``, coldest first, ordered by a
    REPLICATED key (set/dict iteration order is id()-based).

    Locks are deliberately NOT read here: a write-through pin is released at
    its ack, and acks are rank-local (``_count_ready_acks``), so ``lock_ref``
    -- and with it ``evictable_device_leaves`` -- differs across ranks for a
    while after every publish.  The digest covers only replicated structure;
    "unlocked" is part of the per-rank mask and agreed by MIN."""
    if floor is None:
        return []
    ct = _base_ct()
    out = []
    stack = list(tree.root_node.children.values())
    while stack:
        n = stack.pop()
        kids = list(n.children.values())
        stack.extend(kids)
        v = n.component_data[ct].value
        if v is None or int(v.numel()) == 0 or n.last_access_time >= floor:
            continue
        if any(c.component_data[ct].value is not None for c in kids):
            continue
        out.append(n)
    out.sort(key=lambda n: (float(n.last_access_time), float(n.creation_time)))
    return out[: int(k)]


def describe(nodes: list, mod: int, prefix: Sequence[int]) -> Tuple[List[int], List[List[int]], List[Tuple[int, int, int]]]:
    """lengths, per-class slot counts and digest keys of ``nodes`` -- ONE
    device->host transfer for all of them."""
    from sglang.srt.weg2.d_token_placement import class_of

    if not nodes:
        return [], [], []
    ct = _base_ct()
    vals = [n.component_data[ct].value for n in nodes]
    lens = [int(v.numel()) for v in vals]
    cat = torch.cat([v.reshape(-1).to(torch.int64) for v in vals])
    ncls = len(prefix) - 1
    seg = torch.repeat_interleave(torch.arange(len(vals), device=cat.device),
                                  torch.tensor(lens, device=cat.device))
    cls = class_of(cat, mod, prefix)
    counts = torch.zeros(len(vals) * ncls, dtype=torch.int64, device=cat.device)
    counts.index_add_(0, seg * ncls + cls, torch.ones_like(cls, dtype=torch.int64))
    starts = torch.tensor([0] + lens[:-1], device=cat.device).cumsum(0)
    ends = starts + torch.tensor(lens, device=cat.device) - 1
    blob = torch.cat([counts, cat[starts], cat[ends]]).cpu().tolist()
    m = len(vals)
    cc = [blob[i * ncls:(i + 1) * ncls] for i in range(m)]
    first = blob[m * ncls: m * ncls + m]
    last = blob[m * ncls + m: m * ncls + 2 * m]
    keys = [(lens[i], int(first[i]), int(last[i])) for i in range(m)]
    return lens, cc, keys


def backed_in_l2_locally(tree, node) -> bool:
    """THIS rank holds a complete L2 copy: Full host rows present, no write in
    flight, every component with a device value also has a host value (the
    mamba anchor travels with the KV), and -- under the shared arena -- the
    rows are arena slots (the reader-referenced L2 copy), not staging transit."""
    ct = _base_ct()
    if node not in tree.evictable_device_leaves:      # locked here (reader, load-back or write pin)
        return False
    if getattr(node, "write_through_pending_id", None) is not None:
        return False
    full = node.component_data[ct]
    if full.host_value is None:
        return False
    for comp in getattr(tree, "tree_components", (ct,)):
        cd = node.component_data[comp]
        if cd.value is not None and cd.host_value is None:
            return False
    pool = getattr(getattr(tree, "cache_controller", None), "mem_pool_host", None)
    if getattr(pool, "arena_read", False) and getattr(pool, "arena", None) is not None:
        staging = int(getattr(pool, "staging_rows", 0) or 0)
        hv = full.host_value
        if int(hv.numel()) == 0 or int(hv.min()) < staging:
            return False
    return True


def publishable_locally(tree, node) -> bool:
    ct = _base_ct()
    if getattr(tree, "cache_controller", None) is None or not hasattr(tree, "publish_unbacked_sweep"):
        return False
    if node not in tree.evictable_device_leaves:
        return False
    if getattr(node, "write_through_pending_id", None) is not None:
        return False
    if getattr(node, "l3_present", False):
        return False
    cd = node.component_data[ct]
    return cd.host_value is None and cd.value is not None


def chain_root_first(tree, node) -> list:
    chain = []
    while node is not None and node is not tree.root_node:
        chain.append(node)
        node = node.parent
    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class DKvEvictor:
    """One per D scheduler.  ``step()`` is called every tick; a pass runs every
    ``spec.every`` ticks and always enters exactly one collective."""

    def __init__(self, tree, allocator, spec: EvictSpec, reduce_min: Callable[[torch.Tensor], None],
                 rank: int = 0, now_ms: Callable[[], int] = None, counter: Callable[[], float] = None):
        spec.validate()
        placement = getattr(allocator, "_owner_placement", None)
        if placement is None:
            raise EvictError(f"{LOG_PREFIX} needs --d-token-placement bandwidth armed on the allocator")
        self.tree, self.allocator, self.spec = tree, allocator, spec
        self.placement = placement
        self.reduce_min = reduce_min
        self.rank = int(rank)
        self.now_ms = now_ms or (lambda: int(time.monotonic() * 1000.0))
        self.counter = counter or access_counter_now
        self.clock = ColdClock(spec.min_idle_s * 1000.0)
        self.ticks = 0
        self.passes = 0
        self.prev_used_total: Optional[int] = None
        self.evicted_nodes = 0
        self.evicted_tokens = 0
        self.published = 0
        self.last_line = ""

    # -- replicated inputs ---------------------------------------------------
    def _classes(self):
        prefix = self.placement.prefix_fn()
        if prefix is None or len(prefix) - 1 != len(self.placement.spec.shares):
            return None
        from sglang.srt.weg2.d_token_placement import slot_class_totals

        mod = int(prefix[-1])
        return prefix, mod, slot_class_totals(int(self.allocator.size), mod, prefix)

    def _free_ids(self) -> torch.Tensor:
        parts = [self.allocator.free_pages]
        rel = getattr(self.allocator, "release_pages", None)
        if rel is not None and rel.numel() > 0:
            parts.append(rel)
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def step(self) -> Optional[dict]:
        self.ticks += 1
        if self.ticks % int(self.spec.every) != 0:
            return None
        return self.run_pass()

    def run_pass(self) -> dict:
        from sglang.srt.weg2.d_token_placement import class_counts

        self.passes += 1
        ps = self.placement.spec
        k = int(self.spec.max_candidates)
        cls = self._classes()
        stats = {"pass": self.passes, "evicted": 0, "evicted_tokens": 0, "published": 0}
        counter_now = self.counter()
        used_total = int(self.allocator.size) - int(self.allocator.available_size())
        flag = 0
        nodes: list = []
        lens: List[int] = []
        cc: List[List[int]] = []
        keys: List[Tuple[int, int, int]] = []
        used_pc: List[int] = []
        if cls is not None:
            prefix, mod, totals = cls
            # the trigger comes from the PREVIOUS pass's agreed fill: group-uniform by construction
            if self.prev_used_total is not None and triggered_by_total(
                    totals, self.prev_used_total, ps.shares, ps.fill_switch, ps.lookahead, self.spec.hysteresis):
                flag = 1
                free = class_counts(self._free_ids(), mod, prefix)
                used_pc = [int(t) - int(f) for t, f in zip(totals, free)]
                nodes = cold_leaves(self.tree, self.clock.floor(), k)
                lens, cc, keys = describe(nodes, mod, prefix)
        ev_ok = [1 if backed_in_l2_locally(self.tree, n) else 0 for n in nodes]
        pub_ok = [1 if publishable_locally(self.tree, n) else 0 for n in nodes]
        v = pack(flag, self.now_ms(), candidate_digest(keys) if flag else 0, len(nodes), used_total,
                 used_pc, ev_ok, pub_ok, k)
        self.reduce_min(v)
        ag = unpack(v, len(cls[2]) if cls is not None else 0, k)
        self.clock.record(ag.t_ms, counter_now)
        self.prev_used_total = ag.used_total_max
        stats.update(flag=ag.flag, used=ag.used_total_max, candidates=ag.n)
        if ag.used_total_min != ag.used_total_max:
            stats["used_spread"] = ag.used_total_max - ag.used_total_min
        if not ag.flag or cls is None:
            self._log(stats)
            return stats
        prefix, mod, totals = cls
        p = pressure(totals, ag.used_per_class_max, ps.shares, ps.fill_switch, ps.lookahead, self.spec.hysteresis)
        stats.update(need=p.need, switch=int(p.switch), goal=int(p.goal),
                     agreed_backed=sum(ag.evict_ok), agreed_publishable=sum(ag.publish_ok))
        ev_idx, pub_idx = select_victims(lens, cc, ag.evict_ok, ag.publish_ok, p, self.spec)
        stats["victims"] = [keys[i] for i in ev_idx]
        freed_pc = [0] * len(totals)
        for i in ev_idx:
            node = nodes[i]
            if node not in self.tree.evictable_device_leaves or not self.tree._is_device_leaf(node):
                raise RankDivergence(f"{LOG_PREFIX}: agreed victim {keys[i]} is no longer an unlocked device leaf")
            if node.component_data[_base_ct()].host_value is None:
                raise RankDivergence(f"{LOG_PREFIX}: agreed-backed victim {keys[i]} has no host copy on rank {self.rank}")
            tracker = {ct: 0 for ct in getattr(self.tree, "tree_components", (_base_ct(),))}
            self.tree._evict_to_host(node, tracker)
            stats["evicted"] += 1
            stats["evicted_tokens"] += int(lens[i])
            for r, c in enumerate(cc[i]):
                freed_pc[r] += int(c)
        stats["freed_per_class"] = freed_pc
        if pub_idx:
            first, seen = [], set()
            for i in pub_idx:
                for n in chain_root_first(self.tree, nodes[i]):
                    if id(n) not in seen:
                        seen.add(id(n))
                        first.append(n)
            try:
                res = self.tree.publish_unbacked_sweep(max_issue=int(self.spec.publish_max), first=first,
                                                       chain_only=True) or {}
                stats["published"] = int(res.get("issued", 0) or 0)
            except Exception as e:  # noqa: BLE001 -- a publisher never takes the loop down
                logger.warning("%s publish raised %s: %s", LOG_PREFIX, type(e).__name__, e)
        if stats["evicted"]:
            apply = getattr(self.allocator, "_apply_owner_placement", None)
            if callable(apply):
                apply()
        self.evicted_nodes += stats["evicted"]
        self.evicted_tokens += stats["evicted_tokens"]
        self.published += stats["published"]
        self._log(stats)
        return stats

    def _log(self, stats: dict) -> None:
        if self.rank != 0:
            return
        act = stats.get("evicted", 0) or stats.get("published", 0)
        if not (act or self.passes <= 4 or self.passes % 512 == 0 or "used_spread" in stats):
            return
        line = (f"{LOG_PREFIX} pass={self.passes} flag={stats.get('flag')} used={stats.get('used')} "
                f"switch={stats.get('switch', '-')} goal={stats.get('goal', '-')} need={stats.get('need', 0)} "
                f"candidates={stats.get('candidates', 0)} agreed_backed={stats.get('agreed_backed', 0)} "
                f"evicted={stats.get('evicted', 0)} tokens={stats.get('evicted_tokens', 0)} "
                f"freed_per_class={stats.get('freed_per_class', [])} published={stats.get('published', 0)} "
                f"cumulative={self.evicted_nodes}/{self.evicted_tokens}/{self.published}"
                + (f" used_spread={stats['used_spread']} (free lists differ across ranks; MAX used)"
                   if "used_spread" in stats else ""))
        self.last_line = line
        logger.info(line)


_UNSET = object()


def arm(scheduler) -> Optional[DKvEvictor]:
    spec = spec_from_env()
    if spec is None:
        return None
    tree = getattr(scheduler, "tree_cache", None)
    alloc = getattr(tree, "token_to_kv_pool_allocator", None) or getattr(scheduler, "token_to_kv_pool_allocator", None)
    for need in ("_evict_to_host", "_is_device_leaf", "evictable_device_leaves", "_all_reduce"):
        if not hasattr(tree, need):
            raise RuntimeError(f"{LOG_PREFIX} armed ({ENV}) but the tree cache {type(tree).__name__} has no {need}: "
                               "only UnifiedRadixCache (27B D) is supported")
    if getattr(alloc, "_owner_placement", None) is None:
        raise RuntimeError(f"{LOG_PREFIX} armed ({ENV}) but no --d-token-placement bandwidth on the allocator "
                           "(uneven-DCP D only); refusing rather than evicting for a placement that is not there")

    def _reduce(t: torch.Tensor) -> None:
        tree._all_reduce(t, torch.distributed.ReduceOp.MIN, label="d_kv_evict")

    ev = DKvEvictor(tree, alloc, spec, _reduce, rank=int(getattr(scheduler, "tp_rank", 0) or 0))
    if ev.rank == 0:
        logger.info("%s armed: %s (placement %s)", LOG_PREFIX, spec.to_json(), ev.placement.spec.to_json())
    return ev


def scheduler_step(scheduler) -> None:
    """The scheduler hook; a dict lookup when off."""
    ev = scheduler.__dict__.get("_weg2_d_kv_evictor", _UNSET)
    if ev is None:
        return
    if ev is _UNSET:
        ev = arm(scheduler)
        scheduler._weg2_d_kv_evictor = ev
        if ev is None:
            return
    ev.step()
