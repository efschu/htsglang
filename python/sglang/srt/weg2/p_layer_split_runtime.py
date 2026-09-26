"""Rank-side executor of ``--p-layer-split dynamic`` (27B group P).

Design and evidence: /spinning/gpu-arb/docs/DYN_LAYER_SPLIT.md (sec. 4 the
home-mirror design, sec. 7 the hook points). The pure rules -- geometry,
plan, the Besitzkarte (``LeaderCursor``/``FollowerCursor``), the payload
helpers -- live in ``weg2/p_layer_split.py``; this module binds them to ONE
rank's live objects and is the only state the runtime adds:

* the swing MODULES of this stage (the head layers of stage+1's home span),
  built by ``make_layers`` under their own memory-saver tag
  ``weights_swing`` (outside the exchanged weights family) and DETACHED from
  the model tree after the load, so the flip's exchange plan, census and
  coverage vote never see them;
* the swing SLAB's mirrors: a KV shim (a shallow copy of the full-attention
  sub-pool with its own per-layer buffers, same slot frame) and a state shim
  (a shallow copy of the mamba pool with its own per-layer state, same slot
  frame), routed to by the two pool accessors for swing ids only;
* per forward: the adopted row, the executed layer list, the eager bit, the
  write-back to apply / attach, the pulls to receive / send.

Every hook is a no-op while no runtime is installed, i.e. under ``static``
(no env) -- the default path is byte-identical. Nothing here allocates,
plans or talks to a peer unless ``SGLANG_P_LAYER_SPLIT=dynamic``.

TRANSPORT. Write-back rides the proxy frame downstream (the existing PP
send). The prefix PULL and the swing-weight REFILL go UPSTREAM (stage b+1 ->
b) over the same PP group under their own typed-channel kind
(``layer_pull``), async on the sender; announced one forward ahead so the
receiver's blocking receive always has a posted send behind it (see
``p_layer_split.ForwardRow``). The reverse pairs are warmed at boot
(``warmup_reverse_pairs``), sequenced like ``warmup_p2p_pairs``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import dataclasses
import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import p_layer_split as S

logger = logging.getLogger(__name__)

SWING_WEIGHTS_TAG = "weights_swing"
PULL_KIND = "layer_pull"
EAGER_REASON = "layer_split"
#: async planning: PP0 runs this many HOME chunks while the moving-cut plan of
#: the rest is computed in the background (measured plan cost 0.1-0.7 s)
ASYNC_LEAD_CHUNKS = 2
#: moving-cut search over the best N home chunk candidates (0 = all)
DYN_CANDIDATES = 8

_RT: Optional["RankSplitRuntime"] = None


def active() -> Optional["RankSplitRuntime"]:
    """The installed runtime, or None (static / not group P / not armed)."""
    return _RT


def reset_for_tests() -> None:
    global _RT
    if _RT is not None and _RT._pool is not None:
        _RT._pool.shutdown(wait=False)
    _RT = None


def install(spec: S.SplitSpec, stage: int, *, async_plan: bool = True) -> "RankSplitRuntime":
    global _RT
    _RT = RankSplitRuntime(spec, stage, async_plan=async_plan)
    return _RT


def ensure_from_env(num_hidden_layers: int, pp_rank: Optional[int], pp_size: Optional[int],
                    env=None) -> Optional["RankSplitRuntime"]:
    """Install the runtime from ``SGLANG_P_LAYER_SPLIT*`` the first time the
    TARGET model's layers are built on a stage of the armed group; None for
    any other call (static, a draft model's one-layer stack, a TP group)."""
    if _RT is not None:
        g = _RT.geom
        if (pp_rank == _RT.stage and pp_size == g.stages and num_hidden_layers == g.num_layers):
            return _RT
        return None
    if pp_rank is None or pp_size is None:
        return None
    spec = S.split_from_env(os.environ if env is None else env)
    if spec is None:
        return None
    g = spec.geometry
    if int(pp_size) != g.stages or int(num_hidden_layers) != g.num_layers:
        return None
    rt = install(spec, int(pp_rank))
    logger.info("%s", rt.describe())
    return rt


def swing_window_for(num_hidden_layers: int, pp_rank: Optional[int], pp_size: Optional[int],
                     home_span: Optional[Tuple[int, int]] = None) -> Tuple[int, ...]:
    """``make_layers``' question: which extra layers does this stage build?
    ``home_span`` = the (start, end) the PP partition gave this stage; it
    must BE the spec's home interval (the launcher derives both from one
    cut -- two authors of one cut would build the wrong mirrors)."""
    rt = ensure_from_env(num_hidden_layers, pp_rank, pp_size)
    if rt is None:
        return ()
    if home_span is not None:
        want = rt.geom.home(rt.stage)
        if (int(home_span[0]), int(home_span[1])) != (want.start, want.stop):
            raise S.LayerSplitError(
                f"{S.LOG_TAG}: REFUSED -- stage {rt.stage} was partitioned [{home_span[0]}, {home_span[1]}) "
                f"but the split's home cut {list(rt.geom.home_cuts)} gives [{want.start}, {want.stop})")
    return rt.window


def is_swing_layer(layer_id: Optional[int]) -> bool:
    """For the memory saver's chunk scope: a swing layer's allocations carry
    ``weights_swing``, never the chunk tag of its layer band."""
    rt = _RT
    return rt is not None and layer_id is not None and int(layer_id) in rt.window_set


@dataclasses.dataclass(frozen=True)
class ReqSlot:
    """One request of a forward on THIS rank: its own req_to_token row and
    mamba slot (indices differ per rank; none crosses the wire)."""

    rid: object
    row: int
    mamba: int
    pos: int
    width: int


@dataclasses.dataclass
class ForwardCtx:
    reqs: Tuple[ReqSlot, ...]
    out_cache_loc: Any
    req_to_token: Any


class PPGroupTransport:
    """Upstream pulls over the PP group (typed channel kind ``layer_pull``).
    One tensor per message, async on the sender; the works are kept until
    the next forward (or the sleep) and then waited on."""

    def __init__(self, group):
        self.group = group
        self._works: List[Any] = []

    def send(self, dst: int, tensor) -> None:
        from sglang.srt.distributed.pp_typed_channel import send_typed_tensor_dict

        works = send_typed_tensor_dict(self.group, {"x": tensor}, int(dst), PULL_KIND, async_send=True)
        if works:
            self._works.extend(works)

    def recv(self, src: int):
        from sglang.srt.distributed.pp_typed_channel import recv_typed_tensor_dict

        d = recv_typed_tensor_dict(self.group, PULL_KIND, src=int(src))
        return d["x"]

    def drain(self) -> None:
        works, self._works = self._works, []
        for w in works:
            inner = getattr(w, "work", w)
            try:
                inner.wait()
            except Exception:  # noqa: BLE001 -- a finished work may not wait twice
                pass


def _mirror_state(state, n: int):
    """A State of the same dataclass with every per-layer tensor re-made for
    ``n`` layers (leading axis), slot axis and dtype unchanged."""
    import torch

    kwargs = {}
    for f in dataclasses.fields(state):
        v = getattr(state, f.name)
        if v is None:
            kwargs[f.name] = None
        elif isinstance(v, (list, tuple)):
            kwargs[f.name] = [torch.zeros((n,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device) for t in v]
        else:
            kwargs[f.name] = torch.zeros((n,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
    return type(state)(**kwargs)


def _kv_region(pool):
    """The pool's own KV memory-saver region (the mirrors sleep and wake
    with the pool they shadow); a null context without a saver."""
    import contextlib

    adapter = getattr(pool, "memory_saver_adapter", None)
    if adapter is None:
        return contextlib.nullcontext()
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

    return adapter.region(GPU_MEMORY_TYPE_KV_CACHE)


def _state_tensors(layer_state) -> Tuple[Any, ...]:
    """Every tensor of ONE layer's state (``State.at_layer_idx``), in field
    order -- identical order on home and mirror (same dataclass)."""
    out = []
    for f in dataclasses.fields(layer_state):
        v = getattr(layer_state, f.name)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return tuple(out)


def _module_tensors(module) -> List[Tuple[str, Any]]:
    """A layer module's device tensors (parameters, then buffers) by name, in
    registration order -- the refill order on both ends."""
    out = []
    for name, p in module.named_parameters(recurse=True):
        out.append(("p:" + name, p))
    for name, b in module.named_buffers(recurse=True):
        if b is not None:
            out.append(("b:" + name, b))
    return out


def _checksum(t) -> int:
    """Position-weighted byte sum in bounded chunks (boot-time layout check)."""
    import torch

    flat = t.detach().reshape(-1).contiguous().view(torch.uint8)
    total = 0
    step = 1 << 24
    for a in range(0, flat.numel(), step):
        seg = flat[a:a + step].to(torch.int64)
        w = (torch.arange(seg.numel(), device=seg.device, dtype=torch.int64) + a) % 65521 + 1
        total = (total + int((seg * w).sum().item())) % (1 << 61)
    return total


class RankSplitRuntime:
    def __init__(self, spec: S.SplitSpec, stage: int, *, async_plan: bool = True):
        self.spec = spec
        self.geom = spec.geometry
        self.stage = int(stage)
        self.stages = self.geom.stages
        self.families = self.geom.families
        self.follower = S.FollowerCursor(spec, self.stage)
        self._pool = None
        self.leader: Optional[S.LeaderCursor] = None
        if self.stage == 0:
            if async_plan:
                self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                                   thread_name_prefix="p-layer-split-plan")
            self.leader = S.LeaderCursor(spec, on_plan=self._on_plan, executor=self._pool,
                                         lead_chunks=ASYNC_LEAD_CHUNKS, dyn_candidates=DYN_CANDIDATES)
        self.window = tuple(self.geom.swing_window(self.stage))
        self.window_set = frozenset(self.window)
        self.win_attn = tuple(l for l in self.window if self.families[l] == S.FAMILY_ATTENTION)
        self.win_gdn = tuple(l for l in self.window if self.families[l] == S.FAMILY_LINEAR)
        # the window of the stage upstream = what THIS stage is home of and sends
        self.up_window = tuple(self.geom.swing_window(self.stage - 1)) if self.stage > 0 else ()
        self.modules: Dict[int, Any] = {}
        self.home_layers = None
        self.capture_ids: frozenset = frozenset()
        # pools
        self._kv_pool = None
        self._state_pool = None
        self._kv_shim = None
        self._kv_dense: Dict[int, int] = {}
        self._state_shim = None
        self._state_dense: Dict[int, int] = {}
        self.transport = None
        # rows / forward state
        self._rows: "collections.deque" = collections.deque()
        self.row: Optional[S.ForwardRow] = None
        self.ctx: Optional[ForwardCtx] = None
        self.exec_layers: Optional[Tuple[int, ...]] = None
        self.eager = False
        self._recv_next: List[Tuple[S.Pull, ReqSlot]] = []
        self._recv_now: List[Tuple[S.Pull, ReqSlot]] = []
        self._recv_at: Optional[int] = None
        self._send_now: List[Tuple[S.Pull, ReqSlot]] = []
        self._send_at: Optional[int] = None
        self._refill_next = False
        self._req_to_token = None
        self.weights_ready = True
        self.counters: Dict[str, int] = collections.Counter()

    # ------------------------------------------------------------------ info

    def describe(self) -> str:
        g = self.geom
        return (f"{S.LOG_TAG} runtime stage={self.stage}/{self.stages} home={list(g.home(self.stage))[:1]}.."
                f"{list(g.home(self.stage))[-1:]} window={list(self.window)} up_window={list(self.up_window)} "
                f"leader={'yes' if self.leader is not None else 'no'} digest={self.follower.digest}")

    def _on_plan(self, key, plan: S.SplitPlan, pos: int, end: int) -> None:
        self.counters["plans"] += 1
        if plan.candidate.startswith("home") and self.counters["plans"] > 8:
            return
        logger.info("%s", S.plan_line(plan, key=key, end=end))

    # ---------------------------------------------------------- construction

    def register_module(self, idx: int, module) -> None:
        if idx not in self.window_set:
            raise S.LayerSplitError(f"{S.LOG_TAG}: layer {idx} is not in stage {self.stage}'s window")
        self.modules[int(idx)] = module

    def detach(self, layers, missing_factory: Callable[[int], Any]) -> int:
        """After the load (weights loaded and post-processed while the swing
        modules were ordinary members of ``layers``): take them OUT of the
        model tree -- the exchange, its census and coverage vote enumerate
        ``named_parameters`` and must see only home layers."""
        self.home_layers = layers
        n = 0
        for idx in self.window:
            mod = layers[idx]
            if mod is None or type(mod).__name__ == "PPMissingLayer":
                raise S.LayerSplitError(f"{S.LOG_TAG}: swing layer {idx} was not built on stage {self.stage}")
            self.modules[idx] = mod
            layers[idx] = missing_factory(idx)
            n += 1
        return n

    def module(self, idx: int, default):
        m = self.modules.get(idx)
        return default if m is None else m

    def mark_capture(self, ids: Sequence[int]) -> None:
        self.capture_ids = frozenset(int(i) for i in ids)
        for idx, mod in self.modules.items():
            mod._is_layer_to_capture = idx in self.capture_ids

    def swing_weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for m in self.modules.values() for _n, t in _module_tensors(m))

    # ----------------------------------------------------------------- pools

    def bind_pools(self, kv_pool, req_to_token_pool) -> Dict[str, int]:
        """Allocate the mirrors (call inside the KV memory-saver region, so
        they sleep with the pool) and install the accessor routes."""
        name = type(kv_pool).__name__ + "/" + type(getattr(kv_pool, "full_kv_pool", None)).__name__
        if "Unified" in name or "PageMajor" in name or not hasattr(kv_pool, "full_attention_layer_id_mapping"):
            raise S.LayerSplitError(
                f"{S.LOG_TAG}: REFUSED on stage {self.stage} -- the swing mirror supports the static "
                f"HybridLinearKVPool with a list-form MHA sub-pool, not {name}")
        if self.win_gdn and getattr(req_to_token_pool, "enable_mamba_extra_buffer", False):
            raise S.LayerSplitError(
                f"{S.LOG_TAG}: REFUSED on stage {self.stage} -- the mamba extra_buffer strategy keeps "
                f"per-request ping-pong track slots the swing write-back does not carry; group P must "
                f"run --mamba-radix-cache-strategy no_buffer with swing GDN layers")
        self._kv_pool = kv_pool
        self._state_pool = req_to_token_pool
        out = {"kv_bytes": 0, "state_bytes": 0}
        if self.win_attn:
            import torch

            fk = kv_pool.full_kv_pool
            shim = copy.copy(fk)
            with _kv_region(fk):
                shim.k_buffer = [torch.zeros_like(fk.k_buffer[0]) for _ in self.win_attn]
                shim.v_buffer = [torch.zeros_like(fk.v_buffer[0]) for _ in self.win_attn]
            shim.start_layer = 0
            shim.layer_num = len(self.win_attn)
            shim._local_slot_of = None
            shim.layer_transfer_counter = None
            self._kv_shim = shim
            self._kv_dense = {l: j for j, l in enumerate(self.win_attn)}
            kv_pool._p_layer_split_kv = (shim, dict(self._kv_dense))
            out["kv_bytes"] = sum(t.numel() * t.element_size() for t in shim.k_buffer + shim.v_buffer)
        if self.win_gdn:
            mp = req_to_token_pool.mamba_pool
            shim_m = copy.copy(mp)
            with _kv_region(mp):
                shim_m.mamba_cache = _mirror_state(mp.mamba_cache, len(self.win_gdn))
            self._state_shim = shim_m
            self._state_dense = {l: j for j, l in enumerate(self.win_gdn)}
            req_to_token_pool._p_layer_split_state = (shim_m, dict(self._state_dense))
            out["state_bytes"] = sum(t.numel() * t.element_size()
                                     for t in _state_tensors(shim_m.mamba_cache))
        return out

    def kv_of(self, l: int):
        """Raw (store-dtype) K/V buffers of attention layer ``l`` on this
        rank: the mirror for a swing layer, the pool's own for a home one."""
        if l in self._kv_dense:
            j = self._kv_dense[l]
            return self._kv_shim.k_buffer[j], self._kv_shim.v_buffer[j]
        kv = self._kv_pool
        fk = kv.full_kv_pool
        slot = fk.local_slot(kv.full_attention_layer_id_mapping[l])
        return fk.k_buffer[slot], fk.v_buffer[slot]

    def state_of(self, l: int):
        if l in self._state_dense:
            return _state_tensors(self._state_shim.mamba_cache.at_layer_idx(self._state_dense[l]))
        rp = self._state_pool
        return _state_tensors(rp.mamba_pool.mamba_cache.at_layer_idx(rp.mamba_map[l]))

    # ------------------------------------------------------------- transport

    def bind_transport(self, transport) -> None:
        self.transport = transport

    def warmup_reverse_pairs(self, group) -> None:
        """Build the upstream pairs (k -> k-1) before any forward needs them,
        one pair at a time with a barrier between (the lazy pair needs both
        ends; ``GroupCoordinator.warmup_p2p_pairs`` measured the symmetric
        orders deadlock)."""
        import torch

        for grp, dev in ((group.device_group, group.device), (group.cpu_group, torch.device("cpu"))):
            if grp is None:
                continue
            out = torch.zeros(1, dtype=torch.uint8, device=dev)
            inp = torch.zeros(1, dtype=torch.uint8, device=dev)
            for k in range(group.world_size - 1, 0, -1):
                if group.rank_in_group == k:
                    torch.distributed.send(out, group.ranks[k - 1], group=grp)
                elif group.rank_in_group == k - 1:
                    torch.distributed.recv(inp, group.ranks[k], group=grp)
                torch.distributed.barrier(group=grp)
        logger.info("%s reverse pairs warmed on stage %d", S.LOG_TAG, self.stage)

    def verify_refill_layout(self) -> None:
        """Boot: the swing modules must be BYTE-identical to home's (the
        refill after every wake copies home's post-processed tensors). Home
        sends name/shape/dtype digest + per-tensor checksums upstream; a
        mismatch refuses the boot here instead of computing on another
        card's repack layout later."""
        import torch

        def desc(mods):
            names, sums = [], []
            for idx in sorted(mods):
                for n, t in _module_tensors(mods[idx]):
                    names.append(f"{idx}:{n}:{tuple(t.shape)}:{t.dtype}")
                    sums.append(_checksum(t))
            h = hashlib.sha256("|".join(names).encode()).digest()
            return torch.tensor(list(h) + [0] * 0, dtype=torch.uint8), sums

        dev = None
        if self.up_window and self.home_layers is not None:
            mods = {l: self.home_layers[l] for l in self.up_window}
            h, sums = desc(mods)
            dev = next(iter(_module_tensors(mods[self.up_window[0]])))[1].device
            self.transport.send(self.stage - 1, h.to(dev))
            self.transport.send(self.stage - 1, torch.tensor(sums, dtype=torch.int64, device=dev))
        if self.window:
            h, sums = desc(self.modules)
            dev = next(iter(_module_tensors(self.modules[self.window[0]])))[1].device
            got_h = self.transport.recv(self.stage + 1)
            got_s = self.transport.recv(self.stage + 1)
            if not torch.equal(got_h.cpu(), h.cpu()):
                raise S.LayerSplitError(
                    f"{S.LOG_TAG}: REFUSED -- stage {self.stage}'s swing modules {list(self.window)} do not "
                    f"have home's tensor names/shapes/dtypes (a card-specific repack layout); the refill "
                    f"after a wake would copy foreign bytes")
            mine = torch.tensor(sums, dtype=torch.int64)
            if not torch.equal(got_s.cpu(), mine):
                bad = int((got_s.cpu() != mine).sum())
                raise S.LayerSplitError(
                    f"{S.LOG_TAG}: REFUSED -- {bad} swing tensor(s) on stage {self.stage} differ bytewise "
                    f"from home after load (card-specific post-processing); refill would be wrong")
        if self.transport is not None:
            self.transport.drain()
        logger.info("%s refill layout verified on stage %d (window=%s up_window=%s)", S.LOG_TAG,
                    self.stage, list(self.window), list(self.up_window))

    # ------------------------------------------------------------------ rows

    def leader_width(self, rid, pos: int, end: int) -> int:
        return self.leader.next_width(rid, int(pos), int(end))

    def leader_decide(self, entries: Sequence[Tuple[object, int, int]]) -> S.ForwardRow:
        """PP0, at the #791 decision: decide this forward's row and queue it
        for PP0's own forward (the row travels downstream with the frame)."""
        row = self.leader.decide(list(entries))
        self._rows.append(row)
        self.counters["rows"] += 1
        if row.cut != self.geom.home_cuts:
            self.counters["rows_moved"] += 1
        return row

    def push_row(self, raw) -> None:
        """Downstream, at the proxy receive: the row PP0 decided for this
        frame's forward."""
        self._rows.append(raw)

    # --------------------------------------------------------------- forward

    def pre_forward(self, ctx: ForwardCtx, proxy: Optional[Dict[str, Any]]) -> None:
        import torch

        if self.transport is not None:
            self.transport.drain()
        raw = self._rows.popleft() if self._rows else None
        row = self.follower.adopt(raw, len(ctx.reqs))
        if row.order and row.order != S.batch_order([r.rid for r in ctx.reqs]):
            raise S.LayerSplitDivergence(
                f"{S.LOG_TAG}: stage {self.stage} runs batch {[r.rid for r in ctx.reqs]} in another order "
                f"than PP0 decided row {row.version} for (the pull slots index that order)")
        self.row, self.ctx = row, ctx
        self._req_to_token = ctx.req_to_token
        # pulls: announced by the previous row -> received now; announced by
        # this row -> sent now (home side) / received next forward
        self._recv_now, self._recv_next = self._recv_next, [
            (p, ctx.reqs[p[0]]) for p in row.pulls if p[1] == self.stage]
        self._send_now = [(p, ctx.reqs[p[0]]) for p in row.pulls if p[1] == self.stage - 1]
        refill_now, self._refill_next = self._refill_next, bool(row.refill and self.window)
        # 1. refill: home sends at the START of the announcing forward, the
        #    mirror receives at the start of its next one (before any layer)
        if row.refill and self.up_window:
            self._send_refill()
        if refill_now:
            self._recv_refill()
        executed = self.follower.executed(row)
        swing = self.follower.swing(row)
        missing = [l for l in swing if l not in self.modules
                   or (self.families[l] == S.FAMILY_ATTENTION and l not in self._kv_dense)
                   or (self.families[l] == S.FAMILY_LINEAR and l not in self._state_dense)]
        if missing:
            raise S.LayerSplitError(
                f"{S.LOG_TAG}: REFUSED -- stage {self.stage} is told to execute swing layers {missing} "
                f"without their slab (module / KV mirror / state mirror not bound at boot; row "
                f"{row.version} cut {row.cut})")
        if swing and not self.weights_ready:
            raise S.LayerSplitDivergence(
                f"{S.LOG_TAG}: stage {self.stage} would execute swing layers {list(swing)} before the "
                f"refill after a wake landed (row {row.version})")
        # 2. write-back of the chunk's mirror rows, before any home layer
        incoming = self.follower.incoming(row)
        payload = {}
        if proxy is not None:
            for k in [k for k in proxy if isinstance(k, str) and S.is_writeback_key(k)]:
                payload[k] = proxy.pop(k)
        if incoming:
            attn = [l for l in incoming if self.families[l] == S.FAMILY_ATTENTION]
            gdn = [l for l in incoming if self.families[l] == S.FAMILY_LINEAR]
            S.apply_writeback(payload, attn, gdn, self.kv_of, self.state_of, ctx.out_cache_loc,
                              self._state_idx(ctx), row.gdn_wb)
            self.counters["writeback_applied"] += 1
        elif payload:
            raise S.LayerSplitDivergence(
                f"{S.LOG_TAG}: stage {self.stage} got write-back {sorted(payload)} for a row that swings "
                f"nothing onto it (row {row.version} cut {row.cut})")
        # 3. fresh requests: zero the mirror state rows (the pool clears only
        #    its own layers' rows on a new slot)
        if self.win_gdn and self._state_shim is not None:
            fresh = [r.mamba for r in ctx.reqs if r.pos == 0 and r.mamba >= 0]
            if fresh:
                idx = torch.tensor(fresh, dtype=torch.long, device=self._state_device())
                for l in self.win_gdn:
                    for t in self.state_of(l):
                        t.index_fill_(0, idx, 0)
        # 4. where the pulls happen inside the layer loop
        self._recv_at = None
        if self._recv_now:
            lo = min(p[2] for p, _r in self._recv_now)
            self._recv_at = lo if lo in executed else None
        self._send_at = None
        if self._send_now:
            hi = max(p[3] for p, _r in self._send_now)
            if (hi - 1) not in executed:
                raise S.LayerSplitDivergence(
                    f"{S.LOG_TAG}: stage {self.stage} must send pulled layers up to {hi - 1} but does not "
                    f"execute them in row {row.version} (cut {row.cut})")
            self._send_at = hi - 1
        self.exec_layers = executed
        # a home stage that must SEND a pull does so right after the last
        # pulled layer (the planner's precedence) -- a layer hook a captured
        # graph cannot run, so that one forward is eager too
        self.eager = (not self.follower.graph_ok(row)) or bool(self._send_now)
        if self.eager:
            self.counters["eager_forwards"] += 1

    def _state_idx(self, ctx: ForwardCtx):
        import torch

        return torch.tensor([r.mamba for r in ctx.reqs], dtype=torch.long, device=self._state_device())

    def _state_device(self):
        if self._state_pool is not None:
            return self._state_pool.mamba_pool.mamba_cache.temporal.device
        return getattr(self.ctx.out_cache_loc, "device", "cpu")

    def layer_ids(self, default):
        """The model loop's iterator: this forward's executed layers, or the
        default (``owned_layer_ids``) when no row is active."""
        return default if self.exec_layers is None else self.exec_layers

    def before_layer(self, idx: int) -> None:
        if self._recv_at is not None and idx == self._recv_at:
            self._do_recv()

    def after_layer(self, idx: int) -> None:
        if self._send_at is not None and idx == self._send_at:
            self._do_send()

    def force_eager(self) -> bool:
        return bool(self.eager)

    def post_forward(self, out: Optional[Dict[str, Any]]) -> None:
        row, ctx = self.row, self.ctx
        if row is None:
            return
        if self._recv_now:
            self._do_recv()                 # the pulled layers were not executed: land them anyway
        if self._send_now:
            # the layer hook did not run (a graph replayed after all): the
            # state after the whole forward is the same state, only later
            self.counters["late_sends"] += 1
            self._do_send()
        swing = self.follower.swing(row)
        if swing:
            if out is None:
                raise S.LayerSplitDivergence(f"{S.LOG_TAG}: stage {self.stage} has write-back but no frame")
            attn = [l for l in swing if self.families[l] == S.FAMILY_ATTENTION]
            gdn = [l for l in swing if self.families[l] == S.FAMILY_LINEAR]
            out.update(S.writeback_payload(attn, gdn, self.kv_of, self.state_of, ctx.out_cache_loc,
                                           self._state_idx(ctx), row.gdn_wb))
            self.counters["writeback_sent"] += 1
        if out is not None and self.stage < self.stages - 1:
            out[S.ROW_KEY] = row.encode()
        self.row, self.ctx, self.exec_layers, self.eager = None, None, None, False

    # ----------------------------------------------------------------- pulls

    def _do_recv(self) -> None:
        todo, self._recv_now, self._recv_at = self._recv_now, [], None
        for pull, r in todo:
            n = S.pull_recv(pull, self.families, self.kv_of, self.state_of,
                            self._row_loc(r), self._one_idx(r), self._recv_like)
            self.counters["pull_recv_bytes"] += n
            self.counters["pulls_received"] += 1

    def _do_send(self) -> None:
        todo, self._send_now, self._send_at = self._send_now, [], None
        for pull, r in todo:
            n = S.pull_send(pull, self.families, self.kv_of, self.state_of,
                            self._row_loc(r), self._one_idx(r), self._send_up)
            self.counters["pull_send_bytes"] += n
            self.counters["pulls_sent"] += 1

    def _row_loc(self, r: ReqSlot):
        rtt = self._req_to_token
        return lambda a, b: S.request_loc(rtt, r.row, a, b)

    def _one_idx(self, r: ReqSlot):
        import torch

        return torch.tensor([r.mamba], dtype=torch.long, device=self._state_device())

    def _send_up(self, t) -> None:
        self.transport.send(self.stage - 1, t)

    def _recv_like(self, like):
        t = self.transport.recv(self.stage + 1)
        if tuple(t.shape) != tuple(like.shape) or t.dtype != like.dtype:
            raise S.LayerSplitDivergence(
                f"{S.LOG_TAG}: stage {self.stage} pull got {tuple(t.shape)}/{t.dtype}, expected "
                f"{tuple(like.shape)}/{like.dtype}")
        return t

    # ---------------------------------------------------------------- refill

    def _send_refill(self) -> None:
        n = 0
        for l in self.up_window:
            for _name, t in _module_tensors(self.home_layers[l]):
                self._send_up(t.detach())
                n += t.numel() * t.element_size()
        self.counters["refill_sent_bytes"] += n

    def _recv_refill(self) -> None:
        n = 0
        for l in self.window:
            for _name, t in _module_tensors(self.modules[l]):
                got = self._recv_like(t)
                t.data.copy_(got)
                n += t.numel() * t.element_size()
        self.weights_ready = True
        self.counters["refill_recv_bytes"] += n
        logger.info("%s stage %d swing weights refilled from home (%d B)", S.LOG_TAG, self.stage, n)

    # ------------------------------------------------------------------ flip

    def on_sleep(self) -> None:
        """Before the KV pages are released: land whatever the last forward
        announced (its sends are already posted) and finish our own sends."""
        if self.ctx is None and self._recv_next:
            self._recv_now, self._recv_next = self._recv_next, []
            self._do_recv()
        if self._refill_next:
            self._refill_next = False
            self._recv_refill()
        if self.transport is not None:
            self.transport.drain()

    def on_wake(self) -> None:
        """After the swing tag was resumed: its content is undefined until
        the refill; PP0 forces the next forward home and announces it."""
        if self.window:
            self.weights_ready = False
        if self.leader is not None:
            self.leader.reset()
            if any(self.geom.window):
                self.leader.refill_needed = True
        self._rows.clear()
        self.counters["wakes"] += 1

    def census_line(self) -> str:
        c = self.counters
        st = self.leader.stats if self.leader is not None else {}
        return (f"{S.LOG_TAG} census stage={self.stage} rows={c['rows']} moved={c['rows_moved']} "
                f"eager={c['eager_forwards']} wb_sent={c['writeback_sent']} wb_applied={c['writeback_applied']} "
                f"pulls_sent={c['pulls_sent']} ({c['pull_send_bytes']} B) pulls_recv={c['pulls_received']} "
                f"({c['pull_recv_bytes']} B) refill_sent={c['refill_sent_bytes']} B "
                f"refill_recv={c['refill_recv_bytes']} B wakes={c['wakes']} plan={dict(st)}")


# ---------------------------------------------------------------------------
# model-runner glue


def swing_extra_layer_counts(start_layer: int, end_layer: int) -> Tuple[int, int]:
    """The SIZER's post (pool_configurator / the mamba budget): (attention,
    GDN) layers this stage holds BEYOND its home interval -- the swing
    mirrors are pool-shaped, so they are priced exactly like owned layers of
    their family. (0, 0) under static or for a runner whose interval is not
    the armed stage's home (the draft, group D)."""
    rt = _RT
    if rt is None or not rt.window:
        return 0, 0
    home = rt.geom.home(rt.stage)
    if (int(start_layer), int(end_layer)) != (home.start, home.stop):
        return 0, 0
    return len(rt.win_attn), len(rt.win_gdn)


def detach_from_model(model) -> int:
    """After the load: move this stage's swing modules out of the model tree
    (see ``RankSplitRuntime.detach``). 0 under static or on a model without
    swing layers (the draft, group D)."""
    rt = _RT
    if rt is None or model is None:
        return 0
    target = None
    for mod in model.modules():
        layers = getattr(mod, "layers", None)
        if layers is None or not hasattr(layers, "__len__"):
            continue
        if getattr(layers, "swing_layers", None) or (not rt.window and len(layers) == rt.geom.num_layers):
            target = layers
            break
    if target is None:
        return 0
    if not rt.window:
        # a stage without a window (the last) is still HOME of its upstream
        # neighbour's window: it sends pulls and the refill from these modules
        rt.home_layers = target
        return 0
    from sglang.srt.layers.utils import PPMissingLayer

    n = rt.detach(target, lambda idx: PPMissingLayer(return_tuple=False, unowned_layer_id=idx))
    logger.info("%s stage %d detached %d swing module(s) %s from the model tree (%d B, tag %s)",
                S.LOG_TAG, rt.stage, n, list(rt.window), rt.swing_weight_bytes(), SWING_WEIGHTS_TAG)
    return n


def bind_pools(kv_pool, req_to_token_pool) -> None:
    """After the KV pools exist (inside the KV memory-saver region)."""
    rt = _RT
    if rt is None:
        return
    got = rt.bind_pools(kv_pool, req_to_token_pool)
    logger.info("%s stage %d swing slab: weights=%d B kv_mirror=%d B state_mirror=%d B (window=%s)",
                S.LOG_TAG, rt.stage, rt.swing_weight_bytes(), got["kv_bytes"], got["state_bytes"],
                list(rt.window))


def boot_link(pp_group) -> None:
    """Scheduler boot, right after ``warmup_p2p_pairs``: every P rank (in
    lockstep) warms the upstream pairs, binds the transport and verifies the
    swing modules are byte-identical to home's."""
    rt = _RT
    if rt is None:
        return
    rt.bind_transport(PPGroupTransport(pp_group))
    rt.warmup_reverse_pairs(pp_group)
    rt.verify_refill_layout()


# ---------------------------------------------------------------------------
# scheduler-side glue (duck-typed: ScheduleBatch / Req)


def ctx_from_batch(batch, req_to_token) -> ForwardCtx:
    """This forward's per-request slots on THIS rank, in batch order (== the
    #791 decision order after ``order_batch_by_schedule``)."""
    reqs = []
    for req in batch.reqs:
        prefix = getattr(req, "prefix_indices", None)
        pos = 0 if prefix is None else len(prefix)
        mid = getattr(req, "mamba_pool_idx", None)
        try:
            mamba = int(mid.reshape(-1)[0].item()) if hasattr(mid, "reshape") else (-1 if mid is None else int(mid))
        except Exception:  # noqa: BLE001
            mamba = -1
        reqs.append(ReqSlot(req.rid, int(req.req_pool_idx), mamba, int(pos), int(req.extend_input_len)))
    return ForwardCtx(tuple(reqs), batch.out_cache_loc, req_to_token)


__all__ = [
    "SWING_WEIGHTS_TAG", "PULL_KIND", "EAGER_REASON", "active", "install", "ensure_from_env",
    "swing_window_for", "is_swing_layer", "ReqSlot", "ForwardCtx", "PPGroupTransport", "RankSplitRuntime",
    "ctx_from_batch", "reset_for_tests", "detach_from_model", "bind_pools", "boot_link",
    "swing_extra_layer_counts",
]
