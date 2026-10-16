# SPDX-License-Identifier: Apache-2.0
"""NF-968-ROWLB 1014: a follower that EXECUTES PP0's row takes no room verdict of its own for the load-back.

THE DEATHS (h16j = 73f6a4e110, NF P group PP3/TP1, row authority ON since Fix 1 31aa655fff):

* 19:43:19Z (P.log ``..._73f6a4e110_1014_193949`` 12567-12998), rid weg2-0-1, the FIRST requests after the boot:
  weg2-0-1 (121600 from L3) and weg2-0-2 (108800 from L3), one session, common prefix 3008. The two store reads
  completed per stage in a different order: PP0 ``#1423 INSERT-PLACED req=weg2-0-1 matched=0 inserted=121600``,
  PP1/PP2 ``matched=3008 inserted=118592`` (weg2-0-2 was inserted there first). PP0 loaded weg2-0-2, then weg2-0-1
  (``LB-VERDICT VERDICT kv_tokens=118592 ok`` -> ``#988 LOADBACK prefix moved to 121600``) and sent the row. PP1,
  executing that row: ``#1045 FLOOR PUBLISHED floor=153216``, ``WEG2-LOADBACK-EVICT kv_tokens=118592 floor=34560 ...
  this pass refuses`` -> ``#968 PREFIX MATERIALISATION SHORTFALL ... prefix_len=121600, this rank holds 0 ... (load-back
  adopted a GDN anchor at a depth the schedule does not name ...)``.
* 19:49:41Z (``..._1014_194609`` 12385-12998), rid weg2-0-2: the mirror order (PP0 matched=0 for weg2-0-2, the
  followers 3008), ``PR PP-ROOM-CAP binding_rank=0 cap=66176`` (PP0's own room after its load), PP1
  ``WEG2-LOADBACK-EVICT kv_tokens=85184 floor=66240`` -> the same #968 stop.

THE ROOT (the planner's tree-shape hypothesis is refuted, see the report): the matched 0/3008 split is the per-stage
completion order of two concurrent same-session store reads and is harmless -- the host chain each stage loads back
is the same length (118592 / 85184 on PP0's verdict line AND on PP1), and h16h carried the same split without harm
(weg2-18-134 PP0 matched=0, PP1/PP2 3008). What killed the group is the follower's ROOM VERDICT inside ``load_back``:
the floor it read was ``min(own floor, agreed R)`` and the R it held rides a LATER list than the row's pass -- PP0's
room after it had loaded this very prefix. ``local_pp_room`` answered "every stage refuses this pass" (PP0 had not;
it had loaded), the row's load-back is not gated (Fix 1), so the xsn285 refuse-this-pass returned False; the mamba
transfer had been BUILT already (the anchor flag), hence the GDN text.

THE RULE: a row-executed load-back takes no room verdict; it makes its own room on its live pool (evict the shortfall,
load in this pass -- upstream ``HiRadixCache.load_back``) or the group stops by name (``LB-VERDICT ROW-ROOM SHORT``).

Every test drives the real code: ``UnifiedRadixCache.load_back`` / ``init_load_back``, ``execute_scheduled_prefix``,
``pp_room_vote.RoomBook``/``stamp_cap``/``absorb_cap``/``set_cap``, ``pp_loadback_verdict`` -- on stand-in trees.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers import pp_admission_congruence as PAC  # noqa: E402
from sglang.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams  # noqa: E402
from sglang.srt.weg2 import pp_loadback_verdict as LBV  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as PR  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

BASE = URC.BASE_COMPONENT_TYPE
MAMBA = URC.ComponentType.MAMBA

COMMON = 3008                  # common=3045 tokens, 3008 page-aligned (#1423 matched=3008)
R1, R2 = "weg2-0-1", "weg2-0-2"
LEN = {R1: 121600, R2: 108800}  # Tod 1: told 121600 / 108800
POOL = 262144
PP0_FREE_BEFORE_R1 = 153152     # PP0 rem_total_tokens=153087 before weg2-0-1's load, page-aligned
FOLLOWER_FREE = 153216          # PP1 '#1045 FLOOR PUBLISHED n=3 floor=153216' in the row's pass
FOLLOWER_EVICTABLE = 105920     # PP1 'WEG2-LOADBACK-EVICT ... requested=105920'


# ---------------------------------------------------------------------------
# a stand-in radix tree on which the REAL load_back / init_load_back run
# ---------------------------------------------------------------------------


class _CD:
    def __init__(self):
        self.value = None
        self.host_value = None


class _Node:
    _ids = 0

    def __init__(self, parent, label, n):
        _Node._ids += 1
        self.id = _Node._ids
        self.parent = parent
        self.label = label
        self.n = int(n)
        self.children = {}
        self.component_data = {BASE: _CD(), MAMBA: _CD()}
        if parent is not None:
            # host-only after a store read (L3 -> host): evicted on the device, backed on the host
            self.component_data[BASE].host_value = torch.arange(self.n, dtype=torch.int64)
            self.component_data[MAMBA].host_value = torch.zeros(1, dtype=torch.int64)

    @property
    def evicted(self):
        return self.component_data[BASE].value is None


class _Alloc:
    def __init__(self, avail):
        self.avail = int(avail)

    def available_size(self):
        return self.avail


class _Lock:
    delta = 0

    def to_dec_params(self):
        return None


class _KvXfer:
    def __init__(self, nodes):
        self.nodes_to_load = list(nodes)
        self.host_indices = torch.cat([n.component_data[BASE].host_value for n in nodes]) if nodes else torch.empty(0)
        self.device_indices = None


def _evicted_chain(node):
    chain = []
    while node is not None and node.parent is not None and node.evicted:
        chain.append(node)
        node = node.parent
    return list(reversed(chain))


class _BaseComp:
    component_type = BASE

    def build_hicache_transfers(self, node, phase, req=None):
        return [_KvXfer(_evicted_chain(node))]

    def commit_hicache_transfer(self, node, phase, xfers):
        for x in xfers:
            off = 0
            for n in x.nodes_to_load:
                n.component_data[BASE].value = torch.as_tensor(x.device_indices[off:off + n.n])
                off += n.n


class _MambaComp:
    """The #968 FIX-3 writer (mamba_component.build_hicache_transfers): the flag is set when the transfer is BUILT."""

    component_type = MAMBA

    def build_hicache_transfers(self, node, phase, req=None):
        if req is not None:
            req.mamba_loadback_anchor_adopted = True
        return [SimpleNamespace(name="mamba")]

    def commit_hicache_transfer(self, node, phase, xfers):
        pass


class _Controller:
    def __init__(self, tree):
        self.tree = tree
        self.loads = []

    def load(self, host_indices, node_id, extra_pools=None):
        n = len(host_indices)
        alloc = self.tree.token_to_kv_pool_allocator
        if alloc.avail < n:
            return None
        alloc.avail -= n
        self.loads.append(n)
        return torch.arange(n, dtype=torch.int64)


class _Stage:
    """One P stage: the real UnifiedRadixCache.load_back / init_load_back on a stand-in tree."""

    load_back = URC.UnifiedRadixCache.load_back
    init_load_back = URC.UnifiedRadixCache.init_load_back
    enable_storage = False
    prefetch_progress_is_collective_free = None

    def __init__(self, rank, avail, evictable=0):
        self.rank = rank
        self.root_node = _Node(None, "root", 0)
        self.root_node.component_data[BASE].value = torch.empty(0, dtype=torch.int64)
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable = int(evictable)
        self.evicted_log = []
        self.uniform_avail_floor = int(avail)
        self.cache_controller = _Controller(self)
        self.components = {BASE: _BaseComp(), MAMBA: _MambaComp()}
        self._components_tuple = (self.components[BASE], self.components[MAMBA])
        self.load_back_threshold = 10
        self.ongoing_load_back = {}
        self.metrics_collector = None
        self._empty_match_result = SimpleNamespace(device_indices=torch.empty(0, dtype=torch.int64))
        self.locks = 0
        # the carrierless P form: tp group of one under pp 3 -> the local-PP floor, the LB-VERDICT gate armed
        SF.mark_floor_scope(self, True)
        assert LBV.arm(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=rank), tree_cache=self))

    # --- the tree, as the #1423 insert sees it ---------------------------------
    def insert_read(self, rid):
        """A completed store read lands in the host tier (UnifiedRadixCache._insert_helper_host): returns (matched,
        inserted, the request's deepest node). Segments: the common head, then the rid's own tail."""
        matched, node = 0, self.root_node
        for label, n in (("common", COMMON), (rid, LEN[rid] - COMMON)):
            if label in node.children:
                node = node.children[label]
                matched += n
            else:
                child = _Node(node, label, n)
                node.children[label] = child
                node = child
        return matched, LEN[rid] - matched, node

    # --- the URC surface load_back touches ---------------------------------------
    def inc_host_lock_ref(self, node):
        return _Lock()

    def dec_host_lock_ref(self, node, params=None):
        pass

    def inc_lock_ref(self, node):
        self.locks += 1
        return _Lock()

    def dec_lock_ref(self, node, params=None):
        self.locks -= 1

    def _1424_verify_load_chain(self, kv_xfer, req=None):
        return None

    def _build_sidecar_transfers(self, phase, kv, comp):
        return []

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.evicted_log.append(n)
        return SimpleNamespace(num_tokens_evicted=n)

    def _record_store_event(self, node, medium=None):
        pass

    def _update_evictable_leaf_sets(self, node):
        pass

    def check_prefetch_progress(self, rid):
        pass


class _Req:
    def __init__(self, rid, best, root):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.best_match_node = best
        self.last_node = root
        self.host_hit_length = LEN[rid]
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        self.full_untruncated_fill_ids = list(range(LEN[rid] + 38))
        self.cache_protected_len = 0
        self.mamba_loadback_anchor_adopted = False

    def init_next_round_input(self, tree=None):
        pass


#: read by name so that the repro runs (and fails for the metal reason) on 73f6a4e110 too
ROW_ROOM_ENV = getattr(LBV, "ROW_ROOM_ENV", "SGLANG_WEG2_LBV_ROW_ROOM")


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for env in (LBV.ENV, LBV.ROW_YIELD_ENV, ROW_ROOM_ENV, SF.ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(PAC, "store_read_bound_s", lambda deficit: 0.05)
    monkeypatch.setattr(PAC, "MATERIALISE_POLL_S", 0.005)


def _pp0_loads(pp0, rid, best):
    """PP0's OWN plan (not a row): the LB-VERDICT gate holds, PP0 decides, the verdict is absorbed, PP0 loads."""
    req = _Req(rid, best, pp0.root_node)
    assert pp0.load_back(best, None, req=req) is False          # held; PP0's verdict goes on the wire
    for v in LBV.take(pp0):
        LBV.absorb(pp0, v)
    new, last = pp0.init_load_back(InitLoadBackParams(best_match_node=best, host_hit_length=LEN[rid], req=req))
    req.prefix_indices = torch.cat([req.prefix_indices, new])
    return req


def _follower_executes_row(stage, rid, best, scheduled):
    """A follower executes PP0's row: execute_scheduled_prefix (the #968 station) -> init_load_back -> load_back."""
    req = _Req(rid, best, stage.root_node)
    got = PAC.execute_scheduled_prefix(req, stage, scheduled)
    return req, got


def _cap_for_followers(pp0, followers):
    """PP0's pass after its own load: R = MIN(PP0's own room, each follower's fact) -- binding_rank=0 -- stamped on
    the list, taken off and applied by each follower (pp_room_vote, the real functions)."""
    book = PR.RoomBook()
    book.begin_pass()
    for s in followers:
        room = s.token_to_kv_pool_allocator.avail + s._evictable
        book.absorb([PR.Weg2PpRoomFact(rank=s.rank, executed=0, room=room, available=room - s._evictable,
                                       payable=s._evictable, reported=s._evictable)])
    verdict = book.cap(own_room=pp0.token_to_kv_pool_allocator.avail)
    assert verdict is not None and verdict.rank == 0, "PP0's own room binds (PR PP-ROOM-CAP binding_rank=0)"
    wire = PR.stamp_cap(["req"], verdict)
    for s in followers:
        rest, cap = PR.absorb_cap(list(wire))
        PR.set_cap(s, cap)
    return verdict.cap


def _h16j_constellation():
    """Boot, two same-session requests read from L3 on every stage; the reads land in a different order on PP0 than
    on the followers. weg2-0-2 loads first everywhere (it did on PP0 and as a row on PP1/PP2), then PP0 loads
    weg2-0-1 and the followers execute that row under the R of PP0's NEXT pass."""
    pp0 = _Stage(0, PP0_FREE_BEFORE_R1 + LEN[R2])
    fol = [_Stage(1, FOLLOWER_FREE + LEN[R2]), _Stage(2, FOLLOWER_FREE + LEN[R2])]
    placed = {}
    placed[0] = {R1: pp0.insert_read(R1), R2: pp0.insert_read(R2)}          # PP0: weg2-0-1 first
    for s in fol:
        placed[s.rank] = {R2: s.insert_read(R2), R1: s.insert_read(R1)}     # followers: weg2-0-2 first
    # weg2-0-2 first: PP0 by its own plan, the followers by its row (R of that pass is roomy)
    _pp0_loads(pp0, R2, placed[0][R2][2])
    for s in fol:
        PR.set_cap(s, POOL)
        _follower_executes_row(s, R2, placed[s.rank][R2][2], LEN[R2])
        s._evictable = FOLLOWER_EVICTABLE   # weg2-0-2 finished and retained on the follower when the row arrives
        s.token_to_kv_pool_allocator.avail = FOLLOWER_FREE
        s.uniform_avail_floor = FOLLOWER_FREE
    pp0.uniform_avail_floor = pp0.token_to_kv_pool_allocator.avail
    # PP0 decides and loads weg2-0-1 in pass m
    pp0_req = _pp0_loads(pp0, R1, placed[0][R1][2])
    # PP0's pass m+1 stamps its R; the followers hold it when they execute the row of pass m
    cap = _cap_for_followers(pp0, fol)
    return pp0, fol, placed, pp0_req, cap


def test_the_constellation_is_the_metal_one():
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    # #1423 INSERT-PLACED: PP0 matched=0 inserted=121600, PP1/PP2 matched=3008 inserted=118592
    assert placed[0][R1][:2] == (0, 121600)
    assert all(placed[s.rank][R1][:2] == (COMMON, 118592) for s in fol)
    # the matched split moves no load-back depth: PP0 loaded the same host chain the followers will
    assert pp0.cache_controller.loads == [LEN[R2], LEN[R1] - COMMON]       # 118592 = 'kv_tokens=118592'
    assert len(pp0_req.prefix_indices) == LEN[R1]                           # '#988 prefix moved to 121600'
    assert cap == PP0_FREE_BEFORE_R1 - (LEN[R1] - COMMON) == 34560         # 'floor=34560'


def test_h16j_every_stage_holds_the_rows_prefix():
    """THE REPRO: red on 73f6a4e110 (PP1 '#968 PREFIX MATERIALISATION SHORTFALL'), green with the fix."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    for s in fol:
        req, got = _follower_executes_row(s, R1, placed[s.rank][R1][2], LEN[R1])
        assert got == LEN[R1] and len(req.prefix_indices) == len(pp0_req.prefix_indices) == LEN[R1]
        assert s.cache_controller.loads[-1] == LEN[R1] - COMMON           # the same 118592 PP0 loaded
        assert s.evicted_log == []                                         # 153216 free: nothing to evict
        assert s.locks == 2                                    # the two ongoing load-backs' locks, nothing leaked
        assert LBV.row_executed(req) is False


def test_mutant_row_room_off_is_the_h16j_death():
    """RANK-EQUALITY MUTANT 1 (the decision 'a row-executed load-back takes no room verdict'): switch off = the room
    verdict of 73f6a4e110. PP0 holds 121600, the follower refuses against PP0's own post-load R and the group stops."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    os.environ[ROW_ROOM_ENV] = "0"
    try:
        with pytest.raises(RuntimeError, match=r"#968 PREFIX MATERIALISATION SHORTFALL for rid=weg2-0-1: the decision "
                                               r"names prefix_len=121600, this rank holds 0"):
            _follower_executes_row(fol[0], R1, placed[1][R1][2], LEN[R1])
    finally:
        del os.environ[ROW_ROOM_ENV]
    assert len(pp0_req.prefix_indices) == LEN[R1]                           # the split the detector sees
    assert fol[0].evicted_log == [FOLLOWER_EVICTABLE]                       # 'requested=105920 evicted=105920'
    assert fol[0].cache_controller.loads == [LEN[R2]]                       # weg2-0-1 never loaded


def test_mutant_code_level_the_agreed_floor_back_on_the_row():
    """Same decision, code-level mutant: the predicate answers False (the row path not taken) -> the death again."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    orig = LBV.row_room_applies
    LBV.row_room_applies = lambda tree, req: False
    try:
        with pytest.raises(RuntimeError, match="PREFIX MATERIALISATION SHORTFALL"):
            _follower_executes_row(fol[0], R1, placed[1][R1][2], LEN[R1])
    finally:
        LBV.row_room_applies = orig


def test_the_honest_reason_on_a_refusal_with_a_built_anchor():
    """The #968 text no longer blames a GDN depth for a load-back that planted nothing (mutant 4: the old text)."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    os.environ[ROW_ROOM_ENV] = "0"
    try:
        with pytest.raises(RuntimeError) as ei:
            _follower_executes_row(fol[0], R1, placed[1][R1][2], LEN[R1])
    finally:
        del os.environ[ROW_ROOM_ENV]
    msg = str(ei.value)
    assert "load-back refused before any transfer ran (applied=0" in msg
    assert "adopted a GDN anchor at a depth" not in msg


def test_a_row_this_stage_cannot_hold_stops_by_name_never_refuses():
    """Decision 2: even every evictable row short -> LB-VERDICT ROW-ROOM SHORT, the locks unwound, nothing loaded,
    never a False (refuse-this-pass) and never retreated (lb_verdict_short). Mutant 2: the helper's verdict ignored
    (load_back returns False instead) -> the test sees the #968 text, not the named stop."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    s = fol[0]
    s.token_to_kv_pool_allocator.avail = 10000
    s._evictable = 20000
    with pytest.raises(SF.AgreedRoomShort) as ei:
        _follower_executes_row(s, R1, placed[1][R1][2], LEN[R1])
    e = ei.value
    assert "LB-VERDICT ROW-ROOM SHORT rid=weg2-0-1 kv_tokens=118592" in str(e)
    assert e.lb_verdict_short is True and e.avail_after == 30000 and e.evicted == 20000
    assert s.locks == 1                     # weg2-0-2's ongoing lock only: both locks of this attempt unwound
    assert s.cache_controller.loads == [LEN[R2]]


def test_the_row_evicts_the_shortfall_not_the_whole_set():
    """Decision 1's room half (mutant 5: drain every evictable row, the xsn285 shape): a follower short of free rows
    evicts exactly the shortfall and loads in THIS pass -- upstream HiRadixCache.load_back's evict-then-load."""
    pp0, fol, placed, pp0_req, cap = _h16j_constellation()
    s = fol[1]
    s.token_to_kv_pool_allocator.avail = 100000
    s._evictable = 50000
    req, got = _follower_executes_row(s, R1, placed[2][R1][2], LEN[R1])
    assert got == LEN[R1] and s.evicted_log == [LEN[R1] - COMMON - 100000]
    assert s.token_to_kv_pool_allocator.avail == 0 and s._evictable == 50000 - (LEN[R1] - COMMON - 100000)


def test_an_own_plan_load_back_keeps_the_gate_and_the_agreed_room():
    """Scope (mutant 3: the row path for every load-back): a stage's OWN plan still holds for PP0's verdict and, once
    released, still reads the agreed R -- a released load-back above R stops as LB-VERDICT SHORT (853534a9ff)."""
    s = _Stage(1, 200000, 0)
    _m, _i, best = s.insert_read(R1)
    req = _Req(R1, best, s.root_node)
    assert LBV.row_room_applies(s, req) is False
    assert s.load_back(best, None, req=req) is False            # held: no verdict from PP0 yet
    assert s.cache_controller.loads == []
    LBV.absorb(s, LBV.Weg2PpLoadbackVerdict(rid=R1))
    PR.set_cap(s, 50000)                                         # R below the load-back
    with pytest.raises(SF.AgreedRoomShort):
        s.load_back(best, None, req=req)
    assert s.cache_controller.loads == []
    setattr(req, LBV.ROW_ATTR, True)
    assert LBV.row_room_applies(s, req) is True                 # only the row flag opens the row path
    setattr(s, SF.FLOOR_LOCAL_PP_ATTR, False)
    assert LBV.row_room_applies(s, req) is False                # a TP group's floor: never (replicas)


def test_switches_default_on():
    assert LBV.row_room_enabled() and LBV.row_yield_enabled() and LBV.enabled()


def test_source_pins():
    src = inspect.getsource(URC.UnifiedRadixCache.load_back)
    i_row = src.index("if _lbv.row_room_applies(self, req):")
    i_agreed = src.index("floor = _sf.agreed_floor(self, floor)")
    i_gate = src.index("_lbv_gated = _lbv_rid is not None and _lbv.gated(self, req)")
    i_refuse = src.index("if floor < kv_tokens:")
    assert i_row < i_agreed < i_gate < i_refuse
    blk = src[i_row:i_agreed]
    assert "raise _sf.AgreedRoomShort(" in blk and "lb_verdict_short=True" in blk
    assert blk.index("self.dec_lock_ref(best_match_node, ancestor_lock_params)") < blk.index("raise _sf.AgreedRoomShort(")
    assert "else:" in src[i_row:i_agreed + 60]
