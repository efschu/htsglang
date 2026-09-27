"""H105b + D-OOM: the Form A host votes the depth it ADMITS, and a Form A
load-back charges the iteration's availability floor.

H105 (rc12w dkrnfh91dprsabar1dauer09272038_3a10543309, D log 10939-10993,
rid weg2-0-1 after a #248 park/wake-read): TP0's match was device 0 + host
hit 23040 -- the leading 2560-token node counts no host hit -- with key and
mamba anchor at 25600. TP0 VOTED 23040 (``_local_match_len``), the workers
FOLLOWED 23040 (``RU FORM-A FOLLOW tp0_depth=23040 worker_local=25600``), and
TP0's own load-back raised its extent to the anchor
(``#1040 EXTENT STATE-ALIGN kv=23040 extent=25600``, ``#988 LOADBACK prefix
moved to 25600``). TP0 took the tail skip (extend 0) and closed its admission
loop, the workers extended 2571 tokens and went on to the next gate:
``H105 RU FORM-A ADMISSION MALFORMED rid=weg2-0-5``.

D-OOM (rc12v dkrnfh91dprsabar1dauer09272047_4c866d534a, D log ~281870-281925,
rid weg2-28-124): TP1/TP2 loaded 23424 rows back, then the 163-token extend's
eviction trigger read the uncharged floor, skipped the eviction and failed on
a pool with 0 free rows ('EVICTION UNDER-DELIVERED asked 227 received 0').

RED on 3a10543309: the host votes 23040; the host above the group is not
capped; the load-back leaves the ledger at 0 and the eviction is skipped.
"""

from __future__ import annotations

import contextlib
import types
import unittest
from array import array

import torch

from sglang.srt import rank_role
from sglang.srt.managers import pp_admission_congruence as pac
from sglang.srt.managers import tp_match_floor as m
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.base_prefix_cache import MatchResult

SWITCH = "SGLANG_WEG2_ENABLE_FORM_A_TP0_FOLLOW"
FOLLOW_ATTR = "_tp_match_floor_follow_walk"
FLOOR_ATTR = "_tp_match_floor_group"
ROLES = ("host", "worker", "worker")
PROMPT = 25611
LEAD = 2560  # the leading node that carries no host hit on TP0
DEPTH = 25600  # key match and mamba anchor on TP0


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


@contextlib.contextmanager
def _switch(value=True):
    from sglang.srt.environ import envs

    with getattr(envs, SWITCH).override(value):
        yield


def _node(name, anchor=True):
    host = torch.tensor([3]) if anchor else None
    return types.SimpleNamespace(
        name=name,
        component_data=[None, None, types.SimpleNamespace(value=None, host_value=host)],
    )


class _HostTree:
    """TP0 after the #248 wake-read: key and anchor at ``depth``, the host hit
    counted from ``lead`` (the leading node came in hitless)."""

    def __init__(self, depth=DEPTH, lead=LEAD):
        self.depth, self.lead = depth, lead
        self.root_node = _node("root", anchor=False)
        self.cache_controller = object()
        self.is_eagle = False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def match_prefix(self, params):
        d = min(self.depth, len(params.key))
        node = _node(f"n{d}")
        return MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=max(0, d - self.lead),
            state_anchor_depth=d,
            key_match_depth=d,
        )


class _WorkerTree(_HostTree):
    """A byteless worker: its KV reach IS its host hit (no hitless lead)."""

    def __init__(self, reach=DEPTH):
        super().__init__(depth=reach, lead=0)


def _req(rid="weg2-0-1", n=PROMPT):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=array("q"),
        extra_key=None,
        mamba_pool_idx=0,
        positional_embed_overrides=None,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


def _vote(tree, rank, rid="weg2-0-1"):
    with _as_rank(rank):
        req = _req(rid)
        return m.local_usable_matches(tree, {rid: req}, {rid: PROMPT - 1})[rid]


class TestStateAlignedExtentPure(unittest.TestCase):
    def test_one_expression_for_load_back_and_vote(self):
        fn = getattr(pac, "state_aligned_extent", None)
        self.assertIsNotNone(fn, "no pure #1040 extent to share with the host vote")
        self.assertEqual(fn(23040, 25600, 0, 25600), (25600, True))  # rc12w TP0
        self.assertEqual(fn(23040, 20000, 0, 25600), (20000, False))  # anchor below the hit
        self.assertEqual(fn(23040, 25600, 0, 24000), (23040, False))  # key short of the anchor
        self.assertEqual(fn(23040, 30000, 2560, 30000), (27440, True))  # device part

    def test_load_back_len_unchanged(self):
        req = types.SimpleNamespace(
            rid="x",
            host_hit_length=23040,
            state_anchor_depth=25600,
            key_match_depth=25600,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        self.assertEqual(pac.state_aligned_load_back_len(req), 25600)


class TestHostVotesAdmittedDepth(unittest.TestCase):
    def test_rc12w_host_votes_the_load_back_depth(self):
        with _switch():
            host = _vote(_HostTree(), 0)
            workers = [_vote(_WorkerTree(), r) for r in (1, 2)]
        # RED on 3a10543309: 23040 -- the depth TP0 does NOT admit.
        self.assertEqual(host, DEPTH)
        self.assertEqual(workers, [DEPTH, DEPTH])

    def test_vote_equals_the_hosts_own_load_back(self):
        """The invariant: device + #1040 extent at admission == the vote."""
        tree = _HostTree()
        with _switch():
            vote = _vote(tree, 0)
        res = tree.match_prefix(types.SimpleNamespace(key=list(range(PROMPT - 1))))
        req = types.SimpleNamespace(
            rid="weg2-0-1",
            host_hit_length=res.host_hit_length,
            state_anchor_depth=res.state_anchor_depth,
            key_match_depth=res.key_match_depth,
            prefix_indices=res.device_indices,
        )
        self.assertEqual(vote, len(res.device_indices) + pac.state_aligned_load_back_len(req))

    def test_host_admission_does_not_stop_at_its_own_depth(self):
        tree = _HostTree()
        res = tree.match_prefix(types.SimpleNamespace(key=list(range(PROMPT - 1))))
        setattr(tree, FLOOR_ATTR, {"weg2-0-1": DEPTH})
        with _switch(), _as_rank(0):
            # the raw match (23040) is below the planted depth; what the host
            # admits (25600) is not -- no HOST-BELOW-GROUP stop
            self.assertIsNone(m.form_a_follow_admission(tree, _req(), res))
            self.assertIsNone(m.group_floor_cap(tree, _req(), res))

    def test_host_above_group_is_capped(self):
        """A worker whose KV reach is shorter than the host's admitted depth
        sets the group lower; the host must cap, not load back beyond it."""
        tree = _HostTree()
        res = tree.match_prefix(types.SimpleNamespace(key=list(range(PROMPT - 1))))
        setattr(tree, FLOOR_ATTR, {"weg2-0-1": 24000})
        with _switch(), _as_rank(0):
            # RED on 3a10543309: None (raw 23040 <= 24000 reads 'agree') while
            # the load-back takes the host to 25600
            self.assertEqual(m.group_floor_cap(tree, _req(), res), 24000)
            # the re-match at the group depth REALIZES it on the host (anchor
            # there, admitted depth == 24000) -- no CAP-MISS
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixKey

            params = MatchPrefixParams(
                key=RadixKey(array("q", range(PROMPT - 1)), None), cow_mamba=False, req=_req()
            )
            capped = m.rematch_at_group_depth(tree, params, 24000, DEPTH)
            self.assertEqual(m.host_admission_len(capped), 24000)

    def test_worker_follows_the_group_depth(self):
        tree = _WorkerTree()
        res = tree.match_prefix(types.SimpleNamespace(key=list(range(PROMPT - 1))))
        setattr(tree, FLOOR_ATTR, {"weg2-0-1": DEPTH})
        with _switch(), _as_rank(1):
            self.assertEqual(m.form_a_follow_admission(tree, _req(), res), DEPTH)

    def test_classic_boot_untouched(self):
        tree = _HostTree()
        with _switch(False), _as_rank(0, roles=None):
            vote = m.local_usable_matches(tree, {"weg2-0-1": _req()}, {"weg2-0-1": 23040})
        # classic path: the head walk's number, as before
        self.assertEqual(vote["weg2-0-1"], 23040)


class _Alloc:
    def __init__(self, free):
        self.free = free

    def available_size(self):
        return self.free


class _EvictTree:
    """What `evict_from_tree_cache` and the load-back charge read."""

    def __init__(self, floor, free):
        self.uniform_avail_floor = floor
        self.uniform_admitted_since_floor = 0
        self.token_to_kv_pool_allocator = _Alloc(free)
        self.evict_calls = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evict_calls.append(int(params.num_tokens))
        self.token_to_kv_pool_allocator.free += int(params.num_tokens)
        return types.SimpleNamespace(num_tokens_evicted=int(params.num_tokens))


def _charge(tree, rows):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    fn = getattr(urc, "_form_a_note_loaded", None)
    if fn is None:
        return False
    fn(tree, rows)
    return True


class TestFormALoadBackCharged(unittest.TestCase):
    def test_rc12v_worker_extend_evicts_after_the_load(self):
        # floor published at the top of the pass (value illustrative: the log
        # names only its effect), the load takes 23424 of it, 0 rows are free
        tree = _EvictTree(floor=23600, free=0)
        with _switch(), _as_rank(1):
            self.assertTrue(_charge(tree, 23424), "no Form A load-back charge")
        self.assertEqual(tree.uniform_admitted_since_floor, 23424)
        common.evict_from_tree_cache(tree, 227)
        # RED on 3a10543309: [] -- floor 23600 >= 227, eviction skipped, OOM
        self.assertEqual(tree.evict_calls, [227])

    def test_classic_tp_ledger_unchanged(self):
        tree = _EvictTree(floor=23651, free=0)
        with _switch(False), _as_rank(1, roles=None):
            _charge(tree, 23424)
        self.assertEqual(tree.uniform_admitted_since_floor, 0)

    def test_no_floor_no_charge(self):
        tree = _EvictTree(floor=None, free=0)
        with _switch(), _as_rank(1):
            _charge(tree, 23424)
        self.assertEqual(tree.uniform_admitted_since_floor, 0)


if __name__ == "__main__":
    unittest.main()
