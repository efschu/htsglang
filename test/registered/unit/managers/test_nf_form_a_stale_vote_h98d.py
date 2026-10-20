"""H98d: a Form A group depth that went stale INSIDE the pass is refused
through the H105 verdict on every rank -- not an H98 HOST-BELOW-GROUP death.

THE DEATH (NF xc D, c5da548b7c, boot dkrnfint4h6ablxcbar1dauer10071818,
D log 841600-843160, 07.10.2026 21:37:20-21:37:29Z, rid pdflip-130-2092, a
175353-token agent turn, uncached 249):

* 21:37:20 TP0 ``MATCH-CENSUS-DEEP rid=pdflip-130-2092 reached=175104
  accepted=175104 prefix=175104`` -- the whole prefix on TP0's DEVICE, its
  anchor at 175104. The pass's usable vote (scheduler ~10740,
  ``local_usable_matches`` -> ``admission_probe``) plants 175104.
* 21:37:29, the SAME pass, pdflip-130-2093 is admitted first. Under the #239
  token cut its ADMIT is gathered before its load-back; the load-back's
  floor refuses and drains every evictable leaf (xsn285, unified_radix_cache
  ``load_back``; ``EVICT-FRONTIER-CENSUS request=168832`` on all three
  ranks, then ``H105c FORM-A FOLLOW-ROOM ... evicted=0`` and
  ``H105c FORM-A FOLLOW LOAD-BACK``). TP0's mamba host arena is full
  (``ARENA-DROP ... slot_bytes=58834944 ... freed=0``), so the drain backs
  four device nodes up KV-ONLY and drops their anchors (``P-FUND EVICT
  KV-ONLY n=19..22`` tokens 384, 960, 320, 256; 960+320+256 = 1536 =
  175104 - 173568).
* pdflip-130-2092's admission match on TP0 now ends on the deepest SURVIVING
  anchor: ``host_admission_len`` 173568 < planted 175104 -> ``H98 RU FORM-A
  HOST-BELOW-GROUP rid=pdflip-130-2092 local_match=173568 group=175104``,
  RANK_EXCEPTION, the D group dead after 3 h 15 min. TP1/TP2 had already
  followed 175104 (``#1042 EXTENT LIFECYCLE set ... extent=44032``,
  ``X-GATE ... uncached=249 ... verdict=admit``).

The model below is the H98 file's: a rank's tree is its KV reach plus the
depths at which it holds a usable (host-backed) anchor; the drain is the
anchor at 175104 leaving TP0's tree while its KV stays (KV-only backup).

RED on c5da548b7c: TP0's admission raises FormAHostBelowGroup.
GREEN with H98d: TP0 keeps 173568 and records the stale pair; its H105
verdict (OTHER) keeps the request queued on every rank; the next pass votes
173568 and every rank admits 173568. The stop stays with the switch off,
without a verdict channel, and for the same pair three passes running.
"""

from __future__ import annotations

import contextlib
import os
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from flliper.srt import rank_role
from flliper.srt.managers import schedule_policy as sp
from flliper.srt.managers import tp_match_floor as m
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

FLOOR_ATTR = "_tp_match_floor_group"
FOLLOW_ATTR = "_tp_match_floor_follow_walk"
FOLLOW_SWITCH = "FLLIPER_PDFLIP_ENABLE_FORM_A_TP0_FOLLOW"
DEFER_SWITCH = "FLLIPER_PDFLIP_ENABLE_FORM_A_STALE_VOTE_DEFER"

PAGE = 64
SLOTS = 8
ROLES = ("host", "worker", "worker")

#: pdflip-130-2092 geometry (D log 21:37:20-21:37:29Z).
RID = "pdflip-130-2092"
FILL = 175353  # 175104 cached + 249 uncached (X-GATE uncached=249)
KV = 175104  # the KV reach on every rank (the drain backs the tail up KV-only)
GROUP = 175104  # planted at the top of the pass (TP0 anchor on device)
LOCAL = 173568  # TP0's deepest SURVIVING anchor after the drain
HOST_AT_VOTE = (172288, 173568, 175104)
HOST_AFTER_DRAIN = (172288, 173568)


def _floor(n):
    return n // PAGE * PAGE


class _Tree:
    """One rank's radix tree, reduced to what decides a match (the H98 model):
    ``kv`` is the KV reach, ``anchors`` the depths with a usable host-backed
    anchor. An ordinary walk ends on the deepest anchor within reach; a walk
    under the follow attribute ends on the reach itself."""

    def __init__(self, kv, anchors):
        self.kv = kv
        self.anchors = sorted(anchors)
        self.root_node = types.SimpleNamespace(
            name="root",
            component_data=[None, None, types.SimpleNamespace(value=None, host_value=None)],
        )
        self.cache_controller = object()
        self.is_eagle = False
        self.is_chunk_cache = lambda: False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def _node(self, n):
        if n <= 0:
            return self.root_node
        host = torch.tensor([3]) if n in self.anchors else None
        data = [None, None, types.SimpleNamespace(value=None, host_value=host)]
        return types.SimpleNamespace(name=f"n{n}", component_data=data)

    def result(self, n):
        node = self._node(n)
        return MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=n,
        )

    def match_prefix(self, params):
        reach = min(_floor(len(params.key)), self.kv)
        if getattr(self, FOLLOW_ATTR, False):
            n = reach
        else:
            n = max([a for a in self.anchors if a <= reach], default=0)
        res = self.result(n)
        if params.cow_mamba:
            return _component(self).finalize_match_result(
                result=res, params=params, value_chunks=[torch.zeros(1)], best_value_len=1
            )
        return res


def _component(tree):
    comp = types.SimpleNamespace(
        component_type=ComponentType.MAMBA,
        mamba_checkpoint_interval=None,
        mamba_ckpt_strict_resume=False,
        cache=tree,
        _stateless_resume_refusals=0,
        _foreign_pool_resume_refusals=0,
    )
    comp.finalize_match_result = types.MethodType(MambaComponent.finalize_match_result, comp)
    comp._raw_token_pos = lambda depth: depth
    comp.create_match_validator = types.MethodType(MambaComponent.create_match_validator, comp)
    return comp


def _req(rid=RID, n=FILL):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=array("q"),
        extra_key=None,
        mamba_pool_idx=0,
        positional_embed_overrides=None,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


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
def _switches(follow=True, defer=None):
    """The H98 follow switch and (when not None) the H98d deferral switch."""
    from flliper.srt.environ import envs

    with contextlib.ExitStack() as stack:
        stack.enter_context(getattr(envs, FOLLOW_SWITCH).override(follow))
        # Read by NAME so this file collects and runs on c5da548b7c too, where
        # the switch does not exist (= no deferral: that is the red).
        field = getattr(envs, DEFER_SWITCH, None)
        if defer is not None and field is not None:
            stack.enter_context(field.override(defer))
        yield


def _head_walk(tree, req):
    toks = list(req.origin_input_ids) + list(req.output_ids)
    res = tree.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", toks), None), cow_mamba=False, req=None)
    )
    req.best_match_node = res.best_match_node
    n = len(res.device_indices) + int(res.host_hit_length)
    return min(n, req._compute_max_prefix_len(len(toks)))


def _plant(trees, rid=RID, roles=ROLES):
    """The pass's usable vote, rank by rank (MIN arm, MAX arm, H97 round)."""
    canonical = [rid]
    local = {}
    for r, t in trees.items():
        with _as_rank(r, roles):
            req = _req(rid)
            local[r] = m.local_usable_matches(t, {rid: req}, {rid: _head_walk(t, req)})
    red = lambda rows: [min(col) for col in zip(*rows)]  # noqa: E731
    rows_min, rows_max = [], []
    for r in trees:
        with _as_rank(r, roles):
            rows_min.append(m.build_usable_match_payload(canonical, local[r], SLOTS))
            rows_max.append(m.build_usable_max_payload(canonical, local[r], SLOTS))
    g = m.decode_group_usable(canonical, red(rows_min))
    skew = m.skewed_rids(g, m.decode_group_max(canonical, red(rows_max)))
    if skew:
        flags = []
        for r, t in trees.items():
            with _as_rank(r, roles):
                flags.append(
                    m.build_realize_payload(canonical, skew, local[r], t, {rid: _req(rid)}, SLOTS)
                )
        g = m.apply_realize_verdict(g, canonical, skew, red(flags))
    return g


def _admit(tree, rank, planted, rid=RID, roles=ROLES):
    """`Req.init_next_round_input` on this rank inside the planted window."""
    req = _req(rid)
    toks = array("q", list(req.origin_input_ids))
    params = MatchPrefixParams(
        key=RadixKey(toks, None, limit=req._compute_max_prefix_len(len(toks))),
        cow_mamba=True,
        req=req,
    )
    setattr(tree, FLOOR_ATTR, planted)
    try:
        with _as_rank(rank, roles):
            out = tree.match_prefix(params)
    finally:
        setattr(tree, FLOOR_ATTR, None)
    return len(out.device_indices) + int(out.host_hit_length)


def _trees_at_vote():
    # TP0: anchors 172288 / 173568 (host-backed) and 175104 (device, the
    # previous turn's end). TP1/TP2: the same KV reach, byteless anchors
    # elsewhere (their MAMBA-HOST-RESUME depths in the log).
    return {
        0: _Tree(KV, HOST_AT_VOTE),
        1: _Tree(KV, [172288]),
        2: _Tree(KV, [173824]),
    }


def _drain(host_tree):
    """The xsn285 drain for pdflip-130-2093: the 175104 anchor leaves TP0's
    tree (P-FUND KV-ONLY), its KV stays host-backed."""
    host_tree.anchors = sorted(HOST_AFTER_DRAIN)


def _reset_module_state():
    getattr(m, "_STALE_REPEAT", {}).clear()  # absent on c5da548b7c
    m._STATS.pop("stale_vote_defer", None)


class TestPdFlip_130_2092(unittest.TestCase):
    """The death, replayed: vote -> in-pass drain -> admission."""

    def setUp(self):
        _reset_module_state()

    def test_host_admission_after_the_drain_is_not_a_death(self):
        trees = _trees_at_vote()
        with _switches(defer=True):
            planted = _plant(trees)
            self.assertEqual(planted, {RID: GROUP}, "the vote is right when taken")
            _drain(trees[0])
            with self.assertLogs(m.logger, level="WARNING") as cm:
                # RED on c5da548b7c: FormAHostBelowGroup local_match=173568
                # group=175104 (tp_match_floor.py:966)
                geometry = {r: _admit(t, r, planted) for r, t in trees.items()}
        self.assertEqual(geometry, {0: LOCAL, 1: GROUP, 2: GROUP})
        self.assertEqual(m.form_a_host_vote_stale(trees[0], _req()), (LOCAL, GROUP))
        self.assertIsNone(m.form_a_host_vote_stale(trees[1], _req()))
        lines = [l for l in cm.output if "H98d RU FORM-A STALE-VOTE DEFER" in l]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(f"rid={RID} local={LOCAL} group={GROUP} repeat=1 why=admission", lines[0])

    def test_next_pass_converges_on_every_rank(self):
        """The refused pass leaves the request queued; the next vote is taken
        from the trees as they are -- TP0 votes what it admits now."""
        trees = _trees_at_vote()
        with _switches(defer=True):
            planted = _plant(trees)
            _drain(trees[0])
            for r, t in trees.items():
                _admit(t, r, planted)
            m.plant(trees[0], None)  # the next pass's plant resets the record
            planted2 = _plant(trees)
            geometry = {r: _admit(t, r, planted2) for r, t in trees.items()}
        self.assertEqual(planted2, {RID: LOCAL})
        self.assertEqual(geometry, {0: LOCAL, 1: LOCAL, 2: LOCAL})
        self.assertIsNone(m.form_a_host_vote_stale(trees[0], _req()))
        self.assertNotIn(RID, m._STALE_REPEAT, "a clean admission forgets the repeat")


def _isolate_pdflip_group(testcase):
    """As in test_nf_form_a_admission_follow_h105: the chunk-admission module
    globals read FLLIPER_PDFLIP_GROUP once; keep them neutral here."""
    env = patch.dict(os.environ)
    env.start()
    testcase.addCleanup(env.stop)
    os.environ.pop("FLLIPER_PDFLIP_GROUP", None)
    sp._PDFLIP_CHUNK_ADMIT = None
    sp._PDFLIP_PARK_ON = None

    def _reset():
        sp._PDFLIP_CHUNK_ADMIT = None
        sp._PDFLIP_PARK_ON = None

    testcase.addCleanup(_reset)


def _tree_cache():
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _adder(available=10**7):
    b = MagicMock()
    b.reqs = []
    return PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(available),
        running_batch=b,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=4096,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _mreq(*, device_prefix):
    """pdflip-130-2092 at ``add_one_req``: TP0 host-backed (0 device rows),
    a worker with the followed depth."""
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = list(range(device_prefix))
    req.full_untruncated_fill_ids = list(range(FILL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


class _Group:
    """Three ranks of one Form A group; ``exchange`` is the TP broadcast (the
    host posts first, every rank reads the host's post)."""

    def __init__(self):
        self.mailbox = {}

    def scheduler(self, tp_rank):
        s = SimpleNamespace(
            ps=SimpleNamespace(tp_size=3, pp_size=1),
            tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
            tp_cpu_group=None,
        )

        def _exchange(site, payload):
            if payload is not None:
                self.mailbox[site] = payload
            return self.mailbox.get(site)

        s._form_a_tp_exchange = _exchange
        for name in ("_form_a_is_host", "_form_a_admission_follow_fn"):
            setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
        return s


def _install(adder, sched, tp_rank):
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=tp_rank != 0
    ):
        adder.form_a_admission_follow = sched._form_a_admission_follow_fn()


def _host_stale_record():
    """TP0's record after the replayed vote -> drain -> admission (the tree
    object is the scheduler's ``tree_cache``, the adder's too)."""
    trees = _trees_at_vote()
    with _switches(defer=True):
        planted = _plant(trees)
        _drain(trees[0])
        _admit(trees[0], 0, planted)
    return getattr(trees[0], m.HOST_STALE_ATTR)


class TestH105VerdictCarriesTheRefusal(unittest.TestCase):
    def setUp(self):
        _reset_module_state()
        _isolate_pdflip_group(self)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_every_rank_keeps_the_request_queued(self):
        stale = _host_stale_record()
        group = _Group()
        results, adders = {}, {}
        for tp_rank, prefix in ((0, 0), (1, GROUP), (2, GROUP)):
            adder = _adder()
            _install(adder, group.scheduler(tp_rank), tp_rank)
            if tp_rank == 0:
                setattr(adder.tree_cache, m.HOST_STALE_ATTR, stale)
            results[tp_rank] = adder.add_one_req(
                _mreq(device_prefix=prefix), truncation_align_size=None
            )
            adders[tp_rank] = adder
        self.assertEqual(
            results,
            {0: AddReqResult.OTHER, 1: AddReqResult.OTHER, 2: AddReqResult.OTHER},
            "the workers must take the host's refusal, never admit 175104 alone",
        )
        for r in (0, 1, 2):
            self.assertEqual(adders[r].can_run_list, [], r)
        posted = group.mailbox["form-a-admission/tp<-verdict"]
        self.assertEqual(posted[:2], (RID, "OTHER"))

    def test_no_verdict_channel_is_the_h98_stop(self):
        adder = _adder()  # no Form A follow callable installed
        setattr(adder.tree_cache, m.HOST_STALE_ATTR, {RID: (LOCAL, GROUP)})
        with self.assertRaises(m.FormAHostBelowGroup) as cm:
            adder.add_one_req(_mreq(device_prefix=0), truncation_align_size=None)
        self.assertIn(f"local_match={LOCAL} group={GROUP}", str(cm.exception))

    def test_gate_inert_without_a_record(self):
        adder = _adder()
        self.assertIsNone(sp._form_a_stale_vote_gate(adder, _mreq(device_prefix=0), None))
        self.assertFalse(
            m.form_a_stale_vote_refuses(
                adder.tree_cache, _mreq(device_prefix=0), has_verdict_channel=False
            )
        )


class TestTheStopStays(unittest.TestCase):
    def setUp(self):
        _reset_module_state()

    def test_switch_off_is_the_immediate_h98_stop(self):
        trees = _trees_at_vote()
        with _switches(defer=False):
            planted = _plant(trees)
            _drain(trees[0])
            with self.assertRaises(m.FormAHostBelowGroup) as cm:
                _admit(trees[0], 0, planted)
        self.assertIn(
            f"H98 RU FORM-A HOST-BELOW-GROUP rid={RID} local_match={LOCAL} group={GROUP}",
            str(cm.exception),
        )
        self.assertIsNone(m.form_a_host_vote_stale(trees[0], _req()))

    def test_same_pair_three_passes_running_stops_by_name(self):
        tree = _Tree(KV, HOST_AFTER_DRAIN)
        with _switches(defer=True):
            for _ in range(m.STALE_VOTE_REPEAT_STOP - 1):
                m.plant(tree, None)
                self.assertEqual(_admit(tree, 0, {RID: GROUP}), LOCAL)
            m.plant(tree, None)
            with self.assertRaises(m.FormAHostBelowGroup) as cm:
                _admit(tree, 0, {RID: GROUP})
        self.assertIn("H98d REPEAT passes=3", str(cm.exception))

    def test_a_clean_pass_resets_the_repeat(self):
        tree = _Tree(KV, HOST_AFTER_DRAIN)
        with _switches(defer=True):
            _admit(tree, 0, {RID: GROUP})
            m.plant(tree, None)
            _admit(tree, 0, {RID: LOCAL})  # the host admits the planted depth
            m.plant(tree, None)
            _admit(tree, 0, {RID: GROUP})
            m.plant(tree, None)
            self.assertEqual(_admit(tree, 0, {RID: GROUP}), LOCAL)  # 2nd running, no stop

    def test_zero_guard_defers_and_stops_like_the_depth(self):
        tree = _Tree(KV, HOST_AFTER_DRAIN)
        setattr(tree, FLOOR_ATTR, {RID: GROUP})
        with _switches(defer=True), _as_rank(0):
            m.form_a_host_zero_guard(tree, _req(), "mamba slot starvation")
        self.assertEqual(m.form_a_host_vote_stale(tree, _req()), (0, GROUP))
        m.plant(tree, {RID: GROUP})
        with _switches(defer=False), _as_rank(0), self.assertRaises(m.FormAHostBelowGroup):
            m.form_a_host_zero_guard(tree, _req(), "mamba slot starvation")


class TestPassScopeAndUnchangedPaths(unittest.TestCase):
    def setUp(self):
        _reset_module_state()

    def test_plant_and_clear_reset_the_record(self):
        tree = _Tree(KV, HOST_AFTER_DRAIN)
        with _switches(defer=True):
            _admit(tree, 0, {RID: GROUP})
            self.assertIsNotNone(m.form_a_host_vote_stale(tree, _req()))
            m.plant(tree, {RID: LOCAL})
            self.assertIsNone(m.form_a_host_vote_stale(tree, _req()))
            _admit(tree, 0, {RID: GROUP})
            m.clear(tree)
            self.assertIsNone(m.form_a_host_vote_stale(tree, _req()))

    def test_host_at_the_group_depth_writes_nothing(self):
        tree = _Tree(KV, HOST_AT_VOTE)
        with _switches(defer=True):
            self.assertEqual(_admit(tree, 0, {RID: GROUP}), GROUP)
        self.assertFalse(hasattr(tree, m.HOST_STALE_ATTR))

    def test_classic_and_flip_boots_unchanged(self):
        """No Form A role plan (27B, flip P<->D, classic): the RU/H96 path, the
        same result with the H98d switch on or off, nothing recorded."""
        for defer in (True, False):
            tree = _Tree(KV, HOST_AFTER_DRAIN)
            with _switches(defer=defer):
                got = _admit(tree, 0, {RID: GROUP}, roles=None)
            self.assertEqual(got, LOCAL, defer)
            self.assertFalse(hasattr(tree, m.HOST_STALE_ATTR), defer)

    def test_follow_switch_off_unchanged(self):
        tree = _Tree(KV, HOST_AFTER_DRAIN)
        with _switches(follow=False, defer=True):
            self.assertEqual(_admit(tree, 0, {RID: GROUP}), LOCAL)
        self.assertFalse(hasattr(tree, m.HOST_STALE_ATTR))


if __name__ == "__main__":
    unittest.main()
