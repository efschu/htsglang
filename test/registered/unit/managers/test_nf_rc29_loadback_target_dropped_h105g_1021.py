"""H105g: a request's own room step must not hand init_load_back a load-back source it just dropped.

THE DEATH (NF rc29 a773b4ebc4, D log
boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210221_a773b4ebc4_1021_022209.D.log, Form A x #239 token cut):

* 02:30:48Z ``HiCache prefetch success req=pdflip-2-44 ... matched=18880 loaded=150336`` (L3 -> L2, 2349 pages):
  a 150336-token HOST-ONLY leaf below an 18880-token device prefix, its mamba END anchor host-only
  (02:31:11 ``MAMBA-HOST-RESUME ... depth=169216 on a HOST-backed state ... this match triggers load_back``);
* 02:31:11-12Z TP0, the request's H105d room (``_h105d_cut_load_back_room`` -> ``tc.evict``): ``#1469 EVICT
  node=170/167/169``; backing them up claims arena pages, the device round pays short and the host-only round
  drops the one candidate: ``W3-HOST-LEAF SPILL n=25 want=1987 released_pages=2349 leaves=1 candidates=1 ...
  l3=on_disk:2349 ... aux_candidates=1 aux_leaves=1`` -- the request's own load-back source (S109 fa527fb6ea
  made a leaf whose only host-only aux state is its mamba anchor spillable);
* 02:31:13Z TP0 ``LEAF-SET DETACHED n=1 node=166 evicted=True caller=init_load_back/load_back/inc_host_lock_ref``
  -> ``full_component.build_hicache_transfers``: ``assert cd.host_value is not None`` -> RANK_EXCEPTION; TP1/TP2
  follow on the closed gloo pair.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL ``Scheduler._form_a_admission_follow_fn``
(three ranks in threads, the token cut's all-gather; harness of test_nf_form_a_cut_load_back_room_h105d), the
init_load_back double asserting exactly what ``FullComponent.build_hicache_transfers(LOAD_BACK)`` asserts.
RED on a773b4ebc4: TP0 raises the metal AssertionError while TP1/TP2 load back. GREEN with H105g: TP0 votes
NO_TOKEN in the confirming gather, every rank keeps the rid queued, nobody loads back, the gathers stay alike.
Plus the predicate on a REAL UnifiedRadixCache: the drop the spill does (``_evict_host_leaf``) is what it names.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


H = _load("_h105d_harness_for_h105g", _HERE / "test_nf_form_a_cut_load_back_room_h105d.py")


class _Key:
    """The radix child key of the dropped leaf (``key.child_key(page)``)."""

    def child_key(self, page):
        return ("leaf", page)


def _metal_init_load_back(tc, inner):
    """The real ``load_back`` first walks the evicted run exactly like ``build_hicache_transfers(LOAD_BACK)``:
    ``assert cd.host_value is not None`` on every evicted node (full_component.py:389)."""

    def _ilb(params):
        cur = params.best_match_node
        while cur is not None and cur.evicted:
            assert cur.component_data[BASE_COMPONENT_TYPE].host_value is not None
            cur = cur.parent
        return inner(params)

    return _ilb


def _wire(adder, req, *, drops: bool):
    """Give the rank's tree a real parent/child link for the load-back source; on ``drops`` its room eviction
    also does what the W3 host-only spill did on TP0 at 02:31:12 (``_evict_host_leaf``: Full host value None,
    the leaf out of its parent's children)."""
    tc = adder.tree_cache
    tc.page_size = 1
    tc.root_node = object()
    leaf = req.best_match_node
    leaf.key = _Key()
    leaf.id = 166
    leaf.parent.children = {leaf.key.child_key(1): leaf}
    real_evict = tc.evict.side_effect

    def _evict(params):
        res = real_evict(params)
        if drops:
            leaf.component_data[BASE_COMPONENT_TYPE].host_value = None
            leaf.parent.children.pop(leaf.key.child_key(1), None)
        return res

    tc.evict.side_effect = _evict
    tc.init_load_back.side_effect = _metal_init_load_back(tc, tc.init_load_back.side_effect)


def _wire_anchor(adder, req, *, gdn: bool, drops: bool):
    """S109R/S109c in the window: the match accepted the load-back source's mamba state on the HOST
    (``mamba_host_hit_length`` 1 on the GDN rank TP0; the Form A workers hold no recurrent state, 0), and TP0's room
    eviction runs the S109c aux round, which releases that host anchor (its L3 copy proven). The KV stays intact."""
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    tc = adder.tree_cache
    leaf = req.best_match_node
    mcd = SimpleNamespace(value=None, host_value=torch.tensor([5], dtype=torch.int64))
    leaf.component_data[ComponentType.MAMBA] = mcd
    req.mamba_host_hit_length = 1 if gdn else 0
    tc.anchor_lost_loads = 0
    real_evict = tc.evict.side_effect

    def _evict(params):
        res = real_evict(params)
        if drops:
            mcd.host_value = None
        return res

    tc.evict.side_effect = _evict
    inner = tc.init_load_back.side_effect

    def _ilb(params):
        if int(getattr(params.req, "mamba_host_hit_length", 0) or 0) > 0 and mcd.value is None \
                and mcd.host_value is None:
            tc.anchor_lost_loads += 1   # MambaComponent restores nothing: the prefix is clamped and recomputed
        return inner(params)

    tc.init_load_back.side_effect = _ilb


def _group_anchor():
    gather = H._Gather()
    out, adders, reqs = {}, {}, {}
    with patch(H.TOKEN_CUT, return_value=True):
        for r in range(3):
            adders[r] = H._adder(r, gather, deliverable=H.REPORTED_EVICTABLE)
            reqs[r] = H._req(r)
            _wire_anchor(adders[r], reqs[r], gdn=r == 0, drops=r == 0)

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001
                out[r] = e

        with patch.object(sp, "_pp_load_back_extent", side_effect=lambda q: q._h105d_extent), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            ts = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(30)
    return out, adders, gather


def _group(*, drop_on=(0,)):
    """One add_one_req for the rid on TP0/1/2 at once (room makeable on every rank); TP ranks in ``drop_on``
    drop the load-back source inside their room eviction."""
    gather = H._Gather()
    out, adders, reqs = {}, {}, {}
    with patch(H.TOKEN_CUT, return_value=True):
        for r in range(3):
            adders[r] = H._adder(r, gather, deliverable=H.REPORTED_EVICTABLE)
            reqs[r] = H._req(r)
            _wire(adders[r], reqs[r], drops=r in drop_on)

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001 -- the result under test
                out[r] = e

        with patch.object(sp, "_pp_load_back_extent", side_effect=lambda q: q._h105d_extent), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            ts = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(30)
    return out, adders, reqs, gather


class LoadBackTargetDroppedTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sp._H105D_REFUSED["n"] = 0
        if hasattr(sp, "_H105G_DROPPED"):
            sp._H105G_DROPPED["n"] = 0

    def test_rc29_tp0_room_step_drops_its_own_source_the_group_waits(self):
        """02:31:13Z: RED on a773b4ebc4 -- TP0 AssertionError (the metal), TP1/TP2 loaded back. GREEN: NO_TOKEN
        on every rank, nobody loads back, both gathers entered alike."""
        out, adders, _reqs, gather = _group(drop_on=(0,))
        for r in range(3):
            self.assertNotIsInstance(out[r], AssertionError, f"TP{r} died like rc29 02:31:13Z: {out[r]!r}")
            self.assertEqual(out[r], AddReqResult.NO_TOKEN, f"TP{r}: {out[r]!r}")
            self.assertEqual(adders[r].tree_cache.load_backs, [], f"TP{r} loaded back")
            self.assertEqual(adders[r].can_run_list, [])
        self.assertEqual(gather.calls, {0: 2, 1: 2, 2: 2})  # H105f vote + confirm, the same on every rank
        self.assertEqual(sp._H105G_DROPPED["n"], 1)  # TP0 alone saw the drop; its vote carried the group

    def test_source_kept_admits_and_loads_back_as_before(self):
        """No drop: the room step's ADMIT stands and the load-back serves (the false-positive direction -- a
        verdict that refuses a live source would keep the rid queued forever)."""
        out, adders, reqs, gather = _group(drop_on=())
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]!r}")
            self.assertEqual([q.rid for q in adders[r].can_run_list], [H.RID])
            self.assertEqual(len(reqs[r].prefix_indices), H.KV_ROWS)
        self.assertEqual(gather.calls, {0: 2, 1: 2, 2: 2})
        self.assertEqual(sp._H105G_DROPPED["n"], 0)

    def test_mutant_check_only_before_the_room_step_reproduces_the_death(self):
        """Danger direction 1: the check placed BEFORE the eviction (where the source still exists) misses the
        drop the eviction itself causes -- TP0 dies on the metal assert again."""
        real = sp._h105g_target_kept

        def _mutant(adder, req, has_extent, site):
            return True if site == "after_room" else real(adder, req, has_extent, site)

        with patch.object(sp, "_h105g_target_kept", side_effect=_mutant):
            out, _adders, _reqs, _gather = _group(drop_on=(0,))
        self.assertIsInstance(out[0], AssertionError, f"the mutant must reproduce the death: {out!r}")

    def test_mutant_every_evicted_source_counts_as_lost_never_admits(self):
        """Danger direction 2: a predicate that names a LIVE host-backed source lost (e.g. ``node.evicted``
        alone) keeps every host hit queued forever -- the kept-source test must catch it."""
        with patch.object(sp, "_h105g_load_back_target_lost", side_effect=lambda tc, n, req=None: "evicted"):
            out, _adders, _reqs, _gather = _group(drop_on=())
        self.assertEqual(set(out.values()), {AddReqResult.NO_TOKEN}, out)


class AnchorDroppedInWindowTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sp._H105D_REFUSED["n"] = 0
        if hasattr(sp, "_H105G_DROPPED"):
            sp._H105G_DROPPED["n"] = 0

    def test_s109c_releases_the_matched_host_anchor_in_the_room_step_the_group_waits(self):
        """RED before the S109R extension: TP0 loads back with its anchor gone (the prefix is clamped and recomputed,
        ``anchor_lost_loads`` 1). GREEN: TP0 votes NO_TOKEN (``why=anchor_dropped``), every rank keeps the rid; the
        next pass's match reads the anchor from L3 (S109R)."""
        out, adders, gather = _group_anchor()
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.NO_TOKEN, f"TP{r}: {out[r]!r}")
            self.assertEqual(adders[r].tree_cache.load_backs, [])
            self.assertEqual(adders[r].tree_cache.anchor_lost_loads, 0)
        self.assertEqual(gather.calls, {0: 2, 1: 2, 2: 2})
        self.assertEqual(sp._H105G_DROPPED["n"], 1)

    def test_predicate_names_only_a_matched_host_anchor_that_is_gone(self):
        from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        tc = SimpleNamespace(root_node=object(), page_size=1)
        dev = SimpleNamespace(evicted=False, parent=None, component_data={})
        m = SimpleNamespace(value=None, host_value=None)
        node = SimpleNamespace(evicted=False, parent=dev, component_data={ComponentType.MAMBA: m})
        self.assertEqual(sp._h105g_load_back_target_lost(tc, node, SimpleNamespace(mamba_host_hit_length=1)),
                         "anchor_dropped")
        # the false-positive direction: no host hit on the anchor (a worker, or a device state) -> never lost
        self.assertIsNone(sp._h105g_load_back_target_lost(tc, node, SimpleNamespace(mamba_host_hit_length=0)))
        m.host_value = torch.tensor([3])
        self.assertIsNone(sp._h105g_load_back_target_lost(tc, node, SimpleNamespace(mamba_host_hit_length=1)))
        m.host_value, m.value = None, torch.tensor([4])
        self.assertIsNone(sp._h105g_load_back_target_lost(tc, node, SimpleNamespace(mamba_host_hit_length=1)))


# ---- the predicate on a real UnifiedRadixCache ------------------------------------------------------------------

def _urc():
    fx = _load("_urc_unittest_fixture_h105g", _HERE.parent / "mem_cache" / "test_unified_radix_cache_unittest.py")
    fx.get_device = lambda: "cpu"
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.unified_radix_cache import ComponentType

    cfg = fx.CacheConfig(page_size=2, components=(ComponentType.FULL,), kv_size=64, max_context_len=64)
    cache, alloc, _ = fx.build_fixture(cfg)
    for t in ([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 10, 11, 12]):
        v = alloc.alloc(len(t))
        cache.insert(InsertParams(key=RadixKey(array("q", t)), value=v[: len(t)]))
    a = list(cache.root_node.children.values())[0]
    b, c = list(a.children.values())
    return cache, a, b, c, ComponentType.FULL


class RealTreePredicateTest(unittest.TestCase):
    def _host_only(self, cache, node, F):
        node.component_data[F].host_value = torch.arange(len(node.component_data[F].value))
        cache._evict_to_host(node, {ct: 0 for ct in cache.tree_components})
        self.assertTrue(node.evicted and cache._is_host_leaf(node))

    def test_a_backed_host_leaf_is_a_source(self):
        cache, _a, b, _c, F = _urc()
        self._host_only(cache, b, F)
        self.assertIsNone(sp._h105g_load_back_target_lost(cache, b))
        self.assertIsNone(sp._h105g_load_back_target_lost(cache, cache.root_node))

    def test_the_spill_drop_is_named_and_is_the_metal_assert(self):
        """``_evict_host_leaf`` is the W3 host-only spill's drop (host_only_spill.spill_host_only). The handle the
        request kept then fails ``build_hicache_transfers`` exactly like D TP0 02:31:13Z; H105g names it first."""
        from flliper.srt.mem_cache.unified_cache_components.tree_component import CacheTransferPhase

        cache, _a, b, _c, F = _urc()
        self._host_only(cache, b, F)
        cache._evict_host_leaf(b, {ct: 0 for ct in cache.tree_components})
        self.assertIn(sp._h105g_load_back_target_lost(cache, b), ("detached", "no_host_value"))
        with self.assertRaises(AssertionError):
            cache.components[F].build_hicache_transfers(b, CacheTransferPhase.LOAD_BACK)

    def test_device_resident_source_is_never_lost(self):
        cache, _a, b, _c, _F = _urc()
        self.assertIsNone(sp._h105g_load_back_target_lost(cache, b))


if __name__ == "__main__":
    unittest.main()
