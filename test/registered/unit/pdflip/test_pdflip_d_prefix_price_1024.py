"""D-PREFIX-PRICE 1024 (#215; SOLL v2 R7, R11): D's lifetime price credits the device rows the load-back MAPS.

Metal specimen: 27B int4ar ``_1024_003732`` (/spinning/docker-acceptance/27b/evidence), D TP0/TP1 01:02:45:
  ``PDFLIP-HANDBACK rid=pdflip-18-188 N=100380 d_prefix=0 anchor=96704 d_uncached=3676``
  ``D-ADMIT-REFUSED ... gate=lifetime price=104477 budget=64355.368 extend=100380 max_new=4096 prefix=0``
  ``#823 HEAD-VOTE ANCHOR rid=pdflip-18-188 matched=78567 (device 0 + host 78567) anchor_depth=96704``
  after displacing 18-189: budget 100799.776, price still 104477 -> 181 refusals in 18.7 s
  front USAGE later: ``cached_device=18137 cached_l2=78567`` -- the shared system prompt WAS mapped from the device.
The mamba validator cut ``prefix_indices`` to 0 (the 18137-row prompt node is a split node without a state), the
load-back maps it anyway (``init_load_back._collect_new_prefix_indices`` walks ``best_match_node -> last_node``).
Price with the credit: 100380 - 18137 + 4096 + 1 = 86340 -- below the post-displacement budget 100799 (KVD review
SOLL 5: the acceptance case).

The REAL ``PrefillAdder.add_one_req`` runs on a REAL ``UnifiedTreeNode`` path; the load-back is the stop marker
(reaching it = every budget gate admitted).

Danger directions, one group each:
  * zu hoch (the defect)   -- r1 (specimen after the displacement: admitted), r2 (before it: refused, price 86340,
                              ``dev_gap=18137`` on the line), r3 (D's displacement verdict on the corrected price: one
                              victim leaves and the head then fits), r4 (chunk-admit form credits the load-back part)
  * Ueberbuchung (OOM)     -- o1 (an UNLOCKED device node is not credited: it is in the budget's evictable and the
                              load-back locks it out), o2 (no load-back extent: no credit), o3 (a host-only node is
                              the host hit: charged), o4 (credit clipped to the extent), o5 (a path that never meets
                              ``last_node``: 0), o6 (unreadable class / MagicMock nodes: 0, fast), o7 (a node locked
                              only by its own device rows counts once: lock 1 is enough, lock 0 is not)
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.managers.pp_admission_congruence import LOAD_BACK_EXTENT_ATTR  # noqa: E402
from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult  # noqa: E402
from flliper.srt.mem_cache.evict_frontier_census import AUX_LOCKED_ATTR  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

try:  # the red run (before the fix) has no module: the tests that need it fail by name, not the whole file
    from flliper.srt.pdflip import d_prefix_price as dpp  # noqa: E402
except ImportError:  # pragma: no cover - red run only
    dpp = None

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)

HEAD, VICTIM = "pdflip-18-188", "pdflip-18-189"
N = 100380                  # PDFLIP-HANDBACK N
SHARED = 18137              # the shared system prompt: device-resident, no state, locked by the running decodes
HOST = 78567                # the host chain (HEAD-VOTE ANCHOR host 78567)
EXTENT = SHARED + HOST      # 96704: #1040 EXTENT STATE-ALIGN extent=96704 device_len=0
MAX_NEW = 4096
PRICE_OLD = N + MAX_NEW + 1                 # 104477 (metal)
PRICE_NEW = N - SHARED + MAX_NEW + 1        # 86340
BUDGET_BEFORE, BUDGET_AFTER = 64355, 100799  # refuse 1 / refuse 3 budgets (TP0)
VICTIM_SEAT, VICTIM_SPAN, RESERVE = 54190, 54080, 401


class _ReachedLoadBack(Exception):
    """The load-back is the stop marker: every budget gate in front of it admitted the request."""


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_GROUP", "FLLIPER_TP_ADMISSION_AGREE", "FLLIPER_PDFLIP_DUAL_LAYOUT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sp, "_PDFLIP_CHUNK_ADMIT", False)
    monkeypatch.setattr(sp, "_PDFLIP_PARK_ON", None)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


# ------------------------------------------------------------------------------------- real tree nodes, real locks
def _root():
    r = UnifiedTreeNode(COMPONENTS)
    r.key = []
    r.component_data[FULL].value = []
    r.component_data[FULL].lock_ref = 1          # the root is pinned
    return r


def _unode(parent, rows, on_device=True, lock=0):
    n = UnifiedTreeNode(COMPONENTS)
    n.parent = parent
    n.key = list(range(rows))
    n.component_data[FULL].value = list(range(rows)) if on_device else None
    n.component_data[FULL].lock_ref = lock
    return n


def _path(prompt_lock=3, prompt_on_device=True):
    """root -> prompt (18137, no state, ``prompt_lock`` running decodes hold it) -> host chain (78567, evicted)."""
    root = _root()
    prompt = _unode(root, SHARED, on_device=prompt_on_device, lock=prompt_lock)
    host = _unode(prompt, HOST, on_device=False)
    return root, prompt, host


def _tree_cache(root):
    tc = MagicMock()
    tc.root_node = root
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.init_load_back.side_effect = _ReachedLoadBack()
    return tc


def _adder(root, available, chunk=4096):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    rb = MagicMock()
    rb.reqs = []
    adder = PrefillAdder(
        page_size=1, tree_cache=_tree_cache(root), token_to_kv_pool_allocator=a, running_batch=rb,
        new_token_ratio=1.0, rem_input_tokens=10**9, rem_chunk_tokens=chunk, num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    adder.form_a_admission_follow = None
    adder.tp_admission_agree = None
    return adder


def _req(root, best, extent=EXTENT, host_hit=HOST):
    req = MagicMock(spec=Req)
    req.rid = HEAD
    req.priority = 0
    req.prefix_indices = []                      # the mamba-cut device match: d_prefix=0
    req.full_untruncated_fill_ids = list(range(N))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64000, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = host_hit > 0
    req.host_hit_length = host_hit
    req.swa_host_hit_length = 0
    req.mamba_host_hit_length = 0
    req.last_node = root                         # last_device_node = the root (device 0)
    req.last_host_node = best
    req.best_match_node = best
    req.born_spilled = False
    req.born_spilled_deep = False
    req.pdflip_parked_span = 0
    setattr(req, LOAD_BACK_EXTENT_ATTR, extent)
    return req


def _admit(adder, req):
    """``"admit"`` when the request reached the load-back (every budget gate took it), else the AddReqResult."""
    try:
        return adder.add_one_req(req, truncation_align_size=None)
    except _ReachedLoadBack:
        return "admit"


# ======================================================================== zu hoch (the defect)
def test_r1_specimen_after_the_displacement_the_head_is_admitted():
    """budget 100799 after 18-189 left: the true price 86340 fits; the metal price 104477 refused 181 times."""
    root, _prompt, host = _path()
    adder = _adder(root, BUDGET_AFTER)
    got = _admit(adder, _req(root, host))
    assert got == "admit", (got, adder.lifetime_refusal)
    assert adder.lifetime_refusal is None


def test_r2_specimen_before_the_displacement_refused_at_the_true_price_named_on_the_line():
    root, _prompt, host = _path()
    adder = _adder(root, BUDGET_BEFORE)
    req = _req(root, host)
    assert _admit(adder, req) == AddReqResult.NO_TOKEN
    rid, price, budget = adder.lifetime_refusal
    assert (rid, price, budget) == (HEAD, PRICE_NEW, BUDGET_BEFORE)
    nums = _stamped_nums(adder, req)
    assert nums.get("dev_gap") == SHARED and nums.get("prefix") == 0 and nums.get("price") == PRICE_NEW, nums


def _stamped_nums(adder, req):
    """The gate numbers ``_refuse`` stamped for the not-added line (pdflip/d_admit_refusal.stamp)."""
    from flliper.srt.pdflip import d_admit_refusal as dar
    seen = {}

    def _grab(a, r, gate, res=None, nums=None):
        seen.update(nums or {})

    orig = dar.stamp
    dar.stamp = _grab
    try:
        adder.lifetime_refusal = None
        adder.add_one_req(req, truncation_align_size=None)
    except _ReachedLoadBack:
        pass
    finally:
        dar.stamp = orig
    return seen


def _victim_tree(prompt, root):
    own = _unode(prompt, VICTIM_SPAN - SHARED, lock=1)
    prompt.component_data[FULL].lock_ref += 1     # the victim's own lock on the shared prompt
    tree = SimpleNamespace(root_node=root, page_size=1, uniform_avail_floor=None,
                           evictable_size=lambda: 0, supports_mamba=lambda: False)
    setattr(tree, AUX_LOCKED_ATTR, {})
    victim = SimpleNamespace(rid=VICTIM, origin_input_ids=[0] * VICTIM_SEAT, output_ids=[], last_node=own,
                             prefix_indices=[], sampling_params=SimpleNamespace(max_new_tokens=64000),
                             mamba_pool_idx=0, finished=lambda: False)
    return tree, victim, own


def test_r3_d_verdict_on_the_true_price_one_victim_leaves_and_the_head_then_fits():
    """KVD review SOLL 5: 86340 against 100799. D's verdict reads the adder's refusal (price, budget); with the true
    price the guaranteed release of 18-189 (36053 + 401) covers it -> 18-189 leaves; the next pass admits the head."""
    root, prompt, host = _path()
    adder = _adder(root, BUDGET_BEFORE)
    assert _admit(adder, _req(root, host)) == AddReqResult.NO_TOKEN
    _rid, price, budget = adder.lifetime_refusal
    tree, victim, own = _victim_tree(prompt, root)
    older = SimpleNamespace(rid=HEAD, origin_input_ids=[0] * N, output_ids=[], last_node=None, prefix_indices=[],
                            sampling_params=SimpleNamespace(max_new_tokens=64000), mamba_pool_idx=0,
                            finished=lambda: False)
    sched = SimpleNamespace(waiting_queue=[older], tree_cache=tree,
                            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 65447))
    view = {"rid": HEAD, "price": price, "budget": budget, "reserve": {VICTIM: RESERVE}, "pool": 65447, "pass": None}
    assert dpr.kv_displace_would_fit(sched, HEAD, [victim], view=view) is True
    # the victim leaves: its own lock goes, the three other decodes still hold the prompt -> the credit stands
    prompt.component_data[FULL].lock_ref -= 1
    own.component_data[FULL].lock_ref -= 1
    assert prompt.component_data[FULL].lock_ref == 3
    adder2 = _adder(root, BUDGET_AFTER)
    assert _admit(adder2, _req(root, host)) == "admit"


def test_r4_chunk_admit_form_takes_the_gap_off_the_whole_charged_load_back(monkeypatch):
    """Per-chunk admission charges the load-back WHOLE (Q-700) -- the mapped device gap is no load-back row."""
    monkeypatch.setattr(sp, "_PDFLIP_CHUNK_ADMIT", True)
    root, _prompt, host = _path()
    adder = _adder(root, 10)                     # refuse, to read the price
    assert _admit(adder, _req(root, host)) == AddReqResult.NO_TOKEN
    # lb = 96704 - 18137 = 78567, chunk on the rest min(100380 - 18137 - 78567, 4096) = 3676, + 4096 + 1
    assert adder.lifetime_refusal[1] == HOST + 3676 + MAX_NEW + 1


# ======================================================================== Ueberbuchung (danger: OOM)
def test_o1_an_unlocked_device_gap_is_not_credited():
    """lock 0: the rows are in the budget's evictable count and the load-back locks them out of it -- charging them is
    exact (vLLM ``get_num_blocks_to_allocate``: a computed block with ref_cnt 0 counts as one to allocate)."""
    root, _prompt, host = _path(prompt_lock=0)
    adder = _adder(root, BUDGET_AFTER)
    assert _admit(adder, _req(root, host)) == AddReqResult.NO_TOKEN
    assert adder.lifetime_refusal[1] == PRICE_OLD


def test_o2_no_load_back_extent_no_credit():
    root, _prompt, host = _path()
    adder = _adder(root, BUDGET_AFTER)
    assert _admit(adder, _req(root, host, extent=None, host_hit=0)) == AddReqResult.NO_TOKEN
    assert adder.lifetime_refusal[1] == PRICE_OLD


def test_o3_a_host_only_node_is_the_host_hit_and_stays_charged():
    root, _prompt, host = _path(prompt_on_device=False)
    adder = _adder(root, BUDGET_AFTER)
    assert _admit(adder, _req(root, host)) == AddReqResult.NO_TOKEN
    assert adder.lifetime_refusal[1] == PRICE_OLD


def test_o4_credit_is_clipped_to_the_load_back_extent():
    root, _prompt, host = _path()
    req = _req(root, host, extent=10000)
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, 10000) == 10000
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, EXTENT) == SHARED


def test_o5_a_path_that_never_meets_last_node_gives_nothing():
    root, _prompt, host = _path()
    req = _req(root, host)
    req.last_node = _unode(root, 5, lock=1)      # not an ancestor of best_match_node
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, EXTENT) == 0
    host.parent = None                           # broken chain
    req.last_node = root
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, EXTENT) == 0


def test_o6_unreadable_nodes_give_nothing_and_fast():
    req = SimpleNamespace(last_node=MagicMock(), best_match_node=MagicMock())
    t0 = time.perf_counter()
    assert dpp.locked_device_gap(MagicMock(), req, EXTENT) == 0
    assert time.perf_counter() - t0 < 0.5         # stops at the first node, never walks MagicMock parents


def test_o7_lock_one_is_enough_lock_zero_is_not():
    root, prompt, host = _path(prompt_lock=1)
    req = _req(root, host)
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, EXTENT) == SHARED
    prompt.component_data[FULL].lock_ref = 0
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, EXTENT) == 0
