"""NF-RC31 C (rc30 review, Folgefix C): does the S109R ``anchor_dropped`` check need to run without a load-back extent?

The review asked: ``_h105g_target_kept`` returns True at once without ``has_extent``, so a match with
``mamba_host_hit_length > 0`` and ``host_hit_length == 0`` (KV on the device, the anchor host-only) is not checked,
although the room step can release that anchor (S109c). Is this form admitted on D (Form A TP0), and what happens
to the host anchor without ``init_load_back``?

What these tests pin (the answer, so the code stays as it is):

(1) The S109c / S109R form -- device KV WITH its Full host copy, the anchor host-only -- always HAS an extent: the
    match's device validators include the mamba one, so ``last_device_node`` is the deepest node with a DEVICE
    anchor above it and ``FullComponent.finalize_match_result`` counts the node's Full host rows as host hit.
    The H105g/S109R check runs for it.
(2) ``host_hit_length == 0`` with ``mamba_host_hit_length > 0`` is only reachable where no node between the last
    device anchor and ``best_match`` carries a Full host copy (the AUX-HOST-LAW shape S109R handles at the match).
    There ``stamp_state_aligned_extent`` clears the extent, ``_pp_load_back_extent`` is None, and ``add_one_req``
    reaches ``init_load_back`` only under ``if _lb_extent:`` -- the pass extends from ``last_device_node`` (its device
    anchor) and never reads the host anchor. A dropped anchor has no reader in this pass: no stale handle (the H105g
    death needs ``init_load_back``), and a NO_TOKEN vote would only delay the request -- the next pass's match
    resumes from the same device anchor (the host one is gone). So the check stays bound to ``has_extent``.
(3) The predicate itself sees the drop (``anchor_dropped``) -- the room step's verdict for the no-extent form is
    unchanged (True), by decision, not by blindness.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt import rank_role  # noqa: E402
from flliper.srt.managers import pp_admission_congruence as PAC  # noqa: E402
from flliper.srt.managers import schedule_policy as SP  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import MatchResult  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.full_component import FullComponent  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


def _cd(value=None, host=None):
    return types.SimpleNamespace(value=value, host_value=host, lock_ref=0, host_lock_ref=0)


def _node(parent, full_dev, full_host, m_dev=None, m_host=None, nid=0):
    cds = [None] * (max(int(FULL), int(MAMBA)) + 1)
    cds[int(FULL)] = _cd(full_dev, full_host)
    cds[int(MAMBA)] = _cd(m_dev, m_host)
    return types.SimpleNamespace(id=nid, parent=parent, children={}, key=None, evicted=full_dev is None,
                                 backuped=full_host is not None, component_data=cds)


def _tree():
    root = _node(None, torch.zeros(0), None, nid=0)
    a = _node(root, torch.arange(64), None, m_dev=torch.tensor([3]), nid=1)       # device KV + device anchor
    return root, a


def test_the_s109c_form_with_a_full_host_copy_has_an_extent():
    root, a = _tree()
    n = _node(a, torch.arange(64, 128), torch.arange(64), m_host=torch.tensor([5]), nid=2)
    comp = types.SimpleNamespace(component_type=FULL, cache=types.SimpleNamespace(root_node=root))
    res = MatchResult(device_indices=torch.arange(64), last_device_node=a, last_host_node=n, best_match_node=n,
                      mamba_host_hit_length=1)
    out = FullComponent.finalize_match_result(comp, res, None, [], 0)
    assert out.host_hit_length == 64 > 0          # -> stamp_state_aligned_extent sets an extent, H105g/S109R run


def test_the_no_extent_form_clears_the_extent():
    root, a = _tree()
    n = _node(a, torch.arange(64, 128), None, m_host=torch.tensor([5]), nid=2)   # no Full host copy
    comp = types.SimpleNamespace(component_type=FULL, cache=types.SimpleNamespace(root_node=root))
    res = MatchResult(device_indices=torch.arange(64), last_device_node=a, last_host_node=n, best_match_node=n,
                      mamba_host_hit_length=1)
    out = FullComponent.finalize_match_result(comp, res, None, [], 0)
    assert out.host_hit_length == 0 and out.mamba_host_hit_length == 1
    req = types.SimpleNamespace(rid="r", host_hit_length=0, mamba_host_hit_length=1,
                                **{PAC.LOAD_BACK_EXTENT_ATTR: 4096})
    assert PAC.stamp_state_aligned_extent(req) is None
    assert SP._pp_load_back_extent(req) is None


def test_add_one_req_reaches_init_load_back_only_with_an_extent():
    """Guard for the decision: the only load-back call in ``add_one_req`` sits under ``if _lb_extent:``. Should a
    load-back ever run without an extent, the S109R check must move out of ``has_extent`` (this test then fails)."""
    src = inspect.getsource(SP.PrefillAdder.add_one_req)
    assert src.count(".init_load_back(") == 1, "a second load-back site in add_one_req: re-check Folgefix C"
    i = src.index("_lb_extent = _pp_load_back_extent(req)\n")
    j = src.index("if _lb_extent:\n", i)
    k = src.index(".init_load_back(", j)
    assert i < j < k
    assert "_lb_extent =" not in src[j:k]


def test_the_room_step_keeps_its_verdict_for_the_no_extent_form_although_the_predicate_sees_the_drop(monkeypatch):
    root, a = _tree()
    n = _node(a, torch.arange(64, 128), None, m_host=torch.tensor([5]), nid=2)
    req = types.SimpleNamespace(rid="pdflip-2-15", best_match_node=n, host_hit_length=0, mamba_host_hit_length=1,
                                full_untruncated_fill_ids=list(range(200)), prefix_indices=torch.arange(64),
                                **{PAC.LOAD_BACK_EXTENT_ATTR: None})

    class _Tc:
        root_node, page_size = root, 64

        def evictable_size(self):
            return 4096

        def evict(self, params):   # the room step's eviction releases the host-only anchor (S109c)
            n.component_data[int(MAMBA)].host_value = None
            alloc.free += 4096
            return types.SimpleNamespace(num_tokens_evicted=4096)

    class _Alloc:
        free = 0

        def available_size(self):
            return self.free

    alloc = _Alloc()
    adder = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, tree_cache=_Tc(), rem_chunk_tokens=4096,
                                  page_size=64, can_run_list=[], rem_total_tokens=0)
    monkeypatch.setattr(rank_role, "form_a_token_cut_active", lambda: True)
    monkeypatch.setattr(SP, "payable_evictable_or", lambda tc, fn: fn())
    assert SP._h105d_cut_load_back_room(adder, req, evict=True) is True
    assert n.component_data[int(MAMBA)].host_value is None                         # the anchor did go
    assert SP._h105g_load_back_target_lost(adder.tree_cache, n, req) == "anchor_dropped"
