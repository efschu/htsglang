# SPDX-License-Identifier: Apache-2.0
"""Q-1220 DUAL NOT-NAMED GIVE-BACK (27B NVFP4 dual y9d2, 6c7c37327f, P PP1 death
04:18:20Z: '#968 PREFIX MATERIALISATION SHORTFALL ... prefix_len=8192, this rank
holds 0', PP0 resident=8192).

Metal, P mamba pool 8 slots on every PP rank, weg2-0-30 chunked at told=8192 with
weg2-0-35/37/6 queued behind it (told=8192):
  PP0      mamba usage 0.25 on every chunk pass (2 of 8 held)
  PP1/PP2  0.25 -> 0.38 -> 0.50 -> 0.50 -> 0.62 (5 of 8), one step per absorbed
           queued told -- the COW slot each queued request's match drew stayed
           with it at the follower-only ``pp_not_named`` exit.

RED on 6c7c37327f: the exit has no give-back (structural test) and the follower
replica holds 5 slots where PP0 holds 2 (simulation). GREEN: the dual-gated hook
gives the slot back exactly like PP0's #991 admission revert.
FLIP UNCHANGED: off the dual layout the hook reads and writes nothing.
"""
from __future__ import annotations

import inspect
import os
import re
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator  # noqa: E402
from sglang.srt.mem_cache.common import release_admission_acquired_mamba_slot  # noqa: E402

POOL = 8  # 'Mamba Cache is allocated. max_mamba_cache_size: 8' (P PP0/PP1/PP2, y9d2)
DUAL = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}


def _flip_envs():
    """Env shapes that are NOT the dual layout (flip/INT8/NF forms)."""
    return [{}, {"SGLANG_WEG2_GROUP": "P"}, {"SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P"},
            {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "D"}]


class FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.session = None
        self.mamba_pool_idx = None
        self.mamba_slot_acquired_this_admission = False
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.mamba_loadback_anchor_adopted = False


def make_tree():
    pool = types.SimpleNamespace(mamba_allocator=MambaSlotAllocator(POOL, device="cpu"))
    return types.SimpleNamespace(req_to_token_pool=pool)


def held(tree) -> int:
    alloc = tree.req_to_token_pool.mamba_allocator
    return alloc.size - int(alloc.available_size())


def cow_acquire(tree, req):
    """finalize_match_result's speculative COW draw (only when no slot yet)."""
    if req.mamba_pool_idx is None:
        slot = tree.req_to_token_pool.mamba_allocator.alloc(1)
        assert slot is not None
        req.mamba_pool_idx = slot[0]
        req.mamba_slot_acquired_this_admission = True
    req.mamba_cow_src_index = torch.tensor([1])


def _hook():
    from sglang.srt.weg2 import dual_not_named_giveback as G

    return G


# ------------------------------------------------------------------ the specimen

def _run_specimen(env, *, follower: bool, passes: int = 5):
    """y9d2: 2 slots held (the chunked weg2-0-30 + one more), 3 queued told
    requests visited each pass. PP0: the adder refuses each -> #991 revert.
    Follower: pp_not_named -> the Q-1220 exit hook."""
    G = _hook()
    tree = make_tree()
    base = tree.req_to_token_pool.mamba_allocator.alloc(2)
    assert base is not None
    queued = [FakeReq(r) for r in ("weg2-0-35", "weg2-0-37", "weg2-0-6")]
    seen = []
    for _ in range(passes):
        for req in queued:
            cow_acquire(tree, req)
            if follower:
                G.give_back_not_named(req, tree, env=env)
            else:
                release_admission_acquired_mamba_slot(req, tree, site="admission_revert")
        seen.append(held(tree))
    return seen, queued


def test_follower_holds_what_pp0_holds_on_the_dual_layout():
    """RED without the hook: follower 5 of 8 (0.62) vs PP0 2 of 8 (0.25)."""
    pp0, _ = _run_specimen(DUAL, follower=False)
    pp1, queued = _run_specimen(DUAL, follower=True)
    assert pp0 == [2] * 5
    assert pp1 == pp0, f"follower replica holds {pp1} slots, PP0 {pp0}"
    for req in queued:
        assert req.mamba_pool_idx is None
        assert req.mamba_slot_acquired_this_admission is False
        assert req.mamba_cow_src_index is None


def test_the_unfixed_follower_reproduces_the_metal_numbers():
    """The specimen's own numbers, measured on the model: without a give-back
    at the exit the follower ends at 5 of 8 = the 0.62 PP1/PP2 printed."""
    pp1, _ = _run_specimen({}, follower=True)  # hook inert = the pre-fix exit
    assert pp1[-1] == 5
    assert round(pp1[-1] / POOL, 2) == 0.62


def test_the_scheduler_exit_calls_the_hook_before_continue():
    """RED on 6c7c37327f: `_note_skip("pp_not_named", ...)` is followed by a bare
    `continue` -- no give-back on that exit."""
    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    m = re.search(r'_note_skip\("pp_not_named", req\.rid\)(.*?)\n\s*continue\n', src, re.S)
    assert m, "the pp_not_named exit is gone"
    assert "give_back_not_named(req, self.tree_cache)" in m.group(1)


# ------------------------------------------------------------------ the other direction

def test_batch_owned_slot_is_never_given_back():
    G = _hook()
    tree = make_tree()
    req = FakeReq("weg2-0-30")
    req.mamba_pool_idx = tree.req_to_token_pool.mamba_allocator.alloc(1)[0]
    req.mamba_slot_acquired_this_admission = False  # HybridReqToTokenPool.alloc took it
    assert G.give_back_not_named(req, tree, env=DUAL) is False
    assert req.mamba_pool_idx is not None and held(tree) == 1


def test_session_slot_is_never_given_back():
    G = _hook()
    tree = make_tree()
    req = FakeReq("s")
    req.session = object()
    cow_acquire(tree, req)
    assert G.give_back_not_named(req, tree, env=DUAL) is False
    assert held(tree) == 1


def test_no_slot_and_double_exit_are_noops():
    G = _hook()
    tree = make_tree()
    req = FakeReq("r")
    assert G.give_back_not_named(req, tree, env=DUAL) is False
    cow_acquire(tree, req)
    assert G.give_back_not_named(req, tree, env=DUAL) is True
    assert G.give_back_not_named(req, tree, env=DUAL) is False
    assert held(tree) == 0


def test_marker_line(monkeypatch):
    G = _hook()
    tree = make_tree()
    req = FakeReq("weg2-0-35")
    cow_acquire(tree, req)
    lines = []
    monkeypatch.setattr(G, "_N", [0])  # the rate limit is per process
    monkeypatch.setattr(G.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    G.give_back_not_named(req, tree, env=DUAL)
    assert any(G.MARKER in s and "rid=weg2-0-35" in s for s in lines), lines


# ------------------------------------------------------------------ flip unchanged

class TestQ1220FlipUnchanged:
    @pytest.mark.parametrize("env", _flip_envs())
    def test_hook_reads_and_writes_nothing_off_the_dual_layout(self, env):
        G = _hook()
        # a bare tree: any attribute read past the gate would raise
        bare = types.SimpleNamespace()

        class Strict:
            def __getattr__(self, name):
                raise AssertionError("flip form read req.%s" % name)

        assert G.give_back_not_named(Strict(), bare, env=env) is False

    @pytest.mark.parametrize("env", _flip_envs())
    def test_flip_exit_keeps_the_pre_fix_slot_holding(self, env):
        """Byte-identical to 6c7c37327f off the dual layout: the slot stays."""
        tree = make_tree()
        req = FakeReq("r")
        cow_acquire(tree, req)
        slot = int(req.mamba_pool_idx)
        assert _hook().give_back_not_named(req, tree, env=env) is False
        assert int(req.mamba_pool_idx) == slot
        assert req.mamba_slot_acquired_this_admission is True
        assert req.mamba_cow_src_index is not None
        assert held(tree) == 1

    def test_process_env_default_is_inert(self, monkeypatch):
        monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT", raising=False)
        tree = make_tree()
        req = FakeReq("r")
        cow_acquire(tree, req)
        assert _hook().give_back_not_named(req, tree) is False
        assert held(tree) == 1
