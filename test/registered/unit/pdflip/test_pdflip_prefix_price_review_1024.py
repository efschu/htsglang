"""Review #215 D-PREFIX-PRICE + #215b FRONT-DEVICE-LOCKED (SOLL v2 R7, R11) -- reviewer tests.

rv1 (MUSS 1, danger: wrong displacement): the head's credit and the victims' joint release count the SAME rows when
the victims are the only holders of the head's locked gap. D's verdict (``kv_displace_would_fit``, basis=adder) reads
the CREDITED price (``lifetime_refusal``) and adds the joint release (``joint_leaving_rows``), which frees the gap
node because its FULL lock equals the number of victims on its path. Once the victim is gone the gap is lock 0, the
credit is gone, and the head's price is the old one again: the verdict was too optimistic by the gap. D displaces a
running decode and the head still does not fit.

rv3 (test gap, danger: overbooking): ``locked_device_gap`` must never credit ``req.last_node``'s own rows -- they are
``prefix_indices`` already. Every author test uses the root as ``last_node`` (0 rows), so a walk that included it
would survive them all.
"""
from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_rv215_base", os.path.join(_HERE, "test_pdflip_d_prefix_price_1024.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from flliper.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip import d_prefix_price as dpp  # noqa: E402

FULL = base.FULL
_env = base._env  # the author's autouse fixture (server args, env)


def test_rv1_victim_alone_holds_the_gap_the_verdict_must_not_count_it_twice():
    """The victim 18-189 is the ONLY holder of the shared prompt (no other decode). Budget 40000 before.
    Credited price 86340 + 1 <= 40000 + release (35943 own + 18137 prompt + 110 tail + 401 reserve = 94591): the
    verdict says one victim leaves. After it left: prompt lock 0, price back to 104477 > 94591 -- refused."""
    root, prompt, host = base._path(prompt_lock=0)
    tree, victim, own = base._victim_tree(prompt, root)          # the victim's own lock: prompt lock 1
    assert prompt.component_data[FULL].lock_ref == 1
    budget = 40000
    adder = base._adder(root, budget)
    assert base._admit(adder, base._req(root, host)) == AddReqResult.NO_TOKEN
    _rid, price, _b = adder.lifetime_refusal
    assert price == base.PRICE_NEW                               # credited: the victim's lock counts as "another holder"
    release = (base.VICTIM_SPAN - base.SHARED) + base.SHARED + (base.VICTIM_SEAT - base.VICTIM_SPAN) + base.RESERVE
    # the truth: after the displacement the head is refused at the old price
    prompt.component_data[FULL].lock_ref -= 1
    own.component_data[FULL].lock_ref -= 1
    after = base._adder(root, budget + release)
    assert base._admit(after, base._req(root, host)) == AddReqResult.NO_TOKEN
    assert after.lifetime_refusal[1] == base.PRICE_OLD > budget + release
    # D's verdict, taken BEFORE the displacement on the live tree, must therefore say "nobody leaves"
    prompt.component_data[FULL].lock_ref += 1
    own.component_data[FULL].lock_ref += 1
    older = SimpleNamespace(rid=base.HEAD, origin_input_ids=[0] * base.N, output_ids=[], prefix_indices=[],
                            last_node=root, best_match_node=host, host_hit_length=base.HOST,
                            sampling_params=SimpleNamespace(max_new_tokens=64000), mamba_pool_idx=0,
                            finished=lambda: False)
    setattr(older, base.LOAD_BACK_EXTENT_ATTR, base.EXTENT)
    sched = SimpleNamespace(waiting_queue=[older], tree_cache=tree,
                            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 65447))
    view = {"rid": base.HEAD, "price": price, "budget": budget, "reserve": {base.VICTIM: base.RESERVE},
            "pool": 65447, "pass": None}
    assert dpr.kv_displace_would_fit(sched, base.HEAD, [victim], view=view) is False, \
        "credit and joint release counted the 18137 prompt rows twice: a victim leaves and the head still waits"


def test_rv3_last_nodes_own_rows_are_prefix_never_credited():
    """last_node = a state node with 1000 device rows (prefix_indices), locked; below it the stateless prompt gap
    (18137, locked) and the host chain. The credit is the gap only."""
    root = base._root()
    anchor = base._unode(root, 1000, lock=2)                     # last_node: the deepest node WITH a state
    gap = base._unode(anchor, base.SHARED, lock=3)
    host = base._unode(gap, base.HOST, on_device=False)
    req = base._req(root, host)
    req.last_node = anchor
    req.prefix_indices = list(range(1000))
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, base.EXTENT) == base.SHARED
    # extent shorter than the gap: clipped, still without the anchor's rows
    assert dpp.locked_device_gap(SimpleNamespace(root_node=root), req, 500) == 500
