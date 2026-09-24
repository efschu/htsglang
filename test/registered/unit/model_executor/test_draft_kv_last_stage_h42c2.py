"""fnFL2 H42c-2: the draft KV of a draft-KV-only producer is charged where it lives.

x155's dry run (--draft-kv-on-p on) priced P's KV at 2176/1088/816 MiB against
x160's 1904/816/544 (--draft-kv-on-p off): +272 MiB on EVERY stage, and the P
edge fell 0.332 -> 0.324. The rank did the same in its cell (x161
PHASE-FLIP-SEAM-RESERVE cell 8704/4352/3264 vs x160 7616/3264/2176). But the
producer -- and its one draft KV pool -- exists on the last PP stage only
(Scheduler._maybe_init_draft_kv_producer: ``is_last_rank``; x161: "DRAFT-KV-
PRODUCER armed stage=2/3", PP1 "no-drafter", ``KV Cache is allocated`` K
0.88/0.38 GB on PP0/PP1 exactly as in x160, the draft pool K 0.13 GB on PP2).
Pinned: both sides charge the draft on the last stage alone.
"""

import inspect
from types import SimpleNamespace

from sglang.srt.model_executor import pool_configurator as pc
from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


def _mr(pp_rank, pp_size=3, kv_only=True):
    return SimpleNamespace(
        pp_rank=pp_rank,
        pp_size=pp_size,
        server_args=SimpleNamespace(speculative_draft_kv_only=kv_only),
    )


def test_only_the_last_stage_hosts_the_draft_kv_only_pool():
    assert [pc.draft_kv_pool_on_this_rank(_mr(r)) for r in range(3)] == [False, False, True]


def test_every_other_form_is_unchanged():
    assert all(pc.draft_kv_pool_on_this_rank(_mr(r, kv_only=False)) for r in range(3))
    assert pc.draft_kv_pool_on_this_rank(_mr(0, pp_size=1))
    assert pc.draft_kv_pool_on_this_rank(
        SimpleNamespace(pp_rank=0, pp_size=3, server_args=SimpleNamespace())
    )


def test_both_draft_scalings_of_the_cell_are_gated():
    src = inspect.getsource(pc)
    assert "_draft_pool_here = draft_kv_pool_on_this_rank(mr)" in src
    assert ") and not mr.is_draft_worker and _draft_pool_here:" in src
    assert src.count("and not mr.is_draft_worker and _draft_pool_here:") == 2


def test_the_eagle_cell_matches_the_allocation_on_every_stage():
    """The EAGLE scaling on x161's stages (7 / 3 / 2 attention layers, 1088 B
    per layer and token, one draft layer): the phantom 1088 B/token leaves the
    first two stages, the last keeps it."""
    target = {0: 7616, 1: 3264, 2: 2176}
    layers = {0: 7, 1: 3, 2: 2}
    got = {}
    for r in range(3):
        cell = target[r]
        if pc.draft_kv_pool_on_this_rank(_mr(r)):
            cell = int(cell * (1 + 1 / layers[r]))
        got[r] = cell
    assert got == {0: 7616, 1: 3264, 2: 3264}


def test_the_planner_prices_the_draft_on_the_last_stage():
    kw = dict(tokens=262144, attn_layers_by_stage=[7, 3, 2], kv_heads=2, head_dim=256,
              v_head_dim=256, kv_dtype_bytes=1.0)
    base = pp_cut.kv_reserve_mib_per_stage(**kw)
    last = pp_cut.kv_reserve_mib_per_stage(**kw, draft_attn_layers_by_stage=[0, 0, 1])
    assert [round(x) for x in base] == [1904, 816, 544]  # x160's line
    assert [round(x) for x in last] == [1904, 816, 816]  # not x155's 2176/1088/816

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    assert "draft_attn_layers_by_stage=[0] * (n_stages_p - 1) + [_dr]," in src
    assert "draft_attn_layers_by_stage=[_dr] * n_stages_p" not in src
