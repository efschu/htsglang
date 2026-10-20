"""SA-RACE (NF bc35462240, PP0 10:31:31-34, pdflip-56-163).

A SHORT read marks the operation terminated inside the KV transfer; the io
thread then reads the recurrent state at the deepest landed anchor and only
afterwards sets ``pool_transfers_done``. ``can_terminate_prefetch`` must not
let ``terminated`` reap the read in between (the reap would cut on the
store's presence answer and insert a stateless prefix -> #TF told=0).
"""

import time
from types import SimpleNamespace

PAGE = 64


class _Op(SimpleNamespace):
    def is_terminated(self):
        return self.terminated


def _tree(budget_s=100.0):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = object.__new__(UnifiedRadixCache)
    tree.prefetch_stop_policy = "timeout"
    tree.page_size = PAGE
    tree.prefetch_timeout_base = budget_s
    tree.prefetch_timeout_per_page = 0.0
    tree._all_reduce_attn_groups = lambda states, op, label="": None
    return tree


def _short_op(**kw):
    base = dict(
        request_id="pdflip-sa-race",
        hash_value=["h"] * 10,
        token_ids=[0] * 10 * PAGE,
        completed_tokens=4 * PAGE,  # short: 4 of 10 pages landed
        terminated=True,  # the KV transfer marked it on the short batch
        pool_transfers=["mamba"],
        pool_transfers_done=False,  # the SA state read still runs
        read_start_time=time.monotonic(),
        start_time=time.monotonic(),
    )
    base.update(kw)
    return _Op(**base)


def test_short_read_is_held_while_its_state_read_runs():
    assert not _tree().can_terminate_prefetch(_short_op())


def test_short_read_reaps_once_the_state_read_is_done():
    assert _tree().can_terminate_prefetch(_short_op(pool_transfers_done=True))


def test_no_component_pools_no_hold():
    assert _tree().can_terminate_prefetch(_short_op(pool_transfers=None))


def test_queued_never_started_read_is_not_held():
    # terminated by a stop/revoke before the io thread picked it up
    assert _tree().can_terminate_prefetch(_short_op(read_start_time=None))


def test_hold_is_bounded_by_the_timeout_budget():
    op = _short_op(start_time=time.monotonic() - 10.0)
    assert _tree(budget_s=1.0).can_terminate_prefetch(op)
