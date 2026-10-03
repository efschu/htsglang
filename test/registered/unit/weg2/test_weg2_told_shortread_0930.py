"""TS: a follower reads the short tail a TOLD prescribes (NF y4a, b547263cb8).

Death 03:36:24 (P log ...0930_031042, rid weg2-50-159): PP0 registered at head
94080 (span 10241), matched 1152 host tokens and loaded 64 from the store --
told 95296. PP1/PP2 registered one pass later at told with their host walk
already at 95232 (``#1420 WALK-STOP depth=95232``): need 64 < threshold 256,
``#915 PREFETCH REFUSED reason=too_short``, ``#1400 FOLLOWER REGISTRATION
DECLINED ... declined:too_short``, then ``Weg2StoreToldMismatch told=95296
own_prefix=95232 own_loaded=0`` on both followers, W17 GroupDead.

The follower follows PP0's verdict: a read bounded by a told is prescribed,
the #915 threshold (which prices a FRESH read) does not refuse it. PP0 keeps
its threshold for its own intake; nothing is given away (the alternative --
PP0 clamping told to what every follower would open -- gives up the anchor
PP0 already read and falls back to 94080). RED on b547263cb8.
"""

from __future__ import annotations

import inspect
from array import array
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

TOLD, OWN, THRESHOLD, PAGE = 95296, 95232, 256, 64


class _FakeHostPool:
    def alloc(self, num_tokens):
        return torch.arange(num_tokens, dtype=torch.int64)


class _FakeController:
    def __init__(self):
        self.mem_pool_host = _FakeHostPool()
        self.prefetch_tokens_occupied = 0
        self.prefetch_args = None

    def prefetch_rate_limited(self):
        return False

    def prefetch(self, request_id, host_indices, new_input_tokens, last_hash=None,
                 prefix_keys=None, extra_pools=None):
        self.prefetch_args = (request_id, new_input_tokens)
        return mock.Mock()


def _fixture_module():
    import importlib.util
    import pathlib
    import sys

    name = "_urc_unittest_fixture"
    if name not in sys.modules:
        path = (pathlib.Path(__file__).resolve().parents[1] / "mem_cache"
                / "test_unified_radix_cache_unittest.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _follower_tree():
    fx = _fixture_module()
    fx.get_device = lambda: "cpu"
    from sglang.srt.mem_cache.unified_radix_cache import ComponentType

    cfg = fx.CacheConfig(page_size=PAGE, components=(ComponentType.FULL,), kv_size=1024,
                         max_context_len=1024)
    cache, _alloc, _pool = fx.build_fixture(cfg)
    cache.enable_storage = True
    cache.prefetch_threshold = THRESHOLD     # the metal's #915 threshold
    cache.cache_controller = _FakeController()
    return cache


def _follower_read(min_tokens):
    """PP1's registration of weg2-50-159: its host walk ends at 95232, the told
    span beyond it is ONE page (95232..95296)."""
    cache = _follower_tree()
    tail = array("q", range(OWN, TOLD))
    assert len(tail) == TOLD - OWN == 64 < THRESHOLD
    kw = {} if min_tokens is None else {"min_tokens": min_tokens}
    cache.prefetch_from_storage("weg2-50-159", cache.root_node, tail, **kw)
    return "weg2-50-159" in cache.ongoing_prefetch


def test_the_told_read_of_64_tokens_below_the_threshold_is_issued():
    from sglang.srt.managers import weg2_store_told as m

    # the metal: the follower's registration asked with the tree's threshold
    assert _follower_read(None) is False, "the refusal y4a died of (too_short, 64 < 256)"
    # the fix: a told-bounded registration opens the read PP0 already made
    need_min = m.told_read_min_tokens(limit_tokens=TOLD + 1, tail_min=None)
    assert need_min == 1
    assert _follower_read(need_min) is True


def test_an_unbounded_intake_keeps_the_threshold_and_other_minimums():
    from sglang.srt.managers import weg2_store_told as m

    assert m.told_read_min_tokens(limit_tokens=None, tail_min=None) is None   # PP0 intake: 256
    assert m.told_read_min_tokens(limit_tokens=None, tail_min=1) == 1         # store-short tail
    assert m.told_read_min_tokens(limit_tokens=TOLD, tail_min=37) == 1


def test_the_scheduler_registration_applies_it_with_the_told_limit():
    """The follower registration is ``_prefetch_kvcache(req, limit_tokens=...)``
    (``_follower_register``): the minimum it hands the tree comes from the told
    rule, after the tail/park minimums."""
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers import weg2_store_told as m

    src = inspect.getsource(sch.Scheduler._prefetch_kvcache)
    i = src.index("weg2_store_told.told_read_min_tokens(limit_tokens, _tail_min)")
    assert i < src.index("_tail_kw = {\"min_tokens\": _tail_min}")
    assert "limit_tokens=follower_limit_tokens(" in inspect.getsource(m._follower_register)
