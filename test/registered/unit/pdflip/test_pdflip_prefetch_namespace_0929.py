"""Flipzeit-Regression 29.09. (1/3), NF 09292034 y3k-korr: every salted request
(cache_salt -> extra_key) read P's pages from the store under the ROOT anchor's
namespace (None); its own match (namespace = the salt) missed them, D
re-prefilled the whole prompt (hit0, cached_tokens=0): P-Ende -> erstes Token
16-27 s for 10.5k tokens (Bestform x178 2,1-2,8 s)."""

import asyncio
import collections
import inspect
import json
import logging
import types
import unittest
from array import array
from unittest import mock

import pytest
import torch

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------- 1. namespace
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
    """The unified cache suite's own fixture builder (its file, not a package)."""
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


def _unified_cache():
    fx = _fixture_module()
    fx.get_device = lambda: "cpu"   # a CPU tree: the prefetch path moves no bytes here
    CacheConfig, build_fixture = fx.CacheConfig, fx.build_fixture
    from flliper.srt.mem_cache.unified_radix_cache import ComponentType

    cfg = CacheConfig(page_size=4, components=(ComponentType.FULL,), kv_size=64,
                      max_context_len=64)
    cache, _alloc, _pool = build_fixture(cfg)
    cache.enable_storage = True
    cache.prefetch_threshold = 1
    cache.cache_controller = _FakeController()
    return cache


def test_namespace_rule_request_wins_anchor_is_the_fallback():
    from flliper.srt.mem_cache.unified_radix_cache import prefetch_namespace

    assert prefetch_namespace(anchor_extra_key=None, request_extra_key="salt") == "salt"
    assert prefetch_namespace(anchor_extra_key="a", request_extra_key=None) == "a"
    assert prefetch_namespace(anchor_extra_key=None, request_extra_key=None) is None


def test_y3k_korr_a_salted_span_under_the_root_anchor_is_keyed_by_the_salt():
    """pdflip-36-17: last_host_node = root (nothing matched), loaded=10496, then
    addreq hit0 -- the span sat in namespace None. Base: TypeError (no kwarg)
    and the key's namespace was the root's None."""
    cache = _unified_cache()
    tokens = array("q", range(1, 17))
    cache.prefetch_from_storage("pdflip-36-17", cache.root_node, tokens, extra_key="nfcp-a01-p1")
    _rid, storage_key = cache.cache_controller.prefetch_args
    assert storage_key.extra_key == "nfcp-a01-p1"
    rec = cache.ongoing_prefetch["pdflip-36-17"]
    assert rec.prefetch_key.extra_key == "nfcp-a01-p1"


def test_an_unsalted_request_keeps_the_namespace_none():
    cache = _unified_cache()
    cache.prefetch_from_storage("r", cache.root_node, array("q", range(1, 17)), extra_key=None)
    assert cache.cache_controller.prefetch_args[1].extra_key is None


def test_the_scheduler_hands_the_request_namespace_to_the_unified_tree():
    from flliper.srt.managers import scheduler as sch
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    req = types.SimpleNamespace(extra_key="salt")
    unified = UnifiedRadixCache.__new__(UnifiedRadixCache)
    assert sch._prefetch_namespace_kw(unified, req) == {"extra_key": "salt"}
    assert sch._prefetch_namespace_kw(object(), req) == {}
    src = inspect.getsource(sch.Scheduler._prefetch_kvcache)
    assert "_prefetch_namespace_kw(self.tree_cache, req)" in src
