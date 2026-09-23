"""#1417 (boot xsn167): the chain a completed prefetch inserted stays
host-locked until the admission pops the outcome or the request is aborted."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.locks = 0


class _Params:
    def to_dec_params(self):
        return "p"


def _tree():
    t = object.__new__(UnifiedRadixCache)
    t.root_node = _Node()
    t.prefetch_loaded_tokens_by_reqid = {}
    t._prefetch_completed_tokens = {}
    t.ongoing_prefetch = {}
    t.inc_host_lock_ref = lambda node: (setattr(node, "locks", node.locks + 1), _Params())[1]
    t.dec_host_lock_ref = lambda node, params=None: setattr(node, "locks", node.locks - 1)
    return t


def test_pin_covers_the_inserted_chain_and_pops_at_admission():
    t = _tree()
    stop = _Node(t.root_node)
    a = _Node(stop)
    b = _Node(a)
    deepest = _Node(b)
    assert t._pin_prefetched_span("r1", deepest, stop) == 3
    assert (deepest.locks, b.locks, a.locks, stop.locks) == (1, 1, 1, 0)
    t.prefetch_loaded_tokens_by_reqid["r1"] = 7
    assert t.pop_prefetch_loaded_tokens("r1") == 7
    assert (deepest.locks, b.locks, a.locks) == (0, 0, 0)
    assert t.pop_prefetch_loaded_tokens("r1") == 0, "a second pop releases nothing twice"


def test_abort_releases_and_the_dict_is_bounded():
    t = _tree()
    stop = _Node(t.root_node)
    n = _Node(stop)
    t._pin_prefetched_span("r2", n, stop)
    t.release_aborted_request("r2")
    assert n.locks == 0
    nodes = []
    for i in range(UnifiedRadixCache._PREFETCH_SPAN_PINS_MAX + 3):
        m = _Node(stop)
        nodes.append(m)
        t._pin_prefetched_span(f"x{i}", m, stop)
    assert len(t._prefetch_span_pins) == UnifiedRadixCache._PREFETCH_SPAN_PINS_MAX
    assert nodes[0].locks == 0 and nodes[-1].locks == 1
