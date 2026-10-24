"""MAMBA-EVICT-PROBE-1024 (#228b): every device Mamba state that leaves the unified tree is named.

THE METAL (27B nvfp4duo 93eae72472, boot ..._1024_012625.D.log, D TP0; #224 done/reprice-rest-1024.md case 22-194):
``#59 FINISH-TRACK rid=pdflip-22-176 cache_len=124928`` (01:57:35), front ``D-PROBE rid=pdflip-22-194 device=124928``
(01:57:38.186), D ``PDFLIP ANCHOR-ONLY-READ rid=pdflip-22-194 depth=124928 anchor_was=119552
how=store_miss_unsecured`` (01:57:39) -> u 6259 -> ``RESUME-VIA-P CLASS ... reason=x_refusal_midstream`` -> P computed
2163 where D would have computed 883. Nothing in the log names who dropped the device state: no line, second
resolution, the #1469 notes capped at 600, ``PDFLIP-ANCHOR-LOST at=evict`` only on the internal-tombstone branch and
capped at 32 (32/33 in 8 of today's 27B D logs = the cap, not the count).

WHAT MUST HOLD.
(1) A device Mamba state that leaves the tree without a copy on any tier prints ONE ``MAMBA-EVICT-UNSECURED`` line
    with an epoch-ms timestamp, node, token depth, the site from the caller chain, the anchor's rid and the marks.
(2) A state with a surviving copy (host value with a DEVICE-only eviction, store held, secured mark the store cannot
    contradict) prints no single line; it is counted in the window summary.
(3) A host value the SAME call frees (``EvictLayer.ALL``) is no surviving copy.
(4) Volume: a token bucket bounds the single lines; a suppressed line is counted (summary + cumulative), never the
    event; at most one summary per window.
(5) Side doors: the #1055 reclaim (``_disown_reclaimed_value``) and a tree reset (``MAMBA-EVICT-BULK``) are named.
(6) Switch ``FLLIPER_MAMBA_EVICT_PROBE=0``: silent. An instrument that raises never breaks the eviction.
(7) No behaviour: the eviction frees exactly what it freed before (slot back on the free list, node gone).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache import mamba_evict_probe as MEP  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType, EvictLayer  # noqa: E402

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "pdflip", "test_pdflip_d_seat_compact_0930.py")
if not os.path.exists(_BUILDER):  # 27B line: no D-seat compact test -- the same builder, standalone
    _BUILDER = os.path.join(os.path.dirname(__file__), "_hybrid_tree_builder.py")
_spec = importlib.util.spec_from_file_location("_t_build_mep1024", _BUILDER)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
MC = ComponentType.MAMBA
LOGGER = "flliper.srt.mem_cache.mamba_evict_probe"


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    monkeypatch.setattr(MEP, "_PROBE", None)
    monkeypatch.delenv("FLLIPER_MAMBA_EVICT_PROBE", raising=False)
    yield


def _path(n_tokens=128, start=1000):
    cache, pool, kalloc = T._build()
    toks = list(range(start, start + n_tokens))
    slot = pool.mamba_allocator.alloc(1)
    cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(n_tokens), mamba_value=slot.reshape(1)))
    node = next(n for n in cache._collect_all_nodes()
                if n is not cache.root_node and n.component_data[MC].value is not None)
    return cache, pool, node, int(slot.reshape(-1)[0])


def alloc_for_new_request(cache):
    """Stands in for the admission's ``alloc_req_slots`` -> ``evict(mamba_num=1)``: the site rule reads this name."""
    return cache.evict(EvictParams(num_tokens=0, mamba_num=1))


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.getMessage().startswith(tag)]


def _field(line, key):
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    raise AssertionError(f"{key}= not in {line}")


# ------------------------------------------------------------------ (1) + (7): the unsecured eviction, real tree
def test_unsecured_device_eviction_prints_one_line_with_site_depth_and_ms(caplog):
    cache, pool, node, slot = _path(128)
    node.pdflip_anchor_rid = "pdflip-22-176"
    free_before = pool.mamba_allocator.available_size()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        alloc_for_new_request(cache)
    ls = _lines(caplog, MEP.LOG_UNSECURED)
    assert len(ls) == 1, caplog.messages
    ln = ls[0]
    assert _field(ln, "node") == str(node.id)
    assert _field(ln, "depth") == "128"
    assert _field(ln, "site") == "alloc"
    assert _field(ln, "at") == "alloc_for_new_request"
    assert "_evict_device_leaf" in _field(ln, "via") and "drive_eviction" in _field(ln, "via")
    assert _field(ln, "anchor_rid") == "pdflip-22-176"
    assert _field(ln, "host_too") == "1"          # write-through leaf: target ALL
    assert int(_field(ln, "t_ms")) > 1_700_000_000_000   # epoch ms, joinable with the front's ms clock
    assert _field(ln, "n") == "1"
    # (7) no behaviour: the slot is back on the free list, the state left the tree
    assert pool.mamba_allocator.available_size() == free_before + 1
    assert node.component_data[MC].value is None


# ------------------------------------------------------------------ (2) secured: counted, no single line
def test_secured_mark_prints_no_single_line_but_counts():
    cache, _pool, node, _ = _path(128)
    node.pdflip_anchor_secured = True             # no pool behind the tree: the store cannot tell -> mark counts
    alloc_for_new_request(cache)
    p = MEP.probe()
    assert (p.n, p.n_unsecured) == (1, 0)


def test_host_value_with_device_only_eviction_is_secured(caplog):
    cache, _pool, node, _ = _path(128)
    cd = node.component_data[MC]
    cd.host_value = __import__("torch").tensor([5])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        MEP.note(cache, node, cd, site="park", host_too=False)
    assert not _lines(caplog, MEP.LOG_UNSECURED)
    assert MEP.probe().n_unsecured == 0


def test_host_value_freed_by_the_same_call_is_no_copy(caplog):
    cache, _pool, node, _ = _path(128)
    cd = node.component_data[MC]
    cd.host_value = __import__("torch").tensor([5])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        MEP.note(cache, node, cd, site="ud_drop", host_too=True)
    ls = _lines(caplog, MEP.LOG_UNSECURED)
    assert len(ls) == 1 and _field(ls[0], "host_too") == "1" and _field(ls[0], "site") == "ud_drop"


@pytest.mark.parametrize("host,held,mark,host_too,want", [
    (True, None, False, False, (True, "host")),
    (True, None, False, True, (False, "none")),
    (False, True, False, False, (True, "store")),
    (False, None, True, False, (True, "mark")),
    (False, False, True, False, (False, "none")),     # the store contradicts a stale mark
    (False, None, False, False, (False, "none")),
])
def test_secured_state_table(host, held, mark, host_too, want):
    node = types.SimpleNamespace(pdflip_anchor_secured=mark, hash_value=["h1", "hlast"],
                                 get_last_hash_value=lambda: "hlast")
    cd = types.SimpleNamespace(host_value=(object() if host else None), value=[1], lock_ref=0)
    tree = types.SimpleNamespace(_s109r_anchor_held=lambda lh: held)
    sec, by, _h, _w = MEP.secured_state(tree, node, cd, host_too=host_too)
    assert (sec, by) == want


# ------------------------------------------------------------------ (4) volume
def test_bucket_bounds_single_lines_and_counts_the_suppressed():
    p = MEP.EvictProbe(window_s=10.0, rate_per_s=1.0, burst=3.0)
    emitted = [p.note(0.0, "alloc", False)[0] for _ in range(5)]
    assert emitted == [True, True, True, False, False]
    assert (p.n, p.n_unsecured, p.n_suppressed) == (5, 5, 2)
    assert p.note(2.0, "alloc", False)[0] is True        # refilled: 2 s x 1/s


def test_one_summary_per_window_with_sites_and_cumulative_totals():
    p = MEP.EvictProbe(window_s=10.0)
    assert p.note(0.0, "alloc", True)[1] == []
    assert p.note(3.0, "park", False)[1] == []
    assert p.note(9.9, "alloc", True)[1] == []           # window not closed
    emit, pre = p.note(10.5, "flush", False)
    assert emit is True and len(pre) == 1
    s = pre[0]
    assert s.startswith(MEP.LOG_SUMMARY)
    assert "n=3 secured=2 unsecured=1 suppressed=0" in s
    assert "by_site=alloc:2/0,park:1/1" in s
    assert "cum_n=3 cum_unsecured=1" in s
    assert p.note(11.0, "flush", True)[1] == []          # the next window started at 10.5


def test_summary_rides_the_logger_before_the_single_line(caplog):
    cache, _pool, node, _ = _path(128)
    p = MEP.probe()
    p.note(0.0, "alloc", True)                            # a window opened long ago (t=0 s epoch)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        alloc_for_new_request(cache)
    msgs = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert msgs[0].startswith(MEP.LOG_SUMMARY) and msgs[1].startswith(MEP.LOG_UNSECURED), msgs


# ------------------------------------------------------------------ site classification
@pytest.mark.parametrize("names,want", [
    (["_cascade_evict", "_evict_to_host", "_evict_device_leaf", "drive_eviction", "evict", "alloc_token_slots"],
     ("alloc", "alloc_token_slots")),
    (["drive_eviction", "evict", "_alloc_mamba_slot", "prepare_for_caching_req"], ("alloc", "_alloc_mamba_slot")),
    (["_pdflip_cap_release", "_pdflip_cap_after_insert"], ("cap", "_pdflip_cap_release")),
    (["_cascade_evict", "_evict_to_host", "_pdflip_d_park_running"], ("park", "_pdflip_d_park_running")),
    (["_ud_drop_unbacked_leaf", "_evict_device_leaf", "drive_eviction", "evict", "x_unknown"],
     ("ud_drop", "_ud_drop_unbacked_leaf")),
    (["_cascade_evict", "_evict_to_host", "compact_seats", "tick"], ("compact", "compact_seats")),
    (["drive_eviction", "evict"], ("other", "-")),
    (["drive_eviction", "evict", "mystery_caller"], ("other", "mystery_caller")),
])
def test_classify(names, want):
    assert MEP.classify(names) == want


# ------------------------------------------------------------------ (5) side doors
def test_reclaim_side_door_is_named(caplog):
    cache, _pool, node, _ = _path(128)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        cache._disown_reclaimed_value(node, MC)
    ls = _lines(caplog, MEP.LOG_UNSECURED)
    assert len(ls) == 1 and _field(ls[0], "site") == "reclaim"
    assert node.component_data[MC].value is None


def test_reset_prints_one_bulk_line(caplog):
    cache, _pool, _node, _ = _path(128)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        cache.reset()
    ls = _lines(caplog, MEP.LOG_BULK)
    assert len(ls) == 1, caplog.messages
    assert _field(ls[0], "site") == "reset" and _field(ls[0], "states") == "1"
    assert _field(ls[0], "listed_unsecured") == "1"
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        cache.reset()                                     # empty tree: silent
    assert not _lines(caplog, MEP.LOG_BULK)


def test_device_only_funnel_call_with_host_value_is_secured(caplog):
    """The funnel with target DEVICE (the demotion of a backed node): the host value survives the call."""
    cache, _pool, node, _ = _path(128)
    cd = node.component_data[MC]
    cd.host_value = __import__("torch").tensor([5])
    comp = cache.components[MC]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        cache._evict_component_and_detach_lru(node, comp, target=EvictLayer.DEVICE, tracker=None)
    assert not _lines(caplog, MEP.LOG_UNSECURED)
    assert (MEP.probe().n, MEP.probe().n_unsecured) == (1, 0)
    assert cd.value is None and cd.host_value is not None


# ------------------------------------------------------------------ (6) switch + never raises
def test_switch_off_is_silent(caplog, monkeypatch):
    monkeypatch.setenv("FLLIPER_MAMBA_EVICT_PROBE", "0")
    cache, _pool, node, _ = _path(128)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        alloc_for_new_request(cache)
    assert not [r for r in caplog.records if r.name == LOGGER]
    assert node.component_data[MC].value is None


def test_default_is_on():
    assert MEP.enabled() is True


def test_a_raising_store_probe_never_breaks_the_eviction(monkeypatch):
    cache, pool, node, _ = _path(128)

    def boom(*a, **k):
        raise RuntimeError("store down")

    monkeypatch.setattr(type(cache), "_s109r_anchor_held", boom, raising=False)
    monkeypatch.setattr(MEP, "_alt_counts", boom)
    free_before = pool.mamba_allocator.available_size()
    alloc_for_new_request(cache)
    assert pool.mamba_allocator.available_size() == free_before + 1


def test_the_funnel_calls_the_probe():
    """Wiring: the one funnel names the probe (mutant: the call removed -> test 1 red as well)."""
    import inspect

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._evict_component_and_detach_lru)
    assert "_mamba_evict_probe.note_device_evict(self, node, cd, host_too=EvictLayer.HOST in target)" in src
