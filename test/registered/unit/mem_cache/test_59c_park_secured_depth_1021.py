"""#59c PARK-SECURED (rc27 boot ..._893d09fa3f_1021_023225, D TP0 02:44:15): the park answer never credits KV the
flip's sleep drops.

THE METAL. ``PARK-BACKUP REFUSED need=33 ... verdict=stands (P recomputes the park)`` (node 155 of pdflip-2-30,
[133632, 139008), no host copy, no L3 copy), ``D-MEM-SCHED KV-COMPACT verdict=refused ... no free page below 512 for
un-backed node 155 (84 pages)``, and one line later ``#59b PARK-RESUMABLE mode=form-a-dcp-min pdflip-2-30=139008``.
The sleep's reset dropped node 155 (``S85 PARK-END-UNBACKED ... store_end=133632 lost_tokens=5376``); P resumed at
133632 (``PDFLIP-CLAIM-SHORT rid=pdflip-2-30 ... by=kv_prefix tokens=133632``). The answer named 5376 tokens no tier
held.

THE RULE. Each rank caps its park depth at its SECURED end (every node of the path holds its KV off the device: a
host copy / arena claim, or ``l3_present``) and re-probes there; the cap rides the same single MIN reduce as the
depth. The read clamp keeps the uncapped depth (a tail the sleep's flush still secures is read, not recomputed).

On a real ``UnifiedRadixCache`` (the unittest fixture, FULL only, page 1): three nodes of 16 tokens on the device;
keys are ``array("q")`` as the admission probe builds them."""
import os
import sys
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers import pdflip_resumable_depth as RD  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import InsertParams  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

FULL = ComponentType.FULL
SPAN = 16
PS_SOLO = types.SimpleNamespace(pp_size=1, tp_size=1, attn_dp_size=1, attn_tp_rank=0)
PS_TP = types.SimpleNamespace(pp_size=1, tp_size=2, attn_dp_size=1, attn_tp_rank=0)


def _nodes_by_depth(cache):
    out = {}
    for n in cache._collect_all_nodes():
        if n is cache.root_node:
            continue
        d, p = 0, n
        while p is not cache.root_node:
            d += len(p.key)
            p = p.parent
        out[d] = n
    return out


@pytest.fixture
def park():
    """The parked request's path: top [0,16) and mid [16,32) have their host copy (backed), the park end
    [32,48) is on the device only (its backup refused) -- node 155's shape."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(FULL,)))
    for depth in (1, 2, 3):
        n = depth * SPAN
        cache.insert(InsertParams(key=RadixKey(array("q", range(1, 1 + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
    nodes = _nodes_by_depth(cache)
    top, mid, end = nodes[SPAN], nodes[2 * SPAN], nodes[3 * SPAN]
    for i, n in enumerate((top, mid)):
        n.component_data[FULL].host_value = torch.arange(100 + i * SPAN, 100 + (i + 1) * SPAN, dtype=torch.int64)
    req = types.SimpleNamespace(rid="pdflip-2-30", origin_input_ids=list(range(1, 3 * SPAN + 1)) + [999],
                                output_ids=[], extra_key=None, positional_embed_overrides=None)
    return types.SimpleNamespace(cache=cache, top=top, mid=mid, end=end, req=req)


def test_rc27_the_answer_stops_at_the_last_secured_node_the_clamp_keeps_the_admission_depth(park):
    """RED on a773b4ebc4 (no split; the answer = the admission depth 48 over the device-only park end). GREEN: the
    answer is 32 -- what survives the reset -- and the read clamp keeps 48."""
    raw, ans = RD.park_depths_split(park.cache, [park.req], PS_SOLO)
    assert raw == {"pdflip-2-30": 3 * SPAN}
    assert ans == {"pdflip-2-30": 2 * SPAN}, "the park answer credits device-only KV the sleep's reset drops"
    assert RD.park_depths(park.cache, [park.req], PS_SOLO) == ans


def test_a_backed_or_store_present_park_end_is_credited_whole(park):
    park.end.component_data[FULL].host_value = torch.arange(200, 200 + SPAN, dtype=torch.int64)
    assert RD.park_depths_split(park.cache, [park.req], PS_SOLO)[1] == {"pdflip-2-30": 3 * SPAN}
    park.end.component_data[FULL].host_value = None
    park.end.l3_present = True
    assert RD.park_depths_split(park.cache, [park.req], PS_SOLO)[1] == {"pdflip-2-30": 3 * SPAN}


def test_a_device_only_node_in_the_middle_cuts_everything_behind_it(park):
    """Danger direction (a depth beyond the secured pages): a device-only node in the middle of the path takes the
    answer to its start, whatever is backed behind it."""
    park.mid.component_data[FULL].host_value = None
    park.end.component_data[FULL].host_value = torch.arange(200, 200 + SPAN, dtype=torch.int64)
    assert RD.park_depths_split(park.cache, [park.req], PS_SOLO)[1] == {"pdflip-2-30": SPAN}


def test_one_reduce_carries_depth_and_cap_and_the_group_takes_the_min(park):
    """A TP group: ONE MIN reduce over depth + cap (2 slots per rid); a peer whose park end is backed does not lift
    this rank's cap (MIN)."""
    calls = []

    def reduce_min(vals):
        calls.append(list(vals))
        peer = [3 * SPAN, 3 * SPAN]          # the peer secured its whole path
        return [min(a, b) for a, b in zip(vals, peer)]

    raw, ans = RD.park_depths_split(park.cache, [park.req], PS_TP, reduce_min=reduce_min)
    assert len(calls) == 1 and len(calls[0]) == 2
    assert raw == {"pdflip-2-30": 3 * SPAN} and ans == {"pdflip-2-30": 2 * SPAN}


def test_an_unwalkable_path_keeps_the_old_answer_counted(park, monkeypatch):
    before = RD.SECURED_STATS["unknown"]
    monkeypatch.setattr(RD, "secured_end", lambda t, r, d: None)
    assert RD.park_depths_split(park.cache, [park.req], PS_SOLO)[1] == {"pdflip-2-30": 3 * SPAN}
    assert RD.SECURED_STATS["unknown"] == before + 1


def test_the_log_names_the_cap(park, caplog):
    import logging

    caplog.set_level(logging.INFO)
    RD.park_depths_split(park.cache, [park.req], PS_SOLO)
    line = next(m for m in caplog.messages if m.startswith("#59b PARK-RESUMABLE"))
    assert "pdflip-2-30=32" in line and "#59c PARK-SECURED capped=pdflip-2-30:48>32" in line


def test_the_park_answers_the_cap_and_clamps_with_the_admission_depth():
    """Wiring: both park sites take ``park_depths_split``; the read clamp gets the raw depths, the answer the capped."""
    import pathlib

    here = pathlib.Path(__file__).resolve()
    src = next(p for p in here.parents if (p / "python" / "flliper").is_dir())
    text = (src / "python/flliper/srt/pdflip/d_park_runtime.py").read_text()
    assert text.count("pdflip_resumable_depth.park_depths_split(") == 2
    assert "pdflip_resumable_depth.park_depths(" not in text
    assert text.count("resumable_raw.get(str(req.rid))") == 2
    assert "pdflip_resumable_depth=resumable," in text
    lane = text[text.index("def park_rids("):]
    assert "secured=False)" in lane[:lane.index("resumable_raw.get(")], "the lane park (no sleep) is capped"


def test_a_peer_whose_park_end_is_unsecured_takes_the_group_answer_down(park):
    """Rank unity: this rank secured its whole path, a peer (a token owner whose share was refused) did not -- the
    group answer is the peer's secured end (MIN over the cap slots), the read clamp stays at the admission depth."""
    park.end.component_data[FULL].host_value = torch.arange(200, 200 + SPAN, dtype=torch.int64)

    def reduce_min(vals):
        peer = [3 * SPAN, SPAN]               # the peer: admission 48, secured only to 16
        return [min(a, b) for a, b in zip(vals, peer)]

    raw, ans = RD.park_depths_split(park.cache, [park.req], PS_TP, reduce_min=reduce_min)
    assert raw == {"pdflip-2-30": 3 * SPAN} and ans == {"pdflip-2-30": SPAN}


def test_a_park_no_sleep_follows_answers_the_admission_depth_in_one_reduce(park):
    """The priority-lane park keeps D awake (its device KV stays): ``secured=False`` answers the admission depth and
    reduces only the depths (one slot per rid, as before #59c)."""
    calls = []

    def reduce_min(vals):
        calls.append(list(vals))
        return list(vals)

    raw, ans = RD.park_depths_split(park.cache, [park.req], PS_TP, reduce_min=reduce_min, secured=False)
    assert raw == ans == {"pdflip-2-30": 3 * SPAN}
    assert len(calls) == 1 and len(calls[0]) == 1
