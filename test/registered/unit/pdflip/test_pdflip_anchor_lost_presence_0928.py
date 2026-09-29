"""ANCHOR-LOST: presence whose Mamba anchor D's sleep dropped is no credit.

Hermetic (no CUDA). Metal (bridge f833fcbb2d): D wrote the anchor 37376 in
16-46's own extend ('#1469 RETAIN rid=pdflip-16-46 ... cache_len=37376',
19:50:38), 16-48 resumed from it ('#new-token: 195, #cached-token: 37376',
19:50:51) and its finish told the front a resumable depth on that path. The
D sleep at 19:51:15-16 refused host backups ('#1421 BACKUP-REFUSED
why=mamba_pin ... pins=8/8', then 'why=mamba_claim' after '#1427
ARENA-CLAIM REFUSED ... no free slot') and reset the tree ('#1470
FLUSH-PUBLISH ... unbacked_left=1', 'Cache flushed successfully!'). After the
wake the store gave the KV back (store_hit=37376) but no anchor ('[#904
match-census] reached=37376 accepted=0 ... MambaComponent:absent=37376'),
while the front still priced pdflip-16-54 SHORT (front_price 2349, presence
35942 src=d_leg2_cached): 17 s for a D seat, then X-gate W31 on 38291
tokens and a W50 reroute through P. 12 such reroutes in rc12z26, 8 on the
bridge.

What these cases pin:

* the tree names every device-only anchor with its depth (the flush's
  reset drops exactly those) and BACKUP-REFUSED carries depth and rid;
* the sleep leg's answer carries the depths (tokenizer merge: union), the
  front parses them and retracts every entry whose credit stood on one;
* an entry on a surviving anchor keeps its credit.
"""

import json
import logging
from types import SimpleNamespace

from flliper.srt.managers.tokenizer_control_mixin import _merge_memory_occupation_reports
from flliper.srt.mem_cache.unified_radix_cache import ComponentType, UnifiedRadixCache
from flliper.srt.pdflip import front as fr


# ------------------------------------------------------------------ D tree
def _node(parent, n, device=True, host=False, rid=None, mamba=True):
    comp = {}
    if mamba:
        comp[ComponentType.MAMBA] = SimpleNamespace(value=object() if device else None,
                                                    host_value=object() if host else None)
    node = SimpleNamespace(key=list(range(n)), parent=parent, children={}, component_data=comp,
                           pdflip_anchor_rid=rid)
    if parent is not None:
        parent.children[len(parent.children)] = node
    return node


def _tree():
    t = object.__new__(UnifiedRadixCache)
    t.root_node = SimpleNamespace(key=[], parent=None, children={}, component_data={})
    return t


def test_the_tree_names_device_only_anchors_with_their_depth():
    t = _tree()
    a = _node(t.root_node, 32896, host=True)  # backed: survives the reset
    b = _node(a, 4096, mamba=False)  # KV only
    c = _node(b, 384, rid="pdflip-16-46")  # 37376, device only -> lost
    _node(c, 256)  # 37632, device only -> lost
    assert UnifiedRadixCache.pdflip_unbacked_anchors(t) == [(37376, "pdflip-16-46"), (37632, None)]
    assert UnifiedRadixCache.pdflip_node_depth(t, c) == 37376


class _PinnedTree(UnifiedRadixCache):
    _mamba_pin_budget = 8

    def _mamba_pins_held(self):
        return 8


def test_backup_refused_names_depth_and_rid(caplog):
    t = object.__new__(_PinnedTree)
    t.root_node = SimpleNamespace(key=[], parent=None, children={}, component_data={})
    a = _node(t.root_node, 37312)
    node = _node(a, 64, rid="pdflip-16-46")
    node.id, node.evicted, node.backuped = 326, False, False
    with caplog.at_level(logging.WARNING):
        UnifiedRadixCache._1421_refused(t, "mamba_claim", node)
    assert "why=mamba_claim node=326 tokens=64 depth=37376 rid=pdflip-16-46" in caplog.text


def test_the_flush_keeps_the_depths_until_the_sleep_answer(caplog):
    from flliper.srt.managers.scheduler import Scheduler

    sched = SimpleNamespace(tree_cache=SimpleNamespace(
        pdflip_unbacked_anchors=lambda: [(37376, "pdflip-16-46"), (37632, None)]))
    with caplog.at_level(logging.WARNING):
        Scheduler._pdflip_note_lost_anchors(sched)
    assert "PDFLIP-ANCHOR-LOST at=flush n=2 depths=[37376, 37632]" in caplog.text
    assert Scheduler.pdflip_take_anchors_lost(sched) == [37376, 37632]
    assert Scheduler.pdflip_take_anchors_lost(sched) == []  # read once


def test_the_tokenizer_answer_carries_the_union():
    r1 = SimpleNamespace(per_tag={"kv_cache": [1.0, 2.0]}, critical_path="a", anchors_lost=[37376])
    r2 = SimpleNamespace(per_tag={}, critical_path=None, anchors_lost=[37632, 37376])
    out = _merge_memory_occupation_reports([r1, r2])
    assert out["anchors_lost"] == [37376, 37632]
    quiet = _merge_memory_occupation_reports([SimpleNamespace(per_tag={"t": [1.0, 1.0]},
                                                              critical_path="x", anchors_lost=None)])
    assert "anchors_lost" not in quiet  # nothing lost: the answer is the old shape


def test_the_release_leg_only_reads_the_ledger():
    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as WeightUpdater,
    )

    taken = []
    wu = SimpleNamespace(scheduler=SimpleNamespace(
        pdflip_take_anchors_lost=lambda: taken.append(1) or [37376]))
    assert WeightUpdater._pdflip_anchors_lost_for(wu, "resume tags=['kv_cache']") == []
    assert not taken
    assert WeightUpdater._pdflip_anchors_lost_for(wu, "release tags=['kv_cache']") == [37376]


# -------------------------------------------------------------------- front
PREFIX = "S" * 9000  # the agent's shared prefix
T48 = PREFIX + "turn-48 " * 3000
T54 = T48[: len(PREFIX) + 23000] + "turn-54 " * 400


def _spans():
    s = fr.SpanLRU(agent_span=False)
    s.record_presence(T48, cached_tokens=37568, resumable_depth=37376)
    return s


def test_metal_form_the_lost_anchor_retracts_the_short_price():
    s = _spans()
    est = int(len(T54) / fr.CHARS_PER_TOKEN) + 1
    rem, known = s.uncached_tokens(T54, est)
    assert known and rem < est // 2  # before: priced SHORT on 16-48's presence
    body = json.dumps({"per_tag": {"kv_cache": [1, 2]}, "anchors_lost": [37376, 37632]})
    gone = fr.retract_lost_anchors(s, fr.anchors_lost(body))
    assert len(gone) == 1
    assert s.uncached_tokens(T54, est) == (est, False)  # the whole prompt, like D's X-gate


def test_a_surviving_anchor_keeps_its_credit():
    s = _spans()
    assert fr.retract_lost_anchors(s, [16384, 20992]) == []
    assert len(s.entries) == 1


def test_an_uncapped_entry_goes_on_its_measured_depth():
    s = fr.SpanLRU(agent_span=False)
    s.record_presence(T48, cached_tokens=37376)  # an older D without the depth field
    assert len(fr.retract_lost_anchors(s, [37376])) == 1


def test_answers_without_the_field_retract_nothing():
    assert fr.anchors_lost("not json") == []
    assert fr.anchors_lost(json.dumps({"per_tag": {}})) == []
    assert fr.anchors_lost(json.dumps({"anchors_lost": [0, "x", 37376]})) == [37376]


def test_the_flip_reads_the_sleeper_answer_when_d_sleeps():
    import inspect

    src = inspect.getsource(fr.Front.flip)
    i = src.index("s_done, s_per_tag, s_crit = completed_tags(s_body)")
    assert 'if src == "D":\n            self._retract_lost_anchors(anchors_lost(s_body))' in src[i:i + 400]
