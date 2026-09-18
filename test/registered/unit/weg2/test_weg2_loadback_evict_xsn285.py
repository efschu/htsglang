"""weg2xsn285 (18.09.2026): group D held three finished 98k prompts
(retained, backed up, evictable); the fourth's load-back asked for 99570
tokens against a uniform availability floor of 68841 and `load_back`
refused every pass without evicting -- 102,462 WEG2-LOADBACK-WAIT lines in
150 s (the wait counter lived on the per-pass adder), the flip's drain
stalled, IDLE-WEDGE. Now a short floor drains the evictable leaves
rank-uniformly and refuses THIS pass; the counter lives on the tree."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def test_a_short_floor_evicts_rank_uniformly_before_refusing():
    from sglang.srt.mem_cache import unified_radix_cache as urc
    src = open(urc.__file__).read()
    i = src.index("if floor < kv_tokens:")
    blk = src[i:i + 2600]
    assert "_ev = int(self.evictable_size())" in blk
    assert "self.evict(EvictParams(num_tokens=_ev))" in blk
    assert "WEG2-LOADBACK-EVICT" in blk
    # still refuses this pass, after the eviction, releasing both locks
    j = blk.index("self.dec_lock_ref(best_match_node, ancestor_lock_params)")
    assert "return False" in blk[j:j + 300]
    assert blk.index("self.evict(EvictParams") < j


def test_the_loadback_wait_counter_lives_on_the_tree_not_the_adder():
    from sglang.srt.managers import schedule_policy as sp
    src = open(sp.__file__).read()
    i = src.index("WEG2-LOADBACK-WAIT")
    blk = src[i - 1200:i]
    assert '_tc = self.tree_cache' in blk and '_tc._weg2_loadback_no_room = _n' in blk
    assert 'getattr(self, "_weg2_loadback_no_room"' not in blk
