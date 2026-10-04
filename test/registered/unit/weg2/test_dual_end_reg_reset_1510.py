"""#1510: _reset_full must empty the dual end-anchor registry (B9d death 18:04:36Z,
'Sanity check FAILED: H-leaf extra' -- stale end-anchor nodes of the destroyed tree re-entered
the new tree's host-leaf set via weg2_dual_release_ended -> _weg2_dual_release_ref).

The full reset needs a whole tree/pool harness; this pins the three facts that matter:
the registry is the dual module's Registry (entries dict of rid -> (node, t0)), _reset_full clears it,
and it does so after the old tree's host values were released and before the cap's waiting releases.
"""
import inspect

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U
from sglang.srt.weg2 import dual_anchor_release as DAR


def test_registry_is_an_entries_dict_of_node_and_time():
    r = DAR.Registry()
    r.note("weg2-0-1", object())
    assert list(r.entries) == ["weg2-0-1"] and len(r.entries["weg2-0-1"]) == 2
    r.entries.clear()
    assert r.entries == {}


def test_reset_full_clears_the_dual_end_registry_in_the_right_place():
    src = inspect.getsource(U._reset_full)
    assert "_weg2_dual_end_reg" in src and "entries.clear()" in src
    i_rel = src.index("self._release_host_values_before_reset()")
    i_clr = src.index("_dual_end_reg.entries.clear()")
    i_cap = src.index("27B line (24.09.)")
    assert i_rel < i_clr < i_cap


def test_the_clear_is_a_noop_without_the_registry():
    # the registry is created lazily (getattr ... None) only in the dual layout: flip/NF/INT8 never have it
    src = inspect.getsource(U._reset_full)
    assert 'getattr(self, "_weg2_dual_end_reg", None)' in src and "if _dual_end_reg is not None" in src
