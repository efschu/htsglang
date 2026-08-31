"""#1063 + #1062: the fence decider must SEPARATE its three states, and the key
trace must be able to see a WRITE.

MATCHED TO THE FAILURE CLASS OF BOTH EDITS, which is not "does it import".

* #1063's class is a census that collapses: if `evicted_since_flip` cannot be
  told from `still_absent`, the whole two-point design is pointless and the
  boot answers nothing -- which is exactly what a single observation at
  re-admission would have done. So the test drives a real transition.
* #1062's class is the one it exists to fix: an instrument that never observes
  the direction it was built to compare. The old probe sat on
  `_get_component_key`, whose five call sites are all lookups, so it printed
  read keys only and its silence meant nothing (indicator law). The test
  asserts the probe now sits on the funnel BOTH directions traverse.
"""

import os

import sglang.srt.mem_cache.hicache_flip_writeback as fw
from sglang.srt.mem_cache.canonical_page_store import marker_path, part_path


class _Backend:
    """Minimal stand-in with the two path methods the probe uses."""

    def __init__(self, root):
        self.root = root

    def _sharded_path(self, stem):
        return os.path.join(self.root, f"{stem}.bin")

    def _flat_path(self, stem):
        return os.path.join(self.root, f"{stem}.bin")

    def _get_suffixed_key(self, key):
        return f"{key}_cfg"


def _reset():
    fw._1063_AT_FENCE.clear()
    fw._1063_STATE.clear()


def test_three_states_are_distinguishable(tmp_path):
    _reset()
    b = _Backend(str(tmp_path))
    open(b._sharded_path("done_cfg"), "wb").close()
    open(part_path(b._sharded_path("torn_cfg")), "wb").close()
    open(marker_path(b._sharded_path("torn_cfg")), "wb").close()
    assert fw._1063_stem_state(b, "done_cfg") == "readable"
    assert fw._1063_stem_state(b, "torn_cfg") == "assembling"
    assert fw._1063_stem_state(b, "never_cfg") == "absent"


def test_marker_alone_still_reads_as_assembling(tmp_path):
    """`a writer acting alone leaves nothing readable behind` -- either sidecar."""
    _reset()
    b = _Backend(str(tmp_path))
    open(marker_path(b._sharded_path("m_cfg")), "wb").close()
    assert fw._1063_stem_state(b, "m_cfg") == "assembling"


def test_eviction_is_only_countable_against_the_fence_snapshot(tmp_path, caplog):
    """THE POINT OF TWO POINTS: 'gone now' vs 'never there' are the same stat().

    Without the fence snapshot both are `absent` and the census would report one
    number for two conditions with opposite fixes.
    """
    _reset()
    b = _Backend(str(tmp_path))
    open(b._sharded_path("keep_cfg"), "wb").close()
    open(b._sharded_path("lose_cfg"), "wb").close()
    # Fence snapshot: both readable, one never written.
    for stem in ("keep_cfg", "lose_cfg", "ghost_cfg"):
        fw._1063_AT_FENCE[stem] = fw._1063_stem_state(b, stem)
    assert fw._1063_AT_FENCE == {
        "keep_cfg": "readable",
        "lose_cfg": "readable",
        "ghost_cfg": "absent",
    }
    # The evictor takes one of them between the flip and the re-admission.
    os.remove(b._sharded_path("lose_cfg"))

    class _Tree:
        class cache_controller:
            storage_backend = b

    with caplog.at_level("WARNING"):
        fw._1063_probe_since_fence(_Tree())
    text = caplog.text
    assert "present_and_readable=1" in text, text
    assert "evicted_since_flip=1" in text, text
    assert "still_absent=1" in text, text


def test_torn_assembly_is_reported_as_its_own_state(tmp_path, caplog):
    _reset()
    b = _Backend(str(tmp_path))
    open(b._sharded_path("t_cfg"), "wb").close()
    fw._1063_AT_FENCE["t_cfg"] = "readable"
    # The cutover tears the writer group apart: the .bin is gone, a partial remains.
    os.remove(b._sharded_path("t_cfg"))
    open(part_path(b._sharded_path("t_cfg")), "wb").close()

    class _Tree:
        class cache_controller:
            storage_backend = b

    with caplog.at_level("WARNING"):
        fw._1063_probe_since_fence(_Tree())
    assert "complete_marker_absent=1" in caplog.text, caplog.text
    assert "evicted_since_flip=0" in caplog.text, caplog.text


def test_probe_never_raises_on_a_broken_backend(caplog):
    _reset()

    class _Broken:
        def _sharded_path(self, stem):
            raise RuntimeError("no store")

    assert fw._1063_stem_state(_Broken(), "x") == "unknown"


def test_key_trace_now_sits_on_the_symmetric_funnel():
    """#1062: the probe must be where BOTH directions pass.

    Source-level, and deliberately so: the defect being fixed is a PLACEMENT,
    and placement is exactly what a behavioural call on one direction cannot
    demonstrate. Both IO halves must route through the instrumented funnel.
    """
    import inspect

    from sglang.srt.mem_cache.hicache_storage import HiCacheFile

    log_key = inspect.getsource(HiCacheFile._log_key)
    comp_key = inspect.getsource(HiCacheFile._get_component_key)
    write_page = inspect.getsource(HiCacheFile._write_page)
    read_page = inspect.getsource(HiCacheFile._read_page)

    assert "#969G KEY" in log_key, "the trace must live on the symmetric funnel"
    assert "#969G KEY" not in comp_key, "the lookup-only funnel must not carry it"
    assert "_log_key(" in write_page, "write half must traverse the funnel"
    assert "_log_key(" in read_page, "read half must traverse the funnel"
    # And the corrected comment must not re-assert the false claim.
    assert "read and write alike" not in comp_key.lower() or "LIED" in comp_key


def test_key_trace_prints_a_suppressed_count_rather_than_going_silent():
    """A capped trace must name its own denominator (denominator law)."""
    import inspect

    from sglang.srt.mem_cache.hicache_storage import HiCacheFile

    src = inspect.getsource(HiCacheFile._log_key)
    assert "SUPPRESSED" in src, src
    assert "_969g_suppressed" in src, src
