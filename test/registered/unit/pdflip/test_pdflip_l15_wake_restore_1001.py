"""AP L15-12c-A part 2 (1001): the hold-aware D wake restore on cap>0 ranks.

The commit under test (f37b72e1fa) makes ``_pdflip_wake_restore_pools`` ask
``_l15_wake_hold_signal`` first.  With the master (``FLLIPER_PDFLIP_L15``) on and
this rank's local manifest present and its planner cap > 0 the restore KEEPS
the hold instead of wiping it:

  * ``req_to_token_pool.clear(keep_mamba_rows=anchor_slots)`` (fb301a9357
    re-reserves the mamba anchor rows [1, keep)),
  * allocator ``clear()`` + ``l15_retain.reserve_slots`` over the manifest
    span slots (the held slots stay out of ``free_pages``),
  * the ``FLLIPER_FLUSH_ZERO_KV`` scrub bounded to rows >= rows_by_rank[rank]
    (the held compact prefix [0, keep) keeps its KV),
  * the TMS keep set cleared per allocation base after the resume (every L15
    wake, hold AND fallback; master off never touches it).

Master off or no local manifest (absent, cap 0): today's restore, byte
identical -- same call sequence, no manifest consumed when the master is off.

Hermetic: the helpers run UNBOUND on a SimpleNamespace fake self with fake
pools/allocator/adapter; the tensors are real (CPU) so the bounded scrub and
the allocator reservation are checked on actual values, not on mocks.
``current_platform.synchronize()`` is a no-op on the CPU platform (verified),
so no monkeypatch is needed for the scrub.
"""

import os
import pathlib
import sys
from types import SimpleNamespace

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from flliper.srt.pdflip import l15_manifest  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager
SIZE = 16          # fake allocator size
KEEP = 2           # rows_by_rank[rank]: the held compact prefix
ANCHOR = 5         # manifest anchor_slots -> keep_mamba_rows


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _ReqToTokenPool:
    def __init__(self):
        self.clear_calls = []

    def clear(self, *a, **k):
        self.clear_calls.append((a, k))


class _Allocator:
    """The cleared-allocator shape ``l15_retain.reserve_slots`` expects:
    ``clear()`` resets free_pages to arange(1, size+1) (slot 0 padding)."""

    def __init__(self, size=SIZE):
        self.size = size
        self.free_pages = torch.arange(1, size + 1)
        self.clears = 0

    def clear(self):
        self.clears += 1
        self.free_pages = torch.arange(1, self.size + 1)


class _Adapter:
    def __init__(self):
        self.calls = []

    def alloc_info_ok(self, base):
        return True

    def set_keep_byte_spans(self, base, spans):
        self.calls.append((base.data_ptr(), spans))


class _Sched:
    def __init__(self):
        self.tp_size = 2
        self.server_args = SimpleNamespace(tp_size=2, rank_gpu_id=None)
        # pricing pool: one (rows, heads, head_dim) f32 buffer per K and V
        cell_t = torch.zeros(4, 2, 2)
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(k_buffer=[cell_t], v_buffer=[cell_t])))
        self.req_to_token_pool = _ReqToTokenPool()
        self.token_to_kv_pool_allocator = _Allocator()
        # scrub pool set: outer wrapper (resident, no sub-pool limits) whose
        # k_buffer holds TWO views of one base and whose v_buffer one
        # independent tensor -> exactly two allocation bases for the TMS
        # walk; the held prefix lives in full_kv_pool's single k_buffer.
        self.base_a = torch.ones(6, 1)
        self.base_b = torch.ones(6, 1)
        self.kv_held = torch.ones(6, 1)
        outer = SimpleNamespace(
            backing_is_resident=True,
            k_buffer=[self.base_a[0:3], self.base_a[3:6]],
            v_buffer=[self.base_b],
            kv_buffer=None,
            full_kv_pool=SimpleNamespace(k_buffer=[self.kv_held], v_buffer=None,
                                         kv_buffer=None, safe_zero_rows=None),
            swa_kv_pool=None,
            safe_zero_rows=None)
        self._flush_pools = [outer]
        self.flush_calls = []
        self.adapter = _Adapter()
        self.memory_saver_adapter = self.adapter

    def _kv_pools_for_flush(self):
        return self._flush_pools

    def _flush_zero_kv_buffers(self):
        self.flush_calls.append(1)


def _fake_self(sched, rank):
    fs = SimpleNamespace()
    fs.scheduler = sched
    fs._pdflip_group_name = lambda: "D"
    fs._pdflip_rank = lambda: rank
    fs._l15_wake_hold_signal = lambda: WU._l15_wake_hold_signal(fs)
    fs._l15_flush_zero_kv_bounded = lambda s, k: WU._l15_flush_zero_kv_bounded(fs, s, k)
    fs._l15_clear_tms_keep_spans = lambda s: WU._l15_clear_tms_keep_spans(fs, s)
    fs.flushed = []

    def _flush_cache():
        fs.flushed.append(1)
        return False

    fs.flush_cache = _flush_cache
    return fs


def _env(monkeypatch, tmp_path, master, mib):
    monkeypatch.setenv("FLLIPER_PDFLIP_L15", "1" if master else "0")
    if mib is None:
        monkeypatch.delenv("FLLIPER_PDFLIP_L15_MIB", raising=False)
    else:
        monkeypatch.setenv("FLLIPER_PDFLIP_L15_MIB", mib)
    monkeypatch.setenv("FLLIPER_PDFLIP_L15_MANIFEST", str(tmp_path) + os.sep)
    monkeypatch.delenv("FLLIPER_FLUSH_ZERO_KV", raising=False)  # default on


def _write_manifest():
    man = l15_manifest.Manifest(
        epoch=1, pid=os.getpid(),
        spans=(l15_manifest.HoldSpan(rid="a", depth=1, slots=(3, 7), anchor_slot=1,
                                     l2_slots=(), l2_gens=()),
               l15_manifest.HoldSpan(rid="b", depth=2, slots=(7, 12), anchor_slot=1,
                                     l2_slots=(), l2_gens=())),
        rows_by_rank=(KEEP, KEEP), anchor_slots=ANCHOR)
    path = l15_manifest.manifest_path("D", 0, os.environ)
    l15_manifest.write(path, man)
    return man, path


# --------------------------------------------------------------------------
# (a) master on + manifest + cap > 0: the hold survives the restore
# --------------------------------------------------------------------------

def test_hold_path_keeps_mamba_rows_reserves_held_slots_bounds_scrub_clears_keep(
        monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=True, mib="c0=64")
    man, path = _write_manifest()
    sched = _Sched()
    fs = _fake_self(sched, 0)

    assert WU._pdflip_wake_restore_pools(fs) is True and fs.flushed == []

    # req_to_token clear ran with keep_mamba_rows == anchor_slots (mamba
    # anchors [1, keep) survive the reset).
    assert sched.req_to_token_pool.clear_calls == [((), {"keep_mamba_rows": ANCHOR})]

    # allocator cleared once, then the held manifest slots (3, 7, 12; slot 0
    # is padding and dropped by the commit) stay out of free_pages.
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
    assert not free & {3, 7, 12}
    assert free >= {1, 2, 4, 5, 6, 8}

    # KV scrub bounded: rows >= KEEP zeroed, held prefix [0, KEEP) untouched.
    assert bool((sched.kv_held[:KEEP] == 1).all())
    assert bool((sched.kv_held[KEEP:] == 0).all())

    # TMS keep set cleared exactly once per distinct allocation base
    # (base_a via two views, base_b once) AND the hybrid full_kv_pool's
    # held base (F10, 01.10.: the sub-pool buffers must be reached too),
    # each with the EMPTY span set.
    assert [s for _, s in sched.adapter.calls] == [(), (), ()]
    assert {b for b, _ in sched.adapter.calls} == {sched.base_a.data_ptr(),
                                                   sched.base_b.data_ptr(),
                                                   sched.kv_held.data_ptr()}

    # the fallback path's full scrub did NOT run, the manifest was consumed
    # (read-and-clear), and it is stashed for the fence / AP-B.
    assert sched.flush_calls == []
    assert not os.path.exists(path)
    # (the wake re-parses the JSON record, so compare content, not identity)
    assert fs._l15_wake_manifest == man


# --------------------------------------------------------------------------
# (b) master on + no manifest: today's restore, plus the keep-set cleanup
# --------------------------------------------------------------------------

def test_master_on_without_manifest_runs_the_old_sequence(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=True, mib="c0=64")
    sched = _Sched()
    fs = _fake_self(sched, 0)

    assert WU._pdflip_wake_restore_pools(fs) is True and fs.flushed == []

    # the OLD sequence: plain clear()s (no keep kwargs), full scrub hook,
    # allocator left fully free.
    assert sched.req_to_token_pool.clear_calls == [((), {})]
    assert sched.flush_calls == [1]
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
    assert free == set(range(1, SIZE + 1))
    # master on: the sleep-armed keep set is cleared on the fallback too
    # (base_a, base_b and, F10, the hybrid full_kv_pool's held base)...
    assert [s for _, s in sched.adapter.calls] == [(), (), ()]
    # ...and the (absent) manifest is stashed for the fence.
    assert fs._l15_wake_manifest is None


# --------------------------------------------------------------------------
# (c) master off: byte-identical old sequence, manifest NOT read
# --------------------------------------------------------------------------

def test_master_off_is_byte_identical_and_never_reads_the_manifest(
        monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=False, mib="c0=64")
    _, path = _write_manifest()  # a leftover record must NOT be consumed
    sched = _Sched()
    fs = _fake_self(sched, 0)

    assert WU._pdflip_wake_restore_pools(fs) is True and fs.flushed == []

    assert sched.req_to_token_pool.clear_calls == [((), {})]
    assert sched.flush_calls == [1]
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
    assert free == set(range(1, SIZE + 1))
    # master off: no keep-set touch, no manifest stashed, record untouched.
    assert sched.adapter.calls == []
    assert not hasattr(fs, "_l15_wake_manifest")
    assert os.path.exists(path)


# --------------------------------------------------------------------------
# (d) cap == 0 (MIB unset -> auto): manifest read but NO reservation
# --------------------------------------------------------------------------

def test_cap_zero_falls_back_without_reserving_the_held_slots(
        monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=True, mib=None)  # auto -> every cap 0
    _, path = _write_manifest()
    sched = _Sched()
    fs = _fake_self(sched, 0)

    assert WU._pdflip_wake_restore_pools(fs) is True and fs.flushed == []

    # the record was still consumed (read-and-clear happens before the cap
    # check), but nothing was reserved out of the freshly cleared allocator.
    assert not os.path.exists(path)
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages.tolist()}
    assert free == set(range(1, SIZE + 1))
    assert sched.req_to_token_pool.clear_calls == [((), {})]
    assert sched.flush_calls == [1]
