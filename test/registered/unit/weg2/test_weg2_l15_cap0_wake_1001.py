"""AP L15-12c-E2 (1001): the cap-0 rank (TP0) at the wake -- anchor gate,
stash + refill mark, and the verdict-"hold" refill into the fallback.

The cap-0 rank kept nothing mapped through the sleep, so every held row it
owns must come back over H2D from L2 (plan L15-12-PART3-PLAN sec 2):

  * ANCHOR GATE (plan sec 8): while any HoldSpan lacks the GDN anchor's L2
    identity (``anchor_l2_slot`` absent/None -- true for every manifest
    written before C2 records it), a cap-0 wake logs
    ``L15-REFILL rank=%d anchors-missing: votes no hold`` and votes None
    exactly like before E2: no stash, no refill mark, pools untouched.

  * Gate open (spans carry anchor identity): the hold signal stashes the
    manifest and marks the rank a refill rank (``_l15_wake_refill``); the
    hold-aware restore re-reserves the held slots; the fence tail's act on
    verdict "hold" runs ONE generation-checked refill with rid-tagged rows.

  * A generation mismatch (or any refill failure) leaves no half state:
    the refill folds into ``_l15_fallback_drop`` and logs
    ``L15-REFILL rank=%d failed: ... -> fallback``.

  * cap > 0 (AP-A) and master off stay byte-identical.

Hermetic: methods run UNBOUND on SimpleNamespace fakes; CPU tensors only,
no CUDA, no model.  Run:
``python -m pytest test/registered/unit/weg2/test_weg2_l15_cap0_wake_1001.py -q``
"""
import logging
import os
import pathlib
import sys
from types import SimpleNamespace

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import l15_manifest, l15_restore  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager
SIZE = 16
KEEP = 2
ANCHOR = 5
LOGGER = "sglang.srt.managers.scheduler_components.weight_updater"


# ---------------------------------------------------------------------------
# fakes (style: test_weg2_l15_wake_restore_1001.py)
# ---------------------------------------------------------------------------

class _ReqToTokenPool:
    def __init__(self):
        self.clear_calls = []

    def clear(self, *a, **k):
        self.clear_calls.append((a, k))


class _Allocator:
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


class _HostPool:
    """Arena host pool stand-in: current generations + recorded bulk loads."""

    def __init__(self, gens=None):
        self.gens = gens or {}
        self.gen_calls = []
        self.load_calls = []
        self._arena_page_tokens = 1

    def slot_gens(self, slots):
        self.gen_calls.append(tuple(slots))
        return [int(self.gens.get(int(s), -1)) for s in slots]

    def _load_pages_all_layers(self, device_pool, slots_t, didx_t,
                               lanes=None, mode=None):
        self.load_calls.append((tuple(int(x) for x in slots_t),
                                tuple(int(x) for x in didx_t), lanes, mode))


class _TreeCache:
    def __init__(self, host_pool):
        self.resets = 0
        self.cache_controller = SimpleNamespace(mem_pool_host=host_pool)

    def reset(self):
        self.resets += 1


class _Sched:
    def __init__(self, host_pool):
        self.tp_size = 2
        self.server_args = SimpleNamespace(tp_size=2, rank_gpu_id=None)
        cell_t = torch.zeros(4, 2, 2)
        self.device_pool = SimpleNamespace(
            k_buffer=[cell_t], v_buffer=[cell_t])
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(token_to_kv_pool=self.device_pool))
        self.req_to_token_pool = _ReqToTokenPool()
        self.token_to_kv_pool_allocator = _Allocator()
        self.tree_cache = _TreeCache(host_pool)
        self.base_a = torch.ones(6, 1)
        self.base_b = torch.ones(6, 1)
        outer = SimpleNamespace(
            backing_is_resident=True,
            k_buffer=[self.base_a[0:3], self.base_a[3:6]],
            v_buffer=[self.base_b], kv_buffer=None,
            full_kv_pool=None, swa_kv_pool=None, safe_zero_rows=None)
        self._flush_pools = [outer]
        self.memory_saver_adapter = _Adapter()

    def _kv_pools_for_flush(self):
        return self._flush_pools

    def _flush_zero_kv_buffers(self):
        pass


def _span(rid, slots, l2_slots, l2_gens, anchor=9):
    return SimpleNamespace(rid=rid, slots=tuple(slots), anchor_slot=1,
                           l2_slots=tuple(l2_slots), l2_gens=tuple(l2_gens),
                           anchor_l2_slot=anchor, anchor_l2_gen=5)


def _fake_manifest(spans):
    return SimpleNamespace(epoch=1, pid=os.getpid(), spans=tuple(spans),
                           rows_by_rank=(KEEP, KEEP), anchor_slots=ANCHOR)


def _fake_self(sched, rank):
    fs = SimpleNamespace()
    fs.scheduler = sched
    fs._l15_wake_manifest = None
    fs._l15_wake_refill = False
    fs._weg2_group_name = lambda: "D"
    fs._weg2_rank = lambda: rank
    fs._l15_wake_hold_signal = lambda: WU._l15_wake_hold_signal(fs)
    fs._l15_flush_zero_kv_bounded = lambda s, k: WU._l15_flush_zero_kv_bounded(fs, s, k)
    fs._l15_clear_tms_keep_spans = lambda s: WU._l15_clear_tms_keep_spans(fs, s)
    fs._l15_fallback_drop = lambda s: WU._l15_fallback_drop(fs, s)
    fs._l15_do_refill = lambda s: WU._l15_do_refill(fs, s)
    fs._l15_wake_act = lambda s, v, **kw: WU._l15_wake_act(fs, s, v, **kw)
    fs.flushed = []
    fs.flush_cache = lambda: fs.flushed.append(1) or False
    return fs


def _env(monkeypatch, tmp_path, master, mib):
    monkeypatch.setenv("SGLANG_WEG2_L15", "1" if master else "0")
    if mib is None:
        monkeypatch.delenv("SGLANG_WEG2_L15_MIB", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_L15_MIB", mib)
    monkeypatch.setenv("SGLANG_WEG2_L15_MANIFEST", str(tmp_path) + os.sep)
    monkeypatch.delenv("SGLANG_FLUSH_ZERO_KV", raising=False)  # default on


# (rank 0, tp 2: even slots are rank 0's; compact row = slot // 2)
_SPANS = [_span("a", (2, 4), (10, 11), (5, 5))]


def _gate_open(monkeypatch, man):
    monkeypatch.setattr(l15_restore, "load_for_wake", lambda path: man)


# ---------------------------------------------------------------------------
# (1) anchor gate CLOSED (real manifest, no anchor identity): vote None
# ---------------------------------------------------------------------------

def test_cap0_anchor_missing_logs_gate_and_votes_none(monkeypatch, tmp_path, caplog):
    _env(monkeypatch, tmp_path, master=True, mib="c1=64")  # rank 0 cap 0
    man = l15_manifest.Manifest(
        epoch=1, pid=os.getpid(),
        spans=(l15_manifest.HoldSpan(rid="a", depth=2, slots=(2, 4), anchor_slot=1,
                                     l2_slots=(10, 11), l2_gens=(5, 5)),),
        rows_by_rank=(KEEP, KEEP), anchor_slots=ANCHOR)
    l15_manifest.write(l15_manifest.manifest_path("D", 0, os.environ), man)
    sched = _Sched(_HostPool({10: 5, 11: 5}))
    fs = _fake_self(sched, 0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        m, rank, keep, master_on = WU._l15_wake_hold_signal(fs)
    assert (m, rank, keep, master_on) == (None, 0, 0, True)
    assert fs._l15_wake_manifest is None            # not stashed
    assert not fs._l15_wake_refill                   # not a refill rank
    assert sched.token_to_kv_pool_allocator.clears == 0  # pools untouched
    assert any("anchors-missing" in r.getMessage() and "rank=0" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# (2) gate OPEN: stash + refill mark; restore reserves; verdict "hold" refills
# ---------------------------------------------------------------------------

def test_cap0_gate_open_stashes_marks_and_refills_once(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=True, mib="c1=64")
    man = _fake_manifest(_SPANS)
    _gate_open(monkeypatch, man)
    host = _HostPool({10: 5, 11: 5})
    sched = _Sched(host)
    fs = _fake_self(sched, 0)
    m, rank, keep, master_on = WU._l15_wake_hold_signal(fs)
    assert (m, rank, keep, master_on) == (man, 0, KEEP, True)
    assert fs._l15_wake_refill is True
    # the hold-aware restore reserves the held slots (rows unallocatable now)
    assert WU._weg2_wake_restore_pools(fs) is True
    assert fs._l15_wake_manifest is man
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages}
    assert not ({2, 4} & free), sorted(free)
    assert sched.req_to_token_pool.clear_calls == [((), {"keep_mamba_rows": ANCHOR})]
    # the fence tail's act on "hold": ONE gen-checked refill, rid-tagged rows
    n = WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True)
    assert n == 2, n
    assert host.gen_calls == [(10, 11)]
    assert len(host.load_calls) == 1
    slots, didx, lanes, mode = host.load_calls[0]
    assert slots == (10, 11) and didx == (1, 2)      # compact rows slot//2
    assert lanes is None and mode is None
    assert sched.tree_cache.resets == 0             # no fallback on success
    assert fs._l15_wake_manifest is man             # the hold survives the act


# ---------------------------------------------------------------------------
# (3) generation mismatch: no partial copy, folds into the fallback drop
# ---------------------------------------------------------------------------

def test_cap0_gen_mismatch_folds_into_fallback(monkeypatch, tmp_path, caplog):
    _env(monkeypatch, tmp_path, master=True, mib="c1=64")
    man = _fake_manifest(_SPANS)
    _gate_open(monkeypatch, man)
    host = _HostPool({10: 7, 11: 5})                # slot 10 was re-claimed
    sched = _Sched(host)
    fs = _fake_self(sched, 0)
    WU._l15_wake_hold_signal(fs)
    WU._weg2_wake_restore_pools(fs)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        n = WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True)
    assert n == 0
    assert host.load_calls == []                    # nothing half-copied
    assert sched.tree_cache.resets == 1             # fallback drop ran
    assert sched.req_to_token_pool.clear_calls[-1] == ((), {})   # full clear
    free = {int(x) for x in sched.token_to_kv_pool_allocator.free_pages}
    assert free == set(range(1, SIZE + 1))          # nothing half-reserved
    assert fs._l15_wake_manifest is None            # hold disarmed
    assert any("failed" in r.getMessage() and "fallback" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# (4) invariants: master off and cap > 0 unchanged (AP-A stays green)
# ---------------------------------------------------------------------------

def test_master_off_votes_none_and_marks_nothing(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, master=False, mib="c1=64")
    sched = _Sched(_HostPool())
    fs = _fake_self(sched, 0)
    assert WU._l15_wake_hold_signal(fs) == (None, None, 0, False)
    assert not fs._l15_wake_refill


def test_cap_positive_unchanged_no_refill_mark(monkeypatch, tmp_path, caplog):
    _env(monkeypatch, tmp_path, master=True, mib="c0=64")   # rank 0 cap > 0
    man = _fake_manifest(_SPANS)
    _gate_open(monkeypatch, man)
    sched = _Sched(_HostPool({10: 5, 11: 5}))
    fs = _fake_self(sched, 0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        m, rank, keep, master_on = WU._l15_wake_hold_signal(fs)
    assert (m, rank, keep, master_on) == (man, 0, KEEP, True)
    assert not fs._l15_wake_refill                  # AP-A path, not a refill rank
    assert not any("anchors-missing" in r.getMessage() for r in caplog.records)
    # verdict "hold" on a cap>0 rank copies nothing
    WU._weg2_wake_restore_pools(fs)
    assert WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True) == 0
    assert sched.tree_cache.cache_controller.mem_pool_host.load_calls == []
