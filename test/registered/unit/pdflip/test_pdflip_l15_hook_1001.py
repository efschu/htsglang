# SPDX-License-Identifier: Apache-2.0
"""L15-11c (AP D2, 01.10.): the TMS keep-span adapter bindings.

``set_keep_spans(tensor, row_ranges)`` converts row ranges to byte ranges
relative to the tensor's own data_ptr (``row * stride(0) * element_size()``);
``set_keep_byte_spans(base, byte_ranges)`` takes ABSOLUTE byte ranges of one
allocation BASE (tms_set_keep_spans REPLACES the allocation's keep set, so
views of a base must be aggregated into one call per base); ``alloc_info_ok``
asks tms_alloc_info whether a pointer is a tracked allocation base.  Missing
symbol: -100 / False; no-op adapter: -100 / False.
"""
import torch

from flliper.srt.utils import torch_memory_saver_adapter as tmsmod


class _RecordingSymbol:
    """Stands in for the C tms_set_keep_spans; records the ctypes args."""

    def __init__(self, ret=7):
        self.calls = []
        self.ret = ret

    def __call__(self, ptr, n, lo, hi):
        self.calls.append(
            (
                ptr.value,
                n,
                [lo[i] for i in range(n)],
                [hi[i] for i in range(n)],
            )
        )
        return self.ret


def _patch_symbol(monkeypatch, fn):
    monkeypatch.setattr(
        tmsmod,
        "_pdflip_ring_symbol",
        lambda name: fn if name == "tms_set_keep_spans" else None,
    )


def test_set_keep_spans_rows_to_bytes(monkeypatch):
    rec = _RecordingSymbol()
    _patch_symbol(monkeypatch, rec)
    # (10, 4) float32: one row = stride(0) * element_size = 4 * 4 = 16 bytes.
    t = torch.zeros(10, 4, dtype=torch.float32)
    rc = tmsmod._TorchMemorySaverAdapterReal().set_keep_spans(t, [(0, 3)])
    assert rc == 7, "the adapter returns the C code"
    assert len(rec.calls) == 1
    ptr, n, lo, hi = rec.calls[0]
    assert ptr == t.data_ptr(), "the allocation base is the tensor data_ptr"
    assert n == 1
    assert lo == [0] and hi == [48], "rows (0, 3) -> bytes [0, 48)"
    # Unsorted multi-range input is normalised and scaled the same way.
    tmsmod._TorchMemorySaverAdapterReal().set_keep_spans(t, [(2, 5), (0, 1)])
    _, n2, lo2, hi2 = rec.calls[1]
    assert n2 == 2
    assert lo2 == [0, 32] and hi2 == [16, 80], "sorted, row*16 bytes"


def test_set_keep_byte_spans_aggregates_views_of_one_base(monkeypatch):
    # Per-layer views of one (3, 8, 4) f32 allocation (layer stride0 = 128 B,
    # slot stride = 32 B), keeping rows [0, 2): the hook's collector must
    # land ONE byte-span call on the base carrying 3 ranges
    # (l*stride0*esz, l*stride0*esz + 2*slot_stride*esz).
    rec = _RecordingSymbol()
    _patch_symbol(monkeypatch, rec)
    t = torch.zeros(3, 8, 4, dtype=torch.float32)
    ranges = []
    for _l in range(3):
        v = t[_l]
        base = v._base if v._base is not None else v
        assert base is t, "a layer slice is a view of the one allocation"
        off = v.data_ptr() - base.data_ptr()
        unit = int(v.stride(0)) * int(v.element_size())
        for _lo, _hi in [(0, 2)]:
            ranges.append((off + _lo * unit, off + _hi * unit))
    rc = tmsmod._TorchMemorySaverAdapterReal().set_keep_byte_spans(t, ranges)
    assert rc == 7
    assert len(rec.calls) == 1, "one call per allocation BASE, not per view"
    ptr, n, lo, hi = rec.calls[0]
    assert ptr == t.data_ptr()
    assert n == 3
    # 2 kept rows per layer = 2 * stride1 * esz = 2 * 4 * 4 = 32 bytes.
    assert lo == [0, 128, 256] and hi == [32, 160, 288]


def test_set_keep_byte_spans_missing_symbol_is_minus_100(monkeypatch):
    monkeypatch.setattr(tmsmod, "_pdflip_ring_symbol", lambda name: None)
    t = torch.zeros(4, 2, dtype=torch.float32)
    ad = tmsmod._TorchMemorySaverAdapterReal()
    assert ad.set_keep_byte_spans(t, [(0, 8)]) == -100
    assert ad.set_keep_spans(t, [(0, 2)]) == -100


def test_noop_adapter_keep_methods():
    noop = tmsmod.TorchMemorySaverAdapter.create(enable=False)
    t = torch.zeros(4, 2, dtype=torch.float32)
    assert noop.set_keep_spans(t, [(0, 2)]) == -100
    assert noop.set_keep_byte_spans(t, [(0, 2)]) == -100


def test_alloc_info_ok_reports_base_and_missing(monkeypatch):
    state = {"calls": [], "ret": 0}

    def fake(name):
        if name != "tms_alloc_info":
            return None

        def fn(ptr, size, mapped, planned, active):
            state["calls"].append(ptr.value)
            return state["ret"]

        return fn

    monkeypatch.setattr(tmsmod, "_pdflip_ring_symbol", fake)
    t = torch.zeros(4, 2, dtype=torch.float32)
    ad = tmsmod._TorchMemorySaverAdapterReal()
    assert ad.alloc_info_ok(t) is True, "C code 0 means a tracked base"
    assert state["calls"] == [t.data_ptr()]
    state["ret"] = -1
    assert ad.alloc_info_ok(t) is False, "C code -1 means not a base"
    monkeypatch.setattr(tmsmod, "_pdflip_ring_symbol", lambda name: None)
    assert ad.alloc_info_ok(t) is False, "missing symbol -> False"


def test_noop_adapter_alloc_info_ok_is_false():
    noop = tmsmod.TorchMemorySaverAdapter.create(enable=False)
    assert noop.alloc_info_ok(torch.zeros(2, 2)) is False


def _scheduler_flush_block():
    """Text of the D sleep flush, from the retain result branch down."""
    import pathlib

    src = (
        pathlib.Path(tmsmod.__file__).resolve().parent.parent
        / "managers"
        / "scheduler.py"
    ).read_text()
    start = src.index("if _l15_res is not None:")
    end = src.index("self.grammar_manager.clear()", start)
    return src, src[start:end]


def test_flush_retains_only_behind_master_and_in_result_branch():
    src, block = _scheduler_flush_block()
    # The retain round is gated on the master switch before it is called.
    call = "l15_retain.retain_at_sleep"
    assert "l15_plan.master_on(os.environ)" in src
    assert src.index("l15_plan.master_on(os.environ)") < src.index(call)
    # The fallback catch covers the SETUP only; the call sits after it.
    assert src.index("L15-RETAIN failed before the move") < src.index(call)
    # Per-base aggregation: the collector sits before the call, the real
    # keep calls (L15-12c-F2: l15_keep_arm.arm_keep_spans, one adapter
    # call per allocation base inside the helper) go out AFTER it, and
    # the mamba views come from the pool's mamba_cache.
    assert src.index("_keep_by_base") < src.index(call), "collector pre-move"
    assert src.index("l15_keep_arm.arm_keep_spans") > src.index(call), (
        "keep spans are applied per base only after a retain result"
    )
    assert '"mamba_cache"' in src, "mamba views sourced from the pool cache"
    # L15-11c: no mamba state -> the retain round refuses to run (log +
    # today's flush) BEFORE the retain_at_sleep call, so it can never silently
    # rewrite node.anchor_slot while zero mamba bytes move.
    assert "reason=no-mamba-state" in src
    assert src.index("reason=no-mamba-state") < src.index(call)
    # The shadow pricing uses the same span as the retain round: seqlen - 1
    # (the last token has no KV yet) at both the seat and parked sites.
    assert src.count("max(_tok - 1, 0)") == 2, "shadow span at seat + parked"
    head, _, tail = block.partition("else:")
    assert "keep_mamba_rows=_l15_res.a_h" in head, "retained branch keeps rows"
    for _c in (
        "self.tree_cache.reset()",
        "self.req_to_token_pool.clear()",
        "self.token_to_kv_pool_allocator.clear()",
        "self._flush_zero_kv_buffers()",
    ):
        assert _c not in head, f"{_c} must not run when the hold is kept"
        assert _c in tail, f"{_c} must stay in the master-off flush"


def test_retain_set_keep_names_a_defined_function():
    """Lead review 01.10. ~11:10Z: the hook passed set_keep=_set_keep_strict after
    that helper was renamed _set_keep_collect -- a NameError inside the setup
    try, logged as 'failed before the move', so the retain never ran. Every
    set_keep= target in the flush block must be defined in that block."""
    import re

    src, _ = _scheduler_flush_block()
    start = src.index("_l15_kwargs = None")
    end = src.index("l15_retain.retain_at_sleep", start)
    hook = src[start:end]
    targets = re.findall(r"set_keep=(_[A-Za-z0-9_]+)", hook)
    assert targets, "the retain hook must pass a set_keep callable"
    for name in targets:
        assert f"def {name}(" in hook, f"set_keep={name} is not defined in the retain hook"


def test_retain_hook_reads_the_kv_buffers_through_the_hybrid_wrapper():
    """N1 (dkr27browauthoritybar1fs10011036): on HybridLinearKVPool the hook's
    _kv came out EMPTY (k_buffer lives on .full_kv_pool) -- retain would have
    compacted node slots while no KV byte moved. RED on 5ab4b12b28 (no helper;
    the inline read of the wrapper found nothing)."""
    import inspect
    from types import SimpleNamespace

    from flliper.srt.managers import scheduler as S
    from flliper.srt.pdflip import l15_shadow

    inner = SimpleNamespace(k_buffer=[torch.zeros(4, 2), torch.zeros(4, 2)],
                            v_buffer=[torch.zeros(4, 2), torch.zeros(4, 2)])
    hybrid = SimpleNamespace(full_kv_pool=inner)
    kv = l15_shadow.kv_buffers_of(hybrid)
    assert len(kv) == 4 and kv[0] is inner.k_buffer[0] and kv[2] is inner.v_buffer[0]
    assert l15_shadow.kv_pool_of(hybrid) is inner and l15_shadow.kv_pool_of(inner) is inner
    assert l15_shadow.kv_buffers_of(SimpleNamespace()) == []
    src = inspect.getsource(S)
    assert "l15_shadow.kv_buffers_of(_pool)" in src
    assert "L15-RETAIN skipped reason=no-kv-buffers" in src
    assert "if _base_ok and _mb and _kv:" in src
