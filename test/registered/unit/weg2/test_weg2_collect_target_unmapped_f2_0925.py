"""F2 (boot weg2rc5gg, 2026-09-25): a collect target the driver does not know
is a NAMED refusal before any lane writes, not a SIGSEGV.

WHAT THE METAL SHOWED.  weg2rc5gg (RC5 5f13f1aad9, the first flip that carried
GGUF weights) died at its first D->P flip: P PP1 "Fatal Python error:
Segmentation fault" in ``weight_exchange_transport.memcpy_async`` <-
``bar1_lanes._run_bar1_tag_streamed`` (COLLECT, lane p0, seq 0-weights_5).
Three seconds earlier the same rank had printed ``WEG2-RESUME-PTRATTR
tag=weights_5 ... own_tag=weights_5 mapped=60 unmapped=24
first_unmapped=model.layers.42.linear_attn.in_proj_qkvz.qweight
(44564480B,type=0)``: the GGUF qweights are allocated at LOAD time in the
BASE tag, the plan files them under their layer's chunk tag by name, so the
collect of ``weights_5`` wrote into pages no resumed tag held.
``cudaMemcpyAsync(cudaMemcpyDefault)`` takes an address the driver does not
know as pageable host memory, and the CPU copy faults.

WHY THE PLAN'S DESTINATIONS AND NOT THE PTRATTR NAME CENSUS.  The census walks
``_weg2_model_for_group``'s model by NAME.  On D that is the DRAFT model, so
in every RC4 boot (INT8 / FP8 / NVFP4, 51 flips each) D printed
``tag=weights_0 ... unmapped=160 first_unmapped=layers.0.input_layernorm.weight``
(the draft's layers, allocated under ``weights_draft``) and ``tag=weights ...
unmapped=3 first_unmapped=fc.weight_g_idx(0B,type=0)`` (zero-byte tensors) on
150 of 750 readings -- while every collect wrote correctly (325/325 copy-out
probes ``dst_rc=0 dst_type=2`` per group).  A refusal on that census would
have killed every RC4 wake on D.  The collect's OWN targets are the plan's
descriptors for the tag, minus what the collect never writes.

Pure desk: no device, no shm segment, no semaphore -- the probe is injected.
"""

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
from sglang.srt.weg2 import weight_exchange_region as xr

Weg2WakeRefused = wu.Weg2WakeRefused

MAPPED = (0, 2, 0)            # rc 0, CU_MEMORYTYPE_DEVICE, device 0
UNMAPPED = (1, 0, -1)         # the weg2rc5gg reading: rc=1 type=0 device=-1
PROBE_FAILED = (-1, -1, -1)   # ptr_attrs' own "no reading" (exception path)
NO_CONTEXT = (201, 0, -1)     # CUDA_ERROR_INVALID_CONTEXT: not a reading either


class _Desc:
    """The XchgDesc fields the census reads, as an explicit class."""

    def __init__(self, name, *, tag="weights_5", dst_ptr=0x7000_0000,
                 nbytes=4096, kind="flat", dst_off=0):
        self.param_name = name
        self.tag = tag
        self.dst_ptr = dst_ptr
        self.nbytes = int(nbytes)
        self.kind = kind
        self.dst_off = int(dst_off)
        self.rows = 1
        self.run_bytes = int(nbytes)
        self.spitch = int(nbytes)
        self.dpitch = int(nbytes)
        self.src_rank = 0
        self.dst_rank = 0


def _probe(table, calls=None):
    def probe(addr):
        if calls is not None:
            calls.append(int(addr))
        return table.get(int(addr), MAPPED)
    return probe


# ===========================================================================
# THE CENSUS (weight_exchange_bounce.collect_target_census)
# ===========================================================================


def test_census_names_the_unmapped_collect_target():
    """RED on 5f13f1aad9: there was no census, the lane wrote and died."""
    q = _Desc("model.layers.42.linear_attn.in_proj_qkvz.qweight",
              dst_ptr=0x1000, nbytes=44564480)
    n = _Desc("model.layers.42.input_layernorm.weight", dst_ptr=0x2000,
              nbytes=10240)
    d = _Desc("model.layers.42.linear_attn.dt_bias", dst_ptr=0x3000, nbytes=96)
    census = bx.collect_target_census(
        [q, n, d], tag="weights_5", probe=_probe({0x1000: UNMAPPED}))
    assert census.targets == 3
    assert [u[1] for u in census.unmapped] == [
        "model.layers.42.linear_attn.in_proj_qkvz.qweight"]
    tag, name, ptr, nbytes, rc, mtype = census.unmapped[0]
    assert (tag, ptr, nbytes, rc, mtype) == ("weights_5", 0x1000, 44564480, 1, 0)
    assert census.probe_failed == 0


def test_census_skips_what_the_collect_never_writes():
    """The RC4 D shapes that the name census read as unmapped are NOT targets.

    * a zero-byte tensor (``fc.weight_g_idx(0B,type=0)``): nothing is written;
    * a desc without ``dst_ptr``: the lane refuses it by name already;
    * a NO-WRITE desc (the draft's embed IS the target's tensor on D, region
      ``weights``, still paused while ``weights_draft`` is collected -- the
      reason ``_weg2_xchg_no_write`` exists): consumed, never written.
    """
    g_idx = _Desc("fc.weight_g_idx", tag="weights_draft", dst_ptr=0x4000,
                  nbytes=0)
    nodst = _Desc("layers.0.mlp.gate_up_proj.weight", tag="weights_draft",
                  dst_ptr=None)
    embed = _Desc("model.embed_tokens.weight", tag="weights_draft",
                  dst_ptr=0x5000, nbytes=1 << 20)
    body = _Desc("layers.0.self_attn.qkv_proj.weight_packed",
                 tag="weights_draft", dst_ptr=0x6000)
    census = bx.collect_target_census(
        [g_idx, nodst, embed, body], tag="weights_draft",
        no_write=frozenset({("weights_draft", "model.embed_tokens.weight")}),
        probe=_probe({0x4000: UNMAPPED, 0x5000: UNMAPPED}))
    assert census.unmapped == ()
    assert census.targets == 1
    assert census.no_write == 1


def test_a_bare_name_in_no_write_is_honoured_like_the_bar1_lane():
    """bar1_lanes skips ``(tag, name) in _nw or name in _nw``: the census must
    not be narrower than the writer it guards."""
    embed = _Desc("model.embed_tokens.weight", tag="weights_draft",
                  dst_ptr=0x5000)
    census = bx.collect_target_census(
        [embed], tag="weights_draft",
        no_write=frozenset({"model.embed_tokens.weight"}),
        probe=_probe({0x5000: UNMAPPED}))
    assert census.unmapped == () and census.no_write == 1


def test_a_probe_that_cannot_read_never_refuses():
    """(-1,-1,-1) and a no-context rc are not "the driver does not know this
    address" -- a diagnostic that cannot read must not kill a wake."""
    a = _Desc("model.layers.40.mlp.down_proj.weight", dst_ptr=0x1000)
    b = _Desc("model.layers.41.mlp.down_proj.weight", dst_ptr=0x2000)
    census = bx.collect_target_census(
        [a, b], tag="weights_5",
        probe=_probe({0x1000: PROBE_FAILED, 0x2000: NO_CONTEXT}))
    assert census.unmapped == ()
    assert census.probe_failed == 2


def test_each_allocation_is_probed_once():
    """A K-cut tensor arrives as many descriptors on one destination; one
    reading per allocation keeps the census off the flip's clock."""
    calls = []
    descs = [_Desc("model.layers.40.mlp.down_proj.qweight", dst_ptr=0x9000,
                   dst_off=i * 4096) for i in range(5)]
    census = bx.collect_target_census(descs, tag="weights_5",
                                      probe=_probe({}, calls))
    assert calls == [0x9000]
    assert census.targets == 1 and census.unmapped == ()


def test_the_refusal_text_names_tag_tensor_and_reading():
    q = _Desc("model.layers.42.linear_attn.in_proj_qkvz.qweight",
              dst_ptr=0x1000, nbytes=44564480)
    census = bx.collect_target_census(
        [q], tag="weights_5", probe=_probe({0x1000: UNMAPPED}))
    text = census.refusal(group="P", rank=1)
    assert text.startswith("W4 Weg2WakeRefused:")
    assert "tag=weights_5" in text
    assert "model.layers.42.linear_attn.in_proj_qkvz.qweight" in text
    assert "rc=1 type=0" in text
    assert "group=P rank=1" in text


# ===========================================================================
# THE WIRING (_weg2_xchg_inject_from_peer): refuse BEFORE the lanes run
# ===========================================================================


class _FakeServerArgs:
    def __init__(self):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"


class _FakeRunner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


class _FakePlan:
    def __init__(self, descs):
        self.descs = tuple(descs)
        self.raw_descs = tuple(descs)
        self.waves = (("weights_5",),)
        self.byte_matrix = ((0,),)
        self.plan_id = "0xf2f2f2f2"
        self.src_group = "D"
        self.dst_group = "P"
        self.tag_bytes = (("weights_5", sum(d.nbytes for d in descs)),)
        self.skipped_tags = ()


class _FakeTerms:
    total_bytes = 402653184

    def expression(self):
        return "(depth+1) x slot = 3 x 134217728"


class _Ops:
    """Records the device bind the census asks for; no CUDA behind it."""

    def __init__(self):
        self.devices = []

    def set_device(self, device):
        self.devices.append(int(device))


@pytest.fixture()
def armed(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1790313773")
    monkeypatch.delenv(xr.ENV_REGION_PATH, raising=False)


def _manager(monkeypatch, descs, *, ops, calls, no_write=None):
    cls = wu.SchedulerWeightUpdaterManager
    monkeypatch.setattr(cls, "_weg2_server_args", lambda self: _FakeServerArgs())
    monkeypatch.setattr(cls, "_weg2_group_name", lambda self: "P")
    monkeypatch.setattr(cls, "_weg2_rank", lambda self: 1)
    monkeypatch.setattr(cls, "_weg2_device_index", lambda self: 0)
    monkeypatch.setattr(
        cls, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan(descs), ""))
    monkeypatch.setattr(cls, "_weg2_xchg_device_ops", lambda self: ops)
    # no region, no semaphore set: nothing under /dev/shm is opened
    monkeypatch.setattr(cls, "_weg2_shadow_region", lambda self: None)
    monkeypatch.setattr(cls, "_weg2_xchg_sems", lambda self: None)

    def _record_bounce(self, **kw):
        calls.append(kw)
        return None

    monkeypatch.setattr(cls, "_weg2_xchg_bounce_leg", _record_bounce)
    m = cls(tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
            memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
            is_fully_idle=lambda *a, **k: True)
    if no_write is not None:
        m._weg2_xchg_no_write = no_write
    return m


def test_the_collect_refuses_by_name_before_any_lane_runs(monkeypatch, armed):
    """RED on 5f13f1aad9: the bounce leg ran and the lane wrote into the hole."""
    descs = [
        _Desc("model.layers.42.linear_attn.in_proj_qkvz.qweight",
              dst_ptr=0x1000, nbytes=44564480),
        _Desc("model.layers.42.input_layernorm.weight", dst_ptr=0x2000,
              nbytes=10240),
    ]
    monkeypatch.setattr(bx, "ptr_attrs", _probe({0x1000: UNMAPPED}))
    calls, ops = [], _Ops()
    m = _manager(monkeypatch, descs, ops=ops, calls=calls)
    with pytest.raises(Weg2WakeRefused) as ei:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_5")
    text = str(ei.value)
    assert "W4 Weg2WakeRefused" in text and "tag=weights_5" in text
    assert "model.layers.42.linear_attn.in_proj_qkvz.qweight" in text
    assert calls == [], "a lane ran after an unmapped target was read"
    assert ops.devices == [0], "the probe ran without binding the rank's device"


def test_a_mapped_collect_runs_unchanged(monkeypatch, armed):
    """INT8 / FP8 / NVFP4 on RC4: every target mapped -> the leg is called with
    the same descriptors, and nothing is refused.  The RC4 D shapes that the
    name census misread (a no-write embed and a zero-byte g_idx, both reading
    type 0) ride along and change nothing."""
    descs = [
        _Desc("model.layers.40.mlp.gate_up_proj.weight_packed", dst_ptr=0x1000),
        _Desc("model.layers.40.mlp.down_proj.weight_packed", dst_ptr=0x2000),
        _Desc("fc.weight_g_idx", dst_ptr=0x3000, nbytes=0),
        _Desc("model.embed_tokens.weight", dst_ptr=0x4000),
    ]
    monkeypatch.setattr(bx, "ptr_attrs",
                        _probe({0x3000: UNMAPPED, 0x4000: UNMAPPED}))
    calls, ops = [], _Ops()
    m = _manager(monkeypatch, descs, ops=ops, calls=calls,
                 no_write=frozenset({("weights_5", "model.embed_tokens.weight")}))
    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_5")
    assert len(calls) == 1
    assert list(calls[0]["descs"]) == descs
    assert calls[0]["tag"] == "weights_5"
    assert calls[0]["mode"] == wx.INJECT_AUTHORITATIVE


def test_an_empty_tag_slice_probes_nothing(monkeypatch, armed):
    """A rank that holds none of the tag's layers (``pieces=0``) keeps its
    lockstep no-op: no probe, no device bind, the leg still runs."""
    descs = [_Desc("model.layers.0.mlp.down_proj.weight", tag="weights_0",
                   dst_ptr=0x1000)]
    seen = []
    monkeypatch.setattr(bx, "ptr_attrs", _probe({}, seen))
    calls, ops = [], _Ops()
    m = _manager(monkeypatch, descs, ops=ops, calls=calls)
    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_5")
    assert seen == [] and ops.devices == []
    assert len(calls) == 1 and list(calls[0]["descs"]) == []
