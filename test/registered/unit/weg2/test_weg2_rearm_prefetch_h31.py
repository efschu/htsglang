"""fnFL2 H31 (SGLANG_WEG2_REARM_PREFETCH): the Platztausch pad+extra rows load DURING the legs.

x141 (b07a9087cb, D log): ``WEG2-RESUME expert-rearm layers=48
rows_from_store=232 ms=46`` on TP1, serially after the legs and directly before
the group fence -- the rank had waited for the last 5090 lane (weights_8)
before it, so every extra D row (H30: FR_D 0.51/0.48 -> TP1 ~1740 rows) lands
on the P->D flip's critical path. The rows are fixed since the load (the map's
``_moe_offload_refill_runs``), their buffer exists as soon as the chunk tag of
its layer is resumed, and the store is pinned host memory: H31 issues each
tag's rows on a side stream right behind ``resume(tag)`` and the rearm only
joins.

Pinned here in the x141 form (3 ranks, 48 MoE layers, 3 layers per chunk tag,
extras on the 29 layers of P stage 0 for TP1/TP2, none on TP0):
  * the prefetch starts before ``leg_collects`` closes and names it as overlap;
  * the join waits only the rest (fake stream clock);
  * the bytes equal the serial path's (digest), the row count is x141's 232;
  * switch off = the serial path, byte for byte;
  * the mutant "prefetch discarded" (layers marked loaded, copies dropped)
    changes the digest -- the equality check is not vacuous;
  * layers whose tag this wake does not resume, or whose buffer the driver
    does not know, stay on the serial path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)

N_LAYERS = 48
P_STAGE0_LAYERS = 29  # P holds 134 experts/layer there, D more -> D-EXTRA
PREFIX, PAD_ROWS, EXTRA_ROWS, SCRATCH = 6, 1, 2, 3
ATTRS = ("w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale")
STORE_SLOTS = 64

#: the wake tag order of x141 (WEG2-CHUNK-BYTES wake tags=...)
X141_WAKE_ORDER = [
    "weights_13", "weights_9", "weights_0", "weights_14", "weights_10", "weights_1",
    "weights_15", "weights_11", "weights_2", "weights_draft", "weights_12", "weights_3",
    "weights_4", "weights_5", "weights_6", "weights_7", "weights_8", "weights",
]


@pytest.fixture(autouse=True)
def _chunk_geometry(monkeypatch):
    # the boot's geometry: 16 chunk tags of 3 layers, no expert bands
    monkeypatch.setenv("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", "3")
    monkeypatch.setenv("SGLANG_WEG2_WEIGHT_CHUNKS", "16")
    monkeypatch.delenv("SGLANG_WEG2_EXPERT_BAND_SIZE", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_EXPERT_BANDS", raising=False)


def _row_shape(attr):
    return (4, 8) if attr.endswith("packed") else (4,)


def _dtype(attr):
    return torch.int32 if attr.endswith("packed") else torch.float16


class _Experts(torch.nn.Module):
    def __init__(self, layer_id, rank, with_extra):
        super().__init__()
        self.layer_id = layer_id
        g = torch.Generator().manual_seed(1000 * rank + layer_id)
        rows = PREFIX + PAD_ROWS + EXTRA_ROWS + SCRATCH
        presplit = {}
        for attr in ATTRS:
            shape = _row_shape(attr)
            if _dtype(attr) is torch.int32:
                spill = torch.randint(-2**30, 2**30, (STORE_SLOTS, *shape), generator=g,
                                      dtype=torch.int32)
                buf = torch.full((rows, *shape), 7, dtype=torch.int32)  # P's residue
                buf[:PREFIX] = torch.randint(0, 99, (PREFIX, *shape), generator=g,
                                             dtype=torch.int32)  # what the exchange wrote
            else:
                spill = torch.randn((STORE_SLOTS, *shape), generator=g).to(torch.float16)
                buf = torch.full((rows, *shape), 7.0, dtype=torch.float16)
                buf[:PREFIX] = torch.randn((PREFIX, *shape), generator=g).to(torch.float16)
            presplit[attr] = (buf, spill)
        self._moe_offload_presplit = presplit
        if with_extra:
            p0 = (layer_id * 3) % (STORE_SLOTS - EXTRA_ROWS)
            refill = [(PREFIX, -1)] + [(PREFIX + PAD_ROWS + i, p0 + i) for i in range(EXTRA_ROWS)]
            self._moe_offload_refill_runs = tuple(eo._refill_runs(refill))
        else:
            self._moe_offload_refill_runs = ()


class _Layer(torch.nn.Module):
    def __init__(self, layer_id, rank, with_extra):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = _Experts(layer_id, rank, with_extra)


def _x141_model(rank):
    """TP0 (5090): no extra rows; TP1/TP2: extras on the 29 layers of P stage 0."""
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList(
        [_Layer(i, rank, rank in (1, 2) and i < P_STAGE0_LAYERS) for i in range(N_LAYERS)])
    return root


def _digest(model):
    h = hashlib.sha256()
    for name, mod in model.named_modules():
        for attr, (buf, _spill) in sorted(getattr(mod, "_moe_offload_presplit", {}).items()):
            h.update(name.encode())
            h.update(attr.encode())
            h.update(buf.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _serial(rank):
    m = _x141_model(rank)
    layers, rows = eo.rearm_expert_offload_after_wake(m)
    return m, layers, rows


def _prefetched(rank, order=X141_WAKE_ORDER, **kw):
    m = _x141_model(rank)
    phases = [("pre_leg", 1.0)]
    pf = eo.ExpertRearmPrefetch([(m, GPU_MEMORY_TYPE_WEIGHTS)], phases=phases, **kw)
    for tag in order:
        pf.issue(tag)
    phases.append(("leg_collects", 1929.0))
    j = pf.join(phases)
    layers, rows = eo.rearm_expert_offload_after_wake(m, prefetch=pf)
    return m, pf, j, layers, rows


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_x141_bytes_equal_the_serial_rearm(rank):
    ms, _ls, rows_serial = _serial(rank)
    mp, pf, j, _lp, rows_left = _prefetched(rank)
    assert _digest(mp) == _digest(ms)
    expected = 232 if rank in (1, 2) else 0  # x141: rows_from_store=232/232/0
    assert rows_serial == expected
    assert j.rows == expected and rows_left == 0
    assert j.rows + rows_left == rows_serial  # the log's rows_from_store stays the same number
    assert j.left_serial == 0 and j.skipped_unmapped == 0
    if expected:
        assert j.layers == P_STAGE0_LAYERS
        # layers 0..28 live in weights_0..weights_9 -- exactly those tags issue
        assert set(pf.tags_issued) == {f"weights_{k}" for k in range(10)}


def test_prefetch_starts_before_leg_collects_and_names_it():
    _m, pf, j, _l, _r = _prefetched(1)
    assert pf.t_first is not None
    assert j.overlap == "leg_collects"
    fields = j.fields()
    assert "prefetched=232 rows" in fields and "overlap=leg_collects" in fields
    assert "wait_ms=" in fields


def test_wake_source_issues_behind_resume_and_joins_before_the_rearm():
    """The order in resume_memory_occupation: plan -> resume(tag) -> issue ->
    ... -> leg_collects -> join -> rearm(prefetch)."""
    src = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)
    marks = [
        "_rearm_pf = self._weg2_rearm_prefetch_begin(_weg2_ph_l)",
        "weg2_tms_resume(self.memory_saver_adapter, tag)",
        "_rearm_pf = self._weg2_rearm_prefetch_issue(_rearm_pf, tag)",
        "self._weg2_wake_collect_one, tag)))",
        '_weg2_ph("leg_collects")',
        "_pf_join = _rearm_pf.join(_weg2_ph_l)",
        "rearm_expert_offload_after_wake(_m, prefetch=_rearm_pf)",
        "WEG2-RESUME expert-rearm layers=%d rows_from_store=%d",
    ]
    pos = [src.find(mk) for mk in marks]
    assert all(p >= 0 for p in pos), dict(zip(marks, pos))
    assert pos == sorted(pos), dict(zip(marks, pos))


class _Clock:
    def __init__(self):
        self.ms = 0.0

    def __call__(self):
        return self.ms / 1000.0


class _FakeEvent:
    def __init__(self, clock, done_at):
        self.clock, self.done_at = clock, done_at

    def query(self):
        return self.clock.ms >= self.done_at

    def synchronize(self):
        self.clock.ms = max(self.clock.ms, self.done_at)


class _FakeOps:
    """One copy stream: each issue costs ``cost_ms`` behind the previous one."""

    def __init__(self, clock, cost_ms):
        self.clock, self.cost_ms = clock, cost_ms
        self.busy_until = 0.0
        self.waited = []
        self.after = 0

    def new_stream(self):
        return object()

    def stream_ctx(self, _s):
        return contextlib.nullcontext()

    def after_current(self, _s):
        self.after += 1

    def record(self, _s):
        self.busy_until = max(self.busy_until, self.clock.ms) + self.cost_ms
        return _FakeEvent(self.clock, self.busy_until)

    def current_waits(self, ev):
        self.waited.append(ev)


def _timed(issue_gap_ms, legs_end_ms, cost_ms=20.0):
    clock = _Clock()
    ops = _FakeOps(clock, cost_ms)
    m = _x141_model(1)
    phases = [("pre_leg", 1.0)]
    pf = eo.ExpertRearmPrefetch([(m, GPU_MEMORY_TYPE_WEIGHTS)], phases=phases,
                                stream_ops=ops, clock=clock)
    for tag in X141_WAKE_ORDER:
        pf.issue(tag)
        clock.ms += issue_gap_ms
    clock.ms = max(clock.ms, legs_end_ms)
    phases.append(("leg_collects", legs_end_ms))
    j = pf.join(phases)
    return pf, ops, j


def test_join_waits_nothing_when_the_legs_hid_the_copies():
    pf, ops, j = _timed(issue_gap_ms=100.0, legs_end_ms=1929.0)
    assert j.wait_ms == pytest.approx(0.0)
    assert not j.pending_at_join
    assert ops.waited and ops.waited[-1] is pf.event  # GPU order for the rearm's work
    assert ops.after == len(pf.tags_issued)


def test_join_waits_only_the_rest():
    # 10 tags x 20 ms queued back to back from t=0; the legs end at 150 ms
    pf, _ops, j = _timed(issue_gap_ms=0.0, legs_end_ms=150.0)
    assert len(pf.tags_issued) == 10
    assert j.pending_at_join
    assert j.wait_ms == pytest.approx(200.0 - 150.0)


def test_switch_off_is_the_serial_path():
    mgr = _manager(_x141_model(1))
    with envs.SGLANG_WEG2_REARM_PREFETCH.override(False):
        assert mgr._weg2_rearm_prefetch_begin([]) is None
    ms, _l, rows_serial = _serial(1)
    m_off = mgr.tp_worker.model_runner.model
    layers, rows = eo.rearm_expert_offload_after_wake(m_off, prefetch=None)
    assert rows == rows_serial == 232
    assert _digest(m_off) == _digest(ms)
    assert eo.REARM_PREFETCH_OFF_FIELDS == "prefetched=0 rows wait_ms=0.0 overlap=off"


def test_switch_on_plans_by_region_and_the_draft_waits_for_its_tag():
    target, draft = _x141_model(1), _x141_model(2)
    mgr = _manager(target, draft)
    with envs.SGLANG_WEG2_REARM_PREFETCH.override(True):
        pf = mgr._weg2_rearm_prefetch_begin([("pre_leg", 1.0)])
    assert pf is not None
    # the target's 29 layers under their chunk tags, the draft's under weights_draft
    assert pf.planned_layers == 2 * P_STAGE0_LAYERS
    assert pf.planned_rows == 2 * 232
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT in pf.planned_tags
    for tag in X141_WAKE_ORDER:
        if tag != GPU_MEMORY_TYPE_WEIGHTS_DRAFT:
            mgr._weg2_rearm_prefetch_issue(pf, tag)
    j = pf.join()
    assert j.rows == 232 and j.left_serial == P_STAGE0_LAYERS
    _l, rows_t = eo.rearm_expert_offload_after_wake(target, prefetch=pf)
    _l, rows_d = eo.rearm_expert_offload_after_wake(draft, prefetch=pf)
    assert (rows_t, rows_d) == (0, 232)  # the draft's rows load serially
    assert _digest(target) == _digest(_serial(1)[0])
    assert _digest(draft) == _digest(_serial(2)[0])


def test_mutant_prefetch_discarded_changes_the_digest(monkeypatch):
    """Mutant: the prefetch marks its layers loaded but its copies are lost
    (e.g. a join that drops the side stream's work). The rearm then skips the
    rows and the digest must differ -- otherwise the equality test above
    would pass vacuously."""

    def _discarding_issue(self, tag):
        layers = self._by_tag.pop(str(tag), None) or []
        for lay in layers:
            self.loaded.add(lay.key)
        self.tags_issued.append(str(tag))
        return 0

    ms, _l, _r = _serial(1)
    monkeypatch.setattr(eo.ExpertRearmPrefetch, "issue", _discarding_issue)
    mp, _pf, j, _lp, rows_left = _prefetched(1)
    assert rows_left == 0 and j.rows == 0
    assert _digest(mp) != _digest(ms)


def test_rearm_refuses_an_unjoined_prefetch():
    m = _x141_model(1)
    pf = eo.ExpertRearmPrefetch([(m, GPU_MEMORY_TYPE_WEIGHTS)])
    pf.issue("weights_0")
    with pytest.raises(RuntimeError, match="nicht gejoint"):
        eo.rearm_expert_offload_after_wake(m, prefetch=pf)


def test_tag_not_resumed_in_this_wake_stays_serial():
    order = [t for t in X141_WAKE_ORDER if t != "weights_8"]  # layers 24..26
    ms, _l, _r = _serial(1)
    mp, _pf, j, _lp, rows_left = _prefetched(1, order=order)
    assert j.left_serial == 3
    assert rows_left == 3 * 8 and j.rows == 232 - 3 * 8
    assert _digest(mp) == _digest(ms)


def test_unmapped_buffer_is_never_written_early():
    ms, _l, _r = _serial(2)
    mp, _pf, j, _lp, rows_left = _prefetched(2, mapped=lambda _buf: False)
    assert j.rows == 0 and j.skipped_unmapped == P_STAGE0_LAYERS
    assert rows_left == 232
    assert _digest(mp) == _digest(ms)


def test_issue_failure_leaves_the_rest_serial_and_the_rearm_raises_it():
    m = _x141_model(1)
    # layer 4 (weights_1) lost its store: the serial rearm has always raised here
    buf, _spill = m.model.layers[4].mlp.experts._moe_offload_presplit[ATTRS[0]]
    m.model.layers[4].mlp.experts._moe_offload_presplit[ATTRS[0]] = (buf, None)
    mgr = _manager(m)
    with envs.SGLANG_WEG2_REARM_PREFETCH.override(True):
        pf = mgr._weg2_rearm_prefetch_begin([])
    for tag in X141_WAKE_ORDER:
        mgr._weg2_rearm_prefetch_issue(pf, tag)
    assert pf.failed
    j = pf.join()
    assert j.left_serial > 0
    with pytest.raises(RuntimeError, match="keinen Store"):
        eo.rearm_expert_offload_after_wake(m, prefetch=pf)


# --------------------------------------------------------------------------
# manager scaffolding (the H25d test's shape)


class _Runner:
    def __init__(self, model):
        self.model = model
        self.model_config = None


def _manager(target, draft=None):
    tp_worker = SimpleNamespace(model_runner=_Runner(target))
    draft_worker = None if draft is None else SimpleNamespace(draft_model_runner=_Runner(draft))
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=tp_worker, draft_worker=draft_worker, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True)
