"""H91d: the MTP draft KV of a PARKED D request goes with it (weg2/d_park_draft.py).

User decision 25.09. "Ausnahme nur fuers Parken": with the draft tier OFF
(no draft host pool, no draft write-back / load-back) a parked request's draft
rows had no way off the card -- the flip park's sleep flushes the tree, a
pressure park's span is evicted -- and the resumed request (armed draft-cold
at its first admission, the arming never re-runs) drafted over whatever its
NEW slots held.  These tests drive the collaborator against stand-ins:

* the flip park copies the non-zero draft rows off BEFORE ``retract_all``;
* the pressure park snapshots the slot rows BEFORE ``retract_decode``;
* the resume writes the exact prefix state back at the NEW slots (saved rows
  at their positions, zeros elsewhere) and says ``digest=match``;
* off / tier on / no draft pool: nothing is touched (H91b byte for byte);
* abort and the request's death free the buffer; the ledger post returns to 0;
* a flip park over the L2 cap goes to L3 (a file) and comes back from it.
"""
from __future__ import annotations

import gc
import logging
import os
import types
from collections import deque
from contextlib import ExitStack

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))

SIZE = 512
HEADS, DIM = 2, 8
PROMPT = 20  # P's rows: the #993 zero fill
COMMITTED = 30  # D computed rows [20, 30)


@pytest.fixture
def pd():
    from sglang.srt.weg2 import d_park_draft

    return d_park_draft


@pytest.fixture(autouse=True)
def _group_d_tier_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    with ExitStack() as st:
        st.enter_context(envs.SGLANG_WEG2_HICACHE_DRAFT_TIER.override("off"))
        st.enter_context(envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(str(tmp_path)))
        yield


class _Pool:
    """MHA-shaped draft pool (NEXTN: one full-attention layer), bf16."""

    def __init__(self, layers=(40,)):
        self.full_attention_layer_id_mapping = {lid: i for i, lid in enumerate(layers)}
        g = torch.Generator().manual_seed(7)
        self.k = {lid: torch.zeros(SIZE, HEADS, DIM, dtype=torch.bfloat16) for lid in layers}
        self.v = {lid: torch.zeros(SIZE, HEADS, DIM, dtype=torch.bfloat16) for lid in layers}
        self._g = g

    def get_key_buffer(self, lid):
        return self.k[lid]

    def get_value_buffer(self, lid):
        return self.v[lid]

    def fill_random(self, slots):
        for t in list(self.k.values()) + list(self.v.values()):
            t[slots] = torch.randn(len(slots), HEADS, DIM, generator=self._g).to(t.dtype) + 3.0

    def rows(self, slots):
        return torch.cat(
            [t[slots].view(torch.uint8).reshape(len(slots), -1)
             for t in list(self.k.values()) + list(self.v.values())], dim=1
        ).clone()


def _draft_worker(pool):
    return types.SimpleNamespace(
        draft_worker=types.SimpleNamespace(draft_runner=types.SimpleNamespace(token_to_kv_pool=pool))
    )


def _req(rid, seq, *, pool_idx=0, committed=COMMITTED):
    return types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 25, output_ids=[0] * 6,
        is_fast_lane=False, spill_class=None, req_pool_idx=pool_idx,
        kv_committed_len=committed, prefix_indices=torch.empty((0,), dtype=torch.int64),
    )


class _Batch:
    def __init__(self, reqs, sched=None):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.sched = sched

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        # the retraction: slots to the tree, the req slot row freed; the
        # sleep then flushes the tree and the rows are reused by others
        out, self.reqs = self.reqs, []
        for r in out:
            slots = self.sched.req_to_token_pool.req_to_token[r.req_pool_idx, :r.kv_committed_len]
            self.sched.pool.fill_random(slots.long())
            r.req_pool_idx = None
            r.kv_committed_len = 0
        return out


class _Sched:
    def __init__(self, running=(), *, pool="default"):
        self.pool = _Pool() if pool == "default" else pool
        self.draft_worker = _draft_worker(self.pool)
        self.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((4, 64), dtype=torch.int32)
        )
        self.running_batch = _Batch(running, self)
        self.waiting_queue = []
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace()
        self.weg2_dormant = False
        self.sent = []
        self.enable_hicache_storage = False
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o))
        )

    def _969ad_note_retract(self, req, site):
        pass

    def _add_request_to_queue(self, req, is_retracted=False):
        self.waiting_queue.append(req)


def _seat(s, req, row, base):
    """Give ``req`` slot row ``row`` = slots base..base+63; D's rows real."""
    s.req_to_token_pool.req_to_token[row] = torch.arange(base, base + 64, dtype=torch.int32)
    req.req_pool_idx = row
    slots = torch.arange(base, base + 64)
    if s.pool is not None:
        for t in list(s.pool.k.values()) + list(s.pool.v.values()):
            t[slots[:PROMPT]] = 0
        s.pool.fill_random(slots[PROMPT:COMMITTED])
    return slots


def _park(s, epoch=5):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="d-to-p"))


def _resume(s, req, *, row=1, base=300, n_prefix=COMMITTED - 1):
    """The re-admission: NEW slots whose draft rows hold a previous
    occupant's bytes, prefix matched from host up to ``n_prefix``."""
    s.req_to_token_pool.req_to_token[row] = torch.arange(base, base + 64, dtype=torch.int32)
    new = torch.arange(base, base + 64)
    s.pool.fill_random(new)
    req.req_pool_idx = row
    req.prefix_indices = new[:n_prefix].clone()
    return new


def test_flip_park_saves_the_real_rows_before_the_retraction_and_the_resume_restores_them(pd, caplog):
    req = _req("weg2-1-1", 1)
    s = _Sched([req])
    old = _seat(s, req, 0, 100)
    before = s.pool.rows(old[:COMMITTED])
    with caplog.at_level(logging.INFO):
        out = _park(s)
    assert out.success and out.parked == ["weg2-1-1"]
    e = pd.entry_of(req)
    assert e is not None and e.site == ds.SITE_FLIP and e.tier == pd.TIER_L2
    assert e.rows == COMMITTED - PROMPT  # only D's rows, P's zero rows are not carried
    assert e.positions.tolist() == list(range(PROMPT, COMMITTED))
    row_bytes = 2 * HEADS * DIM * 2  # K+V, bf16
    assert e.nbytes == e.rows * row_bytes + e.rows * 8
    assert pd.ledger_of(s).l2 == e.nbytes and pd.ledger_of(s).live == 1
    assert "WEG2-D-PARK DRAFT-SAVE rid=weg2-1-1 site=flip rows=10 of=30" in caplog.text
    assert "tier=L2" in caplog.text and "WEG2-HOST-LEDGER D-PARK-DRAFT event=save" in caplog.text
    # the rows are gone from the card (the retraction stand-in reused them)
    assert not torch.equal(s.pool.rows(old[:COMMITTED]), before)

    new = _resume(s, req)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert pd.restore_admitted(s, types.SimpleNamespace(reqs=[req])) == 1
    after = s.pool.rows(new[:COMMITTED - 1])
    assert torch.equal(after, before[:COMMITTED - 1])  # exact: zeros AND D's rows
    assert "WEG2-D-PARK DRAFT-RESTORE rid=weg2-1-1 site=flip rows=9 of=10 zeroed=20" in caplog.text
    assert "digest=match" in caplog.text
    assert pd.entry_of(req) is None
    assert pd.ledger_of(s).l2 == 0 and pd.ledger_of(s).live == 0


def test_pressure_park_snapshots_before_retract_decode_and_saves_after(pd):
    req, other = _req("young", 2), _req("old", 1, pool_idx=2)
    s = _Sched()
    old = _seat(s, req, 0, 100)
    _seat(s, other, 2, 400)
    before = s.pool.rows(old[:COMMITTED])
    batch = types.SimpleNamespace(reqs=[other, req])
    snap = pd.snapshot(s, batch)
    assert snap == {id(req): (0, COMMITTED), id(other): (2, COMMITTED)}
    # retract_decode: release + reset_for_retract (slot row and length gone)
    req.req_pool_idx, req.kv_committed_len = None, 0
    assert pd.save_retracted(s, [req], snap) == 1
    e = pd.entry_of(req)
    assert e.site == ds.SITE_PRESSURE and e.rows == COMMITTED - PROMPT
    assert pd.entry_of(other) is None  # not retracted: nothing carried
    new = _resume(s, req, row=3, base=200)
    pd.restore_admitted(s, types.SimpleNamespace(reqs=[req]))
    assert torch.equal(s.pool.rows(new[:COMMITTED - 1]), before[:COMMITTED - 1])


def test_switch_off_is_h91b_byte_for_byte(pd):
    with envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.override(False):
        req = _req("a", 1)
        s = _Sched([req])
        _seat(s, req, 0, 100)
        _park(s)
        assert pd.entry_of(req) is None
        assert pd.snapshot(s, types.SimpleNamespace(reqs=[req])) is None
        new = _resume(s, req)
        rows = s.pool.rows(new)
        assert pd.restore_admitted(s, types.SimpleNamespace(reqs=[req])) == 0
        assert torch.equal(s.pool.rows(new), rows)
        assert not hasattr(s, "_weg2_d_park_draft_ledger")


def test_draft_tier_on_carries_nothing_extra(pd):
    with envs.SGLANG_WEG2_HICACHE_DRAFT_TIER.override("auto"):
        req = _req("a", 1)
        s = _Sched([req])
        _seat(s, req, 0, 100)
        _park(s)
        assert pd.entry_of(req) is None and not pd.armed()


def test_park_off_or_group_p_carries_nothing(pd, monkeypatch):
    monkeypatch.setenv(ds.PARK_ENV, "0")
    assert not pd.armed()
    monkeypatch.delenv(ds.PARK_ENV)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert not pd.armed()


def test_a_rank_without_a_draft_pool_skips_deterministically(pd):
    """Form A: the 3080 expert workers are solo-draft SHADOWS (no draft pool).
    Same park, same parked list, no entry, no exception."""
    req = _req("a", 1)
    s = _Sched([req], pool=None)
    s.req_to_token_pool.req_to_token[0] = torch.arange(100, 164, dtype=torch.int32)
    s.running_batch = _Batch([req], s)
    s.running_batch.retract_all = lambda *a, **k: [req]
    out = _park(s)
    assert out.success and out.parked == ["a"]
    assert pd.entry_of(req) is None
    assert pd.snapshot(s, types.SimpleNamespace(reqs=[req])) is not None  # replicated
    assert pd.save_retracted(s, [req], {id(req): (0, COMMITTED)}) == 0


def test_an_abort_of_a_parked_request_frees_its_buffer(pd, caplog):
    req = _req("weg2-2-1", 1)
    s = _Sched([req])
    _seat(s, req, 0, 100)
    _park(s)
    assert pd.ledger_of(s).l2 > 0
    with caplog.at_level(logging.INFO):
        assert rt.park_abort(s, types.SimpleNamespace(rid="weg2-2-1", abort_all=False)) == 1
    assert pd.entry_of(req) is None
    assert pd.ledger_of(s).l2 == 0 and pd.ledger_of(s).live == 0
    assert "WEG2-D-PARK DRAFT-DROP rid=weg2-2-1 reason=abort" in caplog.text


def test_the_buffer_dies_with_the_request(pd):
    req = _req("a", 1)
    s = _Sched([req])
    _seat(s, req, 0, 100)
    _park(s)
    led = pd.ledger_of(s)
    assert led.live == 1
    s.weg2_d_parked = []
    del req
    gc.collect()
    assert led.live == 0 and led.l2 == 0


def test_a_flip_park_over_the_cap_goes_to_l3_and_comes_back(pd, tmp_path, caplog):
    with envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.override(0):
        req = _req("weg2-3-1", 1)
        s = _Sched([req])
        old = _seat(s, req, 0, 100)
        before = s.pool.rows(old[:COMMITTED])
        with caplog.at_level(logging.INFO):
            _park(s)
        e = pd.entry_of(req)
        e._state["future"].result()
        assert e.tier == pd.TIER_L3 and os.path.exists(e.path)
        assert e.path.startswith(os.path.join(str(tmp_path), pd.L3_SUBDIR))
        assert e._blocks is None  # RAM freed once written
        led = pd.ledger_of(s)
        assert led.l2 == 0 and led.l3 == e.nbytes
        assert "tier=L3" in caplog.text
        path = e.path
        new = _resume(s, req)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            pd.restore_admitted(s, types.SimpleNamespace(reqs=[req]))
        assert torch.equal(s.pool.rows(new[:COMMITTED - 1]), before[:COMMITTED - 1])
        assert "tier=L3 digest=match" in caplog.text
        assert not os.path.exists(path) and led.l3 == 0


def test_a_pressure_park_over_the_cap_is_not_carried(pd, caplog):
    with envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.override(0):
        req = _req("young", 2)
        s = _Sched()
        _seat(s, req, 0, 100)
        snap = pd.snapshot(s, types.SimpleNamespace(reqs=[req]))
        with caplog.at_level(logging.INFO):
            assert pd.save_retracted(s, [req], snap) == 0
        assert pd.entry_of(req) is None
        assert "tier=none" in caplog.text and "L3 only across a sleep" in caplog.text


def test_an_unparked_request_is_never_touched(pd):
    req = _req("fresh", 1)
    s = _Sched()
    new = _resume(s, req)
    rows = s.pool.rows(new)
    assert pd.restore_admitted(s, types.SimpleNamespace(reqs=[req])) == 0
    assert torch.equal(s.pool.rows(new), rows)


def test_a_corrupted_buffer_is_named(pd, caplog):
    req = _req("a", 1)
    s = _Sched([req])
    _seat(s, req, 0, 100)
    _park(s)
    pd.entry_of(req)._blocks[0][0].fill_(0x55)
    _resume(s, req)
    with caplog.at_level(logging.INFO):
        pd.restore_admitted(s, types.SimpleNamespace(reqs=[req]))
    assert "digest=MISMATCH" in caplog.text


def test_a_partial_prefix_restores_only_below_it(pd):
    req = _req("a", 1)
    s = _Sched([req])
    old = _seat(s, req, 0, 100)
    before = s.pool.rows(old[:COMMITTED])
    _park(s)
    new = _resume(s, req, n_prefix=24)
    tail = s.pool.rows(new[24:COMMITTED])
    pd.restore_admitted(s, types.SimpleNamespace(reqs=[req]))
    assert torch.equal(s.pool.rows(new[:24]), before[:24])
    assert torch.equal(s.pool.rows(new[24:COMMITTED]), tail)  # the extend's, untouched


def test_the_resume_funnel_without_the_carry_drafts_over_foreign_rows():
    """The finding, driven through the REAL admission funnel: a D request is
    armed draft-cold at its first admission (P's prefix came from host) and
    ``reset_for_retract`` never clears that, so at the resume
    ``arm_draft_cold_for_admission`` skips it -- no scrub, no cold mark -- and
    the new slots keep a previous occupant's draft bytes.  With H91d the
    run_batch sequence (arm, then restore) leaves the exact parked state.
    Red on H91b by assertion, not by import."""
    from sglang.srt.managers.phase_flip_draft_bootstrap import (
        COLD_ARMED_ATTR,
        arm_draft_cold_for_admission,
    )

    req = _req("weg2-9-1", 1)
    setattr(req, COLD_ARMED_ATTR, True)  # the first admission on D armed it
    s = _Sched([req])
    s.tree_cache = None
    old = _seat(s, req, 0, 100)
    before = s.pool.rows(old[:COMMITTED])
    _park(s)
    new = _resume(s, req)
    foreign = s.pool.rows(new[:COMMITTED - 1])
    batch = types.SimpleNamespace(reqs=[req])
    assert arm_draft_cold_for_admission(s, batch) == {"cold": 0, "rows": 0}
    assert torch.equal(s.pool.rows(new[:COMMITTED - 1]), foreign)  # skipped: foreign bytes
    try:  # the run_batch hook right after the arming (scheduler.py, H91d)
        from sglang.srt.weg2 import d_park_draft
    except ImportError:
        d_park_draft = None
    if d_park_draft is not None:
        d_park_draft.restore_admitted(s, batch)
    assert torch.equal(s.pool.rows(new[:COMMITTED - 1]), before[:COMMITTED - 1])


# ---- wiring ratchets (the sites a desk test cannot drive) -------------------

def _read(*parts):
    return open(os.path.join(_SRT, *parts)).read()


def test_wiring_scheduler_delegates_save_restore_drop():
    sch = _read("managers", "scheduler.py")
    i = sch.index("_draft_snap = self._weg2_d_park_draft_snapshot(batch)")
    j = sch.index("batch.retract_decode(", i)
    k = sch.index("self._weg2_d_park_draft_save(retracted_reqs, _draft_snap)", j)
    assert i < j < k
    a = sch.index("arm_draft_cold_for_admission(self, batch)\n")
    assert sch.index("self._weg2_d_park_draft_restore(batch)", a) - a < 200
    h = sch.index("def _weg2_abort_dormant_hold")
    assert '_dpd.drop_all(self, gone, "abort")' in sch[h:h + 2500]
    assert "d_park_draft.restore_admitted(self, batch)" in sch


def test_wiring_flip_park_saves_before_retract_all():
    src = _read("weg2", "d_park_runtime.py")
    assert src.index("d_park_draft.save_parked(sched, running") < src.index(
        "sched.running_batch.retract_all(")
    assert 'd_park_draft.drop_all(sched, gone, "abort")' in src


def test_the_switch_follows_the_env_conventions():
    src = _read("environ.py")
    assert "SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV = EnvBool(True)" in src
    assert "SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB = EnvInt(256)" in src
    # bounding-default value pin (docs/dev/CONVENTION_bounding_defaults.md):
    # 256 MiB = ONE full 262,144-token context of NF's draft (~1 KiB/token),
    # the host-RAM bound of the in-process L2 before a flip park goes to disk.
    with envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.override(None):
        envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.clear()
        assert envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.get() == 256
    with envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.override(None):
        envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.clear()
        assert envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.get() is True
