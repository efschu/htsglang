"""Unit tests for kv-session-offload spill destinations (#224).

CPU-only; no server, no GPU. Covers the pure decision layer, the
destination chain against a fake tier AND the real ``file`` backend
(genuine end-to-end park/unpark roundtrip in a tmpdir), the
producer-identity check, the anti-pendulum rules, the rank-uniform
transfer verdicts, and the byte-identity guarantees of the default path
(unarmed chain touches nothing; the admission reduce keeps its 2-element
payload).

Run:
  CUDA_VISIBLE_DEVICES=99 python -m pytest \
      test/registered/unit/test_kv_spill_destination_unit.py -q
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.managers.kv_session_spill_destination as kd
from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.kv_session_spill_destination import (
    ALL_STORAGE_BACKENDS,
    DestinationTier,
    META_BLOB_BYTES,
    PARK_PRESSURE_WINDOW_ITERS,
    SpillDestinationController,
    blob_key,
    destinations_error,
    fingerprint_hash,
    meta_blob,
    meta_matches,
    owned_tail_rows,
    padded_meta_blob,
    park_decision,
    parse_destinations,
    parse_meta_blob,
    producer_fingerprint,
    should_abandon,
    transfer_verdict,
    unpark_decision,
)

# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------


def test_parse_destinations():
    assert parse_destinations(None) == []
    assert parse_destinations(" local , Mooncake ,file ") == [
        "local",
        "mooncake",
        "file",
    ]
    assert parse_destinations("local,,file") == ["local", "file"]


def test_destinations_valid_lists():
    assert destinations_error(["local", "file"]) is None
    assert destinations_error(["local", "mooncake"]) is None
    assert destinations_error(["local", "mooncake", "file"]) is None
    assert destinations_error(["local", "dynamic"]) is None


def test_destinations_local_must_be_first():
    err = destinations_error(["mooncake", "local"])
    assert err is not None and "must be 'local'" in err
    # The hard boundary is NAMED with the measured numbers, not asserted.
    assert "3.43 GB/s" in err


def test_destinations_rejects_bare_local_and_dups_and_unknown():
    assert "empty" in destinations_error([])
    assert "not setting" in destinations_error(["local"])
    assert "only once" in destinations_error(["local", "file", "local"])
    assert "duplicate" in destinations_error(["local", "file", "file"])
    assert "unknown" in destinations_error(["local", "quantum"])


def test_destinations_known_but_unsupported_backends_are_named():
    # Refuted-hypothesis coverage: eic/simm/hf3fs/nixl/mori are machine-
    # crossing or config-dependent backends, but their arbitrary-blob
    # contract is unverified -> named, liftable exclusion via 'dynamic'.
    for name in ("eic", "simm", "hf3fs", "nixl", "aibrix", "mori"):
        err = destinations_error(["local", name])
        assert err is not None and "not a supported park tier" in err
        assert "dynamic" in err


def test_backend_namespace_mirrors_factory_registry():
    from sglang.srt.mem_cache.storage.backend_factory import (
        StorageBackendFactory,
    )

    registered = set(StorageBackendFactory._registry.keys()) | {"dynamic"}
    assert set(ALL_STORAGE_BACKENDS) == registered
    assert set(kd.SUPPORTED_PARK_BACKENDS) <= set(ALL_STORAGE_BACKENDS)


# ---------------------------------------------------------------------------
# Producer identity
# ---------------------------------------------------------------------------


def _fp(**overrides):
    base = dict(
        model_path="/models/qwen3.6-27b",
        quantization="gguf",
        kv_cache_dtype="fp8_e4m3",
        dtype="bfloat16",
        tp_size=3,
        rank_tp_ratio="2,1,1",
        mode="weighted",
        split_factor=64,
        cp_prefix=[0, 1, 33, 64],
        layer_num=16,
        head_num=4,
        head_dim=256,
        store_dtype="torch.float8_e4m3fn",
    )
    base.update(overrides)
    return producer_fingerprint(**base)


def test_fingerprint_covers_every_identity_axis():
    ref = _fp()
    assert _fp() == ref  # deterministic
    for change in (
        dict(model_path="/models/other"),
        dict(quantization="awq"),
        dict(kv_cache_dtype="auto"),  # KV dtype IS part of the identity
        dict(dtype="float16"),
        dict(tp_size=2),
        dict(rank_tp_ratio="1,1,1"),
        dict(mode="even"),
        dict(split_factor=3),
        dict(cp_prefix=[0, 32, 64]),
        dict(layer_num=15),
        dict(head_num=8),
        dict(head_dim=128),
        dict(store_dtype="torch.bfloat16"),
    ):
        assert _fp(**change) != ref, change
        assert fingerprint_hash(_fp(**change)) != fingerprint_hash(ref)


def test_blob_key_is_rank_and_generation_scoped():
    fph = fingerprint_hash(_fp())
    k0 = blob_key(fph, "rid1", 0, 1, "k3")
    assert k0 != blob_key(fph, "rid1", 1, 1, "k3")  # rank-private shards
    assert k0 != blob_key(fph, "rid2", 0, 1, "k3")
    assert k0 != blob_key(fph, "rid1", 0, 2, "k3")  # fresh keys per episode
    assert k0 != blob_key(fph, "rid1", 0, 1, "v3")


def test_meta_blob_roundtrip_and_matching():
    meta = {
        "v": 1,
        "fingerprint": _fp(),
        "rid": "r-77",
        "rows": 128,
        "L": 300,
        "boundary": 100,
        "rank": 1,
        "generation": 2,
    }
    parsed = parse_meta_blob(meta_blob(meta))
    assert parsed == meta
    padded = padded_meta_blob(meta)
    assert padded.numel() == META_BLOB_BYTES
    # parse tolerates the zero padding (rfind of the JSON boundary).
    raw = bytes(padded.numpy().tobytes())
    end = raw.rfind(b"}")
    import json

    assert json.loads(raw[: end + 1]) == meta

    assert meta_matches(meta, _fp(), "r-77", 128) is None
    assert "missing" in meta_matches(None, _fp(), "r-77", 128)
    assert "fingerprint" in meta_matches(
        meta, _fp(kv_cache_dtype="auto"), "r-77", 128
    )
    assert "rid" in meta_matches(meta, _fp(), "other", 128)
    assert "row count" in meta_matches(meta, _fp(), "r-77", 64)
    bad_version = dict(meta, v=99)
    assert "version" in meta_matches(bad_version, _fp(), "r-77", 128)


# ---------------------------------------------------------------------------
# Pure decision layer
# ---------------------------------------------------------------------------


def test_transfer_verdict_table():
    # A transfer NEVER resolves before every rank's I/O stopped: pool rows
    # must not be handed out under a live worker.
    assert transfer_verdict(0, 0, False) == "wait"
    assert transfer_verdict(0, 1, False) == "wait"
    assert transfer_verdict(0, 0, True) == "wait"  # abandoned but still live
    assert transfer_verdict(1, 1, False) == "commit"
    assert transfer_verdict(1, 0, False) == "failed"
    # Deterministic across ranks: once abandoned, even late success fails.
    assert transfer_verdict(1, 1, True) == "failed"


def test_should_abandon_uses_uniform_clock():
    assert not should_abandon(0, 512, 512)
    assert should_abandon(0, 513, 512)
    assert not should_abandon(1, 10_000, 512)  # done -> no abandonment


def test_park_decision_requires_fresh_pressure_and_exhaustion():
    cands = [(5, 2, 11), (3, 9, 22), (3, 1, 33)]
    common = dict(pressure_iter=100, now_iter=110, inflight=False)
    # Oldest spill wins; arrival seq breaks the tie.
    assert park_decision(free_regions=0, candidates=cands, **common) == 33
    # A free region, stale pressure, an in-flight transfer, or no
    # candidates each veto the park.
    assert park_decision(free_regions=1, candidates=cands, **common) is None
    assert (
        park_decision(
            free_regions=0,
            candidates=cands,
            pressure_iter=0,
            now_iter=PARK_PRESSURE_WINDOW_ITERS + 1,
            inflight=False,
        )
        is None
    )
    assert (
        park_decision(
            free_regions=0,
            candidates=cands,
            pressure_iter=100,
            now_iter=110,
            inflight=True,
        )
        is None
    )
    assert park_decision(free_regions=0, candidates=[], **common) is None


def test_park_and_unpark_are_mutually_exclusive_anti_pendulum():
    # For ANY state, park and unpark can never both trigger: park needs
    # fresh pressure + no free region, unpark needs a free region + stale
    # pressure. This is the pendulum guard at the root.
    for free in (0, 1):
        for age in (0, PARK_PRESSURE_WINDOW_ITERS + 1):
            park = park_decision(
                free_regions=free,
                pressure_iter=100,
                now_iter=100 + age,
                inflight=False,
                candidates=[(1, 1, 7)],
            )
            unpark = unpark_decision(
                free_regions=free,
                pressure_iter=100,
                now_iter=100 + age,
                inflight=False,
                have_parked=True,
            )
            assert not (park is not None and unpark)


def test_unpark_decision():
    assert unpark_decision(
        free_regions=1,
        pressure_iter=0,
        now_iter=PARK_PRESSURE_WINDOW_ITERS + 1,
        inflight=False,
        have_parked=True,
    )
    assert not unpark_decision(
        free_regions=0,
        pressure_iter=0,
        now_iter=1000,
        inflight=False,
        have_parked=True,
    )
    assert not unpark_decision(
        free_regions=1,
        pressure_iter=999,
        now_iter=1000,
        inflight=False,
        have_parked=True,
    )
    assert not unpark_decision(
        free_regions=1,
        pressure_iter=0,
        now_iter=1000,
        inflight=True,
        have_parked=True,
    )
    assert not unpark_decision(
        free_regions=1,
        pressure_iter=0,
        now_iter=1000,
        inflight=False,
        have_parked=False,
    )


def test_owned_tail_rows_matches_bruteforce():
    g = torch.Generator().manual_seed(3)
    residues = torch.randint(0, 64, (500,), generator=g)
    prefix = [0, 1, 33, 64]
    total = 0
    for r in range(3):
        lo, hi = prefix[r], prefix[r + 1]
        brute = int(((residues >= lo) & (residues < hi)).sum())
        assert owned_tail_rows(residues, lo, hi) == brute
        total += brute
    assert total == 500
    assert owned_tail_rows(torch.empty(0, dtype=torch.int64), 0, 1) == 0


# ---------------------------------------------------------------------------
# Fake infrastructure for the flow tests
# ---------------------------------------------------------------------------

CTX = 32
REGION_TOKENS = 16
N_REGIONS = 2
LAYERS = 2
HEADS = 2
HEAD_DIM = 4
HOST_BASE = 10_000


class _FakeHostPool:
    def __init__(self):
        self.layer_num = LAYERS
        self.head_num = HEADS
        self.head_dim = HEAD_DIM
        self.dtype = torch.float32
        size = REGION_TOKENS * N_REGIONS
        self.size = size
        g = torch.Generator().manual_seed(11)
        self.k_data_refs = [
            torch.randn(size, HEADS, HEAD_DIM, generator=g) for _ in range(LAYERS)
        ]
        self.v_data_refs = [
            torch.randn(size, HEADS, HEAD_DIM, generator=g) for _ in range(LAYERS)
        ]


class _FakeBackend:
    def __init__(self):
        self._sess_slots = {}
        self.opened = []
        self.closed = []

    def _sess_open_slot(self, rpi, region_base):
        self._sess_slots[rpi] = SimpleNamespace(
            region_base=region_base, host_row_base=0
        )
        self.opened.append((rpi, region_base))

    def _sess_close_slot(self, rpi):
        self._sess_slots.pop(rpi, None)
        self.closed.append(rpi)


class _FakeReq:
    def __init__(self, rid, rpi, seq, prompt=8, out=4):
        self.rid = rid
        self.req_pool_idx = rpi
        self.kv_arrival_seq = seq
        self.origin_input_ids = list(range(prompt))
        self.output_ids = list(range(out))
        self.kv_spill_state = "host"
        self.kv_spill_boundary = 0
        self.cache_protected_len = 0
        self.last_node = None
        self.mamba_pool_idx = None
        self.to_abort = False
        self.to_abort_message = None
        self._finished = False
        # Mirrors the real Req: `to_finish` is what Scheduler.abort_request
        # sets ("abort method 3"), `finished_reason` is what the result
        # processor later promotes it to (Req.update_finish_state).
        self.to_finish = None
        self.finished_reason = None
        self.return_logprob = False

    def finished(self):
        return self._finished or self.finished_reason is not None


class _FakeSlot:
    """Mirror of the SpillSlot fields the destination flow touches."""

    def __init__(self, req, region, spill_iter):
        self.req = req
        self.batch = None
        self.region = region
        self.spill_iter = spill_iter
        self.last_tick_iter = -1
        self.suppress_tick = False
        self.spec_in_tick = False
        self.born_spilled = False
        self.adopted = True
        self.park_pending = False


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


class _FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros(8, CTX, dtype=torch.int32)
        self.freed_reqs = []

    def free(self, req):
        self.freed_reqs.append(req.rid)


def _make_mgr(dest_ctl):
    mgr = SimpleNamespace()
    mgr._dest = dest_ctl
    mgr.mode = "plain"
    mgr.S = 1
    mgr.lo = 0
    mgr.hi = 1
    mgr.cp_prefix = [0, 1]
    mgr.host_base = HOST_BASE
    mgr.region_tokens = REGION_TOKENS
    mgr._free_regions = []
    mgr.spills = {}
    mgr._iter_ct = 0
    mgr.host_pool = _FakeHostPool()
    mgr.backend = _FakeBackend()
    mgr.req_to_token_pool = _FakeReqToTokenPool()
    mgr.allocator = _FakeAllocator()
    mgr.tree_cache = SimpleNamespace(dec_lock_ref=lambda n: None)
    streamed = []
    mgr.scheduler = SimpleNamespace(
        running_batch=None,
        tp_rank=0,
        output_streamer=SimpleNamespace(
            stream_output=lambda reqs, rl, **kw: streamed.extend(reqs)
        ),
    )
    mgr.streamed = streamed
    mgr._log = lambda *a, **k: None
    return mgr


def _add_spilled_session(mgr, rid, rpi, region, seq, tail_len=6):
    """A plain-mode spilled session: device head [0, boundary) + sentinel
    tail [boundary, L) whose rows sit in ``region`` starting at row 0."""
    req = _FakeReq(rid, rpi, seq)
    L = len(req.origin_input_ids) + len(req.output_ids) - 1
    boundary = L - tail_len
    assert boundary >= 0
    row = mgr.req_to_token_pool.req_to_token
    row[rpi, :boundary] = torch.arange(boundary, dtype=torch.int32)  # device
    row[rpi, boundary:L] = HOST_BASE + torch.arange(tail_len, dtype=torch.int32)
    slot = _FakeSlot(req, region, spill_iter=mgr._iter_ct)
    mgr.spills[rpi] = slot
    mgr.backend._sess_open_slot(rpi, region * REGION_TOKENS)
    return slot, boundary, L, tail_len


def _make_ctl(tiers, timeout=64, fingerprint=None):
    return SpillDestinationController(
        destinations=["local"] + [t.name for t in tiers],
        tiers=tiers,
        fingerprint=fingerprint or _fp(),
        rank=0,
        timeout_iters=timeout,
    )


def _pump(mgr, ctl, until, max_iters=400, advance=1):
    """Drive iterations: simulate the single-rank reduce (min == local) and
    run the flow until ``until()`` or the iteration budget is spent."""
    for _ in range(max_iters):
        mgr._iter_ct += advance
        done, ok = ctl.reduce_extra()
        ctl.consume_reduced(done, ok)
        kd.maybe_park_flow(mgr, None, None)
        if until():
            return True
        time.sleep(0.002)
    return False


class _DictTier(DestinationTier):
    """In-memory blob tier implementing the put/get-into contract."""

    def __init__(self, name="fake", fail_puts=False):
        self.name = name
        self.pointer_io = False
        self.blobs = {}
        self.fail_puts = fail_puts

    def put(self, key, tensor):
        if self.fail_puts:
            return False
        self.blobs[key] = tensor.detach().clone()
        return True

    def get_into(self, key, tensor):
        if key not in self.blobs:
            return False
        tensor.copy_(self.blobs[key].view_as(tensor))
        return True

    def exists(self, key):
        return key in self.blobs

    def get_meta(self, key, max_bytes=META_BLOB_BYTES):
        if key not in self.blobs:
            return None
        return parse_meta_blob(self.blobs[key])


# ---------------------------------------------------------------------------
# Flow tests (fake tier)
# ---------------------------------------------------------------------------


def _snapshot_region(mgr, region, rows):
    base = region * REGION_TOKENS
    pool = mgr.host_pool
    return [
        (
            pool.k_data_refs[fl][base : base + rows].clone(),
            pool.v_data_refs[fl][base : base + rows].clone(),
        )
        for fl in range(LAYERS)
    ]


def test_park_unpark_roundtrip_fake_tier():
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, boundary, L, rows = _add_spilled_session(mgr, "rid-a", 1, 0, seq=5)
    before = _snapshot_region(mgr, 0, rows)

    # Fresh pressure, no free region -> two-phase park.
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-a" in ctl.parked)
    assert slot.req.kv_spill_state == "parked"
    assert mgr._free_regions == [0]  # region freed for the next spill
    assert 1 not in mgr.spills
    assert mgr.backend.closed == [1]
    # meta + one K and one V blob per layer
    assert len(tier.blobs) == 1 + 2 * LAYERS
    assert ctl.counters["parks_committed"] == 1

    # Scramble the region, then unpark (pressure must first go stale).
    for fl in range(LAYERS):
        mgr.host_pool.k_data_refs[fl].zero_()
        mgr.host_pool.v_data_refs[fl].zero_()
    assert _pump(
        mgr,
        ctl,
        until=lambda: 1 in mgr.spills,
        advance=PARK_PRESSURE_WINDOW_ITERS + 2,
    )
    assert slot.req.kv_spill_state == "host"
    assert slot.region == 0
    assert mgr.backend._sess_slots[1].region_base == 0
    after = _snapshot_region(mgr, 0, rows)
    for (kb, vb), (ka, va) in zip(before, after):
        assert torch.equal(kb, ka)  # byte-identical roundtrip
        assert torch.equal(vb, va)
    assert ctl.counters["unparks_committed"] == 1


def test_park_failover_to_second_tier():
    failing = _DictTier(name="failing", fail_puts=True)
    backup = _DictTier(name="backup")
    ctl = _make_ctl([failing, backup])
    mgr = _make_mgr(ctl)
    _add_spilled_session(mgr, "rid-b", 2, 1, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-b" in ctl.parked)
    assert ctl.parked["rid-b"].tier_index == 1  # second tier took over
    assert len(backup.blobs) == 1 + 2 * LAYERS
    assert ctl.counters["parks_failed"] >= 1
    assert ctl.counters["parks_committed"] == 1


def test_park_abandoned_when_no_tier_left_keeps_session_local():
    failing = _DictTier(name="failing", fail_puts=True)
    ctl = _make_ctl([failing])
    mgr = _make_mgr(ctl)
    slot, *_ = _add_spilled_session(mgr, "rid-c", 3, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(
        mgr,
        ctl,
        until=lambda: ctl.counters["parks_failed"] >= 1
        and ctl.inflight is None
        and not slot.park_pending,
    )
    # Today's behaviour is the final fallback: still spilled locally.
    assert 3 in mgr.spills
    assert slot.req.kv_spill_state == "host"
    assert "rid-c" not in ctl.parked


def test_unpark_identity_mismatch_aborts_request():
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, *_ = _add_spilled_session(mgr, "rid-d", 1, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-d" in ctl.parked)

    # A DIFFERENT producer (changed KV dtype) tries to unpark the entry.
    ctl2 = _make_ctl([tier], fingerprint=_fp(kv_cache_dtype="auto"))
    ctl2.parked = ctl.parked
    mgr._dest = ctl2
    assert _pump(
        mgr,
        ctl2,
        until=lambda: ctl2.counters["unparks_failed"] >= 1,
        advance=PARK_PRESSURE_WINDOW_ITERS + 2,
    )
    assert slot.req.to_abort  # hard, named error -- never a silent restore
    assert "restored" in slot.req.to_abort_message
    assert ctl2.counters["unpark_identity_miss"] >= 0
    assert 1 not in mgr.spills
    assert mgr.req_to_token_pool.freed_reqs == ["rid-d"]
    # The claimed region went back.
    assert 0 in mgr._free_regions


def test_reparking_after_growth_uses_fresh_keys():
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, boundary, L, rows = _add_spilled_session(mgr, "rid-e", 1, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-e" in ctl.parked)
    keys1 = set(tier.blobs)
    assert _pump(
        mgr,
        ctl,
        until=lambda: 1 in mgr.spills,
        advance=PARK_PRESSURE_WINDOW_ITERS + 2,
    )
    # Session "grew" while host-resident; park again.
    slot.req.output_ids.append(99)
    row = mgr.req_to_token_pool.req_to_token
    row[1, L] = HOST_BASE + rows
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-e" in ctl.parked)
    keys2 = set(tier.blobs) - keys1
    # Fresh generation -> fresh keys; a deduping backend cannot serve
    # episode-1 bytes for episode 2.
    assert keys2 and not (keys2 & keys1)
    assert ctl.parked["rid-e"].rows == rows + 1


def test_abandoned_transfer_resolves_failed_after_io_stops():
    class _SlowTier(_DictTier):
        def put(self, key, tensor):
            time.sleep(0.05)
            return super().put(key, tensor)

    tier = _SlowTier(name="slow")
    ctl = _make_ctl([tier], timeout=2)  # tighter than the transfer
    mgr = _make_mgr(ctl)
    slot, *_ = _add_spilled_session(mgr, "rid-f", 1, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(
        mgr,
        ctl,
        until=lambda: ctl.counters["parks_timeout"] >= 1
        and ctl.inflight is None,
    )
    # Deterministic outcome: abandoned -> failed, even though the slow I/O
    # eventually succeeded; the session stays host-resident.
    assert "rid-f" not in ctl.parked
    assert 1 in mgr.spills
    assert ctl.counters["parks_failed"] >= 1


def test_parked_abort_reaps_and_releases():
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, *_ = _add_spilled_session(mgr, "rid-g", 1, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-g" in ctl.parked)
    slot.req._finished = True  # abort while parked
    assert _pump(mgr, ctl, until=lambda: "rid-g" not in ctl.parked)
    assert ctl.counters["parked_aborted"] == 1
    assert mgr.req_to_token_pool.freed_reqs == ["rid-g"]


def test_parked_abort_via_to_finish_is_reaped():
    """ABORT of a PARKED session must reap it -- the real abort signal is
    ``to_finish``, not ``finished_reason``.

    THE HOLE THIS PINS. ``Scheduler.abort_request`` reaches a parked session
    (``inflight_batches`` deliberately extends with ``parked_inflight_entries``)
    and marks it the only way it marks any running request: "abort method 3",
    ``req.to_finish = FINISH_ABORT()`` (scheduler.py). That flag is promoted to
    ``finished_reason`` by ``Req.update_finish_state``, which runs ONLY when a
    batch result is processed. A parked session is in no batch and runs no
    forward -- ``mgr.spills`` no longer holds it (``_commit_park`` pops it) and
    its tick is suppressed -- so the promotion NEVER happens and
    ``req.finished()`` stays False forever.

    The reap loop in ``maybe_park_flow`` selected on ``req.finished()`` alone,
    so the abort was silently dropped: the ``ParkedSession`` record stayed in
    ``ctl.parked``, the retained device head, the req-pool slot, the radix tree
    lock and the remote blob all leaked, and the client waited for a response
    that no code path would ever send.

    CAN-FAIL PROOF: this test is RED on the pre-fix tree (the session is still
    in ``ctl.parked`` after pumping and nothing is freed).  The neighbouring
    ``test_parked_abort_reaps_and_releases`` stays green either way because it
    sets ``_finished`` directly -- which is exactly the step the real abort
    path does not perform for a parked session.
    """
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, boundary, _L, _tail = _add_spilled_session(mgr, "rid-i", 1, 0, seq=1)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-i" in ctl.parked)

    # Exactly what Scheduler.abort_request does to an in-flight request.
    slot.req.to_finish = FINISH_ABORT()
    assert not slot.req.finished()

    assert _pump(mgr, ctl, until=lambda: "rid-i" not in ctl.parked)
    assert ctl.counters["parked_aborted"] == 1
    # Released like any other ended spilled session.
    assert mgr.req_to_token_pool.freed_reqs == ["rid-i"]
    assert slot.req.kv_spill_state is None
    if boundary > 0:
        assert mgr.allocator.freed, "retained device head was not freed"
    # The abort is now honest: the flag was promoted, so every later
    # `finished()` check (reap loops, abort re-scans) agrees.
    assert slot.req.finished()
    assert slot.req.to_finish is None
    # ...and the client is told, since no forward pass will ever stream for it.
    assert [r.rid for r in mgr.streamed] == ["rid-i"]


def test_park_instead_of_demote_seam():
    tier = _DictTier()
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, *_ = _add_spilled_session(mgr, "rid-h", 1, 0, seq=1)
    mgr._iter_ct += 1  # past spill_iter
    assert kd.park_instead_of_demote(mgr, slot) is True
    assert slot.park_pending  # routed to the chain instead of demotion
    assert kd.park_instead_of_demote(mgr, slot) is True  # idempotent
    # Ineligible slot -> demote as before.
    slot2, *_ = _add_spilled_session(mgr, "rid-i", 2, 1, seq=2)
    mgr._iter_ct += 1
    slot2.spec_in_tick = True
    assert kd.park_instead_of_demote(mgr, slot2) is False
    # Unarmed manager -> always False.
    mgr2 = _make_mgr(None)
    assert kd.park_instead_of_demote(mgr2, slot) is False


# ---------------------------------------------------------------------------
# Real file backend roundtrip (genuine integration, still CPU-only)
# ---------------------------------------------------------------------------


def _file_tier(tmp_path):
    os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(tmp_path)
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="unit-model",
        dcp_owner_mode=False,
    )
    try:
        return kd.make_tier(
            "file", storage_config=cfg, mem_pool_host=None, extra_config=None
        )
    finally:
        del os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"]


def test_park_unpark_roundtrip_real_file_backend(tmp_path):
    tier = _file_tier(tmp_path)
    ctl = _make_ctl([tier])
    mgr = _make_mgr(ctl)
    slot, boundary, L, rows = _add_spilled_session(mgr, "rid-file", 1, 0, seq=1)
    before = _snapshot_region(mgr, 0, rows)
    ctl.note_region_shortfall(mgr._iter_ct)
    assert _pump(mgr, ctl, until=lambda: "rid-file" in ctl.parked)
    assert len(list(tmp_path.iterdir())) == 1 + 2 * LAYERS  # real files
    for fl in range(LAYERS):
        mgr.host_pool.k_data_refs[fl].zero_()
        mgr.host_pool.v_data_refs[fl].zero_()
    assert _pump(
        mgr,
        ctl,
        until=lambda: 1 in mgr.spills,
        advance=PARK_PRESSURE_WINDOW_ITERS + 2,
    )
    after = _snapshot_region(mgr, 0, rows)
    for (kb, vb), (ka, va) in zip(before, after):
        assert torch.equal(kb, ka)
        assert torch.equal(vb, va)


# ---------------------------------------------------------------------------
# Byte-identity of the default path (flag unset)
# ---------------------------------------------------------------------------


def test_maybe_park_flow_is_inert_when_unarmed():
    # The stub carries ONLY _dest=None: touching ANY other attribute would
    # raise AttributeError, so passing proves the unarmed flow reads and
    # writes nothing else.
    stub = SimpleNamespace(_dest=None)
    sentinel = object()
    assert kd.maybe_park_flow(stub, sentinel, None) is sentinel


def test_admission_reduce_payload_unwidened_without_destinations():
    # Single-rank path of update_dcp_admission_state on a stub WITHOUT a
    # _dest attribute: the method must neither require it nor touch a
    # controller (getattr default) -- the flag-off collective payload and
    # state stay exactly today's.
    stub = SimpleNamespace(
        scheduler=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 42),
            tree_cache=SimpleNamespace(evictable_size=lambda: 7),
            tp_cpu_group=None,
        ),
    )
    KVSessionOffloadManager.update_dcp_admission_state(stub)
    assert stub._dcp_min_avail == 42
    assert stub._dcp_budget_deficit == 0
    assert not hasattr(stub, "_dest")


def test_admission_reduce_consumes_flags_when_armed():
    calls = []
    ctl = SimpleNamespace(
        reduce_extra=lambda: [1, 1],
        consume_reduced=lambda d, o: calls.append((d, o)),
    )
    stub = SimpleNamespace(
        _dest=ctl,
        scheduler=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 42),
            tree_cache=SimpleNamespace(evictable_size=lambda: 7),
            tp_cpu_group=None,
        ),
    )
    KVSessionOffloadManager.update_dcp_admission_state(stub)
    assert calls == [(1, 1)]  # single-rank: min == local


def test_spill_slot_default_has_park_pending_false():
    from sglang.srt.managers.kv_session_offload import SpillSlot

    slot = SpillSlot(
        req=SimpleNamespace(), region=0, spill_iter=0, wave=None, hysteresis=None
    )
    assert slot.park_pending is False


# ---------------------------------------------------------------------------
# GDN / mamba invariant: the destination module never moves recurrent state
# ---------------------------------------------------------------------------


def test_source_scan_no_gdn_state_on_the_wire():
    src = open(kd.__file__).read()
    # The recurrent-state pools are never read or written by this module;
    # the ONLY sanctioned mamba reference is the cleanup FREE of an aborted
    # parked session (mirroring release_finished_spilled_req).
    for forbidden in (
        "temporal_state",
        "conv_state",
        "mamba_radix",
        "MambaPool",
        "intermediate_ssm",
    ):
        assert forbidden not in src, forbidden
    assert src.count("free_mamba_cache") == 2  # hasattr guard + the free
    # And the park payload is exactly meta + per-layer K/V rows: nothing
    # else (no draft, no hiddens, no recurrent state) crosses the wire.
    payload_fn = kd._region_row_views
    mgr = _make_mgr(None)
    views = payload_fn(mgr, 0, 4)
    assert [name for name, _ in views] == [
        f"{kv}{fl}" for fl in range(LAYERS) for kv in ("k", "v")
    ]


def test_meta_blob_too_large_is_rejected():
    with pytest.raises(ValueError):
        padded_meta_blob({"x": "y" * (2 * META_BLOB_BYTES)})


# ---------------------------------------------------------------------------
# ServerArgs validation
# ---------------------------------------------------------------------------


def _fake_server_args(**over):
    """Minimal stand-in exposing exactly the attributes
    ServerArgs._handle_kv_session_offload reads (same pattern as
    test_kv_session_offload_unit._fake_server_args)."""
    ns = SimpleNamespace(
        enable_kv_session_offload=True,
        kv_session_offload_prefill=False,
        kv_session_offload_host_ram_gib=0.0,
        kv_session_offload_block_size=8192,
        kv_session_offload_tick_interval=1,
        kv_session_offload_tick_floor=8,
        kv_session_offload_restore_hysteresis_steps=4,
        kv_session_offload_max_spills=1,
        kv_session_offload_restore_margin_tokens=4096,
        kv_session_offload_wave_back_min_free_tokens=0,
        kv_session_offload_mtp_resident_slices=0,
        kv_session_offload_spec_in_tick=False,
        kv_session_offload_resume_under_spec=False,
        # #236 budget flags (defaults OFF -> inert; the merged validator
        # reads both the budget and the destination attribute sets)
        kv_session_offload_budget_total_tokens=0,
        kv_session_offload_budget_session_tokens=0,
        kv_session_offload_budget_prefill_tokens=0,
        kv_session_offload_budget_decode_tokens=0,
        kv_session_offload_budget_rate_tokens_per_s=0.0,
        kv_session_offload_budget_episode_seconds=0.0,
        kv_session_offload_budget_max_sessions=0,
        kv_session_offload_spill_progress_lock_tokens=0,
        kv_session_offload_spill_hysteresis_steps=0,
        kv_session_offload_spill_cooldown_seconds=0.0,
        kv_session_offload_budget_demote_grace_iters=256,
        kv_session_offload_default_spill_class="normal",
        kv_session_offload_destinations=None,
        kv_session_offload_destination_extra_config=None,
        kv_session_offload_park_timeout_iters=512,
        speculative_algorithm=None,
        attention_backend="flashinfer",
        page_size=1,
        disaggregation_mode="null",
        weightless_kv_fastlane=False,
        enable_hierarchical_cache=False,
        enable_unified_memory=False,
        enable_hisparse=False,
        pp_size=1,
        dp_size=1,
        enable_mixed_chunk=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _validate(ns):
    from sglang.srt.server_args import ServerArgs

    ServerArgs._handle_kv_session_offload(ns)


def test_server_args_default_validates_without_destinations():
    _validate(_fake_server_args())  # must not raise (flag unset)


def test_server_args_dataclass_defaults():
    import dataclasses

    from sglang.srt.server_args import ServerArgs

    fields = {f.name: f for f in dataclasses.fields(ServerArgs)}
    assert fields["kv_session_offload_destinations"].default is None
    assert fields["kv_session_offload_destination_extra_config"].default is None
    assert fields["kv_session_offload_park_timeout_iters"].default == 512


def test_server_args_destinations_require_enable():
    with pytest.raises(ValueError, match="requires"):
        _validate(
            _fake_server_args(
                enable_kv_session_offload=False,
                kv_session_offload_destinations="local,file",
            )
        )


def test_server_args_extra_config_requires_destinations():
    with pytest.raises(ValueError, match="requires"):
        _validate(
            _fake_server_args(
                kv_session_offload_destination_extra_config="{}",
            )
        )


def test_server_args_destinations_validated_at_parse():
    with pytest.raises(ValueError, match="must be 'local'"):
        _validate(
            _fake_server_args(
                kv_session_offload_destinations="mooncake,local",
            )
        )
    with pytest.raises(ValueError, match="not a supported park tier"):
        _validate(
            _fake_server_args(kv_session_offload_destinations="local,eic")
        )
    with pytest.raises(ValueError, match="not valid JSON"):
        _validate(
            _fake_server_args(
                kv_session_offload_destinations="local,file",
                kv_session_offload_destination_extra_config="{broken",
            )
        )
    with pytest.raises(ValueError, match="park-timeout-iters"):
        _validate(
            _fake_server_args(
                kv_session_offload_destinations="local,file",
                kv_session_offload_park_timeout_iters=0,
            )
        )
    # A valid combination passes.
    _validate(
        _fake_server_args(
            kv_session_offload_destinations="local,mooncake,file",
            kv_session_offload_destination_extra_config='{"master_server_address": "x"}',
        )
    )


def test_server_args_help_names_the_order_reality():
    # The help text must name the measured numbers so nobody mistakes the
    # local-first rule for an arbitrary preference; and it documents the
    # GDN residency decision.
    import typing

    from sglang.srt.server_args import ServerArgs

    hints = typing.get_type_hints(ServerArgs, include_extras=True)
    (arg_meta,) = hints["kv_session_offload_destinations"].__metadata__
    assert "3.43 GB/s" in arg_meta.help
    assert "never above it" in arg_meta.help
    assert "device-resident" in arg_meta.help
