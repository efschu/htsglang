"""HiCache write->read roundtrip across the phase flip, hermetic (0828).

THE SPECIMEN THIS REPRODUCES (boot_943bx_0caab9a0c5..._0828_091012.log):

    09:13:55  #706 canonical KV page active: slots [8, 12) of 16   (attach, PP)
    09:14:10  [#719 hicache-rebind] rebound 3 reader(s) to the 'tp' pools
    09:25:23  Canonical read refused for 36161eb0...: read target holds 32768
              bytes but this KV page window is 8192 bytes.          (x128)
    09:25:23  HiCache prefetch success ... completed=0 matched=0 loaded=0
    09:25:31  [#928 anchor] REFUSING resume: node carries no recurrent state
              -> full re-prefill (a Kein-Doppel-Prefill violation).

ROOT (proven here red-first): the #706 canonical windows are built ONCE at
storage attach (`cache_controller._generate_storage_config`) from the pools
bound at that moment, and the #719 cutover rebind swaps every reader's pools
without ever re-deriving the windows.  After the first pp_to_tp rebind the
read buffer comes from the CURRENT (full-width) pool while the store cuts
with the FROZEN (stage) window, and `read_extents` refuses deterministically.
The store was healthy the whole time (#872): the write side ran under the
attach-time binding, where window and pool still agreed.

Every arm below drives the REAL method bodies (borrowed, never imitated --
the 869b pattern): `HiCacheFile` against a real tmp store,
`canonical_page_store.write_extents/read_extents`, the controller's
`_generic_page_set/_generic_page_get/_page_transfer`, and the REAL
`hicache_phase_binding.rebind` / `rebind_for_cutover` seam with #719
generation stamps.  CPU-only, no server, no GPU.
"""

import os
import tempfile
import unittest
from queue import Queue
from types import SimpleNamespace

import torch

import sglang

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache import hicache_phase_binding as binding
from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    build_mamba_window,
    window_for_layers,
)
from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.mamba_ckpt_utils import is_resume_candidate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# ----------------------------------------------------------------------------
# Geometry: the deployment's shape at test scale.  16 attention layers of 64,
# PP stage cut 7/5/4 slots; 12 GDN layers cut 5/4/3; TP head vector 2:1:1
# (the 32,16,16 flip vector's shape).
# ----------------------------------------------------------------------------
ATTN_LAYER_IDS = list(range(3, 64, 4))  # 16 ids
CELL = 64
KV_SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_ATTN_CUT = [(0, 28), (28, 48), (48, 64)]  # stage layer bounds -> 7/5/4 slots

MAMBA_LAYER_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]  # 12 global ids
MAMBA_SPEC = MambaBlobSpec(
    num_layers=12,
    num_heads=8,
    head_dim=4,
    state_size=2,
    conv_dim=24,
    conv_width=3,
    key_dim=8,
    value_dim=8,
    units=4,
    temporal_itemsize=1,
    conv_itemsize=1,
)
PP_MAMBA_CUT = [(0, 5), (5, 9), (9, 12)]  # positions in MAMBA_LAYER_IDS
TP_VECTOR = "2,1,1"
TP_RATIOS = [2, 1, 1]
IDENTITY = "0123456789abcdef"
KEYS = ["cafe01", "cafe02", "cafe03"]


def _stage_attn_ids(stage):
    lo, hi = PP_ATTN_CUT[stage]
    return [i for i in ATTN_LAYER_IDS if lo <= i < hi]


def _stage_mamba_ids(stage):
    lo, hi = PP_MAMBA_CUT[stage]
    return MAMBA_LAYER_IDS[lo:hi]


def _kv_page_bytes(tag):
    """The full canonical page, provenance-tagged per slot."""
    buf = bytearray()
    for slot in range(KV_SPEC.num_attn_layers):
        buf += bytes([(tag + slot) % 256]) * CELL
    return bytes(buf)


def _full_mamba_blob():
    buf = bytearray()
    for layer in range(MAMBA_SPEC.num_layers):
        buf += bytes([(100 + layer) % 256]) * MAMBA_SPEC.temporal_layer_bytes
    for layer in range(MAMBA_SPEC.num_layers):
        buf += bytes([(200 + layer) % 256]) * MAMBA_SPEC.conv_layer_bytes
    return bytes(buf)


MAMBA_BLOB = _full_mamba_blob()


def _window_payload(full_bytes, window):
    """The bytes ``window`` names inside the full canonical blob/page."""
    return torch.frombuffer(
        b"".join(full_bytes[off : off + length] for off, length in window.extents),
        dtype=torch.uint8,
    ).clone()


def _kv_window_for_stage(stage):
    return window_for_layers(KV_SPEC, ATTN_LAYER_IDS, _stage_attn_ids(stage))


def _kv_window_whole():
    return window_for_layers(KV_SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)


def _mamba_window_for_stage(stage):
    lo, hi = PP_MAMBA_CUT[stage]
    return build_mamba_window(
        MAMBA_SPEC, ratios=[1], rank=0, layer_lo=lo, layer_hi=hi
    )


def _mamba_window_for_tp_rank(rank):
    return build_mamba_window(
        MAMBA_SPEC,
        ratios=TP_RATIOS,
        rank=rank,
        layer_lo=0,
        layer_hi=MAMBA_SPEC.num_layers,
    )


# ----------------------------------------------------------------------------
# Pools: the minimum around the REAL storage/controller/binding methods.
# ----------------------------------------------------------------------------
class _KvHostPool:
    """A host KV pool of one phase: page = this phase's flat slot bytes."""

    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.page_size = 1
        self.size = 64
        self.layer_num = n_slots
        self.size_per_token = n_slots * CELL
        self.pages = {}

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.n_slots * CELL, dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]

    def set_from_flat_data_page(self, index, data_page):
        self.pages[int(index)] = data_page.clone()


class _MambaHostPool:
    """A mamba host pool of one phase: page = this rank's shard bytes."""

    def __init__(self, payload_bytes):
        self.page_size = 1
        self.size = 64
        self.size_per_token = payload_bytes
        self.pages = {}

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.size_per_token, dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]

    def set_from_flat_data_page(self, index, data_page):
        if int(data_page.numel()) != self.size_per_token:
            raise AssertionError(
                f"mamba host page is {self.size_per_token} B, got "
                f"{int(data_page.numel())} B"
            )
        self.pages[int(index)] = data_page.clone()


class _HostGroup:
    """The HostPoolGroup surface the controller and the rebind touch."""

    def __init__(self, kv, mamba):
        self.kv = kv
        self.mamba = mamba
        self.entry_map = {PoolName.KV: kv, PoolName.MAMBA: mamba}
        self.entries = [
            SimpleNamespace(name=PoolName.KV, host_pool=kv),
            SimpleNamespace(name=PoolName.MAMBA, host_pool=mamba),
        ]
        self.page_size = 1
        self.size = kv.size

    @property
    def layer_num(self):
        return self.kv.layer_num

    def get_pool(self, name):
        return self.entry_map[name]

    def get_dummy_flat_data_page(self):
        return self.kv.get_dummy_flat_data_page()

    def get_data_page(self, index, flat=True):
        return self.kv.get_data_page(index, flat)

    def set_from_flat_data_page(self, index, data_page):
        return self.kv.set_from_flat_data_page(index, data_page)


class _MambaDevPool:
    def __init__(self, mamba_ids):
        self.mamba_map = {int(i): n for n, i in enumerate(mamba_ids)}


class _DevPool:
    """The wrapped hybrid device pool of one phase (global layer ids)."""

    def __init__(self, attn_ids, mamba_ids):
        self.full_attention_layer_id_mapping = {
            int(i): n for n, i in enumerate(attn_ids)
        }
        self.layer_num = len(attn_ids)
        self.mamba_pool = _MambaDevPool(mamba_ids)
        # phase_pools_for unwraps via full_kv_pool: the inner pool is what
        # check_shapes compares.
        self.full_kv_pool = SimpleNamespace(layer_num=len(attn_ids))


def _pp_world(stage):
    dev = _DevPool(_stage_attn_ids(stage), _stage_mamba_ids(stage))
    host = _HostGroup(
        _KvHostPool(len(_stage_attn_ids(stage))),
        _MambaHostPool(_mamba_window_for_stage(stage).payload_bytes),
    )
    return dev, host


def _tp_world(rank):
    dev = _DevPool(ATTN_LAYER_IDS, MAMBA_LAYER_IDS)
    host = _HostGroup(
        _KvHostPool(len(ATTN_LAYER_IDS)),
        _MambaHostPool(_mamba_window_for_tp_rank(rank).payload_bytes),
    )
    return dev, host


def _config(kv_window, mamba_window):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="Qwen-roundtrip",
        model_identity_hash=IDENTITY,
        canonical_kv_page=kv_window,
        canonical_mamba_blob=mamba_window,
    )


class _Probe:
    """Carries the REAL controller methods over the minimum state they read.

    Borrowed, not imitated (the 869b pattern): a regression in the shipped
    bodies fails here rather than in a stub that quietly kept agreeing.
    """

    _generic_page_get = HiCacheController._generic_page_get
    _generic_page_set = HiCacheController._generic_page_set
    _page_transfer = HiCacheController._page_transfer
    store_presence_pages = HiCacheController.store_presence_pages
    # The fix under test: present only after the fix lands.
    if hasattr(HiCacheController, "rebind_canonical_windows"):
        rebind_canonical_windows = HiCacheController.rebind_canonical_windows
        _canonical_mamba_window = HiCacheController._canonical_mamba_window

    def __init__(self, backend, dev, host, server_args, pp_rank):
        self.storage_backend = backend
        self.enable_storage = True
        self.mem_pool_device_hybrid = dev
        self.mem_pool_host = host
        self.page_size = 1
        self.storage_config = backend_config_of(backend)
        self.host_mem_release_queue = Queue()
        self.prefetch_tokens_occupied = 0
        # attach-time canonical constants (what _generate_storage_config
        # stashes for the rebuild)
        self._canonical_server_args = server_args
        self._canonical_model_config = None
        self._canonical_attn_layer_ids = list(ATTN_LAYER_IDS)
        self._canonical_mamba_spec = MAMBA_SPEC
        self._canonical_mamba_layer_ids = list(MAMBA_LAYER_IDS)
        self.pp_rank = pp_rank
        self.tp_rank = 0
        self.tp_size = 1
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

    def draft_tier_armed(self, direction):
        return False

    def append_host_mem_release(self, host_indices, generation=None):
        pass

    def get_hash_str(self, token_ids, last_hash, page_size=None):
        return list(KEYS)

    def _presence_pool_transfers(self):
        return HiCacheController._presence_pool_transfers(self)


def backend_config_of(backend):
    return _config(backend.canonical_kv_page, backend.canonical_mamba_blob)


def _server_args():
    return SimpleNamespace(
        phase_flip_canonical_kv_page=True,
        phase_flip_rebind_hicache=True,
        phase_flip_tp_vector=TP_VECTOR,
        rank_tp_ratio=None,
    )


def _backend(root, kv_window, mamba_window):
    be = HiCacheFile(_config(kv_window, mamba_window), file_path=root)
    be.file_path = root
    return be


def _write_prefix_from_pp(root):
    """All three PP stages persist KV+mamba through the REAL write chain."""
    for stage in range(3):
        dev, host = _pp_world(stage)
        be = _backend(root, _kv_window_for_stage(stage), _mamba_window_for_stage(stage))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=stage)
        kv_w = be.canonical_kv_page
        for i, key in enumerate(KEYS):
            host.kv.pages[i] = _window_payload(_kv_page_bytes(10), kv_w.as_extents())
            ok = probe._generic_page_set([key], torch.tensor([i]), None)
            assert ok, f"stage {stage} failed to persist KV page {key}"
        # Mamba through the real v2 chain: registered pool -> _write_page.
        be.register_mem_host_pool_v2(host.mamba, PoolName.MAMBA)
        for i, key in enumerate(KEYS):
            host.mamba.pages[i] = _window_payload(
                MAMBA_BLOB, be.canonical_mamba_blob
            )
            res = be.batch_set_v2(
                [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        keys=[key],
                        host_indices=torch.tensor([i]),
                    )
                ]
            )
            assert res[PoolName.MAMBA] == [True]


def _write_prefix_from_tp(root):
    """The other direction: TP ranks persist whole KV pages + mamba shards."""
    for rank in range(3):
        dev, host = _tp_world(rank)
        be = _backend(root, _kv_window_whole(), _mamba_window_for_tp_rank(rank))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=rank)
        if rank == 0:  # whole pages: one writer suffices, others no-op
            for i, key in enumerate(KEYS):
                host.kv.pages[i] = _window_payload(
                    _kv_page_bytes(10), be.canonical_kv_page.as_extents()
                )
                assert probe._generic_page_set([key], torch.tensor([i]), None)
        be.register_mem_host_pool_v2(host.mamba, PoolName.MAMBA)
        for i, key in enumerate(KEYS):
            host.mamba.pages[i] = _window_payload(
                MAMBA_BLOB, be.canonical_mamba_blob
            )
            res = be.batch_set_v2(
                [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        keys=[key],
                        host_indices=torch.tensor([i]),
                    )
                ]
            )
            assert res[PoolName.MAMBA] == [True]


def _prefetch_op(n_keys):
    op = PrefetchOperation(
        "rid-roundtrip",
        torch.arange(n_keys, dtype=torch.int64),
        list(range(n_keys)),
        None,
    )
    op.hash_value = list(KEYS[:n_keys])
    return op


def _phase_pools(phase, dev, host):
    """PhasePools for a direct rebind, pin-compatible.

    On the pin `PhasePools` has no `device_pool_hybrid` field yet; passing the
    WRAPPED pool as `device_pool` keeps the red arm the SPECIMEN (a refusing
    read) rather than a TypeError about a field the fix introduces."""
    kwargs = dict(
        phase=phase, device_pool=dev, host_pool=host, allocator=object()
    )
    if "device_pool_hybrid" in getattr(
        binding.PhasePools, "__dataclass_fields__", {}
    ):
        kwargs["device_pool_hybrid"] = dev
    return binding.PhasePools(**kwargs)


def _fake_scheduler(probe, incoming_dev, incoming_host):
    tree = SimpleNamespace(cache_controller=probe)
    sched = SimpleNamespace(
        server_args=probe._canonical_server_args,
        tree_cache=tree,
        token_to_kv_pool_allocator=object(),
        phase_flip_stacks=SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    token_to_kv_pool=incoming_dev,
                    token_to_kv_pool_allocator=object(),
                )
            )
        ),
        phase_flip_host_pools={"tp": incoming_host, "pp": probe.mem_pool_host},
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=incoming_dev,
                token_to_kv_pool_allocator=object(),
            )
        ),
    )
    return sched


class _Base(CustomTestCase):
    def setUp(self):
        binding.binding_state().reset()
        self.addCleanup(binding.binding_state().reset)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)


class TestSpecimenIsReproduced(_Base):
    """The pin's failure, exact form: stale window + rebound pool -> refusal.

    This arm PASSES on the pin and after the fix alike: it pins the guard's
    behaviour for the hazard direction (a stale window must refuse loudly,
    never reinterpret).  The fix makes this state unreachable via the seam --
    which the roundtrip arms below prove -- without ever weakening the guard.
    """

    def test_stale_stage_window_against_full_width_pool_refuses(self):
        _write_prefix_from_pp(self.root)
        # PP1's attach-time identity ...
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        # ... rebound to the TP pools by the REAL #719 stamp, windows frozen.
        tp_dev, tp_host = _tp_world(1)
        binding.rebind(
            {"cache_controller": probe},
            _phase_pools("tp", tp_dev, tp_host),
        )
        op = _prefetch_op(len(KEYS))
        expected_refusal = (
            f"read target holds {len(ATTN_LAYER_IDS) * CELL} bytes but this "
            f"KV page window is "
            f"{_kv_window_for_stage(1).byte_length} bytes"
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.hicache_storage", level="ERROR"
        ) as logs:
            probe._page_transfer(op)
        self.assertTrue(
            any(expected_refusal in line for line in logs.output),
            f"expected the specimen refusal {expected_refusal!r} in "
            f"{logs.output}",
        )
        self.assertEqual(op.completed_tokens, 0)


class TestRoundtripPpWritesTpReads(_Base):
    """The specimen's direction, green through the REAL cutover seam."""

    def test_roundtrip(self):
        _write_prefix_from_pp(self.root)
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        tp_dev, tp_host = _tp_world(1)
        sched = _fake_scheduler(probe, tp_dev, tp_host)
        generation = binding.rebind_for_cutover(sched, "tp")
        self.assertEqual(generation, 1)

        # The windows must now be the TP phase's, not the attach phase's.
        self.assertTrue(be.canonical_kv_page.is_whole_page)
        self.assertEqual(
            be.canonical_mamba_blob.payload_bytes,
            _mamba_window_for_tp_rank(1).payload_bytes,
        )

        # KV: the real transfer chain, byte-compared.
        op = _prefetch_op(len(KEYS))
        probe._page_transfer(op)
        self.assertEqual(op.completed_tokens, len(KEYS))
        whole = _kv_window_whole().as_extents()
        for i in range(len(KEYS)):
            self.assertTrue(
                torch.equal(
                    tp_host.kv.pages[i], _window_payload(_kv_page_bytes(10), whole)
                ),
                f"KV page {i} did not survive the roundtrip byte-identically",
            )

        # Mamba: the real v2 chain into the CURRENT phase's pool.
        res = be.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=[KEYS[-1]],
                    host_indices=torch.tensor([0]),
                )
            ]
        )
        self.assertEqual(res[PoolName.MAMBA], [True])
        self.assertTrue(
            torch.equal(
                tp_host.mamba.pages[0],
                _window_payload(MAMBA_BLOB, _mamba_window_for_tp_rank(1)),
            ),
            "mamba shard did not survive the roundtrip byte-identically",
        )

        # Token accounting: the issuance gate must promise the full prefix ...
        self.assertEqual(
            probe.store_presence_pages(list(range(len(KEYS))), None), len(KEYS)
        )
        # ... and the anchor rule accepts a host-backed recurrent state, which
        # is what ends the #928 REFUSING-resume -> re-prefill chain.
        self.assertTrue(
            is_resume_candidate(
                len(KEYS),
                None,
                has_device_value=False,
                has_host_value=True,
                device_only=False,
            )
        )


class TestRoundtripTpWritesPpReads(_Base):
    """The reverse leg: decode-phase writes, prefill-phase reads."""

    def test_roundtrip(self):
        _write_prefix_from_tp(self.root)
        dev, host = _tp_world(1)
        be = _backend(self.root, _kv_window_whole(), _mamba_window_for_tp_rank(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        pp_dev, pp_host = _pp_world(1)
        sched = _fake_scheduler(probe, pp_dev, pp_host)
        sched.phase_flip_host_pools["pp"] = pp_host
        sched.tp_worker.model_runner.token_to_kv_pool = pp_dev
        generation = binding.rebind_for_cutover(sched, "pp")
        self.assertEqual(generation, 1)

        self.assertEqual(
            be.canonical_kv_page.byte_length,
            _kv_window_for_stage(1).byte_length,
        )
        op = _prefetch_op(len(KEYS))
        probe._page_transfer(op)
        self.assertEqual(op.completed_tokens, len(KEYS))
        stage_w = _kv_window_for_stage(1).as_extents()
        for i in range(len(KEYS)):
            self.assertTrue(
                torch.equal(
                    pp_host.kv.pages[i],
                    _window_payload(_kv_page_bytes(10), stage_w),
                )
            )
        res = be.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=[KEYS[-1]],
                    host_indices=torch.tensor([0]),
                )
            ]
        )
        self.assertEqual(res[PoolName.MAMBA], [True])
        self.assertTrue(
            torch.equal(
                pp_host.mamba.pages[0],
                _window_payload(MAMBA_BLOB, _mamba_window_for_stage(1)),
            )
        )


class TestProbeHonesty(_Base):
    """Fix 2: presence and readability may never drift apart again.

    On the pin `batch_exists_v2` promised pages the reader could not serve
    (the issuance saw hits, the fetch refused, the match collapsed to 0 and
    the anchor re-prefilled EVERYTHING).  A store the current window cannot
    cut must answer 0 -- an honest cold miss, bounded loss."""

    def test_stale_window_probe_answers_zero_not_a_promise(self):
        _write_prefix_from_pp(self.root)
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        tp_dev, tp_host = _tp_world(1)
        binding.rebind(
            {"cache_controller": probe},
            _phase_pools("tp", tp_dev, tp_host),
        )
        # The stale-window state: pool moved, windows frozen, pools registered.
        be.register_mem_pool_host(tp_host)
        be.register_mem_host_pool_v2(tp_host.mamba, PoolName.MAMBA)
        result = be.batch_exists_v2(KEYS, None)
        self.assertEqual(
            result.kv_hit_pages,
            0,
            "a store the current window cannot cut must not be promised",
        )

    def test_matched_window_probe_still_promises(self):
        _write_prefix_from_pp(self.root)
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        be.register_mem_pool_host(host)
        be.register_mem_host_pool_v2(host.mamba, PoolName.MAMBA)
        result = be.batch_exists_v2(KEYS, None)
        self.assertEqual(result.kv_hit_pages, len(KEYS))


class TestWriteSideStaysGuarded(_Base):
    """The mutant on the danger direction: a stale-window WRITE must refuse.

    A refusal costs a cache miss; silence would deposit the first N bytes of a
    full-width page into another stage's slots -- provenance-wrong bytes under
    a content-addressed key, unfindable later."""

    def test_mismatched_write_is_refused_and_store_untouched(self):
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        tp_dev, tp_host = _tp_world(1)
        binding.rebind(
            {"cache_controller": probe},
            _phase_pools("tp", tp_dev, tp_host),
        )
        tp_host.kv.pages[0] = _window_payload(
            _kv_page_bytes(10), _kv_window_whole().as_extents()
        )
        ok = probe._generic_page_set([KEYS[0]], torch.tensor([0]), None)
        self.assertFalse(ok, "a stale-window write must refuse, not deposit")
        self.assertFalse(be.exists(KEYS[0]))


class TestSeamRebuildsWindows(_Base):
    """The wiring itself: rebind_for_cutover must rebuild the windows.

    RED on the pin: the seam swaps the pools and returns, the windows stay
    the attach phase's."""

    def test_cutover_rebuilds_both_windows(self):
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=1)
        tp_dev, tp_host = _tp_world(1)
        sched = _fake_scheduler(probe, tp_dev, tp_host)
        binding.rebind_for_cutover(sched, "tp")
        self.assertTrue(
            be.canonical_kv_page.is_whole_page,
            "the KV window is still the attach phase's after the cutover",
        )
        self.assertEqual(
            be.canonical_mamba_blob.payload_bytes,
            _mamba_window_for_tp_rank(1).payload_bytes,
            "the mamba window is still the attach phase's after the cutover",
        )
        # And back: the return leg restores the stage windows.
        sched2 = _fake_scheduler(probe, dev, host)
        sched2.phase_flip_host_pools["pp"] = host
        sched2.tp_worker.model_runner.token_to_kv_pool = dev
        binding.rebind_for_cutover(sched2, "pp")
        self.assertEqual(
            be.canonical_kv_page.byte_length,
            _kv_window_for_stage(1).byte_length,
        )


class TestWindowDerivationHasOneSource(_Base):
    """Sibling sweep (user objection, 0828): HiCache is three-tiered
    (VRAM radix -> host staging [#810] -> disk file store), and the specimen
    fired on the disk leg -- so the fix must not be disk-specific. This pins
    the property that MAKES it tier-complete: every path that reads or writes
    canonical bytes consumes the window from the ONE source the cutover
    rebuild swaps (the backend's installed windows), and the derivation of
    those windows lives in exactly ONE module (the controller, attach +
    rebuild). A second, tier-local derivation or extent consumer would be
    rebuild-blind and reintroduce the specimen on that tier; this test makes
    adding one a conscious decision instead of a silent fork.

    Structural pin (source scan), because a runtime arm per tier would cost a
    server: the host tier holds PHASE-LOCAL layouts and is swapped whole by
    the same #719 rebind (pool identity, not byte cut); the geometry-neutral
    cut exists only at the store boundary, and that boundary is what is
    pinned here. Verified against the tree at fix time: extent I/O only in
    hicache_storage.py, derivation only in cache_controller.py, the staging
    ring (staging_write_ring.py) carries no window logic of its own, and
    flip_writeback reads the backend attribute live rather than caching it.
    """

    SRT = os.path.join(os.path.dirname(sglang.__file__), "srt")

    def _files_referencing(self, needles, exclude):
        hits = set()
        for dirpath, _dirs, files in os.walk(self.SRT):
            for name in files:
                if not name.endswith(".py") or name in exclude:
                    continue
                try:
                    with open(
                        os.path.join(dirpath, name), encoding="utf-8"
                    ) as fh:
                        text = fh.read()
                except OSError:
                    continue
                if any(n in text for n in needles):
                    hits.add(name)
        return hits

    def test_extent_io_has_exactly_one_runtime_consumer(self):
        hits = self._files_referencing(
            ("read_extents", "read_slice(", "write_extents", "write_slice("),
            exclude={"canonical_page_store.py"},
        )
        self.assertEqual(
            hits,
            {"hicache_storage.py"},
            "a second consumer of the canonical extent I/O would read or "
            "write with a window the cutover rebuild does not swap -- wire "
            "it through HiCacheFile's installed windows (rebuilt by "
            "rebind_canonical_windows) or extend the rebuild to cover it "
            "before widening this pin",
        )

    def test_window_derivation_lives_only_in_the_controller(self):
        hits = self._files_referencing(
            ("build_page_window", "build_mamba_window", "window_for_layers"),
            exclude={"canonical_page_store.py"},
        )
        self.assertEqual(
            hits,
            {"cache_controller.py"},
            "a second derivation site would mint windows the cutover rebuild "
            "does not know about; derive in the controller (attach + "
            "rebind_canonical_windows) and install via "
            "install_canonical_windows instead",
        )


if __name__ == "__main__":
    unittest.main()


# ============================================================================
# THE MISSION ARM: PP3 writes -> flip -> TP3 reads a PARTIAL prefix.
# ============================================================================
#
# WHY THE ARMS ABOVE DO NOT ALREADY COVER THIS. They are green, and they are
# green honestly: they prove the STORE and the TRANSPORT return PP-written
# bytes to a TP-phase reader byte-identically through the real cutover seam.
# They stop exactly where the live defect starts, because every one of them
# reads the FULL prefix. On metal the mission has never once happened, and the
# measured refusal is not a transport failure:
#
#     verdict=refused reached=45 accepted=0 refused=45 refusers=MambaComponent:45
#
# 671 of 676 walks, MambaComponent every time, and (desk line, 2026-09-01)
# 95985 reason lines all `why=MambaComponent:absent`, `off_grid=0` across all
# 1338 logs. So the refusal predicate (mamba_ckpt_utils.py:126-131) is a REAL
# precondition measuring MISSING BYTES -- not geometry, not a deletion
# candidate. The bytes for the mamba half were never planted.
#
# WHERE THEY FAIL TO GET PLANTED, and the asymmetry that makes it invisible:
#
#   * attention plants unconditionally at the host insert
#     (unified_radix_cache.py:2232, `host_value = host_value.clone()`);
#   * mamba is planted at the same node only when FOUR terms hold
#     (mamba_component.py:1547-1552), the last being `loaded`, i.e.
#     `extra_pool_hit_pages[MAMBA] >= 1`;
#   * and that count is only ever written when the KV hit was TOTAL --
#     `hybrid_cache_controller.py:854-855`,
#     `if operation.pool_transfers and kv_completed_pages == len(hash_value)`.
#
# A PARTIAL prefix hit is the NORMAL case of a prefix match. It leaves the
# mamba pages unfetched, so `host_value` is never planted, so the next resume
# answers `absent` -- and refuses. A self-sustaining hole: the condition that
# would fill it is gated on an outcome it prevents.
#
# THE TWO ARMS MUST BE SEPARATED OR THE PROOF IS EMPTY (provenance axis). The
# same coupling is absent from `_page_backup` (:864-869), which fetches the
# extra pools unconditionally. That is the BACKUP_HOST arm -- one process
# writes and reads its own bytes -- and it reports green in 311 of 1338 boots.
# The PREFETCH arm (`_page_transfer`) is the ONLY path by which a PP3-produced
# prefix can reach a TP3 decode, and it has ZERO log lines. A test that drives
# `_page_backup`, or that does not say which arm it drove, proves the arm that
# was never in question. These drive `_page_transfer`.
#
# PRODUCER PHASE, and why it needs no new field. The key is deliberately
# geometry-neutral (#706/#555) so that PP-written bytes are TP-readable at
# all; stamping a phase into it would re-introduce the two-geometry key that
# HiCache-Phasen-Uniform calls a bug. Here provenance is established BY
# CONSTRUCTION instead: the store starts empty in `setUp`, and the only writer
# is the PP-phase chain. Any hit is therefore provably PP-produced. That is
# available to a hermetic test and NOT to the live census, which is why the
# census still needs a producer-phase axis of its own (see the report).


class _HybridProbe(HybridCacheController):
    """`_Probe` one class up, and a REAL subclass on purpose.

    `_Probe` borrows `HiCacheController._page_transfer` -- the BASE body,
    which knows nothing about extra pools. The coupling under test lives in
    the SUBCLASS override, and that body opens with
    `super()._page_transfer(operation)`. A borrowed function cannot resolve
    that (`super()` binds to the class the body was DEFINED in), so this
    inherits for real: the base half is then exactly the body the arms above
    already exercise, and the extra-pool half is the shipped one.

    `HybridCacheController.__init__` is deliberately NOT run -- it starts
    threads and queues. `_Probe.__init__` sets precisely the fields these two
    bodies read. Borrowed, never imitated (the 869b pattern).
    """

    __init__ = _Probe.__init__
    draft_tier_armed = _Probe.draft_tier_armed
    append_host_mem_release = _Probe.append_host_mem_release
    get_hash_str = _Probe.get_hash_str


def _write_pp_prefix_with_partial_kv(root, n_kv):
    """PP persists the mamba half for the WHOLE prefix but only ``n_kv`` KV
    pages -- the ordinary shape of a prefix that was cached to a shorter
    extent than it is now being matched against."""
    for stage in range(3):
        dev, host = _pp_world(stage)
        be = _backend(root, _kv_window_for_stage(stage), _mamba_window_for_stage(stage))
        probe = _Probe(be, dev, host, _server_args(), pp_rank=stage)
        kv_w = be.canonical_kv_page
        for i, key in enumerate(KEYS[:n_kv]):
            host.kv.pages[i] = _window_payload(_kv_page_bytes(10), kv_w.as_extents())
            assert probe._generic_page_set([key], torch.tensor([i]), None)
        be.register_mem_host_pool_v2(host.mamba, PoolName.MAMBA)
        for i, key in enumerate(KEYS):
            host.mamba.pages[i] = _window_payload(MAMBA_BLOB, be.canonical_mamba_blob)
            res = be.batch_set_v2(
                [PoolTransfer(name=PoolName.MAMBA, keys=[key], host_indices=torch.tensor([i]))]
            )
            assert res[PoolName.MAMBA] == [True]


def _mamba_transfer():
    """The mamba sidecar as the runtime forms it: one trailing page."""
    return PoolTransfer(
        name=PoolName.MAMBA,
        keys=[KEYS[-1]],
        host_indices=torch.tensor([0]),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )


def _hybrid_prefetch_op(n_keys, transfers):
    op = HybridPrefetchOperation(
        "rid-mission",
        torch.arange(n_keys, dtype=torch.int64),
        list(range(n_keys)),
        None,
        pool_transfers=transfers,
    )
    op.hash_value = list(KEYS[:n_keys])
    return op


class _MissionBase(_Base):
    def _tp_reader_after_flip(self):
        """PP wrote; the REAL #719 cutover rebinds to the TP phase; the probe
        that comes back is the TP-phase reader of PP-produced bytes."""
        dev, host = _pp_world(1)
        be = _backend(self.root, _kv_window_for_stage(1), _mamba_window_for_stage(1))
        probe = _HybridProbe(be, dev, host, _server_args(), pp_rank=1)
        tp_dev, tp_host = _tp_world(1)
        sched = _fake_scheduler(probe, tp_dev, tp_host)
        self.assertEqual(binding.rebind_for_cutover(sched, "tp"), 1)
        be.register_mem_host_pool_v2(tp_host.mamba, PoolName.MAMBA)
        return probe


class TestPartialPrefixHitPlantsTheMambaHalf(_MissionBase):
    """RED-FIRST, and red for the MEASURED reason: the mamba half is never
    fetched on a partial hit, so it can never be planted, so the next resume
    refuses `absent` -- which is what 95985 reason lines say."""

    def test_a_partial_prefix_hit_fetches_the_mamba_half(self):
        _write_pp_prefix_with_partial_kv(self.root, n_kv=2)
        probe = self._tp_reader_after_flip()
        op = _hybrid_prefetch_op(len(KEYS), [_mamba_transfer()])

        probe._page_transfer(op)

        # PREMISE CHECK, so the red below can never be vacuous: the partial
        # hit must actually have formed. If this fails the test is refuted,
        # not the code.
        self.assertEqual(
            op.completed_tokens,
            2,
            "the partial KV hit did not form -- this arm proves nothing "
            "about partial hits until it does (premise, not verdict)",
        )
        self.assertLess(op.completed_tokens // probe.page_size, len(op.hash_value))

        got = op.pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0)
        self.assertGreaterEqual(
            got,
            1,
            "THE MISSION FAILS HERE. The KV half of a PP3-written prefix "
            "arrived in the TP3 phase, but the mamba half was never fetched, "
            "because hybrid_cache_controller.py:855 asks for a TOTAL KV hit "
            "(kv_completed_pages == len(hash_value)) before it touches the "
            "extra pools -- and a partial hit is the ordinary shape of a "
            "prefix match. Unfetched means mamba_component.py:1547-1552 sees "
            "`loaded=False` and returns without planting host_value, while "
            "attention plants its own unconditionally at "
            "unified_radix_cache.py:2232. The next resume then answers "
            "`absent` and refuses: MambaComponent, 671/676 walks. The hole "
            "sustains itself -- the fetch that would fill it is gated on an "
            "outcome it prevents. NOTE the machinery for the short hit is "
            "already written and already called one line below the gate: "
            "_sync_trailing_keys re-aligns trailing sidecar keys 'when the "
            "storage hit is shorter than the original target prefix', which "
            "inside `== len(hash_value)` can never be the case.",
        )


class TestTheFullHitControlArm(_MissionBase):
    """The denominator for the arm above: with a TOTAL hit the same chain DOES
    fetch the mamba half. Green today. Without it a red above could equally
    mean 'this harness cannot fetch mamba at all', which would prove nothing
    about partiality."""

    def test_a_total_hit_does_fetch_the_mamba_half(self):
        _write_pp_prefix_with_partial_kv(self.root, n_kv=len(KEYS))
        probe = self._tp_reader_after_flip()
        op = _hybrid_prefetch_op(len(KEYS), [_mamba_transfer()])

        probe._page_transfer(op)

        self.assertEqual(op.completed_tokens, len(KEYS))
        self.assertGreaterEqual(
            op.pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0),
            1,
            "even a TOTAL hit did not fetch the mamba half -- then the red "
            "arm above is about this harness, not about partial hits",
        )


class TestTheTwoArmsAreNotTheSameArm(_MissionBase):
    """PROVENANCE AXIS, structural. The prefetch arm and the backup arm differ
    at exactly this coupling, and only the prefetch arm can carry a PP3 prefix
    to a TP3 decode. Conflating them is how a green BACKUP_HOST number
    (311/1338 boots) gets read as evidence for a path that has never once run
    (PREFETCH: zero log lines)."""

    def test_only_the_prefetch_arm_gates_the_extra_pools_on_a_total_hit(self):
        import inspect as _i

        transfer = _i.getsource(HybridCacheController._page_transfer)
        backup = _i.getsource(HybridCacheController._page_backup)

        self.assertIn(
            "kv_completed_pages == len(operation.hash_value)",
            transfer,
            "the prefetch arm no longer gates the extra pools on a total KV "
            "hit -- if that gate was removed, the red arm above is the fix "
            "and this test should be re-derived, not deleted",
        )
        self.assertNotIn(
            "kv_completed_pages",
            backup,
            "the backup arm has grown the prefetch arm's coupling: the arm "
            "that writes its own bytes in one process must not start "
            "depending on a total KV hit too",
        )

    def test_the_mission_path_is_the_prefetch_arm(self):
        """Named so a future reader cannot mistake which arm was proven: the
        producer here is the PP phase (the store starts empty in setUp and
        only the PP chain wrote), and the reader is the TP phase after the
        real cutover. That is the PREFETCH arm by construction."""
        _write_pp_prefix_with_partial_kv(self.root, n_kv=len(KEYS))
        probe = self._tp_reader_after_flip()
        self.assertEqual(
            binding.binding_state().phase,
            "tp",
            "the reader is not in the TP phase -- the arm under test is not "
            "the mission's arm",
        )
        self.assertIs(
            type(probe)._page_transfer,
            HybridCacheController._page_transfer,
            "the probe is not driving the shipped prefetch body",
        )
