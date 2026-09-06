"""Weg 2 S5: the L3 store as the SOLE carrier between two process groups.

Weg 2 boots two independent groups on the same three cards -- group P
(``pp_size=3, tp_size=1``, prefill) and group D (``tp_size=3, pp_size=1``,
decode) -- and nothing crosses at the flip except what group P already wrote
into the canonical page store. Everything pinned here is a way that carrier
silently delivers nothing:

* F1  -- the two groups must name the SAME bytes. With the format off, each
  group re-appends its own geometry to the KV key
  (``hicache_storage.py:_derive_key_suffixes``) and the key spaces are
  disjoint: a 100 % miss that never raises.
* F7  -- ``_is_storage_owner`` and ``writer_count`` key off ``is_mla_model``
  as a proxy for "do ranks write the same physical files". Under the canonical
  page a GQA model writes shared files too, so all six ranks become eviction
  owners over one directory and the operator's ONE cap is divided by 1 in P
  and by 3 in D. Both predicates are re-pointed at ``writes_shared_keys``.
* W8b -- the eviction index scans on ``config_suffix`` while the canonical
  pages carry ``kv_config_suffix``, so the cap is enforced against a fraction
  of the store (measured on the store of record: 83.7 % of files invisible).
* N2  -- ``touch()`` moved a key to MRU in memory only. The two groups' owners
  each rebuild their LRU from ``st_mtime``, a shared physical fact, so a touch
  that never reaches the inode is invisible to the sibling and the page it
  just served is the next one evicted.
* W7/W9 -- the two refusals that turn a silently empty carrier into a launch
  failure.

Hermetic: no CUDA, no model, no server. The canonical windows are constructed
directly from ``CanonicalPageSpec`` because that dataclass IS the layout
contract (equality is the contract; it carries nothing about the cut).
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import CanonicalPageWindow
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    compute_model_identity_hash,
)
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# One token of a 16-attention-layer hybrid at 2048 B per layer -- the shape of
# the line of record (spec 1.1: the per-token cell stays 2048 B per
# full-attention layer). Small enough to write in a unit test, real in form.
NUM_ATTN_LAYERS = 16
CELL_BYTES = 2048
SPEC = CanonicalPageSpec(
    num_attn_layers=NUM_ATTN_LAYERS,
    kv_bytes_per_token_per_attn_layer=CELL_BYTES,
)
# Group P's three PP stages own contiguous runs of the model's layers, hence
# contiguous runs of page slots.
P_STAGES = ((0, 6), (6, 5), (11, 5))
MODEL_NAME = "Qwen3.8-27B"
IDENTITY = "520526c68d6530e9"
KEY = "0" * 64


def _whole_page_window():
    return CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=NUM_ATTN_LAYERS)


def _stage_window(stage: int):
    first, count = P_STAGES[stage]
    return CanonicalPageWindow(spec=SPEC, first_slot=first, num_slots=count)


def _p_config(stage: int, *, canonical: bool = True):
    """Group P, rank ``stage``: pp_size=3, tp_size=1."""
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=stage,
        pp_size=3,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=IDENTITY,
        canonical_kv_page=_stage_window(stage) if canonical else None,
    )


def _d_config(tp_rank: int, *, canonical: bool = True):
    """Group D, rank ``tp_rank``: tp_size=3, pp_size=1."""
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=3,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=IDENTITY,
        canonical_kv_page=_whole_page_window() if canonical else None,
    )


def _payload(stage: int) -> torch.Tensor:
    """Deterministic per-stage bytes, so a wrong offset is a wrong value."""
    _, count = P_STAGES[stage]
    n = count * CELL_BYTES
    base = (stage + 1) * 37
    return torch.arange(base, base + n, dtype=torch.int64).to(torch.uint8)


def _expected_whole_page() -> torch.Tensor:
    out = torch.zeros(NUM_ATTN_LAYERS * CELL_BYTES, dtype=torch.uint8)
    for stage, (first, count) in enumerate(P_STAGES):
        lo = first * CELL_BYTES
        out[lo : lo + count * CELL_BYTES] = _payload(stage)
    return out


# --- the two-process acceptance worker -----------------------------------
#
# Two real OS processes, driven by subprocess and reporting through a FILE.
# Deliberately not a multiprocessing.Queue: the reader's payload is a whole
# page, and a child that writes more than the pipe buffer blocks forever while
# the parent sits in join() -- a hang, not a red test. The file carries a
# digest, so the byte-identity claim costs one line either way.


def _digest(buf: torch.Tensor) -> str:
    return hashlib.sha256(
        buf.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def worker_main() -> None:
    """Entry point of processes A and B. Never called in-process."""
    role = os.environ["WEG2_S5_ROLE"]
    store_dir = os.environ["WEG2_S5_DIR"]
    out = os.environ["WEG2_S5_OUT"]
    result = {}
    try:
        if role == "prefill":
            keys = []
            for stage in range(3):
                store = HiCacheFile(_p_config(stage), file_path=store_dir)
                keys.append(store._get_suffixed_key(KEY))
                store.set(KEY, _payload(stage))
            result = {"status": "ok", "keys": keys}
        else:
            store = HiCacheFile(_d_config(0), file_path=store_dir)
            target = torch.zeros(NUM_ATTN_LAYERS * CELL_BYTES, dtype=torch.uint8)
            got = store.get(KEY, target)
            result = {
                "status": "ok",
                "keys": [store._get_suffixed_key(KEY)],
                "hit": got is not None,
                "digest": _digest(target) if got is not None else None,
            }
    except Exception as e:  # noqa: BLE001 - the child's failure IS the result
        result = {"status": "err", "error": f"{type(e).__name__}: {e}"}
    with open(out, "w") as f:
        json.dump(result, f)


def _run_worker(role: str, store_dir: str, out_path: str) -> dict:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        WEG2_S5_ROLE=role,
        WEG2_S5_DIR=store_dir,
        WEG2_S5_OUT=out_path,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import test_weg2_s5_store_carrier as t; t.worker_main()" % here
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if not os.path.exists(out_path):
        raise AssertionError(
            f"{role} process produced no result "
            f"(rc={proc.returncode})\nstderr:\n{proc.stderr[-2000:]}"
        )
    with open(out_path) as f:
        return json.load(f)


def _group_server_args(**over):
    """A Weg-2 group launch, validated by ServerArgs itself.

    The base tree cannot build either shape: the format is gated on
    ``--enable-phase-flip``, which in turn demands ``pp_size > 1`` AND
    ``tp_size == 1`` -- so group D is unreachable and group P only reachable
    by arming the arena host ledger this design deletes.
    """
    from sglang.srt.server_args import ServerArgs

    defaults = dict(
        model_path="dummy",
        hicache_canonical_kv_page=True,
        page_size=1,
        hicache_storage_backend="file",
        kv_cache_dtype="fp8_e4m3",
    )
    defaults.update(over)
    args = ServerArgs(**defaults)
    args._handle_hicache_canonical_kv_page()
    return args


def _config_from_server_args(args, *, tp_rank, window):
    from sglang.srt.managers.cache_controller import canonical_identity_hash_for

    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=args.tp_size,
        pp_rank=0 if args.pp_size == 1 else tp_rank,
        pp_size=args.pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=canonical_identity_hash_for(
            args, args.hicache_canonical_kv_page
        ),
        canonical_kv_page=window,
    )


class TestTheKeyIsBuiltFromTheTwoGroupsServerArgs(CustomTestCase):
    """The spec's red-first (i), end to end from the launch arguments.

    RED against the base for two reasons at once: the flag does not exist
    under this name, and the shapes it would have to be set on are refused."""

    def test_the_two_group_launches_produce_one_key(self):
        p_args = _group_server_args(pp_size=3, tp_size=1, rank_kv_ratio=None)
        # Group D on the line of record carries --rank-kv-ratio coupled; the
        # canonical identity hash must not let that geometry term into the key.
        d_args = _group_server_args(tp_size=3, pp_size=1, rank_kv_ratio="coupled")
        with tempfile.TemporaryDirectory() as d:
            p = HiCacheFile(
                _config_from_server_args(p_args, tp_rank=0, window=_stage_window(0)),
                file_path=os.path.join(d, "p"),
            )
            dd = HiCacheFile(
                _config_from_server_args(
                    d_args, tp_rank=0, window=_whole_page_window()
                ),
                file_path=os.path.join(d, "d"),
            )
            self.assertEqual(p._get_suffixed_key(KEY), dd._get_suffixed_key(KEY))

    def test_with_the_format_off_the_two_launches_diverge(self):
        """CAN-FAIL: the premise. With the format off the two launches must
        name different bytes, or the test above proves nothing."""
        p_args = _group_server_args(
            pp_size=3, tp_size=1, rank_kv_ratio=None, hicache_canonical_kv_page=False
        )
        d_args = _group_server_args(
            tp_size=3,
            pp_size=1,
            rank_kv_ratio="coupled",
            hicache_canonical_kv_page=False,
        )
        with tempfile.TemporaryDirectory() as d:
            p = HiCacheFile(
                _config_from_server_args(p_args, tp_rank=0, window=None),
                file_path=os.path.join(d, "p"),
            )
            dd = HiCacheFile(
                _config_from_server_args(d_args, tp_rank=0, window=None),
                file_path=os.path.join(d, "d"),
            )
            self.assertNotEqual(p._get_suffixed_key(KEY), dd._get_suffixed_key(KEY))


class TestW10BothGroupsMustCarryTheFormat(CustomTestCase):
    """W10 ``Weg2CanonicalPageMissing``: the launcher's half. One group with
    the format and one without is the 100 % miss with extra steps."""

    def test_the_format_on_in_one_group_only_refuses(self):
        from sglang.srt.mem_cache.weg2_store_gates import (
            Weg2CanonicalPageMissing,
            check_canonical_page_enabled,
        )

        with self.assertRaises(Weg2CanonicalPageMissing) as cm:
            check_canonical_page_enabled({"P": True, "D": False})
        self.assertIn("D", str(cm.exception))

    def test_both_groups_on_passes(self):
        from sglang.srt.mem_cache.weg2_store_gates import check_canonical_page_enabled

        check_canonical_page_enabled({"P": True, "D": True})


class TestTheTwoGroupsNameTheSameBytes(CustomTestCase):
    """F1 (i): the key test. Byte-identical with the format on, different off."""

    def test_the_key_is_byte_identical_across_the_two_group_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            p = HiCacheFile(_p_config(0), file_path=os.path.join(d, "p"))
            dd = HiCacheFile(_d_config(0), file_path=os.path.join(d, "d"))
            self.assertEqual(
                p._get_suffixed_key(KEY),
                dd._get_suffixed_key(KEY),
                "group P and group D must name one page with one key",
            )

    def test_every_rank_of_both_groups_agrees_on_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            keys = set()
            for stage in range(3):
                keys.add(
                    HiCacheFile(
                        _p_config(stage), file_path=os.path.join(d, f"p{stage}")
                    )._get_suffixed_key(KEY)
                )
            for r in range(3):
                keys.add(
                    HiCacheFile(
                        _d_config(r), file_path=os.path.join(d, f"d{r}")
                    )._get_suffixed_key(KEY)
                )
            self.assertEqual(len(keys), 1, f"six ranks produced {len(keys)} keys")

    def test_with_the_format_off_the_two_shapes_diverge(self):
        """GREEN PIN / premise: this is the 100 % miss F1 exists to remove.
        It must stay true, or the test above proves nothing."""
        with tempfile.TemporaryDirectory() as d:
            p = HiCacheFile(
                _p_config(0, canonical=False), file_path=os.path.join(d, "p")
            )
            dd = HiCacheFile(
                _d_config(0, canonical=False), file_path=os.path.join(d, "d")
            )
            self.assertNotEqual(p._get_suffixed_key(KEY), dd._get_suffixed_key(KEY))

    def test_draft_keys_keep_their_per_rank_suffix(self):
        """GREEN PIN: draft pages are head-sharded and token-complete; no
        suffix rule can neutralise them, so they must NOT be re-keyed."""
        with tempfile.TemporaryDirectory() as d:
            a = HiCacheFile(_d_config(0), file_path=os.path.join(d, "a"))
            b = HiCacheFile(_d_config(1), file_path=os.path.join(d, "b"))
            draft = f"{KEY}.draft-a30db4b7c362c786"
            self.assertNotEqual(a._get_suffixed_key(draft), b._get_suffixed_key(draft))


class TestTheCarrierCrossesTwoProcesses(CustomTestCase):
    """The desk acceptance line: A (P-shape) writes, B (D-shape) reads it
    byte-identically, in two real processes over one directory."""

    def test_process_b_reads_process_as_page_byte_identically(self):
        with tempfile.TemporaryDirectory() as root:
            store_dir = os.path.join(root, "store")
            os.makedirs(store_dir)
            a = _run_worker("prefill", store_dir, os.path.join(root, "a.json"))
            self.assertEqual(a["status"], "ok", f"writer failed: {a}")
            b = _run_worker("decode", store_dir, os.path.join(root, "b.json"))
            self.assertEqual(b["status"], "ok", f"reader failed: {b}")

            self.assertEqual(
                set(a["keys"]),
                set(b["keys"]),
                "writer and reader used different keys for one page",
            )
            self.assertTrue(b["hit"], "reader got a MISS on the sole carrier")
            self.assertEqual(
                b["digest"],
                _digest(_expected_whole_page()),
                "the assembled page is not byte-identical to what P wrote",
            )


class TestF7ExactlyOneEvictionOwnerPerGroup(CustomTestCase):
    """F7: ``_is_storage_owner`` re-pointed at 'writes shared keys'."""

    def _decode_group_evictors(self, d):
        return [
            HiCacheFile(
                _d_config(r),
                file_path=d,
            )._evictor
            for r in range(3)
        ]

    def test_exactly_one_rank_of_the_decode_group_evicts(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                evictors = self._decode_group_evictors(d)
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            owners = [e for e in evictors if e._eviction_enabled]
            self.assertEqual(
                len(owners),
                1,
                f"{len(owners)} of 3 group-D ranks own eviction over one directory",
            )

    def test_the_cap_is_one_number_in_both_groups(self):
        """M6's target: with ``max_size_scope: shared`` and the old
        ``writer_count``, the same operator cap became 200 GiB in P
        (tp_size=1) and 66 GiB in D (tp_size=3) over ONE directory."""
        extra = {"max_size": "1G", "max_size_scope": "shared"}
        with tempfile.TemporaryDirectory() as d:
            pc = _p_config(0)
            pc.extra_config = dict(extra)
            dc = _d_config(0)
            dc.extra_config = dict(extra)
            p = HiCacheFile(pc, file_path=os.path.join(d, "p"))
            dd = HiCacheFile(dc, file_path=os.path.join(d, "d"))
            self.assertEqual(
                p._evictor.max_size_bytes,
                dd._evictor.max_size_bytes,
                "one directory, one cap -- the two groups must agree",
            )

    def test_a_non_canonical_non_mla_store_still_has_every_rank_as_owner(self):
        """GREEN PIN / CAN-FAIL: without the canonical page each rank really
        does write its own files, so the base behaviour must survive."""
        with tempfile.TemporaryDirectory() as d:
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                evictors = [
                    HiCacheFile(_d_config(r, canonical=False), file_path=d)._evictor
                    for r in range(3)
                ]
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            self.assertEqual(len([e for e in evictors if e._eviction_enabled]), 3)


def _seed_store(d, *, kv_suffix, cfg_suffix, n_pages, n_drafts, page_bytes=4096):
    """N canonical pages + M draft files, as the store of record holds them."""
    for i in range(n_pages):
        with open(os.path.join(d, f"{i:064x}{kv_suffix}.bin"), "wb") as f:
            f.write(b"\x01" * page_bytes)
    for i in range(n_drafts):
        with open(os.path.join(d, f"{i:064x}.draft-abc{cfg_suffix}.bin"), "wb") as f:
            f.write(b"\x02" * page_bytes)


def _dir_allocated_bytes(d):
    total = 0
    for fn in os.listdir(d):
        if not fn.endswith(".bin"):
            continue
        st = os.stat(os.path.join(d, fn))
        total += max(st.st_blocks * 512, st.st_size)
    return total


class TestW8bTheIndexCanSeeTheCanonicalPages(CustomTestCase):
    """W8b ``Weg2StoreIndexBlind``. Measured on the store of record: of
    124,610 files, 104,267 (83.7 %) end with no rank's ``config_suffix`` and
    are invisible to the LRU index, so ``max_size`` bounds nothing."""

    def test_the_index_covers_the_whole_directory(self):
        with tempfile.TemporaryDirectory() as d:
            store = HiCacheFile(_d_config(0), file_path=d)
            kv_suffix = store.kv_config_suffix
            cfg_suffix = store.config_suffix
            self.assertNotEqual(
                kv_suffix, cfg_suffix, "the fixture must exercise both suffixes"
            )
            _seed_store(
                d,
                kv_suffix=kv_suffix,
                cfg_suffix=cfg_suffix,
                n_pages=40,
                n_drafts=10,
            )
            evictor = LRUFileEvictor(
                d,
                cfg_suffix,
                kv_config_suffix=kv_suffix,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            coverage = evictor.index_coverage()
            self.assertEqual(
                coverage["indexed_bytes"],
                _dir_allocated_bytes(d),
                f"index sees {coverage['indexed_entries']} of "
                f"{coverage['seen_entries']} files",
            )
            self.assertGreaterEqual(coverage["fraction"], 0.95)

    def test_a_blind_index_refuses_the_launch(self):
        """CAN-IT-FAIL: an owner that indexes under half the directory must
        refuse, because the cap it would enforce is a fiction."""
        from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind

        with tempfile.TemporaryDirectory() as d:
            store = HiCacheFile(_d_config(0), file_path=d)
            _seed_store(
                d,
                kv_suffix=store.kv_config_suffix,
                cfg_suffix=store.config_suffix,
                n_pages=40,
                n_drafts=1,
            )
            with self.assertRaises(Weg2StoreIndexBlind) as cm:
                LRUFileEvictor(
                    d,
                    store.config_suffix,
                    kv_config_suffix=None,  # the base tree's scan filter
                    tp_rank=0,
                    writes_shared_keys=True,
                    extra_config={"max_size": "1G"},
                )
            self.assertIn("of", str(cm.exception))


class TestN2TheTouchReachesTheInode(CustomTestCase):
    """N2. Both groups' owners order their LRU by ``st_mtime`` -- a shared
    physical fact -- so a touch that stays in one process's memory lets the
    sibling evict the page it just served."""

    def test_touch_moves_the_files_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "page_sfx.bin")
            with open(path, "wb") as f:
                f.write(b"\x01" * 4096)
            old = 1_000_000.0
            os.utime(path, (old, old))
            evictor = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            evictor.touch("page_sfx", path)
            self.assertGreater(
                os.stat(path).st_mtime,
                old,
                "the touch never reached the inode the sibling reads",
            )

    def test_a_second_owner_rescan_sees_the_new_order(self):
        """The wake-time re-scan (spec 3.3 (3)): each owner starts its awake
        phase from the true directory state, both ordered by st_mtime."""
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, name in enumerate(("a", "b")):
                p = os.path.join(d, f"{name}_sfx.bin")
                with open(p, "wb") as f:
                    f.write(b"\x01" * 4096)
                os.utime(p, (1_000_000.0 + i, 1_000_000.0 + i))
                paths.append(p)
            sibling = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            self.assertEqual(list(sibling._lru), ["a_sfx", "b_sfx"])
            # The other group touches 'a' while this one sleeps.
            os.utime(paths[0], (2_000_000.0, 2_000_000.0))
            sibling.rescan()
            self.assertEqual(
                list(sibling._lru),
                ["b_sfx", "a_sfx"],
                "a woken owner must re-read the directory, not its stale index",
            )


class TestW9TheTwoGroupsMustShareOneIdentity(CustomTestCase):
    def test_differing_identity_hashes_refuse(self):
        from sglang.srt.mem_cache.weg2_store_gates import (
            Weg2StoreIdentityMismatch,
            check_identity_hashes,
        )

        with self.assertRaises(Weg2StoreIdentityMismatch) as cm:
            check_identity_hashes({"P": "520526c68d6530e9", "D": "aaaaaaaaaaaaaaaa"})
        self.assertIn("P", str(cm.exception))
        self.assertIn("D", str(cm.exception))

    def test_equal_identity_hashes_pass(self):
        from sglang.srt.mem_cache.weg2_store_gates import check_identity_hashes

        check_identity_hashes({"P": IDENTITY, "D": IDENTITY})

    def test_the_byte_format_still_separates(self):
        """GREEN PIN: geometry may drop out of the hash, the KV byte format
        never may -- that is the silent wrong hit the hash exists to catch."""
        from types import SimpleNamespace

        def _a(**over):
            base = dict(
                model_path="/models/Qwen3.8-27B",
                revision=None,
                dtype="auto",
                quantization=None,
                kv_cache_dtype="fp8_e4m3",
            )
            base.update(over)
            return SimpleNamespace(**base)

        self.assertNotEqual(
            compute_model_identity_hash(_a(), include_parallel_vectors=False),
            compute_model_identity_hash(
                _a(kv_cache_dtype="auto"), include_parallel_vectors=False
            ),
        )


class TestW7TheGdnBlobIsMandatoryOnAHybrid(CustomTestCase):
    """W7 ``Weg2MambaBlobAbsent``. A KV-only canonical page delivers ZERO
    usable prefix: ``batch_exists_v2`` takes the MINIMUM across pools and the
    mamba pool is registered TRAILING_PAGES, so a missing blob truncates the
    whole KV prefix to zero."""

    def test_a_hybrid_without_a_blob_refuses(self):
        from sglang.srt.mem_cache.weg2_store_gates import (
            Weg2MambaBlobAbsent,
            check_mamba_blob_present,
        )

        with self.assertRaises(Weg2MambaBlobAbsent):
            check_mamba_blob_present(
                canonical_page_on=True, has_mamba_pool=True, mamba_blob=None
            )

    def test_a_proven_dense_model_needs_no_blob(self):
        """CAN-FAIL GUARD: the refusal must not fire on a dense model, or no
        non-hybrid could ever use the format."""
        from sglang.srt.mem_cache.weg2_store_gates import check_mamba_blob_present

        check_mamba_blob_present(
            canonical_page_on=True, has_mamba_pool=False, mamba_blob=None
        )

    def test_the_format_off_needs_no_blob(self):
        from sglang.srt.mem_cache.weg2_store_gates import check_mamba_blob_present

        check_mamba_blob_present(
            canonical_page_on=False, has_mamba_pool=True, mamba_blob=None
        )


class TestW8TheStoreCapMustBeFundable(CustomTestCase):
    """W8 ``Weg2StoreCapUnfundable`` (operator ruling Q5: 200 GiB cap,
    100 GiB min free on /spinning/hicache-weg2). The tree's existing
    behaviour is a SILENT clamp; on a sole carrier a silently smaller cap is
    a capacity regression nobody sees."""

    def test_a_cap_the_filesystem_cannot_hold_refuses(self):
        from sglang.srt.mem_cache.weg2_store_gates import (
            Weg2StoreCapUnfundable,
            check_store_cap_fundable,
        )

        with tempfile.TemporaryDirectory() as d:
            total = os.statvfs(d).f_blocks * (os.statvfs(d).f_frsize or 4096)
            with self.assertRaises(Weg2StoreCapUnfundable) as cm:
                check_store_cap_fundable(d, total, total)
            self.assertIn(str(total), str(cm.exception))

    def test_a_fundable_cap_passes(self):
        from sglang.srt.mem_cache.weg2_store_gates import check_store_cap_fundable

        with tempfile.TemporaryDirectory() as d:
            check_store_cap_fundable(d, 1 << 20, 1 << 20)


if __name__ == "__main__":
    unittest.main()
