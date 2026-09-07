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


def _p_config(
    stage: int,
    *,
    canonical: bool = True,
    is_mla_model: bool = False,
    dcp_owner_mode: bool = False,
):
    """Group P, rank ``stage``: pp_size=3, tp_size=1."""
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=stage,
        pp_size=3,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        dcp_owner_mode=dcp_owner_mode,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=IDENTITY,
        canonical_kv_page=_stage_window(stage) if canonical else None,
    )


def _d_config(
    tp_rank: int,
    *,
    canonical: bool = True,
    window=None,
    is_mla_model: bool = False,
    extra_config=None,
):
    """Group D, rank ``tp_rank``: tp_size=3, pp_size=1.

    ``window`` overrides the whole-page window with a PARTIAL extent, which is
    what a real TP rank deposits: group D's cut is head channels across every
    layer, so no single rank's write completes the blob. The fixture cuts on
    the layer axis instead -- the axis this page format expresses -- because
    what the write path is being asked here is not which axis was cut but
    whether a rank that writes only PART of a shared file is admitted at all.
    """
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=3,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=IDENTITY,
        canonical_kv_page=(
            window
            if window is not None
            else (_whole_page_window() if canonical else None)
        ),
        extra_config=extra_config,
    )


def _cp_config(attn_cp_rank: int, *, attn_cp_size: int = 2):
    """An MLA rank on the attention-CP axis: tp_size=1, pp_size=1, cp_size=n.

    Not a Weg-2 shape -- a shipping one. It is here because the attn-cp term
    is appended to the suffix with no ``is_mla_model`` guard, so it is one of
    the two axes on which "MLA means every rank names one path" is false.
    """
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_NAME,
        model_identity_hash=IDENTITY,
        canonical_kv_page=None,
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
            owners = []
            wrote = []
            for stage in range(3):
                store = HiCacheFile(_p_config(stage), file_path=store_dir)
                keys.append(store._get_suffixed_key(KEY))
                owners.append(bool(store._evictor.is_storage_owner))
                wrote.append(bool(store.set(KEY, _payload(stage))))
            result = {"status": "ok", "keys": keys, "owners": owners, "wrote": wrote}
        else:
            store = HiCacheFile(_d_config(0), file_path=store_dir)
            owners = [
                bool(
                    HiCacheFile(
                        _d_config(r), file_path=store_dir
                    )._evictor.is_storage_owner
                )
                for r in range(3)
            ]
            target = torch.zeros(NUM_ATTN_LAYERS * CELL_BYTES, dtype=torch.uint8)
            got = store.get(KEY, target)
            result = {
                "status": "ok",
                "keys": [store._get_suffixed_key(KEY)],
                "owners": owners,
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
            # ONE owner PER GROUP, graded on both shapes. Grading only group D
            # let the election miss group P entirely: with pp_size=3, tp_size=1
            # every stage has tp_rank == 0, so a tp-keyed election makes all
            # three of them owners of one directory.
            self.assertEqual(
                sum(1 for o in a["owners"] if o),
                1,
                f"group P elected {a['owners']} storage owners over one directory",
            )
            self.assertEqual(
                sum(1 for o in b["owners"] if o),
                1,
                f"group D elected {b['owners']} storage owners over one directory",
            )
            self.assertTrue(
                all(a["wrote"]),
                f"a P stage was refused its own extent of the blob: {a['wrote']}",
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

    def _prefill_group_evictors(self, d):
        return [
            HiCacheFile(
                _p_config(stage),
                file_path=d,
            )._evictor
            for stage in range(3)
        ]

    def test_exactly_one_rank_of_the_prefill_group_evicts(self):
        """The half the acceptance line never graded.

        Group P is ``pp_size=3, tp_size=1``, so a ``tp_rank == 0`` election
        elects ALL THREE stages: three private LRU indices over one directory,
        each carrying the whole operator cap and each unlinking victims the
        other two still count. The tree names this exact trap 90 lines above
        the call site (``hicache_storage.py``: "with pure PP every stage has
        tp_rank == 0 and attn_cp_rank == 0, so all three ranks elect
        THEMSELVES") -- for the directory, not for the eviction index.
        """
        with tempfile.TemporaryDirectory() as d:
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                evictors = self._prefill_group_evictors(d)
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            owners = [e for e in evictors if e._eviction_enabled]
            self.assertEqual(
                len(owners),
                1,
                f"{len(owners)} of 3 group-P stages own eviction over one directory",
            )

    def test_exactly_one_attn_cp_rank_of_a_shared_key_group_evicts(self):
        """THE THIRD AXIS OF THE ELECTION, ungraded until now.

        The election was deliberately widened from ``tp_rank == 0`` to the
        group's full rank identity, and the suite graded the tp axis (group D)
        and the pp axis (group P) but not attn_cp. Dropping ``and
        attn_cp_rank == 0`` left the whole slice suite green while restoring
        multi-owner-over-one-directory on that axis: two private LRU indices
        over one directory, each carrying the whole cap and each unlinking
        victims the other still counts.

        REACHABILITY, stated honestly: not the Weg-2 P/D shapes, both of which
        are ``attn_cp_size=1``. Reachable on any shared-key file store with
        ``attn_cp_size > 1``, which ``writes_shared_keys`` admits through MLA
        and through dcp owner mode -- shipping, non-Weg-2 shapes.
        """
        with tempfile.TemporaryDirectory() as d:
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                evictors = [
                    HiCacheFile(_cp_config(r), file_path=d)._evictor for r in range(2)
                ]
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            self.assertTrue(
                all(e._writes_shared_keys for e in evictors),
                "the premise of this row is a shared-key group",
            )
            owners = [e for e in evictors if e._eviction_enabled]
            self.assertEqual(
                len(owners),
                1,
                f"{len(owners)} of 2 attn-cp ranks own eviction over one directory",
            )

    def test_a_per_rank_key_store_still_splits_the_cap(self):
        """THE OTHER HALF OF THE ONE PREDICATE ``writer_count`` NOW READS.

        F7 rewrote ``1 if is_mla_model else max(1, tp_size)`` to
        ``1 if shared_keys else max(1, tp_size)``. The only row grading it
        compares group P against group D, and BOTH are shared-key, so it pins
        'P == D' and never 'the split still happens where keys are not
        shared'. MEASURED: collapsing the expression to a constant ``1`` left
        the whole slice suite green, and three TP ranks then each enforce the
        operator's WHOLE cap over one directory -- up to 3x the configured
        budget on disk, with no log line and no refusal.
        """
        cap = 1_000_000_000
        with tempfile.TemporaryDirectory() as d:
            per_rank = HiCacheFile(
                _d_config(0, canonical=False, extra_config={"max_size": str(cap)}),
                file_path=os.path.join(d, "per_rank"),
            )
            shared = HiCacheFile(
                _d_config(0, extra_config={"max_size": str(cap)}),
                file_path=os.path.join(d, "shared"),
            )
            self.assertFalse(per_rank._evictor._writes_shared_keys)
            self.assertTrue(shared._evictor._writes_shared_keys)
            self.assertEqual(
                per_rank._evictor.max_size_bytes,
                cap // 3,
                "a store whose ranks write their own files must still split "
                "the operator's cap by tp_size",
            )
            self.assertEqual(
                shared._evictor.max_size_bytes,
                cap,
                "a shared-key group spends the cap once between its ranks",
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


def _store_allocated_bytes(d):
    """``_dir_allocated_bytes`` over the SHARDED layout a real store writes.

    ``HiCacheFile`` shards its files into subdirectories, so the flat helper
    above answers 0 for a directory a real backend filled -- which would let a
    coverage assertion pass by comparing zero with zero.
    """
    total = 0
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if not fn.endswith(".bin"):
                continue
            st = os.stat(os.path.join(root, fn))
            total += max(st.st_blocks * 512, st.st_size)
    return total


# A key that falls to the DEFAULT branch of ``_suffix_for_key`` -- it is not a
# bare page hash, so no neutralisation rule applies and every rank writes it
# under its OWN ``config_suffix``. That is the population the sole eviction
# owner has to widen its scan filter to see.
DRAFT_KEY = f"{KEY[:-1]}%x.draft-a30db4b7c362c786"


def _every_rank_writes_its_own_drafts(case, d, configs, n_drafts=4):
    """Every rank of ONE group writes its own draft pages; a fresh rank-0
    store then attaches as that group's sole eviction owner.

    ONE fixture for all three axes on purpose. The mechanism under test,
    ``HiCacheStorage._group_scan_suffixes``, derives the owner's scan
    population from ``tp_size x pp_size x attn_cp_size``, so the only thing
    that may differ between the axes is the GEOMETRY -- a fixture per axis
    would be a second bookkeeping of exactly the derivation being graded, and
    would drift the day a fourth axis is added.

    Returns ``(owner, on_disk_bytes)``.
    """
    os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
    try:
        stores = [HiCacheFile(c, file_path=d) for c in configs]
        for r, store in enumerate(stores):
            for i in range(n_drafts):
                case.assertTrue(
                    store.set(
                        DRAFT_KEY % (r * 16 + i),
                        torch.zeros(4096, dtype=torch.uint8),
                    ),
                    f"rank {r} draft {i} was not admitted",
                )
        owner = HiCacheFile(configs[0], file_path=d)
    finally:
        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
    on_disk = _store_allocated_bytes(d)
    case.assertGreater(on_disk, 0, "the fixture wrote nothing")
    return owner, on_disk


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

    def test_the_owner_indexes_every_rank_of_the_groups_files(self):
        """THE SOLE OWNER'S POPULATION IS THE GROUP'S, NOT ITS OWN.

        The row above seeds the fixture from ONE rank's two suffixes, so by
        construction it contains no other rank's file and the 95 % acceptance
        is measured over the one population where the defect is absent. Here
        all three ranks of group D write their own draft pages through the real
        backend, and a fresh owner must bound every byte on disk.

        MEASURED at 69447a36 (fix 2): 120 files on disk, 40 in the owner's
        index -- 80 files outside every byte bound any process in the group can
        apply, because the owner's scan filter was its own ``_0_3`` tail and
        ``reserve`` admits a non-owner's own-suffix write without indexing it.
        """
        with tempfile.TemporaryDirectory() as d:
            owner, on_disk = _every_rank_writes_its_own_drafts(
                self, d, [_d_config(r) for r in range(3)]
            )
            coverage = owner._evictor.index_coverage()
            self.assertEqual(
                coverage["indexed_entries"],
                12,
                "3 ranks x 4 drafts must all be in the one index",
            )
            self.assertEqual(
                coverage["indexed_bytes"],
                on_disk,
                f"the sole owner bounds {coverage['indexed_bytes']} of "
                f"{on_disk} B: another rank's files are outside every bound",
            )

    def test_the_owner_indexes_every_pp_stage_of_group_p(self):
        """THE SAME QUESTION ON THE AXIS WEG 2 ACTUALLY PREFILLS ON.

        The row above is ``_d_config`` -- ``tp_size=3, pp_size=1``. It is the
        only shape any coverage row graded, so the pp loop of
        ``_group_scan_suffixes`` had no can-it-fail proof while group P
        (``pp_size=3, tp_size=1``) is a Weg-2 shape and the store is the SOLE
        carrier across a flip. Dropping that loop leaves the whole suite green
        and the owner scanning one stage's tail.

        MEASURED at 34cd87bd55 on this fixture: group P's derived scan set is
        four suffixes spanning ``pp_rank`` 0/1/2 plus the canonical kv tail
        (``_0_1_3_0``, ``_0_1_3_1``, ``_0_1_3_2``, ``_<model>_<hash>``) and the
        owner indexes 12 of 12. CAN-IT-FAIL, measured: with the pp loop pinned
        to ``range(1)`` this row dies at ``Weg2StoreIndexBlind`` -- *"indexed
        16384 of 49152 B (33.3 %, floor 50 %) across 4 of 12 files"*.

        W8b FIRING HERE IS AN ARTEFACT OF THE FIXTURE, NOT THE PRODUCTION
        BEHAVIOUR, and the row must not be read as "the gate already covers
        this". The fixture is draft-only, so the lost stages are a third of the
        bytes; the store of record is page-dominated (940 pages : 60 drafts,
        spec 3.3 (4)), where the same mutant leaves the group-wide kv tail
        indexed and gives coverage 0.960 -- above the floor, no refusal, and
        two stages' files outside every byte bound. That is why the count is
        asserted and not merely the gate's verdict.
        """
        with tempfile.TemporaryDirectory() as d:
            owner, on_disk = _every_rank_writes_its_own_drafts(
                self, d, [_p_config(r) for r in range(3)]
            )
            coverage = owner._evictor.index_coverage()
            self.assertEqual(
                coverage["indexed_entries"],
                12,
                "3 PP stages x 4 drafts must all be in the one index",
            )
            self.assertEqual(
                coverage["indexed_bytes"],
                on_disk,
                f"the sole owner bounds {coverage['indexed_bytes']} of "
                f"{on_disk} B: another PP stage's files are outside every bound",
            )

    def test_the_owner_indexes_every_attn_cp_rank_of_the_group(self):
        """The third axis of the same derivation, on a shipping shape.

        ``attn_cp`` is not a Weg-2 group, but the term is appended with no
        ``is_mla_model`` guard, so an MLA store's ranks name per-rank paths on
        it -- the same reason ``_cp_config`` already grades the owner election
        (R6). It is also the axis whose suffix is BOTH the config and the kv
        tail, so nothing group-wide survives to pad the coverage.

        THIS IS THE ROW WHERE THE GATE PROVABLY DOES NOT CATCH IT. MEASURED at
        34cd87bd55: 8 of 8 indexed; with the cp loop pinned to ``range(1)`` the
        owner indexes 4 of 8 -- coverage exactly 0.500, and
        ``check_index_coverage`` compares ``fraction >= floor`` against
        ``INDEX_COVERAGE_FLOOR = 0.5``, so W8b returns rather than raises and
        this row fails on the count alone: *"AssertionError: 4 != 8"*. Half the
        group's files outside every byte bound, no refusal anywhere.
        """
        with tempfile.TemporaryDirectory() as d:
            owner, on_disk = _every_rank_writes_its_own_drafts(
                self, d, [_cp_config(r) for r in range(2)]
            )
            coverage = owner._evictor.index_coverage()
            self.assertEqual(
                coverage["indexed_entries"],
                8,
                "2 attn-cp ranks x 4 drafts must all be in the one index",
            )
            self.assertEqual(
                coverage["indexed_bytes"],
                on_disk,
                f"the sole owner bounds {coverage['indexed_bytes']} of "
                f"{on_disk} B: the sibling cp rank's files are outside every bound",
            )

    def test_the_owners_scan_set_is_every_suffix_its_group_can_name(self):
        """GRADE THE DERIVATION ITSELF, not only the count it produces.

        The three rows above count files, so a scan set wrong in a way no
        fixture file happens to land on still passes them. Here the expected
        set is taken from THE RANKS THEMSELVES -- each rank's own
        ``(config_suffix, kv_config_suffix)`` pair, read off a constructed
        store -- and the owner's filter must equal exactly that union.

        That is deliberately NOT a second copy of ``_build_key_suffixes``: the
        ranks are the authority on which paths they name, which is the same
        reason ``_derive_key_suffixes`` answers the group-wide question by
        re-deriving rather than by listing axes. One row therefore grades all
        three loops -- drop any one of them and the owner's set is a strict
        subset of what its own group writes.

        THE ONE THING THIS ROW CANNOT SEE, stated so the green is not read
        wider than it is: on the attn_cp shape ``config_suffix`` and
        ``kv_config_suffix`` are the SAME string, so that subtest alone cannot
        catch a mutant that drops the kv half of the derived set. Measured:
        appending only ``cfg`` inside the loop turns the tp and pp subtests red
        and leaves the cp one green. The tp and pp shapes carry the kv half.
        """
        for label, configs in (
            ("group D, tp axis", [_d_config(r) for r in range(3)]),
            ("group P, pp axis", [_p_config(r) for r in range(3)]),
            ("attn_cp axis", [_cp_config(r) for r in range(2)]),
        ):
            with self.subTest(group=label), tempfile.TemporaryDirectory() as d:
                ranks = [HiCacheFile(c, file_path=d) for c in configs]
                named = set()
                for rank in ranks:
                    named.update(
                        s for s in (rank.config_suffix, rank.kv_config_suffix) if s
                    )
                owner = ranks[0]
                self.assertTrue(
                    owner._evictor._scan_population_is_group_wide,
                    f"{label}: the sole owner still scans its own tail",
                )
                self.assertEqual(
                    set(owner._evictor._scan_suffixes),
                    named,
                    f"{label}: the owner scans "
                    f"{sorted(owner._evictor._scan_suffixes)} while its group "
                    f"writes {sorted(named)}",
                )

    def test_every_rank_of_a_shared_key_group_refuses_a_blind_store(self):
        """W8b IS A GROUP VERDICT AT ATTACH, exactly as W8 is.

        MEASURED at 69447a36 over a store this very group had used once: rank 0
        raised ``Weg2StoreIndexBlind`` at 33.3 % coverage while ranks 1 and 2
        constructed and walked into the next collective -- one rank dead, five
        alive, which is the disagreement user law §0 forbids rather than the
        stop it asks for. It is reachable in normal operation because the
        unindexed non-owner files grow until the fraction crosses the floor.

        The gate can only be a group verdict because the census now is one: the
        filter is the GROUP's suffix set, so numerator, denominator and filter
        are properties of the directory and the geometry, not of the reading
        rank. CAN-IT-FAIL: put the gate back below the owner-only early return
        and ranks 1 and 2 construct, turning this row red.
        """
        from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind

        with tempfile.TemporaryDirectory() as d:
            # Foreign bytes: a suffix no rank of this group can ever name, so
            # the whole group is blind to them and every rank agrees it is.
            for i in range(8):
                with open(
                    os.path.join(d, f"{i:064x}_someone-elses-model.bin"), "wb"
                ) as f:
                    f.write(b"\x03" * 4096)
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                for rank in range(3):
                    with self.assertRaises(
                        Weg2StoreIndexBlind, msg=f"rank {rank} did not refuse"
                    ):
                        HiCacheFile(_d_config(rank), file_path=d)
                # CAN-FAIL COMPANION: the same three ranks over a directory
                # their group's suffixes do name construct, so the rows above
                # grade the coverage and not the constructor.
                for fn in os.listdir(d):
                    os.unlink(os.path.join(d, fn))
                probe = HiCacheFile(_d_config(0), file_path=d)
                _seed_store(
                    d,
                    kv_suffix=probe.kv_config_suffix,
                    cfg_suffix=probe.config_suffix,
                    n_pages=4,
                    n_drafts=2,
                )
                for rank in range(3):
                    HiCacheFile(_d_config(rank), file_path=d)
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)

    def test_the_floor_refuses_the_store_of_record(self):
        """The THRESHOLD, pinned at the population it was chosen for.

        ``INDEX_COVERAGE_FLOOR`` is 0.5 because the store of record measured
        16.3 % coverage (104,267 of 124,610 files invisible). The blind-index
        row below seeds 40 pages + 1 draft, i.e. ~2.4 % coverage, so any floor
        above ~0.025 keeps it green: a floor of 0.10 -- a plausible edit --
        lets the exact measured defect through while the gate still looks
        armed. This row grades the number between the measured defect and a
        healthy store instead.
        """
        from sglang.srt.mem_cache.weg2_store_gates import (
            Weg2StoreIndexBlind,
            check_index_coverage,
        )

        with self.assertRaises(Weg2StoreIndexBlind):
            check_index_coverage(
                store_path="/store-of-record",
                indexed_bytes=163,
                seen_bytes=1000,
                indexed_entries=20343,
                seen_entries=124610,
            )
        # CAN-FAIL COMPANION: a store whose owner sees most of it passes, so
        # the row above grades the floor and not the gate's mere existence.
        self.assertAlmostEqual(
            check_index_coverage(
                store_path="/healthy",
                indexed_bytes=600,
                seen_bytes=1000,
                indexed_entries=600,
                seen_entries=1000,
            ),
            0.6,
        )

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


def _drive_generate_storage_config(*, has_mamba_pool: bool, mamba_blob):
    """Run the PRODUCTION W7 call site: ``HiCacheController._generate_storage_config``.

    Called unbound on a double rather than on a real controller: what is being
    graded is one keyword argument at one call site, and a real controller
    would need pools, a model and a device. Everything the function reaches
    outside itself is patched at the module attribute it looks up, which is
    also why ``build_page_window`` / ``resolve_attn_layer_ids`` are patched on
    ``canonical_page_store`` -- the function imports them at call time.
    """
    import types
    from unittest import mock

    import sglang.srt.managers.cache_controller as cc
    import sglang.srt.mem_cache.canonical_page_store as cps

    controller = types.SimpleNamespace(
        mem_pool_device=object(),
        mem_pool_device_hybrid=types.SimpleNamespace(
            mamba_pool=object() if has_mamba_pool else None
        ),
        mem_pool_host=types.SimpleNamespace(layout="layer_first"),
        enable_storage_metrics=False,
        get_attn_cp_rank_and_size=lambda: (0, 1),
        _canonical_mamba_window=lambda server_args, model_config: mamba_blob,
        _dcp_owner_ctx=lambda: None,
    )
    server_args = types.SimpleNamespace(
        hicache_canonical_kv_page=True,
        get_model_config=lambda: object(),
        hicache_host_role="retention",
    )
    parallel = types.SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    window = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=NUM_ATTN_LAYERS)
    with mock.patch.object(cc, "get_parallel", lambda: parallel), mock.patch.object(
        cc, "is_dp_attention_enabled", lambda: False
    ), mock.patch.object(cc, "get_server_args", lambda: server_args), mock.patch.object(
        cc, "canonical_identity_hash_for", lambda args, on: IDENTITY
    ), mock.patch.object(
        cps, "resolve_attn_layer_ids", lambda mc: list(range(NUM_ATTN_LAYERS))
    ), mock.patch.object(
        cps, "build_page_window", lambda ids, dev, host: window
    ):
        return cc.HiCacheController._generate_storage_config(
            controller, model_name=MODEL_NAME
        )


class TestW7TheWiringCanFire(CustomTestCase):
    """W7's SEAM, the blind spot W8b already had its own row for.

    All three rows of ``TestW7TheGdnBlobIsMandatoryOnAHybrid`` call the pure
    ``check_mamba_blob_present`` directly, so the ONE production call site is
    graded nowhere. MEASURED: replacing
    ``has_mamba_pool=getattr(self.mem_pool_device_hybrid, "mamba_pool", None)
    is not None`` with ``has_mamba_pool=False`` -- after which the gate can
    never fire on any model -- left both S5 files green (44 passed) and all
    five touched neighbour suites green (71 passed). A gate that cannot fire
    is the danger direction here: the failure it guards is the measured #931
    shape, which on the sole carrier is a 100 % miss that reads as a cold
    cache.
    """

    def test_the_production_call_site_refuses_a_hybrid_without_a_blob(self):
        from sglang.srt.mem_cache.weg2_store_gates import Weg2MambaBlobAbsent

        with self.assertRaises(Weg2MambaBlobAbsent):
            _drive_generate_storage_config(has_mamba_pool=True, mamba_blob=None)

    def test_the_production_call_site_lets_a_dense_model_through(self):
        """CAN-FAIL COMPANION: no mamba pool bound is a dense model, and the
        gate must not refuse it -- otherwise the row above would pass with a
        call site that raises unconditionally."""
        config = _drive_generate_storage_config(has_mamba_pool=False, mamba_blob=None)
        self.assertIsInstance(config, HiCacheStorageConfig)

    def test_the_production_call_site_lets_a_hybrid_with_a_blob_through(self):
        config = _drive_generate_storage_config(
            has_mamba_pool=True, mamba_blob=object()
        )
        self.assertIsInstance(config, HiCacheStorageConfig)


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

    def test_the_evictor_grades_its_own_configured_numbers(self):
        """MB3, the SEAM: the two rows above hand the gate numbers by hand.

        MEASURED: replacing ``self.min_free_bytes`` with ``0`` at the
        production call site left the suite green -- and that mutant is not
        cosmetic, because a cap that fits ``total`` but not
        ``total - min_free`` then passes W8 and is SILENTLY clamped to exactly
        that difference by ``_clamp_max_size_to_fs``: the operator's Q5 number
        stays on the launch line while a smaller budget is served.
        """
        from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreCapUnfundable

        with tempfile.TemporaryDirectory() as d:
            st = os.statvfs(d)
            total = st.f_blocks * (st.f_frsize or 4096)
            with self.assertRaises(Weg2StoreCapUnfundable):
                LRUFileEvictor(
                    d,
                    "_sfx",
                    kv_config_suffix=None,
                    tp_rank=0,
                    writes_shared_keys=True,
                    # Fits ``total`` on its own; does not fit total - min_free.
                    extra_config={"max_size": total, "min_free_space": 4096},
                )
            # CAN-FAIL COMPANION: a fundable pair constructs, so the row above
            # grades the numbers and not the constructor.
            LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": total // 4, "min_free_space": total // 4},
            )

    def test_every_rank_of_a_shared_key_group_refuses_together(self):
        """W8 is a GROUP verdict: one rank refusing alone is a disagreement.

        After the election runs over the full rank identity, exactly one rank
        of a shared-key group is the eviction owner, so a gate placed below
        the owner-only early return refuses rank 0 while ranks 1..n-1 build
        their backend and walk into the next collective. Its three inputs are
        rank-invariant by construction (same extra_config, same filesystem),
        so every rank must reach the same verdict and stop together -- user
        law §0, "Ranks never disagree ... STOP, never compensation".
        """
        from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreCapUnfundable

        with tempfile.TemporaryDirectory() as d:
            st = os.statvfs(d)
            total = st.f_blocks * (st.f_frsize or 4096)
            unfundable = {"max_size": total, "min_free_space": 4096}
            for rank in range(3):
                with self.assertRaises(
                    Weg2StoreCapUnfundable, msg=f"rank {rank} did not refuse"
                ):
                    LRUFileEvictor(
                        d,
                        f"_sfx_{rank}_3",
                        kv_config_suffix="_sfx",
                        tp_rank=rank,
                        writes_shared_keys=True,
                        extra_config=unfundable,
                    )
            # CAN-FAIL COMPANION / no stock boot gains a refusal: where every
            # rank owns its own suffixed files there is no shared budget to
            # fund, and the same numbers construct.
            for rank in range(3):
                LRUFileEvictor(
                    d,
                    f"_sfx_{rank}_3",
                    kv_config_suffix=None,
                    tp_rank=rank,
                    writes_shared_keys=False,
                    extra_config=unfundable,
                )


class TestF7WriteAdmissionIsPerFileNotPerRank(CustomTestCase):
    """F7's second half: OWNING THE INDEX IS NOT PERMISSION TO WRITE.

    ``reserve()`` refuses every write from a non-owner. That was safe exactly
    while ``is_mla_model`` chose the non-owners, because MLA ranks write
    byte-identical whole files under one rank-free suffix -- rank 0's write
    already puts every byte of that path on disk. Re-pointing the ELECTION at
    ``writes_shared_keys`` (which the canonical page and dcp owner mode make
    true for a GQA model) carried that refusal to ranks whose bytes nobody
    else writes: their extent of the shared blob, their own suffixed draft
    file. On Weg 2's store, where the cap is always configured, that is a
    carrier that moves nothing -- the failure W7 exists to refuse, arriving
    through the write path instead.
    """

    def _bounded(self, fn):
        os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
        try:
            return fn()
        finally:
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)

    def test_every_decode_rank_deposits_its_extent_of_one_blob(self):
        """Three ranks, three disjoint extents, one blob -- with the cap on."""

        def body(d):
            stores = [
                HiCacheFile(_d_config(r, window=_stage_window(r)), file_path=d)
                for r in range(3)
            ]
            wrote = [stores[r].set(KEY, _payload(r)) for r in range(3)]
            return stores, wrote

        with tempfile.TemporaryDirectory() as d:
            stores, wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                wrote,
                [True, True, True],
                "a decode rank was refused its own extent of the shared blob",
            )
            self.assertTrue(
                stores[0].exists(KEY),
                "the blob never completed: the non-owner extents were dropped",
            )
            # Read back through a WHOLE-page window: each writer above holds
            # only its own extent, so none of them can assemble the blob.
            reader = HiCacheFile(_d_config(0), file_path=d)
            target = torch.zeros(NUM_ATTN_LAYERS * CELL_BYTES, dtype=torch.uint8)
            self.assertIsNotNone(reader.get(KEY, target))
            self.assertEqual(_digest(target), _digest(_expected_whole_page()))

    def test_every_decode_rank_writes_its_own_draft_file(self):
        """Draft keys keep a per-rank suffix, so the owner never writes them.

        The write-side twin of ``test_draft_keys_keep_their_per_rank_suffix``:
        HEAD pins that the keys are per-rank AND that only rank 0 may write
        them, which cannot both be intended.
        """
        draft = f"{KEY}.draft-a30db4b7c362c786"

        def body(d):
            stores = [HiCacheFile(_d_config(r), file_path=d) for r in range(3)]
            return [s.set(draft, torch.zeros(4096, dtype=torch.uint8)) for s in stores]

        with tempfile.TemporaryDirectory() as d:
            wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                wrote, [True, True, True], "a rank's own draft was refused"
            )
            found = []
            for root, _, files in os.walk(d):
                found += [f for f in files if f.endswith(".bin")]
            self.assertEqual(
                len(found), 3, f"3 per-rank draft files expected, found {found}"
            )

    def test_an_mla_non_owner_is_still_refused(self):
        """GREEN PIN / CAN-FAIL: the upstream de-duplication must survive.

        Under MLA the suffix carries no rank term at all
        (``_derive_key_suffixes``: the tp terms are appended only
        ``if not is_mla_model``), so every rank names ONE path and the owner's
        write is the whole file. Admitting the others there would be three
        processes writing one path -- the reason the refusal exists.
        """

        def body(d):
            stores = [
                HiCacheFile(
                    _d_config(r, canonical=False, is_mla_model=True), file_path=d
                )
                for r in range(3)
            ]
            # Rank 1 goes FIRST, on a key nobody has written: otherwise the
            # already-exists fast path answers before the admission question is
            # ever asked, and the pin would grade nothing.
            return [
                stores[1].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[0].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
            ]

        with tempfile.TemporaryDirectory() as d:
            wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                wrote,
                [False, True],
                "MLA ranks write one identical file; only the owner may write it",
            )

    def test_an_mla_stage_still_deposits_its_extent_of_the_canonical_blob(self):
        """The one case where the two questions pull APART inside one write.

        MLA is exactly where ``owner_writes_whole_file`` is True for an
        ordinary page -- and an extent write is still a PART, because the
        canonical format cuts on the layer axis, which MLA does not collapse.
        Deriving the answer from the model shape alone would refuse stages 1
        and 2 here and the blob would never complete, on a model where every
        other write of theirs is correctly de-duplicated.
        """

        def body(d):
            stores = [
                HiCacheFile(_p_config(st, is_mla_model=True), file_path=d)
                for st in range(3)
            ]
            return stores, [stores[st].set(KEY, _payload(st)) for st in range(3)]

        with tempfile.TemporaryDirectory() as d:
            stores, wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                wrote,
                [True, True, True],
                "an MLA PP stage was refused its own layer extent of the blob",
            )
            self.assertTrue(
                stores[0].exists(KEY),
                "the blob never completed: the non-owner stages were dropped",
            )

    def test_an_mla_pp_stage_writes_the_file_only_it_names(self):
        """MEASURED REGRESSION vs the parent, on a shipping non-Weg-2 shape.

        ``is_mla_model`` was read as "every rank names one path". It is not:
        ``_derive_key_suffixes`` drops the TP terms under MLA and appends
        ``_{pp_size}_{pp_rank}`` and ``_cp{rank}_{size}`` with no such guard.
        At 8a71eb87, MLA pp_size=3 gave stage 1 the suffix ``_M_H_3_1`` -- a
        path the elected owner (pp0) never writes -- and stages 1 and 2 were
        refused every write. The parent aef3ae76 (election ``tp_rank == 0``,
        which every PP stage satisfies) admitted all three.
        """

        def body(d):
            stores = [
                HiCacheFile(
                    _p_config(st, canonical=False, is_mla_model=True), file_path=d
                )
                for st in range(3)
            ]
            # Stages 1 and 2 first: on a key already on disk the
            # already-exists fast path answers before admission is asked.
            return stores, [
                stores[1].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[2].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[0].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
            ]

        with tempfile.TemporaryDirectory() as d:
            stores, wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                sorted({s.config_suffix for s in stores}),
                [
                    "_Qwen3.8-27B_520526c68d6530e9_3_0",
                    "_Qwen3.8-27B_520526c68d6530e9_3_1",
                    "_Qwen3.8-27B_520526c68d6530e9_3_2",
                ],
                "the premise of this test is that MLA PP stages name three paths",
            )
            self.assertEqual(
                wrote,
                [True, True, True],
                "an MLA PP stage was refused the file only it names",
            )
            found = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".bin")]
            self.assertEqual(
                len(found), 3, f"3 per-stage files expected, found {found}"
            )

    def test_an_mla_dcp_stage_writes_the_shared_kv_file_only_it_names(self):
        """THE KV HALF of the pair fix 2 created, ungraded until now.

        ``_suffix_for_key`` returns two group-wide answers -- one for
        ``config_suffix``, one for ``kv_config_suffix`` -- and every existing
        write-admission row exercises shapes that fall through to the CONFIG
        branch (the MLA rows are ``canonical=False`` with dcp off). MEASURED:
        hard-wiring ``_kv_config_suffix_is_group_wide`` to True left the slice
        suite green while re-opening exactly the regression fix 2 closed, on
        the untested half: MLA + dcp owner mode + pp_size=3 admitted all three
        stages (3 files) at HEAD and refused two of them under the mutant --
        silent, no raise, reading as a cold cache.

        REACHABILITY, stated honestly: this needs MLA (the group-wide term is
        inert for a GQA model, because ``owner_write_covers_whole_file``
        answers ``bool(is_mla_model)`` last), i.e. an MLA deployment with dcp
        owner mode or the canonical page and pp_size > 1. NOT the Weg-2 shape.
        """

        def body(d):
            stores = [
                HiCacheFile(
                    _p_config(
                        st, canonical=False, is_mla_model=True, dcp_owner_mode=True
                    ),
                    file_path=d,
                )
                for st in range(3)
            ]
            # Stages 1 and 2 first: on a key already on disk the
            # already-exists fast path answers before admission is asked.
            return stores, [
                stores[1].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[2].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[0].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
            ]

        with tempfile.TemporaryDirectory() as d:
            stores, wrote = self._bounded(lambda: body(d))
            self.assertEqual(
                sorted({s.kv_config_suffix for s in stores}),
                [
                    "_Qwen3.8-27B_520526c68d6530e9_3_0",
                    "_Qwen3.8-27B_520526c68d6530e9_3_1",
                    "_Qwen3.8-27B_520526c68d6530e9_3_2",
                ],
                "the premise of this row is that the KV suffix is per-stage here",
            )
            self.assertTrue(
                all(s._is_shared_kv_key(KEY) for s in stores),
                "the premise of this row is that KEY takes the KV branch",
            )
            self.assertEqual(
                wrote,
                [True, True, True],
                "an MLA dcp stage was refused the shared-KV file only it names",
            )
            found = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".bin")]
            self.assertEqual(
                len(found), 3, f"3 per-stage files expected, found {found}"
            )

    def test_an_mla_attn_cp_rank_writes_the_file_only_it_names(self):
        """The second unguarded axis, same shape as the PP one."""

        def body(d):
            stores = [HiCacheFile(_cp_config(r), file_path=d) for r in range(2)]
            return stores, [
                stores[1].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
                stores[0].set(KEY, torch.zeros(4096, dtype=torch.uint8)),
            ]

        with tempfile.TemporaryDirectory() as d:
            stores, wrote = self._bounded(lambda: body(d))
            self.assertNotEqual(
                stores[0].config_suffix,
                stores[1].config_suffix,
                "the premise of this test is that attn-cp ranks name two paths",
            )
            self.assertEqual(
                wrote,
                [True, True],
                "an MLA attn-cp rank was refused the file only it names",
            )
            found = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".bin")]
            self.assertEqual(len(found), 2, f"2 per-cp-rank files, found {found}")

    def test_an_admitted_non_owner_still_honours_the_free_space_watermark(self):
        """MB1: the watermark is the ONLY bound left on an admitted non-owner.

        A non-owner keeps no index (``test_a_non_owner_keeps_no_index_of_what
        _it_wrote``) and does not enforce the byte cap -- that is the owner's
        act. Four of Weg 2's six ranks are non-owners, so if this statvfs
        check goes they write with no bound whatsoever, and the sole carrier
        fills its filesystem until ``set`` rolls back on ENOSPC, which the ack
        path reports as a partial backup, i.e. as a cache miss (spec 3.3 (5)).
        """
        draft = f"{KEY}.draft-a30db4b7c362c786"
        with tempfile.TemporaryDirectory() as d:
            store = HiCacheFile(
                _d_config(1, extra_config={"max_size": "1G", "min_free_space": "1G"}),
                file_path=d,
            )
            self.assertFalse(
                store._evictor.is_storage_owner, "this test needs a non-owner"
            )
            total, _free = store._evictor._fs_stats()
            # The filesystem as it is when the watermark is the thing that
            # matters: 4 KiB left against a 1 GiB floor.
            store._evictor._fs_stats = lambda: (total, 4096)
            self.assertFalse(
                store.set(draft, torch.zeros(4096, dtype=torch.uint8)),
                "an admitted non-owner wrote past the free-space watermark",
            )
            found = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".bin")]
            self.assertEqual(found, [], "bytes landed despite the refusal")
            # CAN-FAIL COMPANION: with room on the filesystem the same write
            # lands, so the row above grades the watermark and not the
            # admission it rides on.
            store._evictor._fs_stats = lambda: (total, total)
            self.assertTrue(
                store.set(draft, torch.zeros(4096, dtype=torch.uint8)),
                "the watermark refused a write the filesystem can fund",
            )

    def test_a_non_owner_keeps_no_index_of_what_it_wrote(self):
        """The non-owner writes UNTRACKED: one index per directory, on the owner.

        A non-owner that indexed its own writes would be the second
        bookkeeping over the same files that F7 removes -- and it would evict
        from it, unlinking pages the owner still counts.
        """
        draft = f"{KEY}.draft-a30db4b7c362c786"

        def body(d):
            store = HiCacheFile(_d_config(1), file_path=d)
            ok = store.set(draft, torch.zeros(4096, dtype=torch.uint8))
            return store, ok

        with tempfile.TemporaryDirectory() as d:
            store, ok = self._bounded(lambda: body(d))
            self.assertTrue(ok)
            self.assertFalse(store._evictor.is_storage_owner)
            self.assertEqual(len(store._evictor._lru), 0)
            self.assertEqual(store._evictor._total_bytes, 0)
            self.assertEqual(len(store._evictor._pending_writes), 0)


class TestW8bTheGateIsGradedWhereTheNumberExists(CustomTestCase):
    """W8b's two blind spots: the SEAM and the MOMENT.

    Every other coverage test hands ``LRUFileEvictor`` its suffixes by hand,
    so the one production wiring (``HiCacheFile`` -> evictor) is never graded
    and re-pointing it at ``config_suffix`` -- the exact 83.7 %-blind defect --
    passes the suite. And the gate ran only in ``__init__``, which on Weg 2's
    first boot sees an EMPTY directory: coverage 1.0 over 0 bytes, a gate with
    no denominator. The populated moment is the wake re-scan.
    """

    def test_the_backend_wires_the_kv_suffix_into_its_evictor(self):
        with tempfile.TemporaryDirectory() as d:
            probe = HiCacheFile(_d_config(0), file_path=d)
            _seed_store(
                d,
                kv_suffix=probe.kv_config_suffix,
                cfg_suffix=probe.config_suffix,
                n_pages=40,
                n_drafts=10,
            )
            os.environ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"] = "1G"
            try:
                store = HiCacheFile(_d_config(0), file_path=d)
            finally:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            coverage = store._evictor.index_coverage()
            self.assertGreaterEqual(
                coverage["fraction"],
                0.95,
                f"the backend's own evictor indexes "
                f"{coverage['indexed_entries']} of {coverage['seen_entries']} "
                f"files ({coverage['fraction']:.1%})",
            )

    def test_the_wake_rescan_grades_the_coverage_it_rebuilt(self):
        """A cold store passes the gate vacuously; the sibling's writes are
        what the woken owner must be graded against."""
        from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind

        with tempfile.TemporaryDirectory() as d:
            evictor = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            self.assertEqual(evictor.index_coverage()["seen_entries"], 0)
            # The sibling group's hours of writes, under a suffix this scan
            # filter cannot see.
            for i in range(20):
                with open(os.path.join(d, f"{i:04d}_other.bin"), "wb") as f:
                    f.write(b"\x03" * 4096)
            with self.assertRaises(Weg2StoreIndexBlind):
                evictor.rescan()

    def test_a_wake_rescan_that_still_sees_the_store_passes(self):
        """CAN-FAIL GUARD: the wake gate must not refuse a healthy store."""
        with tempfile.TemporaryDirectory() as d:
            evictor = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            for i in range(20):
                with open(os.path.join(d, f"{i:04d}_sfx.bin"), "wb") as f:
                    f.write(b"\x03" * 4096)
            census = evictor.rescan()
            self.assertEqual(census["indexed_entries"], 20)


class TestS5RegressionGuards(CustomTestCase):
    """Two invariants the slice asserts in prose and graded nowhere."""

    def test_touch_reaches_the_inode_of_a_file_this_owner_never_indexed(self):
        """N2's actual case: the page the SIBLING wrote while this group slept.

        The indexed key takes the ``move_to_end`` branch; a cross-group hit
        takes the ADOPTION branch, and that is the branch whose recency has to
        reach the inode -- it is the only recency fact the sibling can read.
        """
        with tempfile.TemporaryDirectory() as d:
            evictor = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            self.assertEqual(
                len(evictor._lru), 0, "the adoption branch needs a cold index"
            )
            path = os.path.join(d, "page_sfx.bin")
            with open(path, "wb") as f:
                f.write(b"\x01" * 4096)
            old = 1_000_000.0
            os.utime(path, (old, old))
            evictor.touch("page_sfx", path)
            self.assertIn(
                "page_sfx", evictor._lru, "the sibling's file was not adopted"
            )
            self.assertGreater(
                os.stat(path).st_mtime,
                old,
                "an adopted cross-group hit left no recency the sibling can read",
            )

    def test_a_wake_rescan_preserves_an_in_flight_reservation(self):
        """``rescan``'s documented invariant, graded.

        A reservation belongs to this process and is not on disk yet, so
        re-reading the directory must not drop it -- dropping it under-counts
        ``_total_bytes`` and lets a concurrent ``reserve`` hand out space the
        pending write already holds.
        """
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a_sfx.bin"), "wb") as f:
                f.write(b"\x01" * 4096)
            evictor = LRUFileEvictor(
                d,
                "_sfx",
                kv_config_suffix=None,
                tp_rank=0,
                writes_shared_keys=True,
                extra_config={"max_size": "1G"},
            )
            before = evictor._total_bytes
            self.assertTrue(evictor.reserve("inflight_sfx", 4096))
            evictor.rescan()
            self.assertIn(
                "inflight_sfx",
                evictor._lru,
                "the wake re-scan dropped a write that is still in flight",
            )
            self.assertEqual(evictor._total_bytes, before + 4096)


if __name__ == "__main__":
    unittest.main()
