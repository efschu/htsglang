"""#1035c: a zero from the store probe must say WHICH zero it is.

THE DEFECT THIS PINS. `_storage_hit_query` computes `len(hash_value)` -- the
number of keys it put to the store -- one line before it returns, and then
returns only `(hash_value[:kv_hit_pages], kv_hit_pages * page_size)`. So an
empty key set and a fully-refused 8564-key question produced the byte-identical
`([], 0)`, and the zero that reached `completed_local` was indistinguishable
from that field's `__init__` default. Three states, one output.

The second half is `PoolTransferResult.extra_pool_hit_pages`: it records only
NON-ZERO boundaries (`if boundary: hit_count[name] = boundary`), so the pool
that capped a claim to exactly 0 is ABSENT from it. Its emptiness therefore
reads as "nothing capped this" and means "this capped it to nothing" -- the
`caps={}` misreading, made against the live `#1028B` line before this file
existed.

WHAT IS ASSERTED, and each assert names the mutant it kills:

* ``keys_asked`` survives the call. Mutant: restore
  ``PoolTransferResult(final_pages, hit_count)`` -> the field defaults to 0 and
  NEVER-ASKED becomes indistinguishable from ASKED-AND-NO. RED.
* ``kv_uncapped`` survives. Mutant: drop it -> CAPPED collapses into
  ASKED-AND-NO and a store holding pages reads as a store holding none. RED.
* ``zero_capped_pools`` names the pool that capped to zero. Mutant: delete the
  ``else: _zero_capped.append(...)`` arm -> the tuple is empty and `by=` says
  "-" while a pool did the capping. RED.
* the emitted line exists and carries the cause. Mutant: delete the emitter ->
  assertLogs raises. RED.

NO BEHAVIOUR IS ASSERTED TO CHANGE, and that is deliberate: every assert below
reads a REPORTING field or a log line. `kv_hit_pages` and
`extra_pool_hit_pages` are checked to be UNCHANGED by the widening, because a
reporting change that moves a decision is the defect this family already paid
for.

Hermetic: a tempdir store, no CUDA, no controller construction (the query is
driven against a stand-in `self`, which is what keeps this a unit test).
"""

import logging
import tempfile
import unittest
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.test_utils import CustomTestCase

PAGE_BYTES = 64
KEYS = ["aa01", "aa02", "aa03"]


def _store(tmpdir):
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
        model_name="unit-1035c",
    )
    return HiCacheFile(storage_config=cfg, file_path=tmpdir)


def _mamba_transfer(keys):
    return PoolTransfer(
        name=PoolName.MAMBA,
        keys=list(keys),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )


def _write(store, key, component=None):
    """Put one page on disk under exactly the name the probe will construct."""
    payload = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    store.set(key if component is None else f"{key}.{component}", payload)


class TestZeroAnswerPartition1035c(CustomTestCase):
    # ---------------------------------------------------------------- backend

    def test_never_asked_is_distinguishable(self):
        """keys_asked == 0 is the ONLY signal that nothing was put to the store."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            res = store.batch_exists_v2([], [_mamba_transfer([])], None)
            self.assertEqual(res.kv_hit_pages, 0)
            # THE MUTANT KILLER: without the widening this is 0 for BOTH the
            # never-asked and the asked-and-refused case.
            self.assertEqual(res.keys_asked, 0)
            self.assertEqual(res.kv_uncapped, 0)

    def test_asked_and_no_carries_the_denominator(self):
        """An empty store must report HOW MANY keys it was asked about."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            res = store.batch_exists_v2(KEYS, [_mamba_transfer(KEYS)], None)
            self.assertEqual(res.kv_hit_pages, 0)
            self.assertEqual(res.keys_asked, len(KEYS))  # mutant: 0
            self.assertEqual(res.kv_uncapped, 0)

    def test_capped_to_zero_names_the_pool(self):
        """KV pages present, no anchor -> claim 0, and the pool is NAMED.

        This is the `caps={}` trap in its exact live shape: the mamba boundary
        is 0, so `extra_pool_hit_pages` cannot hold it, and only
        `zero_capped_pools` distinguishes this from an empty store.
        """
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            for k in KEYS:
                _write(store, k)  # KV pages exist; no .mamba anywhere
            res = store.batch_exists_v2(KEYS, [_mamba_transfer(KEYS)], None)

            self.assertEqual(res.kv_hit_pages, 0, "the claim is capped to zero")
            self.assertEqual(res.kv_uncapped, len(KEYS))  # mutant: 0
            self.assertEqual(
                tuple(res.zero_capped_pools),
                (str(PoolName.MAMBA),),
                "the pool that capped to exactly 0 must be named",
            )
            # And the trap itself, pinned: the dict CANNOT carry it.
            self.assertNotIn(
                PoolName.MAMBA,
                res.extra_pool_hit_pages,
                "a zero boundary is absent from extra_pool_hit_pages BY "
                "CONSTRUCTION -- that is why zero_capped_pools exists",
            )

    def test_widening_changes_no_decision(self):
        """The two pre-existing fields are byte-identical to before."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            for k in KEYS:
                _write(store, k)
                _write(store, k, component=PoolName.MAMBA)
            res = store.batch_exists_v2(KEYS, [_mamba_transfer(KEYS)], None)
            self.assertEqual(res.kv_hit_pages, len(KEYS))
            self.assertEqual(res.extra_pool_hit_pages[PoolName.KV], len(KEYS))
            self.assertEqual(tuple(res.zero_capped_pools), ())

    def test_defaults_keep_every_old_construction_site_valid(self):
        """Two-positional construction must still work, unchanged."""
        r = PoolTransferResult(7, {})
        self.assertEqual((r.keys_asked, r.kv_uncapped, r.zero_capped_pools), (0, 0, ()))
        self.assertEqual(PoolTransferResult.empty().kv_hit_pages, 0)

    # ------------------------------------------------------------- the line

    def _drive_query(self, store, keys, transfers):
        """Call the real `_storage_hit_query` against a stand-in `self`."""
        me = SimpleNamespace(
            get_hash_str=lambda *a, **k: list(keys),
            storage_backend=store,
            page_size=1,
            extra_host_mem_release_entries=None,
        )
        op = SimpleNamespace(
            token_ids=[1, 2, 3],
            last_hash=None,
            prefix_keys=None,
            pool_transfers=transfers,
            pool_storage_result=PoolTransferResult.empty(),
        )
        return HybridCacheController._storage_hit_query(me, op)

    def test_zero_answer_emits_the_partition_line(self):
        """A zero result MUST produce the line, naming its cause."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            for k in KEYS:
                _write(store, k)  # KV present, anchor absent -> CAPPED
            with self.assertLogs(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
                level=logging.WARNING,
            ) as cm:
                out = self._drive_query(store, KEYS, [_mamba_transfer(KEYS)])
            self.assertEqual(out, ([], 0), "behaviour unchanged")
            line = "\n".join(cm.output)
            self.assertIn("#1035c ZERO-ANSWER PARTITION", line)
            self.assertIn("cause=CAPPED", line)
            self.assertIn(f"asked={len(KEYS)}", line)
            self.assertIn(f"kv_uncapped={len(KEYS)}", line)
            self.assertIn(f"by={PoolName.MAMBA}", line)

    def test_never_asked_line_says_never_asked(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            with self.assertLogs(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
                level=logging.WARNING,
            ) as cm:
                self._drive_query(store, [], [_mamba_transfer([])])
            line = "\n".join(cm.output)
            self.assertIn("cause=NEVER-ASKED", line)
            self.assertIn("asked=0", line)

    def test_asked_and_no_line_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            with self.assertLogs(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
                level=logging.WARNING,
            ) as cm:
                self._drive_query(store, KEYS, [_mamba_transfer(KEYS)])
            line = "\n".join(cm.output)
            self.assertIn("cause=ASKED-AND-NO", line)
            self.assertIn(f"asked={len(KEYS)}", line)

    def test_non_zero_answer_stays_silent(self):
        """The line is for the AMBIGUOUS case only; a real hit is attributable
        from `#1028B` already and must not gain a second emitter."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            for k in KEYS:
                _write(store, k)
                _write(store, k, component=PoolName.MAMBA)
            logger = logging.getLogger(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"
            )
            with unittest.mock.patch.object(logger, "warning") as warn:
                out = self._drive_query(store, KEYS, [_mamba_transfer(KEYS)])
            self.assertEqual(out[1], len(KEYS))
            self.assertFalse(
                any("#1035c" in str(c) for c in warn.call_args_list),
                "no line on a non-zero claim",
            )


if __name__ == "__main__":
    unittest.main()
