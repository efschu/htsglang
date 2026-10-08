"""L3-PAIR (30.09., NF y3u 5bedac26f1, pdflip-0-5): the L3 LRU must never evict the
QSA index page of a KV page that stays on disk.

Metal: D's L3 owner unlinked 8 ``{h}.qsa_indexer{sfx}`` files at 00:35:43
(journal ``E``) whose KV pages ``{h}{sfx}`` stayed on disk -- in the store index
at the attach the index page of page 47 sat at position 46 of 174419
(mtime 21:41:25), its KV page at 139279 (00:10:24). D's resume then read
``#1028B FETCH CAP kv=1996 claimed=47 caps={qsa_indexer: 47}``, refused the
re-prefill (W50 x_refusal_midstream) and P recomputed 127813 tokens with
cached=0 (32.49 s).

Red on the base (the index page is the oldest entry and goes first), green with
the pair rule. CPU only.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import tempfile
import time
import unittest
from unittest import mock

from flliper.srt.environ import envs
from flliper.srt.mem_cache.hicache_storage import PoolName
from flliper.srt.mem_cache.storage.file import lru_file_evictor as LFE
from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SFX = "_Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist_898fe1bf454ff7c1"
KV_BYTES = 8192
QSA_BYTES = 4096


def _h(i: int) -> str:
    return f"{i:064x}"


def _kv(i: int) -> str:
    return f"{_h(i)}{SFX}"


def _qsa(i: int) -> str:
    return f"{_h(i)}.qsa_indexer{SFX}"


def _write(d: str, stem: str, n: int, mtime: float) -> None:
    p = os.path.join(d, stem + ".bin")
    with open(p, "wb") as f:
        f.write(b"\x07" * n)
    os.utime(p, (mtime, mtime))


def _ev(d: str) -> LRUFileEvictor:
    def _iter_existing():
        for name in os.listdir(d):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_STORE_JOURNAL": "0"}):
        return LRUFileEvictor(d, SFX, tp_rank=0, pp_rank=0, attn_cp_rank=0,
                              writes_shared_keys=True, scan_suffixes=(SFX,),
                              extra_config={"max_size": str(10 ** 9), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing)


def _exists(d: str, stem: str) -> bool:
    return os.path.exists(os.path.join(d, stem + ".bin"))


def _evict_bytes(ev: LRUFileEvictor, n_bytes: int) -> int:
    with ev._lock:
        return ev._evict_while(lambda r: r < n_bytes)


class TestL3SidecarPairEvict(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name
        self.t0 = time.time() - 10_000

    def tearDown(self):
        self._td.cleanup()

    def _y3u_store(self):
        """Page 47: index page OLDEST, KV page young (the y3u store at the attach).
        Pages 1..3: old pairs (both files older than page 47's KV page)."""
        _write(self.d, _qsa(47), QSA_BYTES, self.t0)                # 21:41:25
        for i in (1, 2, 3):
            _write(self.d, _kv(i), KV_BYTES, self.t0 + 10 * i)
            _write(self.d, _qsa(i), QSA_BYTES, self.t0 + 10 * i + 1)
        _write(self.d, _kv(47), KV_BYTES, self.t0 + 1000)            # 00:10:24

    def test_stem_twins_match_the_store_and_the_enum(self):
        self.assertEqual(LFE._PAIR_SIDECAR_TAGS, (f".{PoolName.QSA_INDEXER}",))
        real_kv = ("904ed9b170b2a4021fa5e66a461c413ae318e697eaece746aa1121c01c092167"
                   "_Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist_898fe1bf454ff7c1")
        real_q = ("904ed9b170b2a4021fa5e66a461c413ae318e697eaece746aa1121c01c092167"
                  ".qsa_indexer_Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist_898fe1bf454ff7c1")
        self.assertEqual(LFE._sidecar_kv_twin(real_q), real_kv)
        self.assertEqual(LFE._kv_sidecar_twins(real_kv), (real_q,))
        self.assertIsNone(LFE._sidecar_kv_twin(real_kv))
        self.assertEqual(LFE._kv_sidecar_twins(real_q), ())
        mamba = real_kv.replace("_Qwen", ".mamba_Qwen", 1)
        self.assertIsNone(LFE._sidecar_kv_twin(mamba))
        self.assertEqual(LFE._kv_sidecar_twins(mamba), ())

    def test_index_page_of_a_kv_page_on_disk_is_never_the_victim(self):
        """THE y3u CASE. Base: the oldest entry (page 47's index page) is
        unlinked first while its KV page stays -> the claim caps at 47."""
        self._y3u_store()
        ev = _ev(self.d)
        _evict_bytes(ev, 1)  # one victim: whatever the LRU holds oldest
        self.assertTrue(_exists(self.d, _kv(47)))
        self.assertTrue(_exists(self.d, _qsa(47)),
                        "QSA index page evicted while its KV page stays on disk "
                        "(y3u: FETCH CAP caps={qsa_indexer: 47})")
        # the victim was the oldest real pair's KV page, and its index page went with it
        self.assertFalse(_exists(self.d, _kv(1)))
        self.assertFalse(_exists(self.d, _qsa(1)))
        self.assertTrue(_exists(self.d, _kv(2)) and _exists(self.d, _qsa(2)))

    def test_no_kv_page_on_disk_without_its_index_page_after_a_run(self):
        self._y3u_store()
        ev = _ev(self.d)
        before = ev._total_bytes
        freed = _evict_bytes(ev, 2 * (KV_BYTES + QSA_BYTES))
        for i in (1, 2, 3, 47):
            if _exists(self.d, _kv(i)):
                self.assertTrue(_exists(self.d, _qsa(i)), f"page {i}: KV on disk without its index")
        self.assertEqual(before - ev._total_bytes, freed)
        self.assertGreaterEqual(ev._pair_with_kv, 2)
        self.assertGreaterEqual(ev._pair_deferred, 1)

    def test_orphan_index_page_is_evicted_as_before(self):
        """An index page whose KV page is on no disk is a plain victim."""
        _write(self.d, _qsa(9), QSA_BYTES, self.t0)
        _write(self.d, _kv(1), KV_BYTES, self.t0 + 10)
        _write(self.d, _qsa(1), QSA_BYTES, self.t0 + 11)
        ev = _ev(self.d)
        _evict_bytes(ev, 1)
        self.assertFalse(_exists(self.d, _qsa(9)))
        self.assertTrue(_exists(self.d, _kv(1)) and _exists(self.d, _qsa(1)))

    def test_kv_page_only_on_disk_protects_too(self):
        """The KV page may be outside this owner's index (the sibling group's
        write, or one after the attach): the file itself answers."""
        _write(self.d, _qsa(47), QSA_BYTES, self.t0)
        _write(self.d, _kv(1), KV_BYTES, self.t0 + 10)
        ev = _ev(self.d)
        _write(self.d, _kv(47), KV_BYTES, self.t0 + 20)   # after the attach, not indexed here
        self.assertNotIn(_kv(47), ev._lru)
        _evict_bytes(ev, 1)
        self.assertTrue(_exists(self.d, _qsa(47)))
        self.assertFalse(_exists(self.d, _kv(1)))

    def test_switch_off_is_the_old_path(self):
        self._y3u_store()
        with envs.FLLIPER_HICACHE_L3_SIDECAR_PAIR_EVICT.override(False):
            ev = _ev(self.d)
        _evict_bytes(ev, 1)
        self.assertFalse(_exists(self.d, _qsa(47)))
        self.assertTrue(_exists(self.d, _kv(47)))
        self.assertTrue(_exists(self.d, _qsa(1)))

    def test_switch_default_on(self):
        self.assertTrue(envs.FLLIPER_HICACHE_L3_SIDECAR_PAIR_EVICT.get())


if __name__ == "__main__":
    unittest.main()
