# SPDX-License-Identifier: Apache-2.0
"""Item 170 (open point of 120): D's allocator-cache growth is booked into the 5090's card ledger.

The dual layout runs the 5090 (nvml1) at ~57 MiB physical free. Boot
dkr27bnvfp4dual1mpsleepbar1fs10020527 (D.log): LEDGER-PHYS OVER-PROMISE up to 343 MiB
(:18702), CORRIDOR LAW BREACHED min 56 MiB (:32484), one cuMemCreate OOM rolled back by
GROW-SHORT (:32516) -- while D's own torch allocator cache swung 270 MiB in the same window
(:32482 '#1028c BOUND alloc_cache ... bound=1378.4 instant=1648.1 delta=269.7'; highwater 1665.6 at :42878). The cache starts at the same sample in
every dual boot (TP0 348.4 MiB, the cache when the pools were sized) and grows with traffic;
the ledger budget (the sum of the groups' boot KV) knows nothing of it and books it after the
fact, 3 OVER-PROMISE checks (30 s) at a time.

The fix is ACCOUNTING, not a reserve: a measured record (RECORD > BUILTIN > UNMEASURED, H94)
of the growth the ledger had to book, priced at P's join (CardKvLedger.book_unpriced) on the
card where it binds. These tests pin the record against the series (fixture lines are verbatim
from the evidence boots), the ledger books, the P-side hook and the launcher wiring.
"""

import os
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import budget_rest as BR  # noqa: E402
from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as PK  # noqa: E402
from sglang.srt.weg2 import form as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1 << 20


def _bound(ts, tp, bound, inst):
    return (f"[2026-10-02 {ts} TP{tp}] #1028c BOUND alloc_cache: window=4.3s n=5 bound={bound:.1f} MiB "
            f"instant={inst:.1f} MiB delta={inst - bound:.1f} MiB (bound is a MEASURED MINIMUM over the "
            f"window, not a spot value; it can only narrow)")


def _booked(ts, tp, n):
    return (f"[2026-10-02 {ts} TP{tp}] DUAL-TP3PP3 P-KV LEDGER-PHYS BOOKED {n} B of unbooked card use "
            f"into the budget (group=D, 3 checks over 30 s)")


def _boot(first, high, booked_mib):
    """One D log: per rank the first sample, the highwater, and the BOOKED lines."""
    out = []
    for tp in range(3):
        out.append(_bound("05:29:09", tp, first[tp], first[tp]))
        out.append(_bound("05:35:43", tp, high[tp] - 270.0, high[tp]))
        if booked_mib[tp]:
            out.append(_booked("05:33:34", tp, int(booked_mib[tp] * MIB)))
    return "\n".join(out)


FIRST = (348.4, 266.5, 258.7)  # D.log:6318 / :6334 / :6327 (identical in every dual boot)
#: (tag, first, highwater, LEDGER-PHYS BOOKED MiB) -- the four newest serving boots of
#: 27b-nvfp4-dual1m-psleep, values from their D logs
BOOTS = (
    ("dkr27bnvfp4dual1mpsleepbar1fs10020527", (1665.6, 1757.9, 1766.2), (490.0, 162.5, 120.5)),
    ("dkr27bnvfp4dual1mpsleepbar1fs10020256", (990.5, 969.2, 967.2), (583.3, 0, 0)),
    ("dkr27bnvfp4dual1mpsleepbar1fs10020145", (1002.7, 1399.8, 1342.1), (533.3, 24.5, 0)),
    ("dkr27bnvfp4dual1mpsleepbar1fs10020008", (998.6, 1530.1, 1466.2), (445.3, 0, 0)),
)


def _boots():
    return [(tag, _boot(FIRST, hi, bk)) for tag, hi, bk in BOOTS]


class TheSeriesIsTheMeasurement(CustomTestCase):
    def test_a_rank_reads_first_highwater_and_booked(self):
        cs = BR.cache_series(_boots()[0][1])
        self.assertEqual((cs[0].first, cs[0].highwater), (348.4, 1665.6))
        self.assertAlmostEqual(cs[0].growth, 1317.2, places=1)
        self.assertEqual(cs[0].booked_b, 490 * MIB)
        self.assertEqual(cs[1].booked_b, int(162.5 * MIB))

    def test_the_record_books_the_5090_and_prices_the_3080s_as_absorbed(self):
        vals, lines = BR.alloc_cache_book_from_boots(_boots(), 3)
        # nvml1 (ordinal 0): the ledger booked card use in 4/4 boots -> max growth 1317.2, rounded UP;
        # nvml0 2/4 and nvml2 1/4 are no strict majority -> 0, named
        self.assertEqual(vals, [1318, 0, 0])
        self.assertTrue(any("ordinal=0: BOOK 1318 MiB" in ln and "4/4" in ln for ln in lines))
        self.assertTrue(any("ordinal=1: 0 MiB" in ln and "2/4" in ln and "reserve" in ln for ln in lines))
        self.assertTrue(any("ordinal=2: 0 MiB" in ln and "1/4" in ln for ln in lines))

    def test_an_unsampled_rank_is_unmeasured_never_zero(self):
        txt = _bound("05:29:09", 0, 348.4, 348.4) + "\n" + _booked("05:33:34", 0, 100 * MIB)
        vals, lines = BR.alloc_cache_book_from_boots([("b", txt)], 3)
        self.assertEqual(vals, [0, None, None])  # one boot, 1/1 booked, but growth 0.0 -> 0
        self.assertTrue(any("ordinal=1" in ln and "UNMEASURED" in ln for ln in lines))

    def test_the_registry_record_is_that_measurement(self):
        rec = F.profile_record(BR.alloc_cache_record_name("D"), F.PROFILE_QWEN27B, fmt="nvfp4").record
        self.assertEqual(list(rec.value), BR.alloc_cache_book_from_boots(_boots(), 3)[0])
        self.assertEqual(rec.fmt, "nvfp4")
        # same checkpoint and form only: the INT8 form finds no record
        self.assertIsNone(F.profile_record(BR.alloc_cache_record_name("D"), F.PROFILE_QWEN27B, fmt="int8"))
        for tag, _, _ in BOOTS:
            self.assertIn(tag, rec.boots)


class TheLedgerBooksAtTheJoin(CustomTestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(prefix="wkv"), "card")

    def _joined(self):
        d = K.CardKvLedger(self.path, "D")
        p = K.CardKvLedger(self.path, "P")
        # the real flow (dual_d_kv_stage / dual_p_kv_stage.attach): D contributes its boot KV mapped, shrinks
        # it back to the pool; P's sizing then counts the bytes D released as free, so its boot KV of
        # 4496 MiB adds only 4496 - 2560 = 1936 MiB to the budget
        d.contribute(2560 * MIB, committed=2560 * MIB)
        d.release(2560 * MIB)
        p.contribute(4496 * MIB, committed=0)
        return d, p

    def test_the_budget_falls_by_what_is_booked_from_the_callers_contribution(self):
        d, p = self._joined()
        before = p.state()
        self.assertEqual(before.budget, 4496 * MIB)
        self.assertEqual(p.book_unpriced(1318 * MIB), 1318 * MIB)
        st = p.state()
        self.assertEqual(st.budget, (4496 - 1318) * MIB)
        self.assertEqual(st.contrib["P"], (1936 - 1318) * MIB)
        self.assertEqual(st.contrib["D"], 2560 * MIB)  # D's share is untouched while P's covers it

    def test_what_exceeds_p_contribution_comes_from_the_other_group(self):
        d, p = self._joined()
        self.assertEqual(p.book_unpriced(2000 * MIB), 2000 * MIB)
        st = p.state()
        self.assertEqual((st.contrib["P"], st.contrib["D"]), (0, (2560 - 64) * MIB))
        self.assertEqual(st.budget, (4496 - 2000) * MIB)

    def test_never_below_what_is_committed_i1(self):
        d, p = self._joined()
        d.request(3000 * MIB)
        self.assertEqual(p.book_unpriced(2000 * MIB), (4496 - 3000) * MIB)  # only the free part
        st = p.state()
        self.assertGreaterEqual(st.budget, sum(st.committed.values()))
        self.assertEqual(p.book_unpriced(1), 0)

    def test_zero_books_nothing(self):
        d, p = self._joined()
        self.assertEqual(p.book_unpriced(0), 0)
        self.assertEqual(p.state().budget, 4496 * MIB)


class TheStageHookReadsItsOwnCard(CustomTestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(prefix="wkv"), "card")
        self.led = K.CardKvLedger(self.path, "P")
        self.led.contribute(4496 * MIB, committed=0)

    def test_env_vector_is_indexed_by_the_stage(self):
        env = {PK.ALLOC_CACHE_BOOK_ENV: "1318,0,"}
        self.assertEqual(PK.alloc_cache_book_mib(0, env), 1318)
        self.assertEqual(PK.alloc_cache_book_mib(1, env), 0)
        self.assertIsNone(PK.alloc_cache_book_mib(2, env))   # empty entry = UNMEASURED
        self.assertIsNone(PK.alloc_cache_book_mib(3, env))   # no entry for this stage
        self.assertIsNone(PK.alloc_cache_book_mib(0, {}))     # no record at all

    def test_the_join_books_on_the_priced_card_and_only_there(self):
        env = {PK.ALLOC_CACHE_BOOK_ENV: "1318,0,"}
        self.assertEqual(PK.book_alloc_cache(self.led, 0, "card-a", env), 1318 * MIB)
        self.assertEqual(self.led.state().budget, (4496 - 1318) * MIB)
        self.assertEqual(PK.book_alloc_cache(self.led, 1, "card-b", env), 0)   # absorbed card
        self.assertEqual(PK.book_alloc_cache(self.led, 2, "card-c", env), 0)   # unmeasured card
        self.assertEqual(self.led.state().budget, (4496 - 1318) * MIB)

    def test_without_the_record_nothing_changes(self):
        self.assertEqual(PK.book_alloc_cache(self.led, 0, "card-a", {}), 0)
        self.assertEqual(self.led.state().budget, 4496 * MIB)


class TheLauncherPricesFromTheRecordOfThisCheckpointAndForm(CustomTestCase):
    def _model(self, quant_method):
        d = tempfile.mkdtemp(prefix="m27b")
        import json

        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"],
                       "quantization_config": {"quant_method": quant_method}}, f)
        return d

    def _cards(self):
        return [L.Card(nvml_index=1, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total_mib=32607),
                L.Card(nvml_index=0, uuid="GPU-b", name="NVIDIA GeForce RTX 3080", total_mib=20480),
                L.Card(nvml_index=2, uuid="GPU-c", name="NVIDIA GeForce RTX 3080", total_mib=20480)]

    def _ns(self, model, **kw):
        base = dict(dual_share=True, dual_unified_kv="on", model=model, profile=F.PROFILE_QWEN27B)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_nvfp4_dual_share_gets_the_vector_and_a_record_line(self):
        lines = []
        env = L.dual_alloc_cache_book_env(self._ns(self._model("modelopt")), self._cards(), lines.append)
        self.assertEqual(env, {PK.ALLOC_CACHE_BOOK_ENV: "1318,0,0"})
        self.assertTrue(any(ln.startswith("WEG2-ALLOC-CACHE-BOOK RECORD D_DUAL_ALLOC_CACHE_BOOK_MIB fmt=nvfp4")
                            and "ordinal 0 nvml1 1318 MiB" in ln for ln in lines), lines)

    def test_int8_checkpoint_is_unmeasured_and_books_nothing(self):
        lines = []
        env = L.dual_alloc_cache_book_env(self._ns(self._model("compressed-tensors")), self._cards(),
                                          lines.append)
        self.assertEqual(env, {})
        self.assertTrue(any("WEG2-ALLOC-CACHE-BOOK UNMEASURED" in ln and "fmt=int8" in ln for ln in lines),
                        lines)

    def test_other_layouts_and_profiles_are_untouched(self):
        lines = []
        self.assertEqual(L.dual_alloc_cache_book_env(
            self._ns(self._model("modelopt"), dual_share=False), self._cards(), lines.append), {})
        self.assertEqual(L.dual_alloc_cache_book_env(
            self._ns(self._model("modelopt"), dual_unified_kv="off"), self._cards(), lines.append), {})
        self.assertEqual(lines, [])  # off: byte-identical, no line
        env = L.dual_alloc_cache_book_env(self._ns(self._model("modelopt"), profile=F.PROFILE_NEXTFLASH),
                                          self._cards(), lines.append)
        self.assertEqual(env, {})
        self.assertTrue(any("UNMEASURED" in ln for ln in lines))

    def test_a_record_of_another_card_count_is_refused(self):
        with self.assertRaises(L.Weg2LaunchRefused):
            L.dual_alloc_cache_book_env(self._ns(self._model("modelopt")), self._cards()[:2], lambda _l: None)
