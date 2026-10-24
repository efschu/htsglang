"""27B lifetime gate: a host-tier hit is DEVICE demand, and the refusal line says so (24.10.).

THE READING THAT WAS WRONG. int4ar boot ``..._1023_223241`` D TP0 23:10:59 -> 23:14:07, 1280 refusals::

    PDFLIP D-ADMIT-REFUSED rid=pdflip-30-274 res=NO_TOKEN gate=lifetime price=63008 budget=32930.6
      born=58912 extend=58911 max_new=4096 prefix=0 uncached=1 parts=[available=42809 evictable=0 offset=9878 ...]

was read as "D prices the whole prompt although P/L2/L3 already hold it (prefix=0, uncached=1)". It does not
mis-price. ``prefix`` is D's DEVICE prefix; the 58910 tokens P computed sat in the shared store / L2
(``HiCache prefetch success ... matched=58127 loaded=783``, ``#1042 EXTENT ... held=58910`` -- the
LOAD-BACK extent, host side; front ``PDFLIP-USAGE-DETAILS ... cached_device=0 cached_l2=17966
cached_l3=40529``). A load-back writes those rows INTO D's device pool at admission, so they are charged
like computed rows: ``uncached=1`` is what D must COMPUTE, ``extend=58911`` is what D must ALLOCATE.
``available=42809`` could not take 58911 rows -- the pool was genuinely full (older 98k/115k/58k decodes,
SOLL R5: no decode seat -> it waits, decode later from L2/L3).

THE FIX IS THE LINE, NOT THE PRICE. Crediting the host-tier hit would admit 58911 rows into 42809 free
ones: the KV-OOM / pool-exhaustion direction. The lifetime refusal now names ``host_hit`` and
``load_back`` beside ``prefix``/``extend``, so ``extend - load_back`` (the compute) and ``load_back`` (the
rows the host read will occupy) are both on the line.

Pinned (CPU, real ``PrefillAdder.add_one_req`` up to the gate):
1. the metal specimen refuses with ``gate=lifetime price=63008`` and ``host_hit``/``load_back`` = 58910;
2. a host-tier hit gets NO credit (over-commit guard; mutant "credit host_hit" -> red);
3. a DEVICE-resident prefix IS credited (the price is extend + decode reserve + page);
4. the decode reserve and the ``>=`` boundary are charged (mutants "drop max_new" / ``>`` -> red);
5. the in-lock re-check carries the same terms (AST), and the D-ADMIT-REFUSED line prints them.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
import unittest
import unittest.mock
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.managers.pp_admission_congruence import LOAD_BACK_EXTENT_ATTR  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from flliper.srt.pdflip import d_admit_refusal as DAR  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

N = 58911  # pdflip-30-274's prompt
CLIP = int(sp.CLIP_MAX_NEW_TOKENS)
PAGE = 1  # 27B page_size
#: the gate right after the lifetime gate; reaching it means the lifetime gate let the request through
PASSED = "rem_input_no_chunk"


class _Adder(sp.PrefillAdder):
    _budget = 0

    @property
    def rem_total_tokens(self):  # the pool + evictable - reserves term, fixed per test
        return self._budget


def _adder(budget):
    a = object.__new__(_Adder)
    a._budget = budget
    a.prefill_spill_deep_taken = False
    a.pdflip_skip_extend_taken = False
    a.prefill_delayer_single_pass = None
    a.dsa_prefill_cp_in_seq_split = False
    # non-empty, so the no-chunk input gate right behind the lifetime gate refuses: that name proves
    # the lifetime gate passed without running the admission body
    a.can_run_list = [SimpleNamespace(rid="earlier")]
    a.prefill_max_requests = None
    a.tree_cache = SimpleNamespace(disable=False)
    a.scheduled_extents = None
    a.dllm_config = None
    a.page_size = PAGE
    a._mamba_slot_cost = 0
    a._mamba_slots_per_req = 0
    a.rem_mamba_slots = 0
    a.form_a_admission_follow = None
    a.tp_admission_agree = None
    a.prefill_spill_regions = 0
    a.prefill_spill_deep = False
    a.lifetime_refusal = None
    a.is_hybrid_swa = False
    a.rem_chunk_tokens = None
    a.rem_input_tokens = 0
    return a


def _req(n=N, device_prefix=0, host_hit=0, load_back=None, max_new=64000):
    r = SimpleNamespace(
        rid="pdflip-30-274",
        full_untruncated_fill_ids=[0] * n,
        prefix_indices=[0] * device_prefix,
        host_hit_length=host_hit,
        swa_host_hit_length=0,
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=max_new, ignore_eos=False),
        pdflip_parked_span=0,
        mamba_pool_idx=None,
        last_node=None,
        born_spilled=False,
        born_spilled_deep=False,
    )
    if load_back is not None:
        setattr(r, LOAD_BACK_EXTENT_ATTR, load_back)
    return r


def _run(budget, req):
    adder = _adder(budget)
    with unittest.mock.patch.object(sp, "_form_a_stale_vote_gate", lambda *_a, **_k: None), \
            unittest.mock.patch.object(sp, "_PDFLIP_CHUNK_ADMIT", False):
        res = adder.add_one_req(req, None)
    st = getattr(adder, DAR.STAMP_ATTR, None)
    return res, (st[1] if st else None), (dict(st[3]) if st else {}), adder


def _handback(**kw):
    """P computed N-1 into the store; D holds 0 on its device and loads N-1 back (the handback contract)."""
    return _req(device_prefix=0, host_hit=N - 1, load_back=N - 1, **kw)


class MetalSpecimen(CustomTestCase):
    def test_30_274_refuses_with_its_load_back_named(self):
        res, gate, nums, _ = _run(32930.6, _handback())
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        self.assertEqual(gate, "lifetime")
        self.assertEqual(nums["price"], N + CLIP + PAGE)            # 63008 with the 4096 clip
        self.assertEqual((nums["extend"], nums["prefix"]), (N, 0))  # D allocates every row ...
        self.assertEqual(nums["host_hit"], N - 1)                    # ... and N-1 of them come from host
        self.assertEqual(nums["load_back"], N - 1)

    def test_no_load_back_reads_zero(self):
        res, gate, nums, _ = _run(10, _req(n=100))
        self.assertEqual((res, gate), (AddReqResult.NO_TOKEN, "lifetime"))
        self.assertEqual((nums["host_hit"], nums["load_back"]), (0, 0))


class OverCommitGuard(CustomTestCase):
    """Gefahrenrichtung: KV-OOM / pool exhaustion. A host-tier hit occupies device rows once loaded."""

    def test_a_host_tier_hit_gets_no_credit(self):
        credited = 1 + CLIP + PAGE  # what a "subtract the loadable prefix" price would charge
        for budget in (credited + 1, 32930.6, N + CLIP + PAGE):  # the last one is the >= boundary
            with self.subTest(budget=budget):
                res, gate, _, _ = _run(budget, _handback())
                self.assertEqual((res, gate), (AddReqResult.NO_TOKEN, "lifetime"))

    def test_the_full_price_plus_one_passes(self):
        _res, gate, _, _ = _run(N + CLIP + PAGE + 1, _handback())
        self.assertEqual(gate, PASSED)


class DevicePrefixIsCredited(CustomTestCase):
    def test_a_device_resident_prefix_costs_nothing_new(self):
        price = 1 + CLIP + PAGE
        _res, gate, _, _ = _run(price + 1, _req(device_prefix=N - 1))
        self.assertEqual(gate, PASSED)

    def test_the_decode_reserve_and_the_boundary_are_charged(self):
        price = 1 + CLIP + PAGE
        res, gate, nums, _ = _run(price, _req(device_prefix=N - 1))  # budget == price -> refused (>=)
        self.assertEqual((res, gate), (AddReqResult.NO_TOKEN, "lifetime"))
        self.assertEqual((nums["price"], nums["max_new"], nums["prefix"]), (price, CLIP, N - 1))
        res, gate, _, _ = _run(1 + PAGE + 1, _req(device_prefix=N - 1, max_new=64000))  # no room for decode
        self.assertEqual((res, gate), (AddReqResult.NO_TOKEN, "lifetime"))


def _price_dicts():
    tree = ast.parse(textwrap.dedent(inspect.getsource(sp.PrefillAdder.add_one_req)))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            kws = {k.arg for k in node.keywords}
            if "price" in kws and "extend" in kws:
                out.append((node.lineno, kws))
    return out


class LineCarriesTheTerms(CustomTestCase):
    def test_both_lifetime_gates_carry_host_hit_and_load_back(self):
        dicts = _price_dicts()
        self.assertEqual(len(dicts), 2, dicts)  # pre-lock `lifetime` and in-lock `lifetime_locked`
        for line, kws in dicts:
            with self.subTest(line=line):
                self.assertIn("host_hit", kws)
                self.assertIn("load_back", kws)

    def test_the_refusal_line_prints_them(self):
        res, _gate, _nums, adder = _run(32930.6, _handback())
        req = _handback()
        req._pdflip_xgate_priced = (1, 4097)
        line = DAR.note_not_added(SimpleNamespace(), adder, req, res, site="admission_revert",
                                  env={"FLLIPER_PDFLIP_GROUP": "D"})
        self.assertIsNotNone(line)
        self.assertIn("gate=lifetime", line)
        self.assertIn(f"price={N + CLIP + PAGE}", line)
        self.assertIn(f"host_hit={N - 1} load_back={N - 1}", line)
        self.assertIn("uncached=1", line)


if __name__ == "__main__":
    unittest.main()
