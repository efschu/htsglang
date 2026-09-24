"""Mamba admission debits are read before the host load-back binds the slot (#36415).

Ported from upstream sglang #36415 ("... an undebited Mamba admission slot").
``PrefillAdder.add_one_req`` charged the request's mamba cost twice from
helpers that key on ``req.mamba_pool_idx is None``: once into ``total_tokens``
before ``tree_cache.init_load_back`` and again at the ``_update_prefill_budget``
debit sites AFTER it. A host load-back binds ``req.mamba_pool_idx``
(``MambaComponent.build_hicache_transfers``), so the second read returned 0
and the request's new state was never debited.

Upstream fixed the shared-gap reserve (``_mamba_gap_budget_for_req``, unified
Mamba pool only -- off on this line). The fork's own non-unified slot gate
(#581/#1044: ``rem_mamba_slots`` on ``HybridReqToTokenPool``, i.e. the 27B's
pool) is charged through ``_mamba_slots_for_req``, which has the same trap:
after a load-back the admission debited 0 of the ``_mamba_slots_per_req``
slots the request holds, and the next request of the pass could be admitted
against slots that do not exist (the #581 assert class). Both reads are
hoisted above the load-back and reused at the debit sites.

Pinned here (CPU):
1. the trap itself, on the real helpers: both return their charge for an
   unbound request and 0 once ``mamba_pool_idx`` is bound;
2. ``add_one_req`` reads each helper exactly once, before ``init_load_back``,
   and every ``_update_prefill_budget`` call passes the hoisted values (AST;
   on the base file the helpers are re-read after the load-back -> red);
3. the budget debit actually consumes the hoisted charge.
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

from sglang.srt.managers import schedule_policy as sp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _adder(slot_cost=0, slots_per_req=3, rem_slots=10):
    adder = object.__new__(sp.PrefillAdder)
    adder._mamba_slot_cost = slot_cost
    adder._mamba_slots_per_req = slots_per_req
    adder.rem_mamba_slots = rem_slots
    return adder


class TestTheTrap(CustomTestCase):
    def test_helpers_return_zero_once_the_slot_is_bound(self):
        adder = _adder(slot_cost=7, slots_per_req=3)
        req = SimpleNamespace(mamba_pool_idx=None)
        self.assertEqual(adder._mamba_gap_budget_for_req(req), 7)
        self.assertEqual(adder._mamba_slots_for_req(req), 3)
        req.mamba_pool_idx = 42  # what a host load-back does
        self.assertEqual(adder._mamba_gap_budget_for_req(req), 0)
        self.assertEqual(adder._mamba_slots_for_req(req), 0)


def _add_one_req_ast():
    src = textwrap.dedent(inspect.getsource(sp.PrefillAdder.add_one_req))
    return ast.parse(src)


def _calls(tree, attr):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


class TestDebitOrder(CustomTestCase):
    def test_each_helper_is_read_once_and_before_the_load_back(self):
        tree = _add_one_req_ast()
        load_backs = _calls(tree, "init_load_back")
        self.assertEqual(len(load_backs), 1)
        first_load_back = load_backs[0].lineno
        for helper in ("_mamba_gap_budget_for_req", "_mamba_slots_for_req"):
            with self.subTest(helper=helper):
                calls = _calls(tree, helper)
                self.assertEqual(len(calls), 1, f"{helper} read {len(calls)}x")
                self.assertLess(calls[0].lineno, first_load_back)

    def test_every_budget_debit_passes_the_hoisted_values(self):
        tree = _add_one_req_ast()
        debits = _calls(tree, "_update_prefill_budget")
        self.assertGreaterEqual(len(debits), 2)
        for call in debits:
            kw = {k.arg: k.value for k in call.keywords}
            with self.subTest(line=call.lineno):
                self.assertIsInstance(kw.get("mamba_gap_reserve"), ast.Name)
                self.assertEqual(kw["mamba_gap_reserve"].id, "mamba_gap_reserve")
                self.assertIsInstance(kw.get("mamba_slot_charge"), ast.Name)
                self.assertEqual(kw["mamba_slot_charge"].id, "mamba_slot_charge")


class TestDebitConsumesTheCharge(CustomTestCase):
    def budget_adder(self):
        adder = _adder(slot_cost=0, slots_per_req=3, rem_slots=4)
        adder.page_size = 1
        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 0
        adder.rem_input_tokens = 1000
        adder.is_hybrid_swa = False
        adder.dllm_config = None
        adder.rem_chunk_tokens = None
        adder.log_hit_tokens = adder.log_input_tokens = 0
        adder.reprocessed_log_hit_tokens = adder.reprocessed_log_input_tokens = 0
        return adder

    def test_a_load_back_admission_is_debited_its_slots(self):
        """The 27B shape: non-unified pool (gap cost 0), slot gate on. The
        hoisted charge (read while the slot was unbound) is what the debit
        consumes; a post-load-back re-read would have consumed 0."""
        adder = self.budget_adder()
        req = SimpleNamespace(mamba_pool_idx=None)
        hoisted = adder._mamba_slots_for_req(req)  # read before the load-back
        req.mamba_pool_idx = 5  # the load-back binds the slot
        adder._update_prefill_budget(
            0, 8, 16, False, mamba_gap_reserve=0, mamba_slot_charge=hoisted
        )
        self.assertEqual(adder.rem_mamba_slots, 1)
        # the old order: re-read after the load-back -> nothing debited
        stale = self.budget_adder()
        stale._update_prefill_budget(
            0, 8, 16, False, mamba_slot_charge=stale._mamba_slots_for_req(req)
        )
        self.assertEqual(stale.rem_mamba_slots, 4)


if __name__ == "__main__":
    unittest.main()
