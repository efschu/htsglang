"""#631: an admitted request with no pool slot must not break live-slot enumeration.

MEASURED FAILURE this pins (2026-08-09, metal, all three ranks down):

    phase_flip_runtime.py in _live
      return torch.unique(torch.cat(parts))
    RuntimeError: Tensors must have same number of dimensions: got 1 and 3

THE MECHANISM, and it is a shape bug wearing a lookup's clothes.
``Req.req_pool_idx`` is ``Optional[int]`` and starts as None; a Req is
visible in ``last_batch`` / ``running_mbs`` / ``chunked_req`` from the
moment it is admitted, which is BEFORE a slot is allocated for it. The
enumeration indexed ``req_to_token[req.req_pool_idx, :n]`` regardless, and
None is not "no row" -- it is numpy-style newaxis, so a 2-D (R, C) table
returns a 3-D (1, n, C) tensor. Concatenating that against the tree's 1-D
values raises, inside the flip, and the exception climbs into the event
loop and takes the instance down.

WHY SKIPPING IS THE CORRECT ANSWER and not a papering-over: ``req_to_token``
is indexed BY ``req_pool_idx``. A request that has none cannot have a row
there, so it holds no KV the flip could leave behind -- which is the one
thing ``build_flip_live_slots_fn`` must never do (rows not enumerated are
not moved, and the request's context is then silently wrong). Its slot is
allocated later, in whatever layout is current by then. Reshaping the 3-D
part to 1-D, by contrast, would have injected ``max_context_len`` worth of
unrelated integers into the live set as if they were slot ids: the flip
would move the wrong rows and corrupt quietly instead of crashing loudly.

It is NOT a long-context-specific defect. The long session merely made
flips get armed and re-attempted far more often (each attempt abandoned for
staging room while the pool was full), which widened the window in which
the hook fires at the same moment a fresh request is admitted.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.phase_flip_runtime import build_flip_live_slots_fn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MAX_REQS = 4
MAX_CTX = 64


def _req(rid, seqlen, pool_idx):
    return SimpleNamespace(rid=rid, seqlen=seqlen, req_pool_idx=pool_idx)


def _scheduler(reqs, tree_values=None):
    table = torch.arange(MAX_REQS * MAX_CTX, dtype=torch.int32).reshape(
        MAX_REQS, MAX_CTX
    )
    tree = SimpleNamespace(all_values_flatten=lambda: tree_values)
    return SimpleNamespace(
        tree_cache=tree,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        running_mbs=[SimpleNamespace(reqs=reqs)],
        running_batch=None,
        last_batch=None,
        chunked_req=None,
    )


class TestLiveSlotsWithoutAPoolSlot(CustomTestCase):
    def test_the_measured_crash_is_gone(self):
        # CAN-FAIL: with the guard removed this raises
        # "Tensors must have same number of dimensions: got 1 and 3".
        sched = _scheduler(
            [_req("allocated", 3, 1), _req("just-admitted", 5, None)],
            tree_values=torch.tensor([900, 901], dtype=torch.int32),
        )
        live = build_flip_live_slots_fn(sched)()
        self.assertEqual(live.dim(), 1)

    def test_the_unallocated_request_contributes_nothing(self):
        # Row 1 of the table is 64..127; the first 3 entries are the only
        # rows the allocated request owns. The unallocated one must add
        # NOTHING -- not its would-be row, not the newaxis fallout.
        sched = _scheduler(
            [_req("allocated", 3, 1), _req("just-admitted", 5, None)],
            tree_values=None,
        )
        live = build_flip_live_slots_fn(sched)()
        self.assertEqual(live.tolist(), [64, 65, 66])

    def test_allocated_requests_are_still_fully_enumerated(self):
        # The property the whole function exists for: every row of every
        # allocated request is present, unioned with the tree.
        sched = _scheduler(
            [_req("a", 2, 0), _req("b", 3, 2)],
            tree_values=torch.tensor([5000], dtype=torch.int32),
        )
        live = build_flip_live_slots_fn(sched)()
        self.assertEqual(live.tolist(), [0, 1, 128, 129, 130, 5000])

    def test_slot_zero_is_not_confused_with_no_slot(self):
        # req_pool_idx 0 is a REAL slot. A truthiness test instead of an
        # `is None` test would drop it and lose that request's KV -- the
        # silent-wrong-context class.
        sched = _scheduler([_req("a", 2, 0)], tree_values=None)
        self.assertEqual(build_flip_live_slots_fn(sched)().tolist(), [0, 1])

    def test_every_request_unallocated_yields_an_empty_int64_tensor(self):
        sched = _scheduler(
            [_req("x", 4, None), _req("y", 7, None)], tree_values=None
        )
        live = build_flip_live_slots_fn(sched)()
        self.assertEqual(live.numel(), 0)
        self.assertEqual(live.dtype, torch.int64)

    def test_the_newaxis_mechanism_is_what_it_looks_like(self):
        # Pins the CAUSE, so a future reader does not re-derive it: None as
        # an index is newaxis, and that is where the third dimension came
        # from.
        table = torch.zeros(MAX_REQS, MAX_CTX, dtype=torch.int32)
        self.assertEqual(table[None, :5].dim(), 3)
        self.assertEqual(table[1, :5].dim(), 1)


if __name__ == "__main__":
    unittest.main()
