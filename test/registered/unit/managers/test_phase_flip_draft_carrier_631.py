# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 spec item 6, rung 2: the draft weights on a VA-stable carrier.

WHAT THESE TESTS PIN, and why each one is here rather than assumed.

* THE ADDRESS DOES NOT MOVE. This is the entire reason rung 2 was refused for
  four shifts and the entire content of the fix. The TP decode CUDA graphs
  bake the drafter's parameter addresses at capture; the previous
  implementation restored into a FRESHLY allocated arena, which moves them,
  and the corruption is SILENT -- no exception, just wrong draft logits and a
  decaying accept rate. ``test_addresses_survive_a_full_cycle`` is the pin,
  and ``test_a_fresh_arena_restore_would_move_them`` is its control: it
  performs the OLD restore against the same fixture and asserts the addresses
  DO move, so the first test cannot pass vacuously on a fixture too small or
  too static to relocate anything.

* THE BYTES COME BACK EXACTLY. The fake arena SCRIBBLES the span on decommit
  (0xA5 over every byte) precisely so that "restore" cannot be satisfied by
  doing nothing. Without the scribble, a carrier that released no pages at all
  would pass a byte-identity test trivially -- and "released nothing" is the
  failure mode that turns a capacity headline into a measured zero, which has
  happened six times in this chain.

* THE SPILL ACTUALLY RELEASES. ``decommit_range`` is called with keep=0 and
  its reported bytes are what the rung claims. A rung that logs a payload it
  did not free is worse than one that does nothing, because it is credited.

* THE RESTORE IS PRICED BEFORE IT RUNS. ``pending_restore_bytes`` is what the
  affordability gate reads; it must be the full payload while spilled and 0
  otherwise, or the gate either abandons every flip or protects none.

* THRESHOLD PURITY IS REFUSED AT BOOT. Between spill and restore the
  parameters point at unbacked virtual memory. That is sound only because
  strict purity forbids decode in the PP phase. Under threshold purity a
  PP-phase decode would fault, so the combination is refused where it can
  still be fixed by a flag rather than at the first threshold decode.

Hermetic: a fake arena, CPU tensors, no GPU and no driver. The fixture is 20
layers rather than a token 2 because the layout planner's slot packing and the
alias handling only get exercised at width.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from sglang.srt.managers import phase_flip_spill as spill
from sglang.srt.model_executor.weights_arena import allocate_arena, bind_arena_views

_SCRIBBLE = 0xA5


class _FakeArena:
    """A KvVmmArena stand-in whose decommit really does destroy the bytes.

    The real arena hands physical pages back to the driver, so the virtual
    range keeps its addresses and loses its contents. Zeroing would be a
    weaker model than the truth (zeros are a plausible tensor); 0xA5 is not
    plausible as a weight, so a missing refill shows up as a value mismatch
    rather than a quiet near-miss.
    """

    def __init__(self, granularity: int = 2 << 20):
        self.granularity = granularity
        self.base = 0
        self.reserved = 1 << 40
        self.commits: list = []
        self.decommits: list = []
        self._span = None
        self._committed = 0

    def allocate_carrier(self, nbytes: int) -> torch.Tensor:
        self._span = torch.empty(int(nbytes), dtype=torch.uint8)
        # The real arena's bump allocator returns ``base + granularity-aligned
        # cursor``, so the first allocation sits at offset 0. Publishing the
        # base here reproduces that; leaving it at 0 would hand the carrier a
        # host address as an offset and fail its alignment check for a reason
        # that has nothing to do with the code under test.
        self.base = self._span.data_ptr()
        return self._span

    def commit_range(self, offset: int, want_bytes: int) -> None:
        self.commits.append((int(offset), int(want_bytes)))
        self._committed = int(want_bytes)

    def decommit_range(self, offset: int, keep_bytes: int) -> int:
        self.decommits.append((int(offset), int(keep_bytes)))
        released = max(0, self._committed - int(keep_bytes))
        self._committed = int(keep_bytes)
        if self._span is not None:
            self._span.fill_(_SCRIBBLE)
        return released


class _DraftModel(nn.Module):
    """20 layers of the shapes the drafter actually carries."""

    def __init__(self, layers: int = 20, hidden: int = 32, seed: int = 11):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            block = nn.Module()
            block.qkv = nn.Parameter(torch.randn(hidden, hidden, generator=g))
            block.o = nn.Parameter(torch.randn(hidden, hidden, generator=g))
            block.norm = nn.Parameter(torch.randn(hidden, generator=g))
            self.blocks.append(block)
        self.embed = nn.Parameter(torch.randn(hidden * 4, hidden, generator=g))


class _Inner:
    def __init__(self, model):
        self.draft_runner = type("R", (), {"model": model})()


class _DraftWorker:
    def __init__(self, model):
        self.draft_worker = _Inner(model)


def _carrier(model=None, arena=None):
    model = model or _DraftModel()
    arena = arena or _FakeArena()
    return spill.VmmDraftWeightCarrier(model, 0, arena=arena), model, arena


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _ptrs(model):
    return {n: p.data.data_ptr() for n, p in model.named_parameters()}


class CarrierAddressStabilityTest(unittest.TestCase):
    def test_addresses_survive_a_full_cycle(self):
        carrier, model, _ = _carrier()
        before = _ptrs(model)
        carrier.spill()
        # Even SPILLED the addresses must stand: the graphs hold them while
        # the pages are gone, which is the whole premise.
        self.assertEqual(_ptrs(model), before)
        carrier.restore()
        self.assertEqual(_ptrs(model), before)

    def test_a_fresh_arena_restore_would_move_them(self):
        # The control for the test above. This reproduces the OLD restore --
        # allocate a new arena and rebind onto it -- and asserts it relocates
        # the parameters. If this ever stops moving them, the test above has
        # become vacuous and must be rewritten, not deleted.
        model = _DraftModel()
        named = dict(model.named_parameters())
        from sglang.srt.model_executor.weights_arena import plan_arena_layout

        layout = plan_arena_layout(named)
        before = _ptrs(model)
        fresh = allocate_arena(layout.total_bytes, torch.device("cpu"))
        bind_arena_views(layout, fresh, rebind=list(named.items()))
        self.assertNotEqual(_ptrs(model), before)

    def test_every_param_lies_inside_the_reservation(self):
        carrier, _, _ = _carrier()
        self.assertTrue(carrier.contains_all_params())


class CarrierByteIdentityTest(unittest.TestCase):
    def test_bytes_are_identical_across_a_full_cycle(self):
        carrier, model, arena = _carrier()
        want = _snapshot(model)
        carrier.spill()
        # The fake arena scribbled the span, so this is a real restore.
        self.assertTrue(
            torch.all(arena._span == _SCRIBBLE),
            "the fixture did not actually destroy the span; the byte-identity "
            "assertion below would be vacuous",
        )
        carrier.restore()
        for name, expect in want.items():
            got = dict(model.named_parameters())[name].data
            self.assertTrue(
                torch.equal(got, expect), f"{name} did not come back identical"
            )

    def test_bytes_survive_many_cycles(self):
        carrier, model, _ = _carrier()
        want = _snapshot(model)
        for _ in range(5):
            carrier.spill()
            carrier.restore()
        for name, expect in want.items():
            got = dict(model.named_parameters())[name].data
            self.assertTrue(torch.equal(got, expect), f"{name} drifted")


class CarrierReleaseAccountingTest(unittest.TestCase):
    def test_spill_releases_the_whole_span(self):
        carrier, _, arena = _carrier()
        released = carrier.spill()
        self.assertEqual(arena.decommits, [(0, 0)])
        self.assertAlmostEqual(released, carrier.payload_mib, places=6)

    def test_spill_is_idempotent(self):
        carrier, _, arena = _carrier()
        carrier.spill()
        self.assertEqual(carrier.spill(), 0.0)
        self.assertEqual(len(arena.decommits), 1)

    def test_restore_without_spill_is_a_no_op(self):
        carrier, _, arena = _carrier()
        commits = len(arena.commits)
        self.assertEqual(carrier.restore(), 0.0)
        self.assertEqual(len(arena.commits), commits)

    def test_restore_recommits_the_whole_payload(self):
        carrier, _, arena = _carrier()
        carrier.spill()
        arena.commits.clear()
        carrier.restore()
        self.assertEqual(arena.commits, [(0, carrier.payload_bytes)])


class PendingRestoreBytesTest(unittest.TestCase):
    def test_zero_when_resident_full_payload_when_spilled(self):
        model = _DraftModel()
        worker = _DraftWorker(model)
        # No carrier installed at all: the gate must not charge for one.
        self.assertEqual(spill.pending_restore_bytes(worker), 0)
        carrier = spill.VmmDraftWeightCarrier(model, 0, arena=_FakeArena())
        setattr(worker, spill.CARRIER_ATTR, carrier)
        self.assertEqual(spill.pending_restore_bytes(worker), 0)
        carrier.spill()
        self.assertEqual(spill.pending_restore_bytes(worker), carrier.payload_bytes)
        carrier.restore()
        self.assertEqual(spill.pending_restore_bytes(worker), 0)

    def test_none_worker_is_free(self):
        self.assertEqual(spill.pending_restore_bytes(None), 0)


class InstallRefusalTest(unittest.TestCase):
    def test_threshold_purity_is_refused(self):
        worker = _DraftWorker(_DraftModel())
        args = type("A", (), {"phase_flip_purity": "threshold"})()
        with self.assertRaises(spill.PhaseFlipSpillError) as cm:
            spill.install_draft_weight_carrier(
                worker, 0, server_args=args, arena=_FakeArena()
            )
        self.assertIn("strict", str(cm.exception))

    def test_strict_purity_installs_and_parks_the_carrier(self):
        worker = _DraftWorker(_DraftModel())
        args = type("A", (), {"phase_flip_purity": "strict"})()
        carrier = spill.install_draft_weight_carrier(
            worker, 0, server_args=args, arena=_FakeArena()
        )
        self.assertIsNotNone(carrier)
        self.assertIs(spill.carrier_of(worker), carrier)

    def test_no_draft_model_is_a_quiet_none(self):
        self.assertIsNone(spill.install_draft_weight_carrier(None, 0))
        empty = type("W", (), {"draft_worker": None})()
        self.assertIsNone(spill.install_draft_weight_carrier(empty, 0))


class LadderBindsTheBootCarrierTest(unittest.TestCase):
    def test_ladder_never_builds_a_carrier_itself(self):
        # Building one inside the cutover would move the addresses the graphs
        # baked -- the exact bug rung 2 was refused for.
        worker = _DraftWorker(_DraftModel())
        ladder = spill.PhaseFlipSpillLadder(spill.DEPTH_DRAFT_WEIGHTS)
        self.assertEqual(ladder.on_enter_pp(worker), 0.0)
        self.assertIsNone(spill.carrier_of(worker))

    def test_ladder_drives_the_installed_carrier_both_ways(self):
        model = _DraftModel()
        worker = _DraftWorker(model)
        carrier = spill.VmmDraftWeightCarrier(model, 0, arena=_FakeArena())
        setattr(worker, spill.CARRIER_ATTR, carrier)
        ladder = spill.PhaseFlipSpillLadder(spill.DEPTH_DRAFT_WEIGHTS)
        self.assertGreater(ladder.on_enter_pp(worker), 0.0)
        self.assertTrue(carrier.spilled)
        self.assertGreater(ladder.on_enter_tp(worker), 0.0)
        self.assertFalse(carrier.spilled)

    def test_depth_below_two_leaves_the_drafter_alone(self):
        model = _DraftModel()
        worker = _DraftWorker(model)
        carrier = spill.VmmDraftWeightCarrier(model, 0, arena=_FakeArena())
        setattr(worker, spill.CARRIER_ATTR, carrier)
        ladder = spill.PhaseFlipSpillLadder(spill.DEPTH_ALLOCATOR_CACHE)
        self.assertEqual(ladder.on_enter_pp(worker), 0.0)
        self.assertFalse(carrier.spilled)


class AffordabilityGatePricesTheRestoreTest(unittest.TestCase):
    """The gate is the difference between an abandon and a dead instance.

    The restore's ``commit_range`` runs inside ``_cutover``, past the point of
    no return. ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` there took all
    three ranks down on 2026-08-09. Pricing the commit into the pre-flip
    verdict is what turns that into a unanimous, free refusal.
    """

    def _runtime_stub(self, worker):
        from sglang.srt.managers import phase_flip_runtime as rt

        stub = object.__new__(rt.PhaseFlipRuntime)
        stub._census_scheduler = type(
            "S", (), {"phase_flip_stacks": type("K", (), {"draft_worker": worker})()}
        )()
        return stub, rt

    def _spilled_worker(self):
        model = _DraftModel()
        worker = _DraftWorker(model)
        carrier = spill.VmmDraftWeightCarrier(model, 0, arena=_FakeArena())
        setattr(worker, spill.CARRIER_ATTR, carrier)
        return worker, carrier

    def test_pp_to_tp_is_charged_the_full_payload(self):
        worker, carrier = self._spilled_worker()
        carrier.spill()
        stub, rt = self._runtime_stub(worker)
        self.assertEqual(
            stub._draft_restore_bytes(rt.PP_TO_TP), carrier.payload_bytes
        )

    def test_tp_to_pp_is_free(self):
        # Nothing is restored on the leg that spills; charging it would
        # abandon flips for a cost that is not incurred.
        worker, carrier = self._spilled_worker()
        carrier.spill()
        stub, rt = self._runtime_stub(worker)
        self.assertEqual(stub._draft_restore_bytes(rt.TP_TO_PP), 0)

    def test_a_resident_drafter_is_free(self):
        worker, _ = self._spilled_worker()
        stub, rt = self._runtime_stub(worker)
        self.assertEqual(stub._draft_restore_bytes(rt.PP_TO_TP), 0)

    def test_no_scheduler_does_not_raise_inside_the_gate(self):
        from sglang.srt.managers import phase_flip_runtime as rt

        stub = object.__new__(rt.PhaseFlipRuntime)
        stub._census_scheduler = None
        self.assertEqual(stub._draft_restore_bytes(rt.PP_TO_TP), 0)

    def test_a_broken_scheduler_degrades_to_the_wave_peak(self):
        # A gate that cannot price the restore must not also refuse the flip:
        # the pre-rung-2 behaviour is the safe fallback, not an abort.
        from sglang.srt.managers import phase_flip_runtime as rt

        class _Boom:
            @property
            def phase_flip_stacks(self):
                raise RuntimeError("no stacks here")

        stub = object.__new__(rt.PhaseFlipRuntime)
        stub._census_scheduler = _Boom()
        self.assertEqual(stub._draft_restore_bytes(rt.PP_TO_TP), 0)


if __name__ == "__main__":
    unittest.main()
