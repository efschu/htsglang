"""#602 --rank-kv-ratio corridor: argument resolution and its refusals.

CPU only. The ledger is stubbed, so nothing here touches NVML, CUDA or a
model -- what is under test is which numbers the mode reads and which
configurations it refuses, not how the ledger prices a card.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest -q \
        test/registered/unit/server_args/test_corridor_kv_ratio_args.py
"""

import unittest
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs, _parse_rank_kv_ratio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The ledger line names, in the shape CardVramLedger.term() returns.
ACTIVATION = "runtime activation + metadata"
CAPTURE = "CUDA graph capture"
WORKSPACE = "attention workspaces (capped)"
WEIGHTS = "model weights (shards)"
RESIDUAL = "hardware residual (per process)"
CARVE_OUT = "NVML driver carve-out (not allocatable)"


class _FakeLedger:
    def __init__(self, gpu_id, terms, unbounded=()):
        self.gpu_id = gpu_id
        self.card = f"GPU {gpu_id} (fake)"
        self._terms = dict(terms)
        self.unbounded = tuple(unbounded)

    def term(self, name):
        mib = self._terms.get(name)
        return None if mib is None else SimpleNamespace(name=name, mib=mib)


def _stub(ledgers, **kw):
    """A ServerArgs-shaped stub carrying only what the methods under test read."""
    sa = SimpleNamespace(
        rank_kv_ratio=kw.pop("rank_kv_ratio", "corridor"),
        rank_gpu_id=kw.pop("rank_gpu_id", [0, 1, 2]),
        rank_gpu_memory_mib=kw.pop("rank_gpu_memory_mib", [28868, 17325, 17327]),
        rank_user_reserve_mib=kw.pop("rank_user_reserve_mib", 1024),
        corridor_post_sizing_mib=None,
    )
    sa.uneven_kv_corridor_mode = lambda: sa.rank_kv_ratio == "corridor"
    sa._build_card_ledgers = lambda: ledgers
    sa.corridor_late_term_names = ServerArgs.corridor_late_term_names
    sa.corridor_post_sizing_mib_per_gpu = (
        lambda: ServerArgs.corridor_post_sizing_mib_per_gpu(sa)
    )
    return sa


class TestModeParsing(CustomTestCase):
    def test_corridor_is_accepted(self):
        self.assertEqual(_parse_rank_kv_ratio("corridor"), "corridor")
        self.assertEqual(_parse_rank_kv_ratio("  corridor "), "corridor")

    def test_existing_modes_are_unchanged(self):
        self.assertEqual(_parse_rank_kv_ratio("coupled"), "coupled")
        self.assertEqual(_parse_rank_kv_ratio("capacity"), "capacity")
        self.assertEqual(_parse_rank_kv_ratio("auto"), "capacity")
        self.assertEqual(_parse_rank_kv_ratio("speed"), "speed")
        self.assertEqual(_parse_rank_kv_ratio("3,2,1"), [3, 2, 1])

    def test_unknown_mode_names_the_alternatives(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_rank_kv_ratio("corridorr")
        self.assertIn("corridor", str(ctx.exception))

    @staticmethod
    def _mode_stub(mode):
        """The mode predicates call each other through ``self``, so they have
        to be bound rather than invoked unbound on a bare namespace."""
        sa = SimpleNamespace(rank_kv_ratio=mode)
        for name in (
            "uneven_kv_corridor_mode",
            "uneven_kv_capacity_mode",
            "uneven_kv_speed_mode",
            "uneven_kv_derived_mode",
        ):
            setattr(sa, name, getattr(ServerArgs, name).__get__(sa, SimpleNamespace))
        return sa

    def test_corridor_is_a_derived_mode(self):
        sa = self._mode_stub("corridor")
        self.assertTrue(sa.uneven_kv_corridor_mode())
        self.assertTrue(sa.uneven_kv_derived_mode())
        self.assertFalse(sa.uneven_kv_capacity_mode())
        self.assertFalse(sa.uneven_kv_speed_mode())

    def test_capacity_and_speed_remain_derived_without_corridor(self):
        for mode, cap, speed in (("capacity", True, False), ("speed", False, True)):
            with self.subTest(mode=mode):
                sa = self._mode_stub(mode)
                self.assertTrue(sa.uneven_kv_derived_mode())
                self.assertEqual(sa.uneven_kv_capacity_mode(), cap)
                self.assertEqual(sa.uneven_kv_speed_mode(), speed)
                self.assertFalse(sa.uneven_kv_corridor_mode())

    def test_coupled_is_not_derived(self):
        sa = self._mode_stub("coupled")
        self.assertFalse(sa.uneven_kv_derived_mode())

    def test_other_modes_are_not_corridor(self):
        for mode in ("coupled", "capacity", "speed"):
            with self.subTest(mode=mode):
                sa = SimpleNamespace(rank_kv_ratio=mode)
                self.assertFalse(ServerArgs.uneven_kv_corridor_mode(sa))


class TestPostSizingDemand(CustomTestCase):
    def test_only_the_late_terms_are_charged(self):
        ledgers = [
            _FakeLedger(
                0,
                {
                    ACTIVATION: 900,
                    CAPTURE: 1200,
                    WORKSPACE: 300,
                    # Resident / never-in-free terms must NOT be charged.
                    WEIGHTS: 9000,
                    RESIDUAL: 700,
                    CARVE_OUT: 518,
                },
            ),
            _FakeLedger(1, {ACTIVATION: 400, CAPTURE: 600}),
        ]
        got = _stub(ledgers).corridor_post_sizing_mib_per_gpu()
        self.assertEqual(got, {0: 2400, 1: 1000})

    def test_absent_late_term_counts_as_zero_not_as_an_error(self):
        """A term the ledger did not EMIT is a feature that is not present;
        that is distinct from a term it could not PRICE (next test)."""
        got = _stub([_FakeLedger(0, {ACTIVATION: 500})]).corridor_post_sizing_mib_per_gpu()
        self.assertEqual(got, {0: 500})

    def test_unbounded_term_refuses_and_names_the_card(self):
        ledgers = [_FakeLedger(0, {ACTIVATION: 500}, unbounded=("CUDA graph capture",))]
        with self.assertRaises(ValueError) as ctx:
            _stub(ledgers).corridor_post_sizing_mib_per_gpu()
        msg = str(ctx.exception)
        self.assertIn("GPU 0", msg)
        self.assertIn("CUDA graph capture", msg)
        self.assertIn("0 MiB", msg)

    def test_no_ledger_at_all_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            _stub([]).corridor_post_sizing_mib_per_gpu()
        self.assertIn("corridor", str(ctx.exception))

    def test_term_name_list_comes_from_the_constants(self):
        names = ServerArgs.corridor_late_term_names()
        self.assertIn(ACTIVATION, names)
        self.assertIn(CAPTURE, names)
        self.assertIn(WORKSPACE, names)
        # The residency partition must exclude everything already visible in
        # a pre-pool free reading, and the carve-out which is never in it.
        self.assertNotIn(WEIGHTS, names)
        self.assertNotIn(RESIDUAL, names)
        self.assertNotIn(CARVE_OUT, names)
        self.assertNotIn("mamba/GDN state pool", names)


class TestHandlerRefusals(CustomTestCase):
    def _handle(self, sa):
        return ServerArgs._handle_corridor_kv_ratio(sa)

    def test_non_corridor_mode_is_a_no_op(self):
        sa = _stub([], rank_kv_ratio="coupled", rank_gpu_id=None)
        self._handle(sa)
        self.assertIsNone(sa.corridor_post_sizing_mib)

    def test_missing_placement_is_refused(self):
        sa = _stub([_FakeLedger(0, {ACTIVATION: 1})], rank_gpu_id=None)
        with self.assertRaises(ValueError) as ctx:
            self._handle(sa)
        self.assertIn("--rank-gpu-id", str(ctx.exception))

    def test_missing_budgets_are_refused(self):
        sa = _stub([_FakeLedger(0, {ACTIVATION: 1})], rank_gpu_memory_mib=None)
        with self.assertRaises(ValueError) as ctx:
            self._handle(sa)
        self.assertIn("--rank-gpu-memory-mib", str(ctx.exception))

    def test_happy_path_parks_the_demand(self):
        ledgers = [
            _FakeLedger(0, {ACTIVATION: 900, CAPTURE: 1200}),
            _FakeLedger(1, {ACTIVATION: 400, CAPTURE: 600}),
            _FakeLedger(2, {ACTIVATION: 400, CAPTURE: 600}),
        ]
        sa = _stub(ledgers)
        self._handle(sa)
        self.assertEqual(sa.corridor_post_sizing_mib, {0: 2100, 1: 1000, 2: 1000})


if __name__ == "__main__":
    unittest.main()
