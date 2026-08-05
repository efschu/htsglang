"""#590: the measured footprints must reach KV pool sizing, or they are decor.

#586 produced refusal-honest per-card footprints. Window 6 of 2026-08-05 then
measured that they changed nothing: a tree WITH the ledger and a tree WITHOUT
it sized pools identical to 26 tokens per rank, because the only thing shaping
the budget on the uneven-DCP recipe -- ``derived_rank_auto_reserve_mib`` --
read the raw heuristic, and the one ledger-aware site
(``_profile_available_bytes``) sits behind ``post_capture_kv_active``, which
``dcp_size > 1`` turns off.

These tests pin the wiring, and the first one is the BIND PROOF: change the
ledger's number, observe the reserve (and therefore the pool) move with it. A
test that only checks "the ledger was consulted" would pass against code that
consults it and discards the answer, which is precisely the state #590 fixes.
"""

import math
import sys
import types
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEURISTIC_MIB = 3968.0
CARD = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
ACTIVATION_MODULE = "sglang.srt.mem_ledger.activation"


class _Stub:
    """The smallest object the reserve derivation actually touches.

    Deliberately not a real ServerArgs: constructing one runs the planner and
    reads a checkpoint, which would make this suite a boot test. The methods
    under test are taken off the class and called against this.
    """

    def __init__(self):
        self.heuristic_calls = 0
        self.rank_auto_reserve_mib = "auto"

    def mamba_pre_capture_reserve_mb(self, gpu_mem):
        self.heuristic_calls += 1
        return HEURISTIC_MIB

    def _apply_gpu_mem_capacity_defaults(self, gpu_mem):
        return None

    def speculative_capture_tokens(self):
        return 0

    def ladder_reserve_gpu_id(self):
        return None

    def ladder_reserve_demand(self, colocated):
        return None

    def _reserve_card_uuids(self, gpu_ids):
        return {g: CARD for g in gpu_ids}

    # The REAL implementations, bound to this stub: the derivation calls
    # self.runtime_reserve_mib and the per-GPU map calls
    # self.derived_rank_auto_reserve_mib, and stubbing either of those would
    # test the stub instead of the wiring.
    def runtime_reserve_mib(self, gpu_mem, card_uuid=None):
        return ServerArgs.runtime_reserve_mib(self, gpu_mem, card_uuid=card_uuid)

    def derived_rank_auto_reserve_mib(self, *args, **kwargs):
        return ServerArgs.derived_rank_auto_reserve_mib(self, *args, **kwargs)

    def ledger_full_demand_per_gpu(self):
        """No full-demand model here, on purpose.

        #593 put the FULL per-card demand in front of this path, and these
        tests pin the per-term substitution BEHIND it -- which is still the
        live behaviour whenever the full model refuses or is unavailable (the
        production tree today). Returning None selects that fallback, so what
        this file proves stays true of the boots it describes.
        """
        return None


def runtime_reserve(stub, gpu_mem=20480, card_uuid=None):
    return ServerArgs.runtime_reserve_mib(stub, gpu_mem, card_uuid=card_uuid)


def derived(stub, gpu_mem=20480, colocated=1, card_uuid=None):
    return ServerArgs.derived_rank_auto_reserve_mib(
        stub, gpu_mem, colocated, card_uuid=card_uuid
    )


class _FakeFootprint:
    def __init__(self, activation_mib):
        self.activation_mib = activation_mib
        self.capture_mib = 640


def install_ledger(monkey_target, footprint):
    """A stand-in mem_ledger whose resolver returns ``footprint``."""
    activation = types.ModuleType(ACTIVATION_MODULE)
    activation.profile_from_server_args = lambda sa, arch: ("profile",)
    activation.resolve_phase_footprint = lambda uuid, **kw: footprint
    calibration = types.ModuleType("sglang.srt.mem_ledger.calibration")
    calibration.live_fingerprint = lambda: ("a191a0712717", [], "580")
    engine = types.ModuleType("sglang.srt.mem_ledger.engine")
    engine._model_architectures = lambda sa: ("Qwen3_5ForConditionalGeneration",)
    monkey_target[ACTIVATION_MODULE] = activation
    monkey_target["sglang.srt.mem_ledger.calibration"] = calibration
    monkey_target["sglang.srt.mem_ledger.engine"] = engine


class LedgerModules:
    """Swap the three ledger modules for the duration of a test."""

    def __init__(self, footprint=None, absent=False):
        self.footprint = footprint
        self.absent = absent
        self.saved = {}

    def __enter__(self):
        names = [
            ACTIVATION_MODULE,
            "sglang.srt.mem_ledger.calibration",
            "sglang.srt.mem_ledger.engine",
        ]
        for n in names:
            self.saved[n] = sys.modules.get(n)
        if self.absent:
            # None in sys.modules makes `from X import Y` raise ImportError,
            # which is what a tree without the package looks like.
            for n in names:
                sys.modules[n] = None
        else:
            install_ledger(sys.modules, self.footprint)
        return self

    def __exit__(self, *exc):
        for n, mod in self.saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
        return False


class TestReserveBindsToTheLedger(unittest.TestCase):
    def setUp(self):
        ServerArgs._ledger_reserve_refusal_named = False

    def test_bind_proof_the_pool_moves_with_the_ledger_number(self):
        """THE bind proof: two different footprints, two different reserves.

        The reserve is subtracted from the card total to form the rank budget
        and the KV pool takes what is left, so a reserve that tracks the
        ledger IS a pool that tracks the ledger -- and a smaller reserve is a
        bigger pool, which is the direction #586 was supposed to buy.
        """
        stub = _Stub()
        seen = {}
        for activation in (1766.0, 2644.0, 4195.0):
            with LedgerModules(footprint=_FakeFootprint(activation)):
                mib, source = runtime_reserve(stub, card_uuid=CARD)
                seen[activation] = (mib, source, derived(stub, card_uuid=CARD))

        for activation, (mib, source, total) in seen.items():
            self.assertEqual(source, "ledger")
            self.assertEqual(mib, activation)
            self.assertEqual(total, int(math.ceil(activation)))

        # Distinct inputs must give distinct reserves -- otherwise the number
        # was consulted and thrown away, the exact #590 defect.
        self.assertEqual(len({v[2] for v in seen.values()}), 3, seen)
        # And the direction: the binding 3080 footprint frees budget against
        # the heuristic, the 5090 bound costs some.
        self.assertLess(seen[1766.0][2], HEURISTIC_MIB)
        self.assertGreater(seen[4195.0][2], HEURISTIC_MIB)

    def test_ledger_value_is_used_instead_of_the_heuristic_not_beside_it(self):
        stub = _Stub()
        with LedgerModules(footprint=_FakeFootprint(1766.0)):
            mib, source = runtime_reserve(stub, card_uuid=CARD)
        self.assertEqual((mib, source), (1766.0, "ledger"))
        self.assertEqual(
            stub.heuristic_calls,
            0,
            "the falsified heuristic was still evaluated on the ledger path",
        )


class TestRefusalIsNamedNotSilent(unittest.TestCase):
    def setUp(self):
        ServerArgs._ledger_reserve_refusal_named = False

    def test_ledger_present_but_refusing_is_named_and_falls_back(self):
        """Reachable refusal path: loud, attributed, still bootable."""
        stub = _Stub()
        with LedgerModules(footprint=None):
            with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
                mib, source = runtime_reserve(stub, card_uuid=CARD)
        self.assertEqual((mib, source), (HEURISTIC_MIB, "heuristic:ledger-refused"))
        joined = "\n".join(cm.output)
        self.assertIn("REFUSED", joined)
        self.assertIn(CARD, joined, "the refusal must name the card that missed")

    def test_the_refusal_is_named_once_per_process(self):
        stub = _Stub()
        with LedgerModules(footprint=None):
            with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
                runtime_reserve(stub, card_uuid=CARD)
                runtime_reserve(stub, card_uuid=CARD)
                runtime_reserve(stub, card_uuid=CARD)
        self.assertEqual(len(cm.output), 1, cm.output)


class TestLegacyPathsAreByteIdentical(unittest.TestCase):
    """The production tree today has no mem_ledger at all."""

    def setUp(self):
        ServerArgs._ledger_reserve_refusal_named = False

    def test_no_ledger_module_is_silent_heuristic(self):
        stub = _Stub()
        with LedgerModules(absent=True):
            with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
                mib, source = runtime_reserve(stub, card_uuid=CARD)
        self.assertEqual((mib, source), (HEURISTIC_MIB, "heuristic:no-ledger"))

    def test_no_card_identity_is_silent_heuristic(self):
        """Every legacy caller and the planner call without a UUID."""
        stub = _Stub()
        with LedgerModules(footprint=_FakeFootprint(1766.0)):
            with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
                mib, source = runtime_reserve(stub, card_uuid=None)
        self.assertEqual((mib, source), (HEURISTIC_MIB, "heuristic:not-asked"))

    def test_derived_reserve_without_uuid_matches_the_pre_590_formula(self):
        stub = _Stub()
        with LedgerModules(footprint=_FakeFootprint(1766.0)):
            self.assertEqual(derived(stub, card_uuid=None), int(HEURISTIC_MIB))


class TestPerGpuDemandCarriesTheIdentity(unittest.TestCase):
    """reserve_demand_per_gpu is the pool-sizing entry point."""

    def setUp(self):
        ServerArgs._ledger_reserve_refusal_named = False

    def test_each_gpu_is_priced_with_its_own_card(self):
        per_card = {
            "GPU-aaa": _FakeFootprint(4195.0),
            "GPU-bbb": _FakeFootprint(1766.0),
        }
        uuid_by_gpu = {0: "GPU-aaa", 1: "GPU-bbb"}

        class S(_Stub):
            def _reserve_card_uuids(self, gpu_ids):
                return {g: uuid_by_gpu[g] for g in gpu_ids if g in uuid_by_gpu}

        stub = S()
        activation = types.ModuleType(ACTIVATION_MODULE)
        activation.profile_from_server_args = lambda sa, arch: ("profile",)
        activation.resolve_phase_footprint = lambda uuid, **kw: per_card.get(uuid)
        calibration = types.ModuleType("sglang.srt.mem_ledger.calibration")
        calibration.live_fingerprint = lambda: ("a191a0712717", [], "580")
        engine = types.ModuleType("sglang.srt.mem_ledger.engine")
        engine._model_architectures = lambda sa: ("Qwen3_5ForConditionalGeneration",)
        saved = {
            n: sys.modules.get(n)
            for n in (
                ACTIVATION_MODULE,
                "sglang.srt.mem_ledger.calibration",
                "sglang.srt.mem_ledger.engine",
            )
        }
        sys.modules[ACTIVATION_MODULE] = activation
        sys.modules["sglang.srt.mem_ledger.calibration"] = calibration
        sys.modules["sglang.srt.mem_ledger.engine"] = engine
        try:
            out = ServerArgs.reserve_demand_per_gpu(stub, 20480, {0: 1, 1: 1})
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
        # Two cards, two footprints, two different demands -- a single number
        # for both would mean the per-card identity was dropped on the way.
        self.assertEqual(out[0], 4195)
        self.assertEqual(out[1], 1766)


if __name__ == "__main__":
    unittest.main()
