"""The registry control plane (#333 §7.4), over a real ASGI client.

Hermetic: mocked cards, a fake adapter, no device and no engine process. What
is being tested is the contract the surface promises -- that registering
validates without booting, that a rejection carries numbers a UI can act on,
and that the status codes distinguish "you asked wrongly" from "not right now".

    python -m pytest test/registered/registry/test_control_plane.py -v
"""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.registry.arbiter import EngineRegistry
from sglang.srt.registry.http_api import build_app
from sglang.srt.registry.ledger import MIB, ReservationStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sibling(name):
    """Load a test module next to this one, whatever the rootdir looks like.

    The fake adapter registers itself on import and may only be registered
    once, so both suites have to share the same module object rather than
    each defining their own.
    """
    import importlib.util
    import sys

    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_registry_tests = _sibling("test_registry")
CARD_3080_A = _registry_tests.CARD_3080_A
CARD_5090 = _registry_tests.CARD_5090
RIG = _registry_tests.RIG


def body(engine_id, mib, cards, **extra):
    return {
        "engine_id": engine_id,
        "klass": 1,
        "adapter": "fake",
        "placement": list(cards),
        "launch": {"mib_per_card": mib},
        **extra,
    }


class ControlPlaneTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.now = 1_000_000.0
        store = ReservationStore(
            Path(self._tmp.name),
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        self.registry = EngineRegistry(
            store=store, card_totals=RIG, clock=lambda: self.now
        )
        self.addCleanup(self.registry.shutdown)
        self.client = TestClient(
            build_app(self.registry), raise_server_exceptions=False
        )

    # -- registration ------------------------------------------------------

    def test_post_engines_validates_without_booting(self):
        response = self.client.post(
            "/registry/engines", json=body("a", 8192, [CARD_5090])
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["registered"])
        self.assertTrue(payload["plan"]["fits"])
        self.assertEqual(self.registry.instance("a").state.value, "COLD")
        self.assertEqual(self.registry.adapter("a").history, [])

    def test_an_infeasible_spec_is_rejected_at_registration(self):
        """§7 acceptance gate 4: rejected without booting anything."""
        response = self.client.post(
            "/registry/engines", json=body("huge", 40 * 1024, [CARD_5090])
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["error"], "registration_rejected")
        self.assertIn("32768 MiB", payload["message"])
        self.assertEqual(self.registry.engines(), ())

    def test_a_malformed_spec_is_a_400_that_names_the_field(self):
        response = self.client.post(
            "/registry/engines", json={"engine_id": "x", "klass": 1}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("adapter", response.json()["message"])

    def test_unknown_engine_is_a_404(self):
        self.assertEqual(
            self.client.get("/registry/plan?engine_id=ghost").status_code, 404
        )
        self.assertEqual(self.client.delete("/registry/engines/ghost").status_code, 404)

    # -- state -------------------------------------------------------------

    def test_state_endpoint_walks_the_ladder(self):
        self.client.post("/registry/engines", json=body("a", 8192, [CARD_5090]))
        for target in ("HOT", "WARM_GPU", "COLD"):
            response = self.client.post(
                "/registry/engines/a/state", json={"target": target}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["state"], target)

    def test_an_invalid_state_names_the_valid_ones(self):
        self.client.post("/registry/engines", json=body("a", 1024, [CARD_5090]))
        response = self.client.post(
            "/registry/engines/a/state", json={"target": "LUKEWARM"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("WARM_GPU", response.json()["message"])

    def test_a_promotion_that_needs_too_long_is_a_503_with_the_numbers(self):
        """§7.5: rejection is informative. A UI can offer "wait" or "other model"."""
        self.client.post("/registry/engines", json=body("in", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines", json=body("out", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines/in/state", json={"target": "HOT"})
        response = self.client.post(
            "/registry/engines/out/state",
            json={"target": "HOT", "max_promotion_wait_ms": 1},
        )
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"], "promotion_rejected")
        self.assertEqual(payload["would_evict"], ["in"])
        self.assertGreater(payload["projected_wait_ms"], 0)
        self.assertTrue(payload["projected_wait_is_estimated"])
        self.assertEqual(payload["shortfalls"][0]["card_uuid"], CARD_5090)
        # The incumbent was not touched by a rejected request.
        self.assertEqual(self.registry.instance("in").state.value, "HOT")

    def test_pin_makes_an_engine_uneviectable(self):
        self.client.post("/registry/engines", json=body("in", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines", json=body("out", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines/in/state", json={"target": "HOT"})
        self.client.post("/registry/engines/in/pin", json={"pinned": True})
        response = self.client.post(
            "/registry/engines/out/state", json={"target": "HOT"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("pinned", response.json()["message"])

    def test_delete_releases_the_slot(self):
        self.client.post("/registry/engines", json=body("a", 8192, [CARD_5090]))
        self.client.post("/registry/engines/a/state", json={"target": "HOT"})
        self.assertEqual(self.client.delete("/registry/engines/a").status_code, 200)
        cards = {
            c["card_uuid"]: c
            for c in self.client.get("/registry/cards").json()["cards"]
        }
        self.assertEqual(cards[CARD_5090]["reserved_bytes"], 0)

    # -- reporting ---------------------------------------------------------

    def test_cards_report_total_reserved_measured_corridor_and_waste(self):
        self.client.post("/registry/engines", json=body("a", 8192, [CARD_5090]))
        self.client.post("/registry/engines/a/state", json={"target": "HOT"})
        cards = {
            c["card_uuid"]: c
            for c in self.client.get("/registry/cards").json()["cards"]
        }
        card = cards[CARD_5090]
        self.assertEqual(card["total_mib"], 32768)
        self.assertEqual(card["reserved_mib"], 8192)
        self.assertEqual(card["corridor_bytes"], 400 * MIB)
        self.assertEqual(card["waste_mib"], 4096)
        self.assertEqual(card["tenants"], ["a"])
        self.assertEqual(card["available_mib"], 32768 - 8192 - 400)

    def test_registry_snapshot_reports_the_derived_m(self):
        self.client.post("/registry/engines", json=body("a", 15 * 1024, [CARD_5090]))
        self.client.post("/registry/engines", json=body("b", 15 * 1024, [CARD_5090]))
        self.client.post("/registry/engines", json=body("c", 15 * 1024, [CARD_5090]))
        snapshot = self.client.get("/registry").json()
        self.assertEqual(snapshot["hot_capacity"]["derived_max_hot"], 2)
        self.assertIsNone(snapshot["max_hot"])
        self.assertEqual(len(snapshot["hot_capacity"]["excluded"]), 1)

    # -- plan --------------------------------------------------------------

    def test_post_plan_costs_a_spec_that_is_not_registered(self):
        response = self.client.post(
            "/registry/plan", json=body("what-if", 8192, [CARD_3080_A])
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["fits"])
        self.assertEqual(payload["cards"], [CARD_3080_A])
        self.assertEqual(self.registry.engines(), ())

    def test_plan_reports_the_eviction_it_would_perform(self):
        self.client.post("/registry/engines", json=body("in", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines/in/state", json={"target": "HOT"})
        payload = self.client.post(
            "/registry/plan", json=body("out", 20 * 1024, [CARD_5090])
        ).json()
        self.assertTrue(payload["fits"])
        self.assertEqual(payload["would_evict"], ["in"])
        self.assertFalse(payload["feasible_without_eviction"])
        self.assertGreater(payload["shortfall_bytes"], 0)

    def test_plan_without_an_engine_id_says_what_to_pass(self):
        response = self.client.get("/registry/plan")
        self.assertEqual(response.status_code, 400)
        self.assertIn("engine_id", response.json()["message"])

    # -- idle set ----------------------------------------------------------

    def test_default_hot_is_validated_when_it_is_set(self):
        self.client.post("/registry/engines", json=body("a", 20 * 1024, [CARD_5090]))
        self.client.post("/registry/engines", json=body("b", 20 * 1024, [CARD_5090]))
        response = self.client.post(
            "/registry/default_hot", json={"engines": ["a", "b"]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could never return to it", response.json()["message"])

    def test_idle_returns_to_the_default_set(self):
        self.client.post("/registry/engines", json=body("keep", 8192, [CARD_5090]))
        self.client.post("/registry/engines", json=body("drop", 8192, [CARD_5090]))
        self.client.post("/registry/default_hot", json={"engines": ["keep"]})
        self.client.post("/registry/engines/drop/state", json={"target": "HOT"})
        changed = self.client.post("/registry/idle", json={"force": True}).json()
        self.assertEqual(sorted(changed["changed"]), ["drop", "keep"])
        self.assertEqual(self.registry.instance("drop").state.value, "COLD")
        self.assertEqual(self.registry.instance("keep").state.value, "HOT")


if __name__ == "__main__":
    unittest.main()
