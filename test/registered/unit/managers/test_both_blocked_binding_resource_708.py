"""#708: the BOTH-BLOCKED decline must NAME its binding resource, not assert it.

The decline read "the binding resource is KV, not the layout" as a fixed
string. It was RIGHT on all three live specimens --

    22:22:33   19,004 avail vs  97,922 pending
    22:30:43  107,881 avail vs 160,514 pending
    22:32:00    7,085 avail vs 217,048 pending

-- available well under pending every time. But a claim that cannot come out
the other way is not a diagnosis: the same sentence would have printed with the
pool standing empty, and the next reader would have gone hunting the pool while
the real bound was elsewhere. This is a ROBUSTNESS fix, not a wrong-verdict fix;
I originally filed it as the latter, which over-claimed.

The routing conclusion is deliberately unchanged. When the target cannot admit
either, changing layout cannot help, so this is never a flip regardless of
which resource binds. Only the diagnosis is derived.
"""

import unittest

from sglang.srt.managers.phase_policy import BOTH_BLOCKED
from sglang.test.test_utils import CustomTestCase


def _decline_text(avail, pending=97922, phase="tp"):
    """Drive the real policy branch and return the decline reason."""
    from sglang.srt.managers import phase_policy as pp

    inp = pp.PhasePolicyInputs(
        phase=phase,
        pending_prefill_tokens=pending,
        running_bs=0,
        now=1000.0,
        nothing_can_run=True,
        target_can_admit=False,
        kv_available_tokens=avail,
    )
    # a positive flip threshold is required by PhasePolicyConfig.__post_init__
    cfg = pp.PhasePolicyConfig(enabled=True, flip_tokens=1000)
    state = pp.PhasePolicyState()
    decision = pp.decide(cfg, state, inp)
    return decision.reason or ""


class TestBindingResourceIsDerived708(CustomTestCase):
    def test_the_live_specimens_still_name_KV(self):
        """All three real specimens had available << pending, so the derived
        text must reach the SAME conclusion the hardcoded one did -- with the
        numbers now shown."""
        for avail, pending in ((19004, 97922), (107881, 160514), (7085, 217048)):
            with self.subTest(avail=avail, pending=pending):
                msg = _decline_text(avail, pending)
                self.assertIn(BOTH_BLOCKED, msg)
                self.assertIn("the binding resource is KV", msg, msg)
                self.assertIn(str(avail), msg, "the measurement must be quoted")
                self.assertIn(str(pending), msg)

    def test_kv_not_binding_says_so_and_redirects(self):
        """CAN-FAIL, the synthetic specimen the fix exists for: a pool with room
        to spare must NOT be blamed. The old hardcoded string fails this."""
        msg = _decline_text(500_000, pending=1_000)
        self.assertIn("KV is NOT the binding resource", msg, msg)
        self.assertIn("mamba/GDN", msg, "it must redirect to the state-slot bound")
        self.assertNotIn("the binding resource is KV,", msg, msg)

    def test_unmeasured_admits_it_rather_than_guessing(self):
        msg = _decline_text(None)
        self.assertIn("NOT MEASURED", msg, msg)
        self.assertNotIn("the binding resource is KV", msg, msg)
        self.assertNotIn("KV is NOT the binding resource", msg, msg)

    def test_the_routing_conclusion_never_changes(self):
        """The decline is an evict trigger and never a flip, whichever resource
        binds. A fix that let the diagnosis change the ROUTING would be a much
        worse bug than the one being fixed."""
        for avail in (None, 0, 7085, 500_000):
            with self.subTest(avail=avail):
                msg = _decline_text(avail, pending=1_000)
                self.assertIn("evict trigger and NOT a flip", msg, msg)
                self.assertIn("10:24 ping-pong", msg, msg)

    def test_boundary_equal_available_and_pending(self):
        """avail == pending is not short, so KV is not what binds."""
        msg = _decline_text(50_000, pending=50_000)
        self.assertIn("KV is NOT the binding resource", msg, msg)


class TestSchedulerSuppliesTheMeasurement708(CustomTestCase):
    """Wiring pin: an input the scheduler never populates leaves the fix inert
    and every line reading 'NOT MEASURED' forever."""

    def test_scheduler_passes_kv_available_tokens(self):
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        src = inspect.getsource(scheduler_mod.Scheduler)
        # KEYED ON THE IDENTIFIER, NOT THE CALL FORM. The wiring was later
        # made defensive (getattr(self, "_uniform_kv_available", ...)) so a
        # scheduler STAND-IN without the probe degrades to "not measured"
        # instead of raising -- and this pin, keyed on the literal call,
        # broke on that edit. A pin must survive a legal refactor of the
        # thing it pins, or it punishes the fix instead of the regression.
        self.assertIn("kv_available_tokens=", src)
        self.assertIn("_uniform_kv_available", src)

    def test_the_helper_uses_the_GROUP_MIN_accessor(self):
        """Not the local pool: PhasePolicyInputs fields are replicated by
        contract, and a rank-dependent value here is the #616g divergence."""
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        src = inspect.getsource(scheduler_mod.Scheduler._uniform_kv_available)
        self.assertIn("uniform_avail_for_evict", src)
        self.assertNotIn("allocator.available_size()", src)

    def test_helper_returns_none_instead_of_raising(self):
        from types import SimpleNamespace

        from sglang.srt.managers import scheduler as scheduler_mod

        stub = SimpleNamespace(tree_cache=None)
        self.assertIsNone(scheduler_mod.Scheduler._uniform_kv_available(stub))
        stub2 = SimpleNamespace(tree_cache=SimpleNamespace())
        self.assertIsNone(scheduler_mod.Scheduler._uniform_kv_available(stub2))


if __name__ == "__main__":
    unittest.main()
