"""#713: the idle-box refusal, and the terms that must be printed to explain it.

MEASURED 2026-08-17 03:0x. A TEN-token prompt waited 31.64 s to first token.
Sampled every 2 s for the whole wait:

    num_running_reqs = 0    (nothing running)
    num_queue_reqs   = 1    (the request, queued)
    mamba_available  = 3    (never zero: 0/16 samples)
    kv_available     = hundreds of thousands

and the policy logged, twice:

    BOTH BLOCKED ... 0 req resident, 22 tok pending -- KV is NOT the binding
    resource (72033 rows available against 22 pending)

An 8-arm run put every TTFT between 11.87 s and 62.65 s, with NOT ONE arm under
3 s. So this is the serving floor, not an outlier.

THE SIMULATION IS NOT THE BUG. Replaying _layout_admits with exactly those
numbers returns pp=True / tp=False, i.e. it would have armed the flip. These
tests pin that, so that a future edit cannot quietly make the simulation itself
start refusing this state -- and they establish that the divergence lives in
what the simulation READS in-process, which is why the diagnostic exists.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

# the live-measured state, verbatim
PENDING = 22
ROWS_AVAIL = 72_033
MAMBA_SLOTS = 3


def _sched(avail=ROWS_AVAIL, slots=MAMBA_SLOTS, evictable=0, chunk=512):
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.server_args = SimpleNamespace(chunked_prefill_size=chunk)
    s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: avail)
    s.tree_cache = SimpleNamespace(full_evictable_size=lambda: evictable)
    s.req_to_token_pool = SimpleNamespace(
        mamba_allocator=SimpleNamespace(available_size=lambda: slots)
    )
    return s


class TestIdleBoxMustAdmit713(CustomTestCase):
    def test_pp_admits_the_measured_idle_state(self):
        """The whole complaint in one assertion: with 22 tokens pending, 72033
        rows free and 3 state slots, the PP layout can prefill."""
        self.assertTrue(
            _sched()._layout_admits("pp", 0, PENDING),
            "pp must admit: rows >= min(chunk, pending) and slots >= 1 both hold",
        )

    def test_tp_correctly_refuses_the_same_state(self):
        """CAN-FAIL COUNTERWEIGHT. tp may only decode, and nothing is resident,
        so it must refuse -- otherwise 'nothing_can_run' would be false and the
        refusal would be correct rather than a defect."""
        self.assertFalse(_sched()._layout_admits("tp", 0, PENDING))

    def test_starved_pool_still_refuses(self):
        """The refusal must remain reachable for its real cause."""
        self.assertFalse(_sched(avail=0, evictable=0)._layout_admits("pp", 0, PENDING))

    def test_no_state_slot_still_refuses(self):
        self.assertFalse(_sched(slots=0)._layout_admits("pp", 0, PENDING))

    def test_no_pending_work_still_refuses(self):
        self.assertFalse(_sched()._layout_admits("pp", 0, 0))


class TestIdleLockedDiagnostic713(CustomTestCase):
    """The terms must be printed where they are computed -- external sampling
    could not show the divergence, which is the whole reason this exists."""

    def _run(self, avail, slots, pending, phase="tp"):
        from sglang.srt.managers import scheduler as m

        s = _sched(avail=avail, slots=slots)
        s._round_built_nothing = True
        s.phase_flip_active_stack = phase
        s._idle_locked_diag_at = 0.0
        with self.assertLogs(m.logger, level="WARNING") as cm:
            m.logger.warning("sentinel")
            s._idle_locked_inputs(0, pending)
        return "\n".join(cm.output)

if __name__ == "__main__":
    unittest.main()


class TestDiagnosticProbesAreAllDefended713(CustomTestCase):
    """Review amendment: every probe inside the diagnostic must be armoured.

    A bare call in the logger arguments would let the diagnostic KILL the
    scheduler round it exists to observe -- which is exactly how #715's RADIX
    SHAPE walk died inside the crash it was written to explain.

    ONE HONEST LIMIT, FOUND BY MUTATION. The _post_evict_rows arm of this is
    UNREACHABLE and is therefore not tested: that method swallows its own
    accessor exceptions internally, so it cannot raise, and a test that broke
    the accessors underneath PASSED against a deliberately undefended call
    site -- i.e. it proved nothing. Forcing the raise requires patching the
    bound method, which then raises inside _layout_admits (also a bare call)
    before the diagnostic is reached, so it tests neither. The hardening is
    kept as cheap insurance against a future edit that makes _post_evict_rows
    raise, but it is NOT claimed to fix a reachable defect today, and no test
    pretends otherwise. The mamba probe below IS reachable and is tested.
    """

    def _raising_sched(self, which):

        def boom():
            raise RuntimeError("probe exploded")

        s = _sched(avail=0, slots=0)
        s._round_built_nothing = True
        s.phase_flip_active_stack = "tp"
        s._idle_locked_diag_at = 0.0
        if which == "slots":
            s.req_to_token_pool = SimpleNamespace(
                mamba_allocator=SimpleNamespace(available_size=boom)
            )
        return s

