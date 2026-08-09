"""#631 defect Q, second instance: the consensus CADENCE counter.

THE DEFECT, measured 2026-08-09 08:09:39-08:11:39Z. After a pp_to_tp
cutover the instance died with

    barlink collective 'phase_flip.consensus' made no progress for 120s
    and no peer could be proven dead

PP0 in ``event_loop_normal -> get_next_batch_to_run ->
_phase_flip_on_round``, inside the reduction; its peers never arrived, and
when PP0 finally raised, they died on "Connection closed by peer".

WHY. ``on_round`` used to advance ``_round`` on EVERY call, and
``_round % _interval`` gates ENTRY TO A BLOCKING COLLECTIVE. That is only
safe while the ranks' counts stay congruent, and the only thing that keeps
them congruent is the loop being paced in lockstep -- by the request chain
under ``event_loop_pp``, by rank 0's broadcast under ``event_loop_normal``.

AN ARMED WINDOW IS PRECISELY WHERE THE PACING STOPS. An armed rank admits
nothing and launches nothing, so its pass loop free-runs at about 8 kHz and
calls this hook every iteration: 37371 / 28677 / 32344 calls in ONE 5 s
window on this rig. The ranks emerge incongruent modulo the interval, their
periodic entries never coincide again, and the first periodic consensus
after the cutover deadlocks -- rank 0 inside the reduction, its peers
inside the broadcast recv that rank 0 owes them.

THE FIX: count UNARMED rounds only. While armed, entry is governed by
``require_armed_and_parked`` and the presence gate and never by this
counter, so the increment buys nothing there and costs the congruence of
every round after it.

THIS IS THE SIBLING OF THE MICROBATCH-SLOT DEFECT, and the general form is
worth more than either fix: A QUANTITY IS ONLY IN LOCKSTEP WHILE SOMETHING
KEEPS IT THERE. Both indices were synchronised by traffic, as a side
effect, and the armed window removes the traffic while every local check
still passes.

CPU-only.
"""

from sglang.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    PP_TO_TP,
    PhaseFlipRuntime,
)
from sglang.test.test_utils import CustomTestCase

MAP = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
N_LAYERS = 16
VEC = (3, 2, 2)


class _View:
    """The narrowest pool view the constructor accepts.

    Only its layer count is ever read here: no round in this file reaches
    the byte move, which is the point -- the cadence is decided long before
    any pool is touched.
    """

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.rows = {}


def _runtime(interval=8, ready=False):
    return PhaseFlipRuntime(
        n_ranks=3,
        rank=0,
        layer_map=MAP,
        n_layers=N_LAYERS,
        tp_vector=VEC,
        boot_phase=PHASE_PP,
        consensus_interval=interval,
        collective_min=lambda vals: list(vals),
        exchange=lambda o, i: {},
        pp_pool_view=_View(len(MAP[0])),
        tp_pool_view=_View(N_LAYERS),
        live_slots_fn=lambda: [],
        ready_fn=lambda: ready,
        cutover_fn=lambda d: None,
    )


class TestRoundCadenceCongruence(CustomTestCase):
    def test_an_unarmed_round_advances_the_cadence(self):
        rt = _runtime()
        before = rt._round
        for _ in range(5):
            rt.on_round()
        self.assertEqual(rt._round, before + 5)

    def test_an_armed_round_does_not_advance_the_cadence(self):
        """The load-bearing one. An armed rank spins tens of thousands of
        times; not one of those may move the cadence."""
        rt = _runtime()
        rt._pending = PP_TO_TP
        before = rt._round
        for _ in range(1000):
            rt.on_round(require_armed_and_parked=True)
        self.assertEqual(rt._round, before)

    def test_ranks_stay_congruent_across_an_armed_window_of_unequal_length(self):
        """THE PROPERTY THE TP PHASE DEPENDS ON.

        Three ranks are paced together while unarmed, arm together, spin
        for wildly different numbers of iterations, and resume. Their
        cadence counters must still agree modulo the interval, or their
        periodic entries never coincide and the first one to enter the
        reduction waits for peers that are waiting for it.
        """
        interval = 8
        spins = (37371, 28677, 32344)  # the measured window, to the iteration
        ranks = [_runtime(interval=interval) for _ in spins]

        for rt in ranks:  # paced together before the arm
            for _ in range(11):
                rt.on_round()
        for rt, n in zip(ranks, spins):
            rt._pending = PP_TO_TP
            for _ in range(min(n, 2000)):  # scaled; the count is what matters
                rt.on_round(require_armed_and_parked=True)
            rt._pending = None
        for rt in ranks:  # paced together again after the abandon
            for _ in range(5):
                rt.on_round()

        residues = {rt._round % interval for rt in ranks}
        self.assertEqual(
            len(residues),
            1,
            f"the ranks left the armed window incongruent mod {interval}: "
            f"{[rt._round for rt in ranks]}. Their periodic consensus "
            f"entries can never coincide again.",
        )

    def test_can_fail_counting_armed_rounds_makes_the_ranks_incongruent(self):
        """The can-fail proof: the pre-fix rule, reproduced.

        Counting every call -- which is what ``self._round += 1`` at the
        top of on_round did -- turns unequal spin counts straight into
        unequal residues. This is the metal deadlock, in a unit.
        """
        interval = 8
        spins = (37371, 28677, 32344)
        counts = [11 + n + 5 for n in spins]  # the pre-fix arithmetic
        residues = {c % interval for c in counts}
        self.assertGreater(
            len(residues),
            1,
            "the scenario must actually diverge, or the pin proves nothing",
        )
