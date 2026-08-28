# SPDX-License-Identifier: Apache-2.0
"""#974: the #789 readiness gate must count per MESSAGE KIND, not per wire.

THE DEFECT, measured as a CONTRAST between two boots of window-flip-0828 that
differ by one unrelated fix:

    boot 1   an admission livelock (#944) froze ALL traffic on CHAN_DICT.
             The shared posted/consumed counters froze with it, so the gate
             saw no progress and fired TWICE: "#789 READINESS TIMEOUT".
             A loud, diagnosable death.

    boot 2   #971 fixed that livelock. The upstream therefore stayed busy
             posting `admission_decision` messages while no `output` was
             posted at all. `consumed < posted` was true at EVERY poll, so
             the gate returned early EVERY time, PP2 walked into the
             unbounded output receive, and the rig wedged in silence.
             Zero gate firings.

THE CLASS, and it is the sharpest instance of it this tree has: a compensator
whose TRIGGER CONDITION was being supplied by a foreign defect. Fixing the
foreign defect -- correctly -- disarmed the compensator. Nothing about the
gate changed; the traffic around it did.

THE ROOT is that CHAN_DICT is ONE physical wire carrying three kinds
(`proxy`, `output`, `admission_decision`), demultiplexed by `__msg_type__`
after the fact, and the gate read the wire's counters. So it could answer
"is ANY message coming?" and was being asked "is MINE coming?". The gate's
own docstring named that imprecision as an accepted limit -- the
guard-comment-names-the-hazard shape -- and boot 2 is the measurement that
it was the defect rather than a caveat.

THE FIX gives the counters a KIND AXIS on the mechanism they already have:
one sub-channel file per kind (`dict|output`), same single writer, same
monotonicity, same sweep. The wire counters keep their meaning and their
readers (the drains, which must take off the wire only what is provably on
it). The gate reads the kind's own counters -- but only once
`kind_axis_covers` proves the upstream labels everything it posts, so an
unlabelled or stand-in sender falls back to exactly the pre-#974 behaviour.

WHY THIS TEST IS IN-PROCESS AND TOUCHES NO TRANSPORT. The gate never touches
the tensor-dict wire; it polls an out-of-band /dev/shm side channel. So the
whole defect reproduces with two real `PhaseFlipCounters` (the victim's and
its upstream's, same directory and instance tag, exactly as production) and
the real production send path driven against a stand-in wire. That also
avoids corpse F structurally: there is no `work.wait(timeout)` here to close
a gloo pair, because there is no gloo pair.

HANG-SHAPE IS DISTINGUISHED FROM RAISE-SHAPE. Every arm drives the gate on a
worker thread with its own join deadline, so "the gate returned", "the gate
raised" and "the gate never came back" are three separate observations and
never collapse into one another -- the same discipline the #973 test applies
with its fsynced marker.
"""

import os
import tempfile
import threading
import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30)

WORLD = 3
#: PP2 -- the rank that entered the unbounded output receive in boot 2.
VICTIM = 2
#: PP1, its upstream on the ring.
UPSTREAM = 1
LIVE_MB = 1

#: Short enough to keep the suite fast, long enough that a poll loop at
#: PROXY_READINESS_POLL_STEP_S (0.02s) takes many turns inside it.
BUDGET_S = 0.5
#: The outer bound on the worker thread. Comfortably above the gate's own
#: budget: anything still running at this point is a HANG, not a slow raise.
JOIN_S = 6.0
INSTANCE = "t974"


class _Wire:
    """Only what `resolve_src`, `typed_inbox` and the send path touch."""

    def __init__(self, rank):
        self.rank_in_group = rank
        self.world_size = WORLD
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1
        self.sent = []

    def send_tensor_dict(self, tensor_dict=None, all_gather_group=None, **kw):
        # The counters, not the transport, are what this defect lives in --
        # so the send is recorded and returns no work handles.
        self.sent.append(dict(tensor_dict))
        return []


def _counters(directory, rank):
    from sglang.srt.managers.phase_flip_counters import PhaseFlipCounters

    return PhaseFlipCounters(
        n_ranks=WORLD, rank=rank, directory=directory, instance=INSTANCE
    )


def _upstream_holder(directory):
    """A sender driving the REAL `_pp_send_dict_to_next_stage`.

    Deliberately the production function rather than hand-written counter
    bumps: what is under test is whether the shipped send path labels what it
    posts, so a test that bumped the counters itself would be asserting its
    own arithmetic.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=_Wire(UPSTREAM),
        pp_flip_counters=_counters(directory, UPSTREAM),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        _pp_gapped_wire=False,
    )
    h._pp_boundary_stats = lambda: None
    for name in (
        "_pp_send_dict_to_next_stage",
        "_pp_flip_bump_sent",
        "_pp_flip_bump_attempted",
    ):
        # Bound only if it exists, so the arms that measure BEHAVIOUR can be
        # run against the pre-#974 tree and produce a behavioural red (the
        # gate returning) rather than an AttributeError that would prove
        # nothing about the defect.
        fn = getattr(SchedulerPPMixin, name, None)
        if fn is not None:
            setattr(h, name, types.MethodType(fn, h))
    return h


def _victim_holder(directory):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=_Wire(VICTIM),
        pp_flip_counters=_counters(directory, VICTIM),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
    )
    h._pp_flip_upstream = lambda: UPSTREAM
    for name in (
        "_pp_wait_for_dict_readiness",
        "_pp_wait_for_proxy_readiness",
        "_pp_flip_bump_consumed",
    ):
        # Bound only if it exists, so the arms that measure BEHAVIOUR can be
        # run against the pre-#974 tree and produce a behavioural red (the
        # gate returning) rather than an AttributeError that would prove
        # nothing about the defect.
        fn = getattr(SchedulerPPMixin, name, None)
        if fn is not None:
            setattr(h, name, types.MethodType(fn, h))
    return h


def _post(upstream, kind, n=1):
    """`n` messages of `kind` through the production send path."""
    for i in range(n):
        upstream._pp_send_dict_to_next_stage(
            {"payload": i}, async_send=True, msg_type=kind
        )


def _drive(victim, kind, join_s=JOIN_S):
    """Run the gate on its own thread. Returns one of three outcomes.

    'returned' | 'raised' | 'hung' -- never inferred from one another, which
    is the whole point: boot 2's failure was the gate RETURNING, and a test
    that could not tell a return from a hang would have called it a pass.
    """
    box = {"outcome": "hung", "error": None}

    def _run():
        try:
            victim._pp_wait_for_dict_readiness(LIVE_MB, kind)
        except Exception as exc:  # noqa: BLE001 - the outcome under test
            box["error"] = str(exc)
            box["outcome"] = "raised"
        else:
            box["outcome"] = "returned"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(join_s)
    box["still_running"] = t.is_alive()
    return box


class PPReadinessPerKind974(unittest.TestCase):
    def setUp(self):
        from sglang.srt.managers.scheduler_pp_mixin import ENV_PROXY_READINESS_BUDGET

        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self._env_key = ENV_PROXY_READINESS_BUDGET
        self._env_old = os.environ.get(self._env_key)
        os.environ[self._env_key] = str(BUDGET_S)
        self.upstream = _upstream_holder(self.dir)
        self.victim = _victim_holder(self.dir)

    def tearDown(self):
        if self._env_old is None:
            os.environ.pop(self._env_key, None)
        else:
            os.environ[self._env_key] = self._env_old
        self._tmp.cleanup()

    # ------------------------------------------------------------- arm 1
    def test_boot2_shape_other_kind_traffic_no_longer_disarms_the_gate(self):
        """ARM 1, THE DEFECT. A busy wire carrying somebody else's kind.

        The upstream posts `admission_decision` continuously, as it did in
        boot 2 once #971 unblocked it. Nothing of kind `output` is ever
        posted. Before the kind axis the wire counters read
        `consumed(0) < posted(12)` at every poll and the gate returned
        immediately, sending the caller into the unbounded output receive.
        After it, the gate sees zero posted and zero entered on
        `dict|output` and refuses -- bounded, named, loud.
        """
        from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

        _post(self.upstream, ADMISSION_DECISION_KIND, n=12)

        res = _drive(self.victim, "output")

        self.assertEqual(
            res["outcome"],
            "raised",
            f"the gate did not fire while the wire was busy with another "
            f"kind -- this is boot 2's silent wedge: {res}",
        )
        self.assertIn("#789 OUTPUT READINESS TIMEOUT", res["error"])
        self.assertIn(f"mb_id={LIVE_MB}", res["error"])
        self.assertIn(f"upstream (rank {UPSTREAM})", res["error"])

    def test_the_wire_counter_really_did_show_progress_all_along(self):
        """The disarm condition, asserted rather than assumed.

        If the shared counters did NOT show progress in arm 1's setup, arm 1
        would be passing for a trivial reason (an idle wire) and would prove
        nothing about the per-kind axis. Pin the premise: on the wire the
        upstream is provably ahead, and it is only ON THIS KIND that it is
        not.
        """
        from sglang.srt.managers.phase_flip_counters import CHAN_DICT
        from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

        _post(self.upstream, ADMISSION_DECISION_KIND, n=12)
        c = self.victim.pp_flip_counters

        self.assertEqual(c.sent(CHAN_DICT, UPSTREAM), 12)
        self.assertEqual(c.local_consumed(CHAN_DICT), 0)
        self.assertLess(
            c.local_consumed(CHAN_DICT),
            c.sent(CHAN_DICT, UPSTREAM),
            "the wire counter must read as PROGRESS here -- that reading is "
            "exactly what disarmed the gate in boot 2",
        )
        self.assertEqual(c.sent_of_kind(CHAN_DICT, "output", UPSTREAM), 0)
        self.assertEqual(
            c.sent_of_kind(CHAN_DICT, ADMISSION_DECISION_KIND, UPSTREAM), 12
        )

    # ------------------------------------------------------------- arm 2
    def test_healthy_progress_on_this_kind_returns_immediately(self):
        """ARM 2, THE DIRECTION THAT MUST NOT MOVE. A posted `output`.

        The false-positive direction is the safe one for this gate; the
        false-NEGATIVE direction is an outage. A gate that raised whenever it
        was reached would pass arm 1 and fail here.
        """
        _post(self.upstream, "output", n=1)

        res = _drive(self.victim, "output")

        self.assertEqual(res["outcome"], "returned", f"{res}")

    def test_a_rendezvous_in_progress_still_counts_as_a_signal(self):
        """The #789 `attempted` protection survives the kind axis.

        An upstream INSIDE a send for this rank is positive evidence -- the
        only evidence available while that send is the lazy NCCL communicator
        rendezvous, which cannot return until this rank enters the receive.
        Boots instr7/instr8 died of getting this wrong; it must not regress
        because the counters gained an axis.
        """
        from sglang.srt.managers.phase_flip_counters import CHAN_DICT, kind_channel

        # Entered, not yet posted: exactly what `bump_attempted` publishes on
        # the line before the send call.
        self.upstream.pp_flip_counters.bump_attempted(kind_channel(CHAN_DICT, "output"))
        self.upstream.pp_flip_counters.bump_attempted(CHAN_DICT)
        # ...and one completed post of another kind, so the kind axis is
        # covered and this arm exercises the per-kind path rather than the
        # fallback.
        _post(self.upstream, "proxy", n=1)

        res = _drive(self.victim, "output")

        self.assertEqual(
            res["outcome"],
            "returned",
            f"the gate refused a send its upstream had already entered: {res}",
        )

    # ------------------------------------------------------------- arm 3
    def test_an_upstream_that_posted_nothing_is_unchanged(self):
        """The original #789 case, byte-identical. No posts at all: both axes
        say the same thing, the fallback keeps the shipped path, and the gate
        raises exactly as it always has."""
        res = _drive(self.victim, "proxy")

        self.assertEqual(res["outcome"], "raised", f"{res}")
        self.assertIn("#789 PROXY READINESS TIMEOUT", res["error"])

    def test_an_unlabelled_sender_falls_back_to_the_wire_counter(self):
        """THE COMPATIBILITY CONTRACT, and the reason the axis is not simply
        trusted.

        A sender that bumps the wire counter without labelling -- every
        stand-in holder in the #631/#757/#787/#789/#791/#795/#797/#798 test
        family, and any future send site that forgets -- must not be read as
        "no message of your kind is coming". `kind_axis_covers` detects the
        shortfall and the gate uses the wire counters, which is what those
        tests have always measured.
        """
        from sglang.srt.managers.phase_flip_counters import (
            CHAN_DICT,
            kind_axis_covers,
        )

        self.upstream.pp_flip_counters.bump_sent(CHAN_DICT)

        self.assertFalse(
            kind_axis_covers(self.victim.pp_flip_counters, CHAN_DICT, UPSTREAM),
            "an unlabelled post must make the axis unusable",
        )
        res = _drive(self.victim, "output")
        self.assertEqual(
            res["outcome"],
            "returned",
            f"the fallback did not preserve the pre-#974 behaviour: {res}",
        )

    def test_a_counters_object_without_the_kind_api_is_tolerated(self):
        """The stub-counters holder of test_pp_proxy_readiness_contract_789,
        in one line: presence-tested, not assumed."""
        from sglang.srt.managers.phase_flip_counters import (
            CHAN_DICT,
            kind_axis_covers,
        )

        stub = types.SimpleNamespace(
            sent=lambda chan, rank: 5,
            attempted=lambda chan, rank: 5,
            local_consumed=lambda chan: 0,
        )
        self.assertFalse(kind_axis_covers(stub, CHAN_DICT, UPSTREAM))

    # ------------------------------------------------------------- arm 4
    def test_consuming_another_kind_is_not_progress_on_mine(self):
        """The receive side of the same asymmetry.

        Taking an `admission_decision` off the wire bumps the wire's consumed
        count -- it really did leave the wire, and the upstream's blocking
        commit depends on that being counted. It must NOT be charged to
        `output`, or a rank that consumed somebody else's message would look
        like a rank that had caught up on its own.
        """
        from sglang.srt.managers.phase_flip_counters import CHAN_DICT
        from sglang.srt.managers.scheduler_pp_mixin import (
            ADMISSION_DECISION_KIND,
            _pp_flip_bump_kind,
        )

        _post(self.upstream, ADMISSION_DECISION_KIND, n=3)
        c = self.victim.pp_flip_counters
        for _ in range(3):
            _pp_flip_bump_kind(
                self.victim, "bump_consumed", CHAN_DICT, ADMISSION_DECISION_KIND
            )
            self.victim._pp_flip_bump_consumed(CHAN_DICT)

        self.assertEqual(c.local_consumed(CHAN_DICT), 3)
        self.assertEqual(
            c.local_consumed_of_kind(CHAN_DICT, ADMISSION_DECISION_KIND), 3
        )
        self.assertEqual(c.local_consumed_of_kind(CHAN_DICT, "output"), 0)

        # And the wire now reads "all caught up", which is the OTHER way the
        # shared counter misinforms this gate -- silently, in the opposite
        # direction from arm 1.
        res = _drive(self.victim, "output")
        self.assertEqual(res["outcome"], "raised", f"{res}")

    def test_the_kind_bump_is_reachable_from_a_holder_that_never_bound_it(self):
        """THE REGRESSION THIS FIX ALREADY CAUSED ONCE, pinned.

        The first cut made the kind bump a METHOD, and every stand-in holder
        in this tree is a `types.SimpleNamespace` that binds mixin methods
        one name at a time. `test_pp_retracted_pass_void_797.py` went from 31
        passed to 7 failed, all `AttributeError: 'types.SimpleNamespace'
        object has no attribute '_pp_flip_bump_kind'`.

        So: a holder carrying ONLY the names the shipped send path has always
        needed must be able to complete a send. Constructed here without the
        helper list at all, so the assertion cannot be satisfied by this
        module's own holders happening to bind more than production requires.
        """
        from sglang.srt.managers.phase_flip_counters import CHAN_DICT
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        bare = types.SimpleNamespace(
            pp_group=_Wire(UPSTREAM),
            pp_flip_counters=_counters(self.dir, UPSTREAM),
            require_attn_tp_allgather=False,
            attn_tp_group=None,
            _pp_gapped_wire=False,
        )
        bare._pp_boundary_stats = lambda: None
        for name in (
            "_pp_send_dict_to_next_stage",
            "_pp_flip_bump_sent",
            "_pp_flip_bump_attempted",
        ):
            setattr(bare, name, types.MethodType(getattr(SchedulerPPMixin, name), bare))

        bare._pp_send_dict_to_next_stage({"x": 1}, async_send=True, msg_type="output")

        self.assertEqual(bare.pp_flip_counters.local_sent(CHAN_DICT), 1)
        self.assertEqual(
            bare.pp_flip_counters.sent_of_kind(CHAN_DICT, "output", UPSTREAM),
            1,
            "the labelled post must reach the sub-channel without the holder "
            "binding anything new",
        )

    # ------------------------------------------------------------- arm 5
    def test_can_fail_neutering_the_kind_axis_restores_the_disarm(self):
        """ARM 5, THE MUTANT -- and it is the pre-#974 gate exactly.

        `kind_axis_covers` returning False IS the old code path: the gate
        reads the wire counters and nothing else. So this arm does not
        approximate the defect, it reinstates it. Arm 1's constellation must
        go back to returning, which is the silent wedge.
        """
        import sglang.srt.managers.scheduler_pp_mixin as mod
        from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

        _post(self.upstream, ADMISSION_DECISION_KIND, n=12)

        with mock.patch.object(mod, "kind_axis_covers", lambda *a, **k: False):
            res = _drive(self.victim, "output")

        self.assertEqual(
            res["outcome"],
            "returned",
            "with the kind axis neutered the gate must be disarmed again -- "
            "if it still fires, arm 1 is not measuring the kind axis",
        )

    # ------------------------------------------------------------- arm 6
    def test_the_sub_channel_is_swept_with_everything_else(self):
        """Boot hygiene is not a second mechanism either.

        `sweep` matches on the instance prefix and the role/rank suffix; a
        sub-channel file carries both unchanged, so it is removed by the
        sweep that already existed. Asserted because a counter file that
        outlived its boot would be read as a phantom message next boot.
        """
        from sglang.srt.managers.phase_flip_counters import CHAN_DICT

        _post(self.upstream, "output", n=2)
        files = os.listdir(self.dir)
        self.assertTrue(
            any(f"{CHAN_DICT}|output" in name for name in files),
            f"no per-kind counter file was published: {files}",
        )

        removed = self.upstream.pp_flip_counters.sweep()
        left = [n for n in os.listdir(self.dir) if n.endswith(f".s{UPSTREAM}")]
        self.assertGreaterEqual(removed, 2, "the sweep missed the sub-channel")
        self.assertEqual(left, [], f"counter files survived the sweep: {left}")


if __name__ == "__main__":
    unittest.main()
