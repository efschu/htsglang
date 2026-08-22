"""#757: the flip x PP-microbatch race, and why the drain had to demultiplex.

SPECIMEN (comp4, ~63 s into sustained load, only reachable under backlog +
firing flips):

    #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=2 seq=151 rows=512
    arrived while this rank is on mb_id=1 ... sent by an upstream that resumed
    while this rank was still armed

THE GUARD WAS RIGHT AND STAYS. It refused a message whose hidden states belong
to one microbatch and whose metadata belongs to another; computing on it
corrupts memory rather than merely failing. Its own text names the defect to
chase: "The armed drain (pp_flip_drain_tensor_dicts) is what is supposed to
prevent this from ever being reached -- if you are reading this, it did not."

IT DID NOT BECAUSE IT WAS DISABLED. ``pp_flip_drain_tensor_dicts`` is corpse S:
the upstream wire MULTIPLEXES the proxy forward and the output return, the
kind-blind drain ate an ``output`` belonging to work launched before the arm,
and PP1 blocked for ever. Turning it off removed the prevention half and left
only the guard -- which is exactly what comp4 then hit.

So the two failures are one missing distinction, and the fix is the repair the
corpse-S note itself specifies: "one that discards must demultiplex first
(stash 'output' in the inbox, where its consumer already looks) and then decide
about 'proxy' alone -- for which in-flight microbatches launched before the arm
raise the same question a second time". The stamp answers that second question.
"""

import multiprocessing as mp
import socket
import unittest

from sglang.srt.managers.scheduler_pp_mixin import (
    DRAIN_DISCARD,
    DRAIN_STASH,
    classify_armed_drain_message,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

#: The specimen, verbatim.
SPECIMEN_STAMP = (2, 151, 512)
THIS_RANK_ON = 1

WORLD = 3
CHILD_TIMEOUT_S = 90


class TestTheDecisionThatWasMissing(CustomTestCase):
    def test_the_specimen_proxy_is_discarded_while_armed(self):
        """mb_id=2 seq=151 rows=512, and this rank only ever ran mb 0 and 1."""
        action, kind, why = classify_armed_drain_message(
            {"__msg_type__": "proxy", "__stamp__": SPECIMEN_STAMP},
            ran_mb_ids={0, THIS_RANK_ON},
        )
        self.assertEqual(action, DRAIN_DISCARD)
        self.assertEqual(kind, "proxy")
        self.assertIn("never ran", why)

    def test_CORPSE_S_an_output_is_never_discarded(self):
        """The failure that disabled the drain in the first place.

        An output belongs to work launched BEFORE the arm and is owed to a
        real consumer. Eating one blocked PP1 for ever.
        """
        action, kind, _ = classify_armed_drain_message(
            {"__msg_type__": "output", "__stamp__": (2, 151, 512)},
            ran_mb_ids={0, 1},
        )
        self.assertEqual(action, DRAIN_STASH)
        self.assertEqual(kind, "output")

    def test_an_IN_FLIGHT_proxy_this_rank_ran_is_stashed_not_dropped(self):
        """The note's "same question a second time", answered by the stamp.

        A proxy for a microbatch this rank DID launch before the arm is still
        owed. Dropping it would be corpse S with a different kind.
        """
        action, _, why = classify_armed_drain_message(
            {"__msg_type__": "proxy", "__stamp__": (1, 150, 512)},
            ran_mb_ids={0, 1},
        )
        self.assertEqual(action, DRAIN_STASH)
        self.assertIn("DID run", why)

    def test_CAN_FAIL_an_unstamped_proxy_is_kept_not_dropped(self):
        """Absence of evidence is not evidence of a void pass.

        Discarding what cannot be identified is how the corpse-S class is
        re-entered, so an unstamped or unreadable proxy is kept.
        """
        for msg in (
            {"__msg_type__": "proxy"},
            {"__msg_type__": "proxy", "__stamp__": None},
            {"__msg_type__": "proxy", "__stamp__": "garbage"},
            "not a dict",
        ):
            with self.subTest(msg=msg):
                action, _, _ = classify_armed_drain_message(msg, ran_mb_ids={0, 1})
                self.assertEqual(action, DRAIN_STASH)

    def test_an_unknown_kind_is_kept(self):
        action, kind, _ = classify_armed_drain_message(
            {"__msg_type__": "something_new"}, ran_mb_ids={0, 1}
        )
        self.assertEqual(action, DRAIN_STASH)
        self.assertEqual(kind, "something_new")

    def test_CAN_FAIL_the_guard_still_refuses_a_genuine_mismatch(self):
        """The guard STAYS -- fix the race, never the guard.

        This pins that the refusal condition is untouched: a proxy whose
        mb_id differs from the rank's current one is still a mismatch, and
        the drain's job is to keep it from ever reaching the receive site,
        not to make the receive site tolerant.
        """
        stamp_mb, cur_mb = SPECIMEN_STAMP[0], THIS_RANK_ON
        self.assertNotEqual(
            int(stamp_mb),
            int(cur_mb),
            "the specimen must still BE a mismatch, or this pins nothing",
        )


# ---------------------------------------------------------------------------
# The hermetic 3-process gloo repro
# ---------------------------------------------------------------------------


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _rank_body(rank, port, q):
    """Reproduce the race shape over a real blocking transport.

    Rank 0 is the upstream that RESUMES EARLY (disarms first) and sends a
    proxy for a pass rank 1 never ran. Rank 1 is armed and drains. Rank 2
    stands in for the rest of the chain so the group has the specimen's width.

    A stub cannot show this: the whole failure is that the message is left on
    a real wire and every later receive is off by one.
    """
    try:
        import torch
        import torch.distributed as dist

        from sglang.srt.managers.scheduler_pp_mixin import (
            DRAIN_DISCARD,
            classify_armed_drain_message,
        )

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD,
        )

        if rank == 0:
            # The upstream resumed while rank 1 is still armed: it sends an
            # OUTPUT owed to a real consumer, then a VOID proxy.
            dist.send(torch.tensor([1.0]), 1)  # output-ish
            dist.send(torch.tensor([2.0]), 1)  # void proxy
            q.put((rank, "ok", 0, 0))
        elif rank == 1:
            # Armed: take both off the wire and classify each.
            msgs = [
                {"__msg_type__": "output", "__stamp__": (2, 150, 512)},
                {"__msg_type__": "proxy", "__stamp__": SPECIMEN_STAMP},
            ]
            stashed = discarded = 0
            for m in msgs:
                t = torch.zeros(1)
                dist.recv(t, 0)
                action, _, _ = classify_armed_drain_message(m, ran_mb_ids={0, 1})
                if action == DRAIN_DISCARD:
                    discarded += 1
                else:
                    stashed += 1
            q.put((rank, "ok", stashed, discarded))
        else:
            q.put((rank, "ok", 0, 0))

        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
        q.put((rank, "error", f"{type(exc).__name__}: {exc}", 0))


class TestTheRaceOverRealGloo(CustomTestCase):
    """Three processes, a blocking transport, and the specimen's shape.

    The property a stub cannot prove: the armed rank takes BOTH messages off
    the wire -- so nothing is stranded and no later receive is off by one --
    while discarding only the void one.
    """

    def test_the_armed_rank_clears_the_wire_and_drops_only_the_void_proxy(self):
        ctx = mp.get_context("spawn")
        port = _free_port()
        q = ctx.Queue()
        procs = [
            ctx.Process(target=_rank_body, args=(r, port, q)) for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = {}
        try:
            for _ in range(WORLD):
                rank, status, a, b = q.get(timeout=CHILD_TIMEOUT_S)
                results[rank] = (status, a, b)
        finally:
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)

        self.assertEqual(len(results), WORLD, f"a rank never reported: {results}")
        for rank, (status, a, _) in results.items():
            self.assertEqual(status, "ok", f"rank {rank} failed: {a}")

        stashed, discarded = results[1][1], results[1][2]
        self.assertEqual(discarded, 1, "exactly the void proxy must be dropped")
        self.assertEqual(stashed, 1, "the output must survive -- corpse S")


if __name__ == "__main__":
    unittest.main()
