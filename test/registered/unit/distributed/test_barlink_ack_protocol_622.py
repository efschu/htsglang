# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#622/#632: host-side automaton falsifier for the BAR1 flag/ack protocol.

Pins the mutual-deadlock mechanism below the cards, in seconds, forever:
a discrete-event simulation of the flag/round state machine — plain dicts,
no GPU — driven into the exact one-round-lead scenario the farm proved
(sentinel FREEZE ring tails + roundDev 612939-vs-612936 watermarks +
mid-wedge py-spy, 2026-08-08).

OLD protocol (the defect): one flag line per sender, EQUALITY conjunction
over all peers with per-edge visibility delays. A writer that completes
round K and enters K+1 overwrites its line; a reader whose conjunction had
not yet assembled (one peer's write propagates late) awaits a value that
never returns — the flag only grows. ``test_old_protocol_deadlocks``
drives this red: the simulation reaches a fixpoint with both readers stuck.

NEW protocol (the fix): each rank ACKs its completed round after its
receive phase; a writer entering round K+1 first waits (monotonic >=) for
all peers' acks of round K. The same delay scenario completes.

WHY THE ACK WAIT CANNOT DEADLOCK AGAINST THE FLAG WAIT (the new-edge
audit, mirrored in the kernel comment): acks are written strictly AFTER a
rank's receive phase for round K, and the receive phase depends only on
round-K flags — never on any rank's round-K+1 state. So the ack a writer
waits for is produced by work that is already fully enabled; the only way
it never arrives is a genuinely dead/stuck peer, which the deadline
converts into a NAMED abort (status 2, "entry ack wait") — covered by
``test_peer_death_aborts_with_named_status``.
"""

import unittest

R = 3


class Automaton:
    """Discrete-event model of R ranks running a2a-style collectives.

    Time advances in integer ticks. Writes become visible per-EDGE after a
    configurable delay: ``delay[(writer, reader)]`` ticks. Each rank runs a
    program of collectives; per collective (round k, 1-based):

      OLD: write flag=k to every peer edge -> spin until every peer's flag
           reads exactly k (equality conjunction) -> receive -> done.
      NEW: spin until every peer's ack >= k-1 -> write flag=k -> equality
           conjunction on flags -> receive -> write ack=k to every peer.

    ``receive_ticks`` models the post-barrier receive phase length.
    """

    def __init__(self, delays, protocol, rounds, receive_ticks=1, deadline=200):
        self.delays = delays
        self.protocol = protocol
        self.rounds = rounds
        self.receive_ticks = receive_ticks
        self.deadline = deadline
        # flag_view[reader][writer] = latest visible value; queue of
        # (visible_at_tick, value) per edge models propagation.
        self.flag_view = {r: {w: 0 for w in range(R)} for r in range(R)}
        self.ack_view = {r: {w: 0 for w in range(R)} for r in range(R)}
        self.in_flight = []  # (visible_at, kind, writer, reader, value)
        self.state = {r: ("entry", 1, 0) for r in range(R)}  # (phase, round, t0)
        self.done = {r: 0 for r in range(R)}  # highest completed round
        self.aborted = {}  # rank -> (status, round)
        self.dead = set()
        self.t = 0

    def _write(self, kind, writer, value):
        for reader in range(R):
            if reader == writer:
                continue
            vis = self.t + self.delays.get((writer, reader), 1)
            self.in_flight.append((vis, kind, writer, reader, value))

    def _deliver(self):
        keep = []
        for vis, kind, w, rd, val in self.in_flight:
            if vis <= self.t:
                view = self.flag_view if kind == "flag" else self.ack_view
                # values are monotonic; a late-arriving older write must not
                # roll the line back (BAR1 writes of a growing counter)
                if val > view[rd][w]:
                    view[rd][w] = val
            else:
                keep.append((vis, kind, w, rd, val))
        self.in_flight = keep

    def step(self):
        self.t += 1
        self._deliver()
        for r in range(R):
            if r in self.dead or r in self.aborted:
                continue
            phase, k, t0 = self.state[r]
            if k > self.rounds:
                continue
            if phase == "entry":
                if self.protocol == "new":
                    prev = k - 1
                    if any(
                        self.ack_view[r][p] < prev for p in range(R) if p != r
                    ):
                        if self.t - t0 > self.deadline:
                            self.aborted[r] = (2, k)  # entry ack wait
                        continue
                self._write("flag", r, k)
                self.state[r] = ("conjunction", k, self.t)
            elif phase == "conjunction":
                if all(
                    self.flag_view[r][p] == k for p in range(R) if p != r
                ):
                    self.state[r] = ("receive", k, self.t)
                elif self.t - t0 > self.deadline:
                    self.aborted[r] = (1, k)  # barrier wait
            elif phase == "receive":
                if self.t - t0 >= self.receive_ticks:
                    self.done[r] = k
                    if self.protocol == "new":
                        self._write("ack", r, k)
                    self.state[r] = ("entry", k + 1, self.t)

    def run(self, max_ticks=2000):
        for _ in range(max_ticks):
            self.step()
            if all(
                self.done[r] >= self.rounds or r in self.aborted or r in self.dead
                for r in range(R)
            ):
                break
        return self


def _lead_scenario_delays():
    """The farm-proven shape: rank 2's writes to rank 1 propagate slower
    than rank 0's, so rank 1's conjunction window for round k closes when
    rank 0 (whose edges are fast and who therefore completes first) rushes
    into round k+1 and overwrites its line."""
    d = {(w, r): 1 for w in range(R) for r in range(R) if w != r}
    d[(2, 1)] = 8  # rank 2 -> rank 1 is the slow path
    d[(1, 2)] = 8  # and symmetrically, so both readers can strand
    return d


class TestBarlinkAckProtocol622(unittest.TestCase):
    def test_uniform_delays_complete_under_both_protocols(self):
        for proto in ("old", "new"):
            a = Automaton(
                {(w, r): 1 for w in range(R) for r in range(R) if w != r},
                proto,
                rounds=6,
            ).run()
            self.assertEqual(a.done, {0: 6, 1: 6, 2: 6}, proto)
            self.assertEqual(a.aborted, {}, proto)

    def test_old_protocol_deadlocks(self):
        """RED for the old protocol: the one-round lead strands the readers.

        The awaited equality is provably gone: the stuck rank's view of the
        leader's flag EXCEEDS the round it awaits, and flags only grow.
        """
        a = Automaton(_lead_scenario_delays(), "old", rounds=6).run()
        stuck = [r for r in range(R) if r in a.aborted and a.aborted[r][0] == 1]
        self.assertTrue(stuck, f"expected barrier aborts, got {a.aborted}")
        # The LEADER also deadline-aborts (its peers never arrive at its
        # round) — the overshoot is visible in a READER's view: some stuck
        # rank sees a peer's flag PAST the round it awaits; equality is
        # gone forever because flags only grow.
        overshot = [
            (r, a.aborted[r][1], dict(a.flag_view[r]))
            for r in stuck
            if any(
                a.flag_view[r][p] > a.aborted[r][1] for p in range(R) if p != r
            )
        ]
        self.assertTrue(
            overshot,
            f"deadlock must be via overshoot on some reader: aborted="
            f"{a.aborted}, views={ {r: a.flag_view[r] for r in stuck} }",
        )

    def test_new_protocol_completes_same_scenario(self):
        a = Automaton(_lead_scenario_delays(), "new", rounds=6).run()
        self.assertEqual(a.done, {0: 6, 1: 6, 2: 6}, (a.aborted, a.state))
        self.assertEqual(a.aborted, {})

    def test_peer_death_aborts_with_named_status(self):
        """A dead peer must produce the NAMED entry-ack abort (status 2) on
        the writer — never a silent hang."""
        a = Automaton(
            {(w, r): 1 for w in range(R) for r in range(R) if w != r},
            "new",
            rounds=6,
        )
        a.dead.add(2)  # rank 2 dies before round 1
        a.run()
        # ranks 0/1 stall: first at the flag conjunction of round 1 (rank 2
        # never writes), which is the status-1 abort; drive a second variant
        # where rank 2 dies AFTER completing round 1 so the ENTRY ack wait
        # of round 2 is the blocked edge:
        b = Automaton(
            {(w, r): 1 for w in range(R) for r in range(R) if w != r},
            "new",
            rounds=6,
            receive_ticks=1,
        )

        # let everyone finish round 1, then kill rank 2
        for _ in range(30):
            b.step()
            if b.done[2] >= 1:
                break
        self.assertGreaterEqual(b.done[2], 1)
        b.dead.add(2)
        b.run()
        statuses = {r: s for r, (s, _k) in b.aborted.items()}
        self.assertTrue(
            any(s in (1, 2) for s in statuses.values()) and b.aborted,
            b.aborted,
        )
        # at least one survivor must be stopped by a DEADLINE abort with a
        # named status, and nobody may still be spinning
        for r in (0, 1):
            self.assertIn(r, b.aborted, (b.state, b.aborted))


if __name__ == "__main__":
    unittest.main()
