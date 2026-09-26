"""#1459c: leg-1 completions that land in ONE event-loop round reach D in
dispatch (= arrival) order, not in hash-set order.

`_p_drain_pool` awaits `asyncio.wait(inflight, FIRST_COMPLETED)` and walked the
returned `done` SET. A P batch finishes several leg 1s at once (P prefills
`--max-running-requests` of them together), so several tasks arrive in one
`done` set and `on_done` -- which appends to `_ready_for_d`, the deque the D
admitter pops oldest-first (law 2) -- ran in the set's iteration order, which
follows the Task objects' ids. D then admitted the batch shuffled
(test_weg2_scheduling_slice_a_0907 t3: ['r01', 'r02', 'r03', 'r00', ...]).
Across rounds the order stays completion order, as the docstring states.
"""

import asyncio
import collections
import unittest

from sglang.srt.weg2 import front


def _drain(n, limit, gate_after=None):
    """n items; every item waits on one shared event, so all of them finish
    in the SAME round once it is set (a P batch completing together)."""

    async def main():
        ev = asyncio.Event()
        q = collections.deque(range(n))
        order = []
        started = [0]

        async def one(i):
            started[0] += 1
            if started[0] == min(n, limit):
                asyncio.get_running_loop().call_soon(ev.set)
            await ev.wait()
            return i

        await front._p_drain_pool(q, limit, one, order.append, lambda: True)
        return order

    return asyncio.run(main())


class ABatchThatCompletesTogetherKeepsItsOrder(unittest.TestCase):
    def test_one_round_is_delivered_in_dispatch_order(self):
        self.assertEqual(_drain(64, 64), list(range(64)))

    def test_several_rounds_each_in_dispatch_order(self):
        # limit 8: eight rounds of eight; within each, dispatch order
        self.assertEqual(_drain(64, 8), list(range(64)))


if __name__ == "__main__":
    unittest.main()
