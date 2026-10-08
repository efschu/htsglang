"""FLIPCYCLE H1 (02.10.): an empty flip ledger skips the drain's progress read.

y6z D->P: after the park the ledger is empty (the parked requests are kept,
not waited for), yet `drain` first read D's /get_server_info -- p50 81 / p90
400 ms on the flip's critical path while D's scheduler ran its post-park
publish. No window, no witness: the read is skipped.
"""

import asyncio
import collections
import types
import unittest

from flliper.srt.environ import envs
from flliper.srt.pdflip import front


def _front(outstanding):
    f = types.SimpleNamespace(counters=collections.Counter(), epoch=12, drain_deadline_s=120.0)
    reads = []

    async def progress(g):
        reads.append(g.name)
        return {"tokens": 0}

    f._pdflip_decode_progress = progress
    f._flip_ledger = lambda g: dict(g.outstanding)
    g = types.SimpleNamespace(name="D", outstanding=dict(outstanding))
    return f, g, reads


class TheEmptyLedger(unittest.TestCase):
    def test_no_progress_read_when_nothing_is_outstanding(self):
        f, g, reads = _front({})
        self.assertTrue(asyncio.run(front.Front.drain(f, g)))
        self.assertEqual(reads, [])
        self.assertEqual(f.counters["drain_empty_skip"], 1)

    def test_switch_off_reads_as_before(self):
        f, g, reads = _front({})
        with envs.FLLIPER_PDFLIP_ENABLE_DRAIN_EMPTY_SKIP.override(False):
            self.assertTrue(asyncio.run(front.Front.drain(f, g)))
        self.assertEqual(reads, ["D"])

    def test_a_running_request_still_opens_the_window(self):
        f, g, reads = _front({"pdflip-1-1": 1.0})

        async def run():
            t = asyncio.ensure_future(front.Front.drain(f, g))
            await asyncio.sleep(0.05)
            g.outstanding.clear()
            return await t

        self.assertTrue(asyncio.run(run()))
        self.assertEqual(reads, ["D"])


if __name__ == "__main__":
    unittest.main()
