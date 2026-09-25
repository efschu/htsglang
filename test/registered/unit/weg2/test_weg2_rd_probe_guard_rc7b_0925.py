"""Review V (RC7b): an r_D probe error never turns a SERVED leg 2 into a 503.

`_sample_r_d` (front.py, handle_generate) was guarded by try/except only on the
STREAMED call site; on the non-streamed branch an exception inside it escaped
after D had answered 200 and the client got a 503 for a request that had been
served. The guard now lives inside the function, for both wire shapes: the
error is counted (`r_d_probe_errors`) and logged, never propagated.
"""

import asyncio
from unittest import mock

from sglang.srt.weg2 import front as front_mod
from sglang.test.test_utils import CustomTestCase

from .test_weg2_x_split_rc7_0925 import _probe


def _boom(*_a, **_k):
    raise RuntimeError("probe exploded")


class TheProbeErrorIsCountedNeverServed(CustomTestCase):
    def _run(self, stream):
        with mock.patch.object(front_mod, "r_d_probe", _boom):
            return asyncio.run(_probe(stream=stream))

    def test_non_streamed_leg_stays_200(self):
        """RED before the fix: the exception escaped the non-streamed branch."""
        f, res = self._run(stream=False)
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(f.counters["r_d_probe_errors"], 1)
        self.assertEqual(len(f._x_samples["r_d"]), 0)

    def test_streamed_leg_stays_200(self):
        f, res = self._run(stream=True)
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(f.counters["r_d_probe_errors"], 1)
        self.assertEqual(len(f._x_samples["r_d"]), 0)

    def test_without_an_error_the_sample_is_taken_as_before(self):
        f, res = asyncio.run(_probe(stream=False))
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(f.counters["r_d_probe_errors"], 0)
        self.assertEqual(len(f._x_samples["r_d"]), 1)
