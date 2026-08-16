# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#689: a reclaim ask is judged by what MOVED, not by what was already there.

THE FALSE SUCCESS, MEASURED. 2026-08-16 12:29, three consecutive asks from the
seam on the binding rank, all reporting success and all freeing nothing:

    asked the corridor guard for 178 MiB (pp_to_tp): ok=True,
    spendable now 609 MiB against a need of 788 MiB

609 did not move across any of the three. The guard's verdict was
``ok = free_now >= want`` -- and with 1428 MiB of driver-free on that card,
"are 178 MiB free" is trivially true whether or not the ladder reclaimed a
single byte. The seam then abandoned anyway, having been told it was funded.

That question is the RIGHT one for an allocator about to allocate ``want``,
which is what every existing caller is, so the verdict is not changed
underneath them. What was missing is a way to ask the OTHER question -- "did
you actually free this much" -- which is the only one a caller already holding
the memory can use. ``must_reclaim=True`` asks it.

WHY IT MATTERS BEYOND ONE RECEIPT: fundable_width's pre-arm picture is built
from what the guard says it can deliver. An optimistic guard makes that
picture optimistic, so the window is formed for a width the seam cannot carry
and the failure surfaces later, as an abandon, instead of earlier, as a
narrower window.
"""

import unittest

from sglang.srt.managers.corridor_guard import CorridorGuard
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1024 * 1024


def _guard(free_mib, *, providers=()):
    """A guard over a fake device whose free column we control."""
    state = {"free": int(free_mib) * MIB}
    g = CorridorGuard(
        device_index=0,
        floor_mib=0,
        delta_mib=0,
        probe=lambda: state["free"],
        law_floor_mib=0,
    )
    for name, cost, pays in providers:
        def _mk(pays=pays):
            def _spend(want):
                give = min(int(want), int(pays) * MIB)
                state["free"] += give
                return give
            return _spend
        g.register(name, cost, _mk())
    return g, state


class AnAskThatFreedNothingIsNotASuccess(unittest.TestCase):
    def test_the_1229_specimen(self):
        """1428 MiB free, ask 178, no provider that can pay: must be ok=False."""
        g, _ = _guard(1428)
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertFalse(
            res.ok,
            "the ladder freed nothing, so the ask was not served; reporting "
            "success here is what told the seam it was funded three times "
            "before it abandoned anyway.",
        )
        self.assertEqual(0, res.reclaimed)

    def test_the_reason_names_the_shortfall(self):
        g, _ = _guard(1428)
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertIn("reclaim", (res.detail or "").lower())

    def test_a_provider_that_pays_enough_succeeds(self):
        g, _ = _guard(1428, providers=[("arena", 30, 200)])
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertTrue(res.ok)
        self.assertGreaterEqual(res.reclaimed, 178 * MIB)

    def test_a_provider_that_underpays_still_fails(self):
        g, _ = _guard(1428, providers=[("arena", 30, 50)])
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertFalse(res.ok)


class TheDefaultContractIsUnCHANGED(unittest.TestCase):
    """Every existing caller is an allocator about to allocate ``want``.

    For them "is want allocatable" is the right question and the answer must
    not move, or this fix breaks the callers it was meant to un-lie to.
    """

    def test_plenty_free_and_no_provider_still_passes_by_default(self):
        g, _ = _guard(1428)
        res = g.ensure_headroom(178 * MIB, reason="an allocation")
        self.assertTrue(res.ok)

    def test_not_enough_free_still_fails_by_default(self):
        g, _ = _guard(100)
        res = g.ensure_headroom(178 * MIB, reason="an allocation")
        self.assertFalse(res.ok)


if __name__ == "__main__":
    unittest.main()
