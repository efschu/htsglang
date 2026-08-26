"""#871b: the BOOT GATE read host RAM from a second reimplementation.

`turnkey/preflight.py:_real_mem_available` opened `/proc/meminfo` and returned
`MemAvailable` raw, with zero references to the #407 owner. It feeds
`check_host_headroom`, which raises a HARD `REFUSE_HOST_HEADROOM` -- so this is
not a diagnostic that reads a slightly wrong number, it is the gate that decides
whether a boot is allowed to start.

IT OVER-REPORTS TWICE OVER, and both halves are already documented elsewhere in
this tree:

* LXCFS SCOPE. `memtier/profile.py:honest_host_memory_bytes` states it: inside
  this container `/proc/meminfo` is synthesised, `MemAvailable` can EXCEED
  `MemTotal` (observed on this rig), and with `memory.max` unlimited it reports
  the HOST's figures on a box other containers are also spending.
* CGROUP RESIDENT. The owner additionally clamps by what this cgroup already
  holds. Measured on this box: MemAvailable 113.19 GiB -> honest 112.95 GiB.

A gate that over-reports free RAM ADMITS a boot that should have been refused.
#721 is that outcome: a real container OOM with `oom_kill=17`.

AND THE SCOPE MUST BE IN THE MESSAGE. Measured here, three cgroup levels report
three different `memory.current` -- root 23.5, system.slice 23.3,
system.slice/claude.service 22.3 GiB -- and all three carry `memory.max = max`.
A refusal that says only "12.0 GiB" is not checkable against any of them,
because the reader cannot tell which question was asked.

WHAT THIS SUITE PINS, and the second half is the one that matters more:

1. the default probe reaches the #407 owner instead of re-reading /proc/meminfo;
2. THE GATE STILL REFUSES when host RAM is genuinely short. A fix that removes
   the over-reporting and loses the refusal is WORSE than the bug it fixes --
   #721 was a real OOM, and a gate that cannot refuse is a gate in name only.
   Both directions are asserted, and both are proved able to fail.
3. an unestablishable number degrades to NO GUARD rather than to a refusal,
   which is the rule the owner module states: refusing a boot on a fabricated
   figure is worse than not checking.
4. the refusal names the scope it measured.

Hermetic: pure functions and injected probes. No CUDA, no boot, no card.
"""

import inspect
import unittest

from sglang.srt.turnkey import preflight as PF
from sglang.srt.turnkey.refusal import REFUSE_HOST_HEADROOM
from sglang.test.test_utils import CustomTestCase

GIB = 1 << 30


def _cfg(host_headroom_gib: int):
    """A StackConfig whose only interesting field is the headroom want."""

    class _P:
        pass

    p = _P()
    p.host_headroom_gib = host_headroom_gib

    class _C:
        pass

    c = _C()
    c.preflight = p
    return c


def _probes(mem_available_bytes):
    class _Pr:
        pass

    pr = _Pr()
    pr.mem_available_bytes = mem_available_bytes
    return pr


class TestPreflightHonestHostRam(CustomTestCase):
    # -- 1. no second reimplementation ---------------------------------------

    def test_the_default_probe_reaches_the_407_owner(self):
        """RED before: `_real_mem_available` opened /proc/meminfo itself."""
        src = inspect.getsource(PF)
        self.assertIn(
            "host_memory_bytes_for_pinning",
            src,
            "the boot gate still reads host RAM from its own reimplementation "
            "instead of the #407 owner, so it inherits neither the lxcfs "
            "correction nor the cgroup clamp",
        )
        # Assert on the CODE, not the prose. The docstring legitimately names
        # `/proc/meminfo` several times to record why reading it here was
        # wrong, and an assertion blunt enough to catch that explanation would
        # have to be deleted the first time anyone documented the fix -- the
        # same mistake this suite's sibling (#871a) already made once.
        probe_src = inspect.getsource(PF._real_mem_available)
        body = probe_src.replace(PF._real_mem_available.__doc__ or "", "")
        self.assertNotIn(
            "open(",
            body,
            "the probe still opens a file itself instead of asking the owner",
        )
        self.assertIn("host_memory_bytes_for_pinning", body)

    # -- 2. THE DANGER DIRECTION: it must still refuse ------------------------

    def test_it_STILL_REFUSES_when_host_ram_is_genuinely_short(self):
        """The half that must never regress.

        #721 was a real container OOM. A fix that only removes over-reporting
        and quietly loses the refusal turns a gate into decoration.
        """
        r = PF.check_host_headroom(_cfg(15), _probes(lambda: 1 * GIB))
        self.assertIsNotNone(r, "the gate did not refuse on 1 GiB against 15")
        self.assertEqual(r.name, REFUSE_HOST_HEADROOM)

    def test_it_admits_when_there_is_genuinely_enough(self):
        self.assertIsNone(PF.check_host_headroom(_cfg(15), _probes(lambda: 100 * GIB)))

    def test_a_want_of_zero_disables_the_gate(self):
        self.assertIsNone(PF.check_host_headroom(_cfg(0), _probes(lambda: 1)))

    # -- 3. unknown is NOT a refusal -----------------------------------------

    def test_an_unestablishable_number_stands_the_guard_DOWN(self):
        """The owner's own rule, applied here.

        "A caller that cannot get a number must say so, not invent one" -- and
        refusing a boot on a fabricated figure is worse than not checking. The
        sentinel must not be read as "0 GiB available", which would refuse
        every boot on a box whose cgroup files are absent.
        """
        r = PF.check_host_headroom(_cfg(15), _probes(lambda: PF.MEM_AVAILABLE_UNKNOWN))
        self.assertIsNone(
            r,
            "an unreadable host-RAM figure was treated as a refusal; on a box "
            "without cgroup files that refuses every boot for no reason",
        )

    def test_the_unknown_sentinel_is_negative_so_it_cannot_pass_as_a_size(self):
        self.assertLess(PF.MEM_AVAILABLE_UNKNOWN, 0)

    # -- 4. the scope is in the message --------------------------------------

    def test_the_refusal_names_the_scope_it_measured(self):
        """Three cgroup levels give three answers; say which one was asked."""
        r = PF.check_host_headroom(_cfg(15), _probes(lambda: 1 * GIB))
        self.assertIsNotNone(r)
        blob = f"{r.subject} {r.observed} {r.expected} {r.remedy}"
        self.assertIn(
            "cgroup",
            blob,
            "the refusal reports a bare GiB figure without saying which scope "
            f"it measured: {blob!r}",
        )

    # -- 5. the boot-gated unknown is NAMED, not guessed ---------------------

    def test_the_file_bucket_unknown_is_recorded_in_the_code(self):
        """#534's question is open and must be visible where it bites.

        CUDA pinned host memory is accounted under the cgroup's `file` bucket,
        which the owner deliberately never charges. Whether that makes this
        gate over-report AGAIN, on top of what is fixed here, is decidable only
        by watching anon/file across a real pin allocation. Guessing it would
        be the capping-on-a-fabricated-figure this whole class is about, so it
        is carried as a named unknown instead.
        """
        src = inspect.getsource(PF)
        self.assertIn("#534", src)
        self.assertIn("file", src)


class TestFeasibilityHonestHostRam(CustomTestCase):
    """The second copy of the same reimplementation, closed in the same pass."""

    def test_feasibility_reaches_the_407_owner_too(self):
        """Advisory, not a gate -- but a class with a known second instance is
        not left pending. That is how the first one survived."""
        from sglang.srt.training import feasibility as FE

        src = inspect.getsource(FE)
        self.assertIn(
            "host_memory_bytes_for_pinning",
            src,
            "training/feasibility.py still carries its own /proc/meminfo "
            "reader -- the second instance of the class fixed above",
        )


if __name__ == "__main__":
    unittest.main()
