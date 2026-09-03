"""#1175 (E2/E3/E4a) -- the follower's wait drives the writer it used to
suspend, states its own shortage before it can be stuck, and stops being
silent on the healthy path.

THE SPECIMEN (boot_855_weg1b5, rid 0c34259f, log 113750-113834). PP1 received
the forwarded row naming prefix_len=12288, entered `execute_scheduled_prefix`
holding 0, and polled `match_prefix` 256 times over 14 s while `local` stayed
0. It could not have been otherwise: the prefetched span reaches the radix
tree at exactly one site (`UnifiedRadixCache._insert_helper_host`, reached
only from `check_prefetch_progress`), whose only callers run on the SAME
scheduler thread that was sleeping in this loop. The wait polled a structure
whose sole writer it had itself suspended, so the bound could only expire.

The old comment justified never calling it with the #580 collective-symmetry
hazard -- a statement about a DANGER, not about the mechanism. The danger is
answered by asking whether the call carries a collective at all
(`prefetch_progress_is_collective_free`, world 1 = every all_reduce skipped by
construction), not by never calling it.

WHAT THIS PINS:

 (a) the poll DRIVES `check_prefetch_progress` when the tree reports the path
     collective-free, and does NOT when it does not;
 (b) the expiry text names `prefetch_driven_in_loop=` so an expiry can never
     be read as "the bytes were not there" when nothing drove the writer
     (INDIKATOR-GESETZ);
 (c) E3: EVERY follower prints one bounded per-rid under-coverage line at
     ENTRY, before any bound can expire -- PP2 printed not one line about
     this rid and its silence was indistinguishable from health;
 (d) E4a: `local == scheduled` no longer returns silently, so "0
     materialised lines on the whole boot" is readable as "never needed"
     rather than "never reached".

MUTANTS (each red): drop the drive call; hard-code the collective-free
predicate to True; remove the entry line; restore the silent `return 0`.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

# The function under test imports these lazily, INSIDE the timed region. A
# cold first import costs seconds and would expire the length-priced bound
# before the loop ever polled -- warm them here so the test measures the loop
# and not the interpreter (measured: 4.28 s of a 0.20 s bound, 0 polls).
import torch  # noqa: F401,E402

from sglang.srt.managers import pp_admission_congruence as pac
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams  # noqa: F401,E402


class _Tree:
    def __init__(self, collective_free=True):
        self._free = collective_free
        self.progress_calls = []

    def prefetch_progress_is_collective_free(self):
        return self._free

    def check_prefetch_progress(self, rid):
        self.progress_calls.append(rid)


class _Req(SimpleNamespace):
    def init_next_round_input(self, tree_cache=None):
        """The real Req re-derives its match from the tree each poll. This
        stand-in holds still: the prefix only grows when the (absent) load-back
        grows it, which is exactly the specimen's condition."""


def _req(rid="0c34259f", local=0):
    return _Req(
        rid=rid,
        prefix_indices=list(range(local)),
        cache_protected_len=local,
        host_hit_length=0,
        best_match_node=None,
        mamba_loadback_anchor_adopted=False,
    )


class _ShortBound:
    """Price the wait at ~0.2 s so the expiry arm costs a fifth of a second."""

    def __enter__(self):
        self._p = [
            mock.patch.object(pac, "MATERIALISE_BASE_S", 0.2),
            mock.patch.object(pac, "MATERIALISE_S_PER_KI_TOKEN", 0.0),
            mock.patch.object(pac, "MATERIALISE_MAX_S", 0.2),
        ]
        for p in self._p:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._p:
            p.stop()
        return False


class TestTheWaitDrivesTheWriterItUsedToSuspend(unittest.TestCase):
    def test_it_drives_progress_when_the_path_is_collective_free(self):
        tree = _Tree(collective_free=True)
        with _ShortBound(), self.assertRaises(RuntimeError) as ctx:
            pac.execute_scheduled_prefix(_req(), tree, 12288)
        self.assertGreater(
            len(tree.progress_calls),
            0,
            "the poll must drive the sole writer of the host tier",
        )
        self.assertIn("prefetch_driven_in_loop=True", str(ctx.exception))

    def test_it_does_not_drive_when_the_path_carries_a_collective(self):
        tree = _Tree(collective_free=False)
        with _ShortBound(), self.assertRaises(RuntimeError) as ctx:
            pac.execute_scheduled_prefix(_req(), tree, 12288)
        self.assertEqual(tree.progress_calls, [], "a peer's collective is not ours")
        self.assertIn("prefetch_driven_in_loop=False", str(ctx.exception))

    def test_the_expiry_still_names_the_group_stop_and_the_bound(self):
        tree = _Tree()
        with _ShortBound(), self.assertRaises(RuntimeError) as ctx:
            pac.execute_scheduled_prefix(_req(), tree, 12288)
        text = str(ctx.exception)
        self.assertIn("PREFIX MATERIALISATION SHORTFALL", text)
        self.assertIn("RAENGE-NIE-UNEINS", text)
        self.assertIn("length-priced bound", text)


class TestEveryFollowerSpeaksBeforeItCanBeStuck(unittest.TestCase):
    def test_an_under_covered_entry_prints_a_bounded_per_rid_line(self):
        pac._PREFIX_EXEC_SHORT_SEEN = 0
        tree = _Tree()
        with _ShortBound(), self.assertLogs(
            "sglang.srt.managers.pp_admission_congruence", level=logging.INFO
        ) as cap:
            with self.assertRaises(RuntimeError):
                pac.execute_scheduled_prefix(_req(), tree, 12288)
        joined = "\n".join(cap.output)
        self.assertIn("#1175 PREFIX-EXEC UNDER-COVERAGE", joined)
        self.assertIn("rid=0c34259f", joined)
        self.assertIn("deficit=12288", joined)
        self.assertIn("scheduled=12288", joined)

    def test_the_line_is_bounded_not_a_stream(self):
        pac._PREFIX_EXEC_SHORT_SEEN = 0
        tree = _Tree()
        emitted = 0
        with _ShortBound():
            for _ in range(24):
                with self.assertLogs(
                    "sglang.srt.managers.pp_admission_congruence", level=logging.INFO
                ) as cap:
                    # assertLogs needs at least one record; past the rate limit
                    # this call emits none, which is the point.
                    pac.logger.info("probe")
                    with self.assertRaises(RuntimeError):
                        pac.execute_scheduled_prefix(_req(), tree, 4096)
                emitted += sum(
                    1 for line in cap.output if "PREFIX-EXEC UNDER-COVERAGE" in line
                )
        self.assertEqual(
            emitted, 20, "first 20 then rate-limited: a denominator, not a stream"
        )


class TestTheHealthyPathIsNoLongerSilent(unittest.TestCase):
    def test_local_equals_scheduled_prints_a_bounded_no_op_line(self):
        pac._PREFIX_EXEC_NOOP_SEEN = 0
        with self.assertLogs(
            "sglang.srt.managers.pp_admission_congruence", level=logging.INFO
        ) as cap:
            moved = pac.execute_scheduled_prefix(_req(local=4096), _Tree(), 4096)
        self.assertEqual(moved, 0)
        joined = "\n".join(cap.output)
        self.assertIn("#968 PREFIX-EXEC no-op", joined)
        self.assertIn("local=scheduled=4096", joined)

    def test_the_no_op_line_is_rate_limited(self):
        pac._PREFIX_EXEC_NOOP_SEEN = 0
        seen = 0
        for _ in range(9):
            with self.assertLogs(
                "sglang.srt.managers.pp_admission_congruence", level=logging.INFO
            ) as cap:
                # assertLogs needs at least one record; the warning below is
                # unrelated and only keeps the context manager satisfied.
                pac.logger.info("probe")
                pac.execute_scheduled_prefix(_req(local=8), _Tree(), 8)
            seen += sum(1 for line in cap.output if "PREFIX-EXEC no-op" in line)
        self.assertEqual(seen, 5)


class TestTruncationIsStillTheSafeDirection(unittest.TestCase):
    def test_local_above_scheduled_truncates_and_says_so(self):
        req = _req(local=8192)
        with self.assertLogs(
            "sglang.srt.managers.pp_admission_congruence", level=logging.INFO
        ) as cap:
            moved = pac.execute_scheduled_prefix(req, _Tree(), 4096)
        self.assertEqual(moved, 0)
        self.assertEqual(len(req.prefix_indices), 4096)
        self.assertEqual(req.cache_protected_len, 4096)
        self.assertIn("PREFIX-EXEC truncate", "\n".join(cap.output))


if __name__ == "__main__":
    unittest.main()
