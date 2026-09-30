"""HP1 (cb7f2cdc35) off Form A: the #580 vote is byte-identical for 27B.

27B's D group runs the #580 participation vote under uneven DCP (the host
tiers diverge by construction), with NO Form A role plan installed. HP1 threads
``span_base`` from the one group-decides call site into
``prefetch_from_storage`` and packs the completion MIN as ENDS. Off Form A
``form_a_end_base`` must answer None, so the vote is the unchanged length
vote: same collectives, same payloads, same reduced values, same registration
with or without ``span_base``. Spans of DIFFERENT starts (the case the END
vote would answer on Form A) stay 27B's named W65 stop, message for message.

Driven through the REAL ``UnifiedRadixCache.prefetch_from_storage`` on three
simulated ranks (the H99 harness), in the 27B form: uneven DCP on, no role
plan, no Form A worker. An identity pin: green on cb7f2cdc35 and on its base.
The END pack at base 0 is the identity on the completion slots as well.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import test_nf_form_a_prefetch_span_h99 as h99

from flliper.srt import rank_role
from flliper.srt.mem_cache import unified_radix_cache as urc

END = 33600
PROMPT = list(range(END))
# Server args of the 27B D group: even rank-TP plan, uneven DCP decides
# the asymmetry.
B27_D = types.SimpleNamespace(rank_tp_ratio=None, hicache_size=4)


class _RecordingGroup(h99.MockGlooGroup):
    """The H99 group, keeping every rank's payload and the reduced value."""

    def __init__(self):
        super().__init__()
        self.payloads = {r: [] for r in range(h99.WORLD)}

    def all_reduce(self, rank, tensor, op, label):
        sent = tensor.clone().tolist()
        super().all_reduce(rank, tensor, op, label)
        self.payloads[rank].append((label, str(op), sent, tensor.clone().tolist()))


class _Env27B:
    """27B D form: uneven DCP, NO Form A plan, no worker role on any rank."""

    def __init__(self):
        self.stack = []

    def __enter__(self):
        self.prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = None, None
        for p in (
            mock.patch("flliper.srt.runtime_context.get_server_args", return_value=B27_D),
            mock.patch.object(urc, "uneven_dcp_active", return_value=True),
            mock.patch.object(rank_role, "this_rank_is_form_a_worker", lambda: False),
        ):
            p.__enter__()
            self.stack.append(p)
        return self

    def __exit__(self, *a):
        for p in reversed(self.stack):
            p.__exit__(*a)
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = self.prev


def _intake(bases, rid, *, span_base):
    group = _RecordingGroup()
    caches = {}

    def _rank(r):
        caches[r] = c = h99._carrier(r, group)
        kw = {"span_base": bases[r]} if span_base else {}
        c.prefetch_from_storage(
            rid, h99._host_node(), PROMPT[bases[r]:], last_hash=None, prefix_keys=None, **kw
        )
        return h99._registered_len(c, rid)

    with _Env27B():
        results, errors = h99.run_ranks(_rank)
    return caches, results, errors, group


# Same end, different starts -- where HP1's END vote answers differently
# from the length vote on Form A (host 33600 vs workers 896). Off Form A
# differing spans are 27B's named W65 stop, with or without span_base.
SKEWED = {0: 0, 1: 32704, 2: 32704}
EVEN = {0: 4096, 1: 4096, 2: 4096}


def _named(errors):
    return {r: (type(e).__name__, str(e)) for r, e in errors.items()}


class Hp1OffFormAIsTheLengthVote(unittest.TestCase):
    def test_span_base_changes_no_collective_and_no_registration(self):
        c0, r0, e0, g0 = _intake(EVEN, "pdflip-27b-1", span_base=False)
        c1, r1, e1, g1 = _intake(EVEN, "pdflip-27b-1", span_base=True)
        self.assertEqual((e0, e1, g0.errors, g1.errors), ({}, {}, [], []))
        self.assertEqual(r0, {r: (29504, 29504) for r in range(h99.WORLD)})
        self.assertEqual(r1, r0, "span_base moved a 27B registration")
        self.assertEqual(
            g1.payloads, g0.payloads,
            "span_base changed a 27B vote payload or its reduced value",
        )
        self.assertEqual(g1.log, g0.log)
        self.assertEqual(
            {len(p[2]) for p in (g1.payloads[r][0] for r in g1.payloads)}, {5},
            "the #580 payload off Form A is the five-slot vote",
        )
        for r in c1:
            self.assertIsNone(
                getattr(c1[r], "_hp1_end_base_by_rid", {}).get("pdflip-27b-1"),
                "an END base was recorded off Form A",
            )

    def test_skewed_spans_stay_the_named_w65_stop(self):
        _c0, r0, e0, g0 = _intake(SKEWED, "pdflip-27b-2", span_base=False)
        _c1, r1, e1, g1 = _intake(SKEWED, "pdflip-27b-2", span_base=True)
        self.assertEqual(set(e0), set(range(h99.WORLD)), "27B's W65 must stop every rank")
        for _r, (name, msg) in _named(e0).items():
            self.assertEqual(name, "HiCacheCollectiveDesyncError")
            self.assertIn("W65 PdFlipPrefetchSpanSplit", msg)
            self.assertIn("min=896 max=33600", msg)
        self.assertEqual(_named(e1), _named(e0), "span_base changed 27B's W65 stop")
        self.assertEqual(g1.payloads, g0.payloads)
        self.assertEqual((r0, r1), ({}, {}))

    def test_end_pack_at_base_zero_is_the_identity(self):
        for completed, hit, anchor in (
            (0, 0, 0), (896, 512, 256), (33600, 33600, urc._ANCHOR_ABSTAIN),
        ):
            with self.subTest(completed=completed, hit=hit, anchor=anchor):
                self.assertEqual(
                    urc._hp1_end_pack(0, completed, hit, anchor), (completed, hit, anchor)
                )
                for v in (completed, hit, anchor):
                    self.assertEqual(urc._hp1_end_unpack(0, v), v)


if __name__ == "__main__":
    unittest.main()
