# SPDX-License-Identifier: Apache-2.0
"""27B rc12z30j (dkr27browauthoritybar1w109290020, bb82fbcb68), P->D flip epoch=120, 14283 ms.

front.log:7621 PDFLIP-FLIP-TIMELINE epoch=120 ... sleep-kv@12228 ... done@14283. P.log 00:44:12-24:
PP0 "Cache not flushed ... #1268 group verdict: GROUP VERDICT PENDING", then "#1460 CTRL-FWD
kinds=['FlushCacheReqInput']"; PP1/PP2 "Cache flushed successfully!" + "PDFLIP-ANCHOR-LOST at=flush
n=2 depths=[16383, 20990]"; PP0 "#1466 PASS-STALL pass_ms=12035", "PP-RECV-OBJ site=pdflip/idle-vote-home
waited=12.1s expiries=3"; PP1/PP2 "PP-CHAIN-RECV blocked 11931 ms".

The followers decided the forwarded flush on their own idleness while PP0 refused it. Fix
(pdflip/flush_verdict.py): PP0 stamps the flush on the wire, followers park it and flush only on PP0's
passed verdict, which rides the next pass at the front of the wire.

The ring below is the metal form: PP3 group P, the front polling /flush_cache, PP0 pending for three
polls (no idle-vote lap landed), then passing. On bb82fbcb68 (no flush_verdict module) the same ring
runs with the old wrapper and the followers flush on every pending poll -> red.
"""

from __future__ import annotations

import inspect
import logging
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.io_struct import FlushCacheReqInput  # noqa: E402
from flliper.srt.managers.scheduler_components import flush_wrapper as FW  # noqa: E402

try:
    from flliper.srt.pdflip import flush_verdict as FV  # noqa: E402
except ImportError:  # the pre-fix tree: the ring runs with the old wrapper (red)
    FV = None


class _Chan:
    def __init__(self):
        self.sent = []
        self.send_to_tokenizer = self

    def send_output(self, out, req):
        self.sent.append((out, req))


class _Release:
    """Stands in for the sleep leg's ReleaseMemoryOccupationReqInput on the wire."""


class _Rank:
    """One PP rank: the real SchedulerFlushWrapper over a flush_cache with the
    metal verdicts (PP0: the #1268 group verdict; follower: rank-local idle)."""

    def __init__(self, pp_rank, pp_size=3):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
        self.pp0_pending = True  # PP0: no idle-vote lap landed yet
        self.idle = True
        self.flushes = 0
        self.anchor_lost = 0
        self.dispatched = []
        self.chan = _Chan()
        kw = {}
        if FV is not None:
            kw = dict(
                park_forwarded=lambda req: FV.follower_park(self, req),
                on_decided=lambda req, ok, detail: FV.pp0_record(self, req, ok, detail),
            )
        self.wrapper = FW.SchedulerFlushWrapper(
            flush_cache=self._flush_cache, is_fully_idle=lambda: self.idle,
            ipc_channels=self.chan, is_dormant=lambda: False, **kw)

    def _flush_cache(self, **_kw):
        group_idle = (not self.pp0_pending) if self.ps.pp_rank == 0 else self.idle
        if not group_idle:
            return False
        self.flushes += 1
        if self.ps.pp_rank != 0:
            self.anchor_lost += 2  # PDFLIP-ANCHOR-LOST at=flush n=2 depths=[16383, 20990]
        return True

    def dispatch(self, reqs):
        for r in reqs:
            if isinstance(r, FlushCacheReqInput):
                out = self.wrapper.handle(r)
                if out is not None:
                    self.chan.send_output(out, r)
            else:
                self.dispatched.append(r)


def _pass(ranks, tokenizer_reqs):
    """One pass of _pp_forward_and_process_input_requests down the chain:
    forward first (PP0 stamps + prefixes verdicts), then absorb, then dispatch."""
    recv = list(tokenizer_reqs)
    wires = []
    for k, rank in enumerate(ranks):
        if k == 0:
            wire = FV.pp0_wire(rank, recv) if FV is not None else recv
        else:
            wire = recv
        wires.append(list(wire))
        mine = recv
        if k != 0 and FV is not None:
            mine = FV.follower_absorb(rank, recv, rank.wrapper.apply_pp0_verdict)
        rank.dispatch(mine)
        recv = wire
    return wires


class Epoch120FollowersFollowPP0(unittest.TestCase):
    def _ring(self):
        return [_Rank(0), _Rank(1), _Rank(2)]

    def test_pp0_pending_followers_do_not_flush_no_anchor_lost(self):
        ranks = self._ring()
        for _poll in range(3):  # the front's quiesce polls while PP0 answers pending
            _pass(ranks, [FlushCacheReqInput()])
        _pass(ranks, [])  # PP0's refusals ride this pass
        self.assertEqual(ranks[0].flushes, 0)
        for r in ranks[1:]:
            self.assertEqual(r.flushes, 0, f"PP{r.ps.pp_rank} flushed on its own verdict")
            self.assertEqual(r.anchor_lost, 0, f"PP{r.ps.pp_rank} PDFLIP-ANCHOR-LOST at=flush")

    def test_pp0_passes_then_every_follower_flushes_once(self):
        ranks = self._ring()
        for _poll in range(3):
            _pass(ranks, [FlushCacheReqInput()])
        ranks[0].pp0_pending = False  # the lap landed idle
        _pass(ranks, [FlushCacheReqInput()])
        self.assertEqual(ranks[0].flushes, 1)
        self.assertEqual([r.flushes for r in ranks[1:]], [0, 0], "followers wait for the verdict")
        wires = _pass(ranks, [_Release()])  # the sleep leg's release follows PP0's 200
        self.assertEqual([r.flushes for r in ranks], [1, 1, 1])
        kinds = [type(x).__name__ for x in wires[0]]
        self.assertLess(kinds.index("PdFlipFlushVerdict"), kinds.index("_Release"),
                        "the verdict must reach the followers before the release")
        for r in ranks[1:]:
            self.assertEqual(len(r.dispatched), 1)  # only the release; verdicts never dispatch

    def test_follower_busy_under_pp0_pass_is_named_uneins(self):
        ranks = self._ring()
        ranks[0].pp0_pending = False
        ranks[2].idle = False
        _pass(ranks, [FlushCacheReqInput()])
        with self.assertLogs(level=logging.WARNING) as cm:
            _pass(ranks, [])
        self.assertEqual([r.flushes for r in ranks], [1, 1, 0])
        self.assertTrue(any("UNEINS" in m for m in cm.output))


@unittest.skipIf(FV is None, "pre-fix tree")
class OldFormAndOffPPAreUnchanged(unittest.TestCase):
    def test_kill_switch_restores_the_old_form(self):
        with mock.patch.dict(os.environ, {FV.ENV: "0"}):
            ranks = [_Rank(0), _Rank(1), _Rank(2)]
            _pass(ranks, [FlushCacheReqInput()])
        self.assertEqual([r.flushes for r in ranks], [0, 1, 1])  # bb82fbcb68's behaviour

    def test_unstamped_flush_on_a_follower_takes_the_old_path(self):
        r = _Rank(1)
        out = r.wrapper.handle(FlushCacheReqInput())
        self.assertTrue(out.success)
        self.assertEqual(r.flushes, 1)

    def test_off_pp_nothing_is_stamped(self):
        r = _Rank(0, pp_size=1)
        req = FlushCacheReqInput()
        self.assertEqual(FV.pp0_wire(r, [req]), [req])
        self.assertIsNone(req.pdflip_flush_seq)
        r.pp0_pending = False
        self.assertTrue(r.wrapper.handle(req).success)
        self.assertEqual(FV._state(r)["out"], [])

    def test_pp_forward_wires_both_halves(self):
        from flliper.srt.managers import scheduler_pp_mixin as M

        src = inspect.getsource(M.SchedulerPPMixin._pp_forward_and_process_input_requests)
        self.assertIn("_flush_verdict.pp0_wire(self, _wire_reqs)", src)
        self.assertIn("_flush_verdict.follower_absorb(", src)
        self.assertLess(src.index("pp0_wire"), src.index("_pp_send_pyobj_to_next_stage"))


if __name__ == "__main__":
    unittest.main()
