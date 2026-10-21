"""NF-RC31 D (rc30 review, Folgefix D): an H105g refusal carries its own D-ADMIT-REFUSED gate name.

Before: a refusal because the load-back source left L2 (H105g / S109R ``anchor_dropped``) was stamped
``h105d_cut_room`` / ``h105f_confirm_room`` -- the names of "no room". Monitor class 2 (``nf_scharf.py``,
``H105g LOAD-BACK TARGET DROPPED`` > 2x per rid) had to pair lines to tell them apart. Now:
``h105g_target_dropped``. Instrument only: the verdicts and gathers are unchanged (checked on the real
``add_one_req`` group harness of the H105g test).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from flliper.srt.pdflip import d_admit_refusal as DAR  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


G = _load("_h105g_group_for_rc31_d", _HERE / "test_nf_rc29_loadback_target_dropped_h105g_1021.py")


def _gate(adder):
    st = getattr(adder, DAR.STAMP_ATTR, None)
    return st[1] if st else None


class H105gGateNameTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sp._H105D_REFUSED["n"] = 0
        sp._H105G_DROPPED["n"] = 0

    def test_the_dropped_source_is_named_h105g_target_dropped_on_the_rank_that_saw_it(self):
        out, adders, _reqs, gather = G._group(drop_on=(0,))
        self.assertEqual(set(out.values()), {AddReqResult.NO_TOKEN}, out)
        self.assertEqual(gather.calls, {0: 2, 1: 2, 2: 2})              # the gathers as before
        self.assertEqual(_gate(adders[0]), "h105g_target_dropped")
        self.assertEqual({_gate(adders[1]), _gate(adders[2])}, {"h105_group_verdict"})

    def test_the_anchor_dropped_form_is_named_too(self):
        out, adders, _gather = G._group_anchor()
        self.assertEqual(set(out.values()), {AddReqResult.NO_TOKEN}, out)
        self.assertEqual(_gate(adders[0]), "h105g_target_dropped")

    def test_no_room_keeps_its_room_name(self):
        adder = SimpleNamespace()
        req = SimpleNamespace(rid="r1")
        self.assertEqual(sp._h105_room_gate_name(adder, req, "h105d_cut_room"), "h105d_cut_room")
        setattr(adder, sp.H105G_REFUSED_ATTR, ("r0", "detached"))      # an earlier rid's record never leaks
        self.assertEqual(sp._h105_room_gate_name(adder, req, "h105f_confirm_room"), "h105f_confirm_room")
        setattr(adder, sp.H105G_REFUSED_ATTR, ("r1", "anchor_dropped"))
        self.assertEqual(sp._h105_room_gate_name(adder, req, "h105f_confirm_room"), "h105g_target_dropped")

    def test_a_room_call_clears_the_last_record(self):
        """A later call for the same rid that refuses for want of room is not named H105g by a stale record."""
        adder = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 0),
                                tree_cache=SimpleNamespace(evictable_size=lambda: 0), rem_chunk_tokens=64,
                                page_size=1, can_run_list=[], rem_total_tokens=0)
        req = SimpleNamespace(rid="r1", full_untruncated_fill_ids=[0] * 10, prefix_indices=[],
                              best_match_node=None)
        setattr(adder, sp.H105G_REFUSED_ATTR, ("r1", "detached"))
        with unittest.mock.patch(G.H.TOKEN_CUT, return_value=True), \
                unittest.mock.patch.object(sp, "_pp_load_back_extent", lambda q: None), \
                unittest.mock.patch.object(sp, "payable_evictable_or", lambda tc, fn: 0):
            self.assertFalse(sp._h105d_cut_load_back_room(adder, req, evict=True))
        self.assertEqual(sp._h105_room_gate_name(adder, req, "h105d_cut_room"), "h105d_cut_room")


if __name__ == "__main__":
    unittest.main()
