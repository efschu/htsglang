"""#239 S3d: the seams that assumed "a Form A worker holds 0 KV".

Under Form A x the token cut (form ``kv=qsa_forma_dcp``) a worker OWNS a token
range of the full-attention KV. Every place that read "worker" as "byteless"
must now ask whether this rank holds rows:

* ``rank_role.form_a_token_cut_active`` / ``form_a_worker_holds_kv`` -- the one
  group-uniform predicate (installed plan + installed token vector);
* H98/H97 (tp_match_floor): a worker's usable vote is its REAL KV reach, so it
  can take the group below the host -- a named line, never a silent floor;
* #59 (weg2_resumable_depth): the depth is the group MIN (host admission,
  worker KV reach), not the host's alone;
* H105: the admission gate is the group MIN (a gather in place of the
  host's broadcast);
* S3e: the launch refusal of the cut reads the seam register (F14 -- the
  worker's bytes in the host tier and store -- is still open).

RED on cdd6522981: the predicate does not exist, a Form A worker names no
depth and enters no reduce, and a worker-lowered floor is silent.
"""

from __future__ import annotations

import contextlib
import logging
import types
import unittest
from array import array
from unittest import mock

import torch

from sglang.srt import rank_role
from sglang.srt.mem_cache.base_prefix_cache import MatchResult

ROLES = ("host", "worker", "worker")
#: gcd-reduced token cut [0, 46, 18] (S2b dry run): prefix 0, 0, 46, 64
BOUNDS = {0: (64, 0, 0), 1: (64, 0, 46), 2: (64, 46, 64)}
GRID = 25600


@contextlib.contextmanager
def _as_rank(rank, *, cut=True, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    bounds = BOUNDS[rank] if cut else None
    try:
        with mock.patch(
            "sglang.srt.distributed.utils.uneven_dcp_owner_bounds", lambda: bounds
        ):
            yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


class TestPredicate(unittest.TestCase):
    def test_the_cut_is_group_uniform(self):
        for r in (0, 1, 2):
            with _as_rank(r):
                self.assertTrue(rank_role.form_a_token_cut_active())
            with _as_rank(r, cut=False):
                self.assertFalse(rank_role.form_a_token_cut_active())

    def test_a_worker_holds_kv_only_with_a_share(self):
        with _as_rank(0):
            self.assertFalse(rank_role.form_a_worker_holds_kv())  # the host
        with _as_rank(1):
            self.assertTrue(rank_role.form_a_worker_holds_kv())
        with _as_rank(2):
            self.assertTrue(rank_role.form_a_worker_holds_kv())
        with _as_rank(1, cut=False):
            self.assertFalse(rank_role.form_a_worker_holds_kv())
        # a worker with share 0 under the cut ([32, 0, 32]) stays byteless
        prev = BOUNDS[1]
        BOUNDS[1] = (64, 32, 32)
        try:
            with _as_rank(1):
                self.assertTrue(rank_role.form_a_token_cut_active())
                self.assertFalse(rank_role.form_a_worker_holds_kv())
        finally:
            BOUNDS[1] = prev

    def test_classic_boot_is_untouched(self):
        with _as_rank(0, roles=None):
            self.assertFalse(rank_role.form_a_token_cut_active())
            self.assertFalse(rank_role.form_a_worker_holds_kv())


class TestWorkerFloorIsNamed(unittest.TestCase):
    def _skew(self, rank, cut):
        from sglang.srt.managers import tp_match_floor as f

        with _as_rank(rank, cut=cut), self.assertLogs(f.logger, logging.WARNING) as logs:
            f.logger.warning("sentinel")
            out = f.skewed_rids({"weg2-a": 16384, "weg2-b": 8192}, {"weg2-a": 25600, "weg2-b": 8192})
        return out, [line for line in logs.output if "WORKER-FLOOR" in line]

    def test_a_worker_reach_below_the_host_is_a_named_line(self):
        """RED ON cdd6522981: the skew is acted on (H97) but never named."""
        for rank in (0, 1, 2):
            out, lines = self._skew(rank, cut=True)
            self.assertEqual(out, {"weg2-a": 16384})
            self.assertEqual(len(lines), 1)
            self.assertIn("RU FORM-A-DCP WORKER-FLOOR rid=weg2-a host=25600 group=16384", lines[0])

    def test_no_line_without_the_cut(self):
        out, lines = self._skew(0, cut=False)
        self.assertEqual(out, {"weg2-a": 16384})
        self.assertEqual(lines, [])


def _node(name, anchor=True):
    host = torch.tensor([3]) if anchor else None
    return types.SimpleNamespace(
        name=name,
        component_data=[None, None, types.SimpleNamespace(value=None, host_value=host)],
    )


class _Tree:
    """A finished request's tree on one rank: KV to ``reach``, the recurrent
    anchor there only where the rank holds GDN state (the host)."""

    def __init__(self, reach, anchor):
        self.reach, self.anchor = reach, anchor
        self.root_node = _node("root", anchor=False)
        self.cache_controller = None
        self.is_eagle = False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def match_prefix(self, params):
        d = min(self.reach, len(params.key))
        node = _node(f"n{d}", anchor=self.anchor)
        return MatchResult(
            device_indices=torch.arange(d, dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )


def _req(rid="weg2-59-dcp"):
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(GRID + 11)),
        output_ids=list(range(300)),
        extra_key=None,
        positional_embed_overrides=None,
        cached_tokens=GRID,
        time_stats=SchedulerReqTimeStats(),
        finished=lambda: True,
        finished_output=False,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


def _ps(rank):
    return types.SimpleNamespace(tp_size=3, attn_dp_size=3, pp_size=1, attn_tp_rank=rank)


class TestResumableDepthIsTheGroupMin(unittest.TestCase):
    def test_every_rank_votes_and_the_group_takes_the_min(self):
        """RED ON cdd6522981: under Form A the host alone names the depth and a
        worker enters no reduce -- a worker that lost its rows of the prefix
        would leave the front pricing a resume the group cannot serve."""
        from sglang.srt.managers import weg2_resumable_depth as m

        votes = {}
        trees = {0: _Tree(GRID, anchor=True), 1: _Tree(16384, anchor=False), 2: _Tree(GRID, anchor=False)}
        for rank in (0, 1, 2):
            with _as_rank(rank):
                self.assertEqual(m.group_mode(_ps(rank)), m.MODE_DCP_MIN)
                _, depths = m.group_depths(
                    trees[rank], [_req()], _ps(rank),
                    reduce_min=lambda v, r=rank: votes.setdefault(r, list(v)) and list(v),
                )
        # the host votes its admission (anchor rule), a worker its KV reach
        # (no anchor to test -- without the follow walk it would vote 0)
        self.assertEqual(votes, {0: [GRID], 1: [16384], 2: [GRID]})
        group = [min(v[0] for v in votes.values())]
        reqs = [_req()]
        with _as_rank(0):
            mode = m.stamp_finished(trees[0], reqs, _ps(0), reduce_min=lambda v: group)
        self.assertEqual(mode, m.MODE_DCP_MIN)
        self.assertEqual(reqs[0].time_stats.weg2_resumable_depth, 16384)

    def test_without_the_cut_the_host_still_decides(self):
        from sglang.srt.managers import weg2_resumable_depth as m

        with _as_rank(0, cut=False):
            self.assertEqual(m.group_mode(_ps(0)), m.MODE_HOST)
        with _as_rank(1, cut=False):
            self.assertEqual(m.group_mode(_ps(1)), m.MODE_FOLLOW)


class TestAdmissionVerdictIsTheGroupMin(unittest.TestCase):
    """H105 under the token cut: a worker's pool gate is real."""

    def _verdict(self, codes, rank, rid="weg2-h105"):
        from sglang.srt.managers import tp_match_floor as f

        tuples = [(rid, c, 100 + i, 1000) for i, c in enumerate(codes)]
        return f.form_a_admission_verdict(
            rid, codes[rank], is_host=rank == 0, exchange=None,
            price=100 + rank, budget=1000, gather=lambda payload: list(tuples),
        )

    def test_a_refusing_worker_refuses_for_the_group(self):
        """RED ON cdd6522981: the host's ADMIT was the group's even when a
        worker's own KV pool could not take the extend."""
        codes = ["ADMIT", "NO_TOKEN", "ADMIT"]
        self.assertEqual({self._verdict(codes, r) for r in (0, 1, 2)}, {"NO_TOKEN"})

    def test_the_host_refusal_still_wins(self):
        codes = ["OTHER", "NO_TOKEN", "ADMIT"]
        self.assertEqual({self._verdict(codes, r) for r in (0, 1, 2)}, {"OTHER"})
        self.assertEqual({self._verdict(["ADMIT"] * 3, r) for r in (0, 1, 2)}, {"ADMIT"})

    def test_a_split_or_a_missing_rank_is_a_named_stop(self):
        from sglang.srt.managers import tp_match_floor as f

        with self.assertRaisesRegex(f.FormAAdmissionSplit, "ADMISSION SPLIT"):
            f.form_a_admission_verdict(
                "a", "ADMIT", is_host=True, exchange=None,
                gather=lambda p: [("a", "ADMIT", 1, 1), ("b", "ADMIT", 1, 1)],
            )
        with self.assertRaisesRegex(f.FormAAdmissionSplit, "ADMISSION MALFORMED"):
            f.form_a_admission_verdict(
                "a", "ADMIT", is_host=False, exchange=None,
                gather=lambda p: [("a", "ADMIT", 1, 1), None],
            )

    def test_the_scheduler_gathers_only_under_the_cut(self):
        import inspect

        from sglang.srt.managers import scheduler as sch

        src = inspect.getsource(sch.Scheduler._form_a_admission_follow_fn)
        self.assertIn("form_a_token_cut_active()", src)
        self.assertIn("gather=_gather", src)
        self.assertIn("all_gather_object", inspect.getsource(sch.Scheduler._form_a_tp_gather))


class TestTokenCutRiegel(unittest.TestCase):
    """S3e: the launch refusal of kv=qsa_forma_dcp reads the seam register."""

    def _ns(self, dry=False):
        return types.SimpleNamespace(dry_run=dry, d_kv_token_cut="joint")

    def test_the_riegel_names_the_unwired_seam(self):
        """RED ON cdd6522981: the refusal was unconditional prose ("the
        workers attend nothing") -- true before S3c, false after it -- and
        would never lift."""
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(rank_role.TOKEN_CUT_SEAMS, ("F4", "F5", "F12", "F14"))
        self.assertEqual(rank_role.unwired_token_cut_seams(), ("F14",))
        with self.assertRaisesRegex(L.Weg2TokenCutNotWired, r"seam\(s\) F14 unwired.*#239 S3"):
            L.refuse_unwired_token_cut(self._ns(), types.SimpleNamespace(kv="qsa_forma_dcp"))
        L.refuse_unwired_token_cut(self._ns(dry=True), types.SimpleNamespace(kv="qsa_forma_dcp"))

    def test_the_riegel_lifts_when_every_seam_is_built(self):
        import dataclasses

        from sglang.srt.weg2 import launcher as L

        built = dict(rank_role.SEAMS)
        built["F14"] = dataclasses.replace(built["F14"], wired=True)
        with mock.patch.object(rank_role, "SEAMS", built):
            self.assertEqual(rank_role.unwired_token_cut_seams(), ())
            L.refuse_unwired_token_cut(self._ns(), types.SimpleNamespace(kv="qsa_forma_dcp"))
        f5_open = dict(rank_role.SEAMS)
        f5_open["F5"] = dataclasses.replace(f5_open["F5"], wired=False)
        with mock.patch.object(rank_role, "SEAMS", f5_open):
            with self.assertRaisesRegex(L.Weg2TokenCutNotWired, "F5, F14"):
                L.refuse_unwired_token_cut(self._ns(), types.SimpleNamespace(kv="qsa_forma_dcp"))

    def test_a_boot_without_a_host_tier_needs_no_f13(self):
        """S4a: --weg2-disable-hicache (no L2, no L3) enters no F14 path, so the
        cut boots there -- with the named line -- and nowhere else."""
        import contextlib
        import dataclasses
        import io

        from sglang.srt.weg2 import launcher as L

        ns = types.SimpleNamespace(dry_run=False, d_kv_token_cut="joint", weg2_disable_hicache=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            L.refuse_unwired_token_cut(ns, types.SimpleNamespace(kv="qsa_forma_dcp"))
        self.assertIn("#239 KV-TOKEN-SCHNITT F14-FREI (kein Host-Tier)", out.getvalue())
        f5_open = dict(rank_role.SEAMS)
        f5_open["F5"] = dataclasses.replace(f5_open["F5"], wired=False)
        with mock.patch.object(rank_role, "SEAMS", f5_open):
            with self.assertRaisesRegex(L.Weg2TokenCutNotWired, "F5, F14"):
                L.refuse_unwired_token_cut(ns, types.SimpleNamespace(kv="qsa_forma_dcp"))

    def test_the_f13_anchors_hit(self):
        import os

        root = os.path.dirname(rank_role.__file__)
        for rel, line, needle in rank_role.SEAMS["F14"].anchors:
            lines = open(os.path.join(root, rel)).read().splitlines()
            window = "\n".join(lines[max(0, line - 6): line + 5])
            self.assertIn(needle, window, f"{rel}:{line}")
        self.assertIn("F14", rank_role.UNWIRED_ORDER)


if __name__ == "__main__":
    unittest.main()
