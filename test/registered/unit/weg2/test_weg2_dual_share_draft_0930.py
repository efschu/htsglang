# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 --dual-share: the draft worker is NOT shared between the groups.

Metal 30.09. boot dkr27bnvfp4dual1bbar1fs09300226: P PP2's draft (TP1, full
vocab) tried to bind D's draft image (TP3, vocab-sharded):
'candidate_selector.predecessor_codebook.weight' (248320, 256) here vs
(82816, 256) in the manifest -> UnionShareError. The two groups hold the draft
in different TP forms BY DESIGN, so the plan must keep it out of the union.

DANGER DIRECTIONS guarded here:
* under --dual-share both groups carry SGLANG_WEG2_UNION_ROLES=main, so
  neither D (owner) nor P (peer) touches a draft image;
* the skip happens before any socket / CUDA access;
* without the env (every non-dual boot) the draft role is still shared.
"""
from __future__ import annotations

import os
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import union_arena_bind as B
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class DualShareDraft(CustomTestCase):
    def test_launcher_keeps_draft_out_of_the_share(self):
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share"])
        L.resolve_dual_layout(ns)
        for g in ("D", "P"):
            self.assertEqual(L.dual_share_env(ns, g)[B.UNION_ROLES_ENV], "main")
        off = L.build_parser().parse_args(["--tree", "/x", "--tag", "t"])
        self.assertEqual(L.dual_share_env(off, "P"), {})

    def test_draft_role_skipped_before_any_access(self):
        env = {B.UNION_DIR_ENV: "/nonexistent-union", B.UNION_MODE_ENV: "bind", B.UNION_ROLES_ENV: "main"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(B, "bind_image", side_effect=AssertionError("draft must not bind")), \
                mock.patch.object(B, "own_image", side_effect=AssertionError("draft must not own")):
            self.assertIsNone(B.maybe_union_image(object(), device=0, role="draft"))
        env[B.UNION_MODE_ENV] = "own"
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(B, "own_image", side_effect=AssertionError("draft must not own")):
            self.assertIsNone(B.maybe_union_image(object(), device=0, role="draft"))

    def test_roles_default_shares_every_role(self):
        self.assertTrue(B.union_role_enabled("draft", {}))
        self.assertTrue(B.union_role_enabled("main", {B.UNION_ROLES_ENV: "main"}))
        self.assertFalse(B.union_role_enabled("draft", {B.UNION_ROLES_ENV: "main"}))
