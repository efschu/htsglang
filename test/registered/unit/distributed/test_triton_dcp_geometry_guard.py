"""The Triton backend's DCP geometry gate: loud at boot, never silently wrong.

``reject_unsupported_dcp_geometry`` has three branches:

1. **The uneven-DCP lane** (a ``--rank-tp-ratio`` plan installed, so the KV
   pool is token-sharded with the FULL replicated kv-head set per row).
   #169.1 refused it outright, because the Triton path then implemented only
   the EVEN modulo owner rule and never gathered the kv heads before writing.
   #173 ported the weighted machinery over from flashinfer -- the SAME
   ``layers/dcp/owner.py`` functions on both sides -- so the lane is now
   SERVED, and this branch only refuses the parts of it with no Triton twin:
   the weightless-KV fast lane, MLA, TREE-masked speculative verify, sliding
   window. (#180 ported flashinfer's M4 verify split, so CHAIN speculative
   decoding -- ``--speculative-eagle-topk 1`` -- left that list.)

   The replication arithmetic of branch 3 deliberately does not run here: the
   pool rows carry every kv head by construction under this lane.

2. **A token vector without a plan** -- weighted pool sizing with head-sharded
   rows, a half-installed state no backend serves.

3. **Even DCP without kv-head replication** (the pre-existing rule, kept
   byte-identical): ``tp_size // total_kv_heads >= dcp_size``.

CPU only: the rule is a pure function of the geometry, so it is called here
directly -- no device, no process group, no ModelRunner.
"""

import unittest

from sglang.srt.layers.attention.triton_backend import (
    reject_unsupported_dcp_geometry,
    total_swa_kv_heads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _call(
    dcp,
    tp,
    kv,
    *,
    plan=False,
    tokens=False,
    weightless=False,
    swa=None,
    mla=False,
    spec=False,
    spec_tree=False,
    window=False,
    swa_hybrid_dcp=False,
):
    reject_unsupported_dcp_geometry(
        dcp,
        tp,
        kv,
        uneven_plan=plan,
        weighted_tokens=tokens,
        weightless_kv=weightless,
        swa_kv_heads=swa,
        mla=mla,
        speculative=spec,
        speculative_tree=spec_tree,
        sliding_window=window,
        swa_hybrid_dcp=swa_hybrid_dcp,
    )


class _HF:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Cfg:
    def __init__(self, hf):
        self.hf_text_config = hf


class TestTritonDcpGeometryGuard(CustomTestCase):
    # ---------------------------------------------------------------- uneven

    def test_the_measured_mojibake_config_is_now_served(self):
        """THE acceptance case for #173.

        The config that produced CJK mojibake on the pre-#169 backend is
        uneven-DCP + a dense (qwen2) class on triton: a --rank-tp-ratio plan, a
        non-uniform token vector, dcp == tp. #169.1 rejected it because the
        backend had no weighted wiring; #173 gave it the weighted wiring, so it
        must now pass the gate -- INCLUDING at kv >= tp (8 kv heads over tp 3),
        where the kv heads are head-sharded and the write gathers them.

        Note the replication arithmetic of branch 3 would reject this geometry
        (3 // 8 = 0 < 3). It must not run for this lane: under it every pool row
        holds all 8 kv heads regardless of what any rank projects.
        """
        _call(3, 3, 8, plan=True, tokens=True)
        # kv < tp, the REPLICATED-KV / Nordstern 27B-class case
        _call(3, 3, 2, plan=True, tokens=True)
        # the same lane with the even-modulo rule (uniform / absent vector)
        _call(3, 3, 2, plan=True, tokens=False)

    def test_the_lane_still_refuses_what_has_no_triton_twin(self):
        """Opening the gate must not open it for the pieces that were never
        ported. Each names itself, so the operator sees which feature to drop.

        The sliding window is the CONDITIONAL one since #96 (see
        test_a_windowed_model_is_served_only_under_the_stage_b_preconditions):
        refused unless the Stage-B lane carries it.
        """
        for kwargs, fragment in (
            ({"weightless": True}, "weightless-KV fast lane"),
            ({"mla": True}, "MLA"),
            ({"spec": True, "spec_tree": True}, "TREE-masked speculative verify"),
            ({"window": True}, "sliding window"),
        ):
            with self.subTest(**kwargs):
                _call(3, 3, 2, plan=True, tokens=True)  # without it: served
                with self.assertRaises(ValueError) as ctx:
                    _call(3, 3, 2, plan=True, tokens=True, **kwargs)
                self.assertIn(fragment, str(ctx.exception))

    def test_chain_speculative_decoding_is_now_served(self):
        """THE acceptance case for #180.

        Before it, ANY speculative decoding was refused on this lane because
        Triton's target-verify built FULL un-sharded kv indices. The M4 verify
        split closes that, but only for a CHAIN draft: at topk == 1 the
        draft->draft mask IS the causal mask, so it can be dropped, which an
        owner-sharded prefix requires (the mask's row stride is the GLOBAL
        prefix length).

        So speculative=True alone must now pass -- on both sub-lanes, kv < tp
        (replicated kv heads, the 27B/35B Nordstern case) and kv >= tp.
        """
        _call(3, 3, 2, plan=True, tokens=True, spec=True)
        _call(3, 3, 8, plan=True, tokens=True, spec=True)
        # and with the even-modulo rule inside the same lane
        _call(3, 3, 2, plan=True, tokens=False, spec=True)

    def test_only_the_tree_half_of_speculation_is_refused(self):
        """The narrowing must be exactly one predicate wide.

        ``speculative_tree`` without ``speculative`` is not a real config and
        must not manufacture a refusal on its own; ``speculative`` without the
        tree is the served chain; both together is the refusal, and the message
        has to point at the actionable flag rather than at speculation as such.
        """
        _call(3, 3, 2, plan=True, tokens=True, spec_tree=True)  # incoherent: inert
        _call(3, 3, 2, plan=True, tokens=True, spec=True)  # chain: served
        with self.assertRaises(ValueError) as ctx:
            _call(3, 3, 2, plan=True, tokens=True, spec=True, spec_tree=True)
        msg = str(ctx.exception)
        self.assertIn("--speculative-eagle-topk 1", msg)
        self.assertIn("#76", msg)

    def test_a_windowed_model_is_served_only_under_the_stage_b_preconditions(self):
        """#96 Stage B: the ~10 global layers of an SWA-hybrid are token-sharded
        while the ~50 window layers keep their unsharded local path, so nothing
        has to causally mask a sparse owned-slot subset -- which is what the
        refusal was about. ``swa_hybrid_dcp`` is swa_hybrid_dcp_lane(...): a
        hybrid model (global AND window layers), cap-sized SWA pool, plan,
        target worker. Without it the refusal must stand verbatim, because every
        other windowed configuration (pure-SWA model, ratio-sized SWA pool,
        draft worker) still has no Triton twin.
        """
        # served: the Stage-B lane
        _call(3, 3, 8, plan=True, tokens=True, window=True, swa_hybrid_dcp=True)
        _call(3, 3, 2, plan=True, tokens=True, window=True, swa_hybrid_dcp=True)
        # a larger SWA kv base does not change branch 1's verdict (branch 3's
        # replication arithmetic does not run under a plan)
        _call(3, 3, 2, plan=True, tokens=True, window=True, swa=8, swa_hybrid_dcp=True)
        # refused: window without the lane, same message as before #96
        with self.assertRaises(ValueError) as ctx:
            _call(3, 3, 2, plan=True, tokens=True, window=True)
        self.assertIn("sliding window", str(ctx.exception))
        # and the lane flag does NOT open any of the other refusals
        for kwargs, fragment in (
            ({"weightless": True}, "weightless-KV fast lane"),
            ({"mla": True}, "MLA"),
            ({"spec": True, "spec_tree": True}, "--speculative-eagle-topk 1"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    _call(
                        3,
                        3,
                        2,
                        plan=True,
                        tokens=True,
                        window=True,
                        swa_hybrid_dcp=True,
                        **kwargs,
                    )
                self.assertIn(fragment, str(ctx.exception))

    def test_window_lane_times_chain_verify_is_permitted_but_unvalidated(self):
        """Rebase seam between #96 and the merged #180. REGISTERED, NOT PROVEN.

        Before #180 this pair could not arise: any `speculative=True` was
        refused on the uneven lane, so #96's original test listed it among the
        refusals the Stage-B flag must not open. #180 serves the CHAIN verify
        (topk == 1), so after the rebase the guard lets
        `swa_hybrid_dcp + speculative` through -- the assertion was stale, not
        the code.

        What is NOT established is that the combination is *correct*. #180
        drops `custom_mask`/`mask_indptr` under `dcp_size > 1` on the
        target-verify branch, justified for the DCP paged path ("the dxd
        draft->draft block IS the causal mask"). Under #96 a sliding-window
        layer no longer takes the DCP path at all: it falls through to the
        plain 2-stage kernel and DOES receive a mask, with `window_kv_offsets`
        re-basing the column arithmetic. Whether the drop is still right there
        is unverified in either direction.

        This test pins only what the guard currently decides, so that a future
        change to that decision is visible. It is not evidence of correctness.
        Settle on GPU with a chain-verify (`--speculative-eagle-topk 1`) run of
        an SWA-hybrid model on the uneven-DCP lane, asserting per layer what a
        sliding-window layer receives under `is_target_verify()`.
        """
        kw = dict(plan=True, tokens=True, window=True, swa_hybrid_dcp=True, spec=True)
        _call(3, 3, 2, **kw)
        _call(3, 3, 8, **kw)

    def test_the_lane_names_every_unserved_feature_at_once(self):
        """A config that trips several must name all of them, so the operator
        does not drop one and re-run into the next."""
        with self.assertRaises(ValueError) as ctx:
            _call(
                3,
                3,
                2,
                plan=True,
                tokens=True,
                weightless=True,
                mla=True,
                spec=True,
                spec_tree=True,
            )
        msg = str(ctx.exception)
        for fragment in (
            "weightless-KV fast lane",
            "MLA",
            "TREE-masked speculative verify",
        ):
            self.assertIn(fragment, msg)

    def test_a_token_vector_without_a_plan_is_still_refused(self):
        """The half-installed state: the pool is SIZED by the weighted rule
        (uneven_dcp_active) but its rows carry only this rank's kv-head shard
        (uneven_dcp_kv_replicated is what makes them full). The weighted read
        assumes full rows, so there is no correct interpretation -- and picking
        one silently would be the exact class of bug this guard exists for.
        """
        with self.assertRaises(ValueError) as ctx:
            _call(3, 3, 2, plan=False, tokens=True)
        msg = str(ctx.exception)
        self.assertIn("--rank-tp-ratio", msg)
        self.assertIn("weighted owner rule", msg.lower())

    def test_weightless_kv_fastlane_is_rejected_without_a_plan_too(self):
        """Head counts [all, 0, 0] are not an equal split either.

        Geometry chosen so the replication arithmetic ACCEPTS it (tp=4, kv=2,
        dcp=2 -> replicas 2 >= 2); without the fast-lane branch nothing would
        stop it.
        """
        _call(2, 4, 2)  # same geometry, fast lane off -> accepted
        with self.assertRaises(ValueError) as ctx:
            _call(2, 4, 2, weightless=True)
        self.assertIn("weightless", str(ctx.exception))

    # ------------------------------------------------- even-DCP replication

    def test_replication_arithmetic_on_the_four_measured_cases(self):
        """Unchanged from the predecessor guard, re-pinned behaviourally
        (it used to be asserted only as a source regex)."""
        cases = [
            # (attn_tp, total_kv, dcp, must_reject)
            (2, 2, 2, True),  # Qwen2.5-1.5B TP=2/DCP=2 -> measured mojibake
            (2, 8, 2, True),  # Qwen3-0.6B              -> measured mojibake
            (4, 2, 2, False),  # replicas 2 >= 2        -> measured coherent
            (8, 4, 2, False),  # the path's origin (Qwen3.5-397B)
        ]
        for attn_tp, kv, dcp, must_reject in cases:
            with self.subTest(tp=attn_tp, kv=kv, dcp=dcp):
                if must_reject:
                    with self.assertRaises(ValueError) as ctx:
                        _call(dcp, attn_tp, kv)
                    # names the numbers, not just "unsupported"
                    self.assertIn(f"{attn_tp} // {kv}", str(ctx.exception))
                else:
                    _call(dcp, attn_tp, kv)

    # ---------------------------------------------- hybrid: two kv bases

    def test_a_larger_sliding_window_kv_base_is_the_binding_one(self):
        """THE falsifier for the hybrid half (#169.3).

        A hybrid class can declare a SECOND kv-head total for its
        sliding-window layers. The replication condition has to hold for every
        layer kind the model runs: at tp=8/dcp=2 a full-attention base of 4
        gives replicas 2 (accepted), but if the SWA layers carry 8 kv heads
        their replicas is 1 and those layers attend the gathered q heads
        against the wrong kv head. Reading only the full-attention base lets
        exactly that model through.
        """
        _call(2, 8, 4)  # full-attention base alone: accepted
        with self.assertRaises(ValueError) as ctx:
            _call(2, 8, 4, swa=8)
        msg = str(ctx.exception)
        self.assertIn("sliding-window base", msg)
        self.assertIn("8 // 8", msg)

    def test_a_smaller_or_equal_swa_base_changes_no_verdict(self):
        """max() may only tighten. Every model with one base, or with a
        smaller SWA base, must keep the verdict it had before."""
        for kv, swa in ((4, 2), (4, 4), (4, None), (2, 1)):
            with self.subTest(kv=kv, swa=swa):
                _call(2, 8, kv, swa=swa)
        for kv, swa in ((8, 4), (8, 8), (8, None)):
            with self.subTest(kv=kv, swa=swa):
                with self.assertRaises(ValueError):
                    _call(2, 8, kv, swa=swa)

    def test_the_swa_base_is_read_from_both_config_spellings(self):
        self.assertIsNone(total_swa_kv_heads(_Cfg(_HF())))
        self.assertEqual(total_swa_kv_heads(_Cfg(_HF(swa_num_key_value_heads=8))), 8)
        self.assertEqual(
            total_swa_kv_heads(
                _Cfg(_HF(attention_other_setting={"num_attention_groups": 6}))
            ),
            6,
        )
        # a model config without hf_text_config must not explode
        self.assertIsNone(total_swa_kv_heads(object()))

    def test_zero_kv_heads_does_not_divide_by_zero(self):
        with self.assertRaises(ValueError):
            _call(2, 2, 0)

    # ------------------------------------------------------- default path

    def test_dcp_off_is_inert_under_every_flag_combination(self):
        """dcp_size <= 1 is the default path: nothing may raise there, not even
        with a plan, a token vector and the fast lane all set."""
        for dcp in (0, 1):
            for plan in (False, True):
                for tokens in (False, True):
                    for weightless in (False, True):
                        with self.subTest(
                            dcp=dcp,
                            plan=plan,
                            tokens=tokens,
                            weightless=weightless,
                        ):
                            _call(
                                dcp,
                                3,
                                8,
                                plan=plan,
                                tokens=tokens,
                                weightless=weightless,
                            )

    def test_the_constructor_still_calls_the_rule(self):
        """A guard nobody calls is not a guard. Pins the call site so a
        refactor cannot orphan the rule while every test above stays green.
        """
        import pathlib

        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        self.assertIn("reject_unsupported_dcp_geometry(", src)
        # constructor call site, with all three uneven inputs wired in
        self.assertIn("uneven_plan=uneven_dcp_kv_replicated(self.dcp_size)", src)
        self.assertIn("weighted_tokens=uneven_dcp_active(self.dcp_size)", src)
        self.assertIn("weightless_kv=weightless_kv_active()", src)
        self.assertIn(
            "swa_kv_heads=total_swa_kv_heads(model_runner.model_config)", src
        )
        # #173: the still-unserved parts of the now-open uneven lane
        self.assertIn("mla=self.use_mla", src)
        self.assertIn("speculative=", src)
        self.assertIn("sliding_window=", src)
        # #180: the tree predicate is decided by the SHARED helper, never
        # re-derived at the call site. dcp_verify_mask_mode knows about both
        # doors onto a tree mask (topk > 1 and --speculative-dflash-tree-
        # verify); a local `self.topk > 1` here would silently reopen the
        # second one, which is exactly how it was missed the first time.
        self.assertIn("speculative_tree=", src)
        self.assertIn("dcp_verify_mask_mode(", src)
        self.assertNotIn("speculative_tree=bool(self.topk", src)
        # #96: and the Stage-B lane flag that makes the window one conditional
        self.assertIn("swa_hybrid_dcp=self.swa_hybrid_dcp", src)


if __name__ == "__main__":
    unittest.main()
