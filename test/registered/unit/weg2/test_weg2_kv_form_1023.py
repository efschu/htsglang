# SPDX-License-Identifier: Apache-2.0
"""#1023 S1 -- the cap seam, and the trap that sits in front of it.

THE TRAP, in the engine's own arithmetic.  ``_hybrid_kv_token_cap``
(``model_runner_kv_cache_mixin.py:4834-4877``) computes

    concurrency = max(max_running_requests, max_mamba_cache_size // ratio)
    cap         = concurrency * (context_len + extra)

Today group D boots ``--max-running-requests 8`` with an auto-sized 40-slot
mamba pool, so ``max(8, 40 // 4) = 10``, ``cap = 2,621,510`` and the cap does
not bind: boot weg2sb5e printed a 671,680-token world pool ``bound by none``.

Take the SAME boot to ``--max-running-requests 1`` -- which is what frees 7/8
of both mamba posts and is the first rung of the harvest ladder -- and the
mamba sizer gives 5 slots, so ``max(1, 5 // 4) = 1`` and the cap becomes
``262,151``.  A 767,776-row projection is then clamped to 262,151: a SILENT
-61 % against the pool the boot started from.  Raising the per-request ceiling
is therefore the PRECONDITION of a bs1 form, not its consequence
(WEG2_KV1M_SPEC_0908 sec 2.1).

WHAT IS ASSERTED HERE, in order:

1.  the trap, against the REAL engine method on a double -- red before the
    knob exists, and it stays red-able forever because it is the thing the
    knob is for;
2.  that the launcher's printed ceiling is the SAME NUMBER the engine will
    compute.  ``launcher.d_kv_form_cap_tokens`` is a mirror of an engine
    formula, and a mirror that nothing pins drifts.  Every form position is
    driven through the engine's own method here, so a change to either side
    without the other goes red;
3.  that the DEFAULT position emits today's argv byte-for-byte -- the whole
    licence for shipping this knob at all;
4.  that a bs1 position OWNS the two quantities it derives, and refuses (W52)
    rather than silently winning against an operator value.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no pool allocation, no server, no boot.
The double is a ``SimpleNamespace``; the METHODS under test are the real
unbound functions off ``ModelRunnerKVCacheMixin``.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase

#: Group D's drafter, read off the argv the launcher builds rather than typed:
#: ``--speculative-num-draft-tokens 3`` -> ``extra = 4 + 3``
#: (``mem_cache/common.py:383-397``).  The user law of 2026-09-09 keeps the MTP
#: head on in every position of this knob, so the term is permanent.
EXTRA = 7

#: The projection the harvest ladder's R1 rung produces on this rig
#: (WEG2_KV1M_SPEC_0908 sec 0.4, rests 12,492 / 5,585 / 5,916 MiB).  It is the
#: number that gets clamped, so it is the number the trap must be shown on.
R1_PROJECTED_ROWS = 767_776


class _RunnerDouble(ModelRunnerKVCacheMixin):
    """The mixin itself, over a dict of the fields its two methods read.

    Subclassed rather than duck-typed on a ``SimpleNamespace`` because
    ``_apply_token_constraints`` DISPATCHES: it calls ``self._hybrid_kv_token_
    cap()``, the SWA sibling and ``self._apply_hybrid_kv_token_cap()``.  A
    double that only carried data would prove the clamp against methods nobody
    resolved -- which is a test of the test.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _double(
    *,
    max_running_requests,
    max_mamba_cache_size,
    mamba_ratio,
    context_len,
    max_total_tokens=None,
):
    """A ServerArgs/ModelRunner double thin enough to be honest about itself.

    Only the fields the two methods under test actually read.  ``dcp_size=1``
    and ``pp_size=1`` with inactive uneven budgets keep ``_apply_token_
    constraints`` on the path that needs neither a process group nor CUDA --
    the same path a single-node TP boot takes for its final clamp.
    """
    sa = SimpleNamespace(
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
        max_total_tokens=max_total_tokens,
        # get_req_to_token_extra_context_len's inputs. page_size=1 is group D's
        # own flag, which is what makes the topk>1 branch unreachable here.
        max_speculative_num_draft_tokens=3,
        speculative_algorithm="NEXTN",
        speculative_eagle_topk=1,
        page_size=1,
        uneven_memory_budgets_active=lambda: False,
    )
    return _RunnerDouble(
        server_args=sa,
        model_config=SimpleNamespace(context_len=context_len),
        mambaish_config=object(),
        _calculate_mamba_ratio=lambda: mamba_ratio,
        dcp_size=1,
        pp_size=1,
        tp_rank=0,
        is_dual_group_lane=False,
    )


def _cap(runner):
    return runner._hybrid_kv_token_cap()


def _clamp(runner, projected):
    return runner._apply_token_constraints(projected)


class TheTrapBeforeTheKnob(CustomTestCase):
    """Red-first: bs1 collapses the pool, and nothing in the tree says so."""

    def test_bs1_collapses_the_cap_to_one_context(self):
        runner = _double(
            max_running_requests=1,
            max_mamba_cache_size=5,
            mamba_ratio=4,
            context_len=262144,
        )
        self.assertEqual(_cap(runner), 262_151)

    def test_bs1_clamps_the_r1_projection_by_sixty_one_percent(self):
        runner = _double(
            max_running_requests=1,
            max_mamba_cache_size=5,
            mamba_ratio=4,
            context_len=262144,
        )
        clamped = _clamp(runner, R1_PROJECTED_ROWS)
        self.assertEqual(clamped, 262_151)
        # Stated as the loss, not as an inequality: the point of the slice is
        # that this is a -61 % nobody would see in a log line that says
        # "bound by hybrid mamba cap" three thousand lines in.
        self.assertLess(clamped / R1_PROJECTED_ROWS, 0.35)

    def test_todays_bs8_form_does_not_bind_at_all(self):
        """The control. Without it the assertion above proves only arithmetic."""
        runner = _double(
            max_running_requests=8,
            max_mamba_cache_size=40,
            mamba_ratio=4,
            context_len=262144,
        )
        self.assertEqual(_cap(runner), 2_621_510)
        self.assertEqual(_clamp(runner, R1_PROJECTED_ROWS), R1_PROJECTED_ROWS)


class TheLauncherPrintsTheNumberTheEngineComputes(CustomTestCase):
    """The mirror is pinned to the engine, position by position.

    ``d_kv_form_cap_tokens`` restates an engine formula in a module the engine
    never imports.  Nothing but this test stops the two drifting, and a drifted
    mirror is worse than no line at all: it prints a ceiling for a boot that
    ran under a different one.
    """

    def _form(self, name, **kw):
        kw.setdefault("d_bs_default", launcher.D_BS_DEFAULT)
        kw.setdefault("max_kv_per_request", launcher.CONTEXT_LENGTH_TOKENS)
        kw.setdefault("extra_tokens", EXTRA)
        return launcher.resolve_d_kv_form(name, **kw)

    def test_default_position(self):
        form = self._form("bs8")
        runner = _double(
            max_running_requests=form.d_bs,
            max_mamba_cache_size=40,
            mamba_ratio=launcher.D_MAMBA_RATIO,
            context_len=form.context_length,
        )
        # The launcher solves the concurrency term from the FLAGS it emits (it
        # cannot see the auto-sized pool), so it prints the bs floor; the
        # engine's own answer is at least that and never binds here.
        self.assertEqual(form.cap_tokens, 8 * (262144 + EXTRA))
        self.assertGreaterEqual(_cap(runner), form.cap_tokens)

    def test_bs1_ctx1m_position_reaches_a_million(self):
        form = self._form("bs1-ctx1m")
        self.assertEqual(form.d_bs, 1)
        self.assertEqual(form.context_length, 1_048_576)
        self.assertEqual(form.max_kv_per_request, 1_048_576)
        self.assertIsNone(form.max_mamba_cache_size)
        runner = _double(
            max_running_requests=form.d_bs,
            max_mamba_cache_size=5,
            mamba_ratio=launcher.D_MAMBA_RATIO,
            context_len=form.context_length,
        )
        self.assertEqual(_cap(runner), 1_048_583)
        self.assertEqual(form.cap_tokens, _cap(runner))
        # The whole point: the R1 projection is no longer clamped.
        self.assertEqual(_clamp(runner, R1_PROJECTED_ROWS), R1_PROJECTED_ROWS)

    def test_bs1_slots_position_reaches_a_million_without_touching_the_context(self):
        form = self._form("bs1-slots")
        self.assertEqual(form.d_bs, 1)
        self.assertEqual(form.context_length, 262144)
        self.assertEqual(form.max_mamba_cache_size, 16)
        runner = _double(
            max_running_requests=form.d_bs,
            max_mamba_cache_size=form.max_mamba_cache_size,
            mamba_ratio=launcher.D_MAMBA_RATIO,
            context_len=form.context_length,
        )
        self.assertEqual(_cap(runner), 1_048_604)
        self.assertEqual(form.cap_tokens, _cap(runner))
        self.assertEqual(_clamp(runner, R1_PROJECTED_ROWS), R1_PROJECTED_ROWS)

    def test_the_mirror_is_the_engines_formula_and_not_a_lookup(self):
        """Drive both sides over a grid, so a hardcoded pair cannot pass."""
        for mrr, slots, ratio, ctx in (
            (1, 5, 4, 262144),
            (1, 16, 4, 262144),
            (1, 5, 4, 1048576),
            (8, 40, 4, 262144),
            (2, 9, 3, 131072),
        ):
            runner = _double(
                max_running_requests=mrr,
                max_mamba_cache_size=slots,
                mamba_ratio=ratio,
                context_len=ctx,
            )
            self.assertEqual(
                launcher.d_kv_form_cap_tokens(max(mrr, slots // ratio), ctx, EXTRA),
                _cap(runner),
                f"mirror drifted at mrr={mrr} slots={slots} ratio={ratio} ctx={ctx}",
            )


class TheDefaultArgvIsUnchanged(CustomTestCase):
    """Backward compatibility, asserted on the bytes rather than promised."""

    def _argv(self, **kw):
        return launcher.argv_d(
            "py", "/m", [1, 2, 3], 1, 1, 1.0, [], d_bs=8, **kw
        )

    def test_default_position_emits_todays_flags(self):
        form = launcher.resolve_d_kv_form(
            "bs8",
            d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        )
        base = self._argv()
        derived = self._argv(
            context_length=form.context_length,
            max_mamba_cache_size=form.max_mamba_cache_size,
        )
        self.assertEqual(base, derived)
        self.assertNotIn("--max-mamba-cache-size", base)
        self.assertEqual(
            base[base.index("--context-length") + 1], str(262144)
        )

    def test_ctx1m_position_raises_only_group_ds_ceiling(self):
        form = launcher.resolve_d_kv_form(
            "bs1-ctx1m",
            d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        )
        argv = launcher.argv_d(
            "py", "/m", [1, 2, 3], 1, 1, 1.0, [],
            d_bs=form.d_bs,
            max_kv_per_request=form.max_kv_per_request,
            context_length=form.context_length,
            max_mamba_cache_size=form.max_mamba_cache_size,
        )
        self.assertEqual(argv[argv.index("--context-length") + 1], "1048576")
        self.assertEqual(argv[argv.index("--max-kv-per-request") + 1], "1048576")
        self.assertEqual(argv[argv.index("--max-running-requests") + 1], "1")
        # Group P is untouched by group D's form.
        argv_p = launcher.argv_p("py", "/m", [1, 2, 3], 1, 1, 1.0, [])
        self.assertEqual(argv_p[argv_p.index("--context-length") + 1], "262144")

    def test_slots_position_emits_the_state_pool_flag(self):
        form = launcher.resolve_d_kv_form(
            "bs1-slots",
            d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        )
        argv = launcher.argv_d(
            "py", "/m", [1, 2, 3], 1, 1, 1.0, [],
            d_bs=form.d_bs,
            max_kv_per_request=form.max_kv_per_request,
            context_length=form.context_length,
            max_mamba_cache_size=form.max_mamba_cache_size,
        )
        self.assertEqual(argv[argv.index("--max-mamba-cache-size") + 1], "16")
        self.assertEqual(argv[argv.index("--context-length") + 1], "262144")

    def test_the_drafter_survives_every_position(self):
        """User law 2026-09-09, "mtp bleibt dabei", asserted on the argv.

        No position of this knob is allowed to become an MTP-off knob by
        omission -- that is R6, it is not built, and the ``extra`` term this
        module reads off the argv depends on the head being there.
        """
        for name in launcher.D_KV_FORM_CHOICES:
            form = launcher.resolve_d_kv_form(
                name,
                d_bs_default=8,
                max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
                extra_tokens=EXTRA,
            )
            argv = launcher.argv_d(
                "py", "/m", [1, 2, 3], 1, 1, 1.0, [],
                d_bs=form.d_bs,
                max_kv_per_request=form.max_kv_per_request,
                context_length=form.context_length,
                max_mamba_cache_size=form.max_mamba_cache_size,
            )
            self.assertIn("--speculative-algorithm", argv, name)
            self.assertEqual(
                argv[argv.index("--speculative-num-draft-tokens") + 1], "3", name
            )
            self.assertEqual(launcher.d_kv_form_extra_tokens(argv), EXTRA, name)


class TheFormOwnsWhatItDerives(CustomTestCase):
    """W52, and it is a refusal rather than a precedence rule on purpose."""

    def test_unknown_position_is_refused(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.resolve_d_kv_form(
                "bs1", d_bs_default=8,
                max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
                extra_tokens=EXTRA,
            )
        self.assertIn("W52 Weg2KvFormRefused", str(cm.exception))

    def test_an_explicit_d_bs_beside_a_bs1_form_is_refused(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.resolve_d_kv_form(
                "bs1-slots", d_bs_default=4,
                max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
                extra_tokens=EXTRA, d_bs_was_passed=True,
            )
        self.assertIn("W52 Weg2KvFormRefused", str(cm.exception))
        self.assertIn("--d-bs", str(cm.exception))

    def test_an_explicit_kv_ceiling_beside_a_bs1_form_is_refused(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.resolve_d_kv_form(
                "bs1-ctx1m", d_bs_default=8,
                max_kv_per_request=131072,
                extra_tokens=EXTRA, max_kv_per_request_was_passed=True,
            )
        self.assertIn("W52 Weg2KvFormRefused", str(cm.exception))
        self.assertIn("--max-kv-per-request", str(cm.exception))

    def test_the_default_position_leaves_both_flags_to_the_operator(self):
        form = launcher.resolve_d_kv_form(
            "bs8", d_bs_default=4, max_kv_per_request=131072,
            extra_tokens=EXTRA, d_bs_was_passed=True,
            max_kv_per_request_was_passed=True,
        )
        self.assertEqual(form.d_bs, 4)
        self.assertEqual(form.max_kv_per_request, 131072)

    def test_the_d_bs_flag_default_is_a_sentinel_not_eight(self):
        """The reason the refusal above can exist at all.

        Comparing an operator value against a default cannot tell "asked for
        8" from "said nothing", and that is exactly how an owned quantity ends
        up silently winning.
        """
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertIsNone(ns.d_bs)
        self.assertIsNone(ns.max_kv_per_request)
        self.assertEqual(ns.d_kv_form, launcher.D_KV_FORM_DEFAULT)
        self.assertEqual(launcher.D_BS_DEFAULT, 8)


class TheProvenanceLineSaysWhatItBought(CustomTestCase):
    """Instrument-text law: the line names the form, the ceiling and the price."""

    def test_every_position_names_itself_and_its_ceiling(self):
        for name in launcher.D_KV_FORM_CHOICES:
            form = launcher.resolve_d_kv_form(
                name, d_bs_default=8,
                max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
                extra_tokens=EXTRA,
            )
            self.assertIn("WEG2 D-KV-FORM", form.line)
            self.assertIn("objective=%s" % name, form.line)
            self.assertIn(str(form.cap_tokens), form.line)

    def test_a_selected_position_names_its_price_and_the_trap_it_avoids(self):
        line = launcher.resolve_d_kv_form(
            "bs1-ctx1m", d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        ).line
        self.assertIn("SELECTED", line)
        # the number the bare bs1 form would have clamped to
        self.assertIn("262151", line.replace(",", ""))
        self.assertIn("UNPROVEN", line)

    def test_the_default_line_says_it_did_not_take_one(self):
        line = launcher.resolve_d_kv_form(
            "bs8", d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        ).line
        self.assertIn("the default, unchanged", line)
        self.assertIn("did not take one", line)


if __name__ == "__main__":
    unittest.main()
