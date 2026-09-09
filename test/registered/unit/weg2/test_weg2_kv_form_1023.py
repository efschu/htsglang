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
The ServerArgs half of the double is a ``SimpleNamespace``; the RUNNER is a
real ``ModelRunnerKVCacheMixin`` subclass, because ``_apply_token_constraints``
dispatches and a data-only double would have tested nothing.

ROUND 2 (the must_fix round) added, below the original four: an argv golden
captured on the PARENT commit (the only comparison here that is not the new
code against itself), the wiring into ``main`` (which was the one untested part
and held two surviving mutants), ``--extra-d`` as the back door to every
quantity the form owns, the reserve that ``bs1-slots`` deletes rather than
pays, and the pinned-ratio provenance.
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


# ---------------------------------------------------------------------------
# ROUND 2 -- the must_fix round.  Everything below was written against a
# finding, and every finding names the mutant that lived where no test looked.
# ---------------------------------------------------------------------------

#: Group D's and group P's DEFAULT argv, CAPTURED ON THE PARENT 4f762260ba and
#: pinned here.  Provenance: run on the remote desk at the parent commit,
#: /spinning/gpu-arb/devtools/remote_results/2f36737928_20260909T054211Z.
#:
#: Round-2 review finding 4: ``test_default_position_emits_todays_flags``
#: compares ``argv_d()`` against ``argv_d(**default_form_kwargs)`` -- the new
#: code against itself -- and therefore cannot see a change to ``argv_d``.  A
#: literal captured BEFORE the slice is the only comparison in this file that
#: can actually fail for the reason the slice claims ("the default path is
#: byte-identical"), so it is the oracle and the self-comparison is the
#: consistency check beside it.
GOLDEN_D_AT_PARENT = ['py', '-m', 'sglang.launch_server', '--model-path', '/m', '--trust-remote-code', '--served-model-name', 'Qwen3.8-27B', '--rank-gpu-id', '0,1,2', '--skip-server-warmup', '--kv-cache-dtype', 'fp8_e4m3', '--context-length', '262144', '--max-kv-per-request', '262144', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder', '--chat-template-default-kwargs', '{"preserve_thinking": true}', '--enable-cache-report', '--enable-metrics', '--enable-hierarchical-cache', '--hicache-host-role', 'staging', '--hicache-size', '1', '--hicache-mamba-host-mib', '1', '--hicache-write-policy', 'write_through', '--hicache-storage-backend', 'file', '--hicache-mem-layout', 'layer_first', '--hicache-io-backend', 'direct', '--hicache-storage-backend-extra-config', '{"max_size":"1G","min_free_space":"1G","max_size_scope":"shared"}', '--hicache-canonical-kv-page', '--host', '127.0.0.1', '--chunked-prefill-size', '4096', '--scheduler-distributed-teardown', '--page-size', '1', '--random-seed', '785500001', '--mamba-slot-reorder', '--kv-backing-relief', '--barlink', '--barlink-transport', 'bar1', '--barlink-bar1-cap-cycles', '300000000000', '--collective-census-interval', '50', '--uneven-dcp', '--uneven-dcp-weighted', '--mamba-ssm-dtype', 'bfloat16', '--enable-memory-saver', '--enable-weights-cpu-backup', '--max-running-requests', '8', '--tp-prefill-max-tokens', '0', '--mamba-radix-cache-strategy', 'extra_buffer', '--tp-size', '3', '--pp-size', '1', '--num-continuous-decode-steps', '1', '--rank-gpu-memory-mib', '1,2,3', '--rank-tp-ratio', 'auto', '--speculative-algorithm', 'NEXTN', '--speculative-num-steps', '2', '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '3', '--barlink-bar1-window-mib', '16,TP_0=32,DCP_0=40', '--barlink-uncovered-class', 'refuse', '--port', '30032']

GOLDEN_P_AT_PARENT = ['py', '-m', 'sglang.launch_server', '--model-path', '/m', '--trust-remote-code', '--served-model-name', 'Qwen3.8-27B', '--rank-gpu-id', '0,1,2', '--skip-server-warmup', '--kv-cache-dtype', 'fp8_e4m3', '--context-length', '262144', '--max-kv-per-request', '262144', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder', '--chat-template-default-kwargs', '{"preserve_thinking": true}', '--enable-cache-report', '--enable-metrics', '--enable-hierarchical-cache', '--hicache-host-role', 'staging', '--hicache-size', '1', '--hicache-mamba-host-mib', '1', '--hicache-write-policy', 'write_through', '--hicache-storage-backend', 'file', '--hicache-mem-layout', 'layer_first', '--hicache-io-backend', 'direct', '--hicache-storage-backend-extra-config', '{"max_size":"1G","min_free_space":"1G","max_size_scope":"shared"}', '--hicache-canonical-kv-page', '--host', '127.0.0.1', '--chunked-prefill-size', '4096', '--scheduler-distributed-teardown', '--page-size', '1', '--random-seed', '785500001', '--mamba-slot-reorder', '--kv-backing-relief', '--barlink', '--barlink-transport', 'bar1', '--barlink-bar1-cap-cycles', '300000000000', '--collective-census-interval', '50', '--mamba-ssm-dtype', 'bfloat16', '--enable-memory-saver', '--enable-weights-cpu-backup', '--max-running-requests', '8', '--tp-size', '1', '--pp-size', '3', '--pp-async-batch-depth', '0', '--disable-overlap-schedule', '--pp-stage-ratio', '32,18,14', '--pp-attn-stage-ratio', '8,4,4', '--rank-gpu-memory-mib', '1,2,3', '--barlink-bar1-window-mib', '24,PP_0=96', '--speculative-algorithm', 'NEXTN', '--speculative-num-steps', '2', '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '3', '--speculative-draft-kv-only', '--max-total-tokens', '428000', '--port', '30031']


def _ns(**kw):
    """An argparse namespace double carrying only what the wiring reads."""
    fields = dict(
        model="/m",
        d_bs=None,
        max_kv_per_request=None,
        extra_d="",
        d_kv_form="bs8",
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


class TheArgvGoldenIsFromBeforeTheSlice(CustomTestCase):
    """The one comparison here that is not the new code against itself."""

    def test_group_d_default_argv_is_byte_identical_to_the_parent(self):
        self.maxDiff = None
        self.assertEqual(
            launcher.argv_d("py", "/m", [1, 2, 3], 1, 1, 1.0, [], d_bs=8),
            GOLDEN_D_AT_PARENT,
        )

    def test_group_p_argv_is_byte_identical_to_the_parent(self):
        self.maxDiff = None
        self.assertEqual(
            launcher.argv_p("py", "/m", [1, 2, 3], 1, 1, 1.0, []),
            GOLDEN_P_AT_PARENT,
        )


class TheWiringIntoABootIsTheTestedPart(CustomTestCase):
    """Round-2 review finding 1: two mutants lived in ``main`` and survived.

    Both are asserted here against ``d_kv_form_for_boot``, which is the whole
    wiring now: MR1 was ``max_kv_per_request_was_passed=False`` (the W52
    ceiling refusal disarmed at the CLI) and MR2 was dropping ``d_bs =
    form.d_bs`` (a bs1 position launched at ``--max-running-requests 8`` while
    the log printed 1).
    """

    def test_a_bs1_position_returns_the_forms_concurrency_not_the_default(self):
        d_bs, _, form = launcher.d_kv_form_for_boot(_ns(d_kv_form="bs1-slots"))
        self.assertEqual(d_bs, 1)
        self.assertEqual(form.d_bs, 1)

    def test_the_default_position_returns_the_shipped_eight(self):
        d_bs, max_kv, form = launcher.d_kv_form_for_boot(_ns())
        self.assertEqual(d_bs, 8)
        self.assertEqual(max_kv, launcher.CONTEXT_LENGTH_TOKENS)
        self.assertTrue(form.is_default)

    def test_an_operator_ceiling_beside_a_bs1_position_is_refused_at_the_cli(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.d_kv_form_for_boot(
                _ns(d_kv_form="bs1-ctx1m", max_kv_per_request=262144)
            )
        self.assertIn("W52", str(cm.exception))

    def test_an_operator_concurrency_beside_a_bs1_position_is_refused_at_the_cli(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.d_kv_form_for_boot(_ns(d_kv_form="bs1-slots", d_bs=8))
        self.assertIn("W52", str(cm.exception))

    def test_an_extra_d_draft_width_moves_the_printed_extra_term(self):
        """The printed ceiling must track the argv that will actually run."""
        _, _, base = launcher.d_kv_form_for_boot(_ns(d_kv_form="bs1-slots"))
        _, _, wider = launcher.d_kv_form_for_boot(
            _ns(
                d_kv_form="bs1-slots",
                extra_d="--speculative-num-draft-tokens 5",
            )
        )
        self.assertEqual(base.cap_tokens, 4 * (262144 + 7))
        self.assertEqual(wider.cap_tokens, 4 * (262144 + 9))


class TheFormOwnsTheBackDoorToo(CustomTestCase):
    """Round-2 refuter MUST_FIX 2: ``--extra-d`` is appended LAST and wins."""

    def test_an_extra_d_concurrency_beside_a_bs1_position_is_refused(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.d_kv_form_for_boot(
                _ns(d_kv_form="bs1-slots", extra_d="--max-running-requests 8")
            )
        msg = str(cm.exception)
        self.assertIn("W52", msg)
        self.assertIn("--max-running-requests", msg)

    def test_every_owned_flag_is_refused_from_extra_d(self):
        for flag, value in (
            ("--max-running-requests", "8"),
            ("--context-length", "262144"),
            ("--max-kv-per-request", "262144"),
            ("--max-mamba-cache-size", "40"),
            ("--speculative-algorithm", "NONE"),
        ):
            with self.subTest(flag=flag):
                with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                    launcher.d_kv_form_for_boot(
                        _ns(d_kv_form="bs1-ctx1m", extra_d=f"{flag} {value}")
                    )
                self.assertIn(flag, str(cm.exception))

    def test_the_equals_form_of_the_flag_is_seen_too(self):
        with self.assertRaises(launcher.Weg2LaunchRefused):
            launcher.d_kv_form_for_boot(
                _ns(d_kv_form="bs1-slots", extra_d="--context-length=1048576")
            )

    def test_an_unrelated_extra_d_flag_still_launches(self):
        _, _, form = launcher.d_kv_form_for_boot(
            _ns(d_kv_form="bs1-slots", extra_d="--log-level debug")
        )
        self.assertEqual(form.name, "bs1-slots")

    def test_the_default_position_leaves_extra_d_alone(self):
        """bs8 owns nothing, so it refuses nothing -- today's boot unchanged."""
        _, _, form = launcher.d_kv_form_for_boot(
            _ns(extra_d="--max-running-requests 4")
        )
        self.assertTrue(form.is_default)


class AnAgreeingOperatorValueIsNotADisagreement(CustomTestCase):
    """Round-2 review finding 7: W52 refused a non-conflict."""

    def test_a_ceiling_equal_to_the_forms_own_is_accepted(self):
        _, _, form = launcher.d_kv_form_for_boot(
            _ns(d_kv_form="bs1-slots", max_kv_per_request=262144)
        )
        self.assertEqual(form.max_kv_per_request, 262144)

    def test_a_ceiling_different_from_the_forms_own_is_refused(self):
        with self.assertRaises(launcher.Weg2LaunchRefused):
            launcher.d_kv_form_for_boot(
                _ns(d_kv_form="bs1-slots", max_kv_per_request=131072)
            )

    def test_d_bs_one_beside_a_bs1_position_is_accepted(self):
        d_bs, _, form = launcher.d_kv_form_for_boot(
            _ns(d_kv_form="bs1-ctx1m", d_bs=1)
        )
        self.assertEqual(d_bs, 1)
        self.assertEqual(form.name, "bs1-ctx1m")


class TheSlotsPositionPricesWhatItDeletes(CustomTestCase):
    """Round-2 refuter MUST_FIX 1: the position deletes a 1 GiB/card post.

    Pinning ``--max-mamba-cache-size`` moves group D from the demand-driven
    mamba branch to the explicit one, and the prefill activation reserve is
    subtracted ONLY on the demand branch.  Before this round the launcher
    priced "+11 mamba slots" and said nothing about the reserve.
    """

    def _form(self, name):
        return launcher.resolve_d_kv_form(
            name,
            d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        )

    def test_the_reserve_is_read_from_the_engine_not_typed_here(self):
        import re as _re

        src = open(
            launcher._ENGINE_KV_MIXIN_PATH, encoding="utf-8"
        ).read()
        m = _re.search(
            r"^MAMBA_AUTO_ACTIVATION_RESERVE_MIB\s*=\s*(\d+)\s*$", src, _re.M
        )
        self.assertIsNotNone(m, "the engine constant this position prices is gone")
        self.assertEqual(
            launcher.engine_int_constant("MAMBA_AUTO_ACTIVATION_RESERVE_MIB"),
            int(m.group(1)),
        )

    def test_the_slots_position_carries_the_deleted_reserve_as_a_field(self):
        form = self._form("bs1-slots")
        self.assertEqual(
            form.unpriced_reserve_mib,
            launcher.engine_int_constant("MAMBA_AUTO_ACTIVATION_RESERVE_MIB"),
        )

    def test_the_positions_that_stay_on_the_demand_branch_delete_nothing(self):
        self.assertEqual(self._form("bs8").unpriced_reserve_mib, 0)
        self.assertEqual(self._form("bs1-ctx1m").unpriced_reserve_mib, 0)

    def test_the_price_names_the_reserve_the_fitter_and_the_class_two_gate(self):
        line = self._form("bs1-slots").line
        self.assertIn("prefill activation reserve", line)
        self.assertIn(
            str(launcher.engine_int_constant("MAMBA_AUTO_ACTIVATION_RESERVE_MIB")),
            line,
        )
        self.assertIn("#307", line)
        self.assertIn("CLASS-II", line)

    def test_the_engine_branches_are_still_asymmetric(self):
        """The premise of the price, pinned against the engine's source.

        Characterization, not red-first: the asymmetry exists today.  It is
        pinned so that FIXING it -- subtracting the reserve on the explicit
        branch too -- goes red here and forces the price string to be
        corrected, instead of leaving the launcher charging for a post the
        engine has started paying.
        """
        src = open(
            launcher._ENGINE_KV_MIXIN_PATH, encoding="utf-8"
        ).read()
        head, _, rest = src.partition(
            "        if server_args.max_mamba_cache_size is not None:"
        )
        self.assertTrue(rest, "the explicit mamba branch moved")
        explicit, sep, demand = rest.partition(
            "        elif self._auto_mamba_demand_active():"
        )
        self.assertTrue(sep, "the demand-driven mamba branch moved")
        demand_body = demand.split("\n        elif ")[0]
        self.assertNotIn(
            '_note_mamba_component(self, "prefill activation reserve"', explicit
        )
        self.assertIn(
            '_note_mamba_component(self, "prefill activation reserve"', demand_body
        )
        # ... and the witness this slice added, so the ratio stays readable on
        # the branch that lost [auto-mamba].
        self.assertIn("[explicit-mamba]", explicit)


class TheDefaultLineSaysFloorNotPrediction(CustomTestCase):
    """Round-2 review finding 5: ``cap_tokens`` is not the boot's own cap."""

    def test_the_default_cap_is_the_flag_floor_and_says_so(self):
        form = launcher.resolve_d_kv_form(
            "bs8",
            d_bs_default=8,
            max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
            extra_tokens=EXTRA,
        )
        self.assertEqual(form.cap_tokens, 8 * (262144 + EXTRA))
        self.assertIn("FLOOR", form.line)
        # the number every boot of record actually printed, and the sizing that
        # produces it (weg2sb5e D:351-353 -> 40 slots at ratio 4)
        self.assertIn("2,621,510", form.line)
        self.assertIn("40 slots", form.line)

    def test_every_position_names_the_pinned_ratio_as_pinned(self):
        for name in launcher.D_KV_FORM_CHOICES:
            form = launcher.resolve_d_kv_form(
                name,
                d_bs_default=8,
                max_kv_per_request=launcher.CONTEXT_LENGTH_TOKENS,
                extra_tokens=EXTRA,
            )
            with self.subTest(name=name):
                self.assertIn("PINNED", form.line)
                self.assertIn("mamba_ratio=%d" % launcher.D_MAMBA_RATIO, form.line)


if __name__ == "__main__":
    unittest.main()
