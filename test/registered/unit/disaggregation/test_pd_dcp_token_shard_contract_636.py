# SPDX-License-Identifier: Apache-2.0
"""The DCP token-sharded PD handover, as ONE boot-time contract (#636).

A PD decode server whose KV pool is token-sharded can only receive a transfer
on a narrow combination: ``page_size == 1``, hisparse off, the mooncake
transport, on the uneven-TP replicated-KV layout. All four were already
enforced -- and all four only at RUNTIME. Three fire in
``disaggregation/decode.py`` on the first ``send_metadata``; the transport one
fires from inside the SENDER, later still. So the server boots healthy,
reports 200, accepts traffic, and dies on the FIRST handover, after the weight
load and in production.

The gate decides the whole combination at parse time, from ServerArgs alone.
These tests therefore check two different things, and the second is the one a
future refactor breaks quietly:

1. the contract itself -- each precondition refuses, and the supported
   combination passes;
2. that the gate RUNS, and runs LATE ENOUGH. ``dcp_size`` is auto-set to
   ``tp_size`` by ``_handle_uneven_tp``, which executes AFTER
   ``_handle_pd_disaggregation`` in the same ``__post_init__``. A gate placed
   in the PD hook reads the unresolved value, decides the token-shard path is
   not in force, and passes silently for exactly the configuration it exists
   to protect. #108 hit this same trap and records a measured false refusal
   from it, which is why the ordering is pinned here rather than trusted.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    validate_pd_dcp_token_shard_contract,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _supported(**over):
    """The one combination the handover supports, as a stub."""
    base = dict(
        disaggregation_mode="decode",
        dcp_size=3,
        tp_size=3,
        page_size=1,
        enable_hisparse=False,
        disaggregation_transfer_backend="mooncake",
        # The token-sharded pool exists only under an installed --rank-tp-ratio
        # plan: uneven_dcp_kv_replicated() is `dcp_size > 1 and
        # get_tp_partition_ratios() is not None`. Without it the pool is stock
        # head-sharded DCP and the receive is unimplemented.
        rank_tp_ratio=[2, 1, 1],
    )
    base.update(over)
    return SimpleNamespace(**base)


class DcpTokenShardContractTest(CustomTestCase):
    def test_supported_combination_passes(self):
        validate_pd_dcp_token_shard_contract(_supported())

    def test_each_precondition_refuses_and_names_itself(self):
        """One contract, but the refusal must say WHICH part failed."""
        for over, needle in (
            ({"page_size": 64}, "page_size must be 1"),
            ({"enable_hisparse": True}, "--enable-hisparse must be off"),
            (
                {"disaggregation_transfer_backend": "nixl"},
                "--disaggregation-transfer-backend must be one of",
            ),
            (
                {"disaggregation_transfer_backend": "mori"},
                "--disaggregation-transfer-backend must be one of",
            ),
        ):
            with self.subTest(over=over):
                with self.assertRaises(ValueError) as ctx:
                    validate_pd_dcp_token_shard_contract(_supported(**over))
                self.assertIn(needle, str(ctx.exception))

    def test_several_violations_are_reported_together(self):
        """Fixing one at a time across four reboots is the thing to avoid."""
        with self.assertRaises(ValueError) as ctx:
            validate_pd_dcp_token_shard_contract(
                _supported(
                    page_size=64,
                    enable_hisparse=True,
                    disaggregation_transfer_backend="nixl",
                )
            )
        msg = str(ctx.exception)
        self.assertIn("page_size must be 1", msg)
        self.assertIn("--enable-hisparse must be off", msg)
        self.assertIn("--disaggregation-transfer-backend must be one of", msg)

    def test_inert_off_the_token_sharded_layout(self):
        """Not this layout -> not this contract. Must not refuse other setups.

        ``dcp_size == 1`` is the ONLY genuinely inert DCP shape: with no DCP
        group there is no owner rule, no compact rows, and nothing this
        contract governs. The case that used to sit beside it here --
        ``dcp_size=2, tp_size=3`` -- is NOT inert and is now pinned as a
        refusal below; see
        ``test_head_sharded_dcp_decode_arm_is_refused_at_boot``.
        """
        for over in ({"dcp_size": 1},):
            with self.subTest(over=over):
                validate_pd_dcp_token_shard_contract(
                    _supported(page_size=64, enable_hisparse=True, **over)
                )

    def test_head_sharded_dcp_decode_arm_is_refused_at_boot(self):
        """#636 P2, the precondition the boot gate used to wave through.

        ``uneven_dcp_owner_bounds()`` (``distributed/utils.py:421``) returns
        non-None iff ``uneven_dcp_kv_replicated(dcp_size)`` -- which is
        ``dcp_size > 1 and get_tp_partition_ratios() is not None``
        (``utils.py:346-354``). So a DCP arm WITHOUT an installed
        ``--rank-tp-ratio`` plan takes the ``elif get_parallel().dcp_enabled``
        branch at ``decode.py:1153-1159`` and raises "Stock head-sharded DCP
        receive is not supported" -- on the FIRST handover, after the weight
        load, in production. That is verbatim the lateness this gate exists to
        remove, and it was the one of the four preconditions the gate's own
        applicability test skipped over.
        """
        with self.assertRaises(ValueError) as ctx:
            validate_pd_dcp_token_shard_contract(_supported(rank_tp_ratio=None))
        msg = str(ctx.exception)
        self.assertIn("--rank-tp-ratio", msg)
        self.assertIn("head-sharded", msg)

    def test_the_owner_rule_does_not_require_dcp_equal_tp(self):
        """The gate's applicability must mirror the RUNTIME predicate.

        The old gate applied only when ``dcp_size == tp_size``, but the owner
        rule needs no such equality -- an installed ratio plan and
        ``dcp_size > 1`` are the whole condition. A ``dcp_size=2, tp_size=3``
        arm therefore DOES build compact rows, and its page_size/hisparse/
        transport preconditions are live rather than moot.
        """
        with self.assertRaises(ValueError) as ctx:
            validate_pd_dcp_token_shard_contract(
                _supported(dcp_size=2, tp_size=3, page_size=64)
            )
        self.assertIn("page_size must be 1", str(ctx.exception))

    def test_inert_for_a_monolithic_server(self):
        """The standing production boot is monolithic and must be untouched."""
        validate_pd_dcp_token_shard_contract(
            _supported(disaggregation_mode="null", page_size=64)
        )


class GateRunsAfterDcpResolutionTest(CustomTestCase):
    """Mechanism reach: the gate is called, and called after dcp_size resolves.

    Read from ``__post_init__``'s source. Constructing a ServerArgs that
    actually engages the auto-set needs more of the boot than a desk test
    should carry, and the property at risk is an ORDERING between two
    statements -- which is exactly what the source states.
    """

    def _post_init_source(self):
        import inspect

        from sglang.srt.server_args import ServerArgs

        return inspect.getsource(ServerArgs.__post_init__)

    def test_gate_is_called_from_post_init(self):
        self.assertIn(
            "self._validate_pd_dcp_token_shard_contract()",
            self._post_init_source(),
            "the #636 gate is not called at all",
        )

    def test_gate_runs_after_uneven_tp_resolves_dcp_size(self):
        src = self._post_init_source()
        uneven_tp = src.index("self._handle_uneven_tp()")
        gate = src.index("self._validate_pd_dcp_token_shard_contract()")
        self.assertLess(
            uneven_tp,
            gate,
            "the gate must run AFTER _handle_uneven_tp, which auto-sets "
            "dcp_size = tp_size. Before it, dcp_size is unresolved and the "
            "gate passes silently on exactly the layout it guards",
        )

    def test_gate_runs_after_the_pd_hook(self):
        """The hook rewrites mooncake_tcp -> mooncake; read it before that
        and the transport check refuses a supported launch."""
        src = self._post_init_source()
        self.assertLess(
            src.index("self._handle_pd_disaggregation()"),
            src.index("self._validate_pd_dcp_token_shard_contract()"),
        )


if __name__ == "__main__":
    unittest.main()
