from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if server_args.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        server_args.disaggregation_transfer_backend = "mooncake"
        server_args.disaggregation_ib_device = None
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    # Single-node hetero PD v1 (#99): speculative decoding is not supported in
    # disaggregation mode on this fork -- the MTP/EAGLE draft KV pool is
    # uneven-head-sharded (not DCP token-sharded), so its transfer would need
    # general uneven head reslicing.
    #
    # #631a turned this from an auto-disable into a REFUSAL. The auto-disable
    # was silent in the only way that matters: a decode arm launched with
    # --speculative-algorithm NEXTN came up WITHOUT it, produced correct
    # output, and was merely slower. Nothing downstream can tell that apart
    # from a slow rig, so the loss of the decode optimum survives every smoke
    # test. A configuration that cannot be honoured is refused by name
    # instead. SGLANG_PD_AUTO_DISABLE_SPEC=1 restores the old behaviour for
    # shared launch configs that feed one flagset to both a PD and a non-PD
    # server, which was the original ruling's reason.
    if (
        server_args.disaggregation_mode in ("prefill", "decode")
        and server_args.speculative_algorithm is not None
    ):
        if envs.SGLANG_PD_AUTO_DISABLE_SPEC.get():
            logger.warning(
                "PD disaggregation (%s arm): speculative decoding "
                "(--speculative-algorithm %s) is not supported in "
                "disaggregation mode on this fork and has been DISABLED for "
                "this server, because SGLANG_PD_AUTO_DISABLE_SPEC=1. This "
                "server will decode WITHOUT speculation.",
                server_args.disaggregation_mode,
                server_args.speculative_algorithm,
            )
            server_args.speculative_algorithm = None
            server_args.speculative_draft_model_path = None
        else:
            raise ValueError(
                f"--speculative-algorithm "
                f"{server_args.speculative_algorithm} cannot be honoured on "
                f"the '{server_args.disaggregation_mode}' arm of a PD pair: "
                "speculative decoding is not supported in disaggregation "
                "mode on this fork, because the MTP/EAGLE draft KV pool is "
                "uneven-head-sharded (not DCP token-sharded) and "
                "transferring it across the PD boundary would need general "
                "uneven head reslicing. This is refused rather than "
                "auto-disabled: a server that quietly drops speculation "
                "still answers correctly and only runs slower, so the loss "
                "would not surface in any smoke test. Either launch this arm "
                "WITHOUT --speculative-algorithm (and accept no speculation "
                "on it), or run a monolithic server without "
                "--disaggregation-mode to keep speculation. To restore the "
                "old auto-disable for a shared launch config, set "
                "SGLANG_PD_AUTO_DISABLE_SPEC=1."
            )

    if server_args.disaggregation_mode == "decode":
        if server_args.disaggregation_decode_enable_radix_cache:
            if server_args.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if server_args.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {server_args.speculative_algorithm})"
                )
            from sglang.srt.arg_groups.overrides import resolved_view

            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            server_args.disable_radix_cache = False
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            server_args.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if server_args.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if server_args.max_running_requests is not None:
                per_worker = server_args.max_running_requests // max(
                    1, server_args.dp_size
                )
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            server_args.disaggregation_decode_extra_slots = extra_slots

    elif server_args.disaggregation_mode == "prefill":
        assert (
            server_args.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

    if server_args.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and server_args.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{server_args.disaggregation_transfer_backend}'."
            )

    # Free PD topology choice (#107). Without --disaggregation-topology this
    # returns before touching anything (default placement byte-identical);
    # with it, the topology is validated, probe-gated and normalized, and an
    # infeasible per-card VRAM sum rejects the launch here instead of
    # surfacing as a runtime OOM.
    from sglang.srt.disaggregation.topology import apply_pd_topology

    apply_pd_topology(server_args)
