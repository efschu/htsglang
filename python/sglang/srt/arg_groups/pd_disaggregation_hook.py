from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


#: The transports that implement the DCP token-sharded receive. nixl and mori
#: raise NotImplementedError from inside the SENDER's send_metadata, i.e. even
#: later than the decode-side checks this gate replaces.
_DCP_TOKEN_SHARD_TRANSPORTS = ("mooncake",)


def validate_pd_dcp_token_shard_contract(server_args: ServerArgs) -> None:
    """BOOT GATE (#636): the DCP token-sharded PD handover, as ONE contract.

    A PD decode server whose KV pool is token-sharded (the fork's uneven-TP
    replicated-KV DCP path) can only receive a transfer on a narrow
    combination. All four conditions were enforced, and enforced well, but
    every one of them at RUNTIME: three in ``disaggregation/decode.py``
    (:1135-1159) on the first ``send_metadata``, and the transport one from
    the receiver the decode arm constructs (``NixlKVReceiver``,
    ``nixl/conn.py:2516``; ``MoriKVReceiver``, ``mori/conn.py:1704``, reached
    from ``decode.py:1249``), later still. The server therefore boots healthy,
    reports 200, accepts traffic, and dies on the FIRST handover -- after the
    full weight load, and in production rather than at the operator's
    keyboard.

    THE FOUR, and where each is decided here:

    ==  =============================  ==========================
    P1  ``page_size == 1``             below, with the owner-rule reason
    P2  the uneven-TP replicated-KV    the ``rank_tp_ratio is None``
        layout (``--rank-tp-ratio``)   refusal below
    P3  the mooncake transport         below
    P4  ``--enable-hisparse`` off      below
    ==  =============================  ==========================

    P2 was the one this gate originally did not decide: it sat in the
    APPLICABILITY test rather than among the violations, so an arm that failed
    it was waved through instead of refused (#636's open leg). Its runtime
    raise is the ``elif get_parallel().dcp_enabled`` branch at
    ``decode.py:1153``.

    That lateness is the defect this gate removes; four loud failures in four
    files is not the problem, four loud failures that all arrive too late is.
    Every input is decidable from ServerArgs alone, so the whole combination
    is decided here instead.

    ORDERING, which is the reason this is not folded into
    ``handle_pd_disaggregation``: ``dcp_size`` is auto-set to ``tp_size`` by
    ``_handle_uneven_tp`` (``server_args.py``, the "Uneven DCP: auto-set"
    branch), which runs AFTER ``_handle_pd_disaggregation`` in the same
    ``__post_init__``. Reading it in the hook would see the unresolved value,
    conclude the token-shard path is not in force, and pass silently for
    exactly the configuration this gate exists to protect -- a gate that never
    fires, which is worse than none because it reads as coverage. #108 hit
    this precise trap and records a MEASURED false refusal from it
    (``_reject_unsupported_draft_kv_dcp``); this gate follows that precedent
    and runs after resolution.
    """
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return
    # No DCP group -> no owner rule, no compact rows, nothing this contract
    # governs. The ONLY genuinely inert DCP shape.
    if server_args.dcp_size <= 1:
        return

    # PRECONDITION 2, and the reason this predicate is not `dcp == tp`.
    #
    # The runtime decides on ``uneven_dcp_owner_bounds()``
    # (``distributed/utils.py:421``), which is non-None iff
    # ``uneven_dcp_kv_replicated(dcp_size)`` -- and that is
    # ``dcp_size > 1 and get_tp_partition_ratios() is not None``
    # (``utils.py:346-354``), with NO equality between dcp_size and tp_size in
    # it. So the equality this gate used to test was both too narrow (it waved
    # a dcp != tp arm through P1/P3/P4, which the runtime does enforce there)
    # and blind to P2 itself: an arm with DCP but no installed --rank-tp-ratio
    # plan fell out of the gate entirely and died at ``decode.py:1153-1159``
    # on the first handover. Mirror the runtime predicate instead.
    if server_args.rank_tp_ratio is None:
        raise ValueError(
            "PD disaggregation on a DCP decode server "
            f"(dcp_size={server_args.dcp_size}) requires the uneven-TP "
            "replicated-KV layout, i.e. an installed --rank-tp-ratio plan. "
            "Stock head-sharded DCP receive is not supported: the sender can "
            "only filter its source rows by the receiver's OWNED ORDINALS, "
            "which exist only under the owner rule that --rank-tp-ratio "
            "installs.\n\n"
            "Pass --rank-tp-ratio (the uneven-TP replicated-KV layout), or "
            "run this arm without DCP (--dcp-size 1). Refused at boot on "
            "purpose: without this gate the server loads its weights, reports "
            "200, accepts traffic and raises on the FIRST KV handover."
        )

    violations = []
    if server_args.page_size != 1:
        violations.append(
            f"  - page_size must be 1, got {server_args.page_size}. The owner "
            "rule maps a GLOBAL slot to a compact per-rank row per TOKEN; a "
            "paged pool has no such per-token row."
        )
    if server_args.enable_hisparse:
        violations.append(
            "  - --enable-hisparse must be off: hisparse and the "
            "token-sharded receive have no combined implementation."
        )
    if server_args.disaggregation_transfer_backend not in _DCP_TOKEN_SHARD_TRANSPORTS:
        violations.append(
            f"  - --disaggregation-transfer-backend must be one of "
            f"{list(_DCP_TOKEN_SHARD_TRANSPORTS)}, got "
            f"'{server_args.disaggregation_transfer_backend}'. Only that "
            "backend filters the sender's source rows by the receiver's "
            "owned ordinals; the others raise from inside send_metadata."
        )
    if not violations:
        return

    raise ValueError(
        "PD disaggregation on a DCP token-sharded KV pool "
        f"(dcp_size={server_args.dcp_size}, tp_size={server_args.tp_size}, "
        "--rank-tp-ratio installed) is only supported on one combination, and "
        "this launch is outside it:\n"
        + "\n".join(violations)
        + "\n\nSupported combination: page_size == 1, --enable-hisparse off, "
        "--disaggregation-transfer-backend mooncake, on the uneven-TP "
        "replicated-KV layout (--rank-tp-ratio installed, dcp_size > 1). "
        "Refused at boot on purpose: each of these was previously discovered "
        "on the first KV handover, i.e. after the weight load and in "
        "production."
    )


def validate_pd_draft_kv_layout(server_args: ServerArgs) -> None:
    """BOOT GATE (#642): a token-sharded PD arm needs ``--draft-kv-layout dcp``.

    The draft KV pool rides the main transfer as extra layers, addressed by the
    SAME index array as the target pool -- ``prefill.py:186-195`` /
    ``decode.py:439-448``, "The indices are always shared with a target
    model", and ``mooncake/conn.py`` contains no occurrence of "draft" at all,
    so the transport cannot treat them differently even in principle.

    On a DCP decode arm those indices have been rewritten into COMPACT
    owner-rule rows, ``(L // S) * (hi - lo) + (L % S - lo)``
    (``decode.py:1119-1125``). Whether that is the right address for the draft
    pool depends entirely on ``--draft-kv-layout``:

    ``dcp``
        the draft pool takes ``dcp_compact_pool_rows`` -- the target's own
        coordinate system -- and the shared indices are correct BY
        CONSTRUCTION.
    ``replicated`` (the default)
        the draft pool keeps "the full global context per rank"
        (``model_runner_kv_cache_mixin.py:2819-2823``). Compact rows are then
        addresses in a DIFFERENT coordinate system: token L's draft KV is
        written at row ``compact(L)``, which in a full-context pool is where
        a different token lives. The error grows with the slot id and there is
        no crash -- #345's right-token/wrong-slot class.

    WHY THIS IS NOT REACHABLE TODAY, stated precisely because a latent guard
    that quietly becomes dead code is its own defect. #631a refuses a PD arm
    that asks for speculation (default) and its ``SGLANG_PD_AUTO_DISABLE_SPEC``
    escape hatch disables speculation instead; NEITHER branch lets a draft
    worker exist, so the registration above never runs. It is "no branch
    permits it" rather than "the refusal nulls it" -- the two could diverge,
    and someone hardening only the escape hatch would not realise they were
    also holding a corruption path shut. This gate is what makes lifting #631a
    safe, and it is deliberately in the tree BEFORE that lift rather than
    bundled into it.

    Kept separate from :func:`validate_pd_dcp_token_shard_contract` although
    both run at the same point: that one is about whether the MAIN handover
    can run at all, this one about the DRAFT pool's addressing. One refusal
    text per hazard reads better than a combined one at the moment an operator
    hits it.
    """
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return
    if server_args.speculative_algorithm is None:
        # No draft worker, no draft pool, no registration, no hazard.
        return
    if not (server_args.dcp_size > 1 and server_args.dcp_size == server_args.tp_size):
        return
    if server_args.draft_kv_layout == "dcp":
        return

    raise ValueError(
        f"--draft-kv-layout '{server_args.draft_kv_layout}' cannot be used on "
        f"the '{server_args.disaggregation_mode}' arm of a PD pair whose KV "
        f"pool is token-sharded (dcp_size={server_args.dcp_size} == "
        f"tp_size={server_args.tp_size}).\n\n"
        "The draft KV pool is transferred as extra layers addressed by the "
        "SAME index array as the target pool, and on this arm those indices "
        "are compact owner-rule rows. Only --draft-kv-layout dcp gives the "
        "draft pool the matching compact row space; 'replicated' keeps the "
        "full global context per rank, so every draft row would be written "
        "at the address of a different token -- silently, with the error "
        "growing as slot ids grow.\n\n"
        "Use --draft-kv-layout dcp (its own gate additionally requires a "
        "linear MTP/NEXTN draft with topk == 1 and one draft KV layer), or "
        "run this arm without speculation."
    )


def validate_pd_speculation(server_args: ServerArgs) -> None:
    """BOOT GATE (#631b): speculation on a PD arm, admitted where the draft
    pool's ROW SPACE is congruent across the PD boundary.

    #631a refused ``--speculative-algorithm`` on either arm unconditionally.
    Its reason was specific, not general: "the MTP/EAGLE draft KV pool is
    uneven-head-sharded (not DCP token-sharded), so its transfer would need
    general uneven head reslicing". That is a statement about ONE layout. The
    refusal was written as a statement about all of them, and the blanket is
    what this gate replaces.

    The draft pool is not transferred on its own. It rides the main transfer as
    extra layers addressed by the SAME index array as the target pool
    (``prefill.py:186-195`` / ``decode.py:439-448``, "The indices are always
    shared with a target model"; ``mooncake/conn.py`` contains no occurrence of
    "draft" at all, so the transport cannot address it separately even in
    principle). So the whole question is a single one: do the draft rows and
    the target rows live in the SAME coordinate system on both arms? Where they
    do, the shared index array is already correct and there is nothing to
    reslice. Where they do not, rows land at other tokens' addresses.

    Two shapes answer yes, and they are the two admitted here:

    ``tp_size == 1``
        Nothing is sharded on this arm. Draft rows and target rows are both
        global token ids. This is the trivially congruent case and it was
        swept up by the blanket purely because the blanket did not look.

    ``dcp_size == tp_size > 1`` with ``--draft-kv-layout dcp``
        The token-sharded path. The draft pool takes ``dcp_compact_pool_rows``
        -- the target's OWN coordinate system -- so the shared indices are
        correct by construction. This is not an inference: it is what
        ``--draft-kv-layout``'s own help text calls the "foundation for
        speculation in PD-disaggregation", and what
        :func:`validate_pd_draft_kv_layout` (#642) was pre-positioned to make
        safe, in the tree BEFORE this lift and deliberately not bundled into
        it.

    Everything else still refuses, and now refuses for the reason #631a
    actually had: a head-sharded draft pool on a multi-rank arm, whose rows
    would need general uneven head reslicing to cross the boundary.

    ORDERING: this reads ``dcp_size``, ``draft_kv_layout`` and the resolved
    ``speculative_algorithm``, so it MUST run after ``_handle_uneven_tp``, next
    to the #636 and #642 gates -- not in ``handle_pd_disaggregation``, which
    runs before ``dcp_size`` is auto-set to ``tp_size`` and would therefore
    read 1 for exactly the token-sharded configuration this gate admits. That
    trap has a measured precedent in this file's sibling gates; the refusal
    lived on the early side, which is why it could only ever be a blanket.
    """
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return
    if server_args.speculative_algorithm is None:
        return

    # The pre-#631a behaviour, kept for shared launch configs that feed one
    # flagset to both a PD and a non-PD server. Unchanged in meaning; it just
    # moved here with the rest of the decision so there is exactly one place
    # that decides whether a PD arm speculates.
    if envs.SGLANG_PD_AUTO_DISABLE_SPEC.get():
        logger.warning(
            "PD disaggregation (%s arm): speculative decoding "
            "(--speculative-algorithm %s) has been DISABLED for this server "
            "because SGLANG_PD_AUTO_DISABLE_SPEC=1. This server will decode "
            "WITHOUT speculation.",
            server_args.disaggregation_mode,
            server_args.speculative_algorithm,
        )
        server_args.speculative_algorithm = None
        server_args.speculative_draft_model_path = None
        return

    # Congruent shape 1: nothing is sharded on this arm.
    if server_args.tp_size == 1:
        logger.info(
            "PD disaggregation (%s arm): speculation admitted -- tp_size=1, "
            "so the draft and target pools are both addressed by global token "
            "ids and the shared index array needs no reslicing (#631b).",
            server_args.disaggregation_mode,
        )
        return

    # Congruent shape 2: the token-sharded path, draft in the target's own
    # compact owner-rule row space. draft_kv_layout != 'dcp' on such an arm is
    # NOT accepted silently here -- validate_pd_draft_kv_layout (#642) refuses
    # it by name, with the addressing argument spelled out. Keeping that
    # refusal there rather than duplicating it keeps one hazard to one text.
    if server_args.dcp_size > 1 and server_args.dcp_size == server_args.tp_size:
        logger.info(
            "PD disaggregation (%s arm): speculation admitted on the token-"
            "sharded pool (dcp_size=%d == tp_size=%d, --draft-kv-layout "
            "'%s'); with layout 'dcp' the draft pool uses the target's "
            "compact owner-rule rows and the shared index array is correct by "
            "construction (#631b).",
            server_args.disaggregation_mode,
            server_args.dcp_size,
            server_args.tp_size,
            server_args.draft_kv_layout,
        )
        return

    raise ValueError(
        f"--speculative-algorithm {server_args.speculative_algorithm} cannot "
        f"be honoured on the '{server_args.disaggregation_mode}' arm of a PD "
        f"pair with tp_size={server_args.tp_size}, "
        f"dcp_size={server_args.dcp_size}, "
        f"--draft-kv-layout '{server_args.draft_kv_layout}'.\n\n"
        "The draft KV pool rides the main transfer as extra layers addressed "
        "by the SAME index array as the target pool. On this arm the draft "
        "pool is head-sharded across ranks while the indices being shared are "
        "the target's, so crossing the PD boundary would need general uneven "
        "head reslicing, which is not implemented.\n\n"
        "Two shapes ARE supported on a PD arm:\n"
        "  - tp_size == 1: nothing is sharded, rows are global token ids.\n"
        "  - dcp_size == tp_size > 1 with --draft-kv-layout dcp: the draft "
        "pool takes the target's compact owner-rule rows (this additionally "
        "requires a linear MTP/NEXTN draft with --speculative-eagle-topk 1, "
        "a non-uniform --rank-tp-ratio and the weighted owner rule).\n\n"
        "Otherwise launch this arm without --speculative-algorithm, or run a "
        "monolithic server without --disaggregation-mode. To disable "
        "speculation automatically instead of refusing, set "
        "SGLANG_PD_AUTO_DISABLE_SPEC=1."
    )


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

    # Speculation on a PD arm is NOT decided here. It used to be (#99's
    # auto-disable, then #631a's blanket refusal), and that placement is
    # precisely why it could only ever be a blanket: this hook runs BEFORE
    # _handle_uneven_tp resolves dcp_size and the speculative defaults, so the
    # one distinction that matters -- is the draft pool's row space congruent
    # with the target's across the boundary? -- is not yet answerable here.
    # The decision moved WHOLE (refusal, admission and the
    # SGLANG_PD_AUTO_DISABLE_SPEC escape hatch) to validate_pd_speculation
    # (#631b), which runs after resolution alongside the #636 and #642 gates.
    # Deliberately not split across the two points: two places that can each
    # turn speculation off is how the escape hatch and the refusal drift apart.

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
