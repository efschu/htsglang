# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for --rank-role, the flag that makes a zero a layout.

No CUDA, no model, no device. `ServerArgs.__post_init__` needs a real
checkpoint and an accelerator, so the validation METHOD is exercised
directly on a bare instance carrying only the fields it reads -- which is
also what pins the method's real input surface.

What is pinned:

  * WITHOUT --rank-role the old rule stands, unchanged: a zero in
    --rank-tp-ratio is still an error, and its message says why a zero is
    not self-explaining;
  * WITH --rank-role the zero is admitted, but ONLY for a rank the vector
    names a worker -- the two must agree in both directions;
  * --rank-role alone (no --rank-tp-ratio) refuses and NAMES the vector it
    wants, rather than being silently inert. That inertness was real: the
    method has an early exit for "none of the rank flags is set", and the
    flag had to be added to its condition;
  * a symbolic --rank-tp-ratio ('auto') is refused under Form A, because
    'auto' maximises the KV pool across ALL ranks and Form A is the layout
    where two ranks hold no KV at all;
  * the classic paths -- no flags, and a plain uneven vector -- are
    untouched.
"""

import pytest

from sglang.srt.server_args import ServerArgs


def _args(**kw):
    """A bare ServerArgs carrying only what _handle_uneven_tp reads."""
    sa = ServerArgs.__new__(ServerArgs)
    sa.tp_size = 3
    sa.pp_size = 1
    sa.dp_size = 1
    sa.ep_size = 1
    sa.nnodes = 1
    sa.rank_gpu_id = None
    sa.rank_gpu_memory_mib = None
    sa.rank_role = None
    sa.rank_tp_ratio = None
    sa.rank_moe_resident_fraction = None
    sa.rank_auto_reserve_mib = ServerArgs.AUTO_RANK_MEMORY_RESERVE_MIB
    for k, v in kw.items():
        setattr(sa, k, v)
    return sa


FORM_A = ["host", "worker", "worker"]


def test_the_classic_paths_are_untouched():
    _args()._handle_uneven_tp()  # no rank flags at all
    _args(rank_tp_ratio=[39, 13, 12])._handle_uneven_tp()
    assert _args().form_a_active() is False


def test_a_zero_without_the_role_flag_is_still_an_error():
    with pytest.raises(ValueError) as e:
        _args(rank_tp_ratio=[1, 0, 0])._handle_uneven_tp()
    msg = str(e.value)
    assert "must be positive integers" in msg
    assert "arithmetic accident" in msg  # says WHY, not just that


def test_the_role_flag_admits_the_zero():
    sa = _args(rank_role=FORM_A, rank_tp_ratio=[1, 0, 0])
    sa._handle_uneven_tp()
    assert sa.form_a_active() is True
    plan = sa.form_a_role_plan()
    assert plan.host_rank == 0 and plan.worker_ranks == [1, 2]
    # any positive host weight works -- the ratio describes the partition,
    # and with one dense rank its magnitude is free
    _args(rank_role=FORM_A, rank_tp_ratio=[64, 0, 0])._handle_uneven_tp()


def test_role_and_ratio_must_agree():
    with pytest.raises(ValueError, match="gives them a dense share"):
        _args(rank_role=FORM_A, rank_tp_ratio=[39, 13, 12])._handle_uneven_tp()
    with pytest.raises(ValueError, match="nobody computes attention"):
        _args(rank_role=FORM_A, rank_tp_ratio=[0, 0, 0])._handle_uneven_tp()


def test_the_role_flag_alone_refuses_and_names_the_vector_it_wants():
    """It must not be silently inert: _handle_uneven_tp has an early exit
    for 'none of the rank flags is set', and --rank-role had to be added to
    that condition or the whole feature would have done nothing."""
    with pytest.raises(ValueError) as e:
        _args(rank_role=FORM_A)._handle_uneven_tp()
    msg = str(e.value)
    assert "requires an explicit --rank-tp-ratio" in msg
    assert "1,0,0" in msg  # hands the caller the answer


def test_a_symbolic_ratio_is_refused_under_form_a():
    with pytest.raises(ValueError, match="symbolic sizing mode"):
        _args(
            rank_role=FORM_A,
            rank_tp_ratio="auto",
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[29500, 17800, 17800],
        )._handle_uneven_tp()


def test_a_bad_role_vector_is_refused_through_the_flag():
    with pytest.raises(ValueError, match="exactly ONE attention host"):
        _args(
            rank_role=["host", "host", "worker"], rank_tp_ratio=[1, 1, 0]
        )._handle_uneven_tp()
    with pytest.raises(ValueError, match="one role per rank"):
        _args(
            rank_role=["host", "worker"], rank_tp_ratio=[1, 0, 0]
        )._handle_uneven_tp()


def test_the_flag_parser_and_the_runtime_share_one_definition():
    from sglang.srt.server_args import _parse_rank_role

    assert _parse_rank_role("host, worker ,WORKER") == FORM_A
    with pytest.raises(Exception, match="must be one of"):
        _parse_rank_role("host,driver,worker")


# ==========================================================================
# Form A x the DRAFT (slice 6a): the boot line's one new speculative flag
# ==========================================================================
def test_form_a_requires_the_solo_draft_and_says_why():
    """The refusal that keeps a SECOND model from hanging the rig.

    'split' placement runs a sharded draft forward on every rank, with its
    own per-layer collectives and -- for an MTP draft -- its own MoE
    combine. A Form A worker holds no draft weights at all (the loader veto
    rejects every `mtp.*` name), so the host would block in the draft's
    collectives exactly the way seam F12 describes for the target model.
    Same load-bearing condition as --weightless-kv-fastlane's, one lane
    over."""
    with pytest.raises(ValueError, match="draft-placement solo"):
        _args(
            rank_role=FORM_A,
            rank_tp_ratio=[1, 0, 0],
            speculative_algorithm="NEXTN",
            speculative_draft_placement="split",
        )._handle_uneven_tp()


def test_form_a_with_the_solo_draft_is_accepted():
    sa = _args(
        rank_role=FORM_A,
        rank_tp_ratio=[1, 0, 0],
        speculative_algorithm="NEXTN",
        speculative_draft_placement="solo",
    )
    sa._handle_uneven_tp()
    assert sa.form_a_active() is True


def test_form_a_without_speculation_needs_no_placement_flag():
    """The refusal is scoped to speculative boots. A Form A boot with no
    draft at all must not be told to configure one."""
    _args(
        rank_role=FORM_A,
        rank_tp_ratio=[1, 0, 0],
        speculative_algorithm=None,
        speculative_draft_placement="split",
    )._handle_uneven_tp()
