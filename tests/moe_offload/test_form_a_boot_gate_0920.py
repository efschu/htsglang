# SPDX-License-Identifier: Apache-2.0
"""'RAENGE NIE UNEINS' as a boot-path refusal (slice 6a).

Hermetic: the all-gather is injected as a list, so the function that decides
whether the rig starts is runnable without a process group.

The law is CRASH/STOP on disagreement, never a hang. Form A makes it
load-bearing: the ranks no longer run the same forward, so "they agree about
the collectives" stops holding by construction and becomes a claim to check.
"""

import pytest

from sglang.srt.form_a_boot_gate import (
    FormARanksDisagree,
    _encode,
    assert_ranks_agree,
    declare_layer_collectives,
    gate_form_a_boot,
)
from sglang.srt.rank_role import RankRolePlan

FORM_A = RankRolePlan(("host", "worker", "worker"))


def _decls(wsd, hdu, hme=False, attn=False):
    return [
        declare_layer_collectives(
            FORM_A,
            r,
            is_attention_layer=attn,
            worker_skips_dense=wsd,
            host_dense_is_unsharded=hdu,
            host_uses_moe_exchange=hme,
        )
        for r in range(3)
    ]


def _gather_of(decls):
    blobs = [_encode(d) for d in decls]
    return lambda _mine: blobs


def test_the_confirmed_boot_target_passes_the_gate():
    """The simple Form A: worker skips dense, host unsharded, MoE keeps its
    all-reduce. One collective per layer, and all three ranks issue it."""
    for attn in (False, True):
        d = _decls(True, True, attn=attn)
        assert_ranks_agree(FORM_A, 0, d[0], _gather_of(d))
        assert [op.kind for op in d[0]] == ["all_reduce"]
        assert [op.kind for op in d[1]] == ["all_reduce"]


def test_slice_6a_without_F12_is_STOPPED_at_boot_not_hung():
    """The whole point. Without F12 the host still issues its dense
    collectives; the gate turns that from a silent three-card deadlock into
    a refusal with a stack and a log line."""
    d = _decls(True, False)
    with pytest.raises(FormARanksDisagree) as e:
        assert_ranks_agree(FORM_A, 0, d[0], _gather_of(d))
    msg = str(e.value)
    assert "RANKS DISAGREE" in msg
    assert "stopping the boot instead of hanging it" in msg
    assert "rank 0 (host)" in msg and "rank 1 (worker)" in msg


def test_a_missing_declaration_is_itself_a_disagreement():
    """A rank that never reached the gate must not be read as agreement."""
    d = _decls(True, True)
    short = [_encode(d[0]), _encode(d[1])]
    with pytest.raises(FormARanksDisagree, match="gathered 2 declarations"):
        assert_ranks_agree(FORM_A, 0, d[0], lambda _m: short)


def test_the_gate_checks_BOTH_layer_shapes():
    """36 linear_attn layers and 12 attention layers issue different
    sequences; a gate that checked only one shape would pass a layout that
    hangs on the other."""
    gate_form_a_boot(
        FORM_A,
        0,
        _gather_of(_decls(True, True)),
        worker_skips_dense=True,
        host_dense_is_unsharded=True,
        host_uses_moe_exchange=False,
    )
    # ... and the attention shape alone is what catches this one:
    d_attn = _decls(True, False, attn=True)
    with pytest.raises(FormARanksDisagree):
        assert_ranks_agree(FORM_A, 0, d_attn[0], _gather_of(d_attn))


def test_the_gate_is_free_on_a_classic_boot():
    """plan None -> every rank runs the same code and the property holds by
    construction, so the gate must cost nothing and never raise."""
    def _explode(_mine):
        raise AssertionError("the gate gathered on a classic boot")

    gate_form_a_boot(
        None,
        0,
        _explode,
        worker_skips_dense=True,
        host_dense_is_unsharded=False,
        host_uses_moe_exchange=True,
    )


def test_the_gate_and_the_desk_probe_cannot_drift_apart():
    """Both are built from the same trace_layer. Two spellings of 'which
    collectives' is how a gate comes to pass while the boot hangs."""
    import inspect

    from sglang.srt import form_a_boot_gate

    assert "trace_layer" in inspect.getsource(form_a_boot_gate)
