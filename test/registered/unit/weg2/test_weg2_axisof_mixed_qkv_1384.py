# SPDX-License-Identifier: Apache-2.0
"""#1384 W68 -- the SIXTH shard-cut class in ``_axis_of``: MIXED_FUSED.

RED ON d61481b5d5, reproduced here hermetically (``CUDA_VISIBLE_DEVICES=""``,
no boot, no GPU): boot ``weg2zwerg2`` (BOOT7 Versuch 2,
``/spinning/gpu-arb/weg2/BOOT_weg2zwerg1_0913.md`` section "VERSUCH 2") drove
two clean flips of a Zwerg form (Qwen3.5-2B-shaped, kv=2, D=TP3) and hit
``W68 Weg2XchgPlanDisagree`` 78x on P and 78x on D for
``model.layers.11.qkv_proj.weight``: P holds ``(5120, 2048)`` as one block; D's
three TP ranks hold ``[(3072, 2048), (2048, 2048), (2048, 2048)]``, summing to
``7168 != 5120``.  ``_axis_of``'s docstring named exactly FIVE classes
(REPLICATED / ROWS / COLS / ROWS-padded / COLS-padded) and "no sixth" -- this
geometry is none of them, because it is not ONE axis: within the fused
Q|K|V row range, Q is ratio-partitioned unevenly across TP (#116's
kv-boundary-aware planner) while K and V are FULLY REPLICATED on every rank,
since ``kv_heads (2) < tp_size (3)`` puts ``attn_kv_replicated`` in play (#62,
and #1382's "not exclusive split" for the draft side of the same geometry).

**THE NUMBERS ARE NOT INVENTED.**  ``total_heads=16, total_kv=2, tp=3,
head_size=256, hidden_size=2048`` run through the REAL production functions
(``attn_kv_replicated``, ``attn_q_partition_units``, ``attn_q_partition_groups``,
``tp_partition_size`` -- the exact ones ``QKVParallelLinear.__init__``,
``layers/linear.py:1466-1541``, calls to size its own buffer) reproduce the
boot's shapes byte-for-byte: Q splits ``[2048, 1024, 1024]``, K=V=512 each,
whole ``4096 + 512 + 512 = 5120``, D rows
``[2048+512+512, 1024+512+512, 1024+512+512] = [3072, 2048, 2048]``.  See
``test_the_boot_numbers_come_from_the_real_partition_functions``, which
performs exactly that derivation and is the load-bearing proof that the other
tests' fixtures are not hand-picked to make the classifier happy.

**THE FIX DOES NOT GUESS AN AXIS FROM ROW-COUNT ARITHMETIC.**  A tolerant
width band was already tried and reverted once in this exact function (the
padded-cut comment above ``_axis_of`` in ``xchg_manifest.py``: "a first draft
accepted any surplus below tp_size*pad_unit and immediately swallowed a
THREE-ROW skew on qkv_proj"), and the ticket's own danger direction is the
same class: a wrongly classified axis compares the wrong bytes and can still
report MATCH, which is worse than the refusal it replaces. So the sixth class
fires only from DECLARED per-component boundaries
(``ManifestPiece.component_rows`` / ``ParamGeom.component_rows``), read
straight off ``QKVParallelLinear``'s own already-computed
``q_proj_shard_size``/``kv_proj_shard_size``/``v_proj_shard_size``
(``linear.py:1539-1541``) via ``weight_exchange_shadow._qkv_component_rows``
-- never re-derived in the join. A tensor with no declared components, an
internally inconsistent declaration, or a per-component disagreement still
raises the ORIGINAL W68 -- the guard stays sharp for genuine disagreement,
which the mutant-shaped tests below exercise directly.

kv >= tp (the pin for the 27B production model, kv=4) and non-fused tensors
must be BYTE-IDENTICAL to before this change: the new code is appended after
all five existing checks and only runs once every one of them has already
failed, so it is structurally unreachable for anything that used to resolve.
``test_kv_ge_tp_is_pinned_byte_identical_even_with_declared_components`` and
``test_non_fused_tensors_are_unaffected`` are the direct proof.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.distributed import utils as du  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

QKV_NAME = "model.layers.11.qkv_proj.weight"
HIDDEN = 2048


def _piece(name, rows, cols, *, component_rows=(), item=2, tag="weights_0",
           cls=None):
    return xm.ManifestPiece(
        param_name=name,
        tensor_class=cls or sh.tensor_class(name),
        rows_full=rows, cols_full=cols, itemsize=item,
        tag=tag, nbytes=rows * cols * item,
        component_rows=tuple(int(c) for c in component_rows),
    )


def _qkv_geometry(*, total_heads, total_kv, tp, head_size, v_head_size=None):
    """The REAL per-rank (q, k, v) row split for a REPLICATED-KV QKV tensor.

    Calls the exact functions ``QKVParallelLinear.__init__`` calls
    (``layers/linear.py:1466-1541``) under an installed uneven-TP plan, so the
    shapes this module's tests assert on are the tree's own arithmetic, not a
    second copy of it.
    """
    v_head_size = head_size if v_head_size is None else v_head_size
    du.set_tp_partition_ratios([1] * tp)
    try:
        assert du.attn_kv_replicated(tp, total_kv), (
            "fixture precondition: this helper is for the REPLICATED-KV "
            "geometry (kv < tp) only"
        )
        units = du.attn_q_partition_units(total_heads, total_kv, tp)
        groups = du.attn_q_partition_groups(total_kv, tp)
        q_rows = [
            du.tp_partition_size(total_heads, tp, r, units, groups=groups)
            * head_size
            for r in range(tp)
        ]
        kv_row = total_kv * head_size
        v_row = total_kv * v_head_size
    finally:
        du.set_tp_partition_ratios(None)
    whole_components = (sum(q_rows), kv_row, v_row)
    cut_components = [(q, kv_row, v_row) for q in q_rows]
    return whole_components, cut_components


def _boot_geometry():
    """The exact BOOT7/weg2zwerg2 numbers: P=5120, D=[3072,2048,2048]."""
    whole_components, cut_components = _qkv_geometry(
        total_heads=16, total_kv=2, tp=3, head_size=256)
    assert sum(whole_components) == 5120
    assert [sum(c) for c in cut_components] == [3072, 2048, 2048]
    return whole_components, cut_components


# ---------------------------------------------------------------------------
# 0. THE FIXTURE'S OWN PROOF -- not hand-picked numbers.
# ---------------------------------------------------------------------------


def test_the_boot_numbers_come_from_the_real_partition_functions():
    whole_components, cut_components = _boot_geometry()
    assert whole_components == (4096, 512, 512)
    assert cut_components == [(2048, 512, 512), (1024, 512, 512),
                              (1024, 512, 512)]


# ---------------------------------------------------------------------------
# 1. RED -- today's behaviour without declared components.
# ---------------------------------------------------------------------------


def test_w68_fires_today_for_the_boot_geometry_without_declared_components():
    """Reproduces the boot's W68 hermetically: no component_rows declared."""
    whole = _piece(QKV_NAME, 5120, HIDDEN)
    cut = [_piece(QKV_NAME, r, HIDDEN) for r in (3072, 2048, 2048)]
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm._axis_of(QKV_NAME, whole, cut)
    assert "W68" in str(exc.value)
    assert QKV_NAME in str(exc.value)
    assert "(5120, 2048)" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. GREEN -- the fix, with declared components.
# ---------------------------------------------------------------------------


def test_mixed_fused_classifies_the_boot_geometry_after_the_fix():
    whole_components, cut_components = _boot_geometry()
    whole = _piece(QKV_NAME, 5120, HIDDEN, component_rows=whole_components)
    cut = [
        _piece(QKV_NAME, sum(c), HIDDEN, component_rows=c)
        for c in cut_components
    ]
    axis, rows_full, cols_full, widths, pad = xm._axis_of(QKV_NAME, whole, cut)
    assert axis == wx.MIXED_FUSED
    assert (rows_full, cols_full) == (5120, HIDDEN)
    assert widths == (3072, 2048, 2048)
    assert pad == 0


def test_join_manifests_end_to_end_with_a_mixed_fused_tensor_amid_ordinary_ones():
    """The join, not just ``_axis_of`` in isolation -- ordinary tensors on the
    same boot must classify exactly as before, and the mixed one must not
    stop the whole join."""
    whole_components, cut_components = _boot_geometry()
    p_pieces = [
        _piece(QKV_NAME, 5120, HIDDEN, component_rows=whole_components),
        _piece("model.layers.11.input_layernorm.weight", 1, HIDDEN),
        _piece("model.layers.11.mlp.down_proj.weight", 4096, HIDDEN),
    ]
    d_pieces = [
        [
            _piece(QKV_NAME, sum(cut_components[r]), HIDDEN,
                   component_rows=cut_components[r]),
            _piece("model.layers.11.input_layernorm.weight", 1, HIDDEN),
            _piece("model.layers.11.mlp.down_proj.weight", w, HIDDEN),
        ]
        for r, w in zip(range(3), (1366, 1365, 1365))
    ]
    manifests = [
        xm.RankManifest(group="P", rank=0, card=0, region_tag="weights_0",
                        boot_token="b1", tp_rank=0, pp_rank=0,
                        pieces=tuple(p_pieces)),
    ] + [
        xm.RankManifest(group="D", rank=r, card=r, region_tag="weights_0",
                        boot_token="b1", tp_rank=r, pp_rank=0,
                        pieces=tuple(pieces))
        for r, pieces in enumerate(d_pieces)
    ]
    join = xm.join_manifests(manifests, pp_group="P", tp_group="D")
    assert not join.unsourced
    by_name = join.by_name
    assert by_name[QKV_NAME].shard_axis == wx.MIXED_FUSED
    assert by_name["model.layers.11.input_layernorm.weight"].shard_axis == (
        wx.REPLICATED)
    assert by_name["model.layers.11.mlp.down_proj.weight"].shard_axis == wx.ROWS


# ---------------------------------------------------------------------------
# 3. GRENZFAELLE -- kv==tp, kv>tp (the 27B production model), non-fused.
# ---------------------------------------------------------------------------


def test_kv_ge_tp_is_pinned_byte_identical_even_with_declared_components():
    """The 27B production model's path (kv=4 >= tp): plain ROWS, no
    replication skew. MUST NOT CHANGE -- proven by attaching (irrelevant,
    even nonsensical) component_rows and showing the outer ROWS check still
    wins before the new code ever runs."""
    tp, total_kv, total_heads, head_size = 3, 4, 32, 128
    du.set_tp_partition_ratios([1] * tp)
    try:
        assert not du.attn_kv_replicated(tp, total_kv)
        units = total_kv  # kv >= tp: kv heads are the indivisible unit
        q_rows = [du.tp_partition_size(total_heads, tp, r, units) * head_size
                  for r in range(tp)]
        kv_rows = [du.tp_partition_size(total_kv, tp, r, units) * head_size
                   for r in range(tp)]
    finally:
        du.set_tp_partition_ratios(None)
    d_rows = [q + 2 * kv for q, kv in zip(q_rows, kv_rows)]
    whole_rows = sum(d_rows)  # plain ROWS cut: no rank replicates anything

    def _resolve(component_rows_per_rank):
        whole = _piece(QKV_NAME, whole_rows, HIDDEN)
        cut = [
            _piece(QKV_NAME, r, HIDDEN, component_rows=c)
            for r, c in zip(d_rows, component_rows_per_rank)
        ]
        return xm._axis_of(QKV_NAME, whole, cut)

    baseline = _resolve([()] * tp)
    assert baseline[0] == wx.ROWS
    # Adversarial: declare components that would (if consulted) misclassify
    # -- e.g. claiming the K/V share is "replicated" at a size that does not
    # even match this rank's own row count. The outer ROWS test must still
    # win first, so this must resolve IDENTICALLY to the baseline.
    nonsense = [(r,) for r in d_rows]  # self-consistent but single "component"
    assert _resolve(nonsense) == baseline


def test_kv_eq_tp_boundary_is_pinned_byte_identical():
    """kv == tp is deliberately EXCLUDED from REPLICATED-KV
    (``attn_kv_replicated`` docstring, ``distributed/utils.py:1727-1734``):
    the even q split, no duplication. Must stay a plain ROWS cut."""
    tp = total_kv = 3
    total_heads, head_size = 24, 128
    du.set_tp_partition_ratios([1] * tp)
    try:
        assert not du.attn_kv_replicated(tp, total_kv)
        units = total_kv
        q_rows = [du.tp_partition_size(total_heads, tp, r, units) * head_size
                  for r in range(tp)]
        kv_rows = [du.tp_partition_size(total_kv, tp, r, units) * head_size
                   for r in range(tp)]
    finally:
        du.set_tp_partition_ratios(None)
    d_rows = [q + 2 * kv for q, kv in zip(q_rows, kv_rows)]
    whole = _piece(QKV_NAME, sum(d_rows), HIDDEN)
    cut = [_piece(QKV_NAME, r, HIDDEN) for r in d_rows]
    axis, *_ = xm._axis_of(QKV_NAME, whole, cut)
    assert axis == wx.ROWS


def test_non_fused_tensors_are_unaffected():
    """A plain REPLICATED norm and a plain COLS o_proj -- no component_rows
    anywhere, behaviour identical to before #1384."""
    whole = _piece("model.layers.0.input_layernorm.weight", 1, HIDDEN)
    cut = [_piece("model.layers.0.input_layernorm.weight", 1, HIDDEN)
           for _ in range(3)]
    assert xm._axis_of("x", whole, cut)[0] == wx.REPLICATED

    whole = _piece("model.layers.0.self_attn.o_proj.weight", HIDDEN, 4096)
    cut = [_piece("model.layers.0.self_attn.o_proj.weight", HIDDEN, w)
           for w in (1366, 1365, 1365)]
    assert xm._axis_of("x", whole, cut)[0] == wx.COLS


def test_a_genuine_disagreement_with_no_declared_components_still_raises():
    """A tensor that is really wrong (not mixed-fused at all) must still
    raise W68 when nobody declared a component split."""
    whole = _piece("some.other.weight", 100, HIDDEN)
    cut = [_piece("some.other.weight", 40, HIDDEN),
           _piece("some.other.weight", 40, HIDDEN),
           _piece("some.other.weight", 40, HIDDEN)]  # sums to 120 != 100
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm._axis_of("some.other.weight", whole, cut)
    assert "W68" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. MUTANTS -- the danger direction named in the ticket: a misclassified
#    axis compares the wrong bytes and can still report MATCH. Each of these
#    must still raise, not silently classify.
# ---------------------------------------------------------------------------


def test_mutant_a_corrupted_replica_on_one_rank_still_raises():
    """Rank 1's K component is corrupted (500 instead of the replicated 512)
    -- neither replicated (not all equal) nor a valid ROWS cut (does not sum
    to the whole's 512). A real per-rank corruption must refuse, not be
    smoothed over by the other two ranks agreeing."""
    whole_components, cut_components = _boot_geometry()
    corrupted = list(cut_components)
    q1, _k1, v1 = corrupted[1]
    corrupted[1] = (q1, 500, v1)  # own row count still sums (1024+500+512=2036)
    whole = _piece(QKV_NAME, 5120, HIDDEN, component_rows=whole_components)
    cut = [
        _piece(QKV_NAME, sum(c), HIDDEN, component_rows=c)
        for c in corrupted
    ]
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm._axis_of(QKV_NAME, whole, cut)
    assert "W68" in str(exc.value)


def test_mutant_b_self_inconsistent_declaration_gets_no_free_pass():
    """Rank 0 declares components that do NOT sum to its own reported
    rows_full -- a corrupted/buggy declaration is treated exactly like no
    declaration at all, never as licence to guess."""
    whole_components, cut_components = _boot_geometry()
    whole = _piece(QKV_NAME, 5120, HIDDEN, component_rows=whole_components)
    cut = [
        _piece(QKV_NAME, sum(cut_components[0]) + 7, HIDDEN,  # +7: now inconsistent
               component_rows=cut_components[0]),
        _piece(QKV_NAME, sum(cut_components[1]), HIDDEN,
               component_rows=cut_components[1]),
        _piece(QKV_NAME, sum(cut_components[2]), HIDDEN,
               component_rows=cut_components[2]),
    ]
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm._axis_of(QKV_NAME, whole, cut)
    assert "W68" in str(exc.value)


def test_mutant_c_differing_component_count_across_ranks_still_raises():
    """Rank 2 declares TWO components (Q, KV-combined) where every other
    side declares three (Q, K, V) -- a schema mismatch is a disagreement,
    not something to zip and hope."""
    whole_components, cut_components = _boot_geometry()
    two_comp = (cut_components[2][0], cut_components[2][1] + cut_components[2][2])
    whole = _piece(QKV_NAME, 5120, HIDDEN, component_rows=whole_components)
    cut = [
        _piece(QKV_NAME, sum(cut_components[0]), HIDDEN,
               component_rows=cut_components[0]),
        _piece(QKV_NAME, sum(cut_components[1]), HIDDEN,
               component_rows=cut_components[1]),
        _piece(QKV_NAME, sum(two_comp), HIDDEN, component_rows=two_comp),
    ]
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm._axis_of(QKV_NAME, whole, cut)
    assert "W68" in str(exc.value)


def test_mutant_d_single_axis_components_never_produce_mixed_fused():
    """Every component independently agreeing on ONE axis is one of the
    existing five classes and must not be attributed to the new function --
    ``_mixed_fused_axis`` itself must decline (``None``), keeping "five outer
    cases, no silent sixth" true. (The outer ``_axis_of`` never even reaches
    this helper for such a tensor -- the plain ROWS/REPLICATED test wins
    first -- so this exercises the helper directly for full branch coverage.)
    """
    whole = _piece(QKV_NAME, 300, HIDDEN, component_rows=(200, 100))
    cut = [
        _piece(QKV_NAME, 100, HIDDEN, component_rows=(70, 30)),
        _piece(QKV_NAME, 100, HIDDEN, component_rows=(70, 30)),
        _piece(QKV_NAME, 100, HIDDEN, component_rows=(60, 40)),
    ]
    # both components resolve as ROWS (70+70+60=200, 30+30+40=100) -- ONE
    # axis for the whole tensor, so this must NOT be called MIXED_FUSED.
    assert xm._mixed_fused_axis(whole, cut) is None


# ---------------------------------------------------------------------------
# 5. THE SCHEMA -- manifest JSON round-trip, ParamGeom refusal on consumption.
# ---------------------------------------------------------------------------


def test_manifest_json_roundtrip_preserves_component_rows():
    piece = _piece(QKV_NAME, 5120, HIDDEN, component_rows=(4096, 512, 512))
    back = xm.ManifestPiece.from_json(piece.as_json())
    assert back.component_rows == (4096, 512, 512)


def test_old_manifests_without_the_field_default_to_no_declared_split():
    raw = _piece(QKV_NAME, 5120, HIDDEN).as_json()
    del raw["component_rows"]  # a manifest written before #1384
    back = xm.ManifestPiece.from_json(raw)
    assert back.component_rows == ()


def test_paramgeom_refuses_an_undeclared_mixed_fused_axis_at_consumption():
    """UPDATED by the #1384 follow-up (see
    ``test_weg2_axisof_mixed_qkv_compare_1384.py``): the per-component byte
    copy for MIXED_FUSED is now wired, so ``MIXED_FUSED`` IS a valid
    ``ParamGeom.shard_axis`` -- but only once it carries a real, self-
    consistent component declaration. A geom that names the axis without
    declaring any components still fails LOUD instead of silently treating
    the whole tensor as one guessed block -- the property this test always
    asserted, now expressed against the sharper refusal."""
    geom = wx.ParamGeom(name=QKV_NAME, tag="weights_0",
                        shard_axis=wx.MIXED_FUSED,
                        rows_full=5120, cols_full=HIDDEN, itemsize=2)
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        geom.validate()
    assert "no declared component_rows" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. THE DECLARING AUTHORITY -- weight_exchange_shadow._qkv_component_rows.
# ---------------------------------------------------------------------------


class _FakeQKV(torch.nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        self.q_proj_shard_size = q
        self.kv_proj_shard_size = k
        self.v_proj_shard_size = v
        self.weight = torch.nn.Parameter(torch.zeros(q + k + v, HIDDEN))


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _FakeQKV(2048, 512, 512)
        self.other = torch.nn.Linear(HIDDEN, HIDDEN)


def test_qkv_component_rows_reads_off_the_owning_module_never_recomputes():
    model = _FakeModel()
    assert sh._qkv_component_rows(model, "qkv_proj.weight") == (2048, 512, 512)


def test_qkv_component_rows_is_empty_for_a_plain_linear():
    model = _FakeModel()
    assert sh._qkv_component_rows(model, "other.weight") == ()


def test_qkv_component_rows_is_empty_for_a_non_weight_name():
    model = _FakeModel()
    assert sh._qkv_component_rows(model, "qkv_proj.some_buffer") == ()


def test_qkv_component_rows_is_empty_for_an_unresolvable_module_path():
    model = _FakeModel()
    assert sh._qkv_component_rows(model, "does.not.exist.weight") == ()
