# SPDX-License-Identifier: Apache-2.0
"""#242: group P's posts from NF's OWN records, per stage -- not 27B's borrow.

THE BEFUND (planer_befund_nvml2.md, 27.09.; re-measured 28.09. over 62 NF boots).
The pool model of an NF boot charged three 27B rows and one scalar:

* ``P_PP_STAGE_FIXED_MIB`` 2342/1106/3518 (27B, with a draft head on the last
  stage NF's P no longer carries since H25) -- the runtime's own "weights +
  runtime state" of NF, minus the model's per-layer term at the same FR, is
  1584.0/527.1/1065.4;
* ``P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT`` 1.5588 (27B; NF measures 1.5602);
* ONE activation post for every stage, the maximum of the builtin transient
  support at chunk 16384 (3697 MiB, fnFL2x118..x146, one sequence per chunk).
  NF's Dauerlauf packs two to five sequences into one 16k chunk: PP0 draws up to
  5979 MiB (peak - allocated), PP1/PP2 never more than 2945/2941.

Boot dkrnfh91dprsavisnoadopts0bar1dauer09281111 (rc12z17, FR_P
0.324/0.637/0.734, cut 29,11,8): the PP-POOL-JOIN of priced against realised
available bytes was -758/-988/-2883 MiB. With the records it is <= 1 MiB.

THE FIX. ``nextflash.json`` carries ``P_ACTIVATION_MIB`` 5984/2952/2944,
``P_PP_STAGE_FIXED_MIB`` and ``P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT`` (no longer
borrowed). The record replaces the builtin 16384 support point for the runtime's
post and the pool model, which books it PER STAGE
(``PhasePoolModel.activation_reserve_mib_by_stage``); the P-KARTE takes the
larger of builtin and record per stage. The 27B profile is byte-identical.

Hermetic: no GPU, no torch device.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.planner import pp_cut
from flliper.srt.pdflip import form, profile_records
from flliper.srt.pdflip import launcher as lc

MIB = float(1 << 20)
NF, B27 = "nextflash", "qwen27b"

# The live boot dkrnfh91dprsavisnoadopts0bar1dauer09281111 (front log + P log).
BUDGET = (28240.0, 17680.0, 17168.0)
COUNTS, ATTN = (29, 11, 8), (7, 3, 2)
LAYER = (539.87, 927.21, 1047.11)  # PP-CUT POOL TERM 540/927/1047 at FR 0.324/0.637/0.734
REALISED_OLD_MIB = (6507200512 / MIB, 3426746368 / MIB, 4361814016 / MIB)  # KV pool sizing
RUNTIME_TRANSIENT_OLD = (3696.6, 3287.0, 3266.6)


def _model(stage_fixed, mamba, act, act_by_stage=()):
    return pp_cut.PhasePoolModel(
        free_mib=BUDGET,
        weight_mib_per_layer=61.6,
        weight_mib_per_layer_by_stage=LAYER,
        kv_mib_per_token_per_attn_layer=1024.0 / MIB,
        arming_floor_mib=(1229.0,) * 3,
        stage_fixed_mib=stage_fixed,
        activation_reserve_mib=act,
        activation_reserve_mib_by_stage=act_by_stage,
        corridor_holdback_mib=0.0,
        mamba_mib_per_linear_layer_per_slot=mamba,
        mamba_slots=32,
    )


# -- the records -------------------------------------------------------------


def test_nextflash_carries_its_own_p_records_and_no_longer_borrows_them():
    borrowed = profile_records.borrow_of(NF)[1]
    for name in ("P_PP_STAGE_FIXED_MIB", "P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"):
        assert name not in borrowed
    assert tuple(form.profile_constant("P_ACTIVATION_MIB", NF)) == (5984, 2952, 2944)
    assert form.profile_constant("P_PP_STAGE_FIXED_MIB", NF) == "1584.0,527.1,1065.4"
    assert form.profile_constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", NF) == pytest.approx(1.5602)
    own = {r.name: r for r in profile_records.own_records(NF)}
    for name in ("P_ACTIVATION_MIB", "P_PP_STAGE_FIXED_MIB"):
        assert own[name].pp_layer_ratio == (29, 11, 8)
        assert own[name].boots


def test_the_27b_profile_is_untouched():
    assert form.profile_constant("P_PP_STAGE_FIXED_MIB", B27) == "2342.0,1105.5,3518.0"
    assert form.profile_constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", B27) == pytest.approx(1.5588)
    with pytest.raises(KeyError):
        form.profile_constant("P_ACTIVATION_MIB", B27)
    assert lc.p_activation_record(B27) == (None, "")
    assert lc.p_transient_support(B27) is lc.P_PREFILL_TRANSIENT_SUPPORT
    assert lc.p_prefill_transient_vector_mib(16384, B27) == (3696.6, 3287.0, 3266.6)
    assert lc.p_prefill_activation_reserve_by_stage_mib(None, 16384, B27) == ()


# -- the support point and the posts -----------------------------------------


def test_the_record_replaces_the_16384_point_for_the_runtime_post():
    assert lc.p_prefill_transient_vector_mib(16384, NF) == (5984.0, 2952.0, 2944.0)
    # between the points the record is the upper support: 8192 (builtin) .. 16384 (record)
    v = lc.p_prefill_transient_vector_mib(12288, NF)
    assert v[0] == pytest.approx((1863.7 + 5984.0) / 2, abs=0.1)
    # every other support point stays the builtin measurement
    pts = {p.chunk: p for p in lc.p_transient_support(NF).points}
    assert pts[8192].mib == (1863.7, 1669.1, 1638.4)
    assert "P_ACTIVATION_MIB" in pts[16384].source


def test_the_card_takes_the_larger_of_builtin_and_record():
    card = {p.chunk: p.mib for p in lc.p_transient_support(NF, card=True).points}
    assert card[16384] == (5984.0, 3287.0, 3266.6)


def test_pool_post_per_stage_only_where_the_profile_measured_it():
    assert lc.p_prefill_activation_reserve_by_stage_mib(None, 16384, NF) == (5984.0, 2952.0, 2944.0)
    # a pinned --pp-cut-activation-reserve-mib pins every stage (the scalar stands)
    assert lc.p_prefill_activation_reserve_by_stage_mib(2048.0, 16384, NF) == ()
    # below the smallest support point the reference 1024 dominates: the scalar stands
    assert lc.p_prefill_activation_reserve_by_stage_mib(None, 256, NF) == ()
    # never below the reference 1024 MiB
    assert min(lc.p_prefill_activation_reserve_by_stage_mib(None, 4096, NF)) >= 1024.0


def test_phase_pool_model_charges_the_activation_post_per_stage():
    scalar = _model((0.0, 0.0, 0.0), 1.5602, 3000.0)
    by_stage = _model((0.0, 0.0, 0.0), 1.5602, 3000.0, (5000.0, 2000.0, 1000.0))
    a = pp_cut.stage_pp_capacities(COUNTS, ATTN, scalar)
    b = pp_cut.stage_pp_capacities(COUNTS, ATTN, by_stage)
    cells = [n * 1024 / MIB for n in ATTN]
    for r, delta in enumerate((-2000.0, 1000.0, 2000.0)):
        assert (b[r] - a[r]) * cells[r] == pytest.approx(delta, abs=cells[r])


def test_a_per_stage_post_must_name_every_stage():
    m = _model((0.0, 0.0, 0.0), 1.5602, 3000.0, (5000.0, 2000.0))
    with pytest.raises(ValueError, match="activation_reserve_mib_by_stage"):
        pp_cut.stage_pp_capacities(COUNTS, ATTN, m)


def test_a_per_stage_post_funds_the_activation_post():
    m = _model((1.0, 1.0, 1.0), 1.5602, 0.0, (5000.0, 2000.0, 1000.0))
    assert "activation_reserve_mib" not in m.unfunded_posts


# -- the join on the live boot (the dry-run gate) ----------------------------


def _priced_available(model):
    caps = pp_cut.stage_pp_capacities(COUNTS, ATTN, model)
    return [c * a * 1024 / MIB for c, a in zip(caps, ATTN)]


def test_live_boot_join_borrowed_rows_miss_by_up_to_2_9_gib():
    old = _model((2342.0, 1105.5, 3518.0), 1.5588, 3696.6)
    err = [p - r for p, r in zip(_priced_available(old), REALISED_OLD_MIB)]
    assert err == [pytest.approx(e, abs=2.0) for e in (-758.0, -988.5, -2882.5)]


def test_live_boot_join_with_the_nf_records_is_exact():
    stage_fixed = tuple(float(x) for x in str(form.profile_constant("P_PP_STAGE_FIXED_MIB", NF)).split(","))
    act = lc.p_prefill_activation_reserve_by_stage_mib(None, 16384, NF)
    new = _model(stage_fixed, float(form.profile_constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", NF)),
                 max(act), act)
    # the runtime books the SAME per-rank vector (P_PREFILL_TRANSIENT_ENV) -> its rest moves by the difference
    realised_new = [r - (n - o) for r, o, n in zip(REALISED_OLD_MIB, RUNTIME_TRANSIENT_OLD,
                                                    lc.p_prefill_transient_vector_mib(16384, NF))]
    for p, r in zip(_priced_available(new), realised_new):
        assert p == pytest.approx(r, abs=2.0)
    caps = pp_cut.stage_pp_capacities(COUNTS, ATTN, new)
    # the binding stage is PP0 now (was PP2); 262144 and the YaRN x2 floor 524288 both hold
    assert min(range(3), key=lambda r: caps[r]) == 0
    assert min(caps) >= 524288
