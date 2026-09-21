"""#48: the weg2 launcher anchors group D's capacity model to the last boot.

Every number asserted here is read out of a REAL boot log that reached
D READY -- ``/spinning/evidence-665-f1/boot_weg2_fnFL2v72_...D.log`` and its
v89 sibling -- not out of a fixture invented to match the parser.  The two
load-bearing claims are:

* the anchor is EXACT at the vector it was measured under (rank 0's predicted
  capacity equals the ``max_total_num_tokens=303872`` that boot actually
  sized), and
* it does NOT open the token-vector readback on this form, because the ranks
  disagree about their KV cell by 20x.

The log files are evidence on a rig, so every test that needs one skips
rather than fails when it is not there; the parser/derivation tests run on
the checked-in excerpts and always execute.
"""

import os

import pytest

from sglang.srt.planner.measured_anchor import (
    GIB,
    MIB,
    MeasuredAnchorRefused,
    REQUIRED_POSTS,
    anchor_has_uniform_kv_cell,
    build_components,
    parse_boot_log_posts,
)

V72 = (
    "/spinning/evidence-665-f1/"
    "boot_weg2_fnFL2v72_1da8f29f12_0921_114243.D.log"
)
V89 = (
    "/spinning/evidence-665-f1/"
    "boot_weg2_fnFL2v89_85ef1d0b96_0921_140445.D.log"
)

#: Verbatim line shapes from V72, one per rank, trimmed to the fields the
#: parser reads.  Checked in so the parser is tested even off the rig.
EXCERPT = """\
[2026-09-21 11:47:37 TP0] [vram-census] pp0tp0 after load: model tensors on device 11.78 GiB = {experts 6.91}; torch allocated 12.03 GiB, reserved 18.21 GiB (the gap to allocated is non-model)
[2026-09-21 11:48:53 TP1] [vram-census] pp0tp1 after load: model tensors on device 11.06 GiB = {experts 10.88}; torch allocated 11.46 GiB, reserved 12.94 GiB (the gap to allocated is non-model)
[2026-09-21 11:48:57 TP2] [vram-census] pp0tp2 after load: model tensors on device 10.49 GiB = {experts 10.31}; torch allocated 10.89 GiB, reserved 12.43 GiB (the gap to allocated is non-model)
[2026-09-21 11:48:59 TP2] [vram-census] pp0tp2-draft after load: model tensors on device 0.06 GiB = {other 0.06}; torch allocated 10.63 GiB, reserved 12.50 GiB (the gap to allocated is non-model)
[2026-09-21 11:49:04 TP0] [vram-idle] after pools: card free 1.472 of 31.336 GiB, allocated 17.51, reserved 27.60 -- NO forward in flight
[2026-09-21 11:49:04 TP0] [vram-idle] after pools: card free 1.200 of 31.336 GiB, allocated 17.78, reserved 27.88 -- NO forward in flight
[2026-09-21 11:49:04 TP1] [vram-idle] after pools: card free 5.051 of 19.585 GiB, allocated 11.39, reserved 13.22 -- NO forward in flight
[2026-09-21 11:49:04 TP2] [vram-idle] after pools: card free 5.545 of 19.585 GiB, allocated 10.82, reserved 12.71 -- NO forward in flight
[2026-09-21 11:49:04 TP0] KV pool sizing: available_bytes=4297900032 (4.003 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=303872
[2026-09-21 11:49:04 TP1] KV pool sizing: available_bytes=4762632192 (4.436 GiB), cell_size=768, page_size=64 -> max_total_num_tokens=6201344
[2026-09-21 11:49:04 TP2] KV pool sizing: available_bytes=5309988864 (4.945 GiB), cell_size=768, page_size=64 -> max_total_num_tokens=6914048
[2026-09-21 11:49:04 TP0] [auto-mamba] demand-driven mamba pool: target_concurrency=1 ratio=5 safety=1.25 -> max_mamba_cache_size=7 slots (0.38 GB @ per_req=56.11 MiB; fit_cap=56) -> admits ~1 reqs; activation_reserve=1.00 GB
[2026-09-21 11:49:04 TP1] [auto-mamba] demand-driven mamba pool: target_concurrency=1 ratio=5 safety=1.25 -> max_mamba_cache_size=7 slots (0.00 GB @ per_req=0.00 MiB; fit_cap=7) -> admits ~1 reqs; activation_reserve=1.00 GB
[2026-09-21 11:49:04 TP2] [auto-mamba] demand-driven mamba pool: target_concurrency=1 ratio=5 safety=1.25 -> max_mamba_cache_size=7 slots (0.00 GB @ per_req=0.00 MiB; fit_cap=7) -> admits ~1 reqs; activation_reserve=1.00 GB
"""


def _components(text=EXCERPT, reserve=(0, 0, 0)):
    return build_components(
        parse_boot_log_posts(text, rank_axis="tp"),
        tp_size=3,
        ranks_on_gpu=[1, 1, 1],
        required_free_bytes=list(reserve),
        source="test",
    )


# --- the parser -----------------------------------------------------------


def test_parses_every_rank_not_only_rank_zero():
    """The join that made this possible.

    ``[vram-peak]`` -- the instrument one would reach for first -- is emitted
    by rank 0 ALONE (measured: 1 line in each of V72 and V89), so a parser
    built on it can never produce a 3-rank balance.  ``[vram-idle] after
    pools`` carries the same driver reading per rank and is what this parser
    uses.
    """
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    assert sorted(posts) == [0, 1, 2]
    for rank in (0, 1, 2):
        assert "card_total_gib" in posts[rank]
        assert "available_bytes" in posts[rank]


def test_draft_census_line_is_not_mistaken_for_the_weight_checkpoint():
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    # rank 2's ``-draft`` line reports 10.63 GiB; the weight checkpoint is
    # the 10.89 GiB of the non-draft line.
    assert posts[2]["weights_alloc_gib"] == pytest.approx(10.89)


def test_idle_sample_taken_is_the_tightest_one():
    """Rank 0 emits two ``[vram-idle]`` samples (1.472 and 1.200 GiB free).

    The roomier one would fund KV the boot had already spent.
    """
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    assert posts[0]["card_free_gib"] == pytest.approx(1.200)


def test_main_pool_is_the_largest_cell_not_the_largest_token_count():
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    assert posts[0]["cell_size"] == 14143
    assert posts[0]["available_bytes"] == 4297900032


def test_rank_axis_is_not_wired_in():
    with pytest.raises(ValueError):
        parse_boot_log_posts(EXCERPT, rank_axis="nonsense")
    # group P geometry reads the pp ordinal of the same tag
    pp = parse_boot_log_posts(EXCERPT, rank_axis="pp")
    assert 0 in pp


# --- the derivation -------------------------------------------------------


def test_every_required_registry_post_is_present():
    for comp in _components():
        for key in REQUIRED_POSTS:
            assert key in comp, key


def test_the_balance_closes_exactly():
    """residual is SOLVED, so every rank's posts must sum to its card share."""
    for comp in _components():
        total = (
            comp["residual_residency_bytes"]
            + comp["weights_alloc_bytes"]
            + comp["mamba_aux_pool_bytes"]
            + comp["required_free_bytes"]
            + comp["kv_pool_bytes"]
        )
        assert total == comp["device_total_bytes"] // comp["ranks_on_gpu"]


def test_mamba_pool_comes_from_its_own_instrument_not_an_allocator_difference():
    """On rank 1 the after-pools allocator reading (11.39 GiB) is BELOW the
    after-load one (11.46), so a difference-derived mamba term clamps to zero
    and hides that it is meaningless.  ``slots x per_req`` states 0 because
    the rank holds no state, and states 0.38 GiB on rank 0 because it does.
    """
    comps = _components()
    assert comps[0]["mamba_aux_pool_bytes"] == pytest.approx(
        7 * 56.11 * MIB, rel=1e-6
    )
    assert comps[1]["mamba_aux_pool_bytes"] == 0
    assert comps[2]["mamba_aux_pool_bytes"] == 0


def test_a_missing_post_is_a_named_refusal_never_a_zero():
    text = "\n".join(
        l for l in EXCERPT.splitlines() if "auto-mamba" not in l
    )
    with pytest.raises(MeasuredAnchorRefused) as exc:
        build_components(
            parse_boot_log_posts(text, rank_axis="tp"),
            tp_size=3,
            ranks_on_gpu=[1, 1, 1],
            required_free_bytes=[0, 0, 0],
            source="test",
        )
    msg = str(exc.value)
    assert "mamba_pool_bytes" in msg
    assert "rank0" in msg and "rank1" in msg and "rank2" in msg
    assert "auto-mamba" in msg


def test_the_caller_configuration_has_no_default():
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    with pytest.raises(MeasuredAnchorRefused):
        build_components(
            posts, tp_size=3, ranks_on_gpu=[1, 1], required_free_bytes=[0, 0, 0]
        )
    with pytest.raises(MeasuredAnchorRefused):
        build_components(
            posts, tp_size=3, ranks_on_gpu=[1, 1, 1], required_free_bytes=[0]
        )


def test_an_impossible_balance_refuses_rather_than_clamping():
    """A reserve larger than the card cannot leave a non-negative residue."""
    with pytest.raises(MeasuredAnchorRefused) as exc:
        _components(reserve=(40 * GIB, 0, 0))
    assert "do not balance" in str(exc.value)


def test_co_location_halves_the_share_and_a_card_that_cannot_fit_refuses():
    """``ranks_on_gpu`` is the caller's rank->card map, and it bites.

    Declaring rank 0 co-located halves its share of the 31.34 GiB 5090 to
    15.67 GiB, which cannot hold that rank's measured 12.03 GiB of weights
    plus its 4.00 GiB KV budget.  That is the physical-impossibility check,
    and it REFUSES by name instead of clamping the residue at zero and
    handing the planner a card with more room than it has.
    """
    posts = parse_boot_log_posts(EXCERPT, rank_axis="tp")
    with pytest.raises(MeasuredAnchorRefused) as exc:
        build_components(
            posts,
            tp_size=3,
            ranks_on_gpu=[2, 1, 1],
            required_free_bytes=[0, 0, 0],
        )
    msg = str(exc.value)
    assert "rank0" in msg and "do not balance" in msg
    assert "15.67" in msg  # the halved share, named in the refusal

    # and the solo map, which is this rig's actual D layout, does balance
    solo = build_components(
        posts, tp_size=3, ranks_on_gpu=[1, 1, 1], required_free_bytes=[0, 0, 0]
    )
    assert solo[0]["ranks_on_gpu"] == 1
    assert solo[0]["device_total_bytes"] == int(round(31.336 * GIB))


# --- the readback guard (B) ----------------------------------------------


def test_this_form_does_not_have_a_uniform_kv_cell():
    """The finding that keeps the token-vector readback shut.

    Ranks 1 and 2 own no attention under ``--rank-tp-ratio 1,0,0`` and sized
    their pools at a 768-byte placeholder cell against rank 0's 14143.
    """
    comps = _components()
    assert {c["cell_size_bytes"] for c in comps} == {14143, 768}
    assert anchor_has_uniform_kv_cell(comps) is False


def test_uniform_cell_opens_the_guard():
    comps = [dict(c) for c in _components()]
    for c in comps:
        c["cell_size_bytes"] = 14143
    assert anchor_has_uniform_kv_cell(comps) is True


def test_a_component_without_the_cell_post_is_not_uniform():
    """An unknown cell is not a matching cell."""
    comps = [dict(c) for c in _components()]
    comps[1].pop("cell_size_bytes")
    assert anchor_has_uniform_kv_cell(comps) is False


# --- the launcher wiring --------------------------------------------------


def test_the_readback_guard_also_requires_a_uniform_cell():
    """Tripwire, next to the #62 one it extends.

    Anchoring makes ``measured is not None`` true, so that clause alone no
    longer means "this vector is safe to ship".
    """
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.d_operating_point_rows)
    i = src.index('token_units = tuple(int(v) for v in cap["token_vector"])')
    guard = src[:i]
    assert guard.rindex("anchor_has_uniform_kv_cell") > guard.rindex(
        'position == "maxkv"'
    )


def test_no_configured_reserve_means_no_anchor_and_says_so():
    from sglang.srt.weg2 import launcher

    anchor, note = launcher._d_measured_anchor(
        cards=[], budgets=[1, 1, 1], model="m", maxkv_weights=[1, 1, 1],
        user_reserve_by_card=None, injected=None, evidence_dirs=(),
    )
    assert anchor is None
    assert note.startswith(launcher.ANCHOR_NOTE_PREFIX)
    assert "--user-reserve-mib" in note
    assert "assumed zero" in note


def test_a_missing_evidence_dir_is_a_note_not_a_crash():
    from sglang.srt.weg2 import launcher

    anchor, note = launcher._d_measured_anchor(
        cards=[], budgets=[1, 1, 1], model="m", maxkv_weights=[1, 1, 1],
        user_reserve_by_card={}, injected=None,
        evidence_dirs=("/nonexistent-evidence-dir",),
    )
    assert anchor is None
    assert note.startswith(launcher.ANCHOR_NOTE_PREFIX)


def test_provenance_never_enters_the_refusal_contract():
    """``refusals`` decides what is FATAL; provenance must not ride in it."""
    from sglang.srt.weg2 import launcher

    notes = []
    rows, refusals = launcher.d_operating_point_rows(
        [], [29560, 18512, 18488], "no-such-model", 1, provenance_out=notes,
    )
    assert all(not r.startswith(launcher.ANCHOR_NOTE_PREFIX) for r in refusals)
    assert notes and notes[0].startswith(launcher.ANCHOR_NOTE_PREFIX)


# --- against the real logs ------------------------------------------------


@pytest.mark.parametrize("path", [V72, V89])
def test_real_boot_log_yields_a_complete_balance(path):
    if not os.path.exists(path):
        pytest.skip("evidence log not on this box: %s" % path)
    with open(path, errors="replace") as f:
        comps = build_components(
            parse_boot_log_posts(f.read(), rank_axis="tp"),
            tp_size=3,
            ranks_on_gpu=[1, 1, 1],
            required_free_bytes=[0, 0, 0],
            source=os.path.basename(path),
        )
    assert len(comps) == 3
    for c in comps:
        for key in REQUIRED_POSTS:
            assert key in c
    assert anchor_has_uniform_kv_cell(comps) is False


def test_anchor_reproduces_the_measured_boots_own_pool():
    """THE claim of #48, end to end against fnFL2v72's own numbers.

    Unanchored, the family model prices every routed expert as card-resident
    (72 GiB of "weights" on rank 0 against a 29.5 GiB budget) and declares the
    vector infeasible -- the W64 refusal of a configuration that ran.
    Anchored, rank 0's predicted capacity is the 303872 tokens that boot
    actually sized, to the token.
    """
    if not os.path.exists(V72):
        pytest.skip("evidence log not on this box")
    from sglang.srt.uneven_perf import PerfCostModel
    from sglang.srt.weg2.launcher import _gcd_reduce, d_plan_inputs

    model = (
        "/spinning/llm_stuff/club-3090/models-cache/"
        "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
    )
    if not os.path.exists(model):
        pytest.skip("checkpoint not on this box")
    budgets = [29560, 18512, 18488]
    weights = list(_gcd_reduce(budgets))
    with open(V72, errors="replace") as f:
        comps = build_components(
            parse_boot_log_posts(f.read(), rank_axis="tp"),
            tp_size=3,
            ranks_on_gpu=[1, 1, 1],
            required_free_bytes=[0, 0, 0],
            source="v72",
        )
    plan = d_plan_inputs(model, 3, 1)
    frac, slots = [0.0188, 0.2104, 0.1853], [44, 48, 48]

    bare = PerfCostModel(
        plan, list(weights), list(budgets),
        moe_resident_fraction=frac, moe_scratch_slots=slots,
    )
    cap_bare = bare.predict_capacity(list(weights))
    assert cap_bare["feasible"] is False
    assert cap_bare["token_vector"] is None

    anchored = PerfCostModel(
        plan, list(weights), list(budgets),
        moe_resident_fraction=frac, moe_scratch_slots=slots,
        measured=list(comps), measured_mlp_vector=list(weights),
    )
    cap = anchored.predict_capacity(list(weights))
    assert cap["feasible"] is True
    # EXACT at the measured vector: this is the whole point of the residue.
    assert int(cap["p"][0]) == 303872
    # and the bias is what removed the family model's absolute error
    assert anchored.measured_weight_bias[0] < -50 * GIB
