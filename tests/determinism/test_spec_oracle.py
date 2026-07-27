# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the speculative-decoding oracle (#143 follow-up to #124).

The property under test is the PROJECTION: a speculative run's observable is
not one logits row per decode step but a ``[k+1, V]`` verify matrix per
round, of which ``accept_len`` rows are the parents of emitted tokens.
``SpecRun.to_trajectory()`` projects that into emitted-token space, at which
point a spec run is the *same object* as a non-spec run and every existing
#124 primitive applies verbatim -- no parallel comparison machinery.

The second property is the one Window 5 forced us to learn: under matched
spec arms a single argmax flip changes ``accept_len`` and therefore reshapes
every later forward, so "0 flips over N tokens" (DECODE_CLASS) is
structurally unattainable. The tolerated relation is near-tie-gated, which
is what :data:`ByteIdentityClass.SPEC_NEAR_TIE` encodes.
"""

import pytest
import torch

from determinism_harness import (
    CLASS_SPECS,
    ByteIdentityClass,
    FlipKind,
    SpecRun,
    Trajectory,
    VerifyRound,
    check_accept_length_floor,
    check_accept_rule_exactness,
    check_class,
    get_case,
)
from determinism_harness.matrix import EXCLUDED_CASES, TEST_MATRIX

VOCAB = 32
BAND = 1.0
NEAR_TIE = 0.25


def _row(winner: int, margin: float, vocab: int = VOCAB) -> torch.Tensor:
    """A logits row whose argmax is ``winner`` by exactly ``margin``."""
    row = torch.zeros(vocab, dtype=torch.float32)
    row[winner] = margin
    return row


def _round(winners, margins, candidates=None, accept_len=None) -> VerifyRound:
    """A chain verify round: row i's argmax is winners[i] by margins[i]."""
    logits = torch.stack([_row(w, m) for w, m in zip(winners, margins)])
    d = len(winners)
    if candidates is None:
        # A perfectly-drafting chain: candidates[i+1] == argmax(row i), so
        # every draft is accepted. candidates[0] is the already-committed root.
        candidates = [0] + list(winners[: d - 1])
    if accept_len is None:
        accept_len = d
    return VerifyRound(
        logits=logits,
        candidates=list(candidates),
        emitted=list(winners[:accept_len]),
    )


def _run(rounds, seed: int = 1234, label: str = "") -> SpecRun:
    return SpecRun(rounds=list(rounds), seed=seed, label=label)


def _perfect_run(num_rounds: int = 4, d: int = 4, margin: float = 5.0) -> SpecRun:
    rounds = []
    tok = 1
    for _ in range(num_rounds):
        winners = [(tok + i) % VOCAB for i in range(d)]
        tok += d
        rounds.append(_round(winners, [margin] * d))
    return _run(rounds)


# --------------------------------------------------------------------------
# 1. The projection
# --------------------------------------------------------------------------


def test_projection_is_in_emitted_token_space():
    """One row per EMITTED token, not one per verify slot."""
    run = _run(
        [_round([3, 4, 5, 6], [5.0] * 4, accept_len=2), _round([7, 8], [5.0] * 2)]
    )
    traj = run.to_trajectory()
    assert traj.token_ids == [3, 4, 7, 8]
    assert traj.logits.shape == (4, VOCAB)
    assert len(traj) == 4


def test_projection_alignment_survives_different_accept_lengths():
    """Two runs that emit the same tokens with DIFFERENT round boundaries
    project to token-aligned trajectories. This is what makes a matched-spec
    A/B comparable at all: accept_len must not be an index in the oracle."""
    a = _run([_round([1, 2, 3, 4], [5.0] * 4)])
    b = _run(
        [
            _round([1, 2, 9, 9], [5.0] * 4, accept_len=2),
            _round([3, 4, 9, 9], [5.0] * 4, accept_len=2),
        ]
    )
    assert a.to_trajectory().token_ids == b.to_trajectory().token_ids == [1, 2, 3, 4]


def test_accept_lengths_reported():
    run = _run(
        [_round([1, 2, 3, 4], [5.0] * 4, accept_len=3), _round([5, 6], [5.0] * 2)]
    )
    assert run.accept_lengths() == [3, 2]
    assert run.mean_accept_length() == pytest.approx(2.5)


def test_empty_round_is_rejected():
    """accept_len >= 1 always: a verify round commits at least the bonus token."""
    with pytest.raises(ValueError):
        VerifyRound(logits=torch.zeros(4, VOCAB), candidates=[0, 1, 2, 3], emitted=[])


def test_more_emitted_than_verify_rows_is_rejected():
    with pytest.raises(ValueError):
        VerifyRound(logits=torch.zeros(2, VOCAB), candidates=[0, 1], emitted=[1, 2, 3])


# --------------------------------------------------------------------------
# 2. check_accept_rule_exactness -- the reference-free plumbing invariant
# --------------------------------------------------------------------------


def test_accept_rule_exactness_passes_on_a_consistent_run():
    v = check_accept_rule_exactness(_perfect_run())
    assert v.ok, v.summary()


def test_accept_rule_exactness_catches_an_emitted_non_argmax_token():
    """The greedy accept rule is exact integer equality against the target
    argmax (eagle_utils.verify_tree_greedy_func). An emitted token that is
    NOT its row's argmax means the accept plumbing -- accept_index, the
    rank-0 predict broadcast, the projection -- is wrong. No fp band can
    excuse it."""
    bad = _round([3, 4], [5.0, 5.0])
    bad.emitted = [3, 11]  # row 1's argmax is 4, not 11
    v = check_accept_rule_exactness(_run([bad]))
    assert not v.ok
    assert v.flip_kind is FlipKind.CORRUPTION
    assert "11" in v.message


def test_accept_rule_exactness_is_not_fooled_by_a_tie():
    """A row with two exactly-equal maxima: either winner is admissible."""
    row = torch.zeros(VOCAB, dtype=torch.float32)
    row[5] = 3.0
    row[6] = 3.0
    rnd = VerifyRound(logits=row.unsqueeze(0), candidates=[0], emitted=[6])
    assert check_accept_rule_exactness(_run([rnd])).ok


# --------------------------------------------------------------------------
# 3. check_accept_length_floor -- the (c) instrument
# --------------------------------------------------------------------------


def test_accept_length_floor_passes_and_fails_around_the_floor():
    run = _perfect_run(num_rounds=4, d=4)  # mean accept length 4.0
    assert check_accept_length_floor(run, floor=1.5).ok
    v = check_accept_length_floor(run, floor=4.5)
    assert not v.ok
    assert "4.5" in v.message


def test_accept_length_floor_catches_a_collapsed_accept_rate():
    """The lane-specific failure mode the #143 doc names: verify reads the
    WRONG KV slots, so it still emits self-consistent tokens but rejects
    nearly every draft. Token-level checks cannot see it; this can."""
    collapsed = _run([_round([1, 2, 3, 4], [5.0] * 4, accept_len=1) for _ in range(8)])
    v = check_accept_length_floor(collapsed, floor=1.5)
    assert not v.ok
    assert "below floor" in v.message
    assert collapsed.mean_accept_length() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 4. The SPEC_NEAR_TIE class
# --------------------------------------------------------------------------


def _spec_pair_near_tie():
    """Matched-spec arms that fork on a genuine near-tie at token 2 and then
    diverge structurally (different accept lengths from there on)."""
    ref = _perfect_run(num_rounds=2, d=3, margin=5.0).to_trajectory()
    test_logits = ref.logits.clone()
    # Token 2: make the reference's own margin to the flipped-to token tiny.
    flip_to = (int(ref.token_ids[2]) + 1) % VOCAB
    ref.logits[2, flip_to] = float(ref.logits[2, int(ref.token_ids[2])]) - 0.1
    test_logits = ref.logits.clone()
    test_logits[2, flip_to] += 0.2  # now the test picks flip_to
    test = Trajectory.from_greedy(test_logits, seed=ref.seed, label="test")
    return ref, test


def test_spec_near_tie_class_accepts_a_near_tie_fork():
    ref, test = _spec_pair_near_tie()
    v = check_class(
        ByteIdentityClass.SPEC_NEAR_TIE,
        ref=ref,
        test=test,
        rerun=test,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    assert v.ok, v.summary()
    assert v.flip_kind is FlipKind.NEAR_TIE


def test_spec_near_tie_class_rejects_a_confident_flip():
    """A divergence where the reference was CONFIDENT is a real defect, and
    it is exactly the shape a wrong-KV-context verify would take."""
    ref = _perfect_run(num_rounds=2, d=3, margin=5.0).to_trajectory()
    test_logits = ref.logits.clone()
    test_logits[2, (int(ref.token_ids[2]) + 1) % VOCAB] = 9.0
    test = Trajectory.from_greedy(test_logits, seed=ref.seed, label="test")
    v = check_class(
        ByteIdentityClass.SPEC_NEAR_TIE,
        ref=ref,
        test=test,
        rerun=test,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    assert not v.ok
    assert v.flip_kind is FlipKind.CORRUPTION


def test_spec_near_tie_class_requires_a_rerun():
    ref, test = _spec_pair_near_tie()
    v = check_class(
        ByteIdentityClass.SPEC_NEAR_TIE,
        ref=ref,
        test=test,
        rerun=None,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    assert not v.ok
    assert "rerun" in v.summary()


def test_spec_near_tie_rejects_a_non_self_deterministic_arm():
    ref, test = _spec_pair_near_tie()
    rerun_logits = test.logits.clone()
    rerun_logits[0, 0] += 1e-3
    rerun = Trajectory(
        token_ids=list(test.token_ids), logits=rerun_logits, seed=test.seed
    )
    v = check_class(
        ByteIdentityClass.SPEC_NEAR_TIE,
        ref=ref,
        test=test,
        rerun=rerun,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    assert not v.ok


def test_spec_near_tie_does_not_require_zero_flips():
    """The load-bearing difference to DECODE_CLASS. Same near-tie pair:
    DECODE_CLASS fails it (it demands 0 flips), SPEC_NEAR_TIE passes it --
    because under matched spec a flip changes accept_len and reshapes every
    later forward, so a 0-flip demand is unattainable by construction, not
    by defect."""
    ref, test = _spec_pair_near_tie()
    decode_v = check_class(
        ByteIdentityClass.DECODE_CLASS,
        ref=ref,
        test=test,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    spec_v = check_class(
        ByteIdentityClass.SPEC_NEAR_TIE,
        ref=ref,
        test=test,
        rerun=test,
        band=BAND,
        near_tie_margin=NEAR_TIE,
    )
    assert not decode_v.ok
    assert spec_v.ok


def test_spec_near_tie_has_a_class_spec_entry():
    spec = CLASS_SPECS[ByteIdentityClass.SPEC_NEAR_TIE]
    assert spec.required_inputs == "ref+test+rerun"
    assert spec.needs_band and spec.needs_near_tie_margin
    assert "accept" in spec.provenance.lower()


# --------------------------------------------------------------------------
# 5. Matrix rows
# --------------------------------------------------------------------------


def test_lane_spec_matched_row_exists_and_matches_spec_on_both_arms():
    """Candidate (a): the reference arm must carry the SAME speculative
    configuration, otherwise the spec-shape difference sits in one arm only
    and the comparison measures spec, not the lane."""
    case = get_case("weightless_spec_matched")
    assert case.expected_class is ByteIdentityClass.SPEC_NEAR_TIE
    assert case.needs_rerun
    for key in (
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
    ):
        assert case.test_config[key] == case.reference_config[key], key
    assert case.reference_config["tp_size"] == 1
    assert case.test_config["weightless_kv_fastlane"] is True


def test_lane_spec_replay_row_is_teacher_forced():
    case = get_case("spec_replay_teacher_forced")
    assert case.expected_class is ByteIdentityClass.SPEC_NEAR_TIE
    assert case.reference_config.get("speculative_algorithm") is None
    assert "teacher" in case.notes.lower() or "replay" in case.notes.lower()


def test_no_matrix_row_claims_spec_vs_nospec_token_identity():
    """The Window-5 finding, encoded so it cannot be re-added by accident."""
    assert "spec_vs_nospec_token_identity" in EXCLUDED_CASES
    reason = EXCLUDED_CASES["spec_vs_nospec_token_identity"]
    assert "temperature 0" in reason or "temperature 0" in reason.lower()
    for case in TEST_MATRIX:
        spec_on = case.test_config.get("speculative_algorithm") is not None
        spec_off = case.reference_config.get("speculative_algorithm") is None
        if spec_on and spec_off:
            assert (
                case.expected_class is not ByteIdentityClass.MACHINE_ZERO
            ), case.case_id
            assert (
                case.expected_class is not ByteIdentityClass.DECODE_CLASS
            ), case.case_id


# --------------------------------------------------------------------------
# 6. Reading the GPU-side dump back
# --------------------------------------------------------------------------


def _dump_record(accept_lens, bs=1, d=4, vocab=VOCAB):
    """A record in the shape sglang.srt.speculative.spec_verify_dump writes."""
    logits = torch.zeros(bs * d, vocab, dtype=torch.float32)
    for r in range(bs * d):
        logits[r, (r + 3) % vocab] = 5.0
    predict = [int((r + 3) % vocab) for r in range(bs * d)]
    accepted_rows = [list(range(b * d, b * d + accept_lens[b])) for b in range(bs)]
    return {
        "step": 0,
        "tp_rank": 0,
        "mode": "target_verify",
        "bs": bs,
        "draft_token_num": d,
        "logits": logits,
        "candidates": torch.zeros(bs, d, dtype=torch.int64),
        "predict": torch.tensor(predict, dtype=torch.int64),
        "accept_lens": torch.tensor(accept_lens, dtype=torch.int64),
        "accepted_rows": accepted_rows,
        "emitted": [[predict[r] for r in rows] for rows in accepted_rows],
    }


def test_verify_round_from_dump_record_reorders_rows_into_emitted_order():
    rec = _dump_record([2])
    rnd = VerifyRound.from_dump_record(rec, request=0)
    assert list(rnd.emitted) == rec["emitted"][0]
    assert rnd.accept_len == 2
    # Rows must be REORDERED into emitted order, not sliced positionally --
    # accept_index carries global flat indices and a tree layout would not
    # hand back 0, 1, 2, ...
    for i, tok in enumerate(rnd.emitted):
        assert int(rnd.logits[i].argmax()) == int(tok)


def test_verify_round_from_dump_record_selects_the_request():
    rec = _dump_record([1, 3], bs=2)
    r0 = VerifyRound.from_dump_record(rec, request=0)
    r1 = VerifyRound.from_dump_record(rec, request=1)
    assert r0.accept_len == 1 and r1.accept_len == 3
    assert list(r1.emitted) == rec["emitted"][1]


def test_spec_run_from_dump_records_is_ordered_by_step():
    """Records are read back by step number, not by directory order."""
    recs = []
    for step in (2, 0, 1):
        rec = _dump_record([2])
        rec["step"] = step
        recs.append(rec)
    run = SpecRun.from_dump_records(recs, seed=1234, request=0)
    assert [r for r in run.meta["steps"]] == [0, 1, 2]
    assert check_accept_rule_exactness(run).ok


def test_spec_run_from_dump_records_rejects_a_wrong_mode():
    rec = _dump_record([2])
    rec["mode"] = "decode"
    with pytest.raises(ValueError, match="target_verify"):
        SpecRun.from_dump_records([rec], seed=1234, request=0)
