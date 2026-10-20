# SPDX-License-Identifier: Apache-2.0
"""Adversarial review of D-MEM-SHARE (nf-decode-host-1020 @ 7cd3c35adc), 20.10.

Each test names a danger direction the author's suite does not reach. At 7cd3c35adc the
four ``test_R*`` cases fail (the findings); with the review patch they pass.
"""
import types

import pytest

from flliper.srt.pdflip import d_mem_share as S

PAGE = 64
_PREFIX = list(range(1000, 1000 + 101248))


def _req(rid, ids, protected=0, n_out=0, header=True, extra_key=None):
    r = types.SimpleNamespace(
        rid=rid, origin_input_ids=list(ids), output_ids=[7] * n_out, finished=lambda: False,
        cache_protected_len=protected, extra_key=extra_key,
        sampling_params=types.SimpleNamespace(max_new_tokens=0))
    if header:
        r.weg2_front_x = 7140
        r.weg2_front_priced = len(ids) - protected
    return r


def _sched():
    return types.SimpleNamespace()


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv(S.ENV, raising=False)


# ------------------------------------------------------------------ R1: radix extra_key


def test_R1_same_tokens_other_extra_key_share_nothing():
    """RadixKey carries ``extra_key`` (cache_salt / LoRA namespace): the same token ids under two keys are two
    tree paths, two sets of pages. D-MEM-SHARE compares token ids only -> credit although nothing is shared."""
    run = _req("a", _PREFIX + [5] * 47, protected=101248, n_out=10, extra_key="salt-A")
    q = _req("b", _PREFIX + [9] * 202, protected=101248, extra_key="salt-B")
    used, incoming, _su, sh_in = S.census(_sched(), [run], [q], PAGE)
    assert sh_in == 0 and incoming == 202 + 101248


# ------------------------------------------------------------------ R2: member order -> vector layout


def _vector(running, queue):
    pl = S.prepare(_sched(), running, queue, PAGE)
    return [(pl.rids[i], pl.rids[j]) for i, j in pl.cands], pl.local


def test_R2_vector_layout_is_independent_of_queue_order():
    """The credit vector rides an ELEMENT-WISE MIN. Its slots are the candidate pairs in member order; two
    ranks with the same queue SET in a different ORDER put different pairs into the same slot."""
    r = _req("r", _PREFIX + [5] * 47, protected=101248, n_out=3)
    a = _req("a", _PREFIX + [6] * 300, protected=101248)
    b = _req("b", _PREFIX[:50000] + [8] * 900, protected=50000 // PAGE * PAGE)
    lay0, _ = _vector([r], [a, b])
    lay1, _ = _vector([r], [b, a])
    assert sorted(tuple(sorted(p)) for p in lay0) == sorted(tuple(sorted(p)) for p in lay1)  # same pair SET
    assert lay0 == lay1


def test_R2_candidate_count_is_independent_of_queue_order():
    """A queued request without the front header (resumed / second admission) next to one with it: whether the
    pair is a candidate depends on which of the two comes first -> different vector LENGTH per rank."""
    r_hdr = _req("h", _PREFIX + [6] * 300, protected=101248)
    r_no = _req("n", _PREFIX + [8] * 900, protected=101248, header=False)
    lay0, _ = _vector([], [r_no, r_hdr])
    lay1, _ = _vector([], [r_hdr, r_no])
    assert len(lay0) == len(lay1)


# ------------------------------------------------------------------ R3: the book outlives the guard


def test_R3_a_requeued_member_is_not_credited_from_the_book():
    """j was agreed while queued, admitted, then parked/retracted back into the queue (output_ids set: no front
    header, cache_protected_len 0). It is no candidate any more, so the book's old value is read -- and
    ``note_refused`` cannot void it (refused only gates ``_local_credit``)."""
    sched = _sched()
    i = _req("i", _PREFIX + [5] * 47, protected=101248, n_out=10)
    j = _req("j", _PREFIX + [9] * 202, protected=101248)
    S.census(sched, [i], [j], PAGE)  # agreed 101248
    S.census(sched, [i, j], [], PAGE)  # both run
    # j back in the queue: decoded tokens, no header, protected reset (reset_for_retract)
    j.output_ids = [7] * 40
    j.cache_protected_len = 0
    for a in ("weg2_front_x", "weg2_front_priced"):
        delattr(j, a)
    S.note_refused(sched, j)  # this rank's admission could not fund it
    used, incoming, _su, sh_in = S.census(sched, [i], [j], PAGE)
    assert sh_in == 0 and incoming == len(j.origin_input_ids) + 40


def test_requeued_member_without_book_entry_counts_whole():
    sched = _sched()
    i = _req("i", _PREFIX + [5] * 47, protected=101248, n_out=10)
    j = _req("j", _PREFIX + [9] * 202, protected=0, n_out=40, header=False)
    assert S.census(sched, [i], [j], PAGE)[3] == 0


# ------------------------------------------------------------------ equivalence of the two named mutants


def test_credit_never_exceeds_the_shorter_prompt():
    """Mutant 2 (``min(c, T)`` removed) is equivalent: every booked value is <= LCP(prompts) <= len(prompt_j)."""
    sched = _sched()
    i = _req("i", _PREFIX + [5] * 47, protected=101248, n_out=10)
    j = _req("j", _PREFIX[:4096], protected=4096)
    S.census(sched, [i], [j], PAGE)
    v = S.book(sched).agreed[S._key("i", "j")]
    assert v <= len(j.origin_input_ids)
