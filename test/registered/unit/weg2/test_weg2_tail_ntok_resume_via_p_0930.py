"""TAIL-NTOK (D-Nachlauf Klasse G): D adopts P's tail of a RESUME-VIA-P leg.

Hermetic (no CUDA, no pools): the admission verdict only (``plan_adopt`` /
``skip_joinable`` / ``uniform_refusal``), on an agreed END-only entry exactly
as the vote leaves it.

Metal y3r (c1c012dd5e, ...dauer09292330): of the four ``W50-REROUTE
reason=x_refusal_midstream`` legs, the two ``path=held-uncommitted`` ones (no
output yet) adopted P's tail, the two with output did not:

* weg2-38-53 ep44: D's prompt 131770 + 4 decoded tokens; P prefilled
  ``resume_via_p.context_ids`` = origin + output = 131774 tokens (front:
  ``X-EXACT-TOKENS group=P tokens_front=131770 tokens_group=131774 match=0``)
  and published n_tokens=131774 (page_prefix 131712, c 131772). D keyed the
  tail on ``origin_input_ids`` alone: ``len(ids) 131770 != 131774`` --
  printed as ``adopt=skipped:n_tokens:131774!=131774`` (fill vs want, the
  failing term unnamed) -- and ran the 2-token tail as a real 2118 ms extend;
* weg2-48-71 ep56: 108536 + 631 decoded = 109167, the same refusal.

What these cases pin:
* the ids a non-park tail keys on are D's prompt PLUS its output (the context
  P prefilled), so the RESUME-VIA-P leg takes the END state (skip);
* the byte riegel stays whole: one differing token below c refuses by
  ``key_mismatch``, a differing last token by the END key; one token decoded
  AFTER the context P saw refuses by ``n_tokens``;
* the refusal names the failing term (``n_tokens:ids131770!=131774``);
* a fresh hand-off (no output) adopts exactly as before.
"""

import logging
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

PAGE, GRAIN = 64, 4
FIRST = 20587  # P's sampled token after the context (y3r weg2-38-53)


def _ctx(prompt: int, out: int):
    return [1000 + (i % 50000) for i in range(prompt)], [70000 + i for i in range(out)]


def _entry(rid: str, ctx):
    """The agreed outcome of an END-only (H63 fold) hand-off of ``ctx``, as
    ``agree`` leaves it with group vote 2 (every rank holds the END state)."""
    spec = th.spec_for(rid, ctx, None, PAGE, GRAIN)
    end = th.EndHeader(first_token=FIRST, key=th.tail_key(ctx, len(ctx), None), rows=len(ctx) - spec.page_prefix,
                       groups=0, ring_rows=0, fa_digest="", gdn_digest="", ring_digest="", nbytes=0)
    hdr = th.TailHeader(spec=spec, part="pp0-1", fa_layers=[], gdn_layers=[], fa_row_shapes={}, gdn_row_shapes={},
                        fa_digest="", gdn_digest="", nbytes=0, end=end, n_parts=3, e1=False)
    st = ta.Staged(spec=spec, headers=[hdr], verdict="ready", end_verdict="ready", first_token=FIRST,
                   token_src="publish", e1=False, n_parts=3)
    return ta.Agreed(staged=st, agreed=True, skip=True)


def _d_req(rid: str, prompt, out):
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    return SimpleNamespace(rid=rid, origin_input_ids=list(prompt), output_ids=list(out),
                           full_untruncated_fill_ids=list(prompt) + list(out), extra_key=None,
                           return_logprob=False, return_hidden_states=False, grammar=None, sampling_params=sp)


@pytest.fixture
def adopt_on():
    ta._AGREED.clear()
    with envs.SGLANG_WEG2_TAIL_ADOPT.override(True):
        yield
    ta._AGREED.clear()


@pytest.mark.parametrize(
    "rid, prompt, out, page_prefix, cut",
    [
        ("weg2-38-53", 131770, 4, 131712, 131772),  # y3r ep44: P 131774, D-prompt 131770
        ("weg2-48-71", 108536, 631, 109120, 109164),  # y3r ep56: P 109167, D-prompt 108536
    ],
)
def test_resume_via_p_leg_takes_the_end_state(adopt_on, caplog, rid, prompt, out, page_prefix, cut):
    p_ids, o_ids = _ctx(prompt, out)
    ctx = p_ids + o_ids  # resume_via_p.context_ids: what P prefilled as its prompt
    entry = _entry(rid, ctx)
    spec = entry.staged.spec
    assert (spec.n_tokens, spec.page_prefix, spec.cut) == (prompt + out, page_prefix, cut)  # the metal geometry
    ta._AGREED[rid] = entry
    req = _d_req(rid, p_ids, o_ids)
    assert ta.skip_joinable(req, page_prefix)  # the adder's pre-check agrees with the commit
    assert ta.peek_compute_tokens(req, page_prefix) == 0  # no target forward
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        got = ta.plan_adopt(req, page_prefix)
    assert got is entry and got.skip, caplog.text
    assert "skipped:" not in caplog.text


def test_a_differing_token_below_the_cut_still_refuses_by_key(adopt_on, caplog):
    rid = "weg2-38-53"
    p_ids, o_ids = _ctx(131770, 4)
    ta._AGREED[rid] = _entry(rid, p_ids + o_ids)
    o_ids[0] += 1  # D decoded a token P never saw (position 131770 < c = 131772)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        assert ta.plan_adopt(_d_req(rid, p_ids, o_ids), 131712) is None
    assert "adopt=skipped:key_mismatch" in caplog.text


def test_a_differing_last_token_refuses_by_the_end_key(adopt_on, caplog):
    rid = "weg2-38-53"
    p_ids, o_ids = _ctx(131770, 4)
    ta._AGREED[rid] = _entry(rid, p_ids + o_ids)
    o_ids[-1] += 1  # position 131773 >= c: only the END key covers it
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        assert ta.plan_adopt(_d_req(rid, p_ids, o_ids), 131712) is None
    assert "adopt=skipped:end_only:end_key_mismatch" in caplog.text


def test_a_token_decoded_after_ps_context_refuses_by_n_tokens(adopt_on, caplog):
    rid = "weg2-38-53"
    p_ids, o_ids = _ctx(131770, 4)
    ta._AGREED[rid] = _entry(rid, p_ids + o_ids)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        assert ta.plan_adopt(_d_req(rid, p_ids, o_ids + [5]), 131712) is None
    assert "adopt=skipped:n_tokens:131775!=131774" in caplog.text


def test_the_refusal_names_the_failing_term():
    p_ids, o_ids = _ctx(131770, 4)
    spec = th.spec_for("weg2-38-53", p_ids + o_ids, None, PAGE, GRAIN)
    # the y3r shape: fill 131774 == want 131774, the ids 131770 short
    assert ta.uniform_refusal(spec, p_ids, 131774, None, 131712, end_only=True) == "n_tokens:ids131770!=131774"


def test_fresh_hand_off_without_output_adopts_as_before(adopt_on, caplog):
    rid = "weg2-21-26"
    p_ids, _ = _ctx(102864, 0)  # y3r: held-uncommitted, P 102864 == D 102864
    ta._AGREED[rid] = _entry(rid, p_ids)
    req = _d_req(rid, p_ids, [])
    assert ta.adopt_ids(req, False) is req.origin_input_ids  # no copy on the common path
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        got = ta.plan_adopt(req, th.spec_for(rid, p_ids, None, PAGE, GRAIN).page_prefix)
    assert got is not None and got.skip, caplog.text
