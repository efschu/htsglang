# SPDX-License-Identifier: Apache-2.0
"""HANDBACK N-1 (fe5c55041b, N3 e9736b85c0): the EXACT bigram keying -- the 27B production
form since the qwen27b row turned exact -- for the three hand-back families whose
specimens are pinned to the UPSTREAM keying (fork_anchor_0926, p_trim_end_anchor_0924,
store_short_tail_xsn437 set SGLANG_WEG2_BIGRAM_ANCHOR_EXACT=0).

Same real trees and the same P prefill helpers as those files; what changes is the keying
(a node of k units = KV of k tokens = the state after k tokens) and D's claim (on a
group-D rank with an exact tree, N raw tokens = N-1 units, weg2/handback_claim.py).

DANGER DIRECTIONS guarded here, each with its MUTANT (the exact arithmetic turned back to
N-2 must turn the test red -- asserted in-suite per test, ``test_the_n_minus_2_mutant_...``):
  (1) FORK CUT: D's leg 2 and a sibling resume AT the fork with the fork state (F units,
      not F-1); without the fork cut D resumes at N-1 with the end state.
      Mutant P-key (the trimmed/fork-cut insert loses its next token).
  (2) P-TRIM -> D-RESUME: D resumes at N-1 units with the N-1 state and the #1481 mark;
      nothing deeper than N-1 on P; the #1442 hand-off carries N-1 page keys.
      Mutants P-key (N-2) and D-claim (D claims N-1 raw: the claim ends inside P's exact
      N-1 node, no anchor there, D falls back past the released chunk anchors -- the
      exact keying REQUIRES D's N claim).
  (3) STORE-SHORT TAIL (metal weg2-6-2 N=4316 / weg2-20-37 N=2050): the trimmed chain
      covers D's deliverable N-1; before the finish the published, resumable prefix is
      the interval anchor / the chunk boundary (exact: 4096 / 2048), so the metal
      shortfalls 219 and 1 are unchanged; N=2049 delivers 2048.
      Mutants P-key and D-claim.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.weg2 import form as _F
from sglang.srt.weg2 import handback_claim as HC
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_hbx_{name}", os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FA = _load("test_weg2_fork_anchor_0926")
PT = _load("test_weg2_p_trim_end_anchor_0924")
SS = _load("test_weg2_store_short_tail_xsn437")


@pytest.fixture
def exact(monkeypatch):
    """group P of the 27B form with the production (exact) keying; the D side is
    switched on per claim (``_as_d``)."""
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv(_F.FORM_ENV, _F.Weg2Form(
        arch="dense", experts="none", draft="dflash", p_draft="none", kv="paged_dcp",
        flip="family", vision="off", profile="qwen27b", model="m").env_value())
    monkeypatch.delenv("SGLANG_WEG2_BIGRAM_ANCHOR_EXACT", raising=False)   # the profile decides
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    monkeypatch.setenv(SS.ANCHOR_INTERVAL_ENV, str(SS.INTERVAL))
    old = HC.BIGRAM_EXACT_TREE[0]
    yield monkeypatch
    HC.BIGRAM_EXACT_TREE[0] = old


def _as_d(mp):
    mp.setenv("SGLANG_WEG2_GROUP", "D")


def _as_p(mp):
    mp.setenv("SGLANG_WEG2_GROUP", "P")


def _mutant_p_key(mp):
    """the trimmed / fork-cut finish insert keys only its committed ids (N-2 units)"""
    mp.setattr(urc, "_ids_with_tail", lambda ids, tail: ids)


def _mutant_d_claim(mp):
    """D claims N-1 raw tokens (N-2 bigram units), the upstream reader"""
    mp.setattr(HC, "handback_bigram_claim", lambda env=None: False)


def _d_claim_len(n):
    return Req._compute_max_prefix_len(FA.SimpleNamespace(return_logprob=False, logprob_start_len=-1), n)


# -- (1) FORK CUT ---------------------------------------------------------------


def _fork_run(mp, fork_on, ids):
    if fork_on:
        mp.setenv(FA.TOKEN_ENV, str(FA.IM))
    else:
        mp.delenv(FA.TOKEN_ENV, raising=False)
    fx = FA._fixture(True)
    assert fx.cache.bigram_anchor_exact, "the 27B row keys exact"
    FA._p_leg1(fx)
    _as_d(mp)
    try:
        return FA._claim(fx, ids)
    finally:
        _as_p(mp)


def test_fork_cut_d_leg2_resumes_at_the_fork_exact(exact):
    depth, state = _fork_run(exact, True, FA.PROMPT)
    assert (depth, state) == (FA.F, pytest.approx(FA.S_FORK)), (depth, state)


def test_fork_cut_the_sibling_resumes_at_the_fork_exact(exact):
    depth, state = _fork_run(exact, True, FA.SIBLING)
    assert (depth, state) == (FA.F, pytest.approx(FA.S_FORK)), (depth, state)


def test_without_the_fork_cut_d_resumes_at_n_minus_1_and_the_sibling_below_the_fork_exact(exact):
    depth, state = _fork_run(exact, False, FA.PROMPT)
    assert (depth, state) == (FA.N - 1, pytest.approx(FA.S_END)), (depth, state)
    s_depth, s_state = _fork_run(exact, False, FA.SIBLING)
    assert s_depth < FA.F and s_state != pytest.approx(FA.S_FORK), (s_depth, s_state)


# -- (2) P-TRIM -> D-RESUME -------------------------------------------------------


def _trim_run(mp):
    fx = PT._fixture(True)
    assert fx.cache.bigram_anchor_exact
    req = PT._prefill_trimmed(fx)
    _as_d(mp)
    try:
        claim = _d_claim_len(PT.N)
        mr = fx.cache.match_prefix(MatchPrefixParams(key=PT._key(fx, PT.PROMPT[:claim])))
    finally:
        _as_p(mp)
    node = mr.last_device_node
    slot = node.component_data[PT.ComponentType.MAMBA].value
    state = None if slot is None else float(fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return fx, req, len(mr.device_indices), node, state


def test_p_trim_d_resumes_at_n_minus_1_with_the_state_and_mark_exact(exact):
    fx, _req, depth, node, state = _trim_run(exact)
    assert depth == PT.N - 1, depth
    assert state == pytest.approx(PT.S1)
    assert getattr(node, "_weg2_end_anchor", False), "the trimmed request's final node is the end anchor"
    assert PT._deepest(fx) == PT.N - 1, "nothing deeper than N-1 on P: no 1-token insert"


def test_p_trim_handoff_carries_n_minus_1_page_keys_exact(exact, tmp_path):
    exact.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    from sglang.srt.weg2 import handoff as ho

    fx = PT._fixture(True)
    keys = []
    fx.cache._weg2_handoff_write = lambda req, radix_key: keys.append(radix_key)   # the finish's own key
    req = PT._prefill_trimmed(fx)
    del fx.cache._weg2_handoff_write
    assert keys, "the finished insert hands its key to the #1442 write"
    mr = fx.cache.match_prefix(MatchPrefixParams(key=keys[-1]))
    node, k = mr.last_device_node, 0
    while node is not None and node is not fx.cache.root_node:   # keys the store would carry
        node.hash_value = [f"h{k + i}" for i in range(len(node.key))]
        k += len(node.key)
        node = node.parent
    fx.cache.enable_storage = True
    fx.cache._weg2_handoff_write(req, keys[-1])
    rec = json.load(open(ho.path(PT.RID)))
    assert rec["input_ids"] == PT.PROMPT, "D's tokenizer takes all N as its prompt"
    assert len(rec["page_keys"]) == PT.N - 1          # exact units of [0, N-1): D's whole claim


# -- (3) STORE-SHORT TAIL ----------------------------------------------------------


def _store_run(mp, n):
    """SS._p_leg1 with D's claim from Req._compute_max_prefix_len on group D."""
    fx = SS._fixture()
    assert fx.cache.bigram_anchor_exact
    prompt = list(range(10_000, 10_000 + n))
    recv = SS.SimpleNamespace(rid="weg2-7-1", input_ids=SS.array("q", prompt), input_embeds=None,
                              return_logprob=False, sampling_params=SS.SamplingParams(max_new_tokens=1),
                              session_params=None, session_id=None, mm_inputs=None)
    ids, tail = SS.pt.split_ids(recv)
    ids = list(ids)
    req = Req(rid="weg2-7-1", origin_input_text="", origin_input_ids=SS.array("q", ids),
              sampling_params=SS.SamplingParams(temperature=0, max_new_tokens=1))
    setattr(req, SS.pt.TRIM_ATTR, SS.array("q", list(tail)))
    fx.pool.alloc([req])
    req.output_ids = SS.array("q")
    req.full_untruncated_fill_ids = SS.array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = SS.torch.empty(0, dtype=SS.torch.int64)
    req.cache_protected_len = 0
    req.last_node = fx.cache.root_node
    for end in range(SS.CHUNK, len(ids), SS.CHUNK):
        start = len(req.prefix_indices)
        fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
        req.set_extend_range(start, end)
        fx.cache.cache_unfinished_req(req, chunked=True)
    _as_d(mp)
    d_claim = prompt[: _d_claim_len(n)]
    before = len(fx.cache.match_prefix(MatchPrefixParams(key=SS._key(fx, d_claim))).device_indices)
    _as_p(mp)
    start = len(req.prefix_indices)
    fx.pool.write((req.req_pool_idx, slice(start, len(ids))), fx.allocator.alloc(len(ids) - start))
    req.set_extend_range(start, len(ids))
    req.kv_committed_len = len(ids)
    req.kv_allocated_len = len(ids)
    fx.cache.cache_finished_req(req, is_insert=True)
    _as_d(mp)
    after = len(fx.cache.match_prefix(MatchPrefixParams(key=SS._key(fx, d_claim))).device_indices)
    _as_p(mp)
    return before, after


#: (N, resumable before the finish under the exact keying, the metal shortfall)
EXACT_A = (4316, 4096, 219)      # weg2-6-2: the 4096 interval anchor
EXACT_B = (2050, 2048, 1)        # weg2-20-37: the chunk boundary 2048 = the trimmed N'-1


@pytest.mark.parametrize("case", [EXACT_A, EXACT_B], ids=["weg2-6-2_N4316", "weg2-20-37_N2050"])
def test_store_short_tail_the_trimmed_chain_covers_d_s_deliverable_exact(exact, case):
    n, first, shortfall = case
    before, after = _store_run(exact, n)
    assert after == n - 1, "the publish is WHOLE: [0, N-1) = D's exact deliverable"
    assert before == first, "what the chunk publish put out before the finish"
    assert after - before == shortfall, "the metal race geometry is unchanged by the keying"


def test_store_short_tail_n_2049_delivers_2048_exact(exact):
    _before, after = _store_run(exact, 2049)
    assert after == 2049 - 1


# -- MUTANT PER TEST ----------------------------------------------------------------
#: every exact test above, run with the exact arithmetic turned back to N-2, must fail.
#: p_key: the trimmed / fork-cut finish keys only its committed ids; d_claim: D claims
#: N-1 raw tokens. The fork cut's own claims sit at F < N-1, which D's claim length does
#: not reach -- there only the P key carries the exact arithmetic.
_MUTANTS = {"p_key": _mutant_p_key, "d_claim": _mutant_d_claim}
_CASES = [
    ("fork_leg2", lambda mp, tp: test_fork_cut_d_leg2_resumes_at_the_fork_exact(mp), ("p_key",)),
    ("fork_sibling", lambda mp, tp: test_fork_cut_the_sibling_resumes_at_the_fork_exact(mp), ("p_key",)),
    ("fork_off", lambda mp, tp: test_without_the_fork_cut_d_resumes_at_n_minus_1_and_the_sibling_below_the_fork_exact(mp),
     ("p_key", "d_claim")),
    ("trim_resume", lambda mp, tp: test_p_trim_d_resumes_at_n_minus_1_with_the_state_and_mark_exact(mp), ("p_key", "d_claim")),
    ("trim_handoff", lambda mp, tp: test_p_trim_handoff_carries_n_minus_1_page_keys_exact(mp, tp), ("p_key",)),
    ("store_4316", lambda mp, tp: test_store_short_tail_the_trimmed_chain_covers_d_s_deliverable_exact(mp, EXACT_A),
     ("p_key", "d_claim")),
    ("store_2050", lambda mp, tp: test_store_short_tail_the_trimmed_chain_covers_d_s_deliverable_exact(mp, EXACT_B),
     ("p_key", "d_claim")),
    ("store_2049", lambda mp, tp: test_store_short_tail_n_2049_delivers_2048_exact(mp), ("p_key", "d_claim")),
]


@pytest.mark.parametrize("case,mutant", [(c, m) for c in _CASES for m in c[2]],
                         ids=[f"{c[0]}-{m}" for c in _CASES for m in c[2]])
def test_the_n_minus_2_mutant_turns_every_exact_test_red(exact, tmp_path, case, mutant):
    _MUTANTS[mutant](exact)
    with pytest.raises(AssertionError):
        case[1](exact, tmp_path)


# -- COLD FIRST CLAIM (D boots with --skip-server-warmup) ----------------------------
#: BIGRAM_EXACT_TREE is the process-wide note D's claim reads; a fresh D process holds
#: False until its tree has noted its keying. When that note came only from the lazy
#: accessor (first cache_*_req / match validator), the first hand-back claim -- computed
#: BEFORE its own match_prefix -- read the flag cold and claimed N-1 raw tokens, which on
#: the exact tree ends inside P's N-1 node: D resumed at 0 (a full D re-prefill).


def _cold_d_first_claim(mp):
    p = PT._fixture(True)                   # P: the trimmed request, the N-1 node
    PT._prefill_trimmed(p)
    HC.BIGRAM_EXACT_TREE[0] = False         # a fresh D process: nothing noted yet
    PT._fixture(True)                       # D constructs its tree -- no insert, no match
    _as_d(mp)
    try:
        claim = _d_claim_len(PT.N)
        depth = len(p.cache.match_prefix(MatchPrefixParams(key=PT._key(p, PT.PROMPT[:claim]))).device_indices)
    finally:
        _as_p(mp)
    return claim, depth


def test_the_first_handback_claim_on_a_freshly_built_tree_claims_n_minus_1_units(exact):
    claim, depth = _cold_d_first_claim(exact)
    assert claim == PT.N, "N raw tokens = N-1 exact units, before any match_prefix"
    assert depth == PT.N - 1, "the first hand-back resumes at N-1, no D re-prefill"


def _mutant_lazy_note(mp):
    """the keying is noted lazily again: construction leaves the flag untouched"""
    orig = urc.UnifiedRadixCache.__init__

    def lazy_init(self, *a, **kw):
        was = HC.BIGRAM_EXACT_TREE[0]
        orig(self, *a, **kw)
        self.__dict__.pop("_weg2_bigram_anchor_exact", None)
        HC.BIGRAM_EXACT_TREE[0] = was

    mp.setattr(urc.UnifiedRadixCache, "__init__", lazy_init)


def test_the_lazy_note_mutant_turns_the_cold_first_claim_red(exact):
    _mutant_lazy_note(exact)
    with pytest.raises(AssertionError):
        test_the_first_handback_claim_on_a_freshly_built_tree_claims_n_minus_1_units(exact)
