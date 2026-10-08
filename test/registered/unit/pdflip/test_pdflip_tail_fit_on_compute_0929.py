"""cold-round1: der D-Nachlauf nach dem P->D-Wake (~5 s) ist ein Extend, den
die Korridor-Kuerzung des Chunks am agreed END-Tail vorbeischickt.

Metall y3p (D-Log ...dauer09292250_a22179d7de, TP0), fuenf Wakes mit
wake_to_last_decode 5,2-6,0 s (ep44 23:07:58, ep54 23:10:18, 23:11:55,
23:13:24, 23:15:34). Jedes Mal direkt nach dem Wake:

  '#794 GROUP-NARROWED this prefill chunk from 4096 to 64 tokens: ... the
   tightest card's corridor funds 64'

und ein geparkter Resume mit 112-211 ungecachten Tokens (pdflip-24-27: F4
PARK-END page_prefix=92480 cut=93048 N=93049, Load-back 92928, uncached 122)
nahm den CHUNKED-Zweig von PrefillAdder.add_one_req, der nie einen Tail
nimmt (kein PDFLIP-TAIL-READY, nur PDFLIP-TAIL-CONSUMED): echter Extend 64 Tokens
(gpu-ms 1955, 'MoE offload ... expert-major prefill, 3 waves, 0.30 GiB H2D'
je Schicht) + Rest im naechsten Pass (261 Tokens, 2427 ms). Mit dem Tail
rechnet dieser Resume NICHTS (E2-Skip): 1-4 Token-Extends kosten auf D
11-37 ms.

Gepinnt:
* ``peek_compute_tokens``: 0 unter dem Skip, N - c unter E1, None ohne
  Tail -- ohne das Ergebnis zu nehmen;
* der Adder passt den Chunk gegen das Gerechnete (0 <= 64), der Schalter
  aus gibt das alte Mass (128 > 64) zurueck, ein Batch mit Forward laesst
  den Skip nicht zu (bleibt 128);
* das Chunk-Budget zahlt das Gerechnete, die KV-Budgets das Ganze;
* Verdrahtung in add_one_req: Whole-Fit und H24c-Riegel lesen fit_tokens,
  der Guertel hinter plan_adopt faengt eine abweichende Planung.
"""

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.schedule_policy import PrefillAdder  # noqa: E402
from flliper.srt.pdflip import tail_adopt as ta  # noqa: E402
from flliper.srt.pdflip import tail_handoff as th  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

PAGE, RATIO = 64, 4
PROMPT, OUT = 400, 100  # consumed N = 499 (prompt + all output but the parked last)
WINDOW = 256  # F4 park window [page_prefix, cut): the #59b anchor the park wrote from
RESUME = 384  # the load-back lands inside the window (metal: 92928 in [92480, 93048))
PARKED_TOKEN = 4242


def _park_req(rid="pdflip-24-27", prefix=RESUME, **kw):
    ids = [(7919 + i) % 151000 for i in range(PROMPT)]
    out = [(31 * i) % 151000 for i in range(OUT - 1)] + [PARKED_TOKEN]
    spp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
                          min_new_tokens=0, ignore_eos=False)
    base = dict(rid=rid, origin_input_ids=ids, output_ids=out, full_untruncated_fill_ids=ids + out,
                extra_key=None, prefix_indices=torch.arange(prefix, dtype=torch.int64),
                return_logprob=False, return_hidden_states=False, grammar=None, sampling_params=spp)
    base.update(kw)
    return SimpleNamespace(**base)


def _park_agreed(req, skip=True, e1=False):
    consumed = th.park_ids(req)
    spec = th.park_spec(req.rid, consumed, None, WINDOW, RATIO)
    assert (spec.n_tokens, spec.page_prefix, spec.cut) == (PROMPT + OUT - 1, WINDOW, 496)
    end = SimpleNamespace(key=th.tail_key(consumed, spec.n_tokens, None), first_token=PARKED_TOKEN)
    staged = SimpleNamespace(spec=spec, headers=[SimpleNamespace(end=end)], e1=e1, end_ok=True,
                             park=True, drop_end=lambda: None)
    return ta.Agreed(staged=staged, agreed=True, skip=skip)


@pytest.fixture
def agreed(monkeypatch):
    box = {}
    monkeypatch.setattr(ta, "_AGREED", box)
    with envs.FLLIPER_PDFLIP_TAIL_HANDOFF.override(True), envs.FLLIPER_PDFLIP_TAIL_ADOPT.override(True):
        yield box


# ------------------------------------------------------------ the peek
def test_peek_names_what_the_target_forward_computes(agreed):
    r = _park_req()
    assert ta.peek_compute_tokens(r, RESUME) is None  # no agreed tail: today's extend
    agreed[r.rid] = _park_agreed(r)
    assert ta.peek_compute_tokens(r, RESUME) == 0  # E2 skip: no target forward
    assert r.rid in agreed  # a peek: the commit still finds its entry
    # a batch that holds a forward refuses the skip; with END-only parts
    # (e1=absent(fold), every park tail in y3p) nothing is taken at all
    assert ta.peek_compute_tokens(r, RESUME, batch_empty=False) is None
    # E1 (a P hand-off with its state at c): the forward runs [c, fill)
    agreed[r.rid] = _park_agreed(r, skip=False, e1=True)
    assert ta.peek_compute_tokens(r, RESUME) == len(r.full_untruncated_fill_ids) - 496
    # outside the park window the tail is not taken (the prefix class of
    # pdflip-4-8 '2368!in[3904,4444)')
    assert ta.peek_compute_tokens(r, WINDOW - PAGE) is None


# ----------------------------------------------------------- the adder
class _Adder(PrefillAdder):
    pass


def _adder(rem_chunk=64, can_run_list=(), skip_taken=False):
    ad = object.__new__(_Adder)
    ad.page_size = PAGE
    ad.rem_chunk_tokens = rem_chunk
    ad.can_run_list = list(can_run_list)
    ad.pdflip_skip_extend_taken = skip_taken
    return ad


def test_ep54_a_narrowed_chunk_fits_a_skip_resume(agreed):
    """ep54 23:10:19: chunk 64, pdflip-24-27 uncached 122 -> ceil 128 > 64 ->
    chunked, no tail, 1955 + 2427 ms. With the tail the forward computes 0."""
    r = _park_req()
    agreed[r.rid] = _park_agreed(r)
    uncached = len(r.full_untruncated_fill_ids) - RESUME
    input_tokens = -(-uncached // PAGE) * PAGE
    assert input_tokens == 128 > 64
    ad = _adder()
    assert ad._pdflip_tail_fit_tokens(r, input_tokens) == 0
    with envs.FLLIPER_PDFLIP_TAIL_FIT_ON_COMPUTE.override(False):
        assert ad._pdflip_tail_fit_tokens(r, input_tokens) == 128  # the old test


def test_without_a_takeable_tail_the_fit_is_the_whole_extend(agreed):
    r = _park_req()
    assert _adder()._pdflip_tail_fit_tokens(r, 128) == 128  # nothing agreed
    agreed[r.rid] = _park_agreed(r)
    # a batch already holding a forward: the skip is refused, END-only has no
    # E1 -> the chunked path stays (the request would run a real extend)
    assert _adder(can_run_list=[object()])._pdflip_tail_fit_tokens(r, 128) == 128
    # behind a skip the batch counts as empty for the next skip (H24c)
    assert _adder(can_run_list=[object()], skip_taken=True)._pdflip_tail_fit_tokens(r, 128) == 0


def test_the_chunk_pays_the_computed_tokens_the_kv_the_whole():
    ad = _adder()
    ad.rem_total_token_offset = ad.cur_rem_token_offset = 0
    ad.rem_mamba_slots = None
    ad.rem_input_tokens = 1 << 20
    ad.is_hybrid_swa = False
    ad.dllm_config = None
    ad.log_hit_tokens = ad.log_input_tokens = 0
    ad.reprocessed_log_hit_tokens = ad.reprocessed_log_input_tokens = 0
    ad._update_prefill_budget(496, 128, 0, False, computed_input_len=0, chunk_charge=0)
    assert ad.rem_chunk_tokens == 64  # the next resume still finds the narrowed chunk
    assert ad.cur_rem_token_offset == 128 + PAGE  # the commit allocates [prefix, N) all the same
    assert ad.rem_input_tokens == (1 << 20) - 128
    ad._update_prefill_budget(496, 64, 0, False)
    assert ad.rem_chunk_tokens == 0  # without a tail: unchanged charge


def test_add_one_req_fits_and_charges_on_fit_tokens():
    src = inspect.getsource(PrefillAdder.add_one_req)
    fit = src.index("fit_tokens = self._pdflip_tail_fit_tokens(req, input_tokens)")
    gate = src.index("tail_adopt.skip_joinable(req, len(req.prefix_indices))")
    whole = src.index("elif self.rem_chunk_tokens is None or fit_tokens <= self.rem_chunk_tokens:")
    plan = src.index("_tail = tail_adopt.plan_adopt(")
    assert fit < gate < whole < plan
    assert "fit_tokens > self.rem_chunk_tokens" in src[fit:gate]
    belt = src[plan:plan + 700]
    assert "_tail is None" in belt and "input_tokens > self.rem_chunk_tokens" in belt
    assert "chunk_charge=None if (_ea_forced or _tail is None) else fit_tokens" in src
