"""DUAL-TP3PP3 stage 2: admit a P-prefilled request straight into D's decode batch.

USER ORDER 01.10.: in the dual layout D never stops decoding. Since ef9b3a899c P
prefills everything; what D still computes is the N-1 anchor token (P holds the
last prompt token back), and it computes it as an eager extend forward that
costs 208-446 gpu-ms per admission (metal dual1m ...10011343, 'Prefill rank
batch #new-token: 2' at cached 4059/6524/39185) -- 7-15 decode rounds of a
32 ms median, i.e. the decode still stops.

THE STATE IDENTITY. A request whose KV covers prompt[0..N-2] and whose last
prompt token prompt[N-1] is still without KV is the same state a running decode
request is in after a verify round: KV committed up to its pending input token,
the pending token not yet written. sglang's decode invariant
``committed KV == seqlen - 1`` (seqlen = len(prompt) + len(output)) holds for it
with an EMPTY output: N + 0 - 1 = N - 1. So the request can join the running
batch WITHOUT a forward; the next (graphed) DFlash round computes prompt[N-1]'s
KV and logits together with the draft block and samples output token 1. The
draft prefix is cold as it already is for every P-prefilled request in the dual
layout today (#993 zero fill, 'WEG2 DRAFT-COLD ... rounds_owed=1'; accept len
4.22 mean on dual1m) -- variant (b), no draft KV on P.

THIS MODULE is the riegel and the bookkeeping, pure (no torch): who may join,
and what the joined request's numbers are. The scheduler's admission branch
and the DFlash spec-state merge consume it (stage 2 steps 2-3).
Switch: ``SGLANG_WEG2_DUAL_DECODE_JOIN`` (default 0 = off: every request takes
the 1-token extend exactly as before).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

JOIN_ENV = "SGLANG_WEG2_DUAL_DECODE_JOIN"

#: The verdicts. ``join``: straight into the decode batch. ``extend``: the old
#: path (switch off, not the dual layout, not a P-prefilled leg) -- the
#: pre-stage-2 behaviour, never an error. ``refuse``: the dual layout with the
#: switch on, and the request cannot be joined although it reached D --
#: refused by name instead of silently extended (the user's "laut verweigern").
JOIN, EXTEND, REFUSE = "join", "extend", "refuse"


def join_enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(JOIN_ENV, "0") or "0").strip().lower() not in ("0", "", "false", "no", "off")


def _n(req: Any, field: str) -> int:
    """len() of a sequence field, 0 when absent -- never ``x or ()``, which
    raises on a multi-element tensor (dual1g W17, ead403b7ab)."""
    v = getattr(req, field, None)
    return 0 if v is None else int(len(v))


@dataclass(frozen=True)
class JoinVerdict:
    verdict: str
    reason: str

    def __bool__(self) -> bool:  # truthy only for a join
        return self.verdict == JOIN


def join_verdict(req: Any, *, dual_layout: bool, spec_is_dflash: bool, prefix_len: int,
                 enabled: Optional[bool] = None) -> JoinVerdict:
    """May ``req`` join D's running decode batch without a forward?

    ``prefix_len`` is the KV D holds for it after match_prefix (target pool +
    store read) -- the only point at which the uncached extent exists.
    """
    on = join_enabled() if enabled is None else bool(enabled)
    if not on:
        return JoinVerdict(EXTEND, "switch off")
    if not dual_layout:
        return JoinVerdict(EXTEND, "not the dual layout")
    n = _n(req, "origin_input_ids")
    if n < 2:
        return JoinVerdict(EXTEND, f"prompt of {n} token(s): nothing for P to have prefilled")
    out = _n(req, "output_ids")
    if out:
        return JoinVerdict(REFUSE, f"carries {out} output token(s) -- a resumed/retracted request "
                                   f"is not a fresh P hand-off")
    uncached = n - int(prefix_len)
    if uncached != 1:
        return JoinVerdict(REFUSE, f"uncached={uncached} (prefix {int(prefix_len)} of {n}); a join "
                                   f"needs exactly the N-1 anchor token outstanding -- the rest is "
                                   f"P's (W31 -> P), never a silent D prefill")
    if not spec_is_dflash:
        return JoinVerdict(REFUSE, "speculative algorithm is not DFLASH: the decode-batch spec "
                                   "state is built for DFlash only")
    if getattr(req, "multimodal_inputs", None) is not None:
        return JoinVerdict(REFUSE, "multimodal request: its embeddings exist on P only")
    if getattr(req, "return_logprob", False):
        start = getattr(req, "logprob_start_len", -1)
        start = -1 if start is None else int(start)
        if 0 <= start < n:
            return JoinVerdict(REFUSE, f"input logprobs from {start} requested: the prompt token's "
                                       f"logprob is not produced by a decode round")
    return JoinVerdict(JOIN, "P prefilled [0, N-1), D's next round computes the anchor token")


@dataclass(frozen=True)
class JoinState:
    """The joined request's numbers, as the decode round expects them."""
    committed_kv: int       # KV rows D holds = seq_lens entry = N - 1
    pending_token: int      # the round's input token = prompt[N-1] (NOT an output)
    cached_tokens_add: int  # usage accounting: the prefix D did not compute
    seqlen: int             # len(prompt) + len(output) = N (invariant: committed == seqlen - 1)


def join_state(req: Any, prefix_len: int) -> JoinState:
    _ids = getattr(req, "origin_input_ids", None)
    ids = [] if _ids is None else list(_ids)
    n = len(ids)
    if n - int(prefix_len) != 1 or _n(req, "output_ids"):
        raise ValueError(f"join_state for a request that may not join (n={n} prefix={prefix_len} "
                         f"outputs={_n(req, "output_ids")})")
    already = int(getattr(req, "already_computed", 0) or 0)
    return JoinState(committed_kv=n - 1, pending_token=int(ids[-1]),
                     cached_tokens_add=max(0, int(prefix_len) - already), seqlen=n)


def apply_join_accounting(req: Any, state: JoinState) -> None:
    """The bookkeeping the skipped extend would have done, and nothing else:
    the prefix counts as cached (it was computed on P, not by this group), the
    request is marked as computed up to its committed KV, and NO output token is
    appended -- the pending token is the prompt's last token, and the first
    output token is the one the next decode round samples."""
    req.cached_tokens = int(getattr(req, "cached_tokens", 0) or 0) + state.cached_tokens_add
    req.cached_tokens_device = int(getattr(req, "cached_tokens_device", 0) or 0) + state.cached_tokens_add
    req.already_computed = state.committed_kv
    setattr(req, "_weg2_decode_joined", True)


# ---------------------------------------------------------------------------
# Step 2: the scheduler side (torch, imported lazily so the riegel above stays
# importable without a GPU stack).
# ---------------------------------------------------------------------------
DUAL_LAYOUT_ENV = "SGLANG_WEG2_DUAL_LAYOUT"


def dual_layout_rank(env=None) -> bool:
    """The launcher sets SGLANG_WEG2_DUAL_LAYOUT=1 on every rank of both groups."""
    e = os.environ if env is None else env
    return str(e.get(DUAL_LAYOUT_ENV, "") or "").strip() == "1"


def _prefix_len(req) -> int:
    # never `x or ()` on a tensor: an empty prefix_indices tensor raised on every
    # D rank in dual1g (ead403b7ab, W17) -- len() of what is there, 0 for None
    p = getattr(req, "prefix_indices", None)
    return 0 if p is None else int(len(p))


def split_join_reqs(can_run_list, *, spec_is_dflash: bool, exclude=(), enabled=None, env=None):
    """``(join, rest, fallbacks)``: which admitted requests join the decode batch.

    ``fallbacks`` are ``(req, reason)`` of requests the dual layout would have
    joined but cannot -- they stay on the extend path (they carry at most the
    one anchor token here: ef9b3a899c's X=1 refused every larger remainder at
    D's X gate before admission) and are named in the log by the caller.
    """
    on = join_enabled(env) if enabled is None else bool(enabled)
    if not on:
        return [], list(can_run_list), []
    dual = dual_layout_rank(env)
    join, rest, fallbacks = [], [], []
    excl = {id(x) for x in (exclude or ())}
    for req in can_run_list:
        if id(req) in excl:
            rest.append(req)
            continue
        v = join_verdict(req, dual_layout=dual, spec_is_dflash=spec_is_dflash,
                         prefix_len=_prefix_len(req), enabled=True)
        if v.verdict == JOIN:
            join.append(req)
        else:
            rest.append(req)
            if v.verdict == REFUSE:
                fallbacks.append((req, v.reason))
    return join, rest, fallbacks


def convert_to_joined(batch, future_map, enable_overlap: bool) -> None:
    """Turn a PREPARED extend batch of join requests into decode-ready state,
    WITHOUT a forward.

    ``prepare_for_extend`` already did everything the extend path does before
    its forward -- the req_to_token rows of the prefix, the Mamba slot from
    P's anchor, one KV slot for the pending token at position N-1 (written by
    the next round), the usage accounting (cached_tokens += prefix,
    already_computed) -- so the numbers match the extend path exactly. What
    the skipped forward would have produced is replaced by the decode
    invariant: committed KV = N-1, pending input = prompt[N-1], no output
    token. The DFlash draft input is the shape ``_make_next_draft_input_prefill``
    builds after an extend, with N-1 and the prompt token instead of N and a
    sampled token; under overlap it is published/stashed into the future map
    under the request's pool index exactly like a prebuilt PD-decode batch.
    """
    import torch

    from sglang.srt.managers.overlap_utils import RelayPayload
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

    lens, bonus = [], []
    for req in batch.reqs:
        st = join_state(req, _prefix_len(req))
        req.kv_committed_len = st.committed_kv
        req._kvc_src = "decode_join"
        setattr(req, "_weg2_decode_joined", True)
        lens.append(st.committed_kv)
        bonus.append(st.pending_token)
    dev = batch.device
    batch.seq_lens = torch.tensor(lens, dtype=torch.int64, device=dev)
    batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    batch.seq_lens_sum = int(sum(lens))
    batch.forward_mode = ForwardMode.DECODE
    batch.input_ids = None
    batch.out_cache_loc = None
    batch.mamba_track_indices = None
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None
    spec = make_draft_input_v2(
        bonus_tokens=torch.tensor(bonus, dtype=torch.int64, device=dev),
        new_seq_lens=batch.seq_lens,
    )
    if enable_overlap:
        spec.future_indices = batch.req_pool_indices
        future_map.publish(spec.future_indices, batch.seq_lens)
        future_map.stash(spec.future_indices, RelayPayload.from_draft_input(spec))
    batch.spec_info = spec


def merge_joined(running_batch, joined: list):
    """Merge the ready join batches into the running batch (the hisparse
    staging->decode transition's shape). Returns the running batch."""
    for jb in joined:
        if running_batch is None or running_batch.is_empty():
            running_batch = jb
        else:
            running_batch.merge_batch(jb)
    return running_batch
