# SPDX-License-Identifier: Apache-2.0
"""#631: give a PP-prefilled request draft state at the PP->TP cutover.

THE PROBLEM THIS EXISTS FOR
---------------------------
Speculation belongs to the TP decode phase. The PP prefill phase carries
no draft worker at all -- ``build_flip_draft_worker`` returns None there,
and the cutover documents the PP phase as "bit-for-bit the state an
instance without speculation has". So a request that prefills in PP and is
carried across the cutover has never had a ``draft_extend``, and #631's
central path is exactly that: EVERY request is supposed to prefill in PP
and decode in TP.

Measured consequence, 2026-08-09 03:32:14Z, all three ranks, one pass
after a committed pp_to_tp that carried one request:

    eagle_worker_v2.draft -> eagle_draft_cuda_graph_runner.execute
    -> foreach_copy: output with shape [1, 1] doesn't match the broadcast
       shape [0, 1]                                        -> SIGQUIT

The 0-row source is ``EagleDraftInput.create_idle_input``, which
``forward_batch_generation`` installs when it finds ``spec_info is None``.
An idle input is the right answer for an idle batch and the wrong answer
for a batch of bs real requests.

Both mitigations tried before this module were measured worthless (v4 §5):
ARM AND WAIT parks the full deadline and abandons under any sustained
decode (cadence zero, plus a fault on the abandon path), and DO NOT ARM
pins the instance in PP at 16.8 tok/s against the 113 tok/s TP+MTP does.

WHAT THE HARDWARE ACTUALLY REQUIRES, and it is only two things
--------------------------------------------------------------
1.  THE DRAFT KV IS INDEXED BY THE TARGET'S SLOT IDS. ``req_to_token_pool``
    and ``token_to_kv_pool_allocator`` are SHARED with the target
    (tp_worker.alloc_memory_pool); only the KV buffers themselves are
    separate. There is no draft req_to_token, no draft seq_len, no draft
    allocated-len -- draft-pool row ``s`` is the draft's K/V for whatever
    token occupies target-pool row ``s``.

    So a carried request's draft rows over its whole prefix are ALLOCATED
    but hold WHATEVER THE PREVIOUS OCCUPANT OF THOSE SLOTS WROTE. They are
    read unconditionally: the draft decode's kv_indices come from
    ``req_to_token[req_pool_idx, :seq_lens]`` with the TARGET's seq_lens
    (base_spec_worker.py:227, triton_backend.py:3020). Nothing raises.
    This is the silent-garbage direction the handoff warned about, and it
    is why this module SCRUBS rather than trusting the pool.

    Zero, not stale: zero is a defined and DETERMINISTIC value, and this
    fork's determinism pins are worth more than the arbitrary bytes of
    some finished request. It is not a correctness argument -- the target
    verifies every proposed token either way -- it is a reproducibility
    one, and it also removes the chance of an fp8 NaN/Inf reaching the
    draft's softmax.

2.  THE DRAFT NEEDS A SEED, AND A SEED NEEDS A HIDDEN STATE. The draft
    chain starts from ``EagleDraftInput(hidden_states=..., topk_index=...)``
    and the PP phase captured no hidden states (``capture_hidden_mode`` is
    NULL without speculation). There is no hidden state to carry.

    So the first post-flip round runs WITHOUT drafting:
    ``_build_trivial_verify_input`` builds a 1-node verify rooted at the
    previous bonus token, which "the kernel always accepts ... functionally
    a plain decode", and it captures hidden states in FULL. The ordinary
    ``_draft_extend_for_decode`` that follows every verify then seeds the
    real draft chain off that round's hidden states. From round two the
    carried request is an ordinary speculating request.

    This is not a new mechanism. It is the kv-session-offload BOOTSTRAP
    TICK (kv_session_offload.spec_in_tick_bootstrap_seed +
    eagle_worker_v2:2144) applied at a different boundary, for the same
    reason: a request that must rejoin a speculating batch with no
    captured hidden yet.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not reconstruct the draft's prefix KV. That needs the target's
hidden states for the whole prefix, which the PP phase discards, and
capturing them costs L x hidden_size x 2 bytes per resident request (543
MiB for one 53k-token prompt) plus a cutover broadcast -- against a budget
this strand is trying to TIGHTEN by 7 GiB. The scrubbed prefix therefore
reads as zeros and the carried request's ACCEPTANCE starts low and
recovers as real draft rows accumulate behind the cutover point. Its
ANSWER is unaffected: the target verifies every proposed token, which is
what makes a low-acceptance draft a throughput question and never a
correctness one.

That trade is the reason this cut is measurable rather than believed: the
number to report after a flip into TP is accept length, and the reason it
is allowed to be lower than the 3.7-4.0 an uncarried request gets.

CARRYING THE LAST TARGET HIDDEN INSTEAD (v3 §3 option 2) was considered
and rejected on cost at the OTHER end: it needs ``capture_hidden_mode =
LAST`` on the PP phase's decode batches, which perturbs the default path
this feature is required to leave bit-for-bit unchanged -- and a
capture-mode change on decode batches is exactly what makes
``recapture_if_needed`` tear down and re-record every plain decode graph
(decode_cuda_graph_runner:791). One eager round per flip is cheaper than
that, and it is confined to the phase that owns speculation.

RANK UNIFORMITY. Every quantity here is replicated by construction: the
slot ids come from the shared ``req_to_token`` (rank-replicated between
rounds, DESIGN_631 3.5), the token ids come from the requests' own
``output_ids``, and the scrub writes zeros -- so every rank performs the
same writes to its own shard without a collective. Nothing here needs a
broadcast, which is deliberate: the cutover is inside the no-return
region and a new collective there is a new wedge class.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-DRAFT"

# Set on every Req carried into a speculating TP phase, cleared by the
# worker after the bootstrap round has run. Per-REQUEST rather than
# per-batch because a batch may legitimately mix carried requests with
# ones admitted after the cutover.
BOOTSTRAP_ATTR = "phase_flip_needs_draft_bootstrap"


class DraftBootstrapError(RuntimeError):
    """A carried request cannot be given draft state.

    Raised rather than worked around: the alternative is the draft graph
    runner reading a 0-row or stale-row input one pass later, which is the
    failure this module exists to remove.
    """


def _reqs_of(batch) -> List:
    return list(getattr(batch, "reqs", None) or [])


def draft_kv_pool(draft_worker) -> Optional[Any]:
    """The draft model's own KV buffers, or None.

    ``draft_worker`` is the EAGLE/NEXTN spec worker the cutover arms
    (``EAGLEWorkerV2``); its ``.draft_worker`` is the EagleDraftWorker and
    ``.draft_runner`` that worker's ModelRunner. Reached defensively -- a
    phase-flip instance may be built without speculation at all, and this
    module must then be a no-op rather than an AttributeError inside the
    no-return region.
    """
    if draft_worker is None:
        return None
    inner = getattr(draft_worker, "draft_worker", None)
    runner = getattr(inner, "draft_runner", None)
    if runner is None:
        runner = getattr(inner, "model_runner", None)
    return getattr(runner, "token_to_kv_pool", None)


def committed_slots(scheduler, batch) -> List[torch.Tensor]:
    """Per request, the token slots whose TARGET KV is committed.

    The length is the batch's ``seq_lens`` and NOT ``req.seqlen``: seqlen
    counts the last sampled output token, whose KV row is not written yet,
    so scrubbing to seqlen would touch one row past the allocation -- an
    out-of-bounds write into another request's slot, dressed as a scrub.
    """
    reqs = _reqs_of(batch)
    if not reqs:
        return []
    req_to_token = scheduler.req_to_token_pool.req_to_token
    seq_lens = getattr(batch, "seq_lens", None)
    if seq_lens is None:
        raise DraftBootstrapError(
            f"{LOG_PREFIX} carried batch has no seq_lens; refusing to guess "
            f"the committed KV length of {len(reqs)} request(s)"
        )
    lens = [int(x) for x in seq_lens.tolist()]
    if len(lens) != len(reqs):
        raise DraftBootstrapError(
            f"{LOG_PREFIX} carried batch has {len(reqs)} request(s) but "
            f"{len(lens)} seq_lens entries; the carry and the tensors "
            f"disagree about the resident set"
        )
    out: List[torch.Tensor] = []
    for req, n in zip(reqs, lens):
        if n <= 0:
            out.append(req_to_token.new_empty((0,)))
            continue
        out.append(req_to_token[req.req_pool_idx, :n])
    return out


def draft_kv_layer_ids(pool) -> List[int]:
    """The layer ids whose KV buffers this pool actually holds.

    NOT a range derived from a count, because on this model it is not one.
    The draft pool here is a HYBRID pool: a Qwen3.6 layer stack mixes full
    attention with linear/GDN layers, and only the full-attention ones have
    KV rows at all. Such a pool carries no ``layer_num`` -- it carries
    ``full_attention_layer_id_mapping``, a dict from the model's layer id
    to the inner pool's -- and ``get_key_buffer`` REFUSES any id outside
    it. A range-based scrub therefore either refuses outright (measured
    08:56:39Z on all three ranks: "draft KV pool reports 0 layers") or, on
    a pool that happens to expose a count, walks ids the pool would reject.

    Order of authority: the explicit mapping first, a declared count
    second, and a loud refusal third. The refusal is deliberate -- a scrub
    that silently covers nothing leaves the stale rows this module exists
    to remove, and the failure would then be a quiet acceptance collapse
    rather than an error.
    """
    if pool is None:
        return []
    mapping = getattr(pool, "full_attention_layer_id_mapping", None)
    if mapping:
        return sorted(int(k) for k in mapping.keys())
    n_layers = int(getattr(pool, "layer_num", 0) or 0)
    if n_layers > 0:
        start = int(getattr(pool, "start_layer", 0) or 0)
        return list(range(start, start + n_layers))
    raise DraftBootstrapError(
        f"{LOG_PREFIX} draft KV pool {type(pool).__name__} declares neither "
        f"full_attention_layer_id_mapping nor a non-zero layer_num; refusing "
        f"to scrub a pool whose geometry cannot be read"
    )


def scrub_draft_kv(pool, slot_rows: Sequence[torch.Tensor]) -> tuple:
    """Zero the draft KV rows of the given token slots.

    Returns (rows zeroed, layer ids scrubbed) -- the layer ids are returned
    rather than counted so the boot log records WHICH layers were covered.
    On a hybrid draft stack that is the difference between a scrub that did
    its job and one that covered a fraction of the layers, and the two are
    indistinguishable from a row count alone.

    Pool-generic: MLA-style pools return the same tensor for key and value,
    so the value write is skipped when it aliases the key (zeroing twice is
    harmless, but the guard keeps the traffic honest).
    """
    if pool is None:
        return 0, []
    layer_ids = draft_kv_layer_ids(pool)
    rows = 0
    for slots in slot_rows:
        if slots.numel() == 0:
            continue
        idx = slots.to(torch.long)
        for layer_id in layer_ids:
            k = pool.get_key_buffer(layer_id)
            k[idx] = 0
            v = pool.get_value_buffer(layer_id)
            if v is not None and v.data_ptr() != k.data_ptr():
                v[idx] = 0
        rows += int(slots.numel())
    return rows, layer_ids


def pending_tokens(batch) -> List[int]:
    """The token each carried request has NOT yet fed through the model.

    THE ROOT OF THE BOOTSTRAP VERIFY, and the trivial verify ALWAYS ACCEPTS
    ITS ROOT -- so a wrong value here is committed unconditionally, with no
    target check to catch it. It is the one quantity in this module that
    can corrupt an answer rather than merely cost acceptance.

    Taken from ``batch.input_ids``, not from ``output_ids[-1]``.
    ``input_ids`` on a decode batch IS the pending token by construction --
    "set at the end of the previous run_batch" -- so it agrees with
    ``seq_lens`` whatever the boundary looks like. ``output_ids`` is a
    different clock: it is written by the batch RESULT processor, so at a
    boundary where a forward has advanced the KV but its result has not
    been processed the two disagree by exactly one token, in whichever
    direction the timing fell.

    That disagreement is not hypothetical. It is what the counting probe
    caught at 09:21:59Z with the scrub disabled -- '...18 19 0 21 22...',
    the " 2" of "20" gone, everything after it correct, and in another
    request a duplicated "28 28": one token lost and one token repeated,
    the two directions of the same off-by-one.

    Falls back to ``output_ids[-1]`` only when the batch carries no usable
    input_ids, and refuses when neither exists rather than rooting a verify
    at a guess.
    """
    reqs = _reqs_of(batch)
    ids_t = getattr(batch, "input_ids", None)
    pending: Optional[List[int]] = None
    if ids_t is not None:
        try:
            flat = [int(x) for x in ids_t.tolist()]
        except AttributeError:
            flat = [int(x) for x in ids_t]
        if len(flat) == len(reqs):
            pending = flat
    out: List[int] = []
    for i, req in enumerate(reqs):
        if pending is not None:
            out.append(pending[i])
            continue
        tail = getattr(req, "output_ids", None) or getattr(
            req, "origin_input_ids", None
        ) or []
        if not tail:
            raise DraftBootstrapError(
                f"{LOG_PREFIX} carried request {getattr(req, 'rid', '?')} has "
                f"no pending token: the batch's input_ids do not match the "
                f"resident set and the request has no tokens of its own"
            )
        out.append(int(tail[-1]))
    return out


def bootstrap_clock_report(batch) -> str:
    """The two clocks, side by side, for the boot log.

    Written because the off-by-one above was diagnosed from generated TEXT
    rather than from the state, which cost a boot. Whoever reads this line
    can see directly whether seq_lens, output_ids and input_ids agree at
    the cutover, per request.
    """
    reqs = _reqs_of(batch)
    seq_lens = getattr(batch, "seq_lens", None)
    lens = [int(x) for x in seq_lens.tolist()] if seq_lens is not None else []
    ids_t = getattr(batch, "input_ids", None)
    try:
        inp = [int(x) for x in ids_t.tolist()] if ids_t is not None else []
    except AttributeError:
        inp = list(ids_t or [])
    parts = []
    for i, req in enumerate(reqs):
        out_ids = getattr(req, "output_ids", None) or []
        parts.append(
            "%s: seq_lens=%s seqlen=%s out[-2:]=%s input_id=%s"
            % (
                getattr(req, "rid", "?")[:8],
                lens[i] if i < len(lens) else "?",
                len(getattr(req, "origin_input_ids", None) or []) + len(out_ids),
                list(out_ids[-2:]),
                inp[i] if i < len(inp) else "?",
            )
        )
    return " | ".join(parts)


def build_bootstrap_draft_input(scheduler, batch, topk: int):
    """A shape-complete ``EagleDraftInput`` for a batch with no draft state.

    Only ``bonus_tokens`` is READ this round -- the trivial verify roots
    its single node there (eagle_worker_v2:2241). The other fields exist
    because ``EagleDraftInput.merge_batch`` takes ``len(self.topk_index)``
    unconditionally (eagle_info.py:262): a seed carrying bonus_tokens
    alone would crash the moment a request admitted after the cutover
    merged into the carried batch. They are zeros, and they are never read
    before ``_draft_extend_for_decode`` replaces the whole input at the end
    of the bootstrap round.
    """
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    reqs = _reqs_of(batch)
    device = scheduler.device
    last_tokens = pending_tokens(batch)

    bs = len(reqs)
    bonus = torch.tensor(last_tokens, dtype=torch.int64, device=device)
    hidden_size, dtype = _draft_hidden_spec(scheduler)
    return EagleDraftInput(
        bonus_tokens=bonus,
        hidden_states=(
            torch.zeros((bs, hidden_size), device=device, dtype=dtype)
            if hidden_size is not None
            else None
        ),
        topk_p=torch.zeros((bs, topk), device=device, dtype=torch.float32),
        topk_index=bonus.view(bs, 1).expand(bs, topk).contiguous(),
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )


def _draft_hidden_spec(scheduler):
    from sglang.srt.speculative.eagle_utils import (
        get_draft_recurrent_hidden_state_spec,
    )

    worker = getattr(scheduler, "draft_worker", None)
    inner = getattr(worker, "draft_worker", None)
    runner = getattr(inner, "draft_runner", None)
    if runner is None:
        return None, None
    return get_draft_recurrent_hidden_state_spec(runner)


def arm_draft_bootstrap(scheduler, batch, draft_worker) -> dict:
    """Cutover leg: scrub the stale draft KV and seed the carried batch.

    Called from ``build_production_flip_cutover`` for the PP->TP direction
    only, AFTER the active stack swap -- the draft worker must already be
    the armed one, because its pool is what gets scrubbed.

    A no-op with an honest report when there is nothing to do (no
    speculation, or nothing resident), because the flip that commits on an
    idle server is the common case and must not pay for this.
    """
    reqs = _reqs_of(batch)
    pool = draft_kv_pool(draft_worker)
    if not reqs or pool is None:
        return {"reqs": len(reqs), "rows": 0, "armed": False}

    slot_rows = committed_slots(scheduler, batch)
    # THE SCRUB IS SEPARABLE ON PURPOSE, and this switch is a measuring
    # instrument rather than a preference. Content correctness does not
    # depend on it: the target verifies every proposed token, so stale
    # draft rows cost acceptance and nothing else. A corrupted ANSWER
    # therefore cannot be caused by skipping the scrub -- but it CAN be
    # caused by performing one, if the draft pool turned out to alias the
    # target's KV bytes. Running the probe once with the scrub off
    # separates those two, and nothing else does it as cheaply.
    if os.environ.get("SGLANG_PHASE_FLIP_DRAFT_SCRUB", "1") == "0":
        rows, layer_ids = 0, []
        logger.warning(
            "%s SCRUB DISABLED by SGLANG_PHASE_FLIP_DRAFT_SCRUB=0: the "
            "carried requests' draft KV keeps the previous occupants' "
            "bytes. Acceptance only -- answers are unaffected because the "
            "target verifies every proposed token.",
            LOG_PREFIX,
        )
    else:
        rows, layer_ids = scrub_draft_kv(pool, slot_rows)

    topk = int(getattr(draft_worker, "topk", 1) or 1)
    batch.spec_info = build_bootstrap_draft_input(scheduler, batch, topk)
    for req in reqs:
        setattr(req, BOOTSTRAP_ATTR, True)

    logger.info("%s cutover clocks -- %s", LOG_PREFIX, bootstrap_clock_report(batch))
    logger.info(
        "%s bootstrapped %d carried request(s) into the speculating TP "
        "phase: %d stale draft KV row(s) scrubbed across layer(s) %s of "
        "%s, seed installed; the first decode round runs a 1-node verify "
        "(no draft) and its hidden states seed the real chain",
        LOG_PREFIX,
        len(reqs),
        rows,
        layer_ids,
        type(pool).__name__,
    )
    return {
        "reqs": len(reqs),
        "rows": rows,
        "layers": layer_ids,
        "armed": True,
    }


def retune_carried_batches_for_phase(scheduler, spec_algorithm) -> int:
    """Point every carried batch's OWN spec_algorithm at the new phase.

    ``ScheduleBatch.spec_algorithm`` is a FIELD, copied in at construction
    from the scheduler that built the batch (``init_new``). A batch built
    in the PP phase therefore carries NONE across the cutover no matter
    what step 7 sets on the scheduler, and the batch's field is not
    decoration: ``prepare_for_decode`` branches on it and hands decode
    preparation to ``spec_prepare_for_decode`` only when it is non-NONE.

    Left stale, the two halves of a single decode round disagree about
    which phase they are in -- the batch prepares itself the plain way
    (one token per request, no draft cache locs) and the worker then runs
    the speculating path over it. It is the same class of defect as the
    boot-cached ``batch_result_processor`` the cutover already rebuilds,
    arriving through the batch instead of through a component.

    BOTH DIRECTIONS. The TP->PP leg has the mirror-image hole: a batch
    that keeps a speculating algorithm while the phase it lands in has no
    draft worker at all. Nothing has hit it yet only because every flip
    that has ever committed committed on an idle server.

    Returns the number of batches retuned.
    """
    n = 0
    for batch in _harvest(scheduler):
        if batch is None:
            continue
        if getattr(batch, "spec_algorithm", None) is spec_algorithm:
            continue
        batch.spec_algorithm = spec_algorithm
        n += 1
    return n


# Every scheduler attribute that can hand a ScheduleBatch to the PP event
# loop. This list is DELIBERATELY WIDER than harvest_resident_batches():
# that harvest answers "which batches hold requests I must carry", and it
# filters out empty batches and never looks at ``last_batch``. The question
# here is the opposite one -- "which batches can the next loop iteration
# still REACH" -- and ``last_batch`` is precisely the handle that killed
# PP0 at 2026-08-09 20:31:48 (corpse I below).
_REACHABLE_BATCH_ATTRS = (
    "running_batch",
    "last_batch",
    "cur_batch",
    "cur_batch_for_debug",
)


def _reachable_batches(scheduler):
    """Every batch object the event loop can still reach on this rank.

    Deduplicated by identity: ``running_batch`` is routinely an ALIAS of a
    ``running_mbs`` slot, and clearing the same object twice would
    miscount. Empty batches are INCLUDED on purpose -- an empty batch with
    a stale spec_info is still a live merge target once requests land in
    it.
    """
    out = []
    seen = set()

    def _take(batch):
        if batch is None or id(batch) in seen:
            return
        seen.add(id(batch))
        out.append(batch)

    for mb in getattr(scheduler, "running_mbs", None) or []:
        _take(mb)
    for attr in _REACHABLE_BATCH_ATTRS:
        _take(getattr(scheduler, attr, None))
    return out


def clear_spec_info_for_unspeculated_phase(scheduler) -> tuple:
    """TP->PP leg: leave NO TP draft state reachable from any batch.

    CORPSE I -- MEASURED ON METAL 2026-08-09 20:31:48Z, epoch 8, PP0 dead,
    two requests decoding and a prefill pending across the cutover:

        get_next_batch_to_run (scheduler.py:4368)
          -> running_batch.merge_batch(last_batch)   (schedule_batch:3399)
          -> self.spec_info.merge_batch(other.spec_info)
          -> eagle_info.py:271  len(spec_info.topk_index)
        AttributeError: 'NoneType' object has no attribute 'topk_index'

    What was INFERRED and is now falsified: this module's step-7b comment
    used to read "the TP->PP leg is flipping speculation OFF, and a request
    carried into a phase with no drafter needs no draft state -- its
    spec_info is simply not read there." It IS read there. Not by a
    drafter, which indeed does not exist in PP, but by ``merge_batch``,
    which branches on ``if self.spec_info:`` (TRUTHINESS OF THE CARRIED
    BATCH ALONE) and then dereferences ``other.spec_info`` unconditionally.
    The carried TP batch is the truthy self; the fresh PP prefill batch,
    built in a phase with no drafter, is the None other.

    Why the retune was not enough: ``retune_carried_batches_for_phase``
    rewrites the ``spec_algorithm`` FIELD and nothing else. Nothing reads
    that field on the merge path. The 20:31:48 log shows the retune
    running and reporting success one line before the crash.

    Why the fix is HERE and not a None-check at eagle_info.py:271: a bare
    None-guard would let two batches merge while one silently drops its
    draft state -- a wrong-output bug replacing a crash. The seam is the
    producer of the inconsistency, so the seam is where it is removed. A
    loud assertion in ``merge_batch`` exists ADDITIONALLY (it raises, it
    never continues) so that a future hole in this reach is reported as
    itself rather than as an AttributeError three frames down.

    Why the acceptance run never saw it: 69 flips, and every one of them
    committed with either no resident decode or no arriving prefill. The
    trigger needs BOTH -- a carried batch with live TP spec_info AND a
    fresh extend batch built after the cutover -- which is exactly the
    epoch 6/7/8 back-to-back pattern under mixed load.

    Returns (batches cleared, rids whose batches were cleared).
    """
    cleared = 0
    rids = []
    for batch in _reachable_batches(scheduler):
        if getattr(batch, "spec_info", None) is None:
            continue
        batch.spec_info = None
        cleared += 1
        rids.extend(str(getattr(r, "rid", "?")) for r in _reqs_of(batch))
    return cleared, rids


def arm_draft_bootstrap_all_reachable(scheduler, draft_worker) -> list:
    """PP->TP leg: seed EVERY reachable non-empty batch, not just one.

    THE MIRROR OF CORPSE I, found by reading the other direction after the
    20:31:48 death rather than by another crash. ``arm_draft_bootstrap``
    was called with ``scheduler.running_batch`` alone. If a non-empty PP
    extend batch is sitting in ``last_batch`` at the cutover, the very next
    ``get_next_batch_to_run`` merges it into the freshly bootstrapped
    running_batch -- truthy self, None other, the identical AttributeError
    with the two roles swapped.

    EMPTY batches are correctly skipped: ``get_next_batch_to_run`` merges
    only ``if not last_batch.is_empty()``, and an empty running_batch is
    REPLACED by last_batch rather than merged with it. So an empty batch
    never reaches the dereference, and seeding one would install a draft
    input for zero requests.

    Returns one report dict per batch armed.
    """
    reports = []
    for batch in _reachable_batches(scheduler):
        if not _reqs_of(batch):
            continue
        reports.append(arm_draft_bootstrap(scheduler, batch, draft_worker))
    return reports


def _harvest(scheduler):
    from sglang.srt.managers.phase_flip_resident_carry import (
        harvest_resident_batches,
    )

    return harvest_resident_batches(scheduler)


def batch_needs_bootstrap(batch) -> bool:
    """True while any request in the batch still owes its bootstrap round.

    Whole-batch by design even though the mark is per-request: the trivial
    verify is a batch-level shape. A batch that mixes carried and fresh
    requests therefore spends ONE round not drafting for all of them, which
    costs the fresh ones a single speculation step and keeps the carried
    ones from reading a draft chain they do not have.
    """
    return any(getattr(r, BOOTSTRAP_ATTR, False) for r in _reqs_of(batch))


def clear_bootstrap(batch) -> int:
    """Clear the marks after the bootstrap round. Returns marks cleared."""
    n = 0
    for req in _reqs_of(batch):
        if getattr(req, BOOTSTRAP_ATTR, False):
            setattr(req, BOOTSTRAP_ATTR, False)
            n += 1
    return n
