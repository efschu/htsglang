# SPDX-License-Identifier: Apache-2.0
"""The D side of the transient vision form: D tokenizes images, never encodes them.

Under ``--weg2-vision transient`` group D boots like P: ``language_model_only``
(no tower) with the multimodal tokenizer ON, instead of the old
``--no-enable-multimodal``. Its leg 2 therefore carries the SAME ids as P's
leg 1 -- the image placeholders become the items' ``pad_value`` ids -- so D's
prefix match reaches P's stored pages across the image, and its decode gets
the request's ``mrope_position_delta``. Measured before this form, xsn411 D
(eb5d04453f): ``enable_multimodal=False``, ``json_model_override_args='{}'``
-- no mm_items at all, hence neither.

What D may never do is PREFILL an image position: it has no tower, and
``get_image_feature`` raises ``_require_visual`` inside the scheduler thread
-- the whole group dies (xsn405's shape, on D). The front routes every image
long (P prefills it, D reads it from the store), so an image inside D's
extent means the store read did not cover it. :func:`verdict` names that at
the admission, where the extent is real (after the prefix match and the
store read), and the caller refuses the request BY NAME (W123) instead.

The covered prefix is priced with the GROUP's match, exactly like W31/W50's
extent (``Scheduler.weg2_uncached_extent`` over ``tp_head_congruence``'s
MIN-reduced head): no new collective, and a rank-uniform verdict. Where the
group has no opinion the request is DEFERRED -- left in the queue for the
next pass, never admitted on a rank-local number and never refused on one.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

W_NOT_IN_PREFIX = "W123 Weg2VisionImageNotInPrefixOnD"

ADMIT = "admit"
DEFER = "defer"
REFUSE = "refuse"

#: Passes a rid may be deferred for want of a group match before it is
#: refused by name. The deferral is a group property (the head verdict is
#: MIN-reduced), so every rank counts the same passes and refuses in the same
#: one; without a bound a verdict that never comes would be a silent hang.
DEFER_BOUND_PASSES = 256


def d_guard_armed(model_config: Any, env: Optional[Dict[str, str]] = None) -> bool:
    """Group D, multimodal tokenization on, no tower (language_model_only)."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return False
    if not bool(getattr(model_config, "is_multimodal", False)):
        return False
    hf = getattr(model_config, "hf_config", None)
    return bool(getattr(hf, "language_model_only", False))


def image_spans(req: Any) -> List[Tuple[int, int]]:
    """Every multimodal placeholder span of the request, (start, end) inclusive."""
    mm = getattr(req, "multimodal_inputs", None)
    spans: List[Tuple[int, int]] = []
    for it in getattr(mm, "mm_items", None) or []:
        for off in getattr(it, "offsets", None) or []:
            spans.append((int(off[0]), int(off[1])))
    return spans


def image_end(req: Any) -> int:
    """One past the last placeholder position, 0 without images."""
    spans = image_spans(req)
    return (max(end for _, end in spans) + 1) if spans else 0


def verdict(req: Any, covered: Optional[int], tail_start: Optional[int] = None,
            tail_wait: bool = False) -> str:
    """``admit`` when every placeholder lies before the first position D's
    target computes -- the covered prefix, or the adopted tail's start when
    the group took P's tail (``tail_adopt.peek_target_start``) -- ``defer``
    when the group has no covered length for this rid yet or the tail would
    cover the image in an empty batch, ``refuse`` when D would have to encode
    a placeholder."""
    end = image_end(req)
    if not end:
        return ADMIT
    if covered is None:
        return DEFER
    if end <= int(covered):
        return ADMIT
    if tail_start is not None and end <= int(tail_start):
        return ADMIT
    return DEFER if tail_wait else REFUSE


def bounded(verdict_now: str, rid: str, defers: Dict[str, int]) -> str:
    """Apply :data:`DEFER_BOUND_PASSES` to one verdict; ``defers`` is the
    caller's per-rid count, cleared on every verdict that is not a defer."""
    if verdict_now != DEFER:
        defers.pop(rid, None)
        return verdict_now
    n = defers.get(rid, 0) + 1
    defers[rid] = n
    return REFUSE if n > DEFER_BOUND_PASSES else DEFER


def refusal_message(req: Any, covered: Optional[int]) -> str:
    if covered is None:
        return (
            f"{W_NOT_IN_PREFIX}: group D has no vision tower and never prefills an image "
            f"position, and the group published no covered prefix for this request in "
            f"{DEFER_BOUND_PASSES} passes, so nothing proves its images were read from the "
            f"store. Refused by name instead of admitted on one rank's own number."
        )
    spans = image_spans(req)
    first = min((s for s in spans if s[1] >= covered), default=spans[0] if spans else (0, 0))
    return (
        f"{W_NOT_IN_PREFIX}: group D has no vision tower and never prefills an image "
        f"position, but this request's prefix covers only {covered} tokens and an image "
        f"spans tokens {first[0]}..{first[1]} -- the P leg's pages did not reach D's "
        f"store read. Refused by name instead of reaching _require_visual."
    )


#: (B) safety net switch: an image D cannot cover on a request that has sent no
#: byte yet goes back through P (X-REQUEUE) instead of W123. Default on; "0"
#: = the plain W123 as before.
REROUTE_ENV = "SGLANG_WEG2_VISION_D_REROUTE"


def reroute_enabled(env=None) -> bool:
    import os

    e = os.environ if env is None else env
    return (e.get(REROUTE_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def reroute_eligible(req: Any, env=None) -> bool:
    """(B) NF rc12z10 08:40:07Z weg2-2-12 (image 6290..9528 of 9537, P's END
    anchor failed, D covered 0 -> W123 to the client): a request that has
    generated NOTHING yet can go back through P with its ORIGINAL body --
    image included, P has the tower -- which is the front's X-REQUEUE, keyed
    on D's W50 refusal. An already streamed one cannot: RESUME-VIA-P's P leg
    carries input_ids alone (``resume_via_p.eligible`` excludes multimodal),
    so it keeps the named W123. Replicated: output_ids is the group's."""
    if not reroute_enabled(env):
        return False
    out = getattr(req, "output_ids", None)
    return (0 if out is None else len(out)) == 0


def reroute_message(req: Any, covered: Optional[int]) -> str:
    """The W50 refusal the front re-routes through P (``x_refusal_marker_in``
    and the extent sentence ``_d_refusal_extent`` parses), W123 named inside.
    The front's bound ends it: a second refusal after a P leg is its W35/W53."""
    fill = getattr(req, "full_untruncated_fill_ids", None)
    total = 0 if fill is None else len(fill)
    extent = max(0, total - int(covered or 0))
    return (
        f"W50 Weg2TpPrefillExceeded (vision reroute): group D cannot serve this request "
        f"itself; this request's extent after prefix matching is {extent}. "
        f"Re-routed through the prefill group, which has the vision tower -- "
        f"{refusal_message(req, covered)}"
    )


# -- the draft side (V2 death 11:20:26, weg2-14-72) ---------------------------------
_DRAFT_NO_EMBEDS_N = [0]


def note_draft_mm_without_embeds(input_embeds: Any, forward_batch: Any) -> bool:
    """True when an MTP draft extend carries multimodal inputs but no
    ``mm_input_embeds`` -- the caller then embeds its ids like any text
    extend instead of asserting.

    Upstream's MTP heads reuse the TARGET's input embeddings of an mm extend
    (``general_mm_embed_routine`` leaves them in ``forward_batch.mm_input_embeds``)
    and assert they exist. On a group without a tower that assumption breaks
    where no target forward ran: V2 (dfce479f08) died at 11:20:26 in
    ``_forward_skip_extend`` (E2, P's END state adopted, no target forward)
    -> ``_draft_extend_for_prefill`` -> ``qwen4_exp_mtp._prepare_input_embeds``
    ``assert input_embeds is not None`` on weg2-14-72 (image at the front,
    5837 tokens). The draft only proposes -- the verifier decides -- so the
    placeholder ids' own embeddings are a sound input, the form DFlash always
    uses (models/dflash.py). Never an assert death; counted and named."""
    if input_embeds is not None:
        return False
    _DRAFT_NO_EMBEDS_N[0] += 1
    n = _DRAFT_NO_EMBEDS_N[0]
    if n <= 8 or n % 256 == 0:
        import logging

        logging.getLogger(__name__).warning(
            "W102 VISION-DRAFT NO-MM-EMBEDS n=%d mode=%s bs=%s (an mm extend reached "
            "the MTP draft without target embeddings -- no tower on this group, or no "
            "target forward (E2 skip): the draft embeds the ids, the verifier decides)",
            n,
            getattr(getattr(forward_batch, "forward_mode", None), "name", "?"),
            getattr(forward_batch, "batch_size", "?"),
        )
    return True
