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


def verdict(req: Any, covered: Optional[int]) -> str:
    """``admit`` when every placeholder lies inside the covered prefix (or
    there is none), ``defer`` when the group has no covered length for this
    rid yet, ``refuse`` when D would have to encode a placeholder."""
    spans = image_spans(req)
    if not spans:
        return ADMIT
    if covered is None:
        return DEFER
    return ADMIT if max(end for _, end in spans) < int(covered) else REFUSE


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
