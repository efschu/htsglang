# SPDX-License-Identifier: Apache-2.0
"""S56 D-MATCH-PROBE (NF rc15 19.10.): the front's arrival class from D's EXACT numbers, never from an estimate.

USER 19.10.: "warum wird irgendwas geschätzt? alle zahlen liegen doch schwarz auf weiß vor zu jedem zeitpunkt?" --
binding: the classification (classify_arrival, the set's S, R6's rest, ARRIVAL-SEAT, PARK_NO_SEAT) uses
``uncached = prompt - the exact hit at the source, at the moment of the decision``. SOLL R11 stays: the front decides
alone, on the sources' numbers.

THE DEFECT (rc15 boot ..._489909d138_1019_071429, front + D + P log). The front priced every arrival with its own
tokenizer and its own presence ledger (TokenSpans: D readings, seq marks, END-ANCHORs, the L3 index probe). Of 27
requests with a realised count, 10 were off by more than one page; the worst:
  pdflip-6-11   front 3513   D computed 977   (8 images in Anthropic tool_result blocks: the front counted a TEXT
                                                prompt of 97465 tokens, D's prompt has 99217 -- the clamp of D's
                                                reading to the front's own text, and the seq mark 98240 > 97465
                                                could never credit)
  pdflip-10-17  front 23549  P computed 543    (the L3 probe stops at the first image: image pages are keyed by
                                                D's pad values, which only D's processor forms)
  pdflip-14-74  front 338    P computed 5718   (4 NEW images in the uncached tail: D refused W123 -- a P-only
                                                request the front could not see)
Every one of them is a token-space problem of a SECOND bookkeeping. Group D holds the answer: its tokenizer forms the
exact prompt (images expanded, pad values = the radix key) and its tree knows what it holds right now.

THE PROBE. ``POST /pdflip/match_probe {"path", "payload"}`` on group D:
  1. D's tokenizer manager tokenizes the payload exactly as the POST of the leg will be (``_tokenize_one_request``:
     chat template, Anthropic conversion with tool_result images, the image processor through D-MM-ITEM-CACHE -- a
     later real leg of the same images hits that cache);
  2. ``PdFlipMatchProbeReqInput`` goes to the scheduler like every control request: TP0 receives, the group
     broadcasts, EVERY rank pads the image items as its intake does (a broadcast mm process needs every rank) and
     reads the admission's usable prefix with ``peek_admission_match`` -- READ-ONLY (S56r review A2: no split, no
     LRU, no stamp, no counter, no slot; ``match_prefix`` itself is none of that). TP0 answers. Its number is the
     GROUP's on a Form A group (H98: the workers adopt the host's depth) or on a symmetric host tier (replica
     trees); on an asymmetric tier without Form A the admission takes the RU MIN over the ranks, which one rank
     cannot read -- the probe then answers success=False by name (``group_min_needed``), never TP0's guess;
  3. the answer: D's ids (scheduler token space), the device tokens and the host-tier tokens of the match (the
     admission's usable prefix: KV + the Mamba anchor), the image pad spans.
The front then asks the SHARED store (L2 arenas + L3 stem index, ``front_store``) with D's ids -- image pages
included, since the ids carry the pad values -- and prices
  ``covered = max(device + host, store)``, ``uncached = n - covered``,
  ``p_only`` when an image span ends past ``covered`` (D has no tower: W123), and then ``uncached_p = n - store``.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROBE_PATH = "/pdflip/match_probe"
#: the probe's rid: never the leg's own rid (no state of the real request is touched)
PROBE_RID_SUFFIX = "~probe"
_PATHS = ("/v1/messages", "/v1/chat/completions", "/generate")


# --------------------------------------------------------------------------------------------- D side (tokenizer)
async def build_probe_obj(body: Dict[str, Any], *, serving_chat: Any, serving_anthropic: Any = None,
                          raw_request: Any = None):
    """The ``GenerateReqInput`` the leg's POST would build from ``body`` (``{"path", "payload"}``), under the probe's
    rid; None for a shape the front does not price."""
    path = str(body.get("path") or "")
    payload = dict(body.get("payload") or {})
    if path not in _PATHS:
        return None
    rid = str(payload.get("rid") or "front") + PROBE_RID_SUFFIX
    payload["rid"] = rid
    payload["stream"] = False
    if path == "/generate":
        from flliper.srt.managers.io_struct import GenerateReqInput

        keep = {k: payload[k] for k in ("text", "input_ids", "image_data", "extra_key", "lora_path")
                if k in payload}
        if "input_ids_b64" in payload:
            # the TEXT-ONLY fast path: the front's exact ids (137/137 text prompts of rc14+rc15 matched D's count,
            # every mismatch was an image request) -- D skips template and tokenizer, only its tree is asked
            keep["input_ids"] = ids_from_b64(str(payload["input_ids_b64"])).tolist()
        obj = GenerateReqInput(rid=rid, sampling_params={"max_new_tokens": 1}, **keep)
    else:
        if path == "/v1/messages":
            if serving_anthropic is None:
                return None
            from flliper.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

            req = serving_anthropic._convert_to_chat_completion_request(AnthropicMessagesRequest(**payload))
        else:
            from flliper.srt.entrypoints.openai.protocol import ChatCompletionRequest

            req = ChatCompletionRequest(**payload)
        obj, _ = serving_chat._convert_to_internal_request(req, raw_request)
        obj.rid = rid
        obj.stream = False
    obj.normalize_batch_and_arguments()
    return obj


def ids_to_b64(ids: Any) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int64).astype(np.int32).tobytes()).decode("ascii")


def ids_from_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s.encode("ascii")), dtype=np.int32).copy()


# --------------------------------------------------------------------------------------------- D side (scheduler)
def mm_spans(image_inputs: Any) -> List[List[int]]:
    """[start, end) of every multimodal item's pad run (``MultimodalDataItem.offsets``: inclusive (start, end))."""
    out: List[List[int]] = []
    for item in getattr(image_inputs, "mm_items", None) or ():
        for off in getattr(item, "offsets", None) or ():
            try:
                a, b = int(off[0]), int(off[1])
            except (TypeError, ValueError, IndexError):
                continue
            out.append([a, b + 1])
    out.sort()
    return out


class PeekUnsupported(RuntimeError):
    """S56r: the tree carries a component the read-only walk has no exact rule for (SWA, ...) -- the probe answers
    success=False by name; the front then takes the A1 path (no estimate)."""


def peek_admission_match(tree_cache: Any, ids: Any, extra_key: Optional[str], limit: int) -> Tuple[int, int, int]:
    """S56r (review A2, user 19.10. "alle zahlen liegen doch schwarz auf weiß vor"): the ADMISSION's usable prefix of
    ``ids`` -- (device tokens, host-tier tokens, key depth) -- read off the tree WITHOUT TOUCHING IT.

    ``UnifiedRadixCache.match_prefix`` is not read-only, and S56 called it: ``_split_node`` re-shapes the tree at a
    partial match (new node, LRU remove/insert, clones), ``_match_post_processor`` moves the matched chain to MRU
    in every component LRU and re-stamps ``last_access_time`` up to the root, ``MambaComponent.finalize_match_result``
    bumps the #928 refusal counters and prints MATCH-CENSUS-DEEP / #928 lines under the probe's rid, the mamba host
    validator counts MAMBA-HOST-RESUME, and the #904/#1061/#1420 instruments count the walk. A probe of a request
    that then goes to P would have re-ordered D's eviction for nothing.

    This walk reads the same facts with the same rules and writes NOTHING (no node, no LRU, no stamp, no counter, no
    log line, no slot, no lock):
      * key: ``RadixKey(ids, extra_key, limit)`` -> bigram view (EAGLE) -> page aligned, as ``match_prefix``;
      * the walk of ``_match_prefix_helper``: a dead node (evicted, not backuped) stops it; a PARTIAL child is NOT
        split -- its would-be upper half is judged where ``_split_node`` would put it (FULL: the child's slice ->
        the child's validity; MAMBA: ``redistribute_on_node_split`` leaves the upper half without a state -> never an
        anchor);
      * validators: FULL ``value is not None`` (device) / ``or backuped`` (host tier), MAMBA the component's own
        ``explain_match_refusal`` (side-effect free, defined on the same ``resume_refusal_reason`` as the validator:
        state present + checkpoint grid + the per-path cap), with ``match_device_only`` exactly as the walk builds
        them (``cache_controller is None`` -> device only);
      * ``FullComponent.finalize_match_result``: host hit = the host rows from the best node up to the best device
        node; ``MambaComponent.finalize_match_result``: the #747 strict resume (no host tier only) and the #928 (a)/(b)
        zeroing through ``tp_match_floor.anchor_unusable`` (the side-effect-free twin the RU vote already uses);
      * ``FLLIPER_RADIX_FORCE_MISS`` zeroes like ``Req.init_next_round_input``.
    Any other component type raises ``PeekUnsupported`` (named, never guessed)."""
    from flliper.srt.environ import envs
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    tc = tree_cache
    comps = tuple(getattr(tc, "_components_tuple", ()) or ())
    kinds = {c.component_type for c in comps}
    unknown = kinds - {ComponentType.FULL, ComponentType.MAMBA}
    if unknown:
        raise PeekUnsupported("peek: no read-only rule for component(s) %s" % sorted(str(k) for k in unknown))
    if getattr(tc, "disable", False) or envs.FLLIPER_RADIX_FORCE_MISS.get():
        return 0, 0, 0
    key = RadixKey(token_ids=ids, extra_key=extra_key, limit=int(limit))
    key, _ = key.maybe_to_bigram_view(getattr(tc, "is_eagle", False))
    page = int(getattr(tc, "page_size", 1) or 1)
    if len(key) == 0:
        return 0, 0, 0
    key = key.page_aligned(page)
    if len(key) == 0:
        return 0, 0, 0
    device_only = getattr(tc, "cache_controller", None) is None
    mamba = getattr(tc, "components", {}).get(ComponentType.MAMBA) if ComponentType.MAMBA in kinds else None
    F = ComponentType.FULL

    def _full_ok(node: Any, dev_only: bool) -> bool:
        cd = node.component_data[F]
        return cd.value is not None or (not dev_only and cd.host_value is not None)

    def _ok(node: Any, depth: int, dev_only: bool) -> bool:
        if not _full_ok(node, dev_only):
            return False
        if mamba is not None and mamba.explain_match_refusal(node, depth, match_device_only=dev_only) is not None:
            return False
        return True

    root = tc.root_node
    node = root
    child_key = key.child_key(page)
    value_lens: List[int] = []          # the walk's ``value`` list (device chunks of non-evicted nodes), lengths
    best_node, best_dev_node, best_dev_len = root, root, 0
    best_virtual = None                 # (child, prefix_len) when the best node is the unsplit upper half
    best_dev_virtual = False
    cum = 0
    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        ccd = child.component_data[F]
        if ccd.value is None and ccd.host_value is None:  # evicted and not backuped: a dead node stops the walk
            break
        prefix_len = child.key.match(key, page_size=page)
        if prefix_len < len(child.key):
            # the would-be split parent [0, prefix_len) of ``child`` -- judged, not built
            if ccd.value is not None:
                value_lens.append(int(prefix_len))
            cum += prefix_len
            if mamba is None:  # FULL only: the upper half carries the child's slices (redistribute_on_node_split)
                if _full_ok(child, device_only):
                    best_node, best_virtual = node, (child, int(prefix_len))
                if _full_ok(child, True):  # device validity: the slice of a device value is on the device
                    best_dev_node, best_dev_len, best_dev_virtual = node, len(value_lens), True
            # with MAMBA the upper half holds no state (redistribute_on_node_split): never an anchor
            break
        if ccd.value is not None:
            value_lens.append(len(ccd.value))
        node = child
        cum += prefix_len
        if _ok(node, cum, device_only):
            best_node, best_virtual = node, None
        if device_only:
            if best_node is node:
                best_dev_node, best_dev_len, best_dev_virtual = node, len(value_lens), False
        elif _ok(node, cum, True):
            best_dev_node, best_dev_len, best_dev_virtual = node, len(value_lens), False
        key = key[prefix_len:]
        if len(key):
            child_key = key.child_key(page)
    key_depth = int(cum)
    dev = int(sum(value_lens[:best_dev_len]))
    # FullComponent.finalize_match_result: host rows from the best node up to (not incl.) the best device node
    host = 0
    if best_virtual is not None and not (best_dev_virtual and best_dev_node is best_node):
        ch, pl = best_virtual
        hv = ch.component_data[F].host_value
        if hv is not None:
            host += min(int(pl), len(hv))
    n_ = best_node
    while n_ is not best_dev_node and n_ is not root and n_ is not None:
        hv = n_.component_data[F].host_value
        if hv is not None:
            host += len(hv)
        n_ = n_.parent
    if mamba is not None:
        interval = getattr(mamba, "mamba_checkpoint_interval", None)
        if (interval is not None and getattr(mamba, "mamba_ckpt_strict_resume", False)
                and getattr(tc, "cache_controller", None) is None and best_dev_len > 0):
            from flliper.srt.mem_cache.mamba_ckpt_utils import floor_to_interval

            if dev != floor_to_interval(int(sum(value_lens)), interval):
                return 0, 0, key_depth
        if best_node is not root:
            from flliper.srt.managers.tp_match_floor import anchor_unusable

            if anchor_unusable(tc, best_node):  # #928 (a) no state / (b) foreign pool: the admission zeroes
                return 0, 0, key_depth
    return dev, host, key_depth


def match_numbers(match_result: Any) -> Tuple[int, int, int]:
    """(device tokens, host-tier tokens, key depth) of a ``MatchResult`` -- the admission's usable prefix is
    ``device + host`` (``Req.init_next_round_input``: prefix_indices + host_hit_length)."""
    dev = len(getattr(match_result, "device_indices", ()) or ())
    host = int(getattr(match_result, "host_hit_length", 0) or 0)
    key = int(getattr(match_result, "key_match_depth", 0) or 0)
    return dev, host, key


# ----------------------------------------------------------------------------------------------------- front side
@dataclasses.dataclass(frozen=True)
class ProbeAnswer:
    ids: np.ndarray
    device_tokens: int
    host_tokens: int
    key_match_depth: int
    mm_spans: Tuple[Tuple[int, int], ...]
    tokenize_ms: float = 0.0
    handle_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def n(self) -> int:
        return int(self.ids.size)

    @property
    def tree_tokens(self) -> int:
        return min(self.n, int(self.device_tokens) + int(self.host_tokens))


def parse_answer(body: Any) -> Optional[ProbeAnswer]:
    """The D answer, or None when it is not a complete one (the caller names why)."""
    if not isinstance(body, dict) or not body.get("success") or not body.get("ids"):
        return None
    try:
        ids = ids_from_b64(str(body["ids"]))
        spans = tuple((int(a), int(b)) for a, b in (body.get("mm_spans") or ()))
        return ProbeAnswer(ids=ids, device_tokens=int(body.get("device_tokens", 0) or 0),
                           host_tokens=int(body.get("host_tokens", 0) or 0),
                           key_match_depth=int(body.get("key_match_depth", 0) or 0), mm_spans=spans,
                           tokenize_ms=float(body.get("tokenize_ms", 0.0) or 0.0),
                           handle_ms=float(body.get("handle_ms", 0.0) or 0.0),
                           total_ms=float(body.get("total_ms", 0.0) or 0.0))
    except (TypeError, ValueError, KeyError):
        return None


@dataclasses.dataclass(frozen=True)
class ExactPrice:
    #: what D must prefill: n - max(tree, store)
    uncached: int
    #: what P must prefill: n - store (P reads only the shared store)
    uncached_p: int
    covered: int
    tree_tokens: int
    store_tokens: int
    #: an image item ends past ``covered``: D cannot take it (no tower, W123) -- P-only (SOLL R9)
    image_in_tail: bool
    src: str


def exact_price(ans: ProbeAnswer, store_tokens: int) -> ExactPrice:
    """THE price of the arrival from the sources' numbers (no estimate): D's tree (device + host tier) and the
    shared store, both asked with D's own ids. ``store_tokens`` is the store's page depth of ``ans.ids``."""
    n = ans.n
    tree = ans.tree_tokens
    store = max(0, min(int(store_tokens), n))
    covered = max(tree, store)
    tail = any(int(b) > covered for _a, b in ans.mm_spans)
    src = "d_tree" if tree >= store and tree > 0 else ("store" if store > 0 else "none")
    return ExactPrice(uncached=max(0, n - covered), uncached_p=max(0, n - store), covered=covered,
                      tree_tokens=tree, store_tokens=store, image_in_tail=bool(tail), src=src)


def class_uncached(price: ExactPrice) -> int:
    """The u the class and the set's S use (SOLL section 2): D's figure, unless the request is P-only (an image in
    the uncached tail, R9) -- then what P computes."""
    return int(price.uncached_p if price.image_in_tail else price.uncached)


def nested_image_parts(payload: Any) -> int:
    """Images inside Anthropic ``tool_result`` blocks (``messages[].content[].content[]``), which the top-level
    ``_image_parts`` does not see but D's Anthropic adapter turns into images
    (``serving._convert_tool_result_content``). rc15: pdflip-6-11 carried 8 of them."""
    n = 0
    try:
        for m in (payload or {}).get("messages") or ():
            c = m.get("content") if isinstance(m, dict) else None
            if not isinstance(c, list):
                continue
            for part in c:
                if not isinstance(part, dict) or part.get("type") != "tool_result":
                    continue
                inner = part.get("content")
                if not isinstance(inner, list):
                    continue
                n += sum(1 for it in inner if isinstance(it, dict)
                         and str(it.get("type") or "") in ("image", "image_url", "input_image"))
    except Exception:  # noqa: BLE001 -- a malformed body carries no image
        return 0
    return n


def payload_has_any_media(payload: Any) -> bool:
    """Any image/video/audio anywhere D would expand: a non-text content part, an image inside a ``tool_result``
    block, or the native ``image_data`` / ``video_data`` / ``audio_data``. False = a text prompt, whose front count
    equals D's (X-EXACT-TOKENS rc14+rc15: 137/137)."""
    if not isinstance(payload, dict):
        return False
    for k in ("image_data", "video_data", "audio_data", "input_embeds"):
        if payload.get(k):
            return True
    if nested_image_parts(payload):
        return True
    for m in payload.get("messages") or ():
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                return True
            t = str(part.get("type") or "")
            if t in ("text", "tool_use", "tool_result", "thinking", "redacted_thinking", "tool_reference"):
                continue
            return True
    return False


def seq_spans_ok(spans: Sequence[Sequence[int]], n: int) -> bool:
    return all(0 <= int(a) < int(b) <= n for a, b in spans)
