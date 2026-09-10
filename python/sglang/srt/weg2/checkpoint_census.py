# SPDX-License-Identifier: Apache-2.0
"""#1332 B1b -- ``widest_layer_bytes``, MEASURED from the checkpoint's own headers.

THE OPEN DECIDE THIS CLOSES (WEG2_REUSE_SPEC_0908 §10.8, item 1): the bounce
sizing function ``xchg_bounce.bounce_terms`` refuses to be called without
``widest_layer_bytes`` -- *"Absent is NOT 'use the mean': the refusal grades
against this number and a missing one would silently become a mean-sized
bound"* -- and until this module there was no producer for it.

WHAT IT MEASURES, and the three decisions behind it, all from §10.2 and the
user's law of 2026-09-11 (*"notfalls wird das Layer VOLLSTAENDIG in einem
kleinen Host-Puffer zusammengesetzt, jede Karte nimmt ihren Teil"*):

* **GROUP-WIDE**, not per card: the assemble buffer holds ONE COMPLETE layer
  which all three cards then slice, so its size is that layer's WHOLE byte
  count and not one card's share of it.
* **PER LAYER**, because the layer is the assembly unit.
* **MAX, never MEAN.** A buffer sized on the mean dies on the first layer above
  it, and the mean is carried on the line only so a reader can see the spread.

WHY THE CHECKPOINT AND NOT A LIVE CENSUS.  The number is needed at ARM time --
before either group starts, before any weight is on a card, in the launcher's
own process. Nothing in that moment knows a parameter's true extent except the
checkpoint. The safetensors header of every shard carries ``dtype``, ``shape``
and ``data_offsets`` per tensor, so the exact on-disk byte count is
``data_offsets[1] - data_offsets[0]`` -- read without touching a byte of tensor
data, without torch, without a GPU and without a model load. Measured on this
rig's own 27.52 GiB checkpoint: 33 classes over 64 layers in well under a
second.

The read side existed first as a desk tool (``weg2/tools/class_byte_census.py``,
written for boot weg2xsn8's sizing addendum, where the remainder had to be
approximated per class); this module is that reading turned into a product
producer WITH A REFUSAL.

**IT NEVER RETURNS A DEFAULT.**  An unreadable checkpoint, a truncated header
and a checkpoint carrying no layer-indexed tensor are three NAMED refusals
(:class:`Weg2XchgWidestLayerUnreadable`), each printing the path it read. A
default here would re-enter exactly the pinned constant the whole slice exists
to remove, and it would do so on the one number the refusal grades against.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: ``model.layers.<i>.`` / ``layers.<i>.`` -- the layer index, wherever the
#: checkpoint's naming puts it. Deliberately NOT anchored at the string start:
#: this rig's Qwen checkpoints prefix with ``model.`` and the draft/MTP head
#: does not, and a producer that saw only one of the two would under-count the
#: widest layer by whatever the other holds.
LAYER_RE = re.compile(r"(?:^|\.)layers?\.(\d+)\.")

#: The safetensors metadata key, which is not a tensor and carries no bytes.
METADATA_KEY = "__metadata__"

#: Fallback element sizes, used ONLY when a header omits ``data_offsets``.
#: Every real safetensors shard carries them; this table exists so that a
#: shard which does not is still measured rather than silently skipped.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

WIDEST_LINE_PREFIX = "WEG2-XCHG WIDEST"


class Weg2XchgWidestLayerUnreadable(RuntimeError):
    """W14. The checkpoint could not yield a MEASURED widest layer.

    THE CODE TOOK THREE TRIES, and both wrong ones were the SAME MISTAKE at
    different depths: a census whose PATTERN was narrower than the ways a
    W-code is actually written.

    * Draft 1 wrote **W74**, which is `Weg2XchgSourceMissing`'s.
      `test_weg2_wcode_uniqueness_1263` caught it in the gate:
      `{'W74': {'Weg2XchgWidestLayerUnreadable', 'Weg2XchgSourceMissing'}}`.
    * Draft 2 wrote **W88** after enumerating with `W(\d+)\s+Weg2\w+`, i.e.
      "a number followed by an exception NAME". That pattern cannot see
      `scheduler.py:6314`'s `code: str = "W88",` (its holder,
      `name: str = "Weg2StoreLoadNotProgressing"`, is on the NEXT line), so it
      read 88 as free while a word-bounded text grep over `python/sglang/srt`
      found NINETEEN hits for W88. The uniqueness census could not see it
      either -- form 4 was added in the same commit as this renumber, and it is
      red on the previous tip.
    * **W14** is the lowest code with ZERO textual hits anywhere under
      `python/sglang/srt` (the full zero set: 14, 15, 23, 39 -- every other
      number below 90 is textually taken in SOME form, including the ones
      draft 2's narrow pattern reported free: W5 16 hits, W6 5, W13 13, W19 11,
      W24 17, W27 23, W73 8 and earmarked for `Weg2XchgRollForward`).

    THE RULE THIS LEAVES BEHIND: a free W-code is one with no TEXTUAL hit at
    all, not one the current census pattern happens not to parse.

    A refusal and never a fallback. ``bounce_terms`` grades the assemble
    buffer's coverage against this number, so a guessed one would produce a
    green ``covers_widest=yes`` over a buffer that cannot hold the layer --
    the instrument-that-cannot-go-red shape this campaign has paid for
    repeatedly. The message names the PATH it read and WHAT it could not do
    with it.
    """


def tensor_class(name: str) -> str:
    """The class of a parameter name -- the same last-meaningful-segment rule
    ``weight_exchange_shadow.tensor_class`` uses, applied to checkpoint names.

    Imported rather than re-implemented would be better; it is re-derived here
    on purpose, because that module pulls the whole exchange (region,
    transport, ctypes) into a process that only wants to read file headers, and
    the launcher calls this at ARM time. The two are pinned equal by a test
    over this rig's real class list instead of by an import.
    """
    parts = [p for p in str(name).split(".") if p]
    if not parts:
        return "unknown"
    tail = parts[-1]
    if tail in ("weight", "bias", "weight_scale", "scale", "weight_scale_inv",
                "input_scale", "g_idx", "qweight", "qzeros", "scales"):
        return parts[-2] if len(parts) >= 2 else tail
    return tail


@dataclass(frozen=True)
class LayerCensus:
    """Bytes per layer of one checkpoint, group-wide, measured from headers."""

    #: ``((layer_index, bytes), ...)`` ascending by index.
    layer_bytes: Tuple[Tuple[int, int], ...]
    #: ``((layer_index, (class, ...)), ...)`` -- what that layer is made of.
    layer_classes: Tuple[Tuple[int, Tuple[str, ...]], ...]
    #: Bytes of every tensor that carries NO layer index (embeddings, the head,
    #: the final norm). Counted and reported, never folded into a layer: a
    #: 2.37 GiB ``lm_head`` added to some layer's total would size the assemble
    #: buffer for a layer that does not exist.
    unlayered_bytes: int
    #: How many shards were read, so an absent shard is visible as a number.
    files: int

    @property
    def n_layers(self) -> int:
        return len(self.layer_bytes)

    @property
    def layer_total_bytes(self) -> int:
        return sum(b for _i, b in self.layer_bytes)

    @property
    def mean_layer_bytes(self) -> int:
        return (self.layer_total_bytes // self.n_layers) if self.n_layers else 0

    @property
    def total_bytes(self) -> int:
        return self.layer_total_bytes + int(self.unlayered_bytes)


def _read_header(path: str) -> Dict[str, object]:
    """One shard's header, or a NAMED refusal. Reads the header only."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError("file shorter than the 8-byte header length")
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > (1 << 30):
                raise ValueError(f"implausible header length {n}")
            body = fh.read(n)
            if len(body) != n:
                raise ValueError(
                    f"header truncated: {len(body)} of {n} bytes")
            return json.loads(body)
    except Weg2XchgWidestLayerUnreadable:
        raise
    except BaseException as exc:  # noqa: BLE001 -- every shape is named
        raise Weg2XchgWidestLayerUnreadable(
            f"W14 Weg2XchgWidestLayerUnreadable: the safetensors header of "
            f"{path} could not be read ({type(exc).__name__}: {exc}). The "
            f"assemble buffer is sized on the WIDEST layer of this checkpoint "
            f"and a boot that cannot measure it is refused here, at ARM time, "
            f"rather than sized against a default -- a default is the pinned "
            f"constant this slice exists to remove, on the one number the "
            f"coverage refusal grades against"
        ) from exc


def _tensor_bytes(meta: object) -> int:
    """One tensor's on-disk bytes: the offsets, else dtype x shape."""
    if not isinstance(meta, dict):
        return 0
    off = meta.get("data_offsets")
    if isinstance(off, (list, tuple)) and len(off) == 2:
        span = int(off[1]) - int(off[0])
        if span > 0:
            return span
    el = DTYPE_BYTES.get(str(meta.get("dtype", "")).upper())
    if el is None:
        return 0
    nbytes = el
    for s in (meta.get("shape") or []):
        nbytes *= int(s)
    return int(nbytes)


def layer_census_from_headers(model_dir: str) -> LayerCensus:
    """Bytes per layer, group-wide, from every shard's header. Never a default."""
    root = str(model_dir)
    try:
        names = sorted(f for f in os.listdir(root)
                       if f.endswith(".safetensors"))
    except BaseException as exc:  # noqa: BLE001
        raise Weg2XchgWidestLayerUnreadable(
            f"W14 Weg2XchgWidestLayerUnreadable: the checkpoint directory "
            f"{root} could not be listed ({type(exc).__name__}: {exc})"
        ) from exc
    if not names:
        raise Weg2XchgWidestLayerUnreadable(
            f"W14 Weg2XchgWidestLayerUnreadable: no safetensors shard under "
            f"{root}. The widest layer is measured from the checkpoint's own "
            f"headers; with no shard there is nothing to measure and the boot "
            f"is refused instead of sized against a default"
        )
    per: Dict[int, int] = {}
    classes: Dict[int, set] = {}
    unlayered = 0
    for fname in names:
        header = _read_header(os.path.join(root, fname))
        for name, meta in header.items():
            if name == METADATA_KEY:
                continue
            nbytes = _tensor_bytes(meta)
            if nbytes <= 0:
                continue
            m = LAYER_RE.search(str(name))
            if m is None:
                unlayered += nbytes
                continue
            idx = int(m.group(1))
            per[idx] = per.get(idx, 0) + nbytes
            classes.setdefault(idx, set()).add(tensor_class(name))
    if not per:
        raise Weg2XchgWidestLayerUnreadable(
            f"W14 Weg2XchgWidestLayerUnreadable: no layer-indexed tensor in "
            f"any of the {len(names)} shard(s) under {root} (pattern "
            f"{LAYER_RE.pattern!r}). A checkpoint whose layers cannot be "
            f"identified cannot be assembled a layer at a time, and reporting "
            f"n_layers=0 would divide by zero one call later in bounce_terms"
        )
    layer_bytes = tuple(sorted((i, b) for i, b in per.items()))
    layer_classes = tuple((i, tuple(sorted(classes.get(i, ()))))
                          for i, _b in layer_bytes)
    return LayerCensus(layer_bytes=layer_bytes, layer_classes=layer_classes,
                       unlayered_bytes=int(unlayered), files=len(names))


def widest_layer(census: LayerCensus) -> Tuple[int, int, Tuple[str, ...]]:
    """``(layer_index, bytes, classes)`` of the WIDEST layer. MAX, never mean.

    Ties go to the LOWEST index, so two processes over the same checkpoint name
    the same layer -- the number would be equal either way, but the line a
    reader greps must not depend on dict order.
    """
    if not census.layer_bytes:
        raise Weg2XchgWidestLayerUnreadable(
            "W14 Weg2XchgWidestLayerUnreadable: an empty census has no widest "
            "layer; layer_census_from_headers refuses before this can happen")
    idx, nbytes = max(census.layer_bytes, key=lambda r: (r[1], -r[0]))
    by_index = dict(census.layer_classes)
    return int(idx), int(nbytes), tuple(by_index.get(int(idx), ()))


def widest_line(census: LayerCensus) -> str:
    """The launcher's one line, with its denominators beside the bound.

    ``mean_bytes`` rides along precisely because it is NOT what the buffer is
    sized on: a reader who sees both can tell at a glance how far a mean-sized
    buffer would have been off, which is the mistake this producer exists to
    make impossible.
    """
    idx, nbytes, classes = widest_layer(census)
    return (
        f"{WIDEST_LINE_PREFIX} layer={idx} bytes={nbytes} "
        f"classes={','.join(classes) or 'none'} "
        f"layers={census.n_layers} mean_bytes={census.mean_layer_bytes} "
        f"layer_total_bytes={census.layer_total_bytes} "
        f"unlayered_bytes={census.unlayered_bytes} shards={census.files} "
        f"instrument=safetensors headers (data_offsets), group-wide, MAX over "
        f"layers -- never the mean"
    )


def widest_layer_terms(model_dir: str, *, pairs: int, depth: int,
                       slot_bytes: int = 0):
    """``(BounceTerms, widest_line, widest_layer_name)`` for the launcher.

    ONE CALL SITE'S WORTH of glue, kept here so the launcher holds no
    arithmetic: the census is measured, the widest layer is taken, and
    ``xchg_bounce.bounce_terms`` -- the one sizing function -- is fed with
    MEASURED numbers only. ``bytes_per_direction`` is the checkpoint's
    layer-bound total, i.e. what one direction of the exchange has to move,
    and it is named on the line rather than assumed.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    census = layer_census_from_headers(model_dir)
    idx, nbytes, _classes = widest_layer(census)
    kw = {} if not slot_bytes else {"slot_bytes": int(slot_bytes)}
    terms = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers,
        widest_layer_bytes=nbytes,
        pairs=int(pairs),
        depth=int(depth),
        **kw,
    )
    return terms, widest_line(census), f"layer {idx}"


def class_census(model_dir: str) -> Sequence[Tuple[str, int, int]]:
    """``((class, tensors, bytes), ...)`` bytes-descending -- the sizing table.

    The same reading the boot record's sizing addendum used, as a function, so
    a per-class upper bound never has to be recomputed by hand again.
    """
    root = str(model_dir)
    per: Dict[str, List[int]] = {}
    names = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    for fname in names:
        for name, meta in _read_header(os.path.join(root, fname)).items():
            if name == METADATA_KEY:
                continue
            nbytes = _tensor_bytes(meta)
            if nbytes <= 0:
                continue
            row = per.setdefault(tensor_class(name), [0, 0])
            row[0] += 1
            row[1] += nbytes
    return tuple(sorted(((k, v[0], v[1]) for k, v in per.items()),
                        key=lambda r: (-r[2], r[0])))
