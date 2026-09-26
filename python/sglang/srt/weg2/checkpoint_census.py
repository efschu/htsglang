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
from typing import Dict, List, Optional, Sequence, Tuple

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


def _gguf_named_bytes(model_path: str) -> Optional[Tuple[Tuple[Tuple[str, int], ...], int]]:
    """27B line G6: ``(((hf_name, bytes), ...), n_parts)`` of a GGUF FILE's
    tensor directory in file order, or None for anything else (a directory is read
    from its safetensors shards exactly as before).

    The names are the loader's (``weg2.gguf_census``): the NEXTN block takes
    the ``mtp.*`` names the safetensors checkpoint carries, so ``LAYER_RE``,
    ``exclude_prefixes=MTP_TREE_PREFIXES`` and ``tensor_class`` see the same
    trees on both formats; the bytes are the header's own per-tensor counts.
    """
    if not os.path.isfile(str(model_path)):
        return None
    from sglang.srt.weg2 import gguf_census as _gguf_census

    if not _gguf_census.is_gguf_checkpoint(str(model_path)):
        return None
    try:
        census = _gguf_census.gguf_tensor_census(str(model_path))
    except BaseException as exc:  # noqa: BLE001 -- every shape is named
        raise Weg2XchgWidestLayerUnreadable(
            f"W14 Weg2XchgWidestLayerUnreadable: the GGUF tensor directory of "
            f"{model_path} could not be read into the loader's names "
            f"({type(exc).__name__}: {exc}); the widest layer is measured from "
            f"the checkpoint's own header or the boot is refused"
        ) from exc
    return tuple((hf, int(nbytes)) for hf, _g, _t, _s, nbytes in census.rows), census.n_parts


def layer_census_from_headers(model_dir: str, *,
                              exclude_prefixes: Sequence[str] = (),
                              exclude_segments: Sequence[str] = ()) -> LayerCensus:
    """Bytes per layer, group-wide, from every shard's header -- or (27B line
    G6) from a GGUF file's tensor directory. Never a default."""
    root = str(model_dir)
    gguf = _gguf_named_bytes(root)
    if gguf is not None:
        named, n_parts = gguf
        return _census_from_named_bytes(root, named, n_parts,
                                        exclude_prefixes=exclude_prefixes,
                                        exclude_segments=exclude_segments)
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
    named: List[Tuple[str, int]] = []
    for fname in names:
        header = _read_header(os.path.join(root, fname))
        for name, meta in header.items():
            if name == METADATA_KEY:
                continue
            named.append((name, _tensor_bytes(meta)))
    return _census_from_named_bytes(root, named, len(names),
                                    exclude_prefixes=exclude_prefixes,
                                    exclude_segments=exclude_segments)


def _census_from_named_bytes(root: str, named: Sequence[Tuple[str, int]],
                             files: int, *,
                             exclude_prefixes: Sequence[str] = (),
                             exclude_segments: Sequence[str] = ()) -> LayerCensus:
    """The per-layer aggregation, over ``(name, bytes)`` pairs from either
    format's header (safetensors shards or a GGUF tensor directory)."""
    per: Dict[int, int] = {}
    classes: Dict[int, set] = {}
    unlayered = 0
    for name, nbytes in named:
        if nbytes <= 0:
            continue
        # #1374: A MODULE TREE MAY BE EXCLUDED BY NAME. `LAYER_RE` matches
        # the substring `layers.<k>.`, and this checkpoint has TWO trees
        # carrying that: `model.language_model.layers.<k>` and the MTP
        # draft head's own `mtp.layers.<k>`. Measured on the shipped
        # checkpoint: layer 0 is 366.2 MiB of model plus 355.1 MiB of mtp,
        # and the 721.3 MiB sum is what made the tag bound overshoot boot
        # weg2xsn30's own manifest by exactly that 355 MiB. Default ()
        # keeps every existing caller -- including the widest-layer claim
        # the #1332 guard grades -- byte-identical.
        if exclude_prefixes and str(name).startswith(tuple(exclude_prefixes)):
            continue
        # #78 (21.09., Nutzer): EIN TEIL DES CHECKPOINTS WIRD NIE GELADEN.
        # Die Per-Layer-Embeddings liegen unter `...layers.<k>.ple.<...>`
        # und werden zur LAUFZEIT per mmap gelesen (#54: der PLE-Gather
        # kostet 8,6 ms je Runde aus mmap-Shards) -- sie erreichen weder
        # VRAM noch den Host-Store, also traegt sie auch kein Flip.
        # GEMESSEN am Shipped-Checkpoint Qwen3.8-Flash-Next-INT4-Mixed:
        #   gesamt                    163,20 GiB
        #   davon PLE                  95,40 GiB  (58 %)
        #   groesstes 3-Layer-Band mit PLE  99,21 GiB
        #   groesstes 3-Layer-Band ohne PLE  3,81 GiB
        # Der Bounce-Term dimensionierte also um FAKTOR 26 zu gross.
        # ANDERS ALS `exclude_prefixes`: `ple` ist kein Baum-Praefix,
        # sondern ein SEGMENT mitten im Namen -- derselbe Layer traegt
        # geladene und nie geladene Tensoren nebeneinander.
        if exclude_segments and any(str(seg) in str(name)
                                    for seg in exclude_segments):
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
            f"any of the {files} shard(s) under {root} (pattern "
            f"{LAYER_RE.pattern!r}). A checkpoint whose layers cannot be "
            f"identified cannot be assembled a layer at a time, and reporting "
            f"n_layers=0 would divide by zero one call later in bounce_terms"
        )
    layer_bytes = tuple(sorted((i, b) for i, b in per.items()))
    layer_classes = tuple((i, tuple(sorted(classes.get(i, ()))))
                          for i, _b in layer_bytes)
    return LayerCensus(layer_bytes=layer_bytes, layer_classes=layer_classes,
                       unlayered_bytes=int(unlayered), files=int(files))


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


#: The module trees whose `layers.<k>.` names are NOT the language model's and
#: therefore not part of a `weights_<k>` tag. One entry today, named rather
#: than pattern-matched, so a new tree has to be added deliberately.
MTP_TREE_PREFIXES = ("mtp.",)

#: #78: die Per-Layer-Embedding-Tensoren, die NUR per mmap gelesen werden.
#: Als SEGMENT geschrieben (mit fuehrendem und folgendem Punkt), damit ein
#: Modul, das zufaellig `ple` im Namen traegt, nicht mitgenommen wird.
PLE_SEGMENTS = (".ple.",)


def max_tag_bytes_from_census(model_dir: str, chunk_layers: int) -> int:
    """#1374 OPTION 1: the LARGEST WEIGHT TAG this boot will pause, exactly.

    The pause granularity is the tag and the weight tags are LAYER CHUNKS
    (`weg2_memory_saver.weights_family_tags`: `weights_<k>` per chunk of
    `chunk_layers` consecutive layers, plus the base that carries the
    unlayered tensors). A tag's deposit has to fit the assemble buffer or the
    deposit needs a collector that cannot run until this rank has paused --
    boot weg2xsn30's deadlock -- so this is the number the buffer is sized
    from.

    EXACT, not a bound with a margin: the widest WINDOW of `chunk_layers`
    consecutive layers out of the census's own per-layer bytes.

    THE UNLAYERED TENSORS DO NOT SET THIS MAXIMUM, and that is measured rather
    than assumed. `census.unlayered_bytes` is GROUP-WIDE (4516 MiB here:
    embed_tokens plus lm_head), while the embeddings and the head are
    TP-SHARDED -- boot weg2xsn30's own plan lines read
    `model.embed_tokens.weight planned_mib=404.375` and
    `lm_head.weight planned_mib=808.750` PER RANK, i.e. 1213 MiB, which is
    BELOW the 2907 MiB window. Taking `max(window, unlayered_bytes)` therefore
    overshot the boot's own manifest by +51 % (4516 against a measured
    2988 MiB) -- the group-wide-versus-per-rank confusion, one term over. The
    rank-side manifest guard (W68 against each rank's own tag maximum) is what
    catches a checkpoint where that ordering does not hold.

    VALIDATED AGAINST THE RIGHT QUANTITY. The buffer has to hold what the
    deposit MOVES -- the parameters -- not what the saver's arena occupies.
    weg2xsn30's manifest for `weights_0` lists 114 tensors summing to
    2908.1 MiB of `planned_mib`, and this derives 2907 MiB: -0.04 %. The same
    tag's pause line reads `bytes=2988 MiB`, ~80 MiB more, which is the saver
    ARENA footprint (alignment and padding) and not bytes any band carries.
    Grading this number against 2988 would have reported -2.7 % and invited a
    margin; grading it against the param sum shows there is nothing to add.

    `chunk_layers <= 0` means the weights are not chunked: then one tag is the
    whole layer stack and this returns the layered total plus the unlayered --
    a number the ledger will refuse if it cannot be funded, which is the
    correct answer rather than a buffer that deadlocks.
    """
    # THE WEIGHT TAGS ARE THE LANGUAGE MODEL'S LAYER CHUNKS. The MTP head is a
    # module of its own with its own tag, and its `mtp.layers.<k>` names alias
    # the language model's layer indices -- see `exclude_prefixes` above for the
    # measurement. Enumerated and subtracted by NAME, never by a factor.
    census = layer_census_from_headers(model_dir,
                                      exclude_prefixes=MTP_TREE_PREFIXES,
                                      exclude_segments=PLE_SEGMENTS)
    per_layer = [int(b) for _idx, b in census.layer_bytes]
    if not per_layer:
        raise Weg2XchgWidestLayerUnreadable(
            "W14 Weg2XchgWidestLayerUnreadable: the census read no layer at "
            "all, so the largest tag cannot be stated and a buffer sized "
            "against a default is how the xsn30 deadlock returns")
    width = int(chunk_layers)
    if width <= 0 or width >= len(per_layer):
        window = sum(per_layer)
    else:
        window = max(sum(per_layer[i:i + width])
                     for i in range(0, len(per_layer) - width + 1))
    return int(window)


def widest_layer_terms(model_dir: str, *, pairs: int, depth: int,
                       slot_bytes: int = 0, n_lanes: int = 1,
                       max_tag_bytes: int = 0, lanes_concurrent: int = 0,
                       band_credit: bool = False,
                       n_cross_lanes: Optional[int] = None,
                       price_lane_cap: int = 0):
    """``(BounceTerms, widest_line, widest_layer_name)`` for the launcher.

    ONE CALL SITE'S WORTH of glue, kept here so the launcher holds no
    arithmetic: the census is measured, the widest layer is taken, and
    ``xchg_bounce.bounce_terms`` -- the one sizing function -- is fed with
    MEASURED numbers only. ``bytes_per_direction`` is the checkpoint's
    layer-bound total, i.e. what one direction of the exchange has to move,
    and it is named on the line rather than assumed.

    ``n_lanes`` is passed THROUGH to ``bounce_terms`` (one assemble buffer per
    lane) and defaults to 1, the same default that function declares. #1358
    added the parameter there and to this function's caller but never to this
    function, so every armed boot died with ``widest_layer_terms() got an
    unexpected keyword argument 'n_lanes'`` in ``choose_host_ledger``, before
    either group started -- a producer commit calling a signature that never
    existed. Default 1 rather than required, because the two callers differ:
    the ledger passes the MEASURED count and the rank publication passes the
    same one, while every other caller prices a single lane.

    ``lanes_concurrent`` is passed THROUGH the same way, and defaults to 0
    ("not stated") -- #1385's cap on how many of ``n_lanes`` price at once.

    ``band_credit``/``n_cross_lanes`` are passed THROUGH the same way,
    defaulting to ``False``/``None`` (#1397's own byte-identical "never
    armed" state). #1397's own design doc (section 9.1) named this exact
    gap: DESK11 built ``BounceTerms.band_credit``/``n_cross_lanes`` and
    ``xchg_bounce.resolve_cross_lanes`` behind their own file boundary, but
    this function -- the ONE call site between the launcher and
    ``bounce_terms`` for every OTHER field above -- never grew the two new
    parameters, so a caller passing them would have died with the identical
    ``unexpected keyword argument`` TypeError #1358 already paid for once.
    Verified here rather than assumed: before this change, ``inspect.
    signature(widest_layer_terms)`` carried neither name.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    # #78 (fnFL2w7, 21.09.): DERSELBE AUSSCHLUSS WIE NEBENAN, und dass er hier
    # gefehlt hat, stand eine Zeile spaeter im Boot-Log:
    #   max_tag_mib=3900 (gefixt)  neben
    #   widest_layer_bytes=103797370776 (ungefixt)
    # Zwei Leser desselben Checkpoints, einer mit und einer ohne PLE -- also
    # zwei Antworten auf eine Frage. Die PLE werden per mmap gelesen und nie
    # geladen (#54), der MTP-Kopf hat seinen eigenen Tag; beides gehoert aus
    # dem Term, aus dem der Bounce sizet.
    census = layer_census_from_headers(model_dir,
                                       exclude_prefixes=MTP_TREE_PREFIXES,
                                       exclude_segments=PLE_SEGMENTS)
    idx, nbytes, _classes = widest_layer(census)
    kw = {} if not slot_bytes else {"slot_bytes": int(slot_bytes)}
    terms = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers,
        widest_layer_bytes=nbytes,
        pairs=int(pairs),
        depth=int(depth),
        n_lanes=int(n_lanes),
        max_tag_bytes=int(max_tag_bytes),
        lanes_concurrent=int(lanes_concurrent),
        band_credit=bool(band_credit),
        n_cross_lanes=n_cross_lanes,
        price_lane_cap=int(price_lane_cap or 0),
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
