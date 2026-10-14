# SPDX-License-Identifier: Apache-2.0
"""G4 (NF-GGUF, PLAN-GGUF-NF-1009 section 0 item 6): the weight LAYOUT of a GGUF expert store, as a boot-wide fact.

A GGUF checkpoint stores its experts in ggml blocks whose type is chosen PER LAYER and PER PROJECTION (the
unsloth Qwen3.8-Flash-Next UD-IQ4_XS export: gate/up IQ3_S, down IQ4_NL or Q8_0, layer 2 gate/up IQ4_XS), so the
byte size of one expert ROW is a property of the layer, not of the model: three row classes (2.2217 / 3.0029 /
3.3203 MiB) over 48 layers.  The shared expert store (``expert_store``) names its files ``L<n>-<attr>.bin`` and
sizes them from the row shape; a sentinel (``*.written.json``) only says "these slots are written".  What keeps a
store written for one set of types from vouching for another is the store IDENTITY
(``expert_store.compute_identity``), and for a GGUF source that identity used to hash NOTHING of the file: a
directory contributed only ``config.json`` / ``*.safetensors*`` (a GGUF directory has none of the last two), a
``.gguf`` path only its name.

This module is the one place that says

* what the per-layer expert types of a GGUF set are (:func:`expert_types_per_layer`),
* the layout tag that names them (:func:`gguf_layout_tag`, ``"gguf:<sha of the types per layer>"``) -- the
  same string shape H88-C's ``moe_w4a8_layout.LAYOUTS`` carries (``marlin_w4a16`` / ``marlin_w4a8``), so a merge
  only has to accept the ``gguf:`` prefix there (:func:`is_gguf_layout`),
* a digest of the HEADER of every shard (:func:`header_digest`): the key/value block and the tensor directory --
  names, shapes, ggml types, data offsets.  Not the weights (that was 50 GB of reading per boot, the thing the
  store exists to avoid), but everything that decides what a row IS: re-quantizing a tensor changes its type or
  its offset, and the digest moves.

Pure python, stdlib only (the launcher imports ``expert_store``, and ``expert_store`` imports this lazily).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from typing import Dict, List, Optional, Sequence, Tuple

MARKER = "G4 GGUF-LAYOUT"

#: the prefix of a GGUF layout tag in ``compute_identity``'s ``layout`` argument
LAYOUT_PREFIX = "gguf:"

#: ggml type id -> name, only the ids this tree names elsewhere (``weg2/model_profile.GGML_TYPES``); an id not
#: listed is carried as ``type<id>`` -- the tag must change when a type changes, it need not know its name.
GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}

_SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<no>\d{5})-of-(?P<total>\d{5})\.gguf$")
_EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<proj>gate_up|gate|up|down)_exps\.weight$")

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFLayoutError(RuntimeError):
    """A GGUF header that cannot be read or a set that is incomplete (named; never a guessed layout)."""


class _Reader:
    """Reads the header and feeds every byte it consumes into a digest (header bytes only)."""

    def __init__(self, fh, digest):
        self.fh = fh
        self.digest = digest

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) < n:
            raise GGUFLayoutError("GGUF header ends early")
        self.digest.update(b)
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")

    def skip_value(self, vt: int) -> None:
        if vt in _SCALAR:
            self.read(struct.calcsize(_SCALAR[vt]))
        elif vt == 8:
            self.string()
        elif vt == 9:
            et = self.u32()
            n = self.u64()
            if et in _SCALAR:
                self.read(struct.calcsize(_SCALAR[et]) * n)
            else:
                for _ in range(n):
                    self.skip_value(et)
        else:
            raise GGUFLayoutError("GGUF value type %d unknown" % vt)


def read_header(path: str) -> Tuple[str, List[Tuple[str, Tuple[int, ...], int, int]]]:
    """``(sha256 of the header bytes, [(name, dims, ggml type id, data offset)])`` of ONE ``.gguf`` file.

    The digested bytes are the magic, version, counts, the key/value block and the tensor directory -- up to the
    first byte of the alignment padding in front of the data.  Never a tensor byte."""
    digest = hashlib.sha256()
    tensors: List[Tuple[str, Tuple[int, ...], int, int]] = []
    try:
        with open(path, "rb") as fh:
            r = _Reader(fh, digest)
            if r.read(4) != b"GGUF":
                raise GGUFLayoutError("%s: not a GGUF (magic missing)" % path)
            version = r.u32()
            if version < 2:
                raise GGUFLayoutError("%s: GGUF version %d is not read" % (path, version))
            n_tensors = r.u64()
            n_kv = r.u64()
            for _ in range(n_kv):
                r.string()
                r.skip_value(r.u32())
            for _ in range(n_tensors):
                name = r.string()
                nd = r.u32()
                dims = tuple(r.u64() for _ in range(nd))
                gt = r.u32()
                off = r.u64()
                tensors.append((name, dims, gt, off))
    except (OSError, struct.error) as exc:
        raise GGUFLayoutError("GGUF header of %s not readable: %s" % (path, exc)) from exc
    return digest.hexdigest(), tensors


def source_files(model: str) -> List[str]:
    """The ``.gguf`` files of a model source, sorted: a directory's ``*.gguf``, or a ``.gguf`` path's whole split
    set (``name-00002-of-00003.gguf`` belongs to ``-00001-of-``..); ``[]`` when it is not a GGUF source.  A part
    missing from a split set is a named error, because a digest over a partial set would pass for the whole."""
    model = str(model or "")
    if not model:
        return []
    if os.path.isdir(model):
        return sorted(
            os.path.join(model, n) for n in os.listdir(model) if n.endswith(".gguf")
        )
    if not model.endswith(".gguf"):
        return []
    m = _SPLIT_RE.match(os.path.basename(model))
    if m is None:
        return [model]
    base = os.path.dirname(os.path.abspath(model))
    total = int(m.group("total"))
    out, missing = [], []
    for i in range(1, total + 1):
        p = os.path.join(base, "%s-%05d-of-%05d.gguf" % (m.group("prefix"), i, total))
        (out if os.path.isfile(p) else missing).append(p)
    if missing:
        raise GGUFLayoutError(
            "%s: GGUF split set incomplete, missing %s" % (model, ", ".join(os.path.basename(p) for p in missing))
        )
    return out


def is_gguf_source(model: str) -> bool:
    try:
        return bool(source_files(model))
    except GGUFLayoutError:
        return True  # an incomplete GGUF set is still a GGUF source: the digest must refuse, not skip


def header_digest(files: Sequence[str]) -> str:
    """One hex digest over the headers of ``files``: the header digest of each, in sorted FILE-NAME order, keyed by
    its position in that order -- never by the name or the path (a renamed or moved copy of the same bytes is the
    same checkpoint; the part number of a split set is its position)."""
    h = hashlib.sha256()
    h.update(b"gguf-header-v2\0")
    for i, path in enumerate(sorted(files, key=os.path.basename)):
        d, _t = read_header(path)
        h.update(str(i).encode() + b"\0" + d.encode() + b"\0")
    return h.hexdigest()


def _name_of(gt: int) -> str:
    return GGML_TYPE_NAMES.get(int(gt), "type%d" % int(gt))


def expert_types_per_layer(files: Sequence[str]) -> Dict[int, Dict[str, str]]:
    """``{layer: {projection: ggml type name}}`` over the expert tensors of ``files`` (``gate``/``up``/``down``,
    or the fused ``gate_up``).  An empty dict means the set carries no routed experts."""
    out: Dict[int, Dict[str, str]] = {}
    for path in files:
        _d, tensors = read_header(path)
        for name, _dims, gt, _off in tensors:
            m = _EXPERT_RE.match(name)
            if m is not None:
                out.setdefault(int(m.group("layer")), {})[m.group("proj")] = _name_of(gt)
    return out


def gguf_layout_tag(types: Dict[int, Dict[str, str]]) -> str:
    """``"gguf:<16 hex>"`` -- the sha of the type set, layer by layer.

    Canonical JSON over ``[[layer, [[proj, type], ...]], ...]`` sorted, so the tag is a function of the types only
    (not of the file names, the order of the shards or the offsets: those are the header digest's business)."""
    canon = [[int(layer), sorted(projs.items())] for layer, projs in sorted(types.items())]
    sha = hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()
    return LAYOUT_PREFIX + sha[:16]


def is_gguf_layout(tag: str) -> bool:
    """Is ``tag`` a GGUF layout tag?  What H88-C's ``moe_w4a8_layout.LAYOUTS`` check must accept besides its two
    Marlin tags (it refuses unknown tags by name)."""
    return str(tag or "").startswith(LAYOUT_PREFIX) and len(str(tag)) > len(LAYOUT_PREFIX)


def layout_for_source(model: str) -> Tuple[str, str]:
    """``(layout tag, header digest)`` of a GGUF source, ``("", "")`` when ``model`` is not one.  Raises
    :class:`GGUFLayoutError` for a GGUF set that cannot be read: an identity computed from a guessed layout would
    let a sentinel of another quantization vouch for these rows."""
    files = source_files(model)
    if not files:
        return "", ""
    return gguf_layout_tag(expert_types_per_layer(files)), header_digest(files)


def row_class_bytes(files: Sequence[str]) -> Dict[int, Dict[str, int]]:
    """``{layer: {projection: bytes of ONE expert}}`` from the tensor directory (``ggml block size`` is implicit in
    the next tensor's offset -- not used here: the size comes from dims and type), for the census.  Imports the
    type table lazily from ``weg2/model_profile`` (stdlib-only, already the tree's ggml size table)."""
    from sglang.srt.weg2.model_profile import GGML_TYPES

    out: Dict[int, Dict[str, int]] = {}
    for path in files:
        _d, tensors = read_header(path)
        for name, dims, gt, _off in tensors:
            m = _EXPERT_RE.match(name)
            if m is None:
                continue
            tname, blck, tsize = GGML_TYPES[int(gt)]
            elems = 1
            for d in dims:
                elems *= int(d)
            n_experts = int(dims[-1])
            out.setdefault(int(m.group("layer")), {})[m.group("proj")] = elems // blck * tsize // n_experts
    return out
