# SPDX-License-Identifier: Apache-2.0
"""27B line G6 (2026-09-25): a GGUF checkpoint's tensor directory, in the names
the launch-side weight readers already understand.

THE GAP. Three launch-side readers price a checkpoint from its safetensors
HEADERS -- ``pp_cut.checkpoint_weight_terms`` (the P cut and, through it,
``ring_table.checkpoint_stage_weights``) and
``checkpoint_census.layer_census_from_headers`` (the widest layer, the largest
weight tag). A GGUF ``--model`` names one FILE with no safetensors beside it, so
all three stopped at "no *.safetensors under ...gguf".

THE READING. The GGUF header carries every tensor's name, ggml type, shape and
byte size; :func:`host_ledger.gguf_header_facts` already reads it ONCE per file
per process (8.5 s for the unsloth UD-IQ4_XS) and now keeps those rows. Nothing
here opens the file again and nothing reads a tensor byte.

THE NAMES. A reader keys on HF names (``model.layers.<i>.self_attn.``,
``embed_tokens``, ``lm_head``, ``mtp.``), so every GGUF name goes through the
SAME map the loader builds -- the family adapter's ``build_name_map`` from
``gguf_registry`` (qwen35 for the 27B), fed the tensor names from the cached
facts instead of a second header read. Two groups of tensors are outside that
map by the loader's own design and are named here the way the loader names them:

* the NEXTN/MTP block ``blk.<depth>`` (``block_count`` 65 = 64 backbone blocks
  + 1 nextn block on the unsloth 27B) -- it is NOT a backbone layer. Its
  tensors take the loader's draft names (``Qwen35GGUFAdapter._MTP_TENSOR_MAP``,
  ``mtp.*``), exactly the names the safetensors checkpoint carries, so every
  reader treats it as it treats the safetensors MTP head;
* vision tensors (``v.``/``mm.``) inside the backbone file -- ``model.visual.``.

THE BYTES are the header's own ``n_bytes`` per tensor: a UD mix of IQ2_XS ..
Q6_K inside ONE layer is priced exactly, never as dtype x shape and never as a
per-layer estimate.

Header bytes are ON-DISK bytes. The GGUF loader keeps quantized tensors packed
on the device; the few tensors it builds dense (the F32 carve-out,
``Qwen35GGUFAdapter.unquantized_module_prefixes``) are not re-priced here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


class GgufCensusUnavailable(RuntimeError):
    """The GGUF tensor directory could not be put into HF names -- named, never
    a default: a weight reader that cannot price the checkpoint must refuse."""


@dataclass(frozen=True)
class GgufTensorCensus:
    """Every tensor of one GGUF split set with its HF-side name."""

    source: str
    arch: str
    family: str
    #: backbone layers the loader builds (``block_count - nextn_predict_layers``)
    backbone_depth: int
    #: ``blk.<i>`` indices of the NEXTN/MTP block(s) (``>= backbone_depth``)
    mtp_blocks: Tuple[int, ...]
    n_parts: int
    #: ``(hf_name, gguf_name, ggml_type, shape, n_bytes)`` in file order
    rows: Tuple[Tuple[str, str, str, Tuple[int, ...], int], ...]

    def sizes(self) -> Dict[str, int]:
        """``{hf_name: bytes}`` (summed, should two GGUF tensors ever share one
        HF name)."""
        out: Dict[str, int] = {}
        for hf, _g, _t, _s, nbytes in self.rows:
            out[hf] = out.get(hf, 0) + int(nbytes)
        return out

    @property
    def total_bytes(self) -> int:
        return sum(int(r[4]) for r in self.rows)


#: (realpath, size, mtime_ns) of the named file -> its census (the facts are
#: cached the same way in host_ledger).
_CENSUS_CACHE: Dict[Tuple[str, int, int], GgufTensorCensus] = {}


def is_gguf_checkpoint(model_path: str) -> bool:
    """The server's own GGUF predicate (host_ledger's wrapper of
    ``check_gguf_file``): a FILE that is a GGUF. A directory never is."""
    if not model_path or not os.path.isfile(model_path):
        return False
    from sglang.srt.weg2.host_ledger import _is_gguf_file

    return _is_gguf_file(model_path)


def _sibling_text_config(gguf_file: str) -> dict:
    from sglang.srt.server_args import declared_config_path_for

    cfg_path = declared_config_path_for(gguf_file)
    if cfg_path is None:
        raise GgufCensusUnavailable(
            f"GGUF {gguf_file}: no sibling config.json (declared_config_path_for); "
            f"the family name map needs the model's own config"
        )
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text = dict(cfg.get("text_config") or cfg)
    text.setdefault("model_type", cfg.get("model_type"))
    return text


def gguf_tensor_census(gguf_file: str) -> GgufTensorCensus:
    """The :class:`GgufTensorCensus` of ``gguf_file`` (its whole split set).

    Raises :class:`GgufCensusUnavailable` by name when the model has no bespoke
    GGUF family (the generic GGUF path is not priced here), when the header's
    arch or depth disagrees with the family config, or when a backbone tensor
    has no name in the loader's map.
    """
    st = os.stat(gguf_file)
    key = (os.path.realpath(gguf_file), int(st.st_size), int(st.st_mtime_ns))
    hit = _CENSUS_CACHE.get(key)
    if hit is not None:
        return hit

    from sglang.srt.model_loader import gguf_registry
    from sglang.srt.model_loader.gguf_shards import resolve_gguf_shard_paths
    from sglang.srt.weg2.host_ledger import gguf_header_facts

    facts = gguf_header_facts(gguf_file)
    if not facts.tensors:
        raise GgufCensusUnavailable(
            f"GGUF {gguf_file}: the header facts carry no tensor rows"
        )
    text = _sibling_text_config(gguf_file)
    model_type = text.get("model_type")
    cls = gguf_registry.get_gguf_adapter_class(model_type)
    if cls is None:
        raise GgufCensusUnavailable(
            f"GGUF {gguf_file}: model_type {model_type!r} has no bespoke GGUF family "
            f"in gguf_registry, so there is no loader name map to price its tensors "
            f"with (the generic transformers GGUF path is not priced by G6)"
        )
    adapter = cls(SimpleNamespace(**text), gguf_file)
    if facts.arch and adapter.arch != facts.arch:
        raise GgufCensusUnavailable(
            f"GGUF {gguf_file}: header arch {facts.arch!r} but the config's family "
            f"{cls.FAMILY} maps model_type {model_type!r} to {adapter.arch!r}"
        )
    depth = facts.backbone_depth
    if depth is not None and int(depth) != int(adapter.num_layers):
        raise GgufCensusUnavailable(
            f"GGUF {gguf_file}: header depth {depth} (block_count {facts.block_count} "
            f"- nextn {facts.nextn_predict_layers or 0}) != config num_hidden_layers "
            f"{adapter.num_layers}; the launcher refuses this as W162 first"
        )
    # The adapter's own lazy caches, filled from the one header read: the name
    # map is the loader's, the file is not opened again.
    adapter._shard_paths = resolve_gguf_shard_paths(gguf_file)
    adapter._file_tensor_names = {row.name for row in facts.tensors}
    name_map = adapter.build_name_map()
    mtp_map = dict(getattr(adapter, "_MTP_TENSOR_MAP", {}) or {})
    n = int(adapter.num_layers)
    rows = []
    mtp_blocks = set()
    for row in facts.tensors:
        hf = name_map.get(row.name)
        if hf is None:
            m = _BLK_RE.match(row.name)
            if m is not None and int(m.group(1)) >= n:
                blk, suffix = int(m.group(1)), m.group(2)
                mtp_blocks.add(blk)
                hf = mtp_map.get(suffix) if blk == n else None
                if hf is None:  # a tensor the draft map does not name
                    hf = f"mtp.gguf.{row.name}"
            elif row.name.startswith(("v.", "mm.")):
                hf = f"model.visual.gguf.{row.name}"
            else:
                raise GgufCensusUnavailable(
                    f"GGUF {gguf_file}: tensor {row.name!r} has no name in the "
                    f"{cls.FAMILY} loader map"
                )
        rows.append((hf, row.name, row.ggml_type, tuple(row.shape), int(row.n_bytes)))
    census = GgufTensorCensus(
        source=str(gguf_file),
        arch=str(facts.arch or adapter.arch),
        family=str(cls.FAMILY),
        backbone_depth=n,
        mtp_blocks=tuple(sorted(mtp_blocks)),
        n_parts=int(facts.n_parts),
        rows=tuple(rows),
    )
    _CENSUS_CACHE[key] = census
    return census
