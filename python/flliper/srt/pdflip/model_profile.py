# SPDX-License-Identifier: Apache-2.0
"""PROFIL-EDITOR S3 (Auftrag 960, 03.10.2026): das Modellprofil ``flliper.model/1``,
am Desk aus einem Modellverzeichnis geschätzt -- ohne GPU, ohne Gewichte zu laden.

PURE: nur Standardbibliothek.  Das Rig-Dashboard (ohne flliper-Import) lädt diese Datei per Dateipfad
(wie ``card_identity.py`` im Kartenplaner) und ruft :func:`estimate`.

Was gelesen wird (nie ein Gewichtsbyte):

* ``config.json`` (+ ``text_config``), ``quantization_config``;
* die Kopfzeilen der ``*.safetensors``-Shards (8-Byte-Länge + JSON-Header je Datei) bzw. der Kopf einer
  ``.gguf``-Datei (Tensorverzeichnis mit Typ und Form);
* fehlen die Shards (nur Config, oder Config + ``model.safetensors.index.json`` ohne Dateien), rechnet
  :func:`estimate_weights_from_config` aus der Geometrie und dem Quantisierungsformat.

Jeder Zahlenwert trägt seine Quelle: ``{"v": ..., "src": "config" | "Index" | "geschätzt" | "stat"}``.

* ``config``      steht wörtlich in ``config.json``;
* ``Index``       aus den Shard-/GGUF-Köpfen (exakte Tensorgrößen) oder dem Safetensors-Index;
* ``geschätzt``   aus der Geometrie gerechnet (Formel steht in ``note``);
* ``stat``        Dateigröße auf der Platte.

Die Bausteine des Planers sind hier zusammengeführt, nicht nachgebaut: die Klassifikation der Tensoren und
die Layerfamilien folgen ``planner/pp_cut.checkpoint_weight_terms`` / ``layer_families_from_config``,
die KV-Zelle ``pp_cut.kv_cell_bytes_per_attention_layer`` (gegengeprüft am Metall: 1088 B/Layer NF fp8),
die Extend-Rate der Geometrieformel aus Q-694b (``extend_trim.derived_rate_mib``, 1de15e82ca).
Der Abgleich gegen die Handzeilen steht in ``test_profil_s3_modell_1003.py``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import struct
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.model/1"
MIB = float(1 << 20)

SRC_CONFIG = "config"
SRC_INDEX = "Index"
SRC_ESTIMATE = "geschätzt"
SRC_STAT = "stat"

FAM_ATTN = "attn"
FAM_GDN = "gdn"
FAM_MAMBA = "mamba"

#: Bytes je Element nach Safetensors-Dtype-Kürzel (F4 = zwei Elemente je Byte)
SAFETENSORS_DTYPE_BYTES: Dict[str, float] = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 0.5,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
#: Bytes je Elemente eines Config-``dtype``-Namens
TORCH_DTYPE_BYTES: Dict[str, float] = {
    "float64": 8, "double": 8, "float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2,
    "float8_e4m3fn": 1, "float8_e4m3": 1, "fp8_e4m3": 1, "fp8": 1, "float8_e5m2": 1, "fp8_e5m2": 1, "int8": 1,
}

#: ggml-Typ -> (Name, Elemente je Block, Bytes je Block)
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}


class ModelProfileError(RuntimeError):
    """Das Modellverzeichnis lässt sich nicht lesen oder schätzen -- benannt, nie geraten."""


def _v(value: Any, src: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"v": value, "src": src}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> Tuple[Dict[str, Any], str]:
    """``(config, config_path)`` eines Modellverzeichnisses oder einer ``.gguf``-Datei (Nachbar-``config.json``)."""
    base = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    cfg_path = os.path.join(base, "config.json")
    if not os.path.isfile(cfg_path):
        raise ModelProfileError("%s: no config.json next to the model" % path)
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh), cfg_path


def text_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    t = dict(cfg.get("text_config") or cfg)
    for k in ("quantization_config", "dtype", "torch_dtype", "tie_word_embeddings", "architectures"):
        if k not in t and k in cfg:
            t[k] = cfg[k]
    return t


def _int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def layer_families(t: Mapping[str, Any]) -> Tuple[str, ...]:
    """Mixer je Layer (``attn`` / ``gdn`` / ``mamba``): ``layer_types`` / ``layers_block_type`` ausdrücklich,
    sonst ``full_attention_interval`` (Layer l ist Attention, wenn ``(l+1) % interval == 0``), sonst alles Attention."""
    n = _int(t.get("num_hidden_layers"))
    if n <= 0:
        raise ModelProfileError("num_hidden_layers is missing or not positive: the depth is unknown")
    explicit = t.get("layer_types") or t.get("layers_block_type")
    linear_kind = FAM_MAMBA if (t.get("mamba_num_heads") or t.get("ssm_state_size")) and not t.get("linear_num_value_heads") else FAM_GDN
    if explicit:
        if len(explicit) != n:
            raise ModelProfileError("layer_types has %d entries, num_hidden_layers is %d" % (len(explicit), n))
        out = []
        for k in explicit:
            s = str(k)
            if s in ("full_attention", "attention", "attn", "sliding_attention"):
                out.append(FAM_ATTN)
            elif s in ("mamba", "mamba2"):
                out.append(FAM_MAMBA)
            else:
                out.append(linear_kind)
        return tuple(out)
    interval = _int(t.get("full_attention_interval"))
    if interval > 0:
        return tuple(FAM_ATTN if (i + 1) % interval == 0 else linear_kind for i in range(n))
    return tuple([FAM_ATTN] * n)


def sliding_info(t: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """``(Fenster in Token, Zahl der Gleitfenster-Layer)`` -- nur wenn ``layer_types`` ausdrücklich ``sliding_attention`` nennt UND
    ``sliding_window`` gesetzt ist (DFlash2-Draft: 2048 / 5); sonst ``None`` (volle Attention, nie geraten)."""
    window = _int(t.get("sliding_window"))
    explicit = t.get("layer_types") or t.get("layers_block_type") or []
    n = sum(1 for k in explicit if str(k) == "sliding_attention")
    return (window, n) if window > 0 and n > 0 else None


def rope_info(cfg: Mapping[str, Any], t: Mapping[str, Any]) -> Dict[str, Any]:
    rp = t.get("rope_parameters") or t.get("rope_scaling") or cfg.get("rope_scaling") or {}
    theta = rp.get("rope_theta", t.get("rope_theta", cfg.get("rope_theta")))
    rtype = rp.get("rope_type") or rp.get("type") or "default"
    out: Dict[str, Any] = {"type": _v(rtype, SRC_CONFIG)}
    if theta is not None:
        out["theta"] = _v(theta, SRC_CONFIG)
    if rp.get("factor") is not None:
        out["factor"] = _v(rp["factor"], SRC_CONFIG)
    if rp.get("original_max_position_embeddings") is not None:
        out["original_max_position_embeddings"] = _v(rp["original_max_position_embeddings"], SRC_CONFIG)
    prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor"))
    if prf is not None:
        out["partial_rotary_factor"] = _v(prf, SRC_CONFIG)
    if rp.get("mrope_section"):
        out["mrope_section"] = _v(list(rp["mrope_section"]), SRC_CONFIG)
    return out


# ---------------------------------------------------------------------------
# Tensorverzeichnisse: Safetensors-Köpfe, GGUF-Kopf, Index
# ---------------------------------------------------------------------------


class Tensor:
    __slots__ = ("name", "nbytes", "dtype", "shape")

    def __init__(self, name: str, nbytes: float, dtype: str, shape: Tuple[int, ...]):
        self.name = name
        self.nbytes = nbytes
        self.dtype = dtype
        self.shape = shape


class TensorDir:
    """Das Tensorverzeichnis eines Modells: Namen, Bytes, Dtype, Form -- aus den Köpfen."""

    def __init__(self, kind: str, tensors: List[Tensor], files: Dict[str, int], header_bytes: int,
                 index_total_size: Optional[int] = None, warnings: Optional[List[str]] = None):
        self.kind = kind                    # "safetensors" | "gguf" | "index" | "none"
        self.tensors = tensors
        self.files = files                  # Dateiname -> Größe auf der Platte
        self.header_bytes = header_bytes
        self.index_total_size = index_total_size
        self.warnings = warnings or []

    @property
    def total_bytes(self) -> float:
        return float(sum(t.nbytes for t in self.tensors))

    @property
    def disk_bytes(self) -> int:
        return int(sum(self.files.values()))


def _read_safetensors_header(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                raise ModelProfileError("%s: file shorter than 8 bytes" % path)
            (n,) = struct.unpack("<Q", raw)
            if n > (1 << 30):
                raise ModelProfileError("%s: header length %d implausible" % (path, n))
            blob = fh.read(n)
            if len(blob) < n:
                raise ModelProfileError("%s: header truncated (%d of %d bytes)" % (path, len(blob), n))
        return {"__len__": 8 + n, **json.loads(blob)}
    except (OSError, ValueError) as exc:
        raise ModelProfileError("safetensors header of %s not readable: %s" % (path, exc)) from exc


def scan_safetensors(model_dir: str) -> TensorDir:
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    idx_total: Optional[int] = None
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as fh:
                idx_total = _int((json.load(fh).get("metadata") or {}).get("total_size")) or None
        except (OSError, ValueError):
            idx_total = None
    if not shards:
        return TensorDir("index" if idx_total else "none", [], {}, 0, idx_total)
    tensors: List[Tensor] = []
    files: Dict[str, int] = {}
    warnings: List[str] = []
    header_bytes = 0
    for shard in shards:
        h = _read_safetensors_header(shard)
        header_bytes += h.pop("__len__")
        try:
            files[os.path.basename(shard)] = os.path.getsize(shard)
        except OSError:
            files[os.path.basename(shard)] = 0
        for name, meta in h.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            elems = 1
            for d in meta.get("shape", []):
                elems *= int(d)
            dt = str(meta.get("dtype"))
            if dt not in SAFETENSORS_DTYPE_BYTES:
                warnings.append("unknown dtype %s at %s: calculated with 2 bytes" % (dt, name))
            tensors.append(Tensor(name, elems * SAFETENSORS_DTYPE_BYTES.get(dt, 2), dt, tuple(int(d) for d in meta.get("shape", []))))
    return TensorDir("safetensors", tensors, files, header_bytes, idx_total, warnings[:20])


class _Buf:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) < n:
            raise ModelProfileError("GGUF-Kopf endet vorzeitig")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")


_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_value(b: _Buf, vt: int) -> Any:
    if vt in _GGUF_SCALAR:
        fmt = _GGUF_SCALAR[vt]
        return struct.unpack(fmt, b.read(struct.calcsize(fmt)))[0]
    if vt == 8:
        return b.string()
    if vt == 9:
        et = b.u32()
        n = b.u64()
        if et in _GGUF_SCALAR and n > 64:
            b.read(struct.calcsize(_GGUF_SCALAR[et]) * n)      # grosse Zahlenfelder (Token-Typen) überspringen
            return None
        return [_gguf_value(b, et) for _ in range(n)]
    raise ModelProfileError("GGUF: value type %d unknown" % vt)


def scan_gguf(gguf_file: str) -> Tuple[TensorDir, Dict[str, Any]]:
    """Tensorverzeichnis und Schlüssel/Wert-Kopf einer GGUF-Datei (nur der Kopf, nie die Tensordaten)."""
    kv: Dict[str, Any] = {}
    tensors: List[Tensor] = []
    try:
        with open(gguf_file, "rb") as fh:
            b = _Buf(fh)
            if b.read(4) != b"GGUF":
                raise ModelProfileError("%s: not a GGUF (magic missing)" % gguf_file)
            version = b.u32()
            if version < 2:
                raise ModelProfileError("%s: GGUF version %d is not read" % (gguf_file, version))
            n_tensors = b.u64()
            n_kv = b.u64()
            for _ in range(n_kv):
                key = b.string()
                vt = b.u32()
                kv[key] = _gguf_value(b, vt)
            for _ in range(n_tensors):
                name = b.string()
                nd = b.u32()
                dims = tuple(b.u64() for _ in range(nd))
                gt = b.u32()
                b.u64()                                  # Datenoffset
                if gt not in GGML_TYPES:
                    raise ModelProfileError("%s: ggml type %d at %s unknown" % (gguf_file, gt, name))
                tname, blck, tsize = GGML_TYPES[gt]
                elems = 1
                for d in dims:
                    elems *= d
                tensors.append(Tensor(name, elems // blck * tsize, tname, dims))
    except (OSError, struct.error) as exc:
        raise ModelProfileError("GGUF header of %s not readable: %s" % (gguf_file, exc)) from exc
    files = {os.path.basename(gguf_file): os.path.getsize(gguf_file)}
    return TensorDir("gguf", tensors, files, 0), kv


_GGUF_SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<no>\d{5})-of-(?P<total>\d{5})\.gguf$")

#: ``general.file_type`` (llama.cpp ``llama_ftype``): nur die Werte, die am Rig als Dateiname gegengeprüft sind (IQ4_XS 30, Q6_K 18,
#: Q8_0 7) und die gleich benannten Standardstufen; jeder andere Wert steht als ``ftype N`` ohne Namen, nie geraten.
GGUF_FILE_TYPES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


def gguf_parts(gguf_file: str) -> Tuple[List[str], List[str]]:
    """``(vorhandene Teile, fehlende Teile)`` einer GGUF-Datei.  ``name-00002-of-00003.gguf`` gehört zu einem geteilten Satz
    (Kopf mit den Schlüsseln nur in Teil 1); jede andere Datei ist ihr eigener Satz."""
    m = _GGUF_SPLIT_RE.match(os.path.basename(gguf_file))
    if m is None:
        return [gguf_file], []
    base = os.path.dirname(os.path.abspath(gguf_file))
    total = int(m.group("total"))
    have: List[str] = []
    missing: List[str] = []
    for i in range(1, total + 1):
        name = "%s-%05d-of-%05d.gguf" % (m.group("prefix"), i, total)
        full = os.path.join(base, name)
        if os.path.isfile(full):
            have.append(full)
        else:
            missing.append(name)
    return have, missing


def gguf_sets(names: Iterable[str]) -> Dict[str, List[str]]:
    """Dateinamen eines Verzeichnisses -> Sätze: ein geteilter Satz (gleiches Präfix und Teilezahl) ist EIN Eintrag."""
    out: Dict[str, List[str]] = {}
    for n in sorted(names):
        m = _GGUF_SPLIT_RE.match(n)
        out.setdefault("%s/%s" % (m.group("prefix"), m.group("total")) if m else n, []).append(n)
    return out


def scan_gguf_set(gguf_file: str) -> Tuple[TensorDir, Dict[str, Any], List[str]]:
    """Tensorverzeichnis und Kopf einer GGUF-Datei ODER ihres ganzen geteilten Satzes; ``(td, kv, dateien)``.

    Fehlt ein Teil, wird verweigert (benannt): die Gewichtssumme eines unvollständigen Satzes wäre ein Bruchteil und gälte als Wert."""
    parts, missing = gguf_parts(gguf_file)
    if missing:
        raise ModelProfileError("%s: GGUF set incomplete, missing %s" % (gguf_file, ", ".join(missing)))
    if len(parts) == 1:
        td, kv = scan_gguf(parts[0])
        return td, kv, parts
    tensors: List[Tensor] = []
    files: Dict[str, int] = {}
    kv: Dict[str, Any] = {}
    for part in parts:
        td_i, kv_i = scan_gguf(part)
        tensors.extend(td_i.tensors)
        files.update(td_i.files)
        for k, v in kv_i.items():
            if not k.startswith("split.") or k == "split.count":
                kv.setdefault(k, v)
    return TensorDir("gguf", tensors, files, 0), kv, parts


def config_from_gguf(kv: Mapping[str, Any], td: TensorDir) -> Tuple[Dict[str, Any], List[str]]:
    """Eine HF-artige Konfiguration aus dem GGUF-Kopf, wenn keine ``config.json`` neben der Datei liegt: ``(config, Hinweise)``.

    Gelesen werden nur Schlüssel, die der Kopf wörtlich trägt (``<arch>.block_count``, ``embedding_length``, ``attention.*``,
    ``expert_*``, ``ssm.*``, ``hyper_connection.*``, ``attention.indexer.*``).  Die GDN-Abbildung (``ssm.group_count`` = Schlüsselköpfe,
    ``state_size`` = Kopfdim., ``time_step_rank`` = Wertköpfe, ``inner_size`` / ``time_step_rank`` = Wertkopfdim.) gilt nur für ``qwen*``;
    bei jeder anderen Architektur mit ``ssm.*`` bleibt der Mixer unabgebildet und der Hinweis sagt es.  Der Aktivierungs-Dtype steht nicht
    im Kopf: ``bfloat16`` ist angenommen."""
    arch = str(kv.get("general.architecture") or "")
    if not arch:
        raise ModelProfileError("GGUF without general.architecture: no geometry readable")

    def g(key: str) -> Any:
        return kv.get(arch + "." + key)

    def num(v: Any) -> int:
        if isinstance(v, (list, tuple)):
            vals = [_int(x) for x in v if _int(x) > 0]
            return max(vals) if vals else 0
        return _int(v)

    bc = num(g("block_count"))
    if bc <= 0:
        raise ModelProfileError("GGUF without %s.block_count: the depth is unknown" % arch)
    notes: List[str] = []
    nextn = num(g("nextn_predict_layers"))
    n_layers = bc - nextn if 0 < nextn < bc else bc
    emb = num(g("embedding_length"))
    heads = num(g("attention.head_count"))
    kvh = num(g("attention.head_count_kv")) or heads
    key_len = num(g("attention.key_length")) or (emb // heads if heads else 0)
    t: Dict[str, Any] = {"model_type": arch, "num_hidden_layers": n_layers, "hidden_size": emb, "num_attention_heads": heads,
                         "num_key_value_heads": kvh, "head_dim": key_len, "dtype": "bfloat16"}
    notes.append("config from the GGUF header (%s): activation dtype bfloat16 assumed, vocabulary from token_embd" % arch)
    if num(g("context_length")):
        t["max_position_embeddings"] = num(g("context_length"))
    ffn = g("feed_forward_length")
    if not isinstance(ffn, (list, tuple)) and num(ffn):
        t["intermediate_size"] = num(ffn)
    if num(g("full_attention_interval")):
        t["full_attention_interval"] = num(g("full_attention_interval"))
    for tn in td.tensors:
        if tn.name.startswith("token_embd") and len(tn.shape) == 2:
            t["vocab_size"] = int(tn.shape[1] if tn.shape[0] == emb else tn.shape[0])
            break
    if num(g("ssm.inner_size")):
        if arch.startswith("qwen"):
            rank = num(g("ssm.time_step_rank"))
            t["linear_num_key_heads"] = num(g("ssm.group_count"))
            t["linear_key_head_dim"] = num(g("ssm.state_size"))
            t["linear_num_value_heads"] = rank
            t["linear_value_head_dim"] = num(g("ssm.inner_size")) // rank if rank else 0
            t["linear_conv_kernel_dim"] = num(g("ssm.conv_kernel")) or 4
        else:
            notes.append("ssm.* keys of architecture %s not mapped: mixer state unknown" % arch)
    if num(g("expert_count")):
        t["num_experts"] = num(g("expert_count"))
        t["num_experts_per_tok"] = num(g("expert_used_count"))
        t["moe_intermediate_size"] = num(g("expert_feed_forward_length"))
        if num(g("expert_shared_feed_forward_length")):
            t["shared_expert_intermediate_size"] = num(g("expert_shared_feed_forward_length"))
    if num(g("hyper_connection.count")):
        t["hc_count"] = num(g("hyper_connection.count"))
        t["hc_lowrank"] = num(g("hyper_connection.low_rank"))
    if num(g("attention.indexer.head_count")):
        t["indexer_n_heads"] = num(g("attention.indexer.head_count"))
        t["indexer_head_dim"] = num(g("attention.indexer.key_length"))
        t["indexer_budget"] = num(g("attention.indexer.top_k"))
    if nextn:
        t["mtp_num_hidden_layers"] = nextn
    rope: Dict[str, Any] = {"rope_type": "default"}
    if g("rope.freq_base") is not None:
        rope["rope_theta"] = g("rope.freq_base")
    secs = g("rope.dimension_sections")
    if isinstance(secs, list) and secs:
        rope["mrope_section"] = [int(x) for x in (secs[:-1] if len(secs) == 4 and not secs[-1] else secs)]
    if num(g("rope.dimension_count")) and key_len:
        t["partial_rotary_factor"] = round(num(g("rope.dimension_count")) / key_len, 6)
    t["rope_parameters"] = rope
    # Ausgabe-Gate der Attention: q_proj des ersten Attention-Layers ist doppelt so breit
    interval = t.get("full_attention_interval") or 1
    first_attn = interval - 1 if interval > 1 else 0
    for tn in td.tensors:
        if tn.name == "blk.%d.attn_q.weight" % first_attn and len(tn.shape) == 2 and heads and key_len:
            t["attn_output_gate"] = bool(int(tn.shape[1]) == 2 * heads * key_len)
            break
    return {"architectures": ["gguf:" + arch], "model_type": arch, "text_config": t}, notes


# ---------------------------------------------------------------------------
# Namensklassifikation (folgt checkpoint_weight_terms)
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_EXPERT_ID_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
_GGUF_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def classify(name: str) -> Tuple[str, Optional[int], str]:
    """``(Klasse, Layerindex, Rolle)`` eines HF-Tensornamens.

    Klassen: ``mtp``, ``visual``, ``lm_head``, ``embed``, ``expert``, ``ple``, ``ngram``, ``layer``, ``other``.
    Rolle (nur ``layer``): ``attn`` | ``gdn`` | ``mamba`` | ``mlp`` | ``shared`` | ``router`` | ``norm`` | ``hc`` | ``misc``.
    """
    if name.startswith("mtp"):
        return "mtp", None, ""
    if ".visual." in name or name.startswith("model.visual") or name.startswith("visual."):
        return "visual", None, ""
    if name == "lm_head.weight" or name.endswith(".lm_head.weight") or name.startswith("lm_head."):
        return "lm_head", None, ""
    if "embed_tokens" in name:
        return "embed", None, ""
    m = _LAYER_RE.search(name)
    if m is None:
        return "other", None, ""
    idx = int(m.group(1))
    if ".mlp.experts." in name:
        return "expert", idx, ""
    if "ngram_embedding" in name:
        return "ngram", idx, ""
    if ".ple." in name:
        return "ple", idx, ""
    if ".self_attn." in name:
        return "layer", idx, FAM_ATTN
    if ".linear_attn." in name:
        return "layer", idx, FAM_GDN
    if ".mamba." in name or ".mixer." in name:
        return "layer", idx, FAM_MAMBA
    if ".mlp.shared_expert" in name:
        return "layer", idx, "shared"
    if re.search(r"\.mlp\.(gate|router)\b", name) or ".mlp.shared_expert_gate" in name:
        return "layer", idx, "router"
    if ".mlp." in name:
        return "layer", idx, "mlp"
    if "hyper_connection" in name:
        return "layer", idx, "hc"
    if "norm" in name:
        return "layer", idx, "norm"
    return "layer", idx, "misc"


def classify_gguf(name: str, n_backbone: int) -> Tuple[str, Optional[int], str]:
    """Wie :func:`classify`, für GGUF-Namen (``blk.N.attn_q.weight`` ...)."""
    if name.startswith("v.") or name.startswith("mm."):
        return "visual", None, ""
    if name == "output.weight":
        return "lm_head", None, ""
    if name.startswith("token_embd"):
        return "embed", None, ""
    m = _GGUF_BLK_RE.match(name)
    if m is None:
        return "other", None, ""
    idx, rest = int(m.group(1)), m.group(2)
    if idx >= n_backbone:
        return "mtp", None, ""
    if "_exps" in rest:
        return "expert", idx, ""
    if "_shexp" in rest or "shared_expert" in rest:
        return "layer", idx, "shared"
    if rest.startswith("ssm_") or rest.startswith("attn_qkv") or rest.startswith("attn_gate") or rest.startswith("ssm"):
        return "layer", idx, FAM_GDN
    if rest.startswith("attn_"):
        return "layer", idx, FAM_ATTN if not rest.startswith("attn_norm") and not rest.startswith("attn_post_norm") else "norm"
    if rest.startswith("ffn_gate_inp"):
        return "layer", idx, "router"
    if rest.startswith("ffn_") and "norm" not in rest:
        return "layer", idx, "mlp"
    if "norm" in rest:
        return "layer", idx, "norm"
    return "layer", idx, "misc"


# ---------------------------------------------------------------------------
# Quantisierung: Format aus der Config und aus den Tensoren
# ---------------------------------------------------------------------------

#: Bytes je Gewichtselement (Nutzlast, ohne Skalen) je Speicherklasse
_CLASS_BITS = {"bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "nvfp4": 4, "int4": 4, "int6": 6}


def _qc(cfg: Mapping[str, Any], t: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("quantization_config") or t.get("quantization_config") or {})


def _group_class(group: Mapping[str, Any], qc: Mapping[str, Any]) -> Optional[str]:
    w = group.get("weights") or {}
    bits = _int(w.get("num_bits"))
    wtype = str(w.get("type") or "int")
    if bits == 4 and wtype == "float":
        return "nvfp4"
    if bits == 8 and wtype == "float":
        return "fp8"
    if bits == 8:
        return "int8"
    if bits == 4:
        return "int4"
    if bits:
        return "int%d" % bits
    return None


def quant_format_from_config(cfg: Mapping[str, Any], t: Mapping[str, Any], gguf: bool = False) -> Dict[str, Any]:
    """Das in der Config erklärte Format: ``{"classes": [...], "bits": [...], "method", "group_size"}``.

    Registry-Name (``form.WeightFormat.name``): ``int8`` | ``fp8`` | ``nvfp4`` | ``int4`` | ``gguf`` | ``bf16``; bei
    gemischten Bitbreiten ein ``-mixed``-Zusatz (``int4-mixed`` = vorwiegend 4 Bit, daneben 6/8 Bit)."""
    qc = _qc(cfg, t)
    method = str(qc.get("quant_method") or ("gguf" if gguf else "none"))
    classes: List[str] = []
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        c = _group_class(g, qc)
        if c and c not in classes:
            classes.append(c)
    if method == "fp8" and "fp8" not in classes:
        classes.append("fp8")
    if str(qc.get("quant_algo") or "") == "NVFP4" and "nvfp4" not in classes:
        classes.append("nvfp4")
    return {"method": method, "classes": classes, "group_size": qc.get("group_size"),
            "mixed_bits": len({_int((g.get("weights") or {}).get("num_bits")) for g in groups.values()} - {0}) > 1}


# ---------------------------------------------------------------------------
# Das Schätzprofil
# ---------------------------------------------------------------------------


def _dtype_bytes(name: Any, default: float = 2.0) -> float:
    return TORCH_DTYPE_BYTES.get(str(name or "").replace("torch.", ""), default)


def kv_cell_bytes(kv_heads: int, head_dim: int, v_head_dim: int, dtype_bytes: float, scale_block: int = 16) -> Tuple[float, float]:
    """``(Nutzlast, Skalenpuffer)`` je Token und Attention-Layer.  Dieselben Terme wie
    ``pp_cut.kv_cell_bytes_per_attention_layer``; der Skalenpuffer gilt nur für 1-Byte-Typen (fp8): NF fp8
    kv_heads 2, head_dim 256 -> 1024 + 64 = 1088 B (am Metall gegengeprüft)."""
    payload = float(kv_heads) * (float(head_dim) + float(v_head_dim)) * float(dtype_bytes)
    scales = float(kv_heads) * float(head_dim) * 2.0 * float(dtype_bytes) / max(1, scale_block) if dtype_bytes < 2 else 0.0
    return payload, scales


def extend_rate_mib_per_row(cfg: Mapping[str, Any]) -> Optional[float]:
    """Q-694b ``extend_trim.derived_rate_mib`` (1de15e82ca): Startrate des D-Extends aus der Geometrie.

    ``(6 x hidden + 3 x intermediate + 2 x mixer_width [+ 2 x top_k x hidden + num_experts]) x max(4, dtype)`` je Zeile,
    keine TP-Teilung, aufgerundet auf vier Nachkommastellen.  ``None`` ohne ``hidden_size``."""
    try:
        t = cfg.get("text_config") or cfg
        hidden = _int(t.get("hidden_size"))
        if hidden <= 0:
            return None
        dense_i = _int(t.get("intermediate_size"))
        moe_i = _int(t.get("moe_intermediate_size")) * _int(t.get("num_experts_per_tok"))
        moe_i += _int(t.get("shared_expert_intermediate_size")) if moe_i else 0
        inter = max(dense_i, moe_i) or 4 * hidden
        heads = _int(t.get("num_attention_heads"))
        kv_heads = _int(t.get("num_key_value_heads"), heads)
        head_dim = _int(t.get("head_dim"), hidden // heads if heads else 0)
        full = heads * head_dim * (2 if t.get("attn_output_gate") else 1) + 2 * kv_heads * head_dim
        lk = _int(t.get("linear_num_key_heads")) * _int(t.get("linear_key_head_dim"))
        lv_heads = _int(t.get("linear_num_value_heads"))
        lv = lv_heads * _int(t.get("linear_value_head_dim"))
        linear = 2 * lk + 2 * lv + 2 * lv_heads
        mixer = max(full, linear, hidden)
        top_k = _int(t.get("num_experts_per_tok")) if moe_i else 0
        moe_extra = 2 * top_k * hidden + _int(t.get("num_experts")) if top_k else 0
        dtype = str(t.get("torch_dtype") or t.get("dtype") or cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
        elem = max(4, int(_dtype_bytes(dtype)))
        elems = 6 * hidden + 3 * inter + 2 * mixer + moe_extra
        return math.ceil(elems * elem / MIB * 1e4 - 1e-9) / 1e4
    except Exception:  # noqa: BLE001 -- eine unlesbare Geometrie leitet nichts ab
        return None


def mamba_state_bytes(t: Mapping[str, Any], ssm_dtype: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Zustand eines Linear-Layers je Request (Slot): ``{"conv", "ssm", "total"}`` in Byte, je SSM-Dtype.

    GDN: ``ssm = lv_heads x lv_head_dim x lk_head_dim x dtype``, ``conv = (2 x lk_heads x lk_dim + lv_heads x lv_dim)
    x (kernel - 1) x model_dtype``.  Mamba2: ``ssm = heads x head_dim x state_size``, ``conv = (intermediate + 2 x groups
    x state) x (kernel - 1)``.  ``mamba_ssm_dtype`` der Config ist der Standard; das Rig fährt ``bfloat16``
    (``--mamba-ssm-dtype``): gemessen 1,5588 MiB je Linear-Layer und Slot (P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT)."""
    model_b = _dtype_bytes(t.get("dtype") or t.get("torch_dtype") or "bfloat16")
    cfg_ssm = str(t.get("mamba_ssm_dtype") or "float32")
    dt = str(ssm_dtype or cfg_ssm)
    if _int(t.get("linear_num_value_heads")):
        lk = _int(t.get("linear_num_key_heads")) * _int(t.get("linear_key_head_dim"))
        lvh = _int(t.get("linear_num_value_heads"))
        lvd = _int(t.get("linear_value_head_dim"))
        kern = _int(t.get("linear_conv_kernel_dim"), 4)
        conv = (2 * lk + lvh * lvd) * max(0, kern - 1) * model_b
        ssm_elems = lvh * lvd * _int(t.get("linear_key_head_dim"))
    elif _int(t.get("mamba_num_heads")) or _int(t.get("ssm_state_size")):
        heads = _int(t.get("mamba_num_heads"))
        hd = _int(t.get("mamba_head_dim"))
        st = _int(t.get("ssm_state_size"))
        groups = _int(t.get("n_groups"), 1)
        kern = _int(t.get("conv_kernel"), 4)
        conv = (heads * hd + 2 * groups * st) * max(0, kern - 1) * model_b
        ssm_elems = heads * hd * st
    else:
        return None
    ssm = ssm_elems * _dtype_bytes(dt, 4.0)
    return {"conv": conv, "ssm": ssm, "total": conv + ssm, "ssm_dtype": dt, "ssm_dtype_config": cfg_ssm,
            "ssm_elems": ssm_elems}


def _classify_tensors(td: TensorDir, n_layers: int, gguf_backbone: Optional[int]) -> Dict[str, Any]:
    layer_nonexp: Dict[int, float] = {}
    layer_exp: Dict[int, float] = {}
    layer_ple: Dict[int, float] = {}
    layer_ngram: Dict[int, float] = {}
    role_bytes: Dict[int, Dict[str, float]] = {}
    expert_ids: Dict[int, set] = {}
    fam_seen: Dict[int, str] = {}
    cls: Dict[str, float] = {}
    mtp_layers: set = set()
    for tn in td.tensors:
        if gguf_backbone is not None:
            k, idx, role = classify_gguf(tn.name, gguf_backbone)
        else:
            k, idx, role = classify(tn.name)
        if k == "layer" and idx is not None:
            layer_nonexp[idx] = layer_nonexp.get(idx, 0.0) + tn.nbytes
            rb = role_bytes.setdefault(idx, {})
            rb[role] = rb.get(role, 0.0) + tn.nbytes
            if role in (FAM_ATTN, FAM_GDN, FAM_MAMBA):
                fam_seen.setdefault(idx, role)
        elif k == "expert" and idx is not None:
            layer_exp[idx] = layer_exp.get(idx, 0.0) + tn.nbytes
            m = _EXPERT_ID_RE.search(tn.name)
            if m is not None:
                expert_ids.setdefault(idx, set()).add(int(m.group(1)))
        elif k == "ple" and idx is not None:
            layer_ple[idx] = layer_ple.get(idx, 0.0) + tn.nbytes
        elif k == "ngram" and idx is not None:
            layer_ngram[idx] = layer_ngram.get(idx, 0.0) + tn.nbytes
        elif k == "mtp":
            m = re.search(r"mtp\.layers\.(\d+)\.", tn.name)
            if m:
                mtp_layers.add(int(m.group(1)))
        if k in ("mtp", "visual", "lm_head", "embed", "other"):
            cls[k] = cls.get(k, 0.0) + tn.nbytes
    return {"layer_nonexp": layer_nonexp, "layer_exp": layer_exp, "layer_ple": layer_ple, "layer_ngram": layer_ngram,
            "role_bytes": role_bytes, "expert_ids": expert_ids, "fam_seen": fam_seen, "cls": cls,
            "mtp_layers": mtp_layers}


def _storage_classes(td: TensorDir) -> Dict[str, Dict[str, float]]:
    """Gewichtsspeicherklassen der Tensoren: ``{"class": {"bytes": Nutzlast+Skalen, "params": Elemente}}``.

    Erkannt wird an Dtype und Nachbartensor (``.weight_scale``): U8 + F8-Skala = nvfp4 (zwei Elemente je Byte), I8 =
    int8, F8_E4M3 = fp8, ``.weight_packed`` (I32) = gepackte Bitbreite (``packedN`` bis die Config sie auflöst),
    BF16/F16 = bf16; GGUF je ggml-Typ."""
    out: Dict[str, Dict[str, float]] = {}

    def add(cls: str, b: float, p: float) -> None:
        d = out.setdefault(cls, {"bytes": 0.0, "params": 0.0})
        d["bytes"] += b
        d["params"] += p

    names = {t.name for t in td.tensors}
    for t in td.tensors:
        n = t.name
        if td.kind == "gguf":
            if len(t.shape) >= 2:
                elems = 1
                for d in t.shape:
                    elems *= d
                add("gguf:" + t.dtype, t.nbytes, elems)
            else:
                add("fp32_small" if t.dtype == "F32" else "gguf:" + t.dtype, t.nbytes, 0)
            continue
        if n.endswith(".weight_packed"):
            add("packed", t.nbytes, t.nbytes * 8)          # Elementzahl folgt aus der Bitbreite, siehe Aufrufer
        elif n.endswith(".weight") and t.dtype == "U8" and (n[:-7] + ".weight_scale") in names:
            add("nvfp4", t.nbytes, t.nbytes * 2)
        elif n.endswith(".weight") and t.dtype == "I8" and (n[:-7] + ".weight_scale") in names:
            add("int8", t.nbytes, t.nbytes)
        elif n.endswith(".weight") and t.dtype in ("F8_E4M3", "F8_E5M2") and len(t.shape) >= 2:
            add("fp8", t.nbytes, t.nbytes)
        elif t.dtype in ("BF16", "F16", "F32") and len(t.shape) >= 2 and (n.endswith(".weight") or "embedding" in n):
            add("bf16", t.nbytes, t.nbytes / SAFETENSORS_DTYPE_BYTES[t.dtype])
        else:
            add("aux", t.nbytes, 0)
    return out


def _registry_format_name(qf: Mapping[str, Any], classes: Mapping[str, Mapping[str, float]], gguf: bool) -> Tuple[str, str]:
    """``(Registry-Name, Quelle)`` -- nach den Gewichtselementen der Layer entschieden, die Config bestätigt."""
    if gguf:
        return "gguf", SRC_INDEX
    quant = {k: v for k, v in classes.items() if k in ("nvfp4", "int8", "fp8", "packed")}
    if not quant and not qf["classes"]:
        return "bf16", SRC_INDEX
    if quant:
        dom = max(quant, key=lambda k: quant[k]["params"])
        if dom == "packed":
            bits = [c for c in qf["classes"] if c.startswith("int")]
            dom = "int4" if "int4" in bits or not bits else bits[0]
            if len(set(qf["classes"])) > 1 or qf["mixed_bits"]:
                dom += "-mixed"
        return dom, SRC_INDEX
    return (qf["classes"][0], SRC_CONFIG)


def _vision_params(v: Mapping[str, Any]) -> float:
    h = _int(v.get("hidden_size"))
    depth = _int(v.get("depth"))
    inter = _int(v.get("intermediate_size"))
    if not (h and depth and inter):
        return 0.0
    per_block = 3 * h * h + 3 * h + h * h + h + h * inter + inter + inter * h + h + 4 * h
    patch = _int(v.get("in_channels"), 3) * _int(v.get("temporal_patch_size"), 2) * _int(v.get("patch_size"), 16) ** 2 * h + h
    pos = _int(v.get("num_position_embeddings")) * h
    sm = _int(v.get("spatial_merge_size"), 2) ** 2
    out_h = _int(v.get("out_hidden_size"))
    merger = (h * sm) * (h * sm) + (h * sm) + (h * sm) * out_h + out_h + 2 * h
    return float(depth * per_block + patch + pos + merger)


def estimate(model_path: str, *, draft_path: Optional[str] = None, kv_dtype: Optional[str] = None,
             mamba_ssm_dtype: Optional[str] = None, gguf_file: Optional[str] = None,
             allow_config_only: bool = True) -> Dict[str, Any]:
    """Das Modellprofil ``flliper.model/1`` von ``model_path`` (Verzeichnis oder ``.gguf``-Datei).

    ``kv_dtype`` (``None`` = ``auto``, gleich dem Modell-Dtype) und ``mamba_ssm_dtype`` (``None`` = die Config) wählen den
    Wert, der als ``v`` gilt; beide Varianten stehen daneben.  Nichts wird geladen, nichts geschrieben."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise ModelProfileError("%s does not exist" % model_path)
    warnings: List[str] = []

    # --- Tensorverzeichnis (zuerst: ein GGUF ohne config.json trägt die Geometrie im Kopf) ----------------
    gguf_kv: Dict[str, Any] = {}
    gguf_path = None
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        gguf_path = model_path
    elif os.path.isdir(model_path):
        sets = gguf_sets(os.path.basename(g) for g in glob.glob(os.path.join(model_path, "*.gguf")))
        no_safetensors = not glob.glob(os.path.join(model_path, "*.safetensors"))
        if gguf_file:
            gguf_path = os.path.join(model_path, gguf_file)
        elif len(sets) == 1 and no_safetensors:
            gguf_path = os.path.join(model_path, next(iter(sets.values()))[0])
        elif len(sets) > 1 and no_safetensors:
            raise ModelProfileError("%s: mehrere .gguf (%s) -- die Datei nennen" % (
                model_path, ", ".join(sorted(os.path.basename(g) for g in glob.glob(os.path.join(model_path, "*.gguf"))))))
    gguf_files: List[str] = []
    if gguf_path:
        td, gguf_kv, gguf_files = scan_gguf_set(gguf_path)
    else:
        td = scan_safetensors(model_path)
    warnings.extend(td.warnings)
    cfg_source = "config.json"
    try:
        cfg, cfg_path = load_config(model_path)
    except ModelProfileError:
        if td.kind != "gguf":
            raise
        cfg, notes = config_from_gguf(gguf_kv, td)
        cfg_path, cfg_source = None, "gguf"
        warnings.extend(notes)
    t = text_config(cfg)
    fams = layer_families(t)
    n_layers = len(fams)
    have_tensors = td.kind in ("safetensors", "gguf") and bool(td.tensors)
    gguf_backbone = None
    if td.kind == "gguf":
        arch_name = str(gguf_kv.get("general.architecture") or "")
        bc = _int(gguf_kv.get(arch_name + ".block_count"), n_layers)
        nextn = _int(gguf_kv.get(arch_name + ".nextn_predict_layers"), 0)
        gguf_backbone = n_layers if bc >= n_layers else bc - nextn

    qf = quant_format_from_config(cfg, t, gguf=td.kind == "gguf")
    out: Dict[str, Any] = {"schema": SCHEMA, "path": model_path,
                           "config_path": cfg_path,
                           "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]}
    if cfg_source == "gguf":
        out["config_source"] = _v("gguf", SRC_INDEX, note="no config.json: geometry from the keys of the GGUF header (config_from_gguf)")
    if td.kind == "gguf":
        ft = _int(gguf_kv.get("general.file_type"), -1)
        out["gguf"] = {
            "architecture": _v(str(gguf_kv.get("general.architecture") or ""), SRC_INDEX),
            "block_count": _v(_int(gguf_kv.get(str(gguf_kv.get("general.architecture") or "") + ".block_count")), SRC_INDEX),
            "nextn_predict_layers": _v(_int(gguf_kv.get(str(gguf_kv.get("general.architecture") or "") + ".nextn_predict_layers")), SRC_INDEX),
            "file_type": _v(GGUF_FILE_TYPES.get(ft, "ftype %d" % ft if ft >= 0 else "unknown"), SRC_INDEX,
                            note="general.file_type = %d" % ft),
            "files": _v(len(gguf_files) or 1, SRC_STAT, note="files of the set: %s" % ", ".join(os.path.basename(x) for x in gguf_files)),
        }

    # --- Architektur ---------------------------------------------------------------------------------
    hidden = _int(t.get("hidden_size"))
    heads = _int(t.get("num_attention_heads"))
    kv_heads = _int(t.get("num_key_value_heads"), heads)
    head_dim = _int(t.get("head_dim"), hidden // heads if heads else 0)
    n_experts = max((_int(t.get(k)) for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")), default=0)
    top_k = _int(t.get("num_experts_per_tok"))
    qsa = bool(t.get("indexer_n_heads") or t.get("indexer_budget"))
    counts = {f: sum(1 for x in fams if x == f) for f in (FAM_ATTN, FAM_GDN, FAM_MAMBA)}
    ple_cfg = bool(t.get("ple_layer_ids") or t.get("ple_embed_dim"))
    mtp_cfg = t.get("mtp") or {}
    mtp_n = _int(t.get("mtp_num_hidden_layers"), _int(mtp_cfg.get("num_hidden_layers")) if isinstance(mtp_cfg, dict) else 0)
    out["arch"] = {
        "family": _v("moe" if n_experts > 0 else "dense", SRC_CONFIG),
        "hybrid": _v(counts[FAM_GDN] + counts[FAM_MAMBA] > 0, SRC_CONFIG),
        "model_type": _v(str(t.get("model_type") or cfg.get("model_type") or ""), SRC_CONFIG),
        "architectures": _v(list(cfg.get("architectures") or t.get("architectures") or []), SRC_CONFIG),
        "n_layers": _v(n_layers, SRC_CONFIG),
        "layer_families": _v(list(fams), SRC_CONFIG),
        "layer_counts": _v(counts, SRC_CONFIG),
        "attention": _v("qsa" if qsa else "full", SRC_CONFIG, note="qsa = Indexer-Attention (indexer_* in der Config)"),
        "hidden": _v(hidden, SRC_CONFIG),
        "heads_q": _v(heads, SRC_CONFIG),
        "heads_kv": _v(kv_heads, SRC_CONFIG),
        "head_dim": _v(head_dim, SRC_CONFIG),
        "intermediate": _v(_int(t.get("intermediate_size")), SRC_CONFIG),
        "vocab": _v(_int(t.get("vocab_size")), SRC_CONFIG),
        "tie_word_embeddings": _v(bool(t.get("tie_word_embeddings", False)), SRC_CONFIG),
        "vision": _v(bool(cfg.get("vision_config") or t.get("vision_config")), SRC_CONFIG),
        "ple": _v(ple_cfg, SRC_CONFIG),
        "hyper_connections": _v(_int(t.get("hc_count")), SRC_CONFIG),
    }

    # --- Gewichte ----------------------------------------------------------------------------------------
    weights: Dict[str, Any]
    fmt_name: str
    fmt_src: str
    fmt_detail: Dict[str, Any] = {"method": qf["method"], "config_classes": qf["classes"]}
    layer_family_arr = list(fams)
    moe_layers: List[int] = []
    if have_tensors:
        cl = _classify_tensors(td, n_layers, gguf_backbone)
        ln, le = cl["layer_nonexp"], cl["layer_exp"]
        missing = [i for i in range(n_layers) if i not in ln and i not in le]
        if missing:
            warnings.append("layers without tensors: %s" % missing[:8])
        # Layerfamilie aus den Tensoren bestätigen
        for i, f in cl["fam_seen"].items():
            if i < n_layers and fams[i] != f and not (fams[i] != FAM_ATTN and f != FAM_ATTN):
                warnings.append("Layer %d: Config sagt %s, Tensoren sagen %s" % (i, fams[i], f))
        moe_layers = sorted(le)
        per_layer = [ln.get(i, 0.0) for i in range(n_layers)]
        per_layer_exp = [le.get(i, 0.0) for i in range(n_layers)]
        per_layer_ple = [cl["layer_ple"].get(i, 0.0) for i in range(n_layers)]
        ngram_total = sum(cl["layer_ngram"].values())
        classes = _storage_classes(td)
        fmt_name, fmt_src = _registry_format_name(qf, classes, td.kind == "gguf")
        fmt_detail["storage"] = {k: {"bytes": int(v["bytes"]), "params": int(v["params"])} for k, v in sorted(classes.items())}
        by_fam = {f: [per_layer[i] for i in range(n_layers) if fams[i] == f] for f in (FAM_ATTN, FAM_GDN, FAM_MAMBA)}
        exp_layers = [per_layer_exp[i] for i in moe_layers]
        ids_per_layer = [len(cl["expert_ids"].get(i, ())) for i in moe_layers if cl["expert_ids"].get(i)]
        n_experts_found = max(ids_per_layer) if ids_per_layer else 0
        mtp_bytes = cl["cls"].get("mtp", 0.0)
        weights = {
            "total_bytes": _v(int(td.total_bytes), SRC_INDEX, note="sum of the tensor sizes from the headers (%s)" % td.kind),
            "disk_bytes": _v(td.disk_bytes, SRC_STAT, note="sum of the file sizes; the rest to the tensor sum are the headers (%d B)" % td.header_bytes) if td.disk_bytes else None,
            "header_bytes": _v(td.header_bytes, SRC_STAT, note="safetensors headers (8 B + JSON per shard); tensor sum + headers = file size") if td.header_bytes else None,
            "layers_bytes_nonexpert": _v(int(sum(per_layer)), SRC_INDEX),
            "layers_bytes_expert": _v(int(sum(per_layer_exp)), SRC_INDEX),
            "per_family_mean_bytes": {f: _v(int(sum(v) / len(v)), SRC_INDEX) for f, v in by_fam.items() if v},
            "per_role_bytes": _v({r: int(sum(cl["role_bytes"].get(i, {}).get(r, 0.0) for i in range(n_layers)))
                                  for r in sorted({r for d in cl["role_bytes"].values() for r in d})}, SRC_INDEX),
            "layer_bytes": _v([int(x) for x in per_layer], SRC_INDEX, note="without experts, PLE, n-gram"),
            "layer_expert_bytes": _v([int(x) for x in per_layer_exp], SRC_INDEX),
            "embed_bytes": _v(int(cl["cls"].get("embed", 0.0)), SRC_INDEX),
            "lm_head_bytes": _v(int(cl["cls"].get("lm_head", 0.0)), SRC_INDEX),
            "visual_bytes": _v(int(cl["cls"].get("visual", 0.0)), SRC_INDEX),
            "mtp_bytes": _v(int(mtp_bytes), SRC_INDEX),
            "ple": {"per_layer_bytes": _v([int(x) for x in per_layer_ple], SRC_INDEX),
                    "ngram_table_bytes": _v(int(ngram_total), SRC_INDEX,
                                            note="n-gram embedding: stays on disk (mmap), never reaches the device") if ngram_total else None},
            "other_bytes": _v(int(cl["cls"].get("other", 0.0)), SRC_INDEX),
        }
        weights["ple"] = {k: v for k, v in weights["ple"].items() if v is not None}
        weights = {k: v for k, v in weights.items() if v is not None}
        expert_info = (n_experts_found or n_experts, SRC_INDEX if n_experts_found else SRC_CONFIG)
        mean_exp_layer = (sum(exp_layers) / len(exp_layers)) if exp_layers else 0.0
        mtp_layers_found = len(cl["mtp_layers"]) or (1 if mtp_bytes > 0 else 0)
    else:
        if td.kind == "none" and not allow_config_only:
            raise ModelProfileError("%s: no shards and no GGUF file" % model_path)
        cw = estimate_weights_from_config(cfg, fmt_hint=None)
        fmt_name, fmt_src = cw["format"], SRC_ESTIMATE
        fmt_detail["storage"] = cw["storage"]
        per_layer = cw["layer_bytes"]
        per_layer_exp = cw["layer_expert_bytes"]
        moe_layers = [i for i in range(n_layers) if per_layer_exp[i] > 0]
        by_fam = {f: [per_layer[i] for i in range(n_layers) if fams[i] == f] for f in (FAM_ATTN, FAM_GDN, FAM_MAMBA)}
        total = cw["total_bytes"]
        src_total = SRC_INDEX if td.index_total_size else SRC_ESTIMATE
        weights = {
            "total_bytes": _v(int(td.index_total_size or total), src_total,
                              note=("model.safetensors.index.json total_size" if td.index_total_size else "formula from geometry and quantisation format")),
            "total_bytes_formula": _v(int(total), SRC_ESTIMATE, note="Geometrieformel (estimate_weights_from_config)"),
            "layers_bytes_nonexpert": _v(int(sum(per_layer)), SRC_ESTIMATE),
            "layers_bytes_expert": _v(int(sum(per_layer_exp)), SRC_ESTIMATE),
            "per_family_mean_bytes": {f: _v(int(sum(v) / len(v)), SRC_ESTIMATE) for f, v in by_fam.items() if v},
            "layer_bytes": _v([int(x) for x in per_layer], SRC_ESTIMATE),
            "layer_expert_bytes": _v([int(x) for x in per_layer_exp], SRC_ESTIMATE),
            "embed_bytes": _v(int(cw["embed_bytes"]), SRC_ESTIMATE),
            "lm_head_bytes": _v(int(cw["lm_head_bytes"]), SRC_ESTIMATE),
            "visual_bytes": _v(int(cw["visual_bytes"]), SRC_ESTIMATE),
            "mtp_bytes": _v(int(cw["mtp_bytes"]), SRC_ESTIMATE),
        }
        for w in cw["warnings"]:
            warnings.append(w)
        expert_info = (n_experts, SRC_CONFIG)
        mean_exp_layer = (sum(per_layer_exp[i] for i in moe_layers) / len(moe_layers)) if moe_layers else 0.0
        mtp_layers_found = 0
    out["format"] = _v(fmt_name, fmt_src, detail=fmt_detail)
    out["weights"] = weights
    out["weights_source"] = _v(td.kind if have_tensors else ("config" if td.kind == "none" else td.kind), SRC_INDEX if have_tensors else SRC_ESTIMATE)

    # --- KV ----------------------------------------------------------------------------------------------
    model_dtype = str(t.get("dtype") or t.get("torch_dtype") or "bfloat16")
    auto_b = _dtype_bytes(model_dtype)
    n_attn = counts[FAM_ATTN]
    variants: Dict[str, Dict[str, Any]] = {}
    for name, bytes_ in (("auto", auto_b), ("fp8_e4m3", 1.0)):
        payload, scales = kv_cell_bytes(kv_heads, head_dim, head_dim, bytes_)
        variants[name] = {"payload_bytes": _v(payload, SRC_ESTIMATE), "scale_bytes": _v(scales, SRC_ESTIMATE),
                          "cell_bytes_per_attn_layer_token": _v(payload + scales, SRC_ESTIMATE,
                                                                note="kv_heads x (head_dim + v_head_dim) x Dtype [+ fp8-Skalenpuffer]"),
                          "bytes_per_token_all_attn_layers": _v((payload + scales) * n_attn, SRC_ESTIMATE),
                          "payload_per_token_all_attn_layers": _v(payload * n_attn, SRC_ESTIMATE)}
    chosen = "fp8_e4m3" if str(kv_dtype or "").startswith("fp8") else "auto"
    out["kv"] = {
        "attn_layers": _v(n_attn, SRC_CONFIG),
        "chosen": _v(chosen, SRC_ESTIMATE if kv_dtype else SRC_CONFIG, note="auto = Modell-Dtype %s (%g B)" % (model_dtype, auto_b)),
        "variants": variants,
        "cell_bytes_per_attn_layer_token": variants[chosen]["cell_bytes_per_attn_layer_token"],
    }
    if qsa:
        out["kv"]["indexer"] = _v({k: t.get(k) for k in t if k.startswith("indexer_")}, SRC_CONFIG,
                                  note="indexer cache not calculated in the KV cell (metal: cell = attention layer x 1088 B fp8)")
    slide = sliding_info(t)
    if slide is not None:
        out["kv"]["sliding"] = {"window_tokens": _v(slide[0], SRC_CONFIG), "layers": _v(slide[1], SRC_CONFIG),
                                "note": "sliding-window layers hold at most window_tokens tokens; the KV cell above applies per layer and token"}

    # --- Mamba/GDN-Zustand ------------------------------------------------------------------------------
    ms_cfg = mamba_state_bytes(t, None)
    n_lin = counts[FAM_GDN] + counts[FAM_MAMBA]
    if ms_cfg is not None and n_lin:
        ms_bf = mamba_state_bytes(t, "bfloat16")
        ms_sel = mamba_state_bytes(t, mamba_ssm_dtype)
        out["state"] = {
            "linear_layers": _v(n_lin, SRC_CONFIG),
            "per_linear_layer_per_slot_bytes": _v(int(ms_sel["total"]), SRC_ESTIMATE,
                                                  note="conv + ssm, SSM-Dtype %s" % ms_sel["ssm_dtype"]),
            "per_linear_layer_per_slot_mib": _v(round(ms_sel["total"] / MIB, 4), SRC_ESTIMATE),
            "ssm_dtype": _v(ms_sel["ssm_dtype"], SRC_ESTIMATE if mamba_ssm_dtype else SRC_CONFIG),
            "variants_mib": {"float32": round(mamba_state_bytes(t, "float32")["total"] / MIB, 4),
                             "bfloat16": round(ms_bf["total"] / MIB, 4)},
            "conv_bytes": _v(int(ms_sel["conv"]), SRC_ESTIMATE),
            "ssm_bytes": _v(int(ms_sel["ssm"]), SRC_ESTIMATE),
            "per_request_bytes": _v(int(ms_sel["total"] * n_lin), SRC_ESTIMATE, note="alle Linear-Layer, ein Slot"),
        }
    else:
        out["state"] = {"linear_layers": _v(0, SRC_CONFIG)}

    # --- Experten ----------------------------------------------------------------------------------------
    if n_experts > 0:
        n_e = expert_info[0] or n_experts
        out["experts"] = {
            "n": _v(n_experts, SRC_CONFIG),
            "top_k": _v(top_k, SRC_CONFIG),
            "moe_intermediate": _v(_int(t.get("moe_intermediate_size")), SRC_CONFIG),
            "shared_intermediate": _v(_int(t.get("shared_expert_intermediate_size")), SRC_CONFIG),
            "moe_layers": _v(len(moe_layers), expert_info[1] if have_tensors else SRC_ESTIMATE),
            "bytes_per_expert": _v(int(mean_exp_layer / max(1, n_e)), SRC_INDEX if have_tensors else SRC_ESTIMATE,
                                   note="Mittel der Expertentensoren je Layer / Expertenzahl (Gate+Up+Down samt Skalen)"),
            "bytes_per_moe_layer": _v(int(mean_exp_layer), SRC_INDEX if have_tensors else SRC_ESTIMATE),
            "total_bytes": _v(int(weights["layers_bytes_expert"]["v"]), SRC_INDEX if have_tensors else SRC_ESTIMATE),
        }
    else:
        out["experts"] = {"n": _v(0, SRC_CONFIG)}

    # --- Decode: Gewichtsbytes, die EIN Decode-Token liest (K2: Speicher x Rechengeschwindigkeit) ---------------
    n_e_dec = int(expert_info[0] or n_experts)
    exp_total = float(weights["layers_bytes_expert"]["v"])
    k_dec = min(top_k, n_e_dec) if n_e_dec > 0 else 0
    exp_active = exp_total * k_dec / n_e_dec if n_e_dec > 0 else 0.0
    lm_dec = float(weights["lm_head_bytes"]["v"]) or (float(weights["embed_bytes"]["v"]) if t.get("tie_word_embeddings") else 0.0)
    dec_src = SRC_INDEX if have_tensors else SRC_ESTIMATE
    out["decode"] = {
        "bytes_per_token": _v(int(weights["layers_bytes_nonexpert"]["v"] + exp_active + lm_dec), dec_src,
                              note="non-expert layers + top_k/n of the expert bytes + lm_head (with a shared embedding its bytes); "
                                   "without the embedding row, PLE projections and the n-gram table"),
        "layers_nonexpert_bytes": _v(int(weights["layers_bytes_nonexpert"]["v"]), dec_src),
        "experts_active_bytes": _v(int(exp_active), dec_src, note="%d of %d experts per MoE layer" % (k_dec, n_e_dec) if n_e_dec else "dense: no experts"),
        "lm_head_bytes": _v(int(lm_dec), dec_src),
    }

    # --- Draft / MTP -------------------------------------------------------------------------------------
    draft: Dict[str, Any] = {
        "mtp_layers": _v(mtp_n, SRC_CONFIG),
        "mtp_tensors_found": _v(mtp_layers_found > 0, SRC_INDEX) if have_tensors else None,
        "mtp_bytes": weights["mtp_bytes"],
    }
    draft = {k: v for k, v in draft.items() if v is not None}
    if draft_path:
        draft["external"] = estimate_draft(draft_path)
    out["draft"] = draft

    # --- Kontext, Rope -----------------------------------------------------------------------------------
    rope = rope_info(cfg, t)
    maxpos = _int(t.get("max_position_embeddings"))
    out["context"] = {"max_position_embeddings": _v(maxpos, SRC_CONFIG), "rope": rope}
    if "factor" in rope and rope["type"]["v"] not in ("default",):
        orig = rope.get("original_max_position_embeddings", {}).get("v") or maxpos
        out["context"]["rope_extended_tokens"] = _v(int(orig * float(rope["factor"]["v"])), SRC_ESTIMATE,
                                                    note="original_max_position_embeddings x factor")

    # --- Aktivierung (Extend-Rate, Geometrie) --------------------------------------------------------------
    rate = extend_rate_mib_per_row(cfg)
    out["activation"] = {"extend_rate_mib_per_row": _v(rate, SRC_ESTIMATE if rate is not None else SRC_ESTIMATE,
                                                       note="Q-694b geometry formula; the measurement at the rank replaces it")}
    out["warnings"] = warnings
    out["id"] = profile_id(out)
    return out


#: Zustände von :func:`probe`.  ``estimable``: ``estimate`` liefert ein Profil (Quelle je Wert benannt).
PROBE_STATES: Dict[str, bool] = {
    "complete": True,         # Config (oder GGUF-Kopf) und Gewichtsköpfe lesbar
    "config_only": True,      # nur config.json: Gewichtsbytes aus der Geometrie, Quelle "geschätzt"
    "index_only": True,       # config.json + model.safetensors.index.json ohne Shards: Index-Summe
    "not_mounted": False,     # Pfad existiert nicht
    "empty": False,           # Verzeichnis ohne Einträge: Mountpunkt ohne eingehängtes Modell
    "no_model_files": False,  # Verzeichnis ohne config.json, Shards oder GGUF (Unterverzeichnisse genannt)
    "no_config": False,       # Safetensors ohne config.json
    "gguf_incomplete": False,  # geteilter GGUF-Satz mit fehlendem Teil
    "ambiguous": False,       # mehrere GGUF-Sätze: die Datei nennen
    "unreadable": False,      # Verzeichnis nicht lesbar
}


def probe(model_path: str, *, gguf_file: Optional[str] = None) -> Dict[str, Any]:
    """Der Zustand eines Modellpfads als DATEN, nie als Ausnahme: ``{"state", "estimable", "reason", "path", ...}``.

    Der Planer liest daraus "Modell fehlt" je Wert als ``unverified``, statt an einem Fehlertext zu hängen.  Gelesen werden nur Dateinamen
    (``stat``/``listdir``), bei GGUF-Sätzen die Existenz der Teile; kein Kopf, kein Gewicht."""
    p = os.path.abspath(model_path)
    out: Dict[str, Any] = {"path": p, "has_config": False, "safetensors": 0, "gguf": [], "index": False, "subdirs": []}

    def done(state: str, reason: str) -> Dict[str, Any]:
        out.update({"state": state, "estimable": PROBE_STATES[state], "reason": reason})
        return out

    if not os.path.exists(p):
        return done("not_mounted", "path does not exist (model folder not mounted?)")
    if os.path.isfile(p):
        if not p.endswith(".gguf"):
            return done("no_model_files", "file is neither a model directory nor a .gguf")
        out["gguf"] = [os.path.basename(p)]
        out["has_config"] = os.path.isfile(os.path.join(os.path.dirname(p), "config.json"))
        _, missing = gguf_parts(p)
        if missing:
            return done("gguf_incomplete", "GGUF set incomplete, missing %s" % ", ".join(missing))
        return done("complete", "GGUF file" + ("" if out["has_config"] else " without config.json (geometry from the header)"))
    try:
        entries = sorted(os.listdir(p))
    except OSError as exc:
        return done("unreadable", "directory not readable: %s" % exc)
    if not entries:
        return done("empty", "empty directory (mount point without a mounted model?)")
    out["has_config"] = "config.json" in entries
    out["safetensors"] = sum(1 for e in entries if e.endswith(".safetensors"))
    out["gguf"] = [e for e in entries if e.endswith(".gguf")]
    out["index"] = "model.safetensors.index.json" in entries
    out["subdirs"] = [e for e in entries if not e.startswith(".") and os.path.isdir(os.path.join(p, e))]
    if out["safetensors"]:
        return done("complete", "safetensors shards") if out["has_config"] else done("no_config", "shards without config.json")
    if out["gguf"] or gguf_file:
        sets = gguf_sets(out["gguf"])
        if gguf_file:
            first = gguf_file
        elif len(sets) == 1:
            first = next(iter(sets.values()))[0]
        else:
            return done("ambiguous", "several GGUF sets (%s): name the file" % ", ".join(sorted(sets)))
        _, missing = gguf_parts(os.path.join(p, first))
        if missing:
            return done("gguf_incomplete", "GGUF set incomplete, missing %s" % ", ".join(missing))
        return done("complete", "GGUF" + ("" if out["has_config"] else " without config.json (geometry from the header)"))
    if out["has_config"]:
        return done("index_only", "config and index, no shards") if out["index"] else done("config_only", "only config.json, no weight files")
    if out["index"]:
        return done("no_config", "index without config.json")
    return done("no_model_files", "neither config.json nor weights" + ((" (subdirectories: %s)" % ", ".join(out["subdirs"])) if out["subdirs"] else ""))


def estimate_or_state(model_path: str, **kw: Any) -> Dict[str, Any]:
    """:func:`estimate`, aber ein nicht lesbarer Modellpfad kommt als Zustand zurück: ``{"ok": False, "state", "reason", ...}``;
    sonst ``{"ok": True, "state", "probe", "profile"}``.  Wirft nur bei einem Programmfehler."""
    st = probe(model_path, gguf_file=kw.get("gguf_file"))
    if not st["estimable"]:
        return dict(st, ok=False)
    try:
        prof = estimate(model_path, **kw)
    except ModelProfileError as exc:
        return dict(st, ok=False, state="unreadable", estimable=False, reason=str(exc))
    return {"ok": True, "state": st["state"], "probe": st, "profile": prof}


def profile_id(profile: Mapping[str, Any]) -> str:
    """Hash des kanonischen Inhalts (ohne ``id``, ``path``, ``config_path``): zwei gleiche Modelle, ein Hash."""
    body = {k: v for k, v in profile.items() if k not in ("id", "path", "config_path")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def draft_kind(cfg: Mapping[str, Any], t: Mapping[str, Any], mtp_found: bool, backbone_found: bool) -> Tuple[str, str]:
    """``(Art, Begründung)`` des getrennten Drafts: ``dflash2`` | ``dflash`` (``dflash_config`` oder Architekturname), ``nextn``
    (``mtp.layers.N``-Tensoren ohne eigenen Backbone), ``eagle`` (Architekturname), sonst ``unbekannt`` -- nie geraten."""
    arch = " ".join(str(a) for a in (cfg.get("architectures") or t.get("architectures") or [])).lower()
    if "dflash" in arch or cfg.get("dflash_config") or t.get("dflash_config"):
        return ("dflash2" if "dflash2" in arch else "dflash"), "architecture/dflash_config of the config"
    if mtp_found and not backbone_found:
        return "nextn", "mtp.layers.N tensors and no backbone of its own (config = geometry of the target)"
    if "eagle" in arch:
        return "eagle", "architecture name of the config"
    return "unbekannt", "neither dflash_config nor MTP tensors nor a known architecture name"


def estimate_draft(draft_path: str) -> Dict[str, Any]:
    """Ein getrenntes Draft-Verzeichnis (DFlash2 ``--dflash-draft-path``, NEXTN/MTP ``--speculative-draft-model-path``): Art,
    EIGENE Layerzahl, Gesamtbytes, Aufteilung Einbettung/``lm_head``/Rest, KV-Zelle samt Gleitfenster -- nur Kopfzeilen.

    ``bytes_without_lm_head`` / ``bytes_without_embed_lm_head`` sind dieselben Abzüge wie ``draft_post.p_draft_post_mib`` (P teilt
    ``lm_head`` mit dem Ziel) und ``draft_post.d_draft_host_mib(share_embed=True)`` (D teilt ``embed_tokens`` und ``lm_head``): Teilstring-
    Abzug je Tensorname, ohne Laufzeitpuffer (64,1 MiB) und Produzenten-Transient (622,8 MiB), die in ``draft_post`` stehen.
    Bei einem NEXTN-Verzeichnis ist die Config die des ZIELS: ``n_layers`` ist dort die Zahl der ``mtp.layers.N``, nicht
    ``num_hidden_layers``."""
    draft_path = os.path.abspath(draft_path)
    cfg, _ = load_config(draft_path)
    t = text_config(cfg)
    td = scan_safetensors(draft_path) if os.path.isdir(draft_path) else TensorDir("none", [], {}, 0)
    arch = list(cfg.get("architectures") or [])
    mtp_ids = {int(m.group(1)) for tn in td.tensors for m in [re.match(r"mtp\.layers\.(\d+)\.", tn.name)] if m}
    backbone = any(classify(tn.name)[0] in ("layer", "expert") for tn in td.tensors)
    kind, why = draft_kind(cfg, t, bool(mtp_ids), backbone)
    n_cfg = _int(t.get("num_hidden_layers"))
    n_layers = len(mtp_ids) if kind == "nextn" and mtp_ids else n_cfg
    out: Dict[str, Any] = {"path": draft_path, "architectures": _v(arch, SRC_CONFIG),
                           "n_layers": _v(n_layers, SRC_INDEX if (kind == "nextn" and mtp_ids) else SRC_CONFIG),
                           "format_config": _v(quant_format_from_config(cfg, t)["classes"], SRC_CONFIG),
                           "kind": _v(kind, SRC_CONFIG if kind in ("dflash2", "dflash", "eagle") else SRC_INDEX, note=why),
                           "own_backbone": _v(backbone, SRC_INDEX) if td.tensors else None}
    out = {k: v for k, v in out.items() if v is not None}
    if td.tensors:
        total = float(td.total_bytes)
        embed = sum(tn.nbytes for tn in td.tensors if "embed_tokens" in tn.name)
        lm = sum(tn.nbytes for tn in td.tensors if "lm_head" in tn.name)
        both = sum(tn.nbytes for tn in td.tensors if "embed_tokens" in tn.name or "lm_head" in tn.name)
        out["total_bytes"] = _v(int(total), SRC_INDEX)
        out["disk_bytes"] = _v(td.disk_bytes, SRC_STAT)
        out["embed_bytes"] = _v(int(embed), SRC_INDEX, note="tensors with 'embed_tokens' in the name")
        out["lm_head_bytes"] = _v(int(lm), SRC_INDEX, note="tensors with 'lm_head' in the name")
        out["bytes_without_lm_head"] = _v(int(total - lm), SRC_INDEX, note="P item of the draft before runtime buffer (draft_post.p_draft_post_mib)")
        out["bytes_without_embed_lm_head"] = _v(int(total - both), SRC_INDEX,
                                                note="D host image with a shared embedding before runtime buffer (draft_post.d_draft_host_mib share_embed)")
        if mtp_ids:
            out["mtp_layers_found"] = _v(len(mtp_ids), SRC_INDEX)
    elif td.index_total_size:
        out["total_bytes"] = _v(int(td.index_total_size), SRC_INDEX, note="model.safetensors.index.json")
    # KV-Zelle des Drafts: eigene Attention-Layer (NEXTN: jede MTP-Schicht ist eine volle Attention-Schicht)
    heads = _int(t.get("num_attention_heads"))
    kvh = _int(t.get("num_key_value_heads"), heads)
    hd = _int(t.get("head_dim"), _int(t.get("hidden_size")) // heads if heads else 0)
    if kind == "nextn" and mtp_ids:
        n_attn = len(mtp_ids)
    else:
        try:
            n_attn = sum(1 for f in layer_families(t) if f == FAM_ATTN)
        except ModelProfileError:
            n_attn = 0
    if kvh and hd and n_attn:
        auto_b = _dtype_bytes(t.get("dtype") or t.get("torch_dtype") or "bfloat16")
        kv: Dict[str, Any] = {"attn_layers": _v(n_attn, SRC_INDEX if (kind == "nextn" and mtp_ids) else SRC_CONFIG),
                              "kv_heads": _v(kvh, SRC_CONFIG), "head_dim": _v(hd, SRC_CONFIG),
                              "cell_bytes_per_attn_layer_token": {
                                  "auto": _v(sum(kv_cell_bytes(kvh, hd, hd, auto_b)), SRC_ESTIMATE),
                                  "fp8_e4m3": _v(sum(kv_cell_bytes(kvh, hd, hd, 1.0)), SRC_ESTIMATE)}}
        slide = sliding_info(t)
        if slide is not None:
            kv["sliding"] = {"window_tokens": _v(slide[0], SRC_CONFIG), "layers": _v(slide[1], SRC_CONFIG)}
        out["kv"] = kv
    dcfg = cfg.get("dflash_config") or t.get("dflash_config")
    if isinstance(dcfg, dict) and dcfg:
        out["dflash"] = {k: _v(v, SRC_CONFIG) for k, v in sorted(dcfg.items())}
    return out


# ---------------------------------------------------------------------------
# Config-only: Gewichtsbytes aus Geometrie und Quantisierungsformat
# ---------------------------------------------------------------------------


class _QuantResolver:
    """Welches Modul in welcher Speicherklasse liegt -- aus ``quantization_config`` (compressed-tensors, modelopt, fp8)."""

    def __init__(self, cfg: Mapping[str, Any], t: Mapping[str, Any]):
        qc = _qc(cfg, t)
        self.method = str(qc.get("quant_method") or "")
        self.groups: List[Dict[str, Any]] = []
        self.ignore: List[str] = list(qc.get("ignore") or []) + list(qc.get("modules_to_not_convert") or []) + list(qc.get("exclude_modules") or [])
        self.block = qc.get("weight_block_size")
        self.qc = qc
        for g in (qc.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            cls = _group_class(g, qc)
            tg = list(g.get("targets") or [])
            self.groups.append({"cls": cls, "bits": _int(w.get("num_bits")), "gs": w.get("group_size"), "strategy": w.get("strategy"),
                                "literal": {x for x in tg if not x.startswith("re:") and x != "Linear"},
                                "regex": [re.compile(x[3:]) for x in tg if x.startswith("re:")],
                                "all_linear": "Linear" in tg, "float": str(w.get("type")) == "float"})
        if self.method == "fp8":
            self.groups.append({"cls": "fp8", "bits": 8, "gs": None, "strategy": "block", "literal": set(), "regex": [],
                                "all_linear": True, "float": True})
        if str(qc.get("quant_algo") or "") == "NVFP4" and not self.groups:
            self.groups.append({"cls": "nvfp4", "bits": 4, "gs": _int(qc.get("group_size"), 16), "strategy": "group", "literal": set(),
                                "regex": [], "all_linear": True, "float": True})
        self._ign_re = [re.compile(x[3:]) for x in self.ignore if x.startswith("re:")]
        self._ign_lit = [x for x in self.ignore if not x.startswith("re:")]

    def _ignored(self, mod: str) -> bool:
        for x in self._ign_lit:
            if x == mod or (x.endswith("*") and mod.startswith(x[:-1])):
                return True
        return any(r.match(mod) for r in self._ign_re)

    def group_of(self, mod: str, strict: bool = False) -> Optional[Dict[str, Any]]:
        """Die Gruppe, die ``mod`` quantisiert.  ``strict``: nur ausdrückliche Ziele (Name oder ``re:``), nie das
        pauschale ``Linear`` -- für Einbettungen, die kein ``nn.Linear`` sind."""
        if self._ignored(mod):
            return None
        for g in self.groups:
            if mod in g["literal"] or any(mod.startswith(x + ".") or mod == x for x in g["literal"]) or any(r.match(mod) for r in g["regex"]):
                return g
        if strict:
            return None
        for g in self.groups:
            if g["all_linear"]:
                return g
        return None


def _linear_bytes(n_in: int, n_out: int, g: Optional[Mapping[str, Any]], dtype_b: float = 2.0, bias: bool = False) -> Tuple[float, str]:
    """Bytes eines Linear-Gewichts einschließlich Skalen, und seine Speicherklasse."""
    n = float(n_in) * float(n_out)
    if g is None:
        return n * dtype_b + (n_out * dtype_b if bias else 0.0), "bf16"
    cls = g["cls"] or "bf16"
    bits = g["bits"] or 16
    payload = n * bits / 8.0
    gs = _int(g.get("gs"))
    if cls == "nvfp4":
        scales = n / (gs or 16) * 1.0
    elif gs:
        scales = n / gs * 2.0
    elif g.get("strategy") == "block":
        scales = n / (128 * 128) * 4.0
    else:
        scales = float(n_out) * 4.0 if cls == "int8" else float(n_out) * 2.0
    return payload + scales + (n_out * dtype_b if bias else 0.0), cls


def kern_ple(t: Mapping[str, Any]) -> int:
    return _int(t.get("ple_conv_kernel_size"), 4)


def estimate_weights_from_config(cfg: Mapping[str, Any], fmt_hint: Optional[str] = None) -> Dict[str, Any]:
    """Gewichtsbytes aus der Config allein (kein Tensorverzeichnis).

    Rechnet je Layer die Linear-Gewichte der Mixer (Attention q/k/v/o mit Ausgabe-Gate, GDN in_proj qkv/z/a/b/out/conv),
    des dichten MLP, der Experten (samt gemeinsamem Experten und Router), die Normen, die Einbettungen, den Sichtturm und
    den MTP-Kopf; jedes Linear in der Speicherklasse, die ``quantization_config`` ihm zuweist (compressed-tensors-Ziele und
    ``ignore``, modelopt ``quantized_layers``-Gruppen, fp8-Blockskalen).  Was die Formel nicht modelliert, steht in ``warnings``:
    Hyper-Connections, PLE und n-gram-Tabellen, Indexer."""
    t = text_config(cfg)
    fams = layer_families(t)
    n = len(fams)
    hidden = _int(t.get("hidden_size"))
    heads = _int(t.get("num_attention_heads"))
    kv_heads = _int(t.get("num_key_value_heads"), heads)
    head_dim = _int(t.get("head_dim"), hidden // heads if heads else 0)
    inter = _int(t.get("intermediate_size"))
    moe_i = _int(t.get("moe_intermediate_size"))
    sh_i = _int(t.get("shared_expert_intermediate_size"))
    n_exp = max((_int(t.get(k)) for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")), default=0)
    vocab = _int(t.get("vocab_size"))
    dtype_b = _dtype_bytes(t.get("dtype") or t.get("torch_dtype") or "bfloat16")
    gate = bool(t.get("attn_output_gate"))
    lk_h, lk_d = _int(t.get("linear_num_key_heads")), _int(t.get("linear_key_head_dim"))
    lv_h, lv_d = _int(t.get("linear_num_value_heads")), _int(t.get("linear_value_head_dim"))
    kern = _int(t.get("linear_conv_kernel_dim"), 4)
    qr = _QuantResolver(cfg, t)
    pre = "model.language_model.layers.%d."
    warnings: List[str] = []
    storage: Dict[str, float] = {}
    # Hyper-Connections (hc_count Ströme): je Modul in_mix down/up (hc_lowrank), block_inject, hc_norm -- zwei je Layer, eins global
    hc = _int(t.get("hc_count"))
    hc_module = 0.0
    if hc:
        wdt = hc * hidden
        hc_module = (2 * _int(t.get("hc_lowrank")) * wdt + hc * wdt + wdt) * dtype_b
    # PLE (je ein Layer: key_proj/value_proj/conv1d/Normen) und die n-gram-Tabelle (ngram_vocab_size_base x ple_embed_dim Elemente,
    # an beiden NF-Checkpoints an der Form der 128 Shards belegt: 20 M x 2560 = 128 x 2500012 x 160)
    ple_dim = _int(t.get("ple_embed_dim"))
    ple_b = 0.0
    ngram_b = 0.0
    if ple_dim and hc:
        ple_b = ((hc * ple_dim) * hidden + ple_dim * hidden + hc * ple_dim * kern_ple(t) + 3 * hc * ple_dim) * dtype_b
        ng_elems = float(_int(t.get("ngram_vocab_size_base"))) * ple_dim
        ng_group = qr.group_of("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", strict=True)
        ngram_b = ng_elems * ((ng_group["bits"] / 8.0) if ng_group and ng_group.get("bits") else dtype_b)

    def lin(mod: str, n_in: int, n_out: int, bias: bool = False) -> float:
        b, c = _linear_bytes(n_in, n_out, qr.group_of(mod), dtype_b, bias)
        storage[c] = storage.get(c, 0.0) + float(n_in) * n_out
        return b

    layer_bytes: List[float] = []
    layer_exp: List[float] = []
    for i, f in enumerate(fams):
        p = pre % i
        b = 2 * hidden * dtype_b                                       # input_layernorm + post_attention_layernorm
        if f == FAM_ATTN:
            b += lin(p + "self_attn.q_proj", hidden, heads * head_dim * (2 if gate else 1))
            b += lin(p + "self_attn.k_proj", hidden, kv_heads * head_dim)
            b += lin(p + "self_attn.v_proj", hidden, kv_heads * head_dim)
            b += lin(p + "self_attn.o_proj", heads * head_dim, hidden)
            b += 2 * head_dim * dtype_b if t.get("qk_norm", True) else 0.0
        elif f == FAM_GDN and lv_h:
            key_dim, val_dim = lk_h * lk_d, lv_h * lv_d
            b += lin(p + "linear_attn.in_proj_qkv", hidden, 2 * key_dim + val_dim)
            b += lin(p + "linear_attn.in_proj_z", hidden, val_dim)
            b += lin(p + "linear_attn.in_proj_a", hidden, lv_h)
            b += lin(p + "linear_attn.in_proj_b", hidden, lv_h)
            b += lin(p + "linear_attn.out_proj", val_dim, hidden)
            b += (2 * key_dim + val_dim) * kern * dtype_b + 2 * lv_h * 4.0 + lv_d * 4.0
        e = 0.0
        if n_exp > 0:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                n_in, n_out = (hidden, moe_i) if nm != "down_proj" else (moe_i, hidden)
                bb, c = _linear_bytes(n_in, n_out, qr.group_of(p + "mlp.experts.0." + nm), dtype_b)
                e += bb * n_exp
                storage[c] = storage.get(c, 0.0) + float(n_in) * n_out * n_exp
            b += hidden * n_exp * dtype_b                                # router
            if sh_i:
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    b += lin(p + "mlp.shared_expert." + nm, hidden if nm != "down_proj" else sh_i, sh_i if nm != "down_proj" else hidden)
                b += hidden * dtype_b                                    # shared_expert_gate
        elif inter:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                b += lin(p + "mlp." + nm, hidden if nm != "down_proj" else inter, inter if nm != "down_proj" else hidden)
        if hc:
            b += 2 * hc_module
        layer_bytes.append(b)
        layer_exp.append(e)
    other_b = hc_module if hc else 0.0
    embed = vocab * hidden * dtype_b
    # compressed-tensors: eine Einbettung, die nicht in ``ignore`` steht, ist mit dem pauschalen ``Linear``-Ziel quantisiert
    # (die Variante ``vocabembed`` unterscheidet sich vom Basis-Checkpoint nur durch das fehlende ``re:.*embed_tokens.*``);
    # fp8/modelopt quantisieren nur ausdrücklich genannte Einbettungen.
    qemb = qr.group_of("model.language_model.embed_tokens", strict=qr.method != "compressed-tensors")
    if qemb is not None and qemb["cls"] not in (None, "bf16"):
        embed = _linear_bytes(hidden, vocab, qemb, dtype_b)[0]
    lm_head = 0.0 if t.get("tie_word_embeddings") else vocab * hidden * dtype_b
    qlm = qr.group_of("lm_head")
    if lm_head and qlm is not None and qlm["cls"] not in (None, "bf16"):
        lm_head = _linear_bytes(hidden, vocab, qlm, dtype_b)[0]
    vis = _vision_params(cfg.get("vision_config") or t.get("vision_config") or {}) * dtype_b
    # MTP-Kopf: eine volle Attention-Schicht + MLP/Experten + Fusion (2h -> h) + Normen
    mtp_n = _int(t.get("mtp_num_hidden_layers"), _int((t.get("mtp") or {}).get("num_hidden_layers")) if isinstance(t.get("mtp"), dict) else 0)
    mtp = 0.0
    for k in range(mtp_n):
        p = "mtp.layers.%d." % k
        mtp += lin(p + "self_attn.q_proj", hidden, heads * head_dim * (2 if gate else 1)) + lin(p + "self_attn.k_proj", hidden, kv_heads * head_dim)
        mtp += lin(p + "self_attn.v_proj", hidden, kv_heads * head_dim) + lin(p + "self_attn.o_proj", heads * head_dim, hidden)
        if n_exp > 0:
            g = qr.group_of(p + "mlp.experts")
            mtp += 3 * n_exp * _linear_bytes(hidden, moe_i, g, dtype_b)[0] + hidden * n_exp * dtype_b
            if sh_i:
                mtp += 3 * lin(p + "mlp.shared_expert.up_proj", hidden, sh_i)
        elif inter:
            mtp += 3 * lin(p + "mlp.up_proj", hidden, inter)
        mtp += lin("mtp.fc", 2 * hidden, hidden) + 6 * hidden * dtype_b
    if _int(t.get("indexer_n_heads")):
        warnings.append("formula does not model the indexer projections of the attention layers (indexer_*): a few MiB per attention layer")
    total = sum(layer_bytes) + sum(layer_exp) + embed + lm_head + vis + mtp + hidden * dtype_b + other_b + ple_b + ngram_b
    cls_sorted = sorted(((k, v) for k, v in storage.items() if k != "bf16"), key=lambda kv_: -kv_[1])
    fmt = cls_sorted[0][0] if cls_sorted else "bf16"
    if len({k for k, _ in cls_sorted}) > 1 and qr.groups and len({g["bits"] for g in qr.groups}) > 1 and fmt.startswith("int"):
        fmt += "-mixed"
    return {"total_bytes": total, "layer_bytes": layer_bytes, "layer_expert_bytes": layer_exp, "embed_bytes": embed,
            "lm_head_bytes": lm_head, "visual_bytes": vis, "mtp_bytes": mtp, "ple_bytes": ple_b, "ngram_bytes": ngram_b,
            "format": fmt,
            "storage": {k: {"params": int(v)} for k, v in sorted(storage.items())}, "warnings": warnings}


# ---------------------------------------------------------------------------
# Stufen- und Karten-Summen (für Abgleich und Editor)
# ---------------------------------------------------------------------------


def stage_weight_bytes(profile: Mapping[str, Any], counts: Sequence[int], *, embed_on_device: bool = True,
                       lm_head_on_device: bool = True, expert_fractions: Optional[Sequence[float]] = None,
                       replicated: Iterable[str] = ()) -> List[float]:
    """Gewichtsbytes je PP-Stufe eines zusammenhängenden Schnitts ``counts`` (Layer je Stufe).

    Stufe 0 trägt die Einbettung, die letzte den ``lm_head``; ``expert_fractions`` (je Stufe) skaliert nur die Expertentensoren
    (residenter Anteil), alles andere zählt voll; ``replicated`` nennt Posten, die jede Stufe trägt (``visual``, ``mtp``)."""
    w = profile["weights"]
    lb = w["layer_bytes"]["v"]
    le = w["layer_expert_bytes"]["v"]
    if sum(counts) != len(lb):
        raise ValueError("counts summiert zu %d, das Modell hat %d Layer" % (sum(counts), len(lb)))
    out: List[float] = []
    start = 0
    for s, c in enumerate(counts):
        frac = 1.0 if expert_fractions is None else float(expert_fractions[s])
        b = sum(lb[start:start + c]) + frac * sum(le[start:start + c])
        if s == 0 and embed_on_device:
            b += w["embed_bytes"]["v"]
        if s == len(counts) - 1 and lm_head_on_device:
            b += w["lm_head_bytes"]["v"]
        for r in replicated:
            b += w[r + "_bytes"]["v"]
        out.append(b)
        start += c
    return out


def expert_buffer_fraction(num_experts: int, fraction: float, scratch_rows: int) -> float:
    """Anteil der Expertenzeilen je Layer, den ein Rang auf dem Gerät hält: ``min(ceil(fraction x E) + Scratch, E) / E``.

    Dieselbe Rechnung wie ``planner/expert_residency.buffer_rows``/``resident_rows`` (Metall NF y8: fraction 0,33 ->
    169 resident + 32 Scratch = 201 von 512).  ``fraction >= 1`` hält alle."""
    E = int(num_experts)
    if E <= 0:
        return 1.0
    if float(fraction) >= 1.0:
        return 1.0
    R = max(1, min(E, int(math.ceil(float(fraction) * E))))
    if R >= E:
        return 1.0
    room = E - R
    return float(min(R + min(int(scratch_rows), room), E)) / E


# ---------------------------------------------------------------------------
# Ableitung der Registry-Zeile (form.ModelProfile) aus dem Schätzprofil
# ---------------------------------------------------------------------------

#: Klassen der ModelProfile-Felder.  ``fact``: folgt aus dem Modell und muss bei den Handzeilen GLEICH sein;
#: ``name``: die Zeilenkennung; ``experts``/``format``: nur der aus dem Modell folgende Teil wird verglichen;
#: ``policy``: Betriebsentscheidung (am Metall bewiesen, Draftwahl, Pfade, Schalter) -- die erzeugte Zeile trägt den konservativen
#: Standard, jede Abweichung der Handzeile steht mit Grund in der Vergleichsliste.
FIELD_CLASS: Dict[str, Tuple[str, str]] = {
    "id": ("name", "Kennung der Zeile, vom Aufrufer gewählt"),
    "arch": ("fact", ""), "replayssm": ("fact", ""), "ple": ("fact", ""), "attn": ("fact", ""),
    "d_layout": ("fact", ""), "page_size": ("fact", ""), "kv_dtype": ("fact", ""), "context_tokens": ("fact", ""),
    "experts": ("experts", "store_dir ist ein Einsatzpfad, residency_p/_d sind Planerausgaben (FRACTION-SOLVE), kein Modellfakt"),
    "formats": ("format", "die Handzeile kennt weitere Formate und je Karten-Klasse die Kernelform (sm8x/sm12x): Betriebswissen"),
    "expect": ("policy", "Erwartungsmengen je Achse: die Handzeile erlaubt mehrere Formen (27B: dflash+mtp), die erzeugte nur die eigene"),
    "draft": ("policy", "Draftwahl ist Betriebsentscheidung (DRAFT-ZUORDNUNG: 27B = DFlash2-Verzeichnis, NF = MTP-Kopf mit 3 Schritten); das Schätzprofil meldet nur, ob ein MTP-Kopf da ist"),
    "p_draft": ("policy", "Draft auf P folgt der Draftwahl (27B 'cold' = Produzent ohne Rechnung)"),
    "chunk": ("policy", "P-Chunk: gemessen je Modell (27B 2048/dynamisch, NF 16384/linear aus Messung)"),
    "end_anchor": ("policy", "Anker am Ende des Prefix: am Metall je Linie gewählt (trim / tail_handoff)"),
    "mamba_anchor": ("policy", "Mamba-Anker: 27B Raster 4096, NF tiefster Zustand -- gemessene Wahl"),
    "mamba_carrier_hold": ("policy", "H81 Carrier-Hold: am Metall bewiesen, generisch aus"),
    "repack_outside_pool": ("policy", "H39 Dense-Repack: am Metall bewiesen, generisch aus"),
    "p_cut": ("policy", "P-Schnitt: 27B gelöst aus Konstanten, NF gepinnt; die erzeugte Zeile lässt den Planer lösen"),
    "x_start_tokens": ("policy", "X-Startwert des Flips: Betriebskonstante"), "x_ceiling_tokens": ("policy", "X-Decke: Betriebskonstante"),
    "idle_layout": ("policy", "Leerlauf-Layout: 27B 'pp', NF keines"),
    "x_split": ("policy", "RC7-X-Aufteilung: 27B-Flip-Linie"), "store_short_tail": ("policy", "xsn437: 27B-Linie"),
    "bigram_anchor_exact": ("policy", "exakte Bigram-Verankerung: am Metall bewiesen (NF, dann 27B)"),
    "warm_min_dwell": ("policy", "H34b: NF-Flip-Linie"), "agent_span": ("policy", "#49 Agent-Span: Agentenlast-Schalter"),
    "standard_form": ("policy", "NF H91 Standardform"),
    "inline_system_in_place": ("policy", "RG/MZ Agentenlast-Präfixschalter: am Agentenlast-Boot bewiesen"),
    "told_probe_tree_key": ("policy", "RG/TK Agentenlast-Präfixschalter"), "told_paced": ("policy", "RG/PX2 Agentenlast-Präfixschalter"),
    "p_twin_defer": ("policy", "RG/TW Agentenlast-Präfixschalter"), "front_span_inflight": ("policy", "RG/#49-Rest"),
    "told_group_fallback": ("policy", "RG/PF: 27B nach Boot z30y14"),
    "vision": ("policy", "Sichtturm: 27B 'transient', NF 'off' -- beide tragen ein vision_config, es ist eine Bootwahl"),
    "records": ("policy", "Schlüssel der Messquellen: 27B mit Zeilen-Linie, NF ohne"),
    "early_read_flags": ("policy", "#1235 Frühlese-Flags: 27B TP-D"), "group_env": ("policy", "Gruppen-Env: NF uneven-DCP"),
    "prefill_transient_checkpoints": ("policy", "Checkpoints, an denen der #114-Prefill-Transient gemessen wurde"),
    "constants": ("policy", "Messkonstanten kommen aus Records (profile_records_data): eine erzeugte Zeile hat keine -- UNCALIBRATED, nie borrowed"),
    "d_residue_census": ("policy", "NF cb1575e94e: Zensus statt Konstante"), "d_expect_from_p_records": ("policy", "D-EXPECT: P-Records"),
    "d_early_start_proven": ("policy", "BOOTZEIT 3: nur nach Metallbeweis je Profil"), "p_mamba_slots_from_argv": ("policy", "H92c"),
    "p_pool_posts_as_booked": ("policy", "PP-POSTEN: 27B"), "d_park_immediate": ("policy", "27B-Park, beide Linien am Metall bewiesen"),
    "front_exact_tokens": ("policy", "X-EXACT, beide Linien am Metall bewiesen"),
    "budget_charges_driver_carve": ("policy", "rc12b: Treiber-Carve ins Budget, Metall"), "driver_carve_min_total_mib": ("policy", "27B: nur 5090 (Tod b1)"),
    "p_row_authority": ("policy", "Fix B #631: 27B nach Agentenlast-Beweis"), "budget_rest_from_records": ("policy", "VRAM-Grundgesetz: 27B"),
    "torch_cache_cap": ("policy", "PDFLIP-ALLOC-OVERHANG: aus bis Messzelle"), "d_hostgap_levers": ("policy", "HG: DFlash-D (27B)"),
    "d_hostgap_base": ("policy", "HG-Basis: DFlash-D (27B)"), "d_release_fixes": ("policy", "Release-Draft-Fixes: DFlash (27B)"),
    "front_dc_off_path": ("policy", "LS6: 27B"), "front_quiesce_fast": ("policy", "LS6: 27B"), "d_dcp_lse_merge": ("policy", "LS6: 27B a2a, Nadel-Gate"),
    "p_host_overlap": ("policy", "LS6: 27B"), "d_verify_vocab_argmax": ("policy", "LS12: DFlash (27B)"), "vram_peak_fast_read": ("policy", "LS12: 27B"),
    "hicache_load_async_index": ("policy", "LS12: 27B"), "hicache_drain_agree_every": ("policy", "LS12: 27B"),
    "admission_wedge_recovery_s": ("policy", "LS12: 27B"), "front_ctl_kick": ("policy", "LS12: 27B"),
    "p_prefill_graph": ("policy", "Prefill-Graph: 27B-Formate int8/nvfp4/fp8"), "p_prefill_graph_formats": ("policy", "Prefill-Graph: 27B"),
    "d_token_placement": ("policy", "Zeile 24b: 27B INT8 Bandbreite"), "d_token_placement_formats": ("policy", "Zeile 24b: 27B INT8"),
    "group_switch_defaults": ("policy", "Leistungsschalter NF (a): Gruppenschalter, wie die Linie lief"),
    "d_kv_token_cut": ("policy", "#239: NF 'owned'"), "handback_claim_n": ("policy", "HANDBACK N-1: 27B"),
    # INT8-Fixsatz (cand2-Merge 1005): Q-711 SHORT-KEPT-BOUND kam als ModelProfile-Feld in ``form.ModelProfile``
    "short_kept_max_wait_s": ("policy", "Q-711 SHORT-KEPT-BOUND: 27B-Zeile 30 s, NF 0 (aus), FLLIPER_PDFLIP_SHORT_KEPT_MAX_WAIT_S"),
}

#: Standard der erzeugten Zeile für Pflichtfelder der Klasse ``policy`` (jedes andere Feld nimmt den Dataclass-Standard, Schalter aus)
_GENERIC_POLICY: Dict[str, Any] = {
    "p_cut": "solved by the planner (solve_pp_cut): no pinned ratio",
    "x_start_tokens": 4096, "x_ceiling_tokens": 12288,
    "idle_layout": "", "end_anchor": "none", "mamba_anchor": "none",
    "mamba_carrier_hold": False, "repack_outside_pool": False,
    "vision": "off", "early_read_flags": False, "prefill_transient_checkpoints": (),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def derive_registry_fields(profile: Mapping[str, Any], *, row_id: Optional[str] = None,
                           checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Die Felder einer ``form.ModelProfile``-Zeile aus dem Schätzprofil -- als reines Dict (kein ``form``-Import).

    * Modellfakten (``fact``): aus ``config`` und Tensornamen.  ``attn``/``page_size``/``kv_dtype``/``d_layout`` folgen der
      Attention-Art: Indexer-Attention (``indexer_*``) = ``qsa`` / Seite 64 / ``fp8_e4m3`` / ``qsa_forma`` (die Form, die QSA
      verlangt), sonst ``full`` / 1 / ``auto`` / ``paged_dcp``.
    * Experten: dicht = ``none``, MoE = ``offload`` (mit ``fraction >= 1`` hält es alles resident; der Planer löst den Anteil).
    * Betriebsentscheidungen (Draft, Schalter, Anker, Ketten) nehmen den konservativen Standard: kein Draft, alle Schalter aus.
    """
    arch = profile["arch"]
    moe = arch["family"]["v"] == "moe"
    qsa = arch["attention"]["v"] == "qsa"
    fmt = profile["format"]["v"]
    path = checkpoint or profile.get("path", "")
    rid = row_id or _slug(os.path.basename(str(path).rstrip("/")) or "generic")
    d_layout = "qsa_forma" if qsa else "paged_dcp"
    exp = ("resident", "offload") if moe else ("none",)
    out: Dict[str, Any] = {
        "id": rid,
        "expect": {"arch": (arch["family"]["v"],), "experts": exp, "draft": ("none",), "p_draft": ("none",), "kv": (d_layout,)},
        "arch": arch["family"]["v"],
        "experts": {"store": "offload" if moe else "none", "swap": "platztausch" if moe else ""},
        "draft": {"kind": "none"},
        "p_draft": "none",
        "replayssm": bool(arch["hybrid"]["v"]),
        "ple": bool(arch["ple"]["v"]),
        "attn": "qsa" if qsa else "full",
        "d_layout": d_layout,
        "page_size": 64 if qsa else 1,
        "kv_dtype": "fp8_e4m3" if qsa else "auto",
        "chunk": {"grid": 0, "policy": "fixed", "model": "generic: measured per model at the first boot", "tokens": 2048},
        "formats": {fmt: {"name": fmt, "checkpoint": str(path)}},
        "context_tokens": int(profile["context"]["max_position_embeddings"]["v"]),
        "records": {"fields": ("checkpoint", "form", "power_limit")},
        "group_env": {},
        "constants": {},
    }
    out.update(_GENERIC_POLICY)
    return out


def to_model_profile(fields: Mapping[str, Any]):
    """``form.ModelProfile`` aus :func:`derive_registry_fields` (lädt ``flliper.srt.pdflip.form`` erst hier)."""
    import dataclasses as _dc

    from flliper.srt.pdflip import form as F

    kw: Dict[str, Any] = dict(fields)
    kw["experts"] = F.Experts(**kw["experts"])
    kw["draft"] = F.Draft(**kw["draft"])
    kw["chunk"] = F.Chunk(**kw["chunk"])
    kw["formats"] = {k: F.WeightFormat(**v) for k, v in kw["formats"].items()}
    kw["records"] = F.RecordKey(**kw["records"])
    # Pflichtfelder ohne Modellfakt und ohne Eintrag: die Schalter der Handzeilen, hier alle aus
    for f in _dc.fields(F.ModelProfile):
        if f.name not in kw and f.default is _dc.MISSING and f.default_factory is _dc.MISSING:
            kw[f.name] = False
    return F.ModelProfile(**kw)


def _plain(x: Any) -> Any:
    import dataclasses as _dc

    if _dc.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in _dc.fields(x)}
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def compare_with_registry(derived, hand) -> List[Dict[str, Any]]:
    """Feld für Feld: die erzeugte Zeile gegen die Handzeile.  ``[{field, class, equal, derived, hand, reason}]``.

    Jedes Feld von ``form.ModelProfile`` steht in :data:`FIELD_CLASS` (ein neues Feld ohne Klasse wirft ``KeyError`` -- es
    muss eingeordnet werden).  ``fact`` verlangt Gleichheit; ``experts`` vergleicht ``store`` und ``swap``; ``format`` die
    Zeile des geschätzten Checkpoints in ``formats`` (Name und Pfad); ``policy`` listet die Abweichung mit ihrem Grund."""
    import dataclasses as _dc

    rows: List[Dict[str, Any]] = []
    for f in _dc.fields(type(hand)):
        cls, reason = FIELD_CLASS[f.name]
        d, h = getattr(derived, f.name), getattr(hand, f.name)
        extra = False
        if cls == "experts":
            equal = (d.store, d.swap) == (h.store, h.swap)
            d_show, h_show = _plain(d), _plain(h)
            extra = equal and d_show != h_show
        elif cls == "format":
            (fk, fv), = d.items()
            hv = h.get(fk)
            equal = hv is not None and hv.name == fv.name and hv.checkpoint == fv.checkpoint
            d_show, h_show = _plain(fv), (_plain(hv) if hv else None)
            extra = bool(equal and d_show != h_show)
        else:
            d_show, h_show = _plain(d), _plain(h)
            equal = d_show == h_show
        rows.append({"field": f.name, "class": cls, "equal": bool(equal), "derived": d_show, "hand": h_show,
                     "reason": reason if (not equal or extra) else "", "partial": bool(extra)})
    return rows


def register_profile(row) -> None:
    """Trägt eine erzeugte Zeile in ``form.PROFILES``, ``PROFILE_EXPECT`` und ``PROFILE_SWITCH_DEFAULTS`` ein (Laufzeit, kein Schreiben
    auf Platte).  Eine vorhandene Zeile gleicher Kennung wird nie überschrieben."""
    from flliper.srt.pdflip import form as F

    if row.id in F.PROFILES:
        raise ModelProfileError("registry row %r already exists: never overwritten" % row.id)
    F.PROFILES[row.id] = row
    F.PROFILE_EXPECT[row.id] = {a: tuple(v) for a, v in row.expect.items()}
    F.PROFILE_SWITCH_DEFAULTS[row.id] = row.switch_defaults()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 model_profile.py PFAD [--draft PFAD] [--kv-dtype fp8_e4m3] [--ssm-dtype bfloat16] [--registry ID]`` -> JSON."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Model profile flliper.model/1 from config.json and headers (no weights)")
    ap.add_argument("path", help="model directory or .gguf file")
    ap.add_argument("--draft", help="separate draft directory")
    ap.add_argument("--kv-dtype", help="auto | fp8_e4m3 (selects the value in kv.cell_bytes_per_attn_layer_token)")
    ap.add_argument("--ssm-dtype", help="float32 | bfloat16 (Mamba state; default: the config)")
    ap.add_argument("--registry", metavar="ID", help="additionally the derived registry fields under this identifier")
    a = ap.parse_args(argv)
    try:
        prof = estimate(a.path, draft_path=a.draft, kv_dtype=a.kv_dtype, mamba_ssm_dtype=a.ssm_dtype)
    except ModelProfileError as exc:
        print("FEHLER: %s" % exc, file=sys.stderr)
        return 2
    if a.registry:
        prof["registry_fields"] = derive_registry_fields(prof, row_id=a.registry)
    print(json.dumps(prof, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
