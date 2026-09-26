"""H92: the Next-Flash (NF) side of ``--p-chunk-policy`` (weg2/p_chunk_policy.py).

User order 25.09. ~21:40Z (via the 27B seat): dynamic P chunk width for NF too,
through the ONE interface the 27B agent DC built (``p_chunk_policy``: the flow
shop rule, ``ChunkLimits``, ``PolicySpec``, ``ChunkPlanner``, the env and the
log lines). This module adds only what is NF-specific and keeps the shared
module untouched:

1. THE NF PROFILE -- per-PP-stage ``t_s(M) = a_s + b_s*M (+ attention)`` from
   NF measurements, keyed by the checkpoint (``model_key``), stored as data in
   ``p_stage_model_data/*.pchunk.json``. The lookup is by model key ONLY: an
   NF boot never prices on 27B numbers (``builtin-int8`` is not reachable
   from here) and a checkpoint without its own profile is refused.
2. THE NF HARD LIMITS (``nf_limits``), refused by name instead of clipped:
   * ceiling = the P chunk the P card was priced at: group P's
     ``--chunked-prefill-size`` (SGLANG_WEG2_P_CHUNKED_PREFILL_TOKENS). The P
     activation transient (#114/H41) and the H41c card (row price 71.4 MiB,
     chunk growth ~321 MiB per chunk, saturating) -- and with them FR_P --
     were solved for THAT width; a wider plan would run on VRAM nobody priced.
     A plan may only choose widths <= the ceiling (a smaller chunk has a
     smaller transient, and the chunk-growth term saturates by chunk index,
     so more, smaller chunks never exceed the priced headroom).
   * the ceiling must also lie on the measured transient support (W131:
     never extrapolated).
   * the 4096 RASTER: every non-final chunk END is an absolute multiple of
     4096 (Mamba anchors / publish / H63 tail fold). In the shared module this
     is ``ChunkLimits.page`` -- NOT ``grid``: ``grid`` forbids a chunk to
     CROSS a multiple (it caps every chunk at 4096), while NF runs 16384-token
     chunks whose boundaries sit on the raster.
   * fixed baseline = the ceiling (today's width; a narrowed ceiling for a
     metal probe takes it along), the ladder floor >= raster.
3. THE FR_P COUPLING (interface duty (a), "one residency for both"): ``a_s``
   carries the expert stream, which scales with the non-resident share. A boot
   whose FR_P differs from the profile's ``fr_p_ref`` gets
   ``a_s + fetch_ref_s * ((1-FR)/(1-FR_ref) - 1)`` and a HOCHRECHNUNG mark.
4. THE FORWARD BUDGET OVER SEVERAL REQUESTS (P stau up to 6, H91): NF plans
   the TOKEN STREAM (head rest + every waiting request's rest), not the head
   alone -- a tail ramp is only worth it where the stream drains, never at the
   end of one request while others wait. ``SGLANG_P_CHUNK_BUDGET=stream``
   selects it on the P ranks; unset = the 27B per-request budget.

Pure (stdlib + the shared module). ``fixed`` never reaches this module.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import p_chunk_policy as _pcp

#: NF's chunk raster (Mamba anchors / publish window / H63 tail fold).
NF_RASTER_TOKENS = 4096
#: Where the profiles live (shared with the 27B's stage-model data).
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p_stage_model_data")
PROFILE_GLOB = "*.pchunk.json"
PROFILE_SCHEMA = "nf-pchunk-1"
#: Budget mode on the P ranks: 'stream' (NF) or unset/'request' (27B).
BUDGET_ENV = "SGLANG_P_CHUNK_BUDGET"
BUDGET_STREAM = "stream"
BUDGET_REQUEST = "request"
#: The dry-run rungs the NF launcher prints (the H92 Messplan): 97k single,
#: burst 8 x 4.2k, stau6 6 x 19806 -- as token streams.
NF_DRY_RUN_TOKENS = (97841, 33918, 118836)
#: How many waiting requests the stream end counts (P stau cap is 6).
STREAM_MAX_REQUESTS = 64


class NfChunkRefused(ValueError):
    """A dynamic NF chunk policy that cannot be armed on this boot (named)."""


@dataclasses.dataclass(frozen=True)
class NfProfile:
    path: str
    model_key: str
    profile: str
    ceiling_tokens: int
    fr_p_ref: Tuple[float, ...]
    stages: Tuple[_pcp.StageModel, ...]
    fetch_ref_ms: Tuple[float, ...]
    source: str


def load_profile(path: str) -> NfProfile:
    try:
        with open(path) as fh:
            d = json.load(fh)
        if d.get("schema") != PROFILE_SCHEMA:
            raise NfChunkRefused(f"{path}: schema {d.get('schema')!r}, expected {PROFILE_SCHEMA!r}")
        stages = tuple(_pcp.StageModel.from_json(s) for s in d["stages"])
        fetch = tuple(float(s.get("fetch_ref_ms", 0.0)) for s in d["stages"])
        fr = tuple(float(x) for x in d["fr_p_ref"])
        if not stages or len(fr) != len(stages):
            raise NfChunkRefused(f"{path}: {len(stages)} stages but {len(fr)} fr_p_ref entries")
        return NfProfile(path, str(d["model_key"]), str(d.get("profile", "")), int(d["ceiling_tokens"]),
                         fr, stages, fetch, str(d.get("source", os.path.basename(path))))
    except NfChunkRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, _pcp.ChunkPolicyError) as exc:
        raise NfChunkRefused(f"NF chunk profile {path} unreadable: {type(exc).__name__}: {exc}") from None


def find_profile(model_key: str, directory: str = PROFILE_DIR) -> Optional[str]:
    """The ONE profile whose ``model_key`` is this checkpoint's, else None.
    Two profiles for one key are refused (no silent pick)."""
    hits = []
    for p in sorted(glob.glob(os.path.join(directory, PROFILE_GLOB))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("schema") == PROFILE_SCHEMA and d.get("model_key") == model_key:
            hits.append(p)
    if len(hits) > 1:
        raise NfChunkRefused(f"{len(hits)} NF chunk profiles claim model {model_key!r}: {hits}")
    return hits[0] if hits else None


def resolve_profile(src: str, model_key: str, directory: str = PROFILE_DIR) -> NfProfile:
    """``--p-chunk-model`` on the NF line: 'auto' (the profile of this
    checkpoint) or a path. Either way the profile's model key must be this
    checkpoint's -- never another model's numbers."""
    src = str(src or "auto").strip()
    if src == "auto":
        path = find_profile(model_key, directory)
        if path is None:
            raise NfChunkRefused(
                f"no NF chunk profile for model {model_key!r} in {directory} -- measure one "
                f"(FWD-TIMING-PREFILL fit) or keep --p-chunk-policy fixed")
    elif src.startswith("builtin") or src.startswith("fit:"):
        raise NfChunkRefused(f"--p-chunk-model {src}: the 27B sources are not NF profiles")
    else:
        path = src
    prof = load_profile(path)
    if prof.model_key != model_key:
        raise NfChunkRefused(
            f"NF chunk profile {os.path.basename(path)} is for {prof.model_key!r}, this boot runs "
            f"{model_key!r} -- a profile is a property of one checkpoint")
    return prof


def fr_adjusted_stages(prof: NfProfile, fr_p: Optional[Sequence[float]]
                       ) -> Tuple[Tuple[_pcp.StageModel, ...], str]:
    """``a_s`` for this boot's FR_P (the expert stream scales with 1-FR)."""
    if fr_p is None or len(fr_p) != len(prof.stages):
        return prof.stages, f"FR_P unknown -> profile a_s (fr_p_ref {list(prof.fr_p_ref)})"
    if all(abs(float(a) - float(b)) < 5e-4 for a, b in zip(fr_p, prof.fr_p_ref)):
        return prof.stages, f"FR_P {list(prof.fr_p_ref)} = profile (measured)"
    out = []
    for st, fr, ref, fetch in zip(prof.stages, fr_p, prof.fr_p_ref, prof.fetch_ref_ms):
        if not (0.0 <= float(fr) <= 1.0):
            raise NfChunkRefused(f"FR_P {list(fr_p)} outside [0, 1]")
        if abs(float(fr) - float(ref)) < 5e-4:
            out.append(st)
            continue
        scale = (1.0 - float(fr)) / max(1e-9, 1.0 - float(ref))
        a0 = st.points[0][1]
        b = st.points[1][1] - a0
        out.append(_pcp.StageModel.linear(max(0.0, a0 + fetch * (scale - 1.0)), b,
                                          attn_ms_per_tok_1k=st.attn_ms_per_tok_1k,
                                          eager_floor_ms=st.eager_floor_ms, name=st.name))
    return tuple(out), (f"FR_P {[round(float(x), 6) for x in fr_p]} != profile {list(prof.fr_p_ref)}: "
                        f"a_s scaled by the expert stream (HOCHRECHNUNG)")


def nf_limits(*, ceiling: int, max_tokens: int = 0, min_tokens: int = 0, fixed_tokens: int = 0,
              raster: int = NF_RASTER_TOKENS, transient_support_max: int = 0,
              min_gain: float = _pcp.DEFAULT_MIN_GAIN, dynamic_min_tokens: int = -1,
              stages: int = 3) -> _pcp.ChunkLimits:
    """The NF hard limits as a ``ChunkLimits`` (0 = the NF default).

    Refused by name: a ceiling above the priced P chunk or off the measured
    transient support, a raster that is not NF's, widths off the raster."""
    ceiling = int(ceiling)
    cap = int(max_tokens or ceiling)
    # the baseline is today's width -- the P chunk -- unless the ceiling was
    # narrowed for a metal probe (min = max = fixed forces one width)
    fixed = int(fixed_tokens or min(cap, ceiling))
    low = int(min_tokens or raster)
    if raster != NF_RASTER_TOKENS:
        raise NfChunkRefused(f"raster {raster}: the NF chunk raster is {NF_RASTER_TOKENS}")
    if cap > ceiling:
        raise NfChunkRefused(
            f"--p-chunk-max {cap} above the priced P chunk {ceiling} (SGLANG_WEG2_P_CHUNKED_PREFILL_TOKENS): "
            f"the P transient, the H41c card and FR_P were solved for {ceiling}; raise the P chunk itself "
            f"so the planner re-prices them")
    if transient_support_max and cap > int(transient_support_max):
        raise NfChunkRefused(f"--p-chunk-max {cap} above the measured P transient support "
                             f"{transient_support_max} (W131: never extrapolated)")
    if fixed > cap:
        raise NfChunkRefused(f"--p-chunk-fixed {fixed} above the plan's ceiling {cap}")
    if low < raster:
        raise NfChunkRefused(f"--p-chunk-min {low} below the NF raster {raster}")
    dmin = fixed if dynamic_min_tokens < 0 else int(dynamic_min_tokens)
    try:
        return _pcp.ChunkLimits(max_tokens=cap, min_tokens=low, fixed_tokens=fixed, page=raster, grid=0,
                                graph_buckets=(), eager=True, max_inflight=int(stages),
                                min_gain=float(min_gain), dynamic_min_tokens=dmin)
    except _pcp.ChunkPolicyError as exc:
        raise NfChunkRefused(f"--p-chunk-policy dynamic (NF): {exc}") from None


def budget_is_stream(env: Mapping[str, str]) -> bool:
    return str(env.get(BUDGET_ENV, "") or "").strip().lower() == BUDGET_STREAM


def req_rest_tokens(req) -> int:
    """Tokens a (waiting or chunked) request still has to prefill."""
    fill = getattr(req, "full_untruncated_fill_ids", None)
    if fill is not None and len(fill):
        end = len(fill)
    else:
        end = len(getattr(req, "origin_input_ids", ()) or ()) + len(getattr(req, "output_ids", ()) or ())
    prefix = getattr(req, "prefix_indices", None)
    return max(0, end - (0 if prefix is None else len(prefix)))


def stream_end(pos: int, head_end: int, others: Iterable) -> int:
    """The END of the token stream a stream budget plans: the head's own end
    plus every other waiting request's rest (at most STREAM_MAX_REQUESTS)."""
    end = int(head_end)
    for i, r in enumerate(others):
        if i >= STREAM_MAX_REQUESTS:
            break
        end += req_rest_tokens(r)
    return max(end, int(pos))


def dry_run_lines(spec: _pcp.PolicySpec, tokens: Sequence[int] = NF_DRY_RUN_TOKENS) -> List[str]:
    out = []
    for n in tokens:
        res = _pcp.plan_detail(n, len(spec.stages), spec.stages, spec.limits)
        out.append("WEG2 " + _pcp.plan_line(res, key=f"dry-run-{n}", start=0, end=n)
                   + " (HOCHRECHNUNG on the NF profile as one token stream, not a measurement)")
    return out


def parse_fractions(value: str) -> Optional[Tuple[float, ...]]:
    try:
        vals = tuple(float(x) for x in str(value or "").split(",") if x.strip())
    except ValueError:
        return None
    return vals or None


__all__ = [
    "NF_RASTER_TOKENS", "PROFILE_DIR", "BUDGET_ENV", "BUDGET_STREAM", "BUDGET_REQUEST", "NF_DRY_RUN_TOKENS",
    "NfChunkRefused", "NfProfile", "load_profile", "find_profile", "resolve_profile", "fr_adjusted_stages",
    "nf_limits", "budget_is_stream", "req_rest_tokens", "stream_end", "dry_run_lines", "parse_fractions",
]
