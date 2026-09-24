"""H25 (Nutzer-Order 24.09. 08:25Z): the planner half of "Draft auf P streichen".

Woertlich: *"Draft auf P streichen. dadurch koennen auch mehr experten im P
layout gehalten werden. die draft layerbytes aus D muessen dann waehrend P
laeuft in den systemram offgeloaded werden."*  Praezisierung 08:30Z: *"das ist
ein systemram gegen vram tausch. der draft liegt im systemram, dafuer aber mehr
experten im vram."*

Three pure pieces, each a seam the launcher reads once:

* :func:`p_draft_post` -- WHAT the draft occupied on P's draft card (the last
  P stage): the draft checkpoint's tensors P holds (all but ``lm_head``, which
  the last stage shares with the target, measured x137 PP2
  ``WEG2-XCHG-COVER tag=weights_draft planned_mib=2735.2`` = MTP rest 1522.8
  + own embed 1212.5), the runner's buffers and the draft-KV producer's
  transient.  Priced from the checkpoint HEADERS, never from a remembered
  boot number, so a different draft prices itself.
* :func:`expert_rows_for` -- the freed MiB as expert rows of that stage (the
  stage holds ALL experts of its layers, so one row costs ``layers x row``),
  capped by the H5 buffer rule ``(E-2)/E``.  No reserve is held back (Memory
  keine-korridor-reserve-nie) and nothing is pinned by hand: the fraction the
  arm gave is the input, the raised one is the output.
* :func:`d_draft_host_mib` -- what D's parked draft holds in pinned host RAM
  (the MTP tensors D does NOT share with its target: with
  ``SGLANG_WEG2_DRAFT_SHARE_EMBED=1`` (H1b) that is the checkpoint minus
  ``embed_tokens`` and ``lm_head``, measured x137 D TP0 ``planned_mib=1522.7``)
  plus the runner's buffers -- the ledger post ``d_draft_host``.

and :func:`draft_swap_line`, the one line that states the host balance of the
swap (store before/after from the Platztausch map, the parked draft, the net).
"""

from __future__ import annotations

import glob
import json
import math
import os
import shlex
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import msgspec

MIB = float(1 << 20)

#: x137 (1e49837c84): ``WEG2-XCHG-COVER tag=weights_draft ... buffers_mib=64.1``
#: on P PP2 AND on D TP0 -- the draft runner's registered buffers (rope cache,
#: norms' scratch) beside its parameters.  Same value on both groups.
DRAFT_RUNNER_BUFFER_MIB = 64.1

#: fnFL2x83 (H1-Bericht 23.09.): the draft-KV producer's prefill transient on
#: P's draft stage, 622.8 MiB beyond the resident head.  Without a draft on P
#: nobody allocates it.
P_DRAFT_PRODUCER_TRANSIENT_MIB = 622.8

#: Tensors the LAST P stage shares with its target instead of holding twice.
P_SHARED_WITH_TARGET = ("lm_head",)
#: Tensors D's NEXTN draft shares with its target under H1b (the shared
#: embedding and head are placeholders, ``mtp_vocab_share.py``).
D_SHARED_WITH_TARGET = ("embed_tokens", "lm_head")


class DraftPost(msgspec.Struct, frozen=True):
    """One stage's draft post turned into expert rows."""

    stage: int
    card: str
    freed_mib: float
    weights_mib: float
    transient_mib: float
    layers: int
    row_mib: float
    rows: int
    frac_before: float
    frac_after: float

    def line(self) -> str:
        return (
            f"PP-CUT draft post (H25) card={self.card} stage={self.stage} "
            f"freed_mib={self.freed_mib:.0f} (weights_draft {self.weights_mib:.0f} "
            f"= checkpoint minus {list(P_SHARED_WITH_TARGET)} + buffers "
            f"{DRAFT_RUNNER_BUFFER_MIB}; producer transient {self.transient_mib:.0f}) "
            f"-> experts +{self.rows} rows ({self.rows} per layer x {self.layers} "
            f"layers x {self.row_mib:.3f} MiB) FR_P[{self.stage}] "
            f"{self.frac_before:.4f} -> {self.frac_after:.6f}; no reserve, no pin: "
            f"the arm's fraction is the input, this one ships"
        )


def checkpoint_tensor_mib(path: str, *, exclude: Sequence[str] = ()) -> Optional[float]:
    """MiB of every tensor in ``path``'s ``*.safetensors`` whose name contains
    none of ``exclude``, read from the HEADERS (no tensor byte is read).

    ``None`` when the directory holds no safetensors file -- absent, never 0.
    """
    files = sorted(glob.glob(os.path.join(str(path), "*.safetensors")))
    if not files:
        return None
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if any(x in name for x in exclude):
                continue
            lo, hi = meta["data_offsets"]
            total += int(hi) - int(lo)
    return total / MIB


def p_draft_post_mib(draft_path: str) -> Tuple[Optional[float], float]:
    """``(weights_mib, transient_mib)`` the draft held on P's draft stage."""
    ckpt = checkpoint_tensor_mib(draft_path, exclude=P_SHARED_WITH_TARGET)
    if ckpt is None:
        return None, P_DRAFT_PRODUCER_TRANSIENT_MIB
    return ckpt + DRAFT_RUNNER_BUFFER_MIB, P_DRAFT_PRODUCER_TRANSIENT_MIB


def d_draft_host_mib(draft_path: str, *, share_embed: bool) -> Optional[float]:
    """The pinned host bytes D's parked draft needs (ledger post ``d_draft_host``)."""
    exclude = D_SHARED_WITH_TARGET if share_embed else ("lm_head",)
    ckpt = checkpoint_tensor_mib(draft_path, exclude=exclude)
    if ckpt is None:
        return None
    return ckpt + DRAFT_RUNNER_BUFFER_MIB


def resident_count(num_experts: int, fraction: float) -> int:
    """The rank's own count, ``max(1, ceil(f*E))`` (expert_map #160)."""
    n = int(num_experts)
    return max(1, int(math.ceil(float(fraction) * n)))


def fraction_for_count(num_experts: int, count: int) -> float:
    """A fraction whose rank count is EXACTLY ``count``, robust to 6-digit
    printing: ``(count - 0.25) / E`` -- ``ceil`` of it is ``count`` for any
    rounding error below a quarter row."""
    return (int(count) - 0.25) / float(num_experts)


def expert_rows_for(freed_mib: float, layers: int, row_mib: float,
                    num_experts: int, frac: float) -> Tuple[int, float]:
    """``(rows, new_fraction)``: whole expert rows per layer ``freed_mib`` buys
    on a stage of ``layers`` layers, capped by the H5 buffer rule (the largest
    offload fraction that still builds a Platztausch buffer is ``(E-2)/E``)."""
    if freed_mib <= 0 or layers <= 0 or row_mib <= 0:
        return 0, float(frac)
    before = resident_count(num_experts, frac)
    cap = int(num_experts) - 2
    rows = int(math.floor(float(freed_mib) / (int(layers) * float(row_mib))))
    after = min(cap, before + max(0, rows))
    if after <= before:
        return 0, float(frac)
    return after - before, fraction_for_count(num_experts, after)


def replace_vector_flag(extra: str, flag: str, vector: Sequence[float]) -> str:
    """``extra`` with EVERY ``flag`` value replaced by ``vector`` (argparse
    takes the last one; a stale earlier copy must not survive either)."""
    toks = shlex.split(str(extra or ""))
    val = ",".join(_fmt(v) for v in vector)
    out: List[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == flag and i + 1 < len(toks):
            out.extend([t, val])
            i += 2
            continue
        if t.startswith(flag + "="):
            out.append(f"{flag}={val}")
        else:
            out.append(t)
        i += 1
    return shlex.join(out)


def replace_env_vector(spec: str, key: str, vector: Sequence[float]) -> str:
    """``'K=V;K=V'`` with ``key``'s value replaced (only where it is present)."""
    items = [x for x in str(spec or "").split(";") if x.strip()]
    val = ",".join(_fmt(v) for v in vector)
    out = []
    for item in items:
        k, _, v = item.partition("=")
        out.append(f"{k}={val}" if k.strip() == key else item)
    return ";".join(out)


def _fmt(v: float) -> str:
    return f"{float(v):.6f}".rstrip("0").rstrip(".") if float(v) != 0 else "0"


def draft_swap_line(*, slots_before: int, slots_after: int, layers: int,
                    row_mib: float, d_draft_host: Optional[float]) -> str:
    """``WEG2-HOST DRAFT-SWAP``: the host balance of the swap, in ONE line.

    The store is the Platztausch map's ``slots`` rows per layer, which is
    ``total - |common of all stages|``; its lower bound is D's own residency
    (D holds every layer with the SAME ids, so its cold rows must live in host
    RAM during the D phase no matter what P holds).  Negative ``net`` = lighter.
    """
    before = int(slots_before) * int(layers) * float(row_mib)
    after = int(slots_after) * int(layers) * float(row_mib)
    host = float(d_draft_host or 0.0)
    return (
        f"WEG2-HOST DRAFT-SWAP store_before_mib={before:.0f} store_after_mib={after:.0f} "
        f"d_draft_host_mib={host:.0f} net_host_mib={after - before + host:+.0f} "
        f"(store = slots x {int(layers)} layers x {float(row_mib):.3f} MiB, slots "
        f"{int(slots_before)} -> {int(slots_after)}; negative = lighter)"
    )


def stage_card_label(cards: Sequence[object], stage: int) -> str:
    try:
        return f"nvml{int(cards[stage].nvml_index)}"
    except (IndexError, AttributeError, TypeError, ValueError):
        return f"ordinal{int(stage)}"


def raise_for_draft_post(*, fracs: Sequence[float], stage_layers: Sequence[int],
                         row_mib: float, num_experts: int, draft_path: str,
                         cards: Sequence[object]) -> Tuple[List[float], Optional[DraftPost], str]:
    """``(new_fracs, post, why)``: the last P stage's fraction raised by the
    draft post.  ``post`` is None (and ``why`` names the reason) when nothing
    can be priced -- then the fractions come back unchanged."""
    fr = [float(x) for x in fracs]
    if not fr or len(stage_layers) != len(fr):
        return fr, None, f"stages {list(stage_layers)} do not match fractions {fr}"
    stage = len(fr) - 1
    weights, transient = p_draft_post_mib(draft_path) if draft_path else (None, 0.0)
    if weights is None:
        return fr, None, (f"draft checkpoint {draft_path!r} holds no *.safetensors -- "
                          "the draft post cannot be priced, fractions unchanged")
    freed = weights + transient
    rows, new = expert_rows_for(freed, int(stage_layers[stage]), row_mib,
                                num_experts, fr[stage])
    post = DraftPost(stage=stage, card=stage_card_label(cards, stage),
                     freed_mib=freed, weights_mib=weights, transient_mib=transient,
                     layers=int(stage_layers[stage]), row_mib=float(row_mib),
                     rows=rows, frac_before=fr[stage], frac_after=new)
    out = list(fr)
    out[stage] = new
    return out, post, ""


def summary(post: Optional[DraftPost]) -> Dict[str, float]:
    if post is None:
        return {}
    return {"stage": post.stage, "before": post.frac_before, "after": post.frac_after,
            "freed_mib": post.freed_mib, "rows": post.rows}
