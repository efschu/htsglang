# SPDX-License-Identifier: Apache-2.0
"""WEG2-D-AWAKE-REST: what an awake D rank holds on its card BEYOND its booked form.

THE DEFECT (NF Dauerlauf rc12 / rc12b / rc12c, 26./27.09.). D TP0 on the 5090
died three times at the same size -- per-process NVML 29.14 / 29.19 / 29.24
GiB -- while its budget went 29624 -> 29144 -> 28288 MiB. The budget never
reached an allocation: the runtime sizes only the KV rest from it
(``KV pool sizing`` 437248 / 401664 / 338176 tokens) and then clamps to the
262144-token context (``max_total_num_tokens=262144`` in all three), so every
MiB the planner took away was a MiB of KV nobody would have allocated. The
record that was meant to price the excess, ``D_OVERSHOOT_MIB`` = peak - BUDGET,
therefore grew by exactly what the budget lost (428 -> 826 -> 1798): a ratchet
without a fixpoint. The #145 solve said "Rest nach KV 1202 -> PASST" (its fixed
post from fnFL2x151/x158 is stale) while its card path said "Decode frei -1096
-> KORRIDOR GERISSEN (Befund, kein Stopper)": two sides of one seam, disagreeing.

THE MEASUREMENT, against the BOOKED FORM, never against the budget:

* ``D_FIXED_MIB`` per rank = ``weights + runtime state`` of the runtime's own
  ``KV budget posts`` line minus ``buffer x L x slot`` of the same log (what
  :func:`expert_residency.d_rank_reference_from_logs` computes) -- the #145
  fixed post, measured on the form that runs today.
* ``D_AWAKE_REST_MIB`` per rank = the card demand at the edge minus the posts
  of the form that boot ran::

      demand = capacity - free - sum(other processes) + the refused request
      posts  = weights + runtime state + mamba + spec + activation
               + max_total_num_tokens x cell        (all from the same log)
      rest   = demand - posts

  ``demand`` is read from torch's OOM line (``GPU 0 has a total capacity of
  ... of which ... is free. Process N has ... memory in use``): capacity is the
  torch-visible card (the driver carve is already gone, the budget books it as
  its own term), and the bytes no listed process owns (rc12c: 357 MiB at the
  edge, 0 at a release) count to the awake D rank that meets the edge. It is a
  LOWER bound (the rank died there). What it covers: the CUDA graphs, every
  untagged buffer born after the KV profile, the private-pool and allocator
  cache the extend could not reuse, and the transient beyond the activation
  post -- the ~2.6 GiB the budget model never saw.

Neither term depends on the budget: a second boot on the booked form measures
the same rest, so the record has a fixpoint (tested). The launcher books the
rest INSTEAD of the budget-relative overshoot and INSTEAD of the builtin 404
awake overshoot on the ranks where it is measured (it is the whole rest there;
both would count it twice), and the #145 solve prices the form with
``D_FIXED_MIB``.

``python -m sglang.srt.weg2.d_awake_rest --model <ckpt> <D.log> ...`` prints
both record rows for weg2/profile_records_data/<profile>.json, max over the
boots given (a record only grows with evidence).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

RECORD_FIXED = "D_FIXED_MIB"
RECORD_REST = "D_AWAKE_REST_MIB"
MARKER = "WEG2-D-AWAKE-REST"

GIB_IN_MIB = 1024.0
_MIB = 1024 * 1024

_TP = r"\[(?:[0-9-]+ [0-9:]+ )?TP(\d+)\]"
_POSTS = re.compile(
    _TP + r" \[world_rank \d+\] KV budget posts \(GiB\): weights \+ runtime "
    r"state=([0-9.]+), mamba state pool=([0-9.]+), speculative intermediate "
    r"state=([0-9.]+), prefill activation reserve=([0-9.]+)"
)
_CELL = re.compile(_TP + r" KV pool sizing: available_bytes=\d+ .*?cell_size=(\d+),")
_TOKENS = re.compile(_TP + r" max_total_num_tokens=(\d+), chunked_prefill_size")
_BUFFER = re.compile(
    _TP + r" MoE expert-offload active on layer (\d+): \d+/\d+ experts resident "
    r"\+ \d+ scratch \(buffer=(\d+),"
)
_EXC_HEAD = re.compile(_TP + r" Scheduler hit an exception")
_OOM = re.compile(
    r"Tried to allocate (?P<ask>[0-9.]+) (?P<au>[KMG]iB)\. GPU \d+ has a total "
    r"capacity of (?P<cap>[0-9.]+) (?P<cu>[KMG]iB) of which (?P<free>[0-9.]+) "
    r"(?P<fu>[KMG]iB) is free\.(?P<procs>(?: Process \d+ has [0-9.]+ [KMG]iB "
    r"memory in use\.)*) Including non-PyTorch memory, this process has "
    r"(?P<mine>[0-9.]+) (?P<mu>[KMG]iB) memory in use"
)
_PROC = re.compile(r"Process \d+ has ([0-9.]+) ([KMG]iB) memory in use")


def _mib(value: str, unit: str) -> float:
    return float(value) * {"KiB": 1.0 / 1024.0, "MiB": 1.0, "GiB": GIB_IN_MIB}[unit]


class AwakeRestError(ValueError):
    """A log that cannot price the rest -- refused, never guessed."""


@dataclass(frozen=True)
class EdgeSample:
    """One OOM at the card edge, in MiB."""

    rank: int
    capacity: float
    free: float
    others: float
    mine: float
    ask: float

    @property
    def demand(self) -> float:
        """What the awake rank needed at the edge: everything on the torch-
        visible card no listed OTHER process owns, plus the refused request."""
        return self.capacity - self.free - self.others + self.ask

    @property
    def unattributed(self) -> float:
        return self.capacity - self.free - self.others - self.mine


@dataclass(frozen=True)
class RankForm:
    """The booked form of one rank, as its own log states it (MiB)."""

    rank: int
    weights_runtime: float
    mamba: float
    spec: float
    activation: float
    #: ``None`` where the log never sized this rank's KV pool (the rest
    #: cannot be priced there; the fixed post still can)
    kv_tokens: Optional[int]
    cell_bytes: Optional[int]
    buffer_rows: Optional[int]

    @property
    def kv(self) -> Optional[float]:
        if self.kv_tokens is None or self.cell_bytes is None:
            return None
        return float(self.kv_tokens) * float(self.cell_bytes) / _MIB

    @property
    def posts(self) -> Optional[float]:
        kv = self.kv
        if kv is None:
            return None
        return self.weights_runtime + self.mamba + self.spec + self.activation + kv

    def fixed(self, layer_row_mib: float) -> Optional[float]:
        if self.buffer_rows is None:
            return None
        return self.weights_runtime - self.buffer_rows * float(layer_row_mib)


def rank_forms(text: str, *, n_layers: int) -> Dict[int, RankForm]:
    """Per rank the form the runtime booked, from ONE log. A rank without its
    posts line is left out; one without its cell or final token count keeps
    its fixed post but prices no rest."""
    posts: Dict[int, Tuple[float, float, float, float]] = {}
    cell: Dict[int, int] = {}
    tokens: Dict[int, int] = {}
    buffer: Dict[int, Tuple[int, int]] = {}
    for line in text.splitlines():
        m = _POSTS.search(line)
        if m:
            posts[int(m.group(1))] = tuple(float(m.group(i)) * GIB_IN_MIB for i in range(2, 6))
            continue
        m = _CELL.search(line)
        if m:
            cell[int(m.group(1))] = int(m.group(2))
            continue
        m = _TOKENS.search(line)
        if m:
            tokens[int(m.group(1))] = int(m.group(2))
            continue
        m = _BUFFER.search(line)
        if m:
            r, layer, rows = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if layer < n_layers and layer >= buffer.get(r, (-1, 0))[0]:
                buffer[r] = (layer, rows)
    out: Dict[int, RankForm] = {}
    for r, (wr, mamba, spec, act) in posts.items():
        out[r] = RankForm(
            rank=r, weights_runtime=wr, mamba=mamba, spec=spec, activation=act,
            kv_tokens=tokens.get(r), cell_bytes=cell.get(r),
            buffer_rows=buffer[r][1] if r in buffer else None,
        )
    return out


def edge_samples(text: str) -> List[EdgeSample]:
    """Every OOM of a TP rank in the log. A traceback line carries no rank
    prefix; it belongs to the rank whose ``Scheduler hit an exception`` header
    came last."""
    out: List[EdgeSample] = []
    last_exc: Optional[int] = None
    for line in text.splitlines():
        h = _EXC_HEAD.search(line)
        if h:
            last_exc = int(h.group(1))
        m = _OOM.search(line)
        if not m:
            continue
        p = re.match(_TP, line)
        rank = int(p.group(1)) if p else last_exc
        if rank is None:
            continue
        others = sum(_mib(v, u) for v, u in _PROC.findall(m.group("procs")))
        out.append(EdgeSample(
            rank=rank,
            capacity=_mib(m.group("cap"), m.group("cu")),
            free=_mib(m.group("free"), m.group("fu")),
            others=others,
            mine=_mib(m.group("mine"), m.group("mu")),
            ask=_mib(m.group("ask"), m.group("au")),
        ))
    return out


@dataclass(frozen=True)
class BootRest:
    name: str
    forms: Dict[int, RankForm]
    #: rank -> (rest MiB, the sample that set it)
    rest: Dict[int, Tuple[float, EdgeSample]]


def boot_rest(name: str, text: str, *, n_layers: int) -> BootRest:
    forms = rank_forms(text, n_layers=n_layers)
    rest: Dict[int, Tuple[float, EdgeSample]] = {}
    for s in edge_samples(text):
        f = forms.get(s.rank)
        if f is None or f.posts is None:
            continue
        v = s.demand - f.posts
        if s.rank not in rest or v > rest[s.rank][0]:
            rest[s.rank] = (v, s)
    return BootRest(name=name, forms=forms, rest=rest)


def records(
    boots: Sequence[Tuple[str, str]], *, n_ranks: int, n_layers: int, layer_row_mib: float
) -> Tuple[List[Optional[int]], List[Optional[int]], List[str]]:
    """``(D_FIXED_MIB, D_AWAKE_REST_MIB, lines)``: per rank the max over the
    boots, ``None`` where no boot priced that rank. Refuses when no boot
    carries a single edge sample -- a rest record from zero edges is a guess."""
    fixed: List[Optional[float]] = [None] * n_ranks
    rest: List[Optional[float]] = [None] * n_ranks
    lines: List[str] = []
    for name, text in boots:
        b = boot_rest(name, text, n_layers=n_layers)
        for r, f in sorted(b.forms.items()):
            if r >= n_ranks:
                continue
            fx = f.fixed(layer_row_mib)
            if fx is not None:
                fixed[r] = fx if fixed[r] is None else max(fixed[r], fx)
        for r, (v, s) in sorted(b.rest.items()):
            if r >= n_ranks:
                continue
            f = b.forms[r]
            rest[r] = v if rest[r] is None else max(rest[r], v)
            lines.append(
                "%s %s TP%d: demand %.0f = capacity %.0f - free %.0f - others %.0f "
                "+ ask %.0f (this process %.0f, unattributed %.0f) | posts %.0f = W+R "
                "%.0f + mamba %.0f + spec %.0f + act %.0f + KV %d x %d B %.0f (buffer "
                "%s rows) | rest %.0f"
                % (MARKER, name, r, s.demand, s.capacity, s.free, s.others, s.ask,
                   s.mine, s.unattributed, f.posts, f.weights_runtime, f.mamba, f.spec,
                   f.activation, f.kv_tokens, f.cell_bytes, f.kv, f.buffer_rows, v)
            )
    if all(v is None for v in rest):
        raise AwakeRestError(
            "%s: no edge sample (torch OOM line of a TP rank with its KV budget "
            "posts) in %s -- a rest record needs a measured edge"
            % (MARKER, [b[0] for b in boots]))
    as_int = lambda xs: [None if x is None else int(round(x)) for x in xs]  # noqa: E731
    return as_int(fixed), as_int(rest), lines


def _boot_tag(path: str) -> str:
    m = re.search(r"boot_weg2_([A-Za-z0-9]+?)_[0-9a-f]{10}_", path)
    return m.group(1) if m else path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(prog="python -m sglang.srt.weg2.d_awake_rest")
    ap.add_argument("--model", required=True, help="checkpoint dir (expert slot size)")
    ap.add_argument("--ranks", type=int, default=3)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args(argv)
    from sglang.srt.planner import pp_cut as _pp_cut

    terms = _pp_cut.checkpoint_weight_terms(a.model)
    slot_mib = float(terms.expert_layer_weight_bytes) / int(terms.num_experts) / _MIB
    layer_row = int(terms.n_layers) * slot_mib
    boots = []
    for p in a.logs:
        with open(p, errors="replace") as fh:
            boots.append((_boot_tag(os.path.basename(p)), fh.read()))
    fixed, rest, lines = records(
        boots, n_ranks=a.ranks, n_layers=int(terms.n_layers), layer_row_mib=layer_row)
    for ln in lines:
        print(ln)
    tags = [b[0] for b in boots]
    print(json.dumps({"name": RECORD_FIXED, "value": fixed, "boots": tags}))
    print(json.dumps({"name": RECORD_REST, "value": rest, "boots": tags}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
