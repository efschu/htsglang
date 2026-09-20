# SPDX-License-Identifier: Apache-2.0
"""Form A slice 6a, first: do the ranks still AGREE about the collectives?

This module exists because of what the slice-6a brief asked for and what
that request actually implies. The goal was stated as "the boot form
triggers no known refusal at the desk". Making the refusals stop firing and
making Form A CORRECT are not the same thing, and the gap between them is
the most expensive failure this rig has:

    A worker whose dense modules are placeholders no longer executes the
    per-layer all-reduce that the attention host still executes
    (qwen4_exp.py:1100-1101 o_proj, :1574 attn_tp_all_reduce, :1640
    attn_tp_all_gather). The host then blocks in a collective that two of
    three ranks will never join. That is not an exception. It is a HANG --
    silent, on all three cards, until the deadman fires.

Removing the refusals without changing the HOST side therefore converts a
loud desk-time error into a wedged rig. The rule this repo already carries
for exactly this ("RAENGE NIE UNEINS -- Uneinigkeit = CRASH/STOP") is the
reason this check is written before the surgery, not after it.

So: a rank's forward is modelled here as the SEQUENCE OF COLLECTIVES it
issues. Three fake ranks run the sequence, and the probe answers one
question -- would they all arrive at the same collective, in the same order?
A "no" names the first divergence, the layer it happens in, and which ranks
went which way.

This is a MODEL of the forward, not the forward. It cannot prove the boot
works. It can prove the boot HANGS, which is the half that matters before a
GPU window is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.rank_role import HOST, RankRolePlan, WORKER

__all__ = [
    "CollectiveMismatch",
    "CollectiveOp",
    "FormAWorkerWithoutMoeInput",
    "RankTrace",
    "trace_layer",
    "trace_forward_edges",
    "check_symmetry",
    "probe_form_a_boot",
]


class CollectiveMismatch(RuntimeError):
    """Two ranks would enter different collectives -- i.e. they would hang."""


class FormAWorkerWithoutMoeInput(RuntimeError):
    """A worker would compute experts on a value nobody handed it.

    The blind spot of the first version of this model, and worth naming
    rather than fixing quietly: a collective-sequence model is silent about
    DATA. It checked that three ranks enter the same ops in the same order
    and said nothing about whether the rows they enter with exist. On a
    classic boot that gap is harmless -- every rank computes the MoE input
    itself, because the dense path ends in an all-reduce. Under Form A the
    value lives on one card, and the configuration the model called "the
    simple Form A, 48 collectives" had no op that moves it.
    """


@dataclass(frozen=True)
class CollectiveOp:
    """One collective, identified by what every participant must agree on."""

    kind: str  # "all_reduce" | "all_gather" | "broadcast" | "reduce"
    site: str  # where in the forward, e.g. "layer3.o_proj"
    payload: str = ""  # a shape/dtype tag; differing tags also hang

    def key(self) -> Tuple[str, str, str]:
        return (self.kind, self.site, self.payload)


@dataclass
class RankTrace:
    rank: int
    role: str
    ops: List[CollectiveOp] = field(default_factory=list)

    def issue(self, kind: str, site: str, payload: str = "") -> None:
        self.ops.append(CollectiveOp(kind, site, payload))


def trace_layer(
    trace: RankTrace,
    layer_id: int,
    is_attention_layer: bool,
    *,
    worker_skips_dense: bool,
    host_uses_moe_exchange: bool,
    host_dense_is_unsharded: bool,
    moe_input_carrier: Optional[str] = None,
) -> None:
    """Model ONE decoder layer's collectives for one rank.

    FOUR switches now. The fourth, `moe_input_carrier`, is the slice-6a
    worker-forward finding: see `FormAWorkerWithoutMoeInput`. It is one op
    per MoE layer, issued by EVERY rank (the host publishes, the workers
    contribute zeros or receive), and it sits immediately before the MoE.
    `None` means no carrier -- correct today, and a data gap under Form A
    unless the host-centric exchange provides one, which its broadcast
    does.

    The first three, and that count was the finding of the previous round.
    The slice plan had two:

      `worker_skips_dense`      -- slice 6a: the worker stops executing the
                                   dense path.
      `host_uses_moe_exchange`  -- the later slice: the host broadcasts and
                                   reduces the MoE instead of all-reducing.

    The probe showed that both together STILL hang, and the reason is
    obvious once seen: the host's own dense collectives -- the q all-gather
    and the o_proj all-reduce -- exist only because the dense side is
    SHARDED across ranks. Under Form A it is not sharded at all; rank 0 owns
    every head. Those collectives have no second participant left and must
    disappear with the sharding, which is a change to the HOST, not to the
    worker:

      `host_dense_is_unsharded` -- the host stops issuing per-layer dense
                                   collectives entirely.

    Leaving it out is how "the worker no longer refuses" turns into "all
    three cards are parked in different collectives".
    """
    is_worker = trace.role == WORKER
    dense_silent = is_worker and worker_skips_dense
    if trace.role == HOST and host_dense_is_unsharded:
        # Nothing to gather, nothing to reduce: this rank holds every head
        # and every dense weight. The collective is not skipped as an
        # optimisation, it has no counterpart.
        dense_silent = True

    if not dense_silent:
        # The dense side of the layer, as the model runs it today.
        if is_attention_layer:
            trace.issue("all_gather", f"layer{layer_id}.q_gather", "q")
            trace.issue("all_reduce", f"layer{layer_id}.o_proj", "hidden")
        else:
            trace.issue("all_reduce", f"layer{layer_id}.linear_attn", "hidden")

    # The MoE INPUT carrier. Every rank is here: the host publishes its
    # value, a worker contributes zeros (an all-reduce whose other addends
    # are zero IS a broadcast) or receives one.
    #
    # It exists only because SOME rank cannot compute the input itself, so
    # it is conditioned on `worker_skips_dense` and not on the carrier
    # setting alone: on a classic boot every rank's own dense path ends in
    # an all-reduce and produces the identical value, so a carrier there
    # would be a 49th collective for a value everyone already has.
    # Skipped when the host-centric exchange broadcasts the input below --
    # two carriers would be two collectives for one value.
    if (
        moe_input_carrier is not None
        and worker_skips_dense
        and not host_uses_moe_exchange
    ):
        kind = "broadcast" if moe_input_carrier == "broadcast" else "all_reduce"
        trace.issue(kind, f"layer{layer_id}.moe_in", "hidden")

    # The MoE side. Every rank owns experts, so every rank is here.
    if host_uses_moe_exchange:
        trace.issue("broadcast", f"layer{layer_id}.moe_in", "rows")
        trace.issue("reduce", f"layer{layer_id}.moe_out", "rows")
    else:
        trace.issue("all_reduce", f"layer{layer_id}.moe_combine", "hidden")


def trace_forward_edges(
    trace: RankTrace,
    *,
    worker_skips_dense: bool,
    vocab_is_host_only: bool,
) -> None:
    """The two collectives that sit OUTSIDE the layer loop -- seam F13.

    Kept in its own function, and out of the per-layer count, because that
    is what it is: once per forward, not once per layer. The seam survey
    missed these for exactly that reason -- it was looking at a decoder
    layer -- and they hang the boot one op EARLIER than F12's would.

      embed_tokens : VocabParallelEmbedding all-reduces the masked lookup
                     across the vocab shards (layers/
                     vocab_parallel_embedding.py:730-732).
      lm_head      : LogitsProcessor all-gathers the logit shards.

    `vocab_is_host_only` is the built answer (F13): the host's vocab layers
    are constructed with enable_tp=False, so there is one shard, no mask
    and no collective, and a worker builds neither module. With it False --
    which is what `tp_vocab_ratios` gives you by default, since the vocab
    family deliberately does NOT inherit the base ratio vector -- the host
    issues both and a worker that skips the dense path issues neither.
    """
    if vocab_is_host_only:
        return
    if trace.role == WORKER and worker_skips_dense:
        return
    trace.issue("all_reduce", "embed_tokens", "hidden")
    trace.issue("all_gather", "lm_head", "logits")


def check_symmetry(traces: Sequence[RankTrace]) -> None:
    """Refuse unless every rank issues the identical collective sequence.

    Names the FIRST divergence and who went which way -- the information a
    wedged boot cannot give you, because by then every rank is parked in a
    different place with no log line between them.
    """
    if len(traces) < 2:
        return
    lengths = {t.rank: len(t.ops) for t in traces}
    depth = min(lengths.values())
    for i in range(depth):
        keys: Dict[Tuple[str, str, str], List[int]] = {}
        for t in traces:
            keys.setdefault(t.ops[i].key(), []).append(t.rank)
        if len(keys) > 1:
            groups = "; ".join(
                f"ranks {rs} -> {k[0]} at {k[1]}" for k, rs in keys.items()
            )
            raise CollectiveMismatch(
                f"collective #{i} differs between ranks: {groups}. Every "
                "rank must enter the same collective in the same order or "
                "the ones that arrive block forever on the ones that never "
                "do -- a HANG, not an error. This is the "
                "'RAENGE NIE UNEINS' rule as a check."
            )
    if len(set(lengths.values())) > 1:
        short = min(lengths, key=lambda r: lengths[r])
        long_ = max(lengths, key=lambda r: lengths[r])
        raise CollectiveMismatch(
            f"ranks issue different NUMBERS of collectives: rank {short} "
            f"{lengths[short]}, rank {long_} {lengths[long_]}. The first "
            f"{depth} agree, so rank {long_} would block on collective "
            f"#{depth} ({traces[0].ops[depth - 1].site if depth else 'n/a'} "
            "was the last agreed one) while rank "
            f"{short} has already left the forward."
        )


def probe_form_a_boot(
    plan: RankRolePlan,
    num_layers: int = 48,
    attention_every: int = 4,
    *,
    worker_skips_dense: bool,
    host_uses_moe_exchange: bool,
    host_dense_is_unsharded: bool = False,
    moe_input_carrier: Optional[str] = None,
    vocab_is_host_only: bool = True,
) -> List[RankTrace]:
    """The whole-boot probe: N layers, three fake ranks, two verdicts.

    Returns the traces when the configuration is symmetric AND every rank
    has the rows it computes on; raises `CollectiveMismatch` naming the
    first divergence, or `FormAWorkerWithoutMoeInput` when a worker would
    run its experts on a value no op ever moved to it.

    The HANG verdict comes first and the data verdict second, deliberately:
    a configuration that hangs costs a GPU window and leaves no log line,
    while one that is merely missing its input fails loudly at the first
    shape. Both are checked -- a configuration can be perfectly symmetric
    and still wrong, which is exactly what the first version of this probe
    waved through.
    """
    traces = [
        RankTrace(rank=r, role=plan.role_of(r)) for r in range(plan.tp_size)
    ]
    for t in traces:
        trace_forward_edges(
            t,
            worker_skips_dense=worker_skips_dense,
            vocab_is_host_only=vocab_is_host_only,
        )
    for layer_id in range(num_layers):
        is_attn = (layer_id + 1) % attention_every == 0
        for t in traces:
            trace_layer(
                t,
                layer_id,
                is_attn,
                worker_skips_dense=worker_skips_dense,
                host_uses_moe_exchange=host_uses_moe_exchange,
                host_dense_is_unsharded=host_dense_is_unsharded,
                moe_input_carrier=moe_input_carrier,
            )
    check_symmetry(traces)
    if worker_skips_dense and not host_uses_moe_exchange and moe_input_carrier is None:
        raise FormAWorkerWithoutMoeInput(
            "the ranks AGREE about the collectives and it is still not a "
            "runnable layout: a worker skips the dense path, so it does not "
            "compute the MoE input; the host-centric exchange is off, so "
            "nothing broadcasts it; and no moe_input_carrier is set, so no "
            "op moves it either. THIS IS THE CONFIGURATION THE SWITCH "
            "MATRIX PRICED AT 48 COLLECTIVES -- symmetry is not "
            "sufficiency. Set moe_input_carrier='all_reduce_zero' (the host "
            "publishes, the workers add zeros -- see form_a_worker_forward, "
            "96 collectives per round) or turn the exchange on."
        )
    return traces
