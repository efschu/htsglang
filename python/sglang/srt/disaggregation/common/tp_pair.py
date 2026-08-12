# SPDX-License-Identifier: Apache-2.0
"""Which prefill/decode TP pairs the KV transfer path can actually express (#643).

Deliberately dependency-free: no torch, no triton, no sglang imports. The
refusal below runs on the live PD handshake path
(``common/conn.py.try_ensure_parallel_info``) and on the prefill arm's
registration path, and neither should acquire a heavyweight import to ask an
integer question. ``staging_buffer.py`` re-exports both names so the historical
import site keeps working.
"""


class HeadSplitNotRepresentable(ValueError):
    """A prefill/decode TP pair whose KV head split this path cannot express.

    Raised instead of returning silently wrong slice coordinates. See
    ``validate_tp_pair_divisible`` for why refusal is the correct shipped
    behaviour rather than a truncating best effort.
    """


def validate_tp_pair_divisible(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    total_kv_heads: int,
    where: str,
) -> None:
    """Refuse a prefill/decode TP pair that this head split cannot represent.

    The whole PD transfer path splits KV heads with a bare floor division,
    ``total_kv_heads // attn_tp_size`` on each side, and then maps one source
    rank's slice onto one destination rank's slice. That mapping is a
    partition only when the two TP sizes stand in an integer-multiple
    relationship. When neither divides the other -- prefill TP=3 against
    decode TP=2 is the smallest real case -- the arithmetic produces, all at
    once and all without a word:

      * two source ranks resolving to the same destination head offset, so
        one rank's KV silently overwrites the other's;
      * a ``dst_head_start + num_heads`` past the destination rank's own head
        capacity, i.e. a write outside that rank's slice;
      * a writer count truncated by floor division, so a staging region is
        sized for fewer writers than write into it AND the receiver declares
        a chunk complete after too few arrivals;
      * whole source ranks never contacted, their heads never transferred.

    Refusing rather than truncating is deliberate, and the tree already argues
    the case. A correct split exists -- ``draft_kv_canonical.local_head_window``
    implements the largest-remainder partition -- and its docstring names
    ``compute_head_slice_params`` as the thing it declines to reuse, because
    that function returns ONE head count used as both the gather and the
    scatter extent and therefore cannot represent "this source rank owns 2
    heads, this destination rank wants 1". Making the pair work is not a fix
    to one expression; it is a change to the transfer loop structure in three
    transport backends, and that cannot be shown correct without a
    two-instance PD boot. A wrong answer delivered quietly is worse than no
    answer delivered loudly, so this path refuses and says exactly why.

    Args:
        src_attn_tp_size: the prefill arm's attention TP size.
        dst_attn_tp_size: the decode arm's attention TP size.
        total_kv_heads: model KV head total, reported for context only.
        where: caller site name, echoed in the message so an operator can act
            on the error without reading the source.

    Raises:
        HeadSplitNotRepresentable: if neither TP size divides the other.
    """
    if src_attn_tp_size <= 0 or dst_attn_tp_size <= 0:
        raise HeadSplitNotRepresentable(
            f"[{where}] invalid TP sizes: prefill attn_tp_size="
            f"{src_attn_tp_size}, decode attn_tp_size={dst_attn_tp_size}; "
            "both must be positive."
        )
    if src_attn_tp_size % dst_attn_tp_size == 0:
        return
    if dst_attn_tp_size % src_attn_tp_size == 0:
        return
    larger = max(src_attn_tp_size, dst_attn_tp_size)
    smaller = min(src_attn_tp_size, dst_attn_tp_size)
    raise HeadSplitNotRepresentable(
        f"[{where}] PD KV transfer refuses a non-divisible TP pair: prefill "
        f"attn_tp_size={src_attn_tp_size}, decode attn_tp_size="
        f"{dst_attn_tp_size} (total_kv_heads={total_kv_heads}). Neither size "
        f"divides the other ({larger} % {smaller} = {larger % smaller}), so "
        "the head split this path performs is not a partition: source ranks "
        "overlap on destination heads, writes land past a destination rank's "
        "head capacity, the writer count truncates so the receiver completes "
        "a chunk early, and some prefill ranks are never contacted at all. "
        "Before #643 this corrupted KV silently and answered with fluent "
        "wrong output. Choose prefill and decode attn_tp_size values where "
        f"one is an integer multiple of the other (e.g. decode {smaller} with "
        f"prefill {smaller * 2}, or equal sizes)."
    )
