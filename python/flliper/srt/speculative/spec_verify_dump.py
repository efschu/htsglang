# SPDX-License-Identifier: Apache-2.0
"""Debug tap for the target-verify step, feeding the #124 determinism harness.

WHY THIS EXISTS. ``ModelRunner._determinism_dump_logits`` (the #124 tap) hangs
off ``ModelRunner.sample`` and early-returns unless ``logits.shape[0] == 1``.
A speculative step reaches neither condition: the target verify's logits are
consumed inside ``eagle_utils.eagle_sample``, and they carry
``bs * draft_token_num`` rows. So the existing tap is structurally blind to
speculation, and the only observable a spec gate had was the emitted token
sequence -- which Window 5 showed is not a usable oracle on its own, because
speculation changes the target forward's shape and therefore its floating-
point reduction order.

WHAT IT RECORDS. Per verify step: the full ``[bs*D, V]`` target logits matrix
(before any sampling preprocessing -- the byte-identity classes are
dtype-strict), the draft candidates, and the accept result. The full matrix
is kept rather than only the accepted rows: the REJECTED slots are what
explain an accept-length difference between two arms, and an accept-length
difference is the one lane failure signature the token-level checks are blind
to (a verify attending over the wrong KV slots still emits self-consistent
tokens; it just stops accepting).

Default off. Guarded by the existing ``--determinism-logits-dump-dir``, so
the default serving path is untouched. Volume is not small -- a 128k-vocab
fp32 verify matrix at k=3 is ~2 MB per round -- which is acceptable for a
bounded gate run and is the reason the flag stays a debug surface.

WHAT IT DOES NOT RECORD. A weightless worker rank has no logits at all (it
runs the stripped attention-only verify and receives the accept result off
the rank-0 broadcast), so nothing is written there. That is correct: the
head rank's logits ARE the verified distribution for the whole group.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "accepted_row_indices",
    "build_verify_record",
    "maybe_dump_verify_step",
    "reset_verify_dump_counter",
    "write_verify_record",
]


def accepted_row_indices(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    num_rows: Optional[int] = None,
) -> List[List[int]]:
    """Which verify row produced which emitted token, per request.

    ``accept_index[b, j]`` is the GLOBAL flat index (into the ``bs * D`` row
    space) of request ``b``'s j-th accepted node, and the greedy verify writes
    ``predicts[g] = target_predict[g] = argmax(logits[g])`` at exactly those
    indices (``verify_tree_greedy_kernel_triton``; the sgl_kernel op is
    identical). So row ``accept_index[b, j]`` is the parent of emitted token
    ``j`` -- for chain and tree layouts alike, which is why this reads the
    index rather than assuming the chain's ``0, 1, 2, ...``.

    ``accept_lens`` is the width INCLUDING the trailing bonus token, i.e. the
    ``num_correct_drafts + 1`` that ``eagle_sample`` returns.

    Everything that could make the resulting trajectory quietly wrong is a
    hard error, never a clamp: a ``-1`` inside the accepted span, a
    zero-length accept, an index outside the row space. A misread here
    produces a plausible-looking trajectory, which is worse than a crash.
    """
    if accept_index.dim() != 2:
        raise ValueError(
            f"accept_index must be [bs, W], got {tuple(accept_index.shape)}"
        )
    if accept_lens.dim() != 1 or accept_lens.shape[0] != accept_index.shape[0]:
        raise ValueError(
            f"accept_lens must be [bs] matching accept_index rows; got "
            f"{tuple(accept_lens.shape)} vs {tuple(accept_index.shape)}"
        )
    idx = accept_index.detach().to("cpu", torch.int64)
    lens = accept_lens.detach().to("cpu", torch.int64)
    out: List[List[int]] = []
    for b in range(idx.shape[0]):
        n = int(lens[b])
        if n < 1:
            raise ValueError(
                f"request {b}: accept_len {n} -- a verify round always commits "
                "at least the bonus token"
            )
        if n > idx.shape[1]:
            raise ValueError(
                f"request {b}: accept_len {n} exceeds accept_index width "
                f"{idx.shape[1]}"
            )
        rows = [int(v) for v in idx[b, :n]]
        for j, r in enumerate(rows):
            if r < 0:
                raise ValueError(
                    f"request {b}: accept_index[{b}, {j}] = {r} inside the "
                    f"accepted span (accept_len {n}) -- accept_lens and "
                    "accept_index disagree"
                )
            if num_rows is not None and r >= num_rows:
                raise ValueError(
                    f"request {b}: accept_index[{b}, {j}] = {r} is out of "
                    f"range for {num_rows} verify rows"
                )
        out.append(rows)
    return out


def build_verify_record(
    *,
    step: int,
    tp_rank: int,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> Dict[str, Any]:
    """Assemble one verify step's record. Pure: no I/O, no CUDA requirement.

    ``accept_lens`` is the returned width (drafts + bonus). ``logits`` keeps
    its serving dtype -- upcasting here would silently soften every
    machine-zero comparison downstream.
    """
    if logits.dim() != 2:
        raise ValueError(f"verify logits must be [bs*D, V], got {tuple(logits.shape)}")
    num_rows = int(logits.shape[0])
    rows = accepted_row_indices(accept_index, accept_lens, num_rows=num_rows)
    flat_predict = predict.detach().reshape(-1).to("cpu", torch.int64)
    emitted: List[List[int]] = [[int(flat_predict[r]) for r in req] for req in rows]
    return {
        "step": int(step),
        "tp_rank": int(tp_rank),
        "mode": "target_verify",
        "bs": int(accept_index.shape[0]),
        "draft_token_num": int(draft_token_num),
        "logits": logits.detach().to("cpu", copy=True),
        "candidates": candidates.detach().to("cpu", torch.int64, copy=True),
        "predict": flat_predict,
        "accept_index": accept_index.detach().to("cpu", torch.int64, copy=True),
        "accept_lens": accept_lens.detach().to("cpu", torch.int64, copy=True),
        "accepted_rows": rows,
        "emitted": emitted,
    }


def write_verify_record(dump_dir: str, record: Dict[str, Any]) -> str:
    """Write one record atomically (tmp + rename), mirroring the #124 tap so a
    reader never sees a partial file."""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(
        dump_dir, f"rank{record['tp_rank']}_verify{record['step']:07d}.pt"
    )
    tmp_path = path + ".tmp"
    torch.save(record, tmp_path)
    os.replace(tmp_path, path)
    return path


#: Per-rank verify-step counter. Module-level ON PURPOSE: the natural-looking
#: alternative, hanging it off the EagleVerifyInput, is wrong -- that object is
#: rebuilt every round, so the counter would reset to 0 each step and every
#: record would overwrite the same file.
_STEP_COUNTER: Dict[int, int] = {}


def reset_verify_dump_counter(tp_rank: Optional[int] = None) -> None:
    """Test seam; also lets a gate run start a fresh numbering per request."""
    if tp_rank is None:
        _STEP_COUNTER.clear()
    else:
        _STEP_COUNTER.pop(int(tp_rank), None)


def maybe_dump_verify_step(
    *,
    dump_dir: Optional[str],
    tp_rank: int,
    logits: Optional[torch.Tensor],
    candidates: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> None:
    """Call-site convenience: no-op unless the debug flag is set and this rank
    actually holds logits (a weightless worker does not).

    Never raises into the serving path -- a debug tap that can abort a verify
    step is a worse defect than the missing datapoint it would report.
    """
    if not dump_dir or logits is None:
        return
    step = _STEP_COUNTER.get(int(tp_rank), 0)
    _STEP_COUNTER[int(tp_rank)] = step + 1
    try:
        record = build_verify_record(
            step=step,
            tp_rank=tp_rank,
            logits=logits,
            candidates=candidates,
            predict=predict,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )
        write_verify_record(dump_dir, record)
    except Exception as exc:  # pragma: no cover - debug surface
        import logging

        logging.getLogger(__name__).warning(
            "spec verify dump skipped at step %s: %s", step, exc
        )
