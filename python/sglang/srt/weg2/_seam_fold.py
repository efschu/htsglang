# SPDX-License-Identifier: Apache-2.0
"""#1350 F1: the DEVICE-SIDE, POSITION-SENSITIVE fold behind the seam digest.

WHY THIS EXISTS -- the adversarial review (REVIEW_DIGEST_1350_0912.md, F1).
The first build hashed on the HOST: one synchronous D2H per 32 MiB block, then
``hashlib.blake2b``, on the scheduler thread, inside the timed flip leg.  The
review measured host blake2b at 0.19 GB/s under foreign load (0.9-1.0 GB/s is
the quiet-box reference) against a population the module itself calls
"~9.6 GiB per rank".  That is 10-54 s PER READING and two readings per flip,
against ``front._flip_stall_bound_s`` (``drain_deadline_s`` = 120 s before the
first measured flip) and the 120 s C14 credit budget the co-located WAKING rank
is fenced on -- because the sleeping rank publishes its credit only AFTER its
``before`` hash.  The first armed flip could raise ``WEG2-FLIP STALL`` and take
the deadman with it, with no byte wrong anywhere: the grader as the defect.

WHAT THIS DOES INSTEAD.  The fold runs where the bytes already are, and the
host learns the answer ONCE per reading:

* per piece, the canonical row-major byte image is read in bounded blocks;
* each block is folded into a device-resident ``int64`` accumulator pair;
* NOTHING is transferred or synchronised during the walk;
* after the last piece, ONE accumulator tensor per device goes to the host --
  that transfer IS the synchronisation, and it is counted and printed.

POSITION SENSITIVITY IS THE LOAD-BEARING HALF, and it is why
``weights_arena.uint8_checksum`` (12 callers, device-side int64 SUM) could not
simply be reused: **a sum is permutation-blind.**  A q<->k swap inside a fused
``qkv_proj`` -- the exact class the exchange could get wrong -- leaves the sum
unchanged.  So each 8-byte word is weighted by its GLOBAL index within the
piece, in two independent lanes:

    lane0 = sum_i  w_i * (i + 1)                        (mod 2**64)
    lane1 = sum_i  w_i * ((i + 1) * (i + 1) + ODD)      (mod 2**64)

Swapping words ``a`` and ``b`` at positions ``i`` and ``j`` moves lane0 by
``(a - b) * (i - j)``, which is zero only when the words are equal.  Two lanes
with different index polynomials make an accidental double cancellation
implausible without paying for a cryptographic hash on the critical path.
``int64`` overflow in torch wraps two's-complement, i.e. arithmetic mod 2**64,
which is exactly the intended ring.

THE WORD GRID IS GLOBAL PER PIECE, NOT PER BLOCK, and that is not a detail: the
block boundaries follow the tensor's ROW geometry, so a piece whose row length
is not a multiple of 8 would otherwise be folded on a different word grid
depending on how it happened to be sliced.  A carry of at most 7 bytes is
handed from one block to the next, so the fold is a pure function of the
piece's logical byte sequence and of nothing else.  That is what keeps the
digest identical across a re-arena (different pointer, different pitch, same
content), which is the property the whole instrument rests on.

THE LENGTH IS FOLDED IN at the end, so a truncated piece can never collide with
the whole: a fold over a prefix has the same lane sums as the whole only if the
dropped tail is all zeros, and the byte count separates those two anyway.

WHAT THIS IS NOT.  It is not a cryptographic hash and does not pretend to be:
it is collision-resistant against the ACCIDENTS a transport makes (dropped
blocks, shifted offsets, swapped rows, stale pages), not against an adversary
choosing bytes.  Nothing in this fork writes weights adversarially, and the
alternative -- a real hash on the flip's critical path -- is the defect this
module was written to remove.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

#: Second lane's additive constant.  Odd, so it never shares a factor of two
#: with the index polynomial; otherwise arbitrary.
LANE1_ODD = 0x9E3779B97F4A7C15 - (1 << 64)  # signed image of the golden ratio

#: Bounded transient per fold step.  The widening to ``int64`` costs 8x the
#: block in device memory for the index vector plus the product, so this is the
#: knob that bounds the grader's own VRAM footprint during a flip -- when the
#: card is at its tightest, which is exactly when the exchange runs.
DEFAULT_CHUNK_BYTES = 4 << 20


def _torch():
    import torch

    return torch


class DeviceAccumulators:
    """One ``int64[n, 2]`` accumulator per device, drained ONCE per reading.

    The class exists so that "one sync per reading" is a property of an object
    a test can count, rather than a claim about the shape of a loop.
    """

    def __init__(self, n_pieces: int):
        self.n_pieces = int(n_pieces)
        self._per_device: Dict[Any, Any] = {}
        self._rows: Dict[int, Any] = {}
        self.syncs = 0

    def _acc_for(self, device) -> Any:
        torch = _torch()
        key = str(device)
        acc = self._per_device.get(key)
        if acc is None:
            acc = torch.zeros((self.n_pieces, 2), dtype=torch.int64, device=device)
            self._per_device[key] = acc
            self._rows[key] = device
        return acc

    def add(self, device, row: int, lane0, lane1) -> None:
        acc = self._acc_for(device)
        acc[row, 0] += lane0
        acc[row, 1] += lane1

    def drain(self) -> List[Tuple[int, int]]:
        """THE ONE HOST SYNC.  Returns ``[(lane0, lane1), ...]`` per piece.

        Every accumulator is summed on its own device first, so the number of
        host transfers is the number of DEVICES, not the number of pieces or
        blocks -- one on every real rank, because a rank's weights live on one
        card.
        """
        torch = _torch()
        if not self._per_device:
            return [(0, 0)] * self.n_pieces
        total = None
        for acc in self._per_device.values():
            host = acc.to("cpu") if acc.device.type != "cpu" else acc
            self.syncs += 1
            total = host.clone() if total is None else total + host
        return [(int(r[0]), int(r[1])) for r in total]


def _canonical_blocks(tensor: Any, chunk_bytes: int):
    """Yield the piece's logical content as canonical row-major ``uint8`` blocks.

    ``contiguous()`` is what makes this a CONTENT reading and not a span
    reading: it materialises the row-major image of the logical values, so a
    fresh arena, a non-zero storage offset and a padded pitch all yield the SAME
    bytes, and the pad bytes of a pitched allocation are not in the reading at
    all.  Blocks stay on their own device -- no transfer happens here.
    """
    torch = _torch()
    if tensor.numel() == 0:
        return
    piece = tensor.detach()
    if piece.dim() == 0:
        piece = piece.reshape(1)
    rows = int(piece.shape[0])
    row_elems = max(1, piece.numel() // max(1, rows))
    row_bytes = max(1, row_elems * piece.element_size())
    rows_per_block = max(1, int(chunk_bytes) // row_bytes)
    for start in range(0, rows, rows_per_block):
        block = piece[start : start + rows_per_block].contiguous()
        u8 = block.reshape(-1).view(torch.uint8)
        # ALIGNMENT, and it is not a nicety: ``contiguous()`` on an already
        # contiguous SLICE returns the slice itself, storage offset and all, so
        # ``big[5:45]`` arrives at byte offset 20 and ``view(int64)`` refuses
        # ("storage_offset must be divisible by 8").  A clone resets the offset
        # to 0.  It fires only for a misaligned slice -- a real parameter sits
        # at offset 0 and is never copied here.
        if u8.storage_offset() % 8:
            u8 = u8.clone()
        yield u8


_MASK64 = (1 << 64) - 1


def _s64(value: int) -> int:
    """A Python int as its signed 64-bit image -- the ring torch computes in."""
    v = int(value) & _MASK64
    return v - (1 << 64) if v >= (1 << 63) else v


_IDX_CACHE: Dict[Tuple[str, int], Tuple[Any, Any]] = {}


def _index_vectors(device, k: int):
    """``(i, i*i)`` for ``i = 1..k``, CACHED per (device, length).

    The first fold built a fresh ``arange`` per block, which is the single
    largest avoidable cost on the whole path: it allocates and writes a vector
    the size of the block ON THE CARD, during a flip, when the card is at its
    tightest -- and then reads it straight back.  Every block of a reading has
    the same length except the last, so two entries per device serve an entire
    walk.
    """
    torch = _torch()
    key = (str(device), int(k))
    got = _IDX_CACHE.get(key)
    if got is None:
        i = torch.arange(1, k + 1, dtype=torch.int64, device=device)
        got = (i, i * i)
        _IDX_CACHE[key] = got
    return got


def fold_piece(tensor: Any, acc: DeviceAccumulators, row: int,
               chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Fold ONE piece into ``acc[row]``.  Returns its logical byte count.

    No transfer, no ``.item()``, no synchronisation: everything stays on the
    piece's own device until :meth:`DeviceAccumulators.drain`.

    THE GLOBAL INDEX IS APPLIED ALGEBRAICALLY, not materialised.  With ``off``
    words already consumed and local positions ``i = 1..k``, the global position
    is ``g = off + i``, so

        sum w*g        = off*S0 + S1
        sum w*(g*g+C)  = (off*off + C)*S0 + 2*off*S1 + S2

    with ``S0 = sum w``, ``S1 = sum w*i``, ``S2 = sum w*i*i`` over the block.
    That is three reductions against ONE cached pair of index vectors, instead
    of a fresh position vector per block -- same fold, same position
    sensitivity, without allocating the card's memory to describe it.
    """
    torch = _torch()
    nbytes = int(tensor.numel()) * int(tensor.element_size())
    if nbytes == 0:
        return 0
    device = tensor.device
    carry = None          # < 8 bytes handed across a block boundary
    word_index = 0        # GLOBAL word position within this piece
    for block in _canonical_blocks(tensor, chunk_bytes):
        buf = block if carry is None else torch.cat((carry, block))
        n = int(buf.numel())
        n8 = (n // 8) * 8
        if n8:
            words = buf[:n8].view(torch.int64)
            k = int(words.numel())
            i1, i2 = _index_vectors(device, k)
            s0 = words.sum()
            s1 = (words * i1).sum()
            s2 = (words * i2).sum()
            off = word_index
            acc.add(
                device, row,
                s0 * _s64(off) + s1,
                s0 * _s64(off * off + LANE1_ODD) + s1 * _s64(2 * off) + s2,
            )
            word_index += k
        carry = buf[n8:].clone() if n8 < n else None
    if carry is not None and int(carry.numel()):
        # THE TAIL, folded per BYTE on the same global grid: a piece whose byte
        # count is not a multiple of 8 must still be covered, and silently
        # dropping up to 7 bytes is precisely the edge class the review's
        # MB/MW mutants exist to catch.  At most 7 elements, so the index is
        # built directly -- caching it would cost more than it saves.
        tail = carry.to(torch.int64)
        k = int(tail.numel())
        base = word_index * 8
        idx = torch.arange(
            base + 1, base + 1 + k, dtype=torch.int64, device=device
        )
        acc.add(device, row, (tail * idx).sum(),
                (tail * (idx * idx + LANE1_ODD)).sum())
    return nbytes
