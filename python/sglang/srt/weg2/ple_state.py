"""fnFL2 H63c: the Qwen4-Exp PLE side states across the P -> D hand-off.

THE GAP. A request slot of the mamba pool carries, besides the GDN state
(``mamba_cache.temporal`` / ``conv``), two PLE side states registered as slot
siblings (mem_cache/ple_state_pool.py): the n-gram history (``NGramPool``, the
last ``ngram_size - 1`` tokens the trigram hash of the next position reads) and
the short-conv window (``ShortConvPool``, the last ``(kernel-1) * ngram_size``
inputs of the PLE layer's dilated conv -- 9 on Qwen3.8-Flash-Next, 10240
channels, gated by the hidden state, so NOT derivable from token ids). Every
P -> D path moved the GDN state only: the mamba host pool / arena slot
(``MambaPoolHost``: temporal + conv) and the tail parts (weg2/tail_handoff.py:
``_gdn_slot_to_host``). D's slot therefore resumed with an EOS history and a
zero (or a previous occupant's) window -- the first 2 positions after a resume
hashed their n-grams against EOS, the first 9 convolved against zeros.

THE HAND-OFF (``SGLANG_WEG2_PLE_STATE_HANDOFF``, both groups). The rows of one
slot (``snapshot``: ``{"ngram": [n, ctx], "conv": [layers, n, C, L]}``) go
wherever the GDN state goes, and are only written and installed on a rank that
runs a PLE layer (its short-conv pool is built with the stage's own PLE layers
only -- ``is_owner``):

* E1 (state at c): with the stash capture; D queues them for the request
  (``queue``) and ``apply_pending`` writes them from ``_prepare_ple_batch`` --
  after the model runner's COW/clear of the slot, before the first PLE read;
* END (state after N, E2 skip): with the END capture; D writes them in
  ``tail_adopt._install_end`` (no forward runs; the next decode reads them);
* arena anchors (page resume): a side table beside the mamba arena
  (``PleSideTable``: one row per arena slot, tagged with the slot's stem key),
  written by the rank's arena backup, read into every load target by the arena
  load (on the load stream; ``_prepare_ple_batch`` joins it before reading).

Unarmed, nothing here is called: the default path is byte-identical.
"""

from __future__ import annotations

import hashlib
import logging
import mmap
import os
import struct
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

Rows = Dict[str, torch.Tensor]


def enabled() -> bool:
    return bool(envs.SGLANG_WEG2_PLE_STATE_HANDOFF.get())


# -- the pools ---------------------------------------------------------------------
def ple_pools(pool) -> Tuple[Optional[object], Optional[object]]:
    """(short_conv, ngram) side-state pools of a HybridReqToTokenPool or of its
    MambaPool (slot siblings); each None when absent or disabled."""
    from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

    sc = getattr(pool, "short_conv_pool", None)
    ng = getattr(pool, "ngram_pool", None)
    if sc is None and ng is None:
        for s in getattr(pool, "_slot_siblings", ()) or ():
            if isinstance(s, ShortConvPool):
                sc = s
            elif isinstance(s, NGramPool):
                ng = s
    sc = sc if sc is not None and getattr(sc, "conv_state", None) is not None else None
    ng = ng if ng is not None and getattr(ng, "context", None) is not None else None
    return sc, ng


def is_owner(pool) -> bool:
    """Does this rank run a PLE layer? Its short-conv pool is built with the
    stage's own PLE layers only (model_runner_kv_cache_mixin), the n-gram pool
    on every stage -- so the short-conv pool decides."""
    sc, _ng = ple_pools(pool)
    return sc is not None


def geometry(pool) -> Optional[Tuple[int, int, int, int, torch.dtype]]:
    """(ctx_len, n_layers, channels, window, conv dtype) or None."""
    sc, ng = ple_pools(pool)
    if sc is None:
        return None
    t = sc.conv_state  # [n_layers, slots, C, L]
    ctx = int(ng.context.shape[1]) if ng is not None else 0
    return ctx, int(t.shape[0]), int(t.shape[2]), int(t.shape[3]), t.dtype


def row_bytes(pool) -> int:
    """Bytes of one slot's rows (n-gram history + short-conv window)."""
    g = geometry(pool)
    if g is None:
        return 0
    ctx, layers, ch, win, dt = g
    return ctx * 8 + layers * ch * win * torch.empty((), dtype=dt).element_size()


def _idx(slot, device) -> torch.Tensor:
    t = slot if torch.is_tensor(slot) else torch.as_tensor(slot)
    return t.reshape(-1).to(device=device, dtype=torch.long, non_blocking=True)


def _to_host(t: torch.Tensor) -> torch.Tensor:
    """Async D2H into pinned memory on the current stream (CPU: a copy)."""
    pin = t.is_cuda
    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
    h.copy_(t, non_blocking=pin)
    return h


def snapshot(pool, slot, host=_to_host) -> Optional[Rows]:
    """The PLE rows of ``slot`` gathered on the CURRENT stream into host
    tensors; None on a rank without a PLE layer."""
    sc, ng = ple_pools(pool)
    if sc is None:
        return None
    rows: Rows = {"conv": host(sc.conv_state.index_select(1, _idx(slot, sc.conv_state.device)))}
    if ng is not None:
        rows["ngram"] = host(ng.context.index_select(0, _idx(slot, ng.context.device)))
    return rows


def refusal(pool, rows: Optional[Rows], n: int = 1) -> str:
    """'' when ``rows`` fit this pool for ``n`` slots, else why not."""
    if rows is None:
        return "absent"
    sc, ng = ple_pools(pool)
    if sc is None:
        return "not_owner"
    conv = rows.get("conv")
    want = (int(sc.conv_state.shape[0]), n) + tuple(int(d) for d in sc.conv_state.shape[2:])
    if conv is None or tuple(conv.shape) != want or conv.dtype != sc.conv_state.dtype:
        return f"conv_shape:{None if conv is None else list(conv.shape)}!={list(want)}"
    ctx = rows.get("ngram")
    if ng is not None:
        want_c = (n, int(ng.context.shape[1]))
        if ctx is None or tuple(ctx.shape) != want_c:
            return f"ngram_shape:{None if ctx is None else list(ctx.shape)}!={list(want_c)}"
    return ""


def install(pool, slot, rows: Rows) -> str:
    """Write ``rows`` into ``slot`` on the current stream (non-blocking H2D
    from pinned rows). '' = installed, else the refusal (nothing written)."""
    n = int(torch.as_tensor(slot).numel()) if not torch.is_tensor(slot) else int(slot.numel())
    why = refusal(pool, rows, n)
    if why:
        return why
    sc, ng = ple_pools(pool)
    dev = sc.conv_state.device
    idx = _idx(slot, dev)
    conv = rows["conv"].to(device=dev, non_blocking=True)
    for layer in range(int(sc.conv_state.shape[0])):
        sc.conv_state[layer].index_copy_(0, idx, conv[layer])
    if ng is not None:
        ng.context.index_copy_(0, _idx(slot, ng.context.device),
                               rows["ngram"].to(device=ng.context.device, dtype=ng.context.dtype, non_blocking=True))
    return ""


def digest(rows: Optional[Rows]) -> str:
    """sha1 over the rows' bytes (n-gram history first, then the window)."""
    if rows is None:
        return ""
    h = hashlib.sha1()
    for k in ("ngram", "conv"):
        t = rows.get(k)
        if t is not None:
            h.update(t.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def ctx_tokens(rows: Optional[Rows]) -> List[int]:
    t = None if rows is None else rows.get("ngram")
    return [] if t is None else [int(x) for x in t.reshape(-1).tolist()]


_LOG_N = [0]


def log_state(side: str, path: str, key: str, at: int, rows: Optional[Rows], extra: str = "") -> None:
    """One WEG2-PLE-STATE line (first 32, then every 64th): P and D print the
    same ``digest`` for the same rid/path/position when the rows arrived."""
    _LOG_N[0] += 1
    n = _LOG_N[0]
    if n > 32 and n % 64:
        return
    logger.info(
        "WEG2-PLE-STATE side=%s path=%s key=%s at=%d ctx=%s digest=%s bytes=%d%s (n=%d)",
        side, path, key, int(at), ",".join(str(x) for x in ctx_tokens(rows)) or "-", digest(rows) or "-",
        sum(int(t.numel() * t.element_size()) for t in (rows or {}).values()), extra, n,
    )


# -- D: installs the next forward applies (E1) ----------------------------------------
#: rid -> (slot index tensor, rows): queued at the E1 admission commit, written by
#: ``apply_pending`` from ``_prepare_ple_batch`` of the forward that carries the rid
PENDING: Dict[str, Tuple[torch.Tensor, Rows]] = {}
KEEP_PENDING = 16


def queue(rid: str, slot, rows: Rows) -> None:
    PENDING[str(rid)] = (torch.as_tensor(slot).reshape(-1).clone(), rows)
    while len(PENDING) > KEEP_PENDING:  # an admitted batch that never ran
        PENDING.pop(next(iter(PENDING)))


def discard(rid: str) -> None:
    PENDING.pop(str(rid), None)


def apply_pending(pool, rids: Optional[Sequence[str]]) -> int:
    """Write the queued rows of the rids in this forward (host lookups only:
    no device sync). Returns how many were installed."""
    if not PENDING or not rids:
        return 0
    n = 0
    for rid in rids:
        ent = PENDING.pop(str(rid), None)
        if ent is None:
            continue
        slot, rows = ent
        why = install(pool, slot, rows)
        if why:
            logger.warning("WEG2-PLE-STATE install refused rid=%s (%s): the slot keeps its history", rid, why)
            continue
        n += 1
    return n


def before_ple_read(pool, forward_batch) -> None:
    """``_prepare_ple_batch`` entry, armed only: join the load stream at the
    PLE layer's step (an arena load writes the side rows there), then apply
    the pending E1 rows of this forward's rids -- both before the n-gram
    history is read."""
    sc, _ng = ple_pools(pool)
    if sc is None:
        return
    wait = getattr(pool, "_wait_for_mamba_layer", None)
    if callable(wait) and getattr(sc, "layer_map", None):
        wait(min(sc.layer_map))
    apply_pending(pool, getattr(forward_batch, "rids", None))


# -- the arena side table ------------------------------------------------------------
_MAGIC = b"PLESIDE1"
_HEAD = struct.Struct("<8sqqqqqq")  # magic, slots, row_bytes, ctx, layers, channels, window
_FILE_HEADER = 4096
_ROW_HEADER = 64  # tag_lo u64, tag_hi u64, reserved


def _round64(n: int) -> int:
    return (int(n) + 63) // 64 * 64


class PleSideTable:
    """One row per arena slot beside the mamba arena file (``<path>.ple``),
    shared by every rank process of both groups. Row = 64-B header (the
    stem key of the slot the rows belong to) + n-gram history (int64) +
    short-conv window. A reader installs a row only when its tag equals the
    key of the stem the slot holds NOW, so a row of a previous occupant (or a
    rank that did not write) is never applied."""

    def __init__(self, path: str, slots: int, ctx: int, layers: int, channels: int, window: int,
                 dtype: torch.dtype, pin: bool = True):
        self.path = path
        self.slots = int(slots)
        self.ctx, self.layers, self.channels, self.window, self.dtype = int(ctx), int(layers), int(channels), int(window), dtype
        self.itemsize = torch.empty((), dtype=dtype).element_size()
        self.conv_bytes = self.layers * self.channels * self.window * self.itemsize
        self.row_bytes = _round64(_ROW_HEADER + self.ctx * 8 + self.conv_bytes)
        total = _FILE_HEADER + self.slots * self.row_bytes
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size < total:
                os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        head = _HEAD.pack(_MAGIC, self.slots, self.row_bytes, self.ctx, self.layers, self.channels, self.window)
        have = bytes(self._mm[:_HEAD.size])
        if have[:8] == _MAGIC:
            if have != head:
                raise ValueError(f"{path} holds a PLE side table of another geometry")
        else:
            self._mm[:_HEAD.size] = head
        buf = torch.frombuffer(self._mm, dtype=torch.uint8)
        self._rows = buf[_FILE_HEADER:_FILE_HEADER + self.slots * self.row_bytes].view(self.slots, self.row_bytes)
        self._tags = self._rows[:, :16].view(torch.int64)  # [slots, 2] (u64 bit patterns)
        c0 = _ROW_HEADER
        self._ctx = self._rows[:, c0:c0 + self.ctx * 8].view(torch.int64)  # [slots, ctx]
        v0 = c0 + self.ctx * 8
        self._conv = (self._rows[:, v0:v0 + self.conv_bytes].view(dtype)
                      .view(self.slots, self.layers, self.channels, self.window))
        self.pinned = False
        if pin and torch.cuda.is_available():
            from sglang.srt.mem_cache.pool_host.arena_pool import (
                _CUDA_ERROR_ALREADY_REGISTERED,
                _CUDA_HOST_REGISTER_FLAGS,
            )

            rc = int(torch.cuda.cudart().cudaHostRegister(int(buf.data_ptr()), total, _CUDA_HOST_REGISTER_FLAGS))
            self.pinned = rc in (0, _CUDA_ERROR_ALREADY_REGISTERED)
            if not self.pinned:
                logger.warning("WEG2-PLE-SIDE cudaHostRegister(%s) failed (%d): the side table stays unused", path, rc)

    @staticmethod
    def _tag(stem: str) -> Tuple[int, int]:
        from sglang.srt.mem_cache.storage.file.hicache_arena import key128

        lo, hi = key128(stem)
        # int64 bit patterns of the u64 key words
        return (lo - (1 << 64) if lo >= 1 << 63 else lo), (hi - (1 << 64) if hi >= 1 << 63 else hi)

    def usable(self) -> bool:
        return self.pinned or not torch.cuda.is_available()

    def write(self, slots: Sequence[int], stems: Sequence[str], pool, device_idx: torch.Tensor) -> int:
        """Gather the rows of ``device_idx`` into the arena ``slots``' rows on the
        CURRENT stream (the backup's write stream: its ack covers them), and tag
        each row with its slot's stem. Returns the rows issued."""
        sc, ng = ple_pools(pool)
        if sc is None or not self.usable() or not slots:
            return 0
        dev = sc.conv_state.device
        didx = device_idx.reshape(-1).to(device=dev, dtype=torch.long)
        conv = sc.conv_state.index_select(1, didx)  # [layers, n, C, L]
        ctx = ng.context.index_select(0, didx.to(ng.context.device)) if (ng is not None and self.ctx) else None
        nb = bool(self.pinned and dev.type == "cuda")
        for i, (slot, stem) in enumerate(zip(slots, stems)):
            s = int(slot)
            self._tags[s, 0], self._tags[s, 1] = self._tag(stem)
            self._conv[s].copy_(conv[:, i], non_blocking=nb)
            if ctx is not None:
                self._ctx[s].copy_(ctx[i], non_blocking=nb)
        return len(slots)

    def read_install(self, slots: Sequence[int], stems: Sequence[str], pool, device_idx,
                     sel=None) -> Tuple[int, int]:
        """Install the rows of arena ``slots`` into the device slots
        ``device_idx[sel]`` (``sel`` None = ``device_idx`` is already the
        selection) on the CURRENT stream when their tag matches the slot's
        stem now; returns (installed, stale). No host wait: the tags are
        checked on the host, the rows cross from the pinned table in one
        non-blocking H2D, and a device index is selected on the device."""
        sc, ng = ple_pools(pool)
        if sc is None or not self.usable() or not slots:
            return 0, 0
        keep = [i for i, (s, stem) in enumerate(zip(slots, stems))
                if (int(self._tags[int(s), 0]), int(self._tags[int(s), 1])) == self._tag(stem)]
        stale = len(slots) - len(keep)
        if not keep:
            return 0, stale
        dev = sc.conv_state.device
        pin = bool(self.pinned and dev.type == "cuda")
        rows = torch.tensor([int(slots[i]) for i in keep], dtype=torch.long)
        conv = torch.empty((len(keep),) + tuple(self._conv.shape[1:]), dtype=self._conv.dtype, pin_memory=pin)
        torch.index_select(self._conv, 0, rows, out=conv)
        conv_d = conv.to(dev, non_blocking=pin)
        didx = _select_on(device_idx, sel, keep, dev)
        for layer in range(int(sc.conv_state.shape[0])):
            sc.conv_state[layer].index_copy_(0, didx, conv_d[:, layer])
        if ng is not None and self.ctx:
            ctx = torch.empty((len(keep), self.ctx), dtype=torch.int64, pin_memory=pin)
            torch.index_select(self._ctx, 0, rows, out=ctx)
            cdev = ng.context.device
            ng.context.index_copy_(0, didx.to(cdev), ctx.to(cdev, non_blocking=pin).to(ng.context.dtype))
        return len(keep), stale


def _to_dev(idx: torch.Tensor, dev) -> torch.Tensor:
    """A host int64 index on ``dev`` without a host wait (pinned + non_blocking)."""
    idx = idx.to(dtype=torch.long)
    if torch.device(dev).type == "cuda" and idx.device.type == "cpu":
        return idx.pin_memory().to(dev, non_blocking=True)
    return idx.to(dev)


def _select_on(device_idx, sel, keep: Sequence[int], dev) -> torch.Tensor:
    """``device_idx[sel][keep]`` as an int64 index on ``dev``: positions are
    composed on the host, a card-resident ``device_idx`` is gathered there
    (never ``.cpu()`` of a device tensor)."""
    t = device_idx if torch.is_tensor(device_idx) else torch.as_tensor(list(device_idx), dtype=torch.long)
    t = t.reshape(-1)
    pos = torch.as_tensor(list(keep), dtype=torch.long)
    if sel is not None:
        pos = torch.as_tensor(sel, dtype=torch.long).reshape(-1).to("cpu").index_select(0, pos)
    if t.device.type == "cpu":
        return _to_dev(t.index_select(0, pos), dev)
    return t.index_select(0, _to_dev(pos, t.device)).to(device=dev, dtype=torch.long)


_TABLES: Dict[str, Optional[PleSideTable]] = {}
_TABLES_LOCK = threading.Lock()
_SIDE_N = [0, 0]


def side_table(arena, pool) -> Optional[PleSideTable]:
    """This process's side table for ``arena`` (created or opened beside it),
    None when unarmed, on a rank without a PLE layer, or on any failure
    (named once)."""
    if not enabled() or arena is None or not is_owner(pool):
        return None
    path = f"{getattr(arena, 'path', '')}.ple"
    if path == ".ple":
        return None
    with _TABLES_LOCK:
        if path in _TABLES:
            return _TABLES[path]
        tab = None
        try:
            ctx, layers, ch, win, dt = geometry(pool)
            tab = PleSideTable(path, int(arena.slots), ctx, layers, ch, win, dt)
            logger.info("WEG2-PLE-SIDE bound %s slots=%d row_bytes=%d total_mib=%.1f pinned=%s",
                        path, tab.slots, tab.row_bytes, tab.slots * tab.row_bytes / 1048576.0, tab.pinned)
        except Exception as exc:  # noqa: BLE001 -- no side table = today's arena resume, named
            logger.warning("WEG2-PLE-SIDE unavailable for %s (%s: %s)", path, type(exc).__name__, exc)
            tab = None
        _TABLES[path] = tab
        return tab


def side_write(arena, pool, slots: Sequence[int], device_idx: torch.Tensor) -> int:
    """Arena backup entry (never raises into the write path)."""
    try:
        tab = side_table(arena, pool)
        if tab is None:
            return 0
        stems = [arena.slot_stem(int(s)) for s in slots]
        n = tab.write(slots, stems, pool, device_idx)
        _SIDE_N[0] += n
        if n and (_SIDE_N[0] <= 16 or _SIDE_N[0] % 256 < n):
            logger.info("WEG2-PLE-SIDE write slots=%s stems=%s rows=%d (total %d)",
                        list(slots)[:4], [s[:24] for s in stems[:2]], n, _SIDE_N[0])
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("WEG2-PLE-SIDE write failed (%s: %s)", type(exc).__name__, exc)
        return 0


def side_read(arena, pool, slots: Sequence[int], device_idx, sel=None) -> Tuple[int, int]:
    """Arena load entry (never raises into the load path). ``device_idx`` may
    be the load's full device index (on the card or the host) with ``sel``
    the host positions of the arena rows in it."""
    try:
        tab = side_table(arena, pool)
        if tab is None:
            return 0, 0
        stems = [arena.slot_stem(int(s)) for s in slots]
        got, stale = tab.read_install(slots, stems, pool, device_idx, sel)
        _SIDE_N[1] += got
        if (got or stale) and (_SIDE_N[1] <= 16 or _SIDE_N[1] % 256 < got or stale):
            logger.info("WEG2-PLE-SIDE load slots=%d installed=%d stale=%d (total installed %d)",
                        len(slots), got, stale, _SIDE_N[1])
        return got, stale
    except Exception as exc:  # noqa: BLE001
        logger.warning("WEG2-PLE-SIDE load failed (%s: %s)", type(exc).__name__, exc)
        return 0, 0
