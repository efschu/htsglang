#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- THE DESK REPLAY OF A WHOLE EXCHANGE LEG, six real processes.

WHY THIS EXISTS (operator rule 2026-09-12, after weg2xsn24): no further
fix-boot cycle for walls BELOW the join. Four boots each found one layer --
the rank collision (xsn22), the padded vocabulary cut (xsn23), then the source
addresses and the slot handshake (xsn24) -- and each cost a window to learn
one fact that a desk run could have produced in minutes.

WHAT IT IS. Six OS processes with the real rank roles (3 P + 3 D), the REAL
semaphore names in /dev/shm (``<epoch>-<r>-<r>-<n>-{empty,full}`` and the
diagonal's per-card twelve), the real host-slot layout, and the REAL manifests
of a boot as input. The tensor bytes are synthetic and DETERMINISTIC from
(name, rank), which is what makes the seam digest a real verdict: a byte that
took the wrong path produces the wrong digest, every time, with no GPU.

WHAT IT IS NOT. No CUDA: the device ops are the transport's own fake, which
moves real bytes through mmap and is cross-process by construction. So this
proves the PLAN, the PHASES, the HANDSHAKE and the BYTE ROUTING. It does not
prove cudaMemcpy2DAsync or VRAM residency -- those stay metal questions.

    scripts/weg2/xchg_leg_replay.py --evidence /spinning/evidence-665-f1/weg2xsn24_0912
    scripts/weg2/xchg_leg_replay.py --self-test        # synthetic manifests, no evidence tree
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import struct
import sys
import traceback

# THE CPU ARM PINS CVD="" SO IT IS HERMETIC. The CUDA arm must NOT inherit
# that: a spawn child re-imports this module, and torch loading under CVD=""
# caches "no CUDA-capable device" before any per-rank UUID can be set (rc=100
# from cudaSetDevice, measured in window dfuqb8). The parent clears CVD and
# sets this marker before spawning, so the child starts with no device policy
# and `make_ops` installs the rank's own UUID first.
if not os.environ.get("WEG2_REPLAY_CUDA"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "test", "registered", "unit", "weg2"))

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

# WARM THE PADDED-CUT IMPORT IN THE PARENT, BEFORE ANY FORK.
#
# `xchg_manifest._pad_vocab_size` imports `layers.vocab_parallel_embedding`
# lazily, and that import is only REACHED when a tensor is padded -- so the
# synthetic self-test never took it and the real manifests always did. Taken
# for the first time inside a forked child of a process that already holds
# torch, it SEGFAULTS the child: six "Segfault encountered" and
# `ranks_reported=0/6 errors=0`, an absence that reads like nothing happened.
#
# In production this does not arise (the scheduler has long imported it), so
# the lazy import stays where it is and the REPLAY warms it instead -- the
# harness adapts to the product, not the other way round.
xm.vocab_pad_unit()
xm._pad_vocab_size(1)

# THE NONCE IS THE PARENT'S, ALWAYS. Computed per process it differed in every
# spawn child (each re-imports this module and has its own pid), so the six
# ranks opened six DIFFERENT semaphore sets and every wait hit ENOENT --
# measured in window dfuqb8. The parent passes it down and the child installs
# it before anything is named.
NONCE = os.environ.get("WEG2_REPLAY_NONCE") or f"replay{os.getpid()}"
#: THE BOUNCE DEPTH-SLOT, from the environment for the same reason the device
#: budget is: a spawn child must not size it differently from the parent.
#: At 4 MiB the arm refused full-width real layers by W71 -- correctly, since
#: a layer that does not fit a slot cannot be assembled whole.
SLOTBYTES_ENV = "WEG2_REPLAY_SLOTBYTES"


def slot_bytes_cfg() -> int:
    return int(os.environ.get(SLOTBYTES_ENV, "") or (4 << 20))
#: Must stay under FakeDeviceOps' FAKE_DEV_BYTES (8 MiB per rank).
#: THE PER-RANK DEVICE BUDGET, read from the environment on every call so a
#: spawn child and the parent that budgeted for it cannot disagree.
#:
#: IT WAS A 2 MiB CONSTANT, and against the real INT4 checkpoint that silently
#: dropped every one of the 37 `weight_packed` tensors -- a 26 MiB tensor
#: cannot fit a 2 MiB rank -- so the first run on real bytes read
#: `tensors_ok=84 verdict=MATCH` while grading not a single quantised weight.
#: The CKPT-GRADED census is what showed it; the number is now a knob.
DEVBUDGET_ENV = "WEG2_REPLAY_DEVBUDGET"


def dev_budget() -> int:
    return int(os.environ.get(DEVBUDGET_ENV, "") or (2 << 20))
DEPTH = 1


def seed_bytes(name: str, nbytes: int) -> bytes:
    """Deterministic content for one tensor, from its NAME alone.

    From the name and not from the rank: a shard of ``lm_head.weight`` on D
    rank 1 must contain exactly the rows of P's ``lm_head.weight`` that the
    plan assigns to it, so both ends derive the same stream and the digest can
    tell a correct route from a plausible one.
    """
    out = bytearray()
    h = hashlib.blake2b(name.encode(), digest_size=64)
    while len(out) < nbytes:
        out += h.digest()
        h = hashlib.blake2b(h.digest(), digest_size=64)
    return bytes(out[:nbytes])


def load_manifests(evidence: str):
    mans = []
    for f in sorted(os.listdir(evidence)):
        if not (f.startswith("phase_manifest_") and f.endswith(".json")):
            continue
        with open(os.path.join(evidence, f), encoding="utf-8") as fh:
            mans.append(xm.RankManifest.from_json(json.load(fh), path=f))
    return mans


def synthetic_manifests():
    """A six-rank set shaped like the real one, for --self-test."""
    cards = tuple(range(xr.N_CARDS))
    classes = (("self_attn.qkv_proj.weight", 512, 256),
               ("mlp.down_proj.weight", 256, 384))
    cut = (2, 1, 1)
    out = []

    def piece(n, r, c):
        return xm.ManifestPiece(param_name=n, tensor_class=n.rsplit(".", 2)[-2],
                                rows_full=r, cols_full=c, itemsize=1,
                                tag="weights_0", nbytes=r * c)

    def stage(layer):
        acc = 0
        for st, k in enumerate(cut):
            acc += k
            if layer < acc:
                return st
        raise AssertionError

    for pp in range(3):
        ps = [piece(f"model.layers.{l}.{sfx}", r, c)
              for l in range(sum(cut)) for sfx, r, c in classes
              if stage(l) == pp]
        out.append(xm.RankManifest(group="P", rank=pp, card=cards[pp],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=0, pp_rank=pp, pieces=tuple(ps)))
    for t in range(3):
        ps = []
        for l in range(sum(cut)):
            for sfx, r, c in classes:
                w = [r // 3] * 3
                w[-1] += r - sum(w)
                ps.append(piece(f"model.layers.{l}.{sfx}", w[t], c))
        out.append(xm.RankManifest(group="D", rank=t, card=cards[t],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=t, pp_rank=0, pieces=tuple(ps)))
    return out


# ---------------------------------------------------------------------------
# THE REAL CHECKPOINT -- bytes that exist before this script runs.
# ---------------------------------------------------------------------------
#
# USER ORDER 2026-09-13: the test vehicle is RedHatAI/Qwen3.8-27B-INT4,
# UNCHANGED. This reads the safetensors file and never writes it.
#
# THE FOUR CLASSES THE CHECKPOINT ACTUALLY CARRIES, measured from the header
# of model.safetensors (2016 tensors, dtypes I64 400 / I32 400 / BF16 1216):
#
#   weight_packed  I32  [out_features, in_features // 8]
#       compressed-tensors pack-quantized W4A16: eight int4 weights per int32
#       word, packed ALONG THE INPUT AXIS. The packing axis is the COLUMN
#       axis here, which is what makes a row cut safe and a column cut a
#       nibble-splitting operation (see `refuse_packed_column_cut`).
#   weight_scale   BF16 [out_features, in_features // 128]
#       one scale per group of 128 input columns (g128).
#   weight_shape   I64  [2]
#       the LOGICAL shape the packed words decode to. Replicated.
#   plain BF16          norms, A_log, dt_bias, conv1d, in_proj_a/b, embed,
#       lm_head -- the GDN and normalisation weights, unquantised.
#
# NO `g_idx` EXISTS IN THIS CHECKPOINT. The order named actorder and g_idx as
# a class to declare; the header carries 400 weight_packed, 400 weight_scale
# and 400 weight_shape and no g_idx tensor at all, so the actorder permutation
# is not materialised here. Naming that is the point of a declared census: a
# class I had silently not implemented would read the same as a class that is
# not there.
#
# THE CHECKPOINT LAYOUT IS THE SOURCE, NOT THE VRAM LAYOUT. Marlin repacking
# happens in the loader on the way to the device; what this arm moves and
# grades is the file's own byte order, which is the thing a flip would have to
# preserve.
CKPT_ENV = "WEG2_REPLAY_CKPT"


def _one_header(path: str):
    """(name -> entry, data_start) for ONE safetensors file, without torch."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head, 8 + n


def checkpoint_header(path: str):
    """The tensor index of a checkpoint -- ONE FILE OR A SHARDED DIRECTORY.

    The production vehicle (user order 2026-09-13, superseding the INT4 one)
    is sharded over 18 files with a `model.safetensors.index.json`, so a
    reader that only knows a single file would have refused the very
    checkpoint the xsn24 manifests belong to. Each entry carries the file it
    lives in and that file's data start, so the byte range is absolute.
    """
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".safetensors"))
        if not files:
            raise SystemExit(f"no safetensors under {path}")
        merged = {}
        for fn in files:
            full = os.path.join(path, fn)
            head, start = _one_header(full)
            for k, v in head.items():
                v = dict(v)
                v["_file"], v["_start"] = full, start
                # ONE NAME, ONE FILE. A duplicate across shards would mean two
                # different byte ranges answer to one name and the digest
                # would grade whichever won -- named, not last-wins.
                if k in merged:
                    raise SystemExit(
                        f"{k} appears in two shards ({merged[k]['_file']} and "
                        f"{full}); the checkpoint index is ambiguous")
                merged[k] = v
        return merged, 0
    head, start = _one_header(path)
    for v in head.values():
        v["_file"], v["_start"] = path, start
    return head, start


#: Bytes per element, per safetensors dtype string.
CKPT_ITEMSIZE = {"I64": 8, "I32": 4, "BF16": 2, "F32": 4, "F16": 2,
                 "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def ckpt_geom(entry):
    """(rows, cols, itemsize) for one checkpoint tensor.

    A 1-D tensor is a single row, and a conv1d's [C, 1, K] is C rows of K --
    the trailing axes are folded into the column axis, which is exactly what
    the row cut needs and what ``StorageGeom`` reads off a live tensor.
    """
    shape = [int(x) for x in entry["shape"]]
    it = CKPT_ITEMSIZE[str(entry["dtype"])]
    if len(shape) == 1:
        return 1, shape[0], it
    rows = shape[0]
    cols = 1
    for d in shape[1:]:
        cols *= d
    return rows, cols, it


#: One scale group is 128 logical input weights; the packed representation
#: holds 8 of those per int32 word, so ONE GROUP IS 16 PACKED COLUMNS.
PACK_PER_WORD = 8
GROUP_INPUTS = 128
WORDS_PER_GROUP = GROUP_INPUTS // PACK_PER_WORD


def refuse_packed_column_cut(name: str, widths, *, packed: bool) -> None:
    """A COLUMN cut of the quantised classes must fall on a scale group.

    THE PREDICATE, stated rather than assumed. ``weight_packed`` holds eight
    int4 weights per int32 word along the INPUT axis, and ``weight_scale``
    holds one scale per 128 inputs -- so a packed column boundary is only
    decodable if it is a multiple of 16 words (= 128 inputs = one group).
    A finer boundary puts one group's inputs on two ranks and NEITHER rank can
    dequantise its own slice: the bytes would arrive intact and decode to
    nonsense, which is the failure mode a byte digest cannot see.

    THIS ARM CUTS ROWS (the output axis), where the question does not arise.
    The predicate exists so that a later arm which cuts the other way meets a
    refusal instead of a silent corruption -- and it is exercised in both
    directions by ``--self-test`` rather than merely declared. The first draft
    of it read ``acc % 1 != 0``, which is never true: a check that cannot go
    red is not a check.
    """
    step = WORDS_PER_GROUP if packed else 1
    acc = 0
    ws = [int(w) for w in widths]
    for w in ws[:-1]:
        acc += w
        if acc % step:
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {name} is quantised along its "
                f"column axis; a cut at column {acc} is not a multiple of "
                f"{step} (one scale group of {GROUP_INPUTS} inputs). One "
                f"group's inputs would land on two ranks and neither could "
                f"dequantise its own slice."
            )


def _selfcheck_column_predicate() -> None:
    """The predicate BOTH WAYS, on the path that runs."""
    refuse_packed_column_cut("t.weight_packed", (32, 16, 16), packed=True)
    try:
        refuse_packed_column_cut("t.weight_packed", (30, 18, 16), packed=True)
    except wx.Weg2XchgPlanDisagree:
        return
    raise SystemExit("refuse_packed_column_cut did not refuse a cut at "
                     "column 30 -- the predicate cannot go red")


def checkpoint_manifests(path: str, max_bytes: int, skip=None):
    """Six manifests over REAL tensors of the real checkpoint.

    P holds whole tensors, one PP stage each; D holds a ROW shard of every
    tensor. Same shape as ``synthetic_manifests``, with the geometry and the
    names read from the file instead of invented -- so the join, the plan and
    the digest all run on the checkpoint's own classes.
    """
    head, _ = checkpoint_header(path)
    cards = tuple(range(xr.N_CARDS))
    skipped = collections.Counter()
    # WHOLE LAYERS, IN ORDER -- never a name-sorted greedy fill. The first
    # version took tensors in sorted name order and the budget filled up with
    # `weight_shape` (81 of 149 tensors) while the load-bearing packed class
    # got 6: a population that is 54 % two-element metadata grades almost
    # nothing of what the flip actually has to move. A layer is all-or-nothing
    # so every class this checkpoint has is walked at its real proportion.
    by_layer = collections.defaultdict(list)
    for name in sorted(head):
        if ".layers." not in name:
            skipped["not-a-layer-tensor"] += 1
            continue
        rows, cols, it = ckpt_geom(head[name])
        by_layer[int(name.split(".layers.")[1].split(".")[0])].append(
            (name, rows, cols, it, rows * cols * it, str(head[name]["dtype"])))
    chosen, total = [], 0
    for lay in sorted(by_layer):
        want = by_layer[lay]
        cost = sum(x[4] for x in want)
        if total + cost > max_bytes:
            skipped["over-byte-budget-whole-layer"] += len(want)
            continue
        chosen.extend(want)
        total += cost
    _selfcheck_column_predicate()
    if not chosen:
        raise SystemExit("checkpoint selection empty -- budget too small")

    layers = sorted({int(n.split(".layers.")[1].split(".")[0])
                     for n, *_ in chosen})
    # CONTIGUOUS PP STAGES over the layers actually selected, in the tree's
    # own 44/10/10 proportion (the P split this campaign boots with).
    def stage_of(layer: int) -> int:
        """Contiguous PP stages, EVERY STAGE NON-EMPTY.

        The first version applied the boot's 44/10/10 proportion directly and
        at three selected layers stage 2 came out empty -- the rank then wrote
        no manifest and the join refused all six legs
        (`join-manifest-missing`, group rank 2). The proportion is a property
        of the 64-layer model, not of a byte-budgeted subset; what the subset
        must preserve is that all three stages carry work, or the replay
        grades a two-rank exchange.
        """
        i_ = layers.index(layer)
        n = len(layers)
        base, rem = divmod(n, xr.N_CARDS)
        if base == 0:
            raise SystemExit(
                f"checkpoint selection has {n} layers for {xr.N_CARDS} PP "
                f"stages -- raise --ckpt-bytes; a stage with no tensors makes "
                f"the join refuse every leg")
        # Front-heavy like the boot's own split: the remainder goes to stage 0.
        first = base + (1 if rem > 0 else 0)
        second = base + (1 if rem > 1 else 0)
        if i_ < first:
            return 0
        return 1 if i_ < first + second else 2

    def piece(name, rows, cols, it):
        return xm.ManifestPiece(
            param_name=name, tensor_class=name.rsplit(".", 1)[-1],
            rows_full=rows, cols_full=cols, itemsize=it,
            tag="weights_0", nbytes=rows * cols * it)

    out = []
    for pp in range(xr.N_CARDS):
        ps = [piece(n, r, c, it) for (n, r, c, it, _, _dt) in chosen
              if stage_of(int(n.split(".layers.")[1].split(".")[0])) == pp]
        out.append(xm.RankManifest(group="P", rank=pp, card=cards[pp],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=0, pp_rank=pp, pieces=tuple(ps)))
    for t in range(xr.N_CARDS):
        ps = []
        for (n, r, c, it, _, _dt) in chosen:
            if r < xr.N_CARDS:
                ps.append(piece(n, r, c, it))          # replicated whole
                continue
            w = [r // xr.N_CARDS] * xr.N_CARDS
            w[-1] += r - sum(w)
            ps.append(piece(n, w[t], c, it))
        out.append(xm.RankManifest(group="D", rank=t, card=cards[t],
                                   region_tag="weights_0", boot_token="rp",
                                   tp_rank=t, pp_rank=0, pieces=tuple(ps)))
    return out, chosen, skipped


class _CkptReader:
    """The checkpoint's bytes for one tensor, by name.  Read-only, mmapped."""

    def __init__(self, path: str):
        self.path = path
        self.head, _ = checkpoint_header(path)
        self._fh = {}

    def whole(self, name: str) -> bytes:
        e = self.head[name]
        a, b = (int(x) for x in e["data_offsets"])
        fh = self._fh.get(e["_file"])
        if fh is None:
            fh = self._fh[e["_file"]] = open(e["_file"], "rb")
        fh.seek(int(e["_start"]) + a)
        return fh.read(b - a)


_CKPT_READER = [None]


def ckpt_reader():
    """This process's reader, or None when the arm runs on seeded bytes."""
    path = os.environ.get(CKPT_ENV, "")
    if not path:
        return None
    if _CKPT_READER[0] is None:
        _CKPT_READER[0] = _CkptReader(path)
    return _CKPT_READER[0]


def narrow(mans, col_div: int):
    """Scale COLUMNS for materialisation -- never rows, never the padding.

    THE BYTE BUDGET, and it is deliberately the one axis that is not under
    test. What the replay grades is the ROW mapping: which rows of the PP-side
    tensor land on which TP rank, and what happens to the padded tail. Columns
    are payload width. Scaling them uniformly on BOTH sides leaves the join's
    axis decision, the shard vector and ``pad_units`` bit-for-bit identical
    while turning a 2.54 GiB ``lm_head.weight`` into something six processes
    can hold.

    ONLY WHERE THE COLUMN COUNTS ALREADY AGREE, so a COLUMN cut is never
    touched -- scaling one there would break the sum the join checks, i.e. it
    would change the arithmetic instead of the budget.
    """
    if int(col_div) <= 1:
        return mans
    cols = {}
    for m in mans:
        for p in m.pieces:
            cols.setdefault(p.param_name, set()).add(int(p.cols_full))
    scalable = {n for n, c in cols.items() if len(c) == 1}
    out = []
    for m in mans:
        ps = []
        for p in m.pieces:
            if p.param_name in scalable:
                c = max(1, int(p.cols_full) // int(col_div))
                ps.append(xm.ManifestPiece(
                    param_name=p.param_name, tensor_class=p.tensor_class,
                    rows_full=p.rows_full, cols_full=c, itemsize=p.itemsize,
                    tag=p.tag, nbytes=p.rows_full * c * p.itemsize))
            else:
                ps.append(p)
        out.append(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(ps)))
    return out


BOOT_TOKEN = "replay"


def card_uuids():
    """The three cards' NVML UUIDs, in NVML order. Never torch enumeration.

    The fork's own rule (device identity): PyTorch's ordering and NVML's can
    diverge, so a rank is pinned by UUID and isolated at the PROCESS level --
    `CUDA_VISIBLE_DEVICES=<uuid>` before anything CUDA is touched, after which
    `cuda:0` is unambiguous inside that process.
    """
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30)
    return [u.strip() for u in out.stdout.splitlines() if u.strip()]


def make_ops(root: str, rank: int, group: str, cuda: bool, uuid: str = ""):
    """The ONE place the two arms differ -- that is the whole design.

    `FakeDeviceOps` and `CudartDeviceOps` implement the same `tp.DeviceOps`
    surface (`raw_malloc`, `memcpy_async`, `memcpy2d_async`, `create_stream`,
    `synchronize`, `host_register`), so join, plan, phase split, rendezvous,
    budget and region cut are untouched by the arm.
    """
    if not cuda:
        from test_weg2_xchg_transport_1273 import FakeDeviceOps

        return FakeDeviceOps(root, rank=(rank if group == "D" else 3 + rank))
    # CVD IS SET BEFORE THE CUDART HANDLE EXISTS. With `spawn` the child has
    # imported nothing CUDA yet, so this is the process-level isolation the
    # device-identity rule asks for rather than an in-process mapping table.
    if uuid:
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    from sglang.srt.weg2 import weight_exchange_transport as _tp

    return _tp.CudartDeviceOps()


def device_read(ops, ptr: int, nbytes: int, cuda: bool) -> bytes:
    """Bytes back from wherever they live -- host mmap, or the card."""
    if not cuda:
        return ctypes.string_at(ops.real(ptr), nbytes)
    buf = (ctypes.c_char * nbytes)()
    stream = ops.create_stream(0)
    try:
        ops.memcpy_async(ctypes.addressof(buf), int(ptr), int(nbytes), stream)
        ops.synchronize(stream)
    finally:
        ops.destroy_stream(stream)
    return bytes(buf)


def device_write(ops, ptr: int, payload: bytes, cuda: bool) -> None:
    if not cuda:
        ctypes.memmove(ops.real(ptr), payload, len(payload))
        return
    buf = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
    stream = ops.create_stream(0)
    try:
        ops.memcpy_async(int(ptr), ctypes.addressof(buf), len(payload), stream)
        ops.synchronize(stream)
    finally:
        ops.destroy_stream(stream)


def expected_bytes(t, group: str, rank: int, nbytes: int) -> bytes:
    """What THIS rank must hold of one tensor -- the one definition both
    directions share.

    The whole tensor's content is `seed_bytes(name)`; a TP rank holds the
    slice of it its shard vector names. Seeding the SOURCE and verifying the
    DESTINATION from the same function is what makes a direction swap a flag
    rather than a second expectation to keep in step.
    """
    rows_content = t.rows_full - t.pad_units
    want_len = rows_content * t.cols_full * t.itemsize
    rd = ckpt_reader()
    # THE SOURCE OF TRUTH FOR THE CONTENT. With a checkpoint configured the
    # whole tensor IS the file's bytes, so the digest compares against the
    # safetensors file and not against another expression of this script.
    whole = rd.whole(t.param_name)[:want_len] if rd is not None else \
        seed_bytes(t.param_name, want_len)
    if group == "P":
        return whole[:nbytes]
    # A REPLICATED TENSOR IS HELD WHOLE BY EVERY TP RANK -- it is not cut, so
    # `tp_widths` is not a shard vector for it (the join records (1, 1, 1)).
    # WITHOUT THIS BRANCH the ROWS formula below took that (1,1,1) as row
    # counts and computed `start = rank * cols_full * itemsize`, which for
    # rank 1 and 2 lies PAST THE END of a one-row tensor: `want` came back
    # EMPTY and the comparison passed on zero bytes. Measured: the pp_to_tp
    # arm read MATCH 302/302 on D while every replicated tensor of ranks 1
    # and 2 was graded vacuously, and only the reverse direction -- where the
    # PP destination's expectation is non-empty -- could ever show it. A
    # checker that cannot go red is the defect, not the direction.
    if t.shard_axis == wx.REPLICATED:
        return whole[:nbytes]
    if t.shard_axis == wx.COLS:
        w = t.tp_widths[rank]
        off = sum(t.tp_widths[:rank]) * t.itemsize
        pitch = t.cols_full * t.itemsize
        run = w * t.itemsize
        return b"".join(whole[i * pitch + off: i * pitch + off + run]
                        for i in range(rows_content))[:nbytes]
    start = sum(t.tp_widths[:rank]) * t.cols_full * t.itemsize
    return whole[start:start + nbytes]


def rank_extents(t, group: str, rank: int):
    """(rows, cols) THIS rank holds of one joined tensor -- AXIS-AWARE.

    The first version took ``tp_widths[rank]`` as the ROW count for every
    class, which is right for a row cut and wrong for a column cut. The
    product's own materialisation check caught it immediately and by name:
    ``A_log: the manifest records (1, 30) ... and the materialised tensor
    holds (30, 48)``. That guard is doing exactly what it was wired for -- the
    manifest is what every other reader plans from -- so the fixture was the
    thing that had drifted.
    """
    if group == "P":
        rows = t.rows_full - (t.pad_units if t.shard_axis == wx.ROWS else 0)
        cols = t.cols_full - (t.pad_units if t.shard_axis == wx.COLS else 0)
        return max(rows, 1), max(cols, 1)
    if t.shard_axis == wx.COLS:
        return max(t.rows_full, 1), max(t.tp_widths[rank], 1)
    if t.shard_axis == wx.ROWS:
        return max(t.tp_widths[rank], 1), max(t.cols_full, 1)
    return max(t.rows_full, 1), max(t.cols_full, 1)


def budget_manifests(mans, max_tensors: int):
    """Drop tensors that do not fit, BY NAME, identically on all six ranks.

    THE COST IS COMPUTED FROM THE JOIN, not estimated from the manifests, and
    that correction is the second one this budget needed. A per-manifest
    estimate under-counted a D rank by 8x (`allocation 16032228 B over budget
    2097152 B at visual.blocks.0.attn.qkv_proj.bias`) because what a rank
    actually allocates is a function of the JOIN's extents -- the padded total,
    the per-rank width -- and not of any single manifest row. Budgeting on a
    quantity that is not the one allocated is the same defect twice.

    SELECTED IN THE PARENT so the six children cannot disagree about the
    population: six plans that differ is the one thing this slice prevents.
    """
    # THE SAME REGION FILTER AS THE LEG AND THE CHILD. Joining unfiltered now
    # REFUSES by name (`lm_head.weight` has no counterpart in P's draft
    # region), which is the identity cut working: a draft tensor needs a draft
    # source, and that is the DRAFT leg's job.
    mans = [xm.RankManifest(
        group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
        boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
        pieces=tuple(p_ for p_ in m.pieces
                     if xm.region_of_tag(p_.tag) == "weights"))
        for m in mans]
    mans = [m for m in mans if m.pieces]
    join = xm.join_manifests(mans, pp_group="P", tp_group="D")

    def rank_costs(t):
        """(worst P stage cost, worst D rank cost) for one tensor."""
        pr, pc = rank_extents(t, "P", 0)
        whole = pr * pc * t.itemsize
        shard = max(
            rank_extents(t, "D", r)[0] * rank_extents(t, "D", r)[1] * t.itemsize
            for r in range(len(t.tp_widths)))
        return max(whole, 1), max(shard, 1)

    # STRATIFIED BY SOURCE RANK, and this is the defect the train seat's call
    # exposed. Picking globally cheapest-first put the whole subset on ONE P
    # stage: the DEFAULT form read
    #   `the joined plan has 72 descriptors and none with src_rank=2`
    # on P ranks 1 and 2 -- `ranks_reported=4/6` -- while my own call with
    # `--col-div 512` saw all three stages and read MATCH. Two invocations of
    # one script, and the SELECTION was the whole difference. A filter that can
    # starve a rank grades a lane it never exercised.
    #
    # Round-robin over the stages keeps every source rank represented, which is
    # what makes a six-rank replay a six-rank replay.
    by_stage = {}
    for t in join.tensors:
        by_stage.setdefault(int(t.pp_stage), []).append(t)
    for st in by_stage:
        by_stage[st].sort(key=lambda t: sum(rank_costs(t)))
    keep, p_used, d_used, dropped = set(), {}, 0, 0
    order = sorted(by_stage)
    idx = {st: 0 for st in order}
    while True:
        progressed = False
        for st in order:
            if idx[st] >= len(by_stage[st]):
                continue
            t = by_stage[st][idx[st]]
            idx[st] += 1
            progressed = True
            w, sh = rank_costs(t)
            # BOTH SIDES BOUNDED, SEPARATELY: a P stage holds whole tensors of
            # its OWN layers (cap per stage) while a D rank holds a shard of
            # EVERY tensor (cap global).
            if (p_used.get(st, 0) + w > dev_budget()
                    or d_used + sh > dev_budget()):
                dropped += 1
                continue
            p_used[st] = p_used.get(st, 0) + w
            d_used += sh
            keep.add(t.param_name)
            if max_tensors and len(keep) >= max_tensors:
                progressed = False
                break
        if not progressed:
            break
    p_used_max = max(p_used.values()) if p_used else 0

    # COVERAGE PER RANK AND PER DIRECTION, because those are the two things a
    # leg needs to exist at all. Under `pp_to_tp` the SOURCE ranks are the P
    # stages; under `tp_to_pp` they are the D ranks, which hold a shard of
    # every kept tensor.
    kept_t = [t for t in join.tensors if t.param_name in keep]
    stages_full = sorted({int(t.pp_stage) for t in join.tensors})
    stages_kept = sorted({int(t.pp_stage) for t in kept_t})
    d_full = sorted(range(len(join.cards)))
    d_kept = sorted({r for t in kept_t for r in range(len(t.tp_widths))
                     if t.tp_widths[r] > 0})
    if stages_kept != stages_full or d_kept != d_full:
        raise SystemExit(
            f"WEG2-XCHG-LEG-REPLAY REFUSED reason=subset-starves-a-rank "
            f"subset covers pp_to_tp src ranks {set(stages_kept)} of "
            f"{set(stages_full)} and tp_to_pp src ranks {set(d_kept)} of "
            f"{set(d_full)}; kept={len(keep)} of {len(join.tensors)} "
            f"cap={dev_budget()} max_tensors={max_tensors} -- a source rank "
            f"with no descriptor cannot deposit, so the run would grade four "
            f"ranks of six and call it a result. THIS IS THE SELECTION'S "
            f"DEFECT, NOT THE PLAN'S: the full join covers {set(stages_full)}. "
            f"Raise --max-tensors or lower --col-div.")
    print(f"  BUDGET kept={len(keep)} dropped={dropped} of {len(join.tensors)} "
          f"pp_to_tp_src_ranks={stages_kept}/{stages_full} "
          f"tp_to_pp_src_ranks={d_kept}/{d_full} p_bytes_max={p_used_max} "
          f"d_bytes={d_used} cap={dev_budget()} -- the dropped tensors are "
          f"NOT byte-routed here; their byte path stays a metal question")
    out = []
    for m in mans:
        out.append(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(p for p in m.pieces if p.param_name in keep)))
    return out


class _FakeParam:
    """A tensor as the product path reads one: address AND geometry.

    ``data_ptr()`` is what the address books ask; ``shape``/``stride()``/
    ``element_size()`` are what `refuse_on_materialisation_drift` reads through
    `StorageGeom.of`. A double with only the pointer made every rank refuse
    with ``derivation-failed: '_FakeParam' object has no attribute 'shape'`` --
    which is the materialisation check doing its job on a fixture that had not
    materialised anything.
    """

    def __init__(self, ptr: int, rows: int, cols: int, itemsize: int):
        self._ptr = int(ptr)
        self.shape = (int(rows), int(cols))
        self._stride = (int(cols), 1)
        self._itemsize = int(itemsize)

    def data_ptr(self) -> int:
        return self._ptr

    def stride(self):
        return self._stride

    def element_size(self) -> int:
        return self._itemsize

    def dim(self) -> int:
        return 2


class _FakeModel:
    def __init__(self, params):
        self._p = list(params)

    def named_parameters(self):
        return list(self._p)


class _FakeRunner:
    def __init__(self, model):
        self.model = model


class _FakeWorker:
    def __init__(self, model):
        self.model_runner = _FakeRunner(model)


class _FakeDraftWorker:
    """The shape ``_get_draft_model_runner`` resolves."""

    def __init__(self, model):
        self.draft_model_runner = _FakeRunner(model)


class _Stub:
    """Borrows the REAL unbound methods -- nothing is re-implemented.

    ``draft_params=None`` is the CAN-FAIL ARM: it reproduces the single-runner
    address book weg2xsn24 shipped, where ``fc.weight`` had no home and the
    leg refused with W74 at ``dst_resolved=894/904``.
    """

    def __init__(self, main_params, draft_params, group="P", rank=0):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        self._group, self._rank = str(group), int(rank)
        self.tp_worker = _FakeWorker(_FakeModel(main_params))
        self.draft_worker = (None if draft_params is None
                             else _FakeDraftWorker(_FakeModel(draft_params)))
        cls = wu.SchedulerWeightUpdaterManager
        for name in ("_weg2_rank_param_table", "_weg2_join_src_addr",
                     "_weg2_join_dst_addr", "_weg2_shadow_plan",
                     "_weg2_xchg_bounce_leg"):
            setattr(type(self), name, getattr(cls, name))

    # #1358: the adapter reads its own identity for the host-slot lines.
    def _weg2_group_name(self):
        return self._group

    def _weg2_rank(self):
        return self._rank


def rank_proc(group, rank, evidence, root, q, self_test, max_tensors,
              col_div, single_runner, manifest_dir, cuda=False,
              uuid="", direction="pp_to_tp"):
    """ONE rank, THROUGH THE PRODUCT ADAPTER.

    THE ENTRY POINT IS THE BOOT'S, and that is the whole point of this
    revision. The first version called ``wb.run_bounce_leg`` with an address
    book the replay built itself, so it graded the library and left the two
    things weg2xsn24 actually died on -- the adapter's two-runner address
    table and its lazy import -- UNTESTED at the desk. Now the stub borrows the
    real unbound methods (`_weg2_shadow_plan`, `_weg2_rank_param_table`, the
    two address books, `_weg2_xchg_bounce_leg`) and the bytes travel the path a
    flip leg travels.
    """
    try:
        os.environ[xm.DIR_ENV] = manifest_dir
        os.environ[xr.ENV_REGION_BOOT] = BOOT_TOKEN
        os.environ["SGLANG_WEG2_WEIGHT_SOURCE"] = "exchange"
        os.environ["SGLANG_WEG2_GROUP"] = group

        mans = xm.load_manifests(manifest_dir, boot_token=BOOT_TOKEN)
        # THE SAME REGION FILTER THE LEG APPLIES. Without it the replay's own
        # join (which sizes the tensors and computes the expectation) saw BOTH
        # runners while the PLAN saw one -- and for the eight names both
        # runners carry, the expectation described a different tensor than the
        # plan moved. Measured as `tensors_bad=2` on exactly
        # model.layers.0.{q,k}_norm.weight, the drafter's own.
        mans = [xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(p_ for p_ in m.pieces
                         if xm.region_of_tag(p_.tag) == "weights"))
            for m in mans]
        mans = [m for m in mans if m.pieces]
        join = xm.join_manifests(mans, pp_group="P", tp_group="D")

        ops = make_ops(root, rank, group, cuda, uuid)

        # THE TWO RUNNERS, as the real rank has them. The draft head's tensors
        # (`fc.weight` and friends) live ONLY in the draft runner -- which is
        # exactly why a single-runner address book resolved 894 of 904 and
        # refused the leg.
        mine = [m for m in mans if m.group == group and m.rank == rank]
        # WHICH RUNNER A TENSOR BELONGS TO IS THE FILE IT CAME FROM, not the
        # tag on the piece. The can-fail arm deliberately re-tags the draft
        # PIECES into the main region (so the region cut cannot separate them,
        # which is the pre-fix state) while the FILE keeps its own region_tag
        # -- so this is the reading that still places them in the draft runner,
        # where a single-runner address book cannot reach them.
        draft_names = {p.param_name for m in mine
                       if str(m.region_tag) == "weights_draft"
                       for p in m.pieces}
        # In the can-fail arm the draft pieces were re-tagged into the
        # main region, so `draft_names` is empty and every piece lands in
        # the MAIN double -- except the ones the single-runner book drops.

        # WHICH GROUP HOLDS THE BYTES AT THE START -- needed before the
        # allocation loop, because it decides who is seeded and who is zeroed.
        src_group = "P" if direction == "pp_to_tp" else "D"
        main_params, draft_params, own = [], [], {}
        allocated = 0
        for t in join.tensors:
            if group == "P" and t.pp_stage != rank:
                continue
            rows, cols = rank_extents(t, group, rank)
            nb = max(rows * cols * t.itemsize, 1)
            allocated += nb
            if allocated > dev_budget():
                # THE CHILD'S OWN CEILING, named rather than asserted deep in
                # the fake device. The parent budgets on a per-tensor cost; if
                # that estimate is ever short, this says so with the number
                # instead of dying as `fake device out of memory` mid-leg.
                q.put(("error", group, rank,
                       f"allocation {allocated} B over budget "
                       f"{dev_budget()} B at {t.param_name} -- the "
                       f"parent's per-tensor cost under-counted this rank",
                       "", 0))
                q.put(("done", group, rank, 0, 0, 0))
                return
            ptr = ops.raw_malloc(0, nb)
            own[t.param_name] = (ptr, nb)
            if group == src_group:
                device_write(ops, ptr,
                             expected_bytes(t, group, rank, nb), cuda)
            else:
                device_write(ops, ptr, b"\0" * nb, cuda)
            # A TENSOR WHOSE data_ptr() IS THE FAKE DEVICE ADDRESS: the address
            # books call `.data_ptr()`, so the double must answer it.
            param = _FakeParam(ptr, rows, cols, t.itemsize)
            (draft_params if t.param_name in draft_names
             else main_params).append((t.param_name, param))

        stub = _Stub(main_params, None if single_runner else draft_params,
                     group=group, rank=rank)

        hook = "source" if group == src_group else "destination"
        plan, reason = stub._weg2_shadow_plan(
            hook, group, rank, agreed=None, require_agreement=False)
        if plan is None:
            q.put(("error", group, rank, f"no plan: {reason}", "", 0))
            q.put(("done", group, rank, 0, 0, 0))
            return
        prof = wx.pointer_profile(plan.descs)
        q.put(("profile", group, rank, prof.src_resolved, prof.dst_resolved,
               prof.descs_total))
        q.put(("alloc", group, rank, allocated, len(own), 0))

        sems = tp.SemSet(NONCE)
        # #1358: the PRODUCT's own host-slot lines, forwarded verbatim, so the
        # replay reads the same form a boot does instead of a line of its own.
        def _capture(line: str) -> None:
            text = str(line)
            if text.startswith(wb.HOST_SLOT_MARKER):
                q.put(("hostslot", group, rank, text, 0, 0))

        stub._weg2_xchg_bounce_leg(
            descs=list(plan.descs), ops=ops, boot_nonce=NONCE,
            slot_bytes=slot_bytes_cfg(), depth=DEPTH,
            mode=wx.INJECT_AUTHORITATIVE, shm_root=root, device=0,
            hook=hook, region=None, sems=sems)
        q.put(("hostslotleg", group, rank,
               wb.host_slot_leg_line(group=group, rank=rank,
                                     leg=f"{NONCE}/{hook}"), 0, 0))
        q.put(("entry", group, rank, "adapter", 0, 0))

        if group != src_group:
            good = bad = 0
            bad_names = []
            for t in join.tensors:
                # A PP DESTINATION HOLDS ONLY ITS OWN STAGE'S TENSORS, so the
                # verdict asks only about what this rank actually received.
                if t.param_name not in own:
                    continue
                ptr, nb = own[t.param_name]
                want = expected_bytes(t, group, rank, nb)
                if len(want) != nb:
                    # AN EXPECTATION THAT IS NOT THE RANK'S OWN SIZE GRADES
                    # NOTHING. Named as a failure rather than counted as a
                    # pass -- this is exactly how the replicated defect above
                    # stayed invisible for a whole direction.
                    bad += 1
                    if len(bad_names) < 3:
                        bad_names.append(
                            f"{t.param_name}(expectation {len(want)}B != "
                            f"held {nb}B)")
                    continue
                got = device_read(ops, ptr, len(want), cuda)
                if got == want:
                    good += 1
                else:
                    bad += 1
                    if len(bad_names) < 3:
                        bad_names.append(t.param_name)
                    if len(bad_names) == 1 and os.environ.get("WEG2_DIAG"):
                        fd = next((i for i in range(min(len(got), len(want)))
                                   if got[i] != want[i]), -1)
                        q.put(("badnames", group, rank,
                               f"DIAG {t.param_name} axis={t.shard_axis} "
                               f"rows={t.rows_full} cols={t.cols_full} "
                               f"pad={t.pad_units} widths={t.tp_widths} "
                               f"nb={nb} first_diff={fd} "
                               f"zeros={got.count(0)}/{len(got)} "
                               f"got_is_shard0={got[:32] == want[:32]} "
                               f"got_head={got[:16].hex()} "
                               f"want_head={want[:16].hex()}", 0, 0))
            q.put(("digest", group, rank, good, bad, len(join.tensors)))
            if bad_names:
                q.put(("badnames", group, rank, ",".join(bad_names), 0, 0))
        q.put(("done", group, rank, 0, 0, 0))
    except BaseException as exc:  # noqa: BLE001
        q.put(("error", group, rank, f"{type(exc).__name__}: {exc}",
               traceback.format_exc()[-900:], 0))
        q.put(("done", group, rank, 0, 0, 0))


def _run(ns) -> int:

    if not ns.self_test and not os.path.isdir(ns.evidence):
        print(f"WEG2-XCHG-LEG-REPLAY REFUSED: no evidence dir {ns.evidence}")
        return 2

    root = f"/dev/shm/weg2-xchg-{NONCE}"
    os.makedirs(root, exist_ok=True)
    # CREATED AFTER THE BUDGET, and that ordering is a leak I caused:
    # `budget_manifests` can `SystemExit` on a starved subset, which skips
    # every line below it -- so a REFUSAL left 36 named semaphores in
    # /dev/shm. A refusal that leaks is a refusal that costs the next run.
    # The teardown below is in a `finally` for the same reason.
    print(f"WEG2-XCHG-LEG-REPLAY nonce={NONCE} slot_bytes={slot_bytes_cfg()} "
          f"depth={DEPTH} max_tensors={ns.max_tensors} col_div={ns.col_div} "
          f"source={'synthetic' if ns.self_test else ns.evidence}")

    # THE PARENT WRITES THE MANIFESTS ONCE, narrowed and budgeted, under ONE
    # boot token -- the children then read them exactly as a rank reads its
    # boot's, through `manifests_for_boot`, which is the product's own entry.
    mdir = os.path.join(root, "manifests")
    os.makedirs(mdir, exist_ok=True)
    if ns.slot_bytes:
        os.environ[SLOTBYTES_ENV] = str(int(ns.slot_bytes))
    if ns.dev_budget:
        os.environ[DEVBUDGET_ENV] = str(int(ns.dev_budget))
    ckpt_pick, ckpt_skip = None, None
    if ns.checkpoint:
        # THE CHECKPOINT IS READ-ONLY AND ITS PATH IS INHERITED, never
        # recomputed per child -- the same reason WEG2_REPLAY_NONCE is.
        os.environ[CKPT_ENV] = ns.checkpoint
        src_mans, ckpt_pick, ckpt_skip = checkpoint_manifests(
            ns.checkpoint, int(ns.ckpt_bytes))
    elif ns.self_test:
        _selfcheck_column_predicate()
        src_mans = synthetic_manifests()
    else:
        src_mans = load_manifests(ns.evidence)
    if ckpt_pick is not None:
        # CLASS *AND* DTYPE. Keyed on the suffix alone, the INT8 vehicle's
        # quantised `weight` (I8) and its bf16 `weight` (norms, conv1d,
        # in_proj_a/b) counted as one class of 94 -- and the whole point of the
        # census is to say whether the QUANTISED bytes were walked.
        cls = collections.Counter(f"{n.rsplit('.', 1)[-1]}:{dt}"
                                  for (n, _r, _c, _i, _b, dt) in ckpt_pick)
        print("WEG2-XCHG-CKPT-POPULATION file=" + os.path.basename(ns.checkpoint)
              + " tensors=" + str(len(ckpt_pick))
              + " bytes=" + str(sum(x[4] for x in ckpt_pick))
              + " classes=" + ",".join(f"{k}:{v}" for k, v in sorted(cls.items()))
              + " skipped=" + ",".join(f"{k}:{v}" for k, v in sorted(ckpt_skip.items()))
              + " -- the classes this run WALKED and the ones it did not, with "
                "the reason; a skip that is not named is a silent hole in the "
                "denominator")
    src_mans = narrow(src_mans, ns.col_div)
    # THE CAN-FAIL RE-TAG RUNS BEFORE THE BUDGET, and the ordering is the whole
    # arm: the budget now filters by region, so a re-tag applied afterwards
    # would find the draft pieces already gone and the arm would read MATCH --
    # a can-fail that cannot fail.
    if ns.single_runner:
        src_mans = [xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=tuple(
                xm.ManifestPiece(
                    param_name=p_.param_name, tensor_class=p_.tensor_class,
                    rows_full=p_.rows_full, cols_full=p_.cols_full,
                    itemsize=p_.itemsize, tag="weights", nbytes=p_.nbytes)
                for p_ in m.pieces))
            for m in src_mans]
    src_mans = budget_manifests(src_mans, ns.max_tensors)
    if ckpt_pick is not None:
        # THE POPULATION THAT IS ACTUALLY GRADED, after the per-rank device
        # budget has had its say. The selection line above says what was read
        # out of the file; this one says what survived to the seam. Measured
        # gap on the first run: 157 selected, 84 joined -- 73 tensors dropped
        # by a budget that printed nothing, which is precisely the unnamed
        # hole in a denominator this campaign keeps paying for.
        kept = {p_.param_name for m in src_mans for p_ in m.pieces}
        dts = {n: dt for (n, _r, _c, _i, _b, dt) in ckpt_pick}
        final = collections.Counter(f"{n.rsplit('.', 1)[-1]}:{dts[n]}"
                                    for n in kept if n in dts)
        gone = collections.Counter(f"{n.rsplit('.', 1)[-1]}:{dt}"
                                   for (n, _r, _c, _i, _b, dt) in ckpt_pick
                                   if n not in kept)
        print("WEG2-XCHG-CKPT-GRADED tensors=" + str(len(kept))
              + " classes=" + ",".join(f"{k}:{v}" for k, v in sorted(final.items()))
              + " dropped_by_device_budget="
              + (",".join(f"{k}:{v}" for k, v in sorted(gone.items())) or "none")
              + " -- this is the tensors_ok denominator; every name outside it "
                "was named above")
    for m in src_mans:
        # THE FILE NAME KEEPS ITS OWN region_tag: the region cut reads the
        # PIECE's tag, while the name's third axis exists so two runners of one
        # rank cannot collide (weg2xsn20/22).
        xm.write_rank_manifest(xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=BOOT_TOKEN, tp_rank=m.tp_rank, pp_rank=m.pp_rank,
            pieces=m.pieces), mdir)
    print(f"  manifests={len(src_mans)} entry={ns.entry} "
          f"address_book="
          f"{'single-runner (CAN-FAIL ARM)' if ns.single_runner else 'both runners'}")

    # CREATED ONLY ONCE THE SUBSET IS ACCEPTED. `budget_manifests` can
    # `SystemExit` on a starved subset, and creating the handshake before that
    # point leaked 36 named semaphores per refusal -- a refusal that leaks is a
    # refusal that costs the next run.
    xr.create_semaphores(NONCE)
    # SPAWN FOR THE CUDA ARM, fork otherwise. A forked child inherits whatever
    # the parent already imported; the CUDA arm must set CUDA_VISIBLE_DEVICES
    # BEFORE anything CUDA exists in that process, which only a fresh
    # interpreter guarantees.
    os.environ["WEG2_REPLAY_NONCE"] = NONCE
    if ns.cuda:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["WEG2_REPLAY_CUDA"] = "1"
    uuids = card_uuids() if ns.cuda else []
    if ns.cuda and len(uuids) < xr.N_CARDS:
        raise SystemExit(
            f"WEG2-XCHG-LEG-REPLAY REFUSED reason=cards-unresolved "
            f"nvml returned {len(uuids)} uuids, need {xr.N_CARDS} -- the arm "
            f"pins a rank by UUID and never by torch enumeration")
    ctx = mp.get_context("spawn" if ns.cuda else "fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=rank_proc,
                         args=(g, r, ns.evidence, root, q, ns.self_test,
                               ns.max_tensors, ns.col_div, ns.single_runner,
                               mdir, ns.cuda,
                               uuids[r] if ns.cuda else "", ns.direction))
             for g in ("P", "D") for r in range(xr.N_CARDS)]
    for p in procs:
        p.start()
    seen, errors = [], []
    alive = len(procs)
    while alive:
        try:
            kind, g, r, a, b, c = q.get(timeout=300)
        except Exception:
            break
        if kind == "error":
            errors.append((g, r, a, b))
            alive -= 1
        elif kind == "done":
            alive -= 1
        else:
            seen.append((kind, g, r, a, b, c))
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    for kind, g, r, a, b, c in sorted(seen):
        if kind == "profile":
            print(f"  WEG2-XCHG-POINTER-PROFILE group={g} rank={r} "
                  f"src_resolved={a}/{c} dst_resolved={b}/{c}")
        elif kind == "legs":
            print(f"  WEG2-XCHG-LEGS group={g} rank={r} legs={a}")
        elif kind == "alloc":
            print(f"  WEG2-XCHG-ALLOC group={g} rank={r} bytes={a} tensors={b}")
        elif kind in ("hostslot", "hostslotleg"):
            print(f"  {a}")
        elif kind == "entry":
            print(f"  WEG2-XCHG-ENTRY group={g} rank={r} entry={a}")
        elif kind == "slot":
            print(f"  WEG2-XCHG-HOST-SLOT group={g} rank={r} {a}")
        elif kind == "badnames":
            print(f"    first mismatches group={g} rank={r}: {a}")
        elif kind == "digest":
            v = "MATCH" if b == 0 and a else ("MISMATCH" if b else "NO-COMPARE")
            print(f"  WEG2-XCHG-SEAM-DIGEST group={g} rank={r} "
                  f"tensors_ok={a} tensors_bad={b} of={c} verdict={v}")
    for g, r, msg, tb in errors:
        print(f"  ERROR group={g} rank={r}: {msg}")
        print("    " + tb.replace("\n", "\n    ")[-700:])

    digests = [s for s in seen if s[0] == "digest"]
    ok = (not errors and digests
          and all(b == 0 and a for _k, _g, _r, a, b, _c in digests))
    print(f"WEG2-XCHG-LEG-REPLAY verdict={'MATCH' if ok else 'FAIL'} "
          f"ranks_reported={len({(s[1], s[2]) for s in seen})}/6 "
          f"errors={len(errors)}")

    return 0 if ok else 1


def _teardown(root: str) -> None:
    """Remove EVERYTHING this run made, on every exit path.

    IN A `finally` BECAUSE A REFUSAL IS AN EXIT PATH TOO: `budget_manifests`
    raises `SystemExit` on a starved subset, and with the teardown inline each
    refusal left its root directory (and, before the ordering fix, 36 named
    semaphores) behind. A refusal that leaks is a refusal that costs the next
    run -- the same class as a test that does not sweep its own shm.
    """
    import shutil

    try:
        xr.unlink_semaphores(NONCE)
    except BaseException:  # noqa: BLE001
        pass
    try:
        wb.BounceSlots(NONCE, shm_root=root).unlink()
    except BaseException:  # noqa: BLE001
        pass
    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="/spinning/evidence-665-f1/weg2xsn24_0912")
    ap.add_argument("--direction", choices=("pp_to_tp", "tp_to_pp"),
                    default="pp_to_tp",
                    help="which way the bytes move. pp_to_tp seeds P and "
                         "collects on D; tp_to_pp is the mirror. The "
                         "acceptance is MATCH in BOTH.")
    ap.add_argument("--cuda", action="store_true",
                    help="THE CUDA ARM: six processes on the three REAL cards "
                         "(one P and one D rank per card), CUDA_VISIBLE_DEVICES "
                         "per process resolved by NVML UUID, real device "
                         "memory, real bounce slot, real semaphores. Same "
                         "pipeline as the CPU arm -- join, plan, phase split, "
                         "rendezvous, digest -- because the arm is only the "
                         "DeviceOps implementation. No host ring, no HiCache, "
                         "no L2, no front, no launcher ledger.")
    ap.add_argument("--checkpoint", default="",
                    help="path to a safetensors file whose REAL bytes seed "
                         "and grade the seam (user order 2026-09-13: "
                         "RedHatAI/Qwen3.8-27B-INT4, read-only)")
    ap.add_argument("--slot-bytes", type=int, default=0,
                    help="bounce depth-slot bytes (default 4 MiB); must cover "
                         "the widest layer unit or the arm refuses by W71")
    ap.add_argument("--dev-budget", type=int, default=0,
                    help="per-rank device byte budget (default 2 MiB); the "
                         "INT4 arm needs room for whole packed tensors")
    ap.add_argument("--ckpt-bytes", type=int, default=192 << 20,
                    help="byte budget for the checkpoint tensor selection")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--single-runner", action="store_true",
                    help="CAN-FAIL ARM: build the address book from the MAIN "
                         "runner only, as weg2xsn24 shipped it. The draft "
                         "head then has no address and the leg must refuse "
                         "with W74 on fc.weight (dst_resolved=894/904)")
    ap.add_argument("--entry", choices=("adapter", "library"),
                    default="adapter",
                    help="which entry point the bytes travel; the boot's is "
                         "the adapter")
    ap.add_argument("--col-div", type=int, default=1,
                    help="divide every tensor's COLUMN count by N for "
                         "materialisation, uniformly on both sides and only "
                         "where the column count already agrees (i.e. never on "
                         "a column cut). The ROW vector and the padding are "
                         "untouched, so the routing under test is unchanged; "
                         "this is a byte budget, not a change of arithmetic")
    ap.add_argument("--max-tensors", type=int, default=0,
                    help="materialise at most N tensors (0 = all that fit the byte "
                         "budget, which IS the default form the xsn25 "
                         "acceptance cites); "
                         "the JOIN always sees every one")
    ns = ap.parse_args()
    root = f"/dev/shm/weg2-xchg-{NONCE}"
    try:
        return _run(ns)
    except SystemExit as exc:
        # A NAMED REFUSAL, printed as one line and exited 2 -- not a traceback,
        # and not a silent 0.
        print(str(exc))
        return 2
    finally:
        _teardown(root)


if __name__ == "__main__":
    sys.exit(main())
