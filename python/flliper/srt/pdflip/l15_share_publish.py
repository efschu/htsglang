"""L15-10 S4n-a: D publishes its held KV for the waking P (hot handover).

After a successful keep arm (the hold extents survive the kv pause), each D
rank writes ONE descriptor and serves the extents' fds:

* ``<dir>/D.<rank>.json`` -- epoch, rank, pid, the owner vector, and per
  hold base: its role (``k``/``v`` of attention layer ``layer``, or
  ``mamba``), the view offset and row unit inside the base, and the hold
  extents ``[[offset, size], ...]`` in fd order; plus the held spans
  (rid, depth, global D slots, anchor slot) from the published manifest.
* ``<dir>/D.<rank>.sock`` -- a unix socket; every connection gets the JSON
  header and ALL fds (l15_hold_share.send_hold), in the descriptor's order.

The fds stay open (D owns them) until :meth:`SharePublisher.close` at the
next wake -- the extents are D's and live exactly as long as the hold.
Gated by FLLIPER_PDFLIP_L15_HOT_SHARE=1 at the call site; any failure here is a
named log line and no share (the hold itself is untouched).
"""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import Dict, List, Optional, Sequence

DEFAULT_DIR = "/dev/shm/pdflip-l15-share"


def share_dir(env) -> str:
    return str(env.get("FLLIPER_PDFLIP_L15_SHARE_DIR", DEFAULT_DIR) or DEFAULT_DIR)


def build_descriptor(*, epoch: int, rank: int, prefix: Sequence[int],
                     bases: Sequence[dict], spans) -> dict:
    """``bases``: [{role, layer, view_off, unit, extents: [[off, size]...]}]
    in the order their fds are sent; ``spans``: manifest HoldSpans."""
    from flliper.srt.pdflip.l15_tree_cand import match_key_of_rid

    n_fds = (1 + max((int(e[2]) for b in bases for e in b["extents"]
                      if len(e) > 2), default=-1)) if any(
        len(e) > 2 for b in bases for e in b["extents"]) else sum(
        len(b["extents"]) for b in bases)
    return {
        "epoch": int(epoch), "rank": int(rank), "pid": os.getpid(),
        "prefix": [int(x) for x in prefix],
        "bases": [dict(b) for b in bases],
        "spans": [{"rid": str(s.rid), "depth": int(s.depth),
                   # L15-EXTRAKEY (240): the tree tip's token digest WITHOUT its
                   # extra_key -- what P's front match compares the prompt to
                   # (pure function of the tokens: identical on every rank)
                   "match_key": match_key_of_rid(s.rid),
                   "slots": [int(x) for x in s.slots],
                   "anchor_slot": int(s.anchor_slot),
                   # L15-10c: the END anchor's canonical L2 slot -- P loads the
                   # anchor from there (all heads of its layers, a few MiB)
                   "anchor_l2_slot": int(getattr(s, "anchor_l2_slot", -1)),
                   "anchor_l2_gen": int(getattr(s, "anchor_l2_gen", -1)),
                   # L15-10c: every token's canonical L2 page (slot, gen) --
                   # P loads the cap-0 ranks' tokens from there (base64 int64)
                   "l2_slots_b64": _b64(getattr(s, "l2_slots", ())),
                   "l2_gens_b64": _b64(getattr(s, "l2_gens", ()))}
                  for s in spans],
        "n_fds": int(n_fds),
    }


def _b64(values) -> str:
    import base64

    import numpy as np

    return base64.b64encode(np.asarray(list(values), dtype="<i8").tobytes()).decode()


def unb64(text) -> list:
    """Inverse of the descriptor's ``*_b64`` int lists."""
    import base64

    import numpy as np

    if not text:
        return []
    return np.frombuffer(base64.b64decode(text), dtype="<i8").tolist()


class SharePublisher:
    """Writes the descriptor and serves header + fds to every connection."""

    def __init__(self, directory: str, rank: int, descriptor: dict,
                 fds: Sequence[int]):
        self.directory = directory
        self.rank = int(rank)
        self.descriptor = descriptor
        self.fds = list(fds)
        self.json_path = os.path.join(directory, "D.%d.json" % self.rank)
        self.sock_path = os.path.join(directory, "D.%d.sock" % self.rank)
        self._srv: Optional[socket.socket] = None
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(8)
        srv.settimeout(0.2)
        self._srv = srv
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.descriptor, fh, sort_keys=True)
        os.replace(tmp, self.json_path)
        self._thr = threading.Thread(target=self._serve, daemon=True,
                                     name="l15-share-D%d" % self.rank)
        self._thr.start()

    def update(self, **fields) -> None:
        """L15-16c: add fields (the parked rows after the release's park) to
        the served descriptor and its JSON; later connections see them."""
        d = dict(self.descriptor)
        d.update(fields)
        self.descriptor = d
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, sort_keys=True)
        os.replace(tmp, self.json_path)

    def _serve(self) -> None:
        from flliper.srt.pdflip.l15_hold_share import send_hold

        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                send_hold(conn, self.descriptor, self.fds)
            except Exception:  # noqa: BLE001 -- one bad peer must not stop the server
                pass
            finally:
                conn.close()

    def close(self) -> None:
        """At D's wake: stop serving, remove the descriptor, close the fds
        (P's imports hold their own references)."""
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        for p in (self.json_path, self.sock_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for f in self.fds:
            try:
                os.close(f)
            except OSError:
                pass
        self.fds = []


def fetch_share(directory: str, rank: int, timeout_s: float = 5.0):
    """P side: (descriptor, fds) of D rank ``rank`` (raises
    l15_hold_share.L15ShareError when nothing is published)."""
    from flliper.srt.pdflip.l15_hold_share import L15ShareError, recv_hold

    path = os.path.join(directory, "D.%d.sock" % int(rank))
    if not os.path.exists(path):
        raise L15ShareError("no hold share published for D rank %d" % rank)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect(path)
        return recv_hold(s)
    except OSError as exc:
        raise L15ShareError("hold share of D rank %d: %r" % (rank, exc)) from exc
    finally:
        s.close()


def publish_for_sched(sched, manifest, env, log) -> Optional["SharePublisher"]:
    """D side, after a successful keep arm: publish this rank's hold for the
    waking P (FLLIPER_PDFLIP_L15_HOT_SHARE=1). Bases: the pool's k_buffer[i]
    (role k, layer i), v_buffer[i] (role v), mamba temporal[i] and conv[0][i]
    -- exactly the views the retain hook armed; extents: the split hold
    extents (l15_keep_split). Any refusal: a named line, no share."""
    import torch

    from flliper.srt.pdflip import l15_keep_split, l15_shadow
    from flliper.srt.pdflip.l15_hold_share import L15ShareError, export_hold_extents

    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    wrapper = getattr(mr, "token_to_kv_pool", None)
    pool = l15_shadow.kv_pool_of(wrapper)
    # L15-10c: a cap-0 rank keeps nothing on its card -- nothing to share
    # (P loads that rank's tokens from L2); the capped ranks name it "cap0"
    try:
        from flliper.srt.distributed.utils import get_cp_token_ratios as _gr

        _tp = len(_gr() or [1])
        _rk = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
        _rg = getattr(getattr(sched, "server_args", None), "rank_gpu_id", None)
        _cards = (list(_rg) if isinstance(_rg, (list, tuple)) and len(_rg) == _tp
                  else list(range(_tp)))
        _caps = l15_shadow.caps_from_env(
            env, _tp, [l15_shadow.cell_bytes_from(pool)] * _tp, _cards)
        cap0 = [r for r in range(_tp) if int(_caps[r]) <= 0]
        if _rk in cap0:
            log("L15-SHARE rank=%d cap 0: nothing on this card to share (P "
                "loads its tokens from L2)" % _rk)
            return None
    except Exception as exc:  # noqa: BLE001 -- no cap view, publish as before
        cap0 = []
        log("L15-SHARE cap view unavailable (%s: %s)" % (type(exc).__name__, exc))
    # layers are published as GLOBAL model layer ids (P stages hold subsets):
    # the hybrid wrapper's dense index -> global id, identity without one
    amap = getattr(wrapper, "full_attention_layer_id_mapping", None) or {}
    att_gid = {int(i): int(g) for g, i in amap.items()}
    rtp = getattr(sched, "req_to_token_pool", None)
    mmap = getattr(rtp, "mamba_map", None) or {}
    mam_gid = {int(i): int(g) for g, i in mmap.items()}
    views = []  # (role, global layer id, tensor)
    for role, lst in (("k", getattr(pool, "k_buffer", None)),
                      ("v", getattr(pool, "v_buffer", None))):
        for i, t in enumerate(lst or ()):
            if isinstance(t, torch.Tensor):
                views.append((role, att_gid.get(i, i), t))
    mc = getattr(getattr(rtp, "mamba_pool", None), "mamba_cache", None)
    temp = getattr(mc, "temporal", None)
    if temp is not None:
        views += [("mamba_temporal", mam_gid.get(i, i), temp[i])
                  for i in range(int(temp.shape[0]))]
    for c, ct in enumerate(getattr(mc, "conv", None) or []):
        views += [("mamba_conv%d" % c, mam_gid.get(i, i), ct[i])
                  for i in range(int(ct.shape[0]))]
    bases, fds = [], []
    exported = {}  # base ptr -> [(off, size, fd_index)] -- one export per base
    try:
        for role, layer, t in views:
            b = t._base if t._base is not None else t
            ptr = int(b.data_ptr())
            holds = l15_keep_split._HOLD.get(ptr)
            if not holds:
                raise L15ShareError("%s layer %d: base not split" % (role, layer))
            if ptr not in exported:
                got = export_hold_extents(ptr, holds)
                exported[ptr] = []
                for o, sz, f in got:
                    exported[ptr].append((o, sz, len(fds)))
                    fds.append(f)
            off = int(t.data_ptr()) - ptr
            unit = int(t.stride(0)) * int(t.element_size())
            vend = off + int(t.numel()) * int(t.element_size())
            # only the base's hold extents inside this view, with their fd
            # index (views of one base share the fds)
            ext = [[o, sz, i] for o, sz, i in exported[ptr]
                   if o < vend and o + sz > off]
            bases.append({"role": role, "layer": layer, "view_off": off,
                          "unit": unit, "extents": ext})
    except Exception as exc:  # noqa: BLE001 -- no share, the hold stays
        for f in fds:
            try:
                os.close(f)
            except OSError:
                pass
        log("L15-SHARE refused (%s: %s) -- P reads the store as today"
            % (type(exc).__name__, exc))
        return None
    rank = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
    from flliper.srt.distributed.utils import get_cp_token_ratios

    ratios = get_cp_token_ratios() or [1]
    prefix = [0]
    for x in ratios:
        prefix.append(prefix[-1] + int(x))
    desc = build_descriptor(epoch=int(manifest.epoch), rank=rank, prefix=prefix,
                            bases=bases, spans=manifest.spans)
    desc["cap0"] = [int(r) for r in cap0]
    # L15-14b: the deposit region a running P prefill may write into
    try:
        from flliper.srt.pdflip import l15_deposit

        tp = len(prefix) - 1
        rgid = getattr(getattr(sched, "server_args", None), "rank_gpu_id", None)
        cards = (list(rgid) if isinstance(rgid, (list, tuple)) and len(rgid) == tp
                 else list(range(tp)))
        caps = l15_shadow.caps_from_env(
            env, tp, [l15_shadow.cell_bytes_from(pool)] * tp, cards)
        reg = l15_deposit.deposit_region(
            list(manifest.rows_by_rank), prefix, caps,
            int(manifest.anchor_slots), l15_keep_split.anchor_cap(env))
        if reg is not None:
            desc["deposit"] = {"e0": reg.e0, "e1": reg.e1, "a0": reg.a0,
                               "a1": reg.a1, "skip_ranks": list(reg.skip_ranks)}
    except Exception as exc:  # noqa: BLE001 -- no deposit, the share stays
        log("L15-DEPOSIT region refused (%s: %s)" % (type(exc).__name__, exc))
    pub = SharePublisher(share_dir(env), rank, desc, fds)
    pub.start()
    log("L15-SHARE published rank=%d bases=%d fds=%d spans=%d"
        % (rank, len(bases), len(fds), len(manifest.spans)))
    return pub
