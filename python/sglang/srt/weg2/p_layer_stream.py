"""#1962 P-LAYER-STREAM -- dual P, PP0: weights out, KV in, staged by the prompt's level.

Goal (user order 05.10.): Dual-P prefills 262144 tokens. Measured/modelled in
deskq/done/1959 + 1960: the PP0 card (K0, 5090) needs 5.998 GB of KV for level
266240 at 11 full-attention layers, its card pool holds 3.33 GB (2.79 after the
ratchet), and with D running (~1.25 GB of D rows on K0) no static cut carries
262144 (O3 34,15,15 tops out near 193k, r30 near 250k, at -29/-40 % P speed).

What this module does -- the dense-model analogue of NF's "experts out, KV in"
(``weg2/d_seat_vram`` + ``d_mem_sched``): when PP0's atomic group grant is short
ONLY on PP0's own card, PP0 pauses whole WEIGHT-CHUNK TAGS of its P-private
layer bytes (``weights_<k>``, the per-layer TMS tags the flip already uses; on
the dual P each holds ~0.75 GiB, 6 of them on PP0 -- f9 P.log ``WEG2-TAG-POOL
occupancy tag=weights_0..5 active_gib=0.75``), lends the freed bytes to the card
pool and from then on streams those layers' tensors from pinned host images,
``prefetch`` layers ahead on a side stream, swapped in by a forward pre-hook on
the hull's decoder layer and swapped back by its post-hook. The number of paused
units follows the level the prompt asks for: a prompt whose level fits pauses
nothing (the default path runs), a 262144 prompt pauses as many units as its
deficit needs.

What it is NOT: no KV layout change, no rung change of the PP cut, no new
collective. The math is identical (the same tensors, only their residence
moves), so the outputs are identical by construction; only PP0 changes
behaviour, and only its timing and its memory. Rank agreement is trivial: the
decision is taken by PP0 alone on PP0's own card (the grant is PP0's already).

Invariants enforced here rather than documented:

* a unit is streamable only if EVERY live block of its tag pool is the storage
  of a tensor this module found on a module of the hull or of a P part, at a
  parsable layer index (``coverage``). A tensor nobody knows would be read at a
  paused (unmapped) address: that unit is refused at build, never paused;
* the pause and the resume of an owned tag happen ONLY through this module; the
  memory-saver adapter's ordinary pause/resume skip an owned tag (the P sleep
  leg would otherwise pause it twice -- both legs are not idempotent,
  weight_updater.py #1285);
* graphs captured before the pause read the paused addresses: while any unit is
  paused PP0's prefill graph runner runs eager (``force_eager``); after the
  regain the VA is the same and the graphs are valid again;
* the loan comes back only through ``CardKvLedger.reclaim`` (free covers it) and
  a verified resume; when D grew into it, P keeps streaming (P never presses D).

Default off (``SGLANG_WEG2_DUAL_P_LAYER_STREAM`` unset): nothing here is built,
no hook is installed and the adapter's ``owns`` answers False from an empty set.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import os
import re
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARK = "#1962 P-LAYER-STREAM"
ENV = "SGLANG_WEG2_DUAL_P_LAYER_STREAM"
PREFETCH_ENV = "SGLANG_WEG2_DUAL_P_LAYER_STREAM_PREFETCH"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

#: the installed streamer of this process (PP0 of a dual P), or None
_STREAMER: Optional["LayerStreamer"] = None
#: tags paused BY this module right now -- what the adapter must not touch
_OWNED: set = set()
#: True only while this module itself calls the adapter's pause/resume
_ACTING = [False]
#: levels whose NO-PLAN line was written (log rate only)
_NOPLAN_SAID: set = set()


def armed(env: Optional[Mapping[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    if str(e.get(ENV, "") or "").strip().lower() not in ("1", "true", "yes", "y"):   # EnvBool's own set
        return False
    if str(e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() != "1":
        return False
    return str(e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "P"


def prefetch_layers(env: Optional[Mapping[str, str]] = None) -> int:
    e = os.environ if env is None else env
    try:
        return max(1, int(str(e.get(PREFETCH_ENV, "") or "2").strip()))
    except ValueError:
        return 2


def active() -> Optional["LayerStreamer"]:
    return _STREAMER


def install(streamer: Optional["LayerStreamer"]) -> None:
    global _STREAMER
    _STREAMER = streamer


def reset_for_tests() -> None:
    global _STREAMER
    _STREAMER = None
    _OWNED.clear()
    _ACTING[0] = False
    _NOPLAN_SAID.clear()


def owns(tag) -> bool:
    """The memory-saver adapter's question: is ``tag`` paused by the streamer
    (and is the caller someone else)? False whenever nothing is paused."""
    return bool(_OWNED) and not _ACTING[0] and str(tag) in _OWNED


def force_eager() -> bool:
    """The prefill graph runner's question: True while a unit is paused (a graph
    would read the paused addresses). False in every process that never paused."""
    return bool(_OWNED)


# ----------------------------------------------------------------------------- the pure rule


def plan_units(units: Sequence[Tuple[str, int]], paused: Iterable[str], deficit: int,
               staging: int) -> List[str]:
    """The additional units to pause for ``deficit`` bytes, in the given order.

    ``units``: (tag, bytes a pause frees), in the order the caller wants them
    paused (the tail of the stage first). ``staging``: the bytes the prefetch
    ring holds while anything is paused -- charged once, when the first unit is
    paused. Returns [] when nothing is needed (deficit <= 0) or when even every
    remaining unit cannot cover it (then pausing would only slow P down without
    granting the request: the request waits as before).
    """
    deficit = int(deficit)
    if deficit <= 0:
        return []
    done = set(paused)
    need = deficit + (0 if done else max(0, int(staging)))
    out: List[str] = []
    got = 0
    for tag, nbytes in units:
        if tag in done:
            continue
        out.append(tag)
        got += max(0, int(nbytes))
        if got >= need:
            return out
    return []


def level_need_on_card0(stage: Mapping, level_tokens: int, covered0: int) -> int:
    """What ``group_grant`` asks of PP0's card for ``level_tokens`` (same rule)."""
    step = int(stage["step"])
    k = min(int(stage["top"]), (int(level_tokens) + step - 1) // step * step) // step
    return max(0, int(stage["bytes"][k]) - int(covered0))


# ----------------------------------------------------------------------------- catalog


@dataclasses.dataclass
class StreamUnit:
    tag: str
    nbytes: int
    #: layer index -> the tensor OBJECTS whose storage lives in this tag's pool
    tensors: Dict[int, List[object]]

    @property
    def layers(self) -> Tuple[int, ...]:
        return tuple(sorted(self.tensors))


def _module_tensors(mod) -> Iterable[object]:
    import torch

    seen = set()
    for t in list(getattr(mod, "_parameters", {}).values()) + list(getattr(mod, "_buffers", {}).values()):
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            yield t
    for v in list(vars(mod).values()):
        vals = v if isinstance(v, (list, tuple)) else (v,)
        for t in vals:
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                yield t


def _on_device(t) -> bool:
    return getattr(t, "device", None) is not None and t.device.type == "cuda"


def catalog(named_modules: Iterable[Tuple[str, object]],
            segments: Mapping[str, Sequence[Mapping]],
            is_device: Callable[[object], bool] = _on_device,
            forbidden: Mapping[int, str] = None) -> Tuple[List[StreamUnit], Dict[str, str]]:
    """Units from the tag pools' live blocks and the tensors found on modules.

    ``segments``: tag -> the pool's snapshot segments (``address``,
    ``total_size``, ``blocks`` with ``size``/``state`` and optionally
    ``address``). ``forbidden``: storage base -> why a reader OUTSIDE the
    forward touches it (the sleep leg's static-state export of the hull's
    buffers, the NVFP4 Marlin->native conversion); a unit holding one is refused
    -- that reader would hit an unmapped page while the unit is paused.
    Returns (streamable units ordered tail-first, {tag: refusal}).
    """
    forbidden = forbidden or {}
    spans: List[Tuple[int, int, str]] = []
    live: Dict[str, Dict[int, int]] = {}
    total: Dict[str, int] = {}
    for tag, segs in segments.items():
        live[tag] = {}
        total[tag] = 0
        for seg in segs or ():
            a = int(seg.get("address", 0))
            n = int(seg.get("total_size", 0))
            total[tag] += n
            spans.append((a, a + n, tag))
            off = a
            for blk in seg.get("blocks", ()):
                size = int(blk.get("size", 0))
                addr = int(blk.get("address", off))
                if blk.get("state") == "active_allocated":
                    live[tag][addr] = size
                off = addr + size
    spans.sort()
    starts = [s[0] for s in spans]

    def tag_of(ptr: int) -> Optional[str]:
        i = bisect.bisect_right(starts, ptr) - 1
        if i >= 0 and spans[i][0] <= ptr < spans[i][1]:
            return spans[i][2]
        return None

    found: Dict[str, Dict[int, List[object]]] = {t: {} for t in segments}
    bases: Dict[str, set] = {t: set() for t in segments}
    refused: Dict[str, str] = {}
    # (tensor, layer) pairs: a tensor object several layers' modules hold (a shared workspace) is swapped at
    # EACH of those layers -- deduplicated per layer, never "first layer wins" (the others would read the pause)
    seen = set()
    for name, mod in named_modules:
        m = _LAYER_RE.search(str(name))
        li = int(m.group(1)) if m is not None else None
        for t in _module_tensors(mod):
            if (id(t), li) in seen or not is_device(t):
                continue
            try:
                base = int(t.untyped_storage().data_ptr())
            except Exception:  # noqa: BLE001
                continue
            tag = tag_of(base)
            if tag is None:
                continue
            seen.add((id(t), li))
            bases[tag].add(base)
            if li is None:
                refused.setdefault(tag, "tensor without a layer index on module %r" % (name,))
                continue
            found[tag].setdefault(li, []).append(t)
    units = []
    for tag in segments:
        hit = sorted(b for b in bases[tag] if b in forbidden)
        if hit and tag not in refused:
            refused[tag] = "holds %d storage(s) a reader outside the forward touches (%s)" % (
                len(hit), forbidden[hit[0]])
        missing = [a for a in live[tag] if a not in bases[tag]]
        if missing and tag not in refused:
            refused[tag] = "coverage: %d of %d live blocks (%d B) belong to no known tensor" % (
                len(missing), len(live[tag]), sum(live[tag][a] for a in missing))
        if not found[tag] and tag not in refused:
            refused[tag] = "no tensor of a known layer lives in this pool"
        if tag in refused:
            continue
        units.append(StreamUnit(tag, total[tag], found[tag]))
    units.sort(key=lambda u: -max(u.tensors))
    return units, refused


def uncatalogued_holders(units: Sequence[StreamUnit], live_bases: Mapping[str, Iterable[int]],
                         objects: Iterable[object], is_device: Callable[[object], bool] = _on_device
                         ) -> Dict[str, int]:
    """tag -> how many Python tensor OBJECTS outside the catalog hold storage of that
    unit's live blocks (a cached ``.T``, a dict entry, a closure). Such an object is
    not swapped by the hooks and would be read at a paused address -> the caller
    refuses the unit. Objects only C++ holds stay invisible (the CUDA smoke)."""
    import torch

    known = {id(t) for u in units for ts in u.tensors.values() for t in ts}
    owner: Dict[int, str] = {}
    for tag, bs in live_bases.items():
        for b in bs:
            owner[int(b)] = tag
    out: Dict[str, int] = {}
    for o in objects:
        if not isinstance(o, torch.Tensor) or id(o) in known or not is_device(o):
            continue
        try:
            b = int(o.untyped_storage().data_ptr())
        except Exception:  # noqa: BLE001
            continue
        tag = owner.get(b)
        if tag is not None:
            out[tag] = out.get(tag, 0) + 1
    return out


def _rounded(nbytes: int) -> int:
    """What the caching allocator really takes for one tensor: >1 MiB -> 2 MiB blocks, else 512 B."""
    n = max(0, int(nbytes))
    g = (2 << 20) if n > (1 << 20) else 512
    return (n + g - 1) // g * g


# ----------------------------------------------------------------------------- the actuator


class LayerStreamer:
    """PP0's streamed layers: pause/regain whole units, swap tensors per layer."""

    def __init__(self, units: Sequence[StreamUnit], *, pause: Callable[[str], None],
                 resume: Callable[[str], None], prefetch: int = 2, device=None,
                 phys_free: Optional[Callable[[], Optional[int]]] = None,
                 pin: bool = True, sync: Optional[Callable[[], None]] = None,
                 tag_mapped: Optional[Callable[[str], Optional[int]]] = None,
                 empty_cache: Optional[Callable[[], None]] = None):
        import torch

        self._torch = torch
        self.units = list(units)
        self._by_tag = {u.tag: u for u in self.units}
        self._pause = pause
        self._resume = resume
        self.prefetch = max(1, int(prefetch))
        self._device = device
        self._phys_free = phys_free
        self._pin = bool(pin)
        self._sync = sync or (lambda: None)
        self._tag_mapped = tag_mapped
        self._empty_cache = empty_cache or (lambda: None)
        #: consumed staging sets whose layer may still run on the GPU: (devs, done event, bytes) -- held until
        #: the event is done, so the ring is BOUNDED by back-pressure, never by how far the CPU runs ahead
        import collections

        self._inflight = collections.deque()
        self._executing: Dict[int, Tuple[List[object], int]] = {}
        self._live_bytes = 0
        #: levels a stream-out was already done for (no cascade while D takes the loan)
        self.levels_streamed: set = set()
        self._cuda = device is not None and getattr(device, "type", str(device)).startswith("cuda")
        self._side = torch.cuda.Stream(device=device) if self._cuda else None
        #: tag -> bytes that pause freed (what the regain must reclaim)
        self.freed: Dict[str, int] = {}
        #: tag -> bytes LENT to the card pool for it (the first unit's loan is net of the staging ring,
        #: which the prefetch keeps on the card while anything is paused)
        self.loan: Dict[str, int] = {}
        #: layer -> [(tensor, its original data, pinned host image)]
        self._images: Dict[int, List[Tuple[object, object, object]]] = {}
        self._staged: Dict[int, Tuple[List[object], object]] = {}
        self._order: List[int] = []
        self.counters = {"out": 0, "regain": 0, "staged": 0, "staged_late": 0, "swapped": 0, "waited": 0,
                         "peak_live_sets": 0, "peak_live_bytes": 0}
        layer_bytes = {}
        for u in self.units:
            for li, ts in u.tensors.items():
                layer_bytes[li] = layer_bytes.get(li, 0) + sum(_rounded(int(t.numel()) * int(t.element_size()))
                                                               for t in ts)
        self.max_layer_bytes = max(layer_bytes.values(), default=0)

    # -- state ---------------------------------------------------------------
    def paused(self) -> Tuple[str, ...]:
        return tuple(t for t in (u.tag for u in self.units) if t in self.freed)

    def lent(self) -> int:
        return int(sum(self.loan.values()))

    def staging_bytes(self) -> int:
        """The ring's bound: ``prefetch`` staged + the executing layer + one whose
        GPU work is still in flight (back-pressure keeps it there), allocator-rounded."""
        return int((self.prefetch + 2) * self.max_layer_bytes)

    def room(self) -> int:
        """What pausing every remaining unit could still free (net of the ring
        when nothing is paused yet) -- the infeasibility check's extra."""
        rest = sum(u.nbytes for u in self.units if u.tag not in self.freed)
        return max(0, int(rest) - (0 if self.freed else self.staging_bytes()))

    def plan(self, deficit: int) -> List[str]:
        return plan_units([(u.tag, u.nbytes) for u in self.units], self.freed, deficit,
                          self.staging_bytes())

    # -- out / regain --------------------------------------------------------
    def _image(self, t):
        torch = self._torch
        if self._pin:
            from sglang.srt.layers.moe.expert_offload import pinned_exact_empty   # page-locked at its exact size

            host = pinned_exact_empty(tuple(t.size()), t.dtype)
        else:
            host = torch.empty(tuple(t.size()), dtype=t.dtype, device="cpu")
        host.copy_(t)
        return host

    def stream_out(self, tags: Sequence[str]) -> int:
        """Pause ``tags``; returns the bytes freed (measured when a reading
        exists, else the units' pool bytes). A unit whose pause freed nothing
        is resumed at once and does not count."""
        total = 0
        for tag in tags:
            u = self._by_tag.get(tag)
            if u is None or tag in self.freed:
                continue
            self._sync()
            images = {li: [(t, t.data, self._image(t)) for t in ts] for li, ts in u.tensors.items()}
            self._sync()
            # the instrument is THIS process's own mapped bytes of the tag (no D noise on the shared card); only
            # without it the device free delta, capped at the unit's pool (D freeing meanwhile cannot inflate it)
            m0 = self._tag_mapped(tag) if self._tag_mapped is not None else None
            before = self._phys_free() if (self._phys_free is not None and m0 is None) else None
            _ACTING[0] = True
            try:
                self._pause(tag)
            finally:
                _ACTING[0] = False
            m1 = self._tag_mapped(tag) if m0 is not None else None
            after = self._phys_free() if before is not None else None
            if m0 is not None and m1 is not None:
                freed = min(int(m0) - int(m1), int(u.nbytes))
            elif before is not None and after is not None:
                freed = min(int(after) - int(before), int(u.nbytes))
            else:
                freed = int(u.nbytes)
            if freed <= 0:
                _ACTING[0] = True
                try:
                    self._resume(tag)
                finally:
                    _ACTING[0] = False
                logger.warning("%s OUT-REFUSED tag=%s freed=%d B -- the pause gave the card nothing; the unit is "
                               "resumed and stays resident", MARK, tag, freed)
                continue
            self.freed[tag] = freed
            _OWNED.add(tag)
            for li, rows in images.items():          # a layer may have tensors in two units: merge, never replace
                self._images.setdefault(li, []).extend(rows)
                self._staged.pop(li, None)            # a copy staged for the old row list would miss the new rows
            self._order = sorted(self._images)
            self.counters["out"] += 1
            total += freed
            logger.warning("%s OUT tag=%s layers=%s freed=%d B (pool %d B) -- PP0 streams these layers' tensors "
                           "per forward from host, prefetch=%d", MARK, tag, list(u.layers), freed, u.nbytes,
                           self.prefetch)
        if total:
            self._prime()
        return total

    def regain(self, tag: str) -> int:
        """Resume ``tag`` (the caller reclaimed its loan first); returns the
        bytes back on the card. A refused resume raises (the adapter verifies)."""
        if tag not in self.freed:
            return 0
        u = self._by_tag[tag]
        self.release_ring()
        mine = {id(t) for ts in u.tensors.values() for t in ts}
        _ACTING[0] = True
        try:
            self._resume(tag)
        finally:
            _ACTING[0] = False
        # the content comes from OUR host image, never from the saver's backup alone (a weights-resident
        # profile pauses without one): copy back, then point the tensors at their storage again
        done = set()
        for li in u.tensors:
            for t, orig, host in self._images.get(li, ()):
                if id(t) in mine and id(t) not in done:
                    orig.copy_(host)
                    done.add(id(t))
        self._sync()
        for li in u.tensors:
            for t, orig, _h in self._images.get(li, ()):
                if id(t) in mine:
                    t.data = orig
        for li in u.tensors:
            rest = [row for row in self._images.get(li, ()) if id(row[0]) not in mine]
            if rest:
                self._images[li] = rest
            else:
                self._images.pop(li, None)
        self._order = sorted(self._images)
        n = self.freed.pop(tag)
        self.loan.pop(tag, None)
        _OWNED.discard(tag)
        if not self.freed:
            self.levels_streamed.clear()
        self.counters["regain"] += 1
        logger.warning("%s REGAIN tag=%s layers=%s %d B -- resident again (graphs valid: same VA)", MARK, tag,
                       list(u.layers), n)
        return n

    # -- per forward ---------------------------------------------------------
    def _set_bytes(self, devs) -> int:
        return sum(_rounded(int(d.numel()) * int(d.element_size())) for d in devs)

    def _gauge(self) -> None:
        sets = len(self._staged) + len(self._inflight) + len(self._executing)
        self.counters["peak_live_sets"] = max(self.counters["peak_live_sets"], sets)
        self.counters["peak_live_bytes"] = max(self.counters["peak_live_bytes"], self._live_bytes)

    def _make_room(self) -> None:
        """Back-pressure: before a new set is staged, the oldest consumed set whose
        layer may still run is waited for (CPU waits on its done event) until at
        most ``prefetch`` sets are staged or in flight."""
        while self._inflight and len(self._staged) + len(self._inflight) >= self.prefetch + 1:
            devs, ev, nb = self._inflight.popleft()
            if ev is not None:
                ev.synchronize()
                self.counters["waited"] += 1
            self._live_bytes -= nb
            del devs

    def release_ring(self) -> None:
        """Everything staged or in flight is dropped (after the device is idle) and the
        allocator's cache handed back -- before a regain needs the unit's whole bytes."""
        self._sync()
        while self._inflight:
            _devs, ev, nb = self._inflight.popleft()
            if ev is not None:
                ev.synchronize()
            self._live_bytes -= nb
        for _li, (devs, _ev) in list(self._staged.items()):
            self._live_bytes -= self._set_bytes(devs)
        self._staged.clear()
        self._live_bytes = max(0, self._live_bytes)
        self._empty_cache()

    def _stage(self, li: int) -> None:
        if li in self._staged or li not in self._images:
            return
        self._make_room()
        torch = self._torch
        imgs = self._images[li]
        # the forward runs under inference_mode; a staged copy made there would be an INFERENCE tensor, which
        # a normal parameter's .data cannot take -- staged copies are ordinary tensors
        with torch.inference_mode(False):
            self._stage_into(li, imgs)

    def _stage_into(self, li: int, imgs) -> None:
        torch = self._torch
        if self._cuda:
            with torch.cuda.stream(self._side):
                devs = []
                for _t, orig, host in imgs:
                    d = torch.empty_strided(tuple(orig.size()), tuple(orig.stride()), dtype=orig.dtype,
                                            device=self._device)
                    d.copy_(host, non_blocking=True)
                    devs.append(d)
                ev = torch.cuda.Event()
                ev.record(self._side)
        else:  # hermetic: the "device" is the host
            devs = [host.clone() for _t, _o, host in imgs]
            ev = None
        self._staged[li] = (devs, ev)
        self._live_bytes += self._set_bytes(devs)
        self.counters["staged"] += 1
        self._gauge()

    def _prime(self) -> None:
        for li in self._order[: self.prefetch]:
            self._stage(li)

    def _ahead(self, li: int) -> None:
        if not self._order:
            return
        i = bisect.bisect_right(self._order, li)
        n = len(self._order)
        for k in range(min(self.prefetch, n - 1 if li in self._images else n)):
            self._stage(self._order[(i + k) % n])

    def pre_layer(self, li: int) -> None:
        if li not in self._images:
            return
        got = self._staged.pop(li, None)
        if got is None:
            self.counters["staged_late"] += 1
            self._stage(li)
            got = self._staged.pop(li)
        devs, ev = got
        if self._cuda:
            cur = self._torch.cuda.current_stream()
            cur.wait_event(ev)
            for d in devs:
                d.record_stream(cur)
        with self._torch.inference_mode(False):
            for (t, _orig, _h), d in zip(self._images[li], devs):
                t.data = d
        self._executing[li] = (devs, self._set_bytes(devs))
        self.counters["swapped"] += 1
        self._ahead(li)

    def post_layer(self, li: int) -> None:
        rows = self._images.get(li, ())
        with self._torch.inference_mode(False):
            for t, orig, _h in rows:
                t.data = orig
        got = self._executing.pop(li, None)
        if got is None:
            return
        devs, nb = got
        ev = None
        if self._cuda:
            ev = self._torch.cuda.Event()
            ev.record(self._torch.cuda.current_stream())
        self._inflight.append((devs, ev, nb))     # held until the layer's GPU work is done (back-pressure)
        self._gauge()

    def install_hooks(self, layers, start: int, end: int) -> int:
        """Forward pre/post hooks on the hull's decoder layers that any unit
        streams; a hook is a dict lookup while its layer is resident."""
        wanted = set()
        for u in self.units:
            wanted.update(u.tensors)
        n = 0
        for li in range(int(start), int(end)):
            if li not in wanted or li >= len(layers):
                continue
            layers[li].register_forward_pre_hook(lambda _m, _a, li=li: self.pre_layer(li))
            layers[li].register_forward_hook(lambda _m, _a, _o, li=li: self.post_layer(li))
            n += 1
        return n


# ----------------------------------------------------------------------------- PP0 wiring


def _decoder_layers(model):
    import torch

    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and str(name).split(".")[-1] == "layers":
            if best is None or len(mod) > len(best):
                best = mod
    return best


def build_for_runner(runner) -> Optional[LayerStreamer]:
    """PP0 of a dual P with the switch on: catalog the weight-chunk tag pools,
    build the streamer and hook the hull. None (with one line) otherwise."""
    if not armed() or int(getattr(runner, "pp_rank", 0) or 0) != 0:
        return None
    import torch

    from sglang.srt.managers import weg2_memory_saver as _ms

    pools = {t: p for t, p in getattr(_ms, "_TAG_MEM_POOLS", {}).items() if _ms.is_weights_chunk_tag(t)}
    if not pools:
        logger.warning("%s not built: this rank has no weight-chunk tag pools (SGLANG_WEG2_WEIGHT_CHUNK_LAYERS / "
                       "SGLANG_WEG2_WEIGHT_CHUNKS unset?) -- nothing can be paused per layer", MARK)
        return None
    segments = {}
    for tag, pool in pools.items():
        try:
            segments[tag] = pool.snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s tag=%s snapshot unreadable (%r): the unit is not streamable", MARK, tag, exc)
    named: List[Tuple[str, object]] = list(runner.model.named_modules())
    for r, part in enumerate(getattr(runner, "dual_share_part_models", None) or ()):
        if part is not None:
            named.extend(("part%d.%s" % (r, n), m) for n, m in part.named_modules())
    # readers OUTSIDE the forward (the sleep leg): the hull's buffers (static-state export/import) and the
    # NVFP4 Marlin<->native flip of flagged linears -- a unit holding their storage is refused
    forbidden: Dict[int, str] = {}
    for _n, b in runner.model.named_buffers():
        try:
            forbidden[int(b.untyped_storage().data_ptr())] = "hull buffer %s (sleep static-state export)" % _n
        except Exception:  # noqa: BLE001
            pass
    try:
        from sglang.srt.layers.quantization import nvfp4_marlin_inplace as _mi

        for lay in _mi.flagged_layers([runner.model] + [p for p in (getattr(runner, "dual_share_part_models", None)
                                                                    or ()) if p is not None]):
            for t in _module_tensors(lay):
                if _on_device(t):
                    forbidden[int(t.untyped_storage().data_ptr())] = "NVFP4 Marlin<->native flip at sleep"
    except Exception as exc:  # noqa: BLE001 -- unreadable: no unit may be proven free of it
        logger.warning("%s marlin flag scan failed (%r): every unit refused", MARK, exc)
        return None
    units, refused = catalog(named, segments, forbidden=forbidden)
    # Python objects outside the modules that hold a unit's memory (cached views, dict entries): refuse
    import gc

    live_bases = {t: [a for seg in segments.get(t, ()) for a in
                      (int(b.get("address", 0)) for b in seg.get("blocks", ()) if b.get("state") == "active_allocated")]
                  for t in segments}
    extra = uncatalogued_holders(units, live_bases, gc.get_objects())
    if extra:
        for tag, n in sorted(extra.items()):
            refused[tag] = "%d Python tensor object(s) outside the modules hold its memory (not swappable)" % n
        units = [u for u in units if u.tag not in extra]
    # TMS pauses by ITS tag record, not by torch's pool: an allocation tagged weights_<k> outside the tag pool
    # (a load-time transient re-homed elsewhere) would be unmapped too and is in no snapshot -> refuse the unit
    adapter = getattr(runner, "memory_saver_adapter", None)
    tag_bytes = getattr(adapter, "tag_bytes", None)
    if tag_bytes is not None:
        kept = []
        for u in units:
            try:
                tms = int(tag_bytes(u.tag) or 0)
            except Exception:  # noqa: BLE001 -- unreadable: cannot prove coverage
                tms = -1
            if tms < 0 or tms > u.nbytes + (2 << 20):
                refused[u.tag] = "TMS records %d B under the tag, the pool's segments hold %d B" % (tms, u.nbytes)
            else:
                kept.append(u)
        units = kept
    for tag, why in sorted(refused.items()):
        logger.warning("%s REFUSED tag=%s: %s", MARK, tag, why)
    if not units:
        logger.warning("%s not built: no streamable unit (%d refused)", MARK, len(refused))
        return None
    if adapter is None or not getattr(adapter, "enabled", False):
        logger.warning("%s not built: no active memory-saver adapter (the pause is a TMS pause)", MARK)
        return None
    dev = torch.device("cuda", int(runner.gpu_id))
    from sglang.srt.weg2.dual_p_kv_stage import phys_free_bytes

    tag_mapped = getattr(adapter, "tag_mapped_bytes", None)
    st = LayerStreamer(units, pause=adapter.pause, resume=adapter.resume, prefetch=prefetch_layers(),
                       device=dev, phys_free=phys_free_bytes, sync=lambda: torch.cuda.synchronize(dev),
                       tag_mapped=tag_mapped, empty_cache=torch.cuda.empty_cache)
    layers = _decoder_layers(runner.model)
    hooked = st.install_hooks(layers, int(getattr(runner, "start_layer", 0) or 0),
                              int(getattr(runner, "end_layer", len(layers)) or len(layers))) if layers is not None else 0
    install(st)
    logger.warning("%s BUILT units=%d (%s) bytes=%d hooked_layers=%d prefetch=%d staging=%d B refused=%d", MARK,
                   len(units), ",".join("%s:%d-%d" % (u.tag, min(u.tensors), max(u.tensors)) for u in units),
                   sum(u.nbytes for u in units), hooked, st.prefetch, st.staging_bytes(), len(refused))
    return st


def try_stream_for_grant(actor, stages: Sequence[Mapping], level_tokens: int, covered0: int, peek) -> int:
    """PP0's grant was short. If ONLY PP0's card is short and pausing units
    covers its deficit, pause them and lend the bytes; returns the bytes lent
    (0 = nothing done, the request waits as before)."""
    st = getattr(actor, "streamer", None)
    if st is None or not stages:
        return 0
    step = int(stages[0]["step"])
    top = min(int(s["top"]) for s in stages)
    k = min(top, (int(level_tokens) + step - 1) // step * step) // step
    deficit = 0
    for i, s in enumerate(stages):
        need = max(0, int(s["bytes"][k]) - (int(covered0) if i == 0 else 0))
        state = peek(s["ledger"])
        if state is None:
            return 0
        short = need - int(state.free)
        if i != 0 and short > 0:
            return 0                       # another card is short too: streaming PP0 would not grant it
        if i == 0:
            deficit = short
    if int(level_tokens) in st.levels_streamed and st.freed and deficit > 0:
        # this level already paused units and the grant is still short: the loan went to D (legal). No
        # cascade of further pauses for the same level -- the request waits like any short grant (P never
        # presses D); a NEW (higher) level may still stream.
        if int(level_tokens) not in _NOPLAN_SAID:
            _NOPLAN_SAID.add(int(level_tokens))
            logger.info("%s STREAM-HELD level=%d deficit=%d B -- units already paused for this level, the card "
                        "pool gave the loan to D; the request waits", MARK, int(level_tokens), deficit)
        return 0
    tags = st.plan(deficit)
    if not tags:
        if deficit > 0 and int(level_tokens) not in _NOPLAN_SAID:   # once per level (the grant retries every N ms)
            if len(_NOPLAN_SAID) > 1024:
                _NOPLAN_SAID.clear()
            _NOPLAN_SAID.add(int(level_tokens))
            logger.info("%s NO-PLAN level=%d deficit=%d B room=%d B -- the request waits", MARK, int(level_tokens),
                        deficit, st.room())
        return 0
    ring = 0 if st.freed else st.staging_bytes()
    before = set(st.freed)
    freed = st.stream_out(tags)
    if freed <= 0:
        return 0
    st.levels_streamed.add(int(level_tokens))
    lent = 0
    for tag in st.paused():
        if tag in before:
            continue
        n = int(st.freed[tag])
        take = min(n, ring)               # the ring the prefetch now holds is not lent (first unit pays it)
        ring -= take
        st.loan[tag] = n - take
        lent += n - take
    if lent > 0:
        actor.ledger.lend(lent)
    actor._stream_lent = int(getattr(actor, "_stream_lent", 0) or 0) + lent
    logger.warning("%s LEND level=%d deficit=%d B freed=%d B lent=%d B (stream loan total %d B, ring %d B kept) "
                   "-- P's own weight bytes, D untouched", MARK, int(level_tokens), deficit, freed, lent,
                   actor._stream_lent, st.staging_bytes())
    return lent


def regain_at_idle(actor, phys_free=None) -> int:
    """P holds no KV: take units back, newest first, while the ledger hands the
    loan back AND the card physically has it. Returns the bytes regained."""
    st = getattr(actor, "streamer", None)
    if st is None or not st.freed or int(getattr(actor, "mapped_tokens", 0) or 0) > 0:
        return 0
    got = 0
    st.release_ring()                         # the ring's cached blocks back to the driver before the phys check
    for tag in reversed(st.paused()):
        n = int(st.loan.get(tag, st.freed[tag]))
        if phys_free is not None:
            pf = phys_free()
            if pf is not None and int(pf) < int(st.freed[tag]):   # the resume maps the whole unit
                break
        if not actor.ledger.reclaim(n):
            if not getattr(actor, "_stream_held_said", False):
                logger.info("%s REGAIN-HELD tag=%s %d B -- the card pool has it committed (D grew into the loan); "
                            "P keeps streaming", MARK, tag, n)
                actor._stream_held_said = True
            break
        try:
            st.regain(tag)
        except Exception:
            actor.ledger.lend(n)           # the loan stands again; the unit stays paused
            raise
        actor._stream_lent = max(0, int(getattr(actor, "_stream_lent", 0) or 0) - n)
        actor._stream_held_said = False
        got += n
    return got


__all__ = [
    "MARK", "ENV", "PREFETCH_ENV", "armed", "prefetch_layers", "active", "install", "reset_for_tests", "owns",
    "force_eager", "plan_units", "level_need_on_card0", "StreamUnit", "catalog", "LayerStreamer",
    "build_for_runner", "try_stream_for_grant", "regain_at_idle",
]
