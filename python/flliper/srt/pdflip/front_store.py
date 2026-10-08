# SPDX-License-Identifier: Apache-2.0
"""L3-INDEX / L2-ARENA PRICE (02.10.): the X credit of an arrival is the page-granular
STORE prefix of its exact token ids -- asked with the key P and D read with.

METAL, boot y7d (9bdfe50185, front log ..._1002_085437.front.log):
  * boot start, 08:57:43-08:58:09: pdflip-0-2 / 0-3 / 0-4 / 1-5 priced
    ``presence_span=0 presence_src=none`` (115213 / 90888 / 91083 / 100906),
    LONG, flip pair -- P then hit 111232 of 111256 (24 new tokens) and
    100736 / 100864 / 100864 of theirs. The prefixes sat in the PERSISTENT L3
    store from earlier boots; the front's presence records only knew D readings
    and P's END-ANCHORs of this boot (``STORE-PRESENCE src=p_flush`` came at
    08:58:14, after the flip).
  * 09:01:20 (tokenizer ready): pdflip-2-19 ``X-EXACT-PRICE pending=30076
    tokens=30076 credit=0 src=none``, LONG, flip pair, P hit 30016 of 30076
    (60 new tokens): a PAGE prefix of an earlier, longer request of this boot,
    not an END-ANCHOR any presence record names.

THE QUESTION ASKED (``StorePresence.depth``), the one ``batch_exists_v2`` of the
file backend answers on the fetch path, against the shared L3 stem index
(#1459, ``<arena dir>-l3idx/l3idx.bin``: seeded from the persistent snapshot +
journals at attach, kept exact by the evictor -- every commit of this boot,
every page of earlier boots, every eviction):
  1. page keys: ``get_hash_str(RadixKey(ids, is_bigram), None, page_size)`` --
     the tree's own key form (``pdflip_store_told._probe_key``; bigram on every
     Weg 2 group, FLLIPER_HICACHE_BIGRAM_KEYS), complete pages only;
  2. the LEADING run of pages whose KV stem is listed (stop at the first miss);
  3. cut by every ALL_PAGES component the store holds (the QSA index page);
  4. the deepest page within that run that also carries the TRAILING
     component's blob (the mamba anchor: a hybrid group resumes only at a
     recurrent state -- the same proof the END-ANCHOR carries).
The credit is ``pages * page_size`` tokens; a bigram chain keys ``n - 1``
units, so at least one token always stays to prefill.

The stems carry the store's GROUP-WIDE suffix (``L3_SUFFIXES.<group>.json``,
the suffix every group of the shared-key store scans: geometry-neutral KV,
mamba and QSA pages, #706). No common suffix = no shared-key store = no credit.

L2 TOO (L2-ARENA PRICE, 02.10., law "L2/L3 are shared, no context only on D
or P"): a page COMPLETE in any shared host arena under the arena dir counts
like an L3 one -- the union ``batch_exists_v2`` reads, arena first. The L3
write-behind lags L2 by minutes on metal (27B N4a 'L3-REUSE WRITE-BEHIND ...
deferred=86750'), so an in-boot prefix is often L2-only when priced. The
depth names the tier it needs (``l2_arena`` when L3 alone proves less).

Over-crediting costs the request and every decode a reroute parks -- the
reader still finds an evicted page missing, and D's X gate stays the
backstop; under-crediting costs a P leg.
"""

from __future__ import annotations

import glob
import json
import os
import time
from array import array
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import msgspec
import numpy as np

#: component pools whose hit policy is TRAILING_PAGES (one trailing page):
#: ``MambaComponent.build_hicache_transfers``
TRAILING_COMPONENTS = frozenset({"mamba"})
#: presence-only pools (``caps_claim=False``): never cut the claim
PRESENCE_ONLY_PREFIX = "draft"
#: stems asked per index call; the walk stops at the chunk with the first miss
CHUNK = 512
#: PROBE-FAST: the forward chunk doubles from CHUNK up to this many stems (a
#: long hit costs log2 calls, an early miss still at most CHUNK stems asked)
CHUNK_MAX = 8192
#: shard directories sampled for the store's component set
SAMPLE_SHARDS = 8
SAMPLE_NAMES = 4096


class Depth(msgspec.Struct, frozen=True):
    tokens: int
    kv_pages: int
    pages: int
    ms: float
    #: the tier the depth needs: "l3_index" when L3 alone proves it, else
    #: "l2_arena" (pages / anchor COMPLETE only in the shared host arena)
    tier: str = "none"
    #: the depth L3 alone proves (pages)
    l3_pages: int = 0
    #: PROBE-FAST (02.10.): "fast" (stems encoded once into one NUL-terminated
    #: buffer, every tier asked over the same char ** in C, numpy verdicts) or
    #: "list" (the per-stem form); the answer is the same
    form: str = "list"
    #: stem lookups made (per tier ask, not per tier)
    asked: int = 0

def shared_suffix(store_dir: str) -> Tuple[Optional[str], str]:
    """The suffix every group of the shared-key store scans (the group-wide,
    geometry-neutral one), or ``(None, why)``."""
    paths = sorted(glob.glob(os.path.join(store_dir, "L3_SUFFIXES.*.json")))
    if not paths:
        return None, "no L3_SUFFIXES record in the store"
    common = None
    for p in paths:
        try:
            with open(p) as f:
                sfx = set(json.load(f).get("suffixes") or [])
        except (OSError, ValueError) as e:
            return None, f"{os.path.basename(p)} unreadable ({type(e).__name__})"
        common = sfx if common is None else common & sfx
    if not common:
        return None, "the groups share no store suffix (no shared-key store)"
    return min(common, key=len), f"{len(paths)} group record(s)"


def store_components(store_dir: str, suffix: str) -> FrozenSet[str]:
    """The component pools the store holds pages of under ``suffix`` (from a
    sample of its shard directories; names only, no stat)."""
    tail = suffix + ".bin"
    out = set()
    seen = shards = 0
    try:
        with os.scandir(store_dir) as it:
            dirs = [e.path for e in it if len(e.name) == 2 and e.is_dir(follow_symlinks=False)]
    except OSError:
        return frozenset()
    for d in dirs:
        if shards >= SAMPLE_SHARDS or seen >= SAMPLE_NAMES:
            break
        try:
            with os.scandir(d) as it:
                names = [e.name for e in it]
        except OSError:
            continue
        if names:
            shards += 1
        for name in names:
            seen += 1
            dot = name.find(".")
            if name.endswith(tail) and 0 <= dot < len(name) - len(tail):
                out.add(name[dot + 1:len(name) - len(tail)])
    return frozenset(out)


def bigram_page_hasher(ids: np.ndarray, page_size: int, bigram: bool,
                       extra_key: Optional[str] = None) -> List[str]:
    """The page keys of ``ids`` as the tree keys them (complete pages only).

    Q-460 SALT-ISOLATION: ``extra_key`` is the request's namespace (cache_salt
    + extra_key); the store keys a namespaced chain from its own root seed."""
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.utils import get_hash_str

    units = int(ids.size) - 1 if bigram else int(ids.size)
    pages = max(0, units) // int(page_size)
    if pages <= 0:
        return []
    raw = array("q")
    raw.frombytes(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    hashes = get_hash_str(RadixKey(raw, extra_key, is_bigram=bool(bigram)), None, page_size=int(page_size))
    return list(hashes[:pages])


class StorePresence:
    """The page-granular store depth of a token sequence (see the module note).

    TWO TIERS, ONE QUESTION (L2-ARENA PRICE, 02.10., law "L2/L3 are shared"):
    a page is present when its stem is COMPLETE in any shared host arena
    under ``arena_dir`` (L2, ``arena_find_stems``) or listed by the L3 stem
    index -- exactly the union ``batch_exists_v2`` reads (arena first, disk
    for the rest). The L3 write-behind lags L2 by minutes on metal (27B N4a
    'L3-REUSE WRITE-BEHIND ... deferred=86750'), so an in-boot prefix is
    often L2-only when it is priced. Both tiers are JOINED, never created:
    an arena or index that does not exist yet is looked for again later."""

    def __init__(self, index, suffix: str, page_size: int, bigram: bool,
                 all_pages: Sequence[str] = (), trailing: Sequence[str] = (),
                 hasher: Callable[[np.ndarray, int, bool], List[str]] = bigram_page_hasher,
                 arena_dir: str = "", index_path: str = ""):
        self.index = index
        self.suffix = str(suffix)
        self.page_size = max(1, int(page_size))
        self.bigram = bool(bigram)
        self.all_pages = tuple(sorted(all_pages))
        self.trailing = tuple(sorted(trailing))
        self.hasher = hasher
        self.arena_dir = str(arena_dir or "")
        self.index_path = str(index_path or "")
        self.arenas: Dict[str, object] = {}

    def describe(self) -> str:
        return (f"suffix={self.suffix} page={self.page_size} bigram={int(self.bigram)} "
                f"all_pages={list(self.all_pages)} trailing={list(self.trailing)} "
                f"l3_index={'joined' if self.index is not None else 'absent'} "
                f"l2_arenas={len(self.arenas)} arena_dir={self.arena_dir or '-'}")

    # -- the two tiers --------------------------------------------------------
    def _rejoin(self) -> None:
        """Join the tiers that appeared since (arenas are created lazily, per
        page width, by the first rank that uses one). One glob of the arena
        dir per probe; a failed join is a failed open(2), nothing more."""
        if self.index is None and self.index_path:
            try:
                from flliper.srt.mem_cache.storage.file.l3_index import L3Index

                self.index = L3Index(self.index_path, create=False)
            except Exception:  # noqa: BLE001 -- not there yet: asked again later
                self.index = None
        if not self.arena_dir:
            return
        for path in sorted(glob.glob(os.path.join(self.arena_dir, "arena-*.bin"))):
            if path in self.arenas:
                continue
            try:
                from flliper.srt.mem_cache.storage.file.hicache_arena import ArenaView

                self.arenas[path] = ArenaView(path)
            except Exception:  # noqa: BLE001 -- not initialised yet: asked again later
                continue

    def _ask(self, stems: List[str], memo: Dict[str, Tuple[bool, bool]]) -> None:
        """(in L3, COMPLETE in L2) per stem into ``memo``."""
        todo = [s for s in stems if s not in memo]
        if not todo:
            return
        l3 = self.index.has(todo) if self.index is not None else [False] * len(todo)
        l2 = [False] * len(todo)
        for arena in self.arenas.values():
            for i, st in enumerate(arena.find_states(todo)):
                if st == 2:
                    l2[i] = True
        for s, a, b in zip(todo, l3, l2):
            memo[s] = (bool(a), bool(b))

    def _stem(self, h: str, comp: Optional[str]) -> str:
        return f"{h}{self.suffix}" if comp is None else f"{h}.{comp}{self.suffix}"

    def _leading(self, hashes: Sequence[str], comp: Optional[str], memo, tiers) -> int:
        n = 0
        for off in range(0, len(hashes), CHUNK):
            stems = [self._stem(h, comp) for h in hashes[off:off + CHUNK]]
            self._ask(stems, memo)
            hit = next((i for i, s in enumerate(stems) if not any(memo[s][t] for t in tiers)),
                       len(stems))
            n += hit
            if hit < len(stems):
                break
        return n

    def _pages(self, hashes: Sequence[str], memo, tiers) -> Tuple[int, int]:
        """(anchored pages, leading KV pages) over the tiers ``tiers``
        (0 = L3, 1 = L2): the KV run, cut by every all-pages pool, ended at
        the deepest page carrying every trailing pool's blob."""
        kv = self._leading(hashes, None, memo, tiers)
        pages = kv
        for comp in self.all_pages:
            if pages <= 0:
                break
            pages = min(pages, self._leading(hashes[:pages], comp, memo, tiers))
        for comp in self.trailing:
            if pages <= 0:
                break
            stems = [self._stem(h, comp) for h in hashes[:pages]]
            self._ask(stems, memo)
            pages = 1 + max((i for i, s in enumerate(stems) if any(memo[s][t] for t in tiers)),
                            default=-1)
        return pages, kv

    # -- PROBE-FAST (02.10.) ----------------------------------------------------
    # N5q epoch 4 (front log ..._1002_151409, 15:17:13.127-13.540): pdflip-3-5
    # 'L3-INDEX-PRESENCE tier=l2_arena depth=73728 kv_pages=74064 tokens=98798
    # probe_ms=413.3' -- serial after the 132 ms X-EXACT count, before the
    # ROUTE-VERDICT. The 27B keys pages of ONE token, so the probe asks ~74k
    # stems per tier (L3 index + 2 arenas). Desk profile of the same shape
    # (213-238 ms): the C lookups are a minority; the time is Python per stem --
    # str.encode x3 tiers, a ctypes c_char_p array built per call and per tier,
    # int8 -> list -> bool, the memo dict and an any() generator per stem.
    # The fast form encodes each stem ONCE into a fixed-width NUL-terminated
    # numpy buffer, hands the SAME char ** to every tier, and reads the
    # verdicts as numpy arrays. Same stems, same tiers, same leading-run /
    # cut / deepest-anchor rules, so the same Depth (pinned by the tests
    # against the list form).
    class _Asked:
        """Per-probe verdicts per component: (in L3, COMPLETE in L2, L3 asked,
        L2 asked). A tier is asked a page only when the answer needs it: the
        union walk asks the arenas first (``batch_exists_v2``'s order) and the
        L3 index only where no arena holds the page; the L3-only walk asks the
        index for the leading pages it has not asked yet."""

        def __init__(self, n: int, hlen: int = 0):
            self.n = n
            #: the one width of every page hash, 0 = not uniform
            self.hlen = hlen
            self.comps: Dict[Optional[str], Tuple[np.ndarray, ...]] = {}
            self.lookups = 0

        def arrays(self, comp: Optional[str]):
            a = self.comps.get(comp)
            if a is None:
                a = self.comps[comp] = tuple(np.zeros(self.n, dtype=bool) for _ in range(4))
            return a

    def _tail(self, comp: Optional[str]) -> str:
        return self.suffix if comp is None else f".{comp}{self.suffix}"

    def _stems_fast(self, hashes: Sequence[str], comp: Optional[str], todo: np.ndarray, st: "_Asked"):
        """(stems thunk, char ** over one NUL-terminated buffer, n, keep-alive).

        Uniform hash width (the probe checked it once): the rows are written
        as bytes -- ''.join + one ASCII encode + two column copies, no per-stem
        Python. Otherwise via a str list. Non-ASCII raises UnicodeEncodeError
        (the caller falls back to the list form)."""
        import ctypes

        tail = self._tail(comp)
        lo, hi = int(todo[0]), int(todo[-1]) + 1
        contiguous = hi - lo == todo.size
        sel = hashes[lo:hi] if contiguous else [hashes[int(i)] for i in todo]
        n = len(sel)
        if st.hlen:
            tb = tail.encode("ascii")
            w = st.hlen + len(tb) + 1
            buf = np.zeros((n, w), dtype=np.uint8)
            buf[:, :st.hlen] = np.frombuffer("".join(sel).encode("ascii"), dtype=np.uint8).reshape(n, st.hlen)
            buf[:, st.hlen:w - 1] = np.frombuffer(tb, dtype=np.uint8)
            row = w
        else:
            u = np.array([h + tail for h in sel])
            # +1: every stem NUL-terminated, the longest one too
            buf = np.zeros(n, dtype="S%d" % (u.dtype.itemsize // 4 + 1))
            buf[:] = u
            row = buf.dtype.itemsize
        addr = np.uint64(buf.ctypes.data) + np.arange(n, dtype=np.uint64) * np.uint64(row)

        def stems():
            return [h + tail for h in sel]

        return stems, addr.ctypes.data_as(ctypes.POINTER(ctypes.c_char_p)), n, (buf, addr)

    def _ask_l3(self, hashes, comp, todo: np.ndarray, st: "_Asked") -> None:
        l3, _, a3, _ = st.arrays(comp)
        if todo.size == 0:
            return
        if self.index is not None:
            stems, ptrs, n, keep = self._stems_fast(hashes, comp, todo, st)
            st.lookups += n
            if hasattr(self.index, "has_ptrs"):
                l3[todo] = self.index.has_ptrs(ptrs, n) != 0
            else:
                l3[todo] = np.asarray(self.index.has(stems()), dtype=bool)
            del keep  # alive through the C call above
        a3[todo] = True

    def _ask_fast(self, hashes: Sequence[str], comp: Optional[str], lo: int, hi: int,
                  st: "_Asked", tiers=(0, 1)) -> None:
        """Ask what ``tiers``' answer over [lo, hi) of ``comp`` still needs."""
        l3, l2, a3, a2 = st.arrays(comp)
        if 1 in tiers:
            todo = lo + np.flatnonzero(~a2[lo:hi])
            if todo.size:
                # COMPLETE in ANY arena: each further arena is asked only for the
                # pages the ones before it did not hold (largest arena first --
                # the KV pages live there; the order changes no answer)
                rest = todo
                for arena in sorted(self.arenas.values(),
                                    key=lambda a: -int(getattr(a, "slots", 0) or 0)):
                    if rest.size == 0:
                        break
                    stems, ptrs, n, keep = self._stems_fast(hashes, comp, rest, st)
                    st.lookups += n
                    if hasattr(arena, "find_states_ptrs"):
                        got = arena.find_states_ptrs(ptrs, n) == 2
                    else:
                        got = np.asarray(arena.find_states(stems()), dtype=np.int8) == 2
                    del keep  # alive through the C call above
                    l2[rest[got]] = True
                    rest = rest[~got]
                a2[todo] = True
            # the index only where no arena holds the page (the union needs no more)
            todo3 = lo + np.flatnonzero(~a3[lo:hi] & ~l2[lo:hi])
        else:
            todo3 = lo + np.flatnonzero(~a3[lo:hi])
        self._ask_l3(hashes, comp, todo3, st)

    @staticmethod
    def _present(st: "_Asked", comp: Optional[str], lo: int, hi: int, tiers) -> np.ndarray:
        l3, l2, _, _ = st.arrays(comp)
        return (l3[lo:hi] | l2[lo:hi]) if 1 in tiers else l3[lo:hi].copy()

    def _leading_fast(self, hashes, comp, upto: int, st, tiers) -> int:
        off, size = 0, CHUNK
        while off < upto:
            hi = min(upto, off + size)
            self._ask_fast(hashes, comp, off, hi, st, tiers)
            miss = np.flatnonzero(~self._present(st, comp, off, hi, tiers))
            if miss.size:
                return off + int(miss[0])
            off, size = hi, min(CHUNK_MAX, size * 2)
        return int(upto)

    def _deepest_fast(self, hashes, pages: int, comp: str, st, tiers) -> int:
        hi = int(pages)
        while hi > 0:
            lo = max(0, hi - CHUNK)
            self._ask_fast(hashes, comp, lo, hi, st, tiers)
            hit = np.flatnonzero(self._present(st, comp, lo, hi, tiers))
            if hit.size:
                return lo + int(hit[-1])
            hi = lo
        return -1

    def _pages_fast(self, hashes, upto: int, st, tiers) -> Tuple[int, int]:
        """``_pages`` over ``hashes[:upto]`` -- same rules, verdicts from ``st``."""
        kv = self._leading_fast(hashes, None, upto, st, tiers)
        pages = kv
        for comp in self.all_pages:
            if pages <= 0:
                break
            pages = min(pages, self._leading_fast(hashes, comp, pages, st, tiers))
        for comp in self.trailing:
            if pages <= 0:
                break
            pages = self._deepest_fast(hashes, pages, comp, st, tiers) + 1
        return pages, kv

    def _depth_fast(self, hashes: Sequence[str], t0: float) -> Depth:
        widths = set(map(len, hashes))
        st = StorePresence._Asked(len(hashes), widths.pop() if len(widths) == 1 else 0)
        pages, kv = self._pages_fast(hashes, len(hashes), st, (0, 1))
        l3_pages = self._pages_fast(hashes, pages, st, (0,))[0] if pages > 0 else 0
        tier = "none" if pages <= 0 else ("l3_index" if l3_pages >= pages else "l2_arena")
        return Depth(tokens=int(pages) * self.page_size, kv_pages=int(kv), pages=int(pages),
                     ms=(time.perf_counter() - t0) * 1000.0, tier=tier, l3_pages=int(l3_pages),
                     form="fast", asked=int(st.lookups))

    def _hash(self, ids: np.ndarray, extra_key: Optional[str]) -> List[str]:
        # the default namespace keeps the 3-argument hasher call (test doubles)
        if extra_key is None:
            return self.hasher(ids, self.page_size, self.bigram)
        return self.hasher(ids, self.page_size, self.bigram, extra_key)

    def depth(self, ids: Optional[np.ndarray], fast: Optional[bool] = None,
              extra_key: Optional[str] = None) -> Depth:
        """``fast``: None = the switch FLLIPER_PDFLIP_FRONT_PROBE_FAST (default on).
        ``extra_key``: the request's namespace (Q-460) -- the store depth OF
        THAT NAMESPACE; another tenant's pages are not this request's credit."""
        t0 = time.perf_counter()
        if ids is None or ids.size == 0:
            return Depth(tokens=0, kv_pages=0, pages=0, ms=0.0)
        self._rejoin()
        if fast is None:
            fast = probe_fast_on()
        if fast:
            hashes = self._hash(ids, extra_key)
            try:
                return self._depth_fast(hashes, t0)
            except UnicodeEncodeError:
                pass  # a non-ASCII stem: the list form below asks it utf-8 encoded
        hashes = self._hash(ids, extra_key)
        memo: Dict[str, Tuple[bool, bool]] = {}
        pages, kv = self._pages(hashes, memo, (0, 1))
        l3_pages = self._pages(hashes[:pages], memo, (0,))[0] if pages > 0 else 0
        tier = "none" if pages <= 0 else ("l3_index" if l3_pages >= pages else "l2_arena")
        return Depth(tokens=int(pages) * self.page_size, kv_pages=int(kv), pages=int(pages),
                     ms=(time.perf_counter() - t0) * 1000.0, tier=tier, l3_pages=int(l3_pages),
                     form="list", asked=len(memo))


def probe_fast_on() -> bool:
    """PROBE-FAST switch (FLLIPER_PDFLIP_FRONT_PROBE_FAST, default on)."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_FRONT_PROBE_FAST.get())


def l3_index_path(arena_dir: str) -> str:
    """The #1459 index beside the arena (``HiCacheFile._l3_index``)."""
    return os.path.join(arena_dir.rstrip("/") + "-l3idx", "l3idx.bin")


def open_store_presence(store_dir: str, arena_dir: str, page_size: int, bigram: bool,
                        hybrid: bool) -> Tuple[Optional[StorePresence], str]:
    """``(probe, provenance)`` or ``(None, why)``. JOINS the L3 index and the
    host arenas the ranks created (never creates either: the index creator
    seeds it from the snapshot, an arena's creator sets its geometry); a tier
    not there yet is joined when it appears."""
    if not store_dir or not os.path.isdir(store_dir):
        return None, f"no store directory ({store_dir!r})"
    if not arena_dir:
        return None, "no arena dir (FLLIPER_HICACHE_ARENA_DIR empty): no shared L2/L3 index"
    suffix, why = shared_suffix(store_dir)
    if suffix is None:
        return None, why
    comps = store_components(store_dir, suffix)
    trailing = sorted(c for c in comps if c in TRAILING_COMPONENTS)
    if hybrid and "mamba" not in trailing:
        trailing.append("mamba")  # a hybrid group resumes only at an anchor
    all_pages = sorted(c for c in comps
                       if c not in TRAILING_COMPONENTS and not c.startswith(PRESENCE_ONLY_PREFIX))
    probe = StorePresence(None, suffix, page_size, bigram, all_pages=all_pages, trailing=trailing,
                          arena_dir=arena_dir, index_path=l3_index_path(arena_dir))
    probe._rejoin()
    entries = probe.index.count() if probe.index is not None else 0
    return probe, f"index={probe.index_path} entries={entries} {why} {probe.describe()}"
