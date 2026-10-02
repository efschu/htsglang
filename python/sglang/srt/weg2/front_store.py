# SPDX-License-Identifier: Apache-2.0
"""L3-INDEX PRICE (02.10.): the X credit of an arrival is the page-granular
STORE prefix of its exact token ids -- asked with the key P and D read with.

METAL, boot y7d (9bdfe50185, front log ..._1002_085437.front.log):
  * boot start, 08:57:43-08:58:09: weg2-0-2 / 0-3 / 0-4 / 1-5 priced
    ``presence_span=0 presence_src=none`` (115213 / 90888 / 91083 / 100906),
    LONG, flip pair -- P then hit 111232 of 111256 (24 new tokens) and
    100736 / 100864 / 100864 of theirs. The prefixes sat in the PERSISTENT L3
    store from earlier boots; the front's presence records only knew D readings
    and P's END-ANCHORs of this boot (``STORE-PRESENCE src=p_flush`` came at
    08:58:14, after the flip).
  * 09:01:20 (tokenizer ready): weg2-2-19 ``X-EXACT-PRICE pending=30076
    tokens=30076 credit=0 src=none``, LONG, flip pair, P hit 30016 of 30076
    (60 new tokens): a PAGE prefix of an earlier, longer request of this boot,
    not an END-ANCHOR any presence record names.

THE QUESTION ASKED (``StorePresence.depth``), the one ``batch_exists_v2`` of the
file backend answers on the fetch path, against the shared L3 stem index
(#1459, ``<arena dir>-l3idx/l3idx.bin``: seeded from the persistent snapshot +
journals at attach, kept exact by the evictor -- every commit of this boot,
every page of earlier boots, every eviction):
  1. page keys: ``get_hash_str(RadixKey(ids, is_bigram), None, page_size)`` --
     the tree's own key form (``weg2_store_told._probe_key``; bigram on every
     Weg 2 group, SGLANG_HICACHE_BIGRAM_KEYS), complete pages only;
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

WHAT IS NOT CREDITED: a page only in the L2 arena and not yet in L3 (the
write-behind copies L2 to L3 every 2 s; P's sleep flush publishes before the
flip). Under-crediting costs a P leg; over-crediting costs the request and
every decode a reroute parks -- the reader stats/opens, never trusts the index,
and D's X gate stays the backstop.
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
#: shard directories sampled for the store's component set
SAMPLE_SHARDS = 8
SAMPLE_NAMES = 4096


class Depth(msgspec.Struct, frozen=True):
    tokens: int
    kv_pages: int
    pages: int
    ms: float


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


def bigram_page_hasher(ids: np.ndarray, page_size: int, bigram: bool) -> List[str]:
    """The page keys of ``ids`` as the tree keys them (complete pages only)."""
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_hash_str

    units = int(ids.size) - 1 if bigram else int(ids.size)
    pages = max(0, units) // int(page_size)
    if pages <= 0:
        return []
    raw = array("q")
    raw.frombytes(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    hashes = get_hash_str(RadixKey(raw, None, is_bigram=bool(bigram)), None, page_size=int(page_size))
    return list(hashes[:pages])


class StorePresence:
    """The page-granular store depth of a token sequence (see the module note)."""

    def __init__(self, index, suffix: str, page_size: int, bigram: bool,
                 all_pages: Sequence[str] = (), trailing: Sequence[str] = (),
                 hasher: Callable[[np.ndarray, int, bool], List[str]] = bigram_page_hasher):
        self.index = index
        self.suffix = str(suffix)
        self.page_size = max(1, int(page_size))
        self.bigram = bool(bigram)
        self.all_pages = tuple(sorted(all_pages))
        self.trailing = tuple(sorted(trailing))
        self.hasher = hasher

    def describe(self) -> str:
        return (f"suffix={self.suffix} page={self.page_size} bigram={int(self.bigram)} "
                f"all_pages={list(self.all_pages)} trailing={list(self.trailing)}")

    def _stem(self, h: str, comp: Optional[str]) -> str:
        return f"{h}{self.suffix}" if comp is None else f"{h}.{comp}{self.suffix}"

    def _leading(self, hashes: Sequence[str], comp: Optional[str]) -> int:
        n = 0
        for off in range(0, len(hashes), CHUNK):
            got = self.index.has([self._stem(h, comp) for h in hashes[off:off + CHUNK]])
            hit = next((i for i, p in enumerate(got) if not p), len(got))
            n += hit
            if hit < len(got):
                break
        return n

    def depth(self, ids: Optional[np.ndarray]) -> Depth:
        t0 = time.perf_counter()
        if ids is None or ids.size == 0:
            return Depth(tokens=0, kv_pages=0, pages=0, ms=0.0)
        hashes = self.hasher(ids, self.page_size, self.bigram)
        kv = self._leading(hashes, None)
        pages = kv
        for comp in self.all_pages:
            if pages <= 0:
                break
            pages = min(pages, self._leading(hashes[:pages], comp))
        for comp in self.trailing:
            if pages <= 0:
                break
            got = self.index.has([self._stem(h, comp) for h in hashes[:pages]])
            last = max((i for i, p in enumerate(got) if p), default=-1)
            pages = last + 1
        return Depth(tokens=int(pages) * self.page_size, kv_pages=int(kv), pages=int(pages),
                     ms=(time.perf_counter() - t0) * 1000.0)


def l3_index_path(arena_dir: str) -> str:
    """The #1459 index beside the arena (``HiCacheFile._l3_index``)."""
    return os.path.join(arena_dir.rstrip("/") + "-l3idx", "l3idx.bin")


def open_store_presence(store_dir: str, arena_dir: str, page_size: int, bigram: bool,
                        hybrid: bool) -> Tuple[Optional[StorePresence], str]:
    """``(probe, provenance)`` or ``(None, why)``. Joins the EXISTING index the
    ranks opened (never creates it: the creator seeds it from the snapshot)."""
    if not store_dir or not os.path.isdir(store_dir):
        return None, f"no store directory ({store_dir!r})"
    if not arena_dir:
        return None, "no arena dir (SGLANG_HICACHE_ARENA_DIR empty): no shared L3 index"
    suffix, why = shared_suffix(store_dir)
    if suffix is None:
        return None, why
    path = l3_index_path(arena_dir)
    try:
        from sglang.srt.mem_cache.storage.file.l3_index import L3Index

        index = L3Index(path, create=False)
    except FileNotFoundError:
        return None, f"no L3 index at {path} yet (no rank opened it)"
    except Exception as e:  # noqa: BLE001 -- no index, no credit, named
        return None, f"L3 index {path} n/a ({type(e).__name__}: {e})"
    comps = store_components(store_dir, suffix)
    trailing = sorted(c for c in comps if c in TRAILING_COMPONENTS)
    if hybrid and "mamba" not in trailing:
        trailing.append("mamba")  # a hybrid group resumes only at an anchor
    all_pages = sorted(c for c in comps
                       if c not in TRAILING_COMPONENTS and not c.startswith(PRESENCE_ONLY_PREFIX))
    probe = StorePresence(index, suffix, page_size, bigram, all_pages=all_pages, trailing=trailing)
    return probe, f"index={path} entries={index.count()} {why} {probe.describe()}"
