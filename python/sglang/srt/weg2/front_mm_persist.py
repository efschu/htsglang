# SPDX-License-Identifier: Apache-2.0
"""MM-PERSIST-1002: what the front learned about images, across a restart.

METAL, NF boot y7t (fc03626e0e, front log ..._1002_150037.front.log):
``15:05:18 WEG2 X-EXACT-FALLBACK rid=weg2-10-10 reason=multimodal-unseen-image
est_uncached=438`` -> LONG -> D->P flip -> ``WEG2-SERVED group=P leg=1
prompt_tokens=727 cached_tokens=704``: P computed 23 tokens, the image KV sat
in the PERSISTENT L3 store from the previous boot (y7o served the same
conversation). The front learns an image's token count only from a served
leg (``X-EXACT-MM-LEARN image=9cbec2e1b5d3 tokens=64``) and kept it in RAM.

Two facts are kept, both front-local and both only ever learned from a leg:

* ``k`` -- an image key's token count (``front_tokens.mm_learn``), so a
  request carrying it is priced exactly from the first arrival on;
* ``anchors`` -- ``(depth, digest)`` of an image prompt's FRONT ids (the
  surrogate expansion, ``front_tokens.mm_expand``) up to a depth P's sleep
  flush published to the shared store (``STORE-PRESENCE src=p_flush``).

Why the anchors and not only K: the store's page keys of an image position
are the group's ``pad_value`` (a hash of the processed pixels), which the
front cannot compute -- the L3-INDEX probe of an image prompt stops at its
first image (``front.py``: ``c.ids[: mm.first_image]``). With K alone the
restarted front prices weg2-10-10 exactly but still finds no credit past
the image and still flips. The anchor carries what the previous boot's
front knew: the store holds this prefix, image included.

The file lives in the STORE directory (``WEG2_FRONT_MM.json``): it is valid
exactly as long as the store it describes (a new model identity is a new
store directory; a removed store takes the file with it; the store's walk
skips non-``.bin`` files). A restored anchor is credited only when the live
store still proves every full page BEFORE the first image (the probe the
front can ask); the image pages themselves are then the backstop's:
D's admission refuses an image outside its covered prefix by name (W123,
``vision_d_guard``) and the request re-routes through P -- the pre-fix path.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FILE_NAME = "WEG2_FRONT_MM.json"
VERSION = 1
#: images kept (oldest dropped first), anchors kept per image (deepest kept)
MAX_IMAGES = 4096
MAX_ANCHORS = 8


def ids_digest(ids: Any) -> str:
    """The digest of a front id prefix -- ``TokenSpans._key``'s form."""
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


class MMPersist:
    """The table in RAM plus its file. Mutators only change RAM and mark it
    dirty; :meth:`snapshot` + :func:`write_snapshot` persist it (the caller
    runs the write off the event loop)."""

    def __init__(self, path: str = ""):
        self.path = path
        #: key -> {"k": int, "t": float, "anchors": [[depth, digest], ...]}
        self.images: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        self.dirty = False
        self.loaded = 0
        self.why = ""

    # -- file ------------------------------------------------------------------
    @classmethod
    def open(cls, store_dir: str) -> "MMPersist":
        """The table of ``store_dir`` (an empty one, named, when there is no
        store directory or no readable file)."""
        if not store_dir or not os.path.isdir(store_dir):
            p = cls("")
            p.why = f"no store directory ({store_dir!r})"
            return p
        p = cls(os.path.join(store_dir, FILE_NAME))
        try:
            with open(p.path) as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            p.why = "no file yet"
            return p
        except (OSError, ValueError) as exc:
            p.why = f"unreadable ({type(exc).__name__}: {exc})"
            return p
        if not isinstance(doc, dict) or int(doc.get("v", 0) or 0) != VERSION:
            p.why = f"version {doc.get('v') if isinstance(doc, dict) else '?'} != {VERSION}"
            return p
        for key, rec in (doc.get("images") or {}).items():
            try:
                k = int(rec["k"])
                anchors = [[int(d), str(g)] for d, g in (rec.get("anchors") or ())][:MAX_ANCHORS]
            except (KeyError, TypeError, ValueError):
                continue
            if k > 0:
                p.images[str(key)] = {"k": k, "t": float(rec.get("t") or 0.0), "anchors": anchors}
        while len(p.images) > MAX_IMAGES:
            p.images.popitem(last=False)
        p.loaded = len(p.images)
        p.why = "loaded"
        return p

    def snapshot(self) -> Optional[Tuple[str, str]]:
        """(path, JSON text) of the table when it changed since the last
        snapshot and has a file; None otherwise. Clears ``dirty``."""
        if not self.dirty or not self.path:
            return None
        self.dirty = False
        return self.path, json.dumps({"v": VERSION, "images": self.images}, separators=(",", ":"))

    # -- facts -----------------------------------------------------------------
    def ktok(self) -> Dict[str, int]:
        return {k: int(r["k"]) for k, r in self.images.items()}

    def note_k(self, key: str, k: int) -> bool:
        """An image's token count as a group served it. A changed count drops
        the anchors (they were taken over the old expansion)."""
        k = int(k)
        if k <= 0:
            return False
        rec = self.images.get(key)
        if rec is not None and int(rec["k"]) == k:
            return False
        self.images.pop(key, None)
        self.images[key] = {"k": k, "t": round(time.time(), 1), "anchors": []}
        while len(self.images) > MAX_IMAGES:
            self.images.popitem(last=False)
        self.dirty = True
        return True

    def note_anchor(self, keys: Sequence[str], ids: Any, depth: int) -> int:
        """The store holds ``ids[:depth]`` (front ids). Recorded under every
        known image key of the prompt; returns how many keys took it."""
        depth = int(depth)
        ids = np.asarray(ids, dtype=np.int32)
        if depth <= 0 or depth > int(ids.size):
            return 0
        dig = ids_digest(ids[:depth])
        n = 0
        for key in dict.fromkeys(keys):
            rec = self.images.get(key)
            if rec is None:
                continue
            anchors: List[List[Any]] = rec["anchors"]
            if any(int(d) == depth and g == dig for d, g in anchors):
                continue
            anchors.append([depth, dig])
            anchors.sort(key=lambda a: -int(a[0]))
            del anchors[MAX_ANCHORS:]
            rec["t"] = round(time.time(), 1)
            self.images.move_to_end(key)
            n += 1
        if n:
            self.dirty = True
        return n

    def credit(self, keys: Sequence[str], ids: Any) -> int:
        """The deepest recorded anchor of any image of ``keys`` that is a
        prefix of ``ids`` (front ids); 0 = none."""
        ids = np.asarray(ids, dtype=np.int32)
        n = int(ids.size)
        best = 0
        cache: Dict[int, str] = {}
        for key in dict.fromkeys(keys):
            rec = self.images.get(key)
            if rec is None:
                continue
            for d, g in rec["anchors"]:
                d = int(d)
                if d <= best or d > n:
                    continue
                if d not in cache:
                    cache[d] = ids_digest(ids[:d])
                if cache[d] == g:
                    best = d
        return best

    def drop_anchors(self, keys: Sequence[str]) -> int:
        """Forget every anchor of ``keys`` (a group refuted them)."""
        n = 0
        for key in dict.fromkeys(keys):
            rec = self.images.get(key)
            if rec is not None and rec["anchors"]:
                n += len(rec["anchors"])
                rec["anchors"] = []
        if n:
            self.dirty = True
        return n


def write_snapshot(path: str, text: str) -> None:
    """Atomic replace (tmp + rename; the store walk removes a leftover
    ``.tmp.`` file and never reads this one)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("WEG2 MM-PERSIST write to %s failed: %s (the next boot reads the "
                       "previous file)", path, exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def preimage_verified(store_depth: int, first_image: int, page: int) -> Tuple[bool, int]:
    """(verified, need): the live store proves every FULL page before the
    first image (``need`` tokens). A first image inside page 0 needs nothing
    -- there is nothing the front can ask the store before it."""
    page = max(1, int(page))
    need = max(0, int(first_image)) // page * page
    return int(store_depth) >= need, need
