"""What the running boot image carries, per seat (27B, NF).

The 27B operator keeps /spinning/gpu-arb/docs/image_changes.json at every delta
build (user order 28.09.: "im dashboard sollen auch die aktuellen aenderungen am
boot img aufgelistet werden und die erwartete verbesserung durch den fix"):
``images -> <rev, 10 chars> -> {rc, base, built_utc, changes[{id, who, title,
expected, status, evidence}]}``.  The rev a boot runs is the one its weg2
launcher line names (``tree=/opt/htsglang/src-nf @ f833fcbb2d``, parsed into
``meta.sha``/``meta.tree`` by live.py) -- the same rev the image's sglang
version carries.  Pure except for ``ImageChanges.load`` (one stat per call,
reread only when the file changed).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from . import redact

DEFAULT_PATH = "/spinning/gpu-arb/docs/image_changes.json"
STATUSES = ("erwartet", "unbelegt", "belegt", "wirkungslos")
SEATS = ("27B", "NF")
TEXT_FIELDS = ("id", "who", "title", "expected", "status", "evidence")


def seat_of(meta: dict) -> Optional[str]:
    """27B or NF from the boot's tree, else its form profile, else its evidence dir."""
    tree = (meta.get("tree") or "").rstrip("/")
    if tree.endswith("src-nf"):
        return "NF"
    if tree.endswith("src-27b"):
        return "27B"
    form = meta.get("form") or ""
    if "profile=nextflash" in form:
        return "NF"
    if "profile=qwen27b" in form:
        return "27B"
    parts = os.path.normpath(meta.get("dir") or "").split(os.sep)
    if "nf" in parts:
        return "NF"
    if "27b" in parts:
        return "27B"
    return None


def _running(b: dict) -> bool:
    return bool(b.get("live") or (b.get("container") or {}).get("State") == "running")


def current_boot_per_seat(boots) -> dict:
    """seat -> the boot that seat runs now (a live one, else the newest by its last log line)."""
    out = {}
    for b in boots or []:
        seat = seat_of(b.get("meta") or {})
        if seat is None or not (b.get("meta") or {}).get("sha"):
            continue
        cur = out.get(seat)
        key = (_running(b), b.get("last_log_t") or 0)
        if cur is None or key > (_running(cur), cur.get("last_log_t") or 0):
            out[seat] = b
    return out


def entry_for(images: dict, rev: str):
    """(key, entry) of the image entry for a rev; either side may be the longer hash."""
    if not rev or len(rev) < 7:
        return None, None
    for k, v in (images or {}).items():
        if k.startswith(rev) or rev.startswith(k):
            return k, v
    return None, None


def _clean_change(c: dict) -> dict:
    out = {f: redact.clean(str(c.get(f) or "")) or "" for f in TEXT_FIELDS}
    if out["status"] not in STATUSES:
        out["status_raw"], out["status"] = out["status"], "unbekannt"
    return out


def view(boots, images: Optional[dict], error: Optional[str] = None, path: str = DEFAULT_PATH) -> dict:
    seats = []
    cur = current_boot_per_seat(boots)
    for seat in SEATS:
        b = cur.get(seat)
        if b is None:
            continue
        m = b.get("meta") or {}
        key, e = entry_for(images or {}, m.get("sha") or "")
        seats.append({
            "seat": seat,
            "rev": m.get("sha"),
            "stem": b.get("stem"),
            "tag": m.get("tag"),
            "running": _running(b),
            "image": (b.get("container") or {}).get("Image"),
            "found": e is not None,
            "key": key,
            "rc": (e or {}).get("rc"),
            "base": (e or {}).get("base"),
            "built_utc": (e or {}).get("built_utc"),
            "changes": [_clean_change(c) for c in (e or {}).get("changes") or [] if isinstance(c, dict)],
        })
    return {"path": path, "error": error, "seats": seats}


class ImageChanges:
    """The json file, reread only when its mtime or size changed."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._sig = None
        self.images: dict = {}
        self.error: Optional[str] = None

    def load(self):
        try:
            st = os.stat(self.path)
        except OSError as e:
            self._sig, self.images, self.error = None, {}, "%s: %s" % (type(e).__name__, e)
            return self.images, self.error
        sig = (st.st_mtime_ns, st.st_size)
        if sig != self._sig:
            try:
                with open(self.path) as fh:
                    d = json.load(fh)
                imgs = d.get("images") if isinstance(d, dict) else None
                if not isinstance(imgs, dict):
                    raise ValueError("kein Objekt 'images'")
                self.images, self.error = imgs, None
            except (OSError, ValueError) as e:
                # keep the last good content; name the broken read
                self.error = "%s: %s" % (type(e).__name__, e)
            self._sig = sig
        return self.images, self.error
