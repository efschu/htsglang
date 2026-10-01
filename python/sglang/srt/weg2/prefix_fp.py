"""L15-18 (AP): prefix fingerprint ledger for the front's leg-1 line.

L3-RETURN stage 2, both lines. Monitor M2's L3-RETURN check judges whether
a restarted boot gets its long prompts back from the L3 store. A prompt the
front has NEVER seen before the restart is not a MISS -- it was never in
the store to come back from. So the front marks each leg-1 request with a
fingerprint of its leading prefix and says whether that fingerprint was
already known BEFORE this boot started.

Pure module: no GPU, no I/O at import, no sglang imports. The ledger file
is a plain newline list of 16-hex-char fingerprints, oldest first.
"""

import hashlib
import os
import struct
import tempfile
from collections import OrderedDict

# 16 hex chars, the sha1[:16] fingerprint format
_FP_LEN = 16
_HEX = set("0123456789abcdef")

# ledger cap: keep the newest 8192 fingerprints
CAP = 8192


def fingerprint(ids_or_text, n=4096):
    """sha1 hex[:16] over the leading prefix of a prompt.

    Token ids: the first ``n`` ids serialised as int64 little-endian bytes.
    Text: the first ``n * 4`` characters (utf-8). Prefix-only by design --
    a different tail beyond the window must not change the fingerprint.
    """
    if isinstance(ids_or_text, str):
        data = ids_or_text[: n * 4].encode("utf-8", errors="replace")
    else:
        data = b"".join(
            struct.pack("<q", int(t)) for t in list(ids_or_text)[:n]
        )
    return hashlib.sha1(data).hexdigest()[:_FP_LEN]


def _valid_fp(line):
    return len(line) == _FP_LEN and all(c in _HEX for c in line)


class PrefixSeen:
    """Which fingerprints existed BEFORE this boot.

    ``_loaded`` is the set read from the file at construction time (i.e.
    at boot start) and is never mutated afterwards -- not by add() and not
    by save(). That is what makes ``seen_before`` answer the monitor's
    question ("known before the restart?") instead of "known to this
    process right now": this boot's own additions never count as seen.
    """

    def __init__(self, path):
        self.path = str(path)
        self._loaded = self._load()
        self._boot = OrderedDict()

    def _load(self):
        # A missing or corrupt file yields an empty set; never raises.
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except OSError:
            return OrderedDict()
        out = OrderedDict()
        for line in raw.split(b"\n"):
            try:
                fp = line.decode("ascii").strip()
            except UnicodeDecodeError:
                continue  # a corrupt line is skipped, not fatal
            if _valid_fp(fp) and fp not in out:
                out[fp] = None
        while len(out) > CAP:  # keep the newest (file is oldest-first)
            out.popitem(last=False)
        return out

    def seen_before(self, fp):
        return fp in self._loaded

    def add(self, fp):
        self._boot[fp] = None
        while len(self._boot) > CAP:
            self._boot.popitem(last=False)

    def save(self):
        """Atomic merge of the boot-start set and this boot's additions.

        Newest entries are the boot's additions (file order: oldest first);
        the whole ledger is capped to the newest CAP entries. Writes a tmp
        file in the same directory and os.replace()s it into place; never
        raises (a failed save must not kill the serving path).
        """
        merged = OrderedDict()
        for fp in self._loaded:
            merged[fp] = None
        for fp in self._boot:
            merged.pop(fp, None)  # re-insert at the end: this boot is newest
            merged[fp] = None
        while len(merged) > CAP:
            merged.popitem(last=False)
        try:
            directory = os.path.dirname(self.path) or "."
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".prefix_fp_")
            try:
                with os.fdopen(fd, "w", encoding="ascii") as f:
                    for fp in merged:
                        f.write(fp + "\n")
                os.replace(tmp, self.path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except OSError:
            pass
