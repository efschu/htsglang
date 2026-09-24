"""THE 27B LINE'S CALIBRATION IDENTITY: same checkpoint AND same line.

User order 2026-09-24 11:4xZ (verbatim): "warum nutzt du ueberhaupt den nf
planer? ich hab doch gesagt getrennt von nf und separat fuers 27b ... die nf
werte passen nicht fuers 27b, wie sollten sie auch. und umgekehrt."

This line (desk/27b-up-line-0924) starts from 829ebd09f8, the tree of weg2xsn411
-- the last 27B boot that worked. Its readers of MEASURED sources (the shared
sidecar ``weg2_measured_record.json``, the P logs the cut and the depth are
calibrated from) all take "the newest" entry, and since 2026-09-20 the newest
entries are Next-Flash boots (fnFL2x*) and 27B boots of the NF line
(weg2xsn412..419) -- measured on other code and, for fnFL2, another model.
xsn418 died W19 on exactly that: the #1444 D residue of fnFL2x144 (3080: 768
MiB) priced for a 27B D that leaves 1404.

A measured source is accepted only when BOTH hold:

1. its boot ran THIS checkpoint -- the ``--model-path`` basename in the boot's
   own group argv (front log) equals this boot's; and
2. its boot's commit is an ANCESTOR of the commit this launcher runs
   (``git merge-base --is-ancestor``): the boots of this line's own history --
   weg2xsn407..411 ran desk/dflash2-pick, whose tip 829ebd09f8 is this line's
   base -- and every later boot of this line. A tree of the NF line is never
   an ancestor of this one, so neither its 27B boots nor its NF boots count.

Everything that cannot be proven (no front log for a record's boot tag, no
model in it, git unreadable) is NOT accepted: an unproven provenance is the
exact mix-up this module exists to stop.

Torch-free; used by the launcher, the front and ring-side readers.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_LOG_RE = re.compile(
    r"^boot_weg2_(?P<tag>.+)_(?P<tip>[0-9a-f]{7,40})_(?P<day>\d{4})_(?P<time>\d{6})"
    r"\.(?P<kind>front|P|D)\.log$"
)
_MODEL_PATH_RE = re.compile(r"--model-path[= ]'?([^\s']+)")
_SCAN_MAX_BYTES = 16 << 20


def model_key(model: str) -> str:
    """A checkpoint's identity: its directory name."""
    return os.path.basename(str(model or "").rstrip("/").strip("'\""))


@lru_cache(maxsize=8192)
def _front_log_model(path: str, _mtime: float) -> Optional[str]:
    """The checkpoint a front log's group argv names (first --model-path)."""
    try:
        with open(path, "rb") as f:
            seen = 0
            for raw in f:
                seen += len(raw)
                if seen > _SCAN_MAX_BYTES:
                    return None
                if b"argv" in raw and b"--model-path" in raw:
                    m = _MODEL_PATH_RE.search(raw.decode("utf-8", "replace"))
                    if m:
                        return model_key(m.group(1))
    except OSError:
        return None
    return None


@lru_cache(maxsize=4096)
def _is_ancestor(repo: str, commit: str, head: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", repo, "merge-base", "--is-ancestor", commit, head],
            capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@lru_cache(maxsize=16)
def _front_logs_by_tag(evidence_dir: str, _stamp: int) -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """tag -> ((tip, front log path), ...), newest first. One listing."""
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    try:
        names = os.listdir(evidence_dir)
    except OSError:
        return {}
    for n in names:
        m = _LOG_RE.match(n)
        if m and m.group("kind") == "front":
            out.setdefault(m.group("tag"), []).append(
                (m.group("day") + m.group("time"), m.group("tip"), os.path.join(evidence_dir, n)))
    return {t: tuple((tip, p) for _, tip, p in sorted(v, reverse=True)) for t, v in out.items()}


@dataclass(frozen=True)
class LineIdentity:
    """The identity every measured source must share with this boot."""

    model: str          # checkpoint path (or its basename) this boot runs
    repo: str           # the tree this launcher runs (a git work tree)
    head: str           # the commit this launcher runs (``tip``)
    evidence_dir: str   # where the boots' own logs are

    @property
    def model_name(self) -> str:
        return model_key(self.model)

    def describe(self) -> str:
        return (f"checkpoint {self.model_name} AND a boot commit that is an ancestor of "
                f"{self.head} (this line's own history)")

    def accepts_boot(self, tip: str, front_log: Optional[str]) -> bool:
        if not front_log or not os.path.exists(front_log):
            return False
        try:
            mtime = os.path.getmtime(front_log)
        except OSError:
            return False
        if _front_log_model(front_log, mtime) != self.model_name:
            return False
        return _is_ancestor(self.repo, tip, self.head)

    def accepts_log(self, path: str) -> bool:
        """A ``boot_weg2_*.{front,P,D}.log``: its boot, judged by its own
        front log (the group logs carry no argv line of the launcher)."""
        m = _LOG_RE.match(os.path.basename(path))
        if not m:
            return False
        front = os.path.join(os.path.dirname(path),
                             os.path.basename(path)[: -len(f".{m.group('kind')}.log")] + ".front.log")
        return self.accepts_boot(m.group("tip"), front)

    def accepts_sample(self, sample: dict) -> bool:
        """A measured-record sample, by its ``boot_tag``: accepted when the
        newest front log of that tag in the evidence dir belongs to this line."""
        tag = str((sample or {}).get("boot_tag", "") or "")
        if not tag:
            return False
        try:
            stamp = int(os.path.getmtime(self.evidence_dir))
        except OSError:
            return False
        for tip, front in _front_logs_by_tag(self.evidence_dir, stamp).get(tag, ()):
            return self.accepts_boot(tip, front)
        return False

    def spec(self) -> str:
        """The one-string form the launcher hands the front (--record-line)."""
        return f"model={self.model}|repo={self.repo}|head={self.head}|evidence={self.evidence_dir}"


def parse_spec(spec: str) -> Optional[LineIdentity]:
    """Inverse of :meth:`LineIdentity.spec`; ``None`` for an empty/unusable one."""
    if not spec:
        return None
    kv = dict(p.split("=", 1) for p in str(spec).split("|") if "=" in p)
    try:
        return LineIdentity(model=kv["model"], repo=kv["repo"], head=kv["head"],
                            evidence_dir=kv["evidence"])
    except KeyError:
        return None
