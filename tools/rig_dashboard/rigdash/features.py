"""Feature table per model (user order 29.09.): what was built, whether it is
finished, in the running image, active in the running boot, what it brought.

"das dashboard muss jetzt eine liste mit features haben die entwickelt haben,
dahinter muss stehen ob schon fertig entwickelt sind, ob sie im image sind und ob
sie auch aktiv sind und was es gebracht hat an verbesserung [...] falls sie im
image sind und ausgeschaltet sind, dann begruendung. das muss immer aktuell
gehalten werden."

One source, /spinning/gpu-arb/docs/features.json, kept by the operators and the
builders (rigdash/features_update.py, deployed under /opt/rigdash/current). Two columns are NOT typed in
there but computed here, so they cannot go stale:

  * im Image  -- a feature's commit is an ancestor of the image rev the model's
    boot runs, OR a commit with the same ``git patch-id`` is on that line (27B
    picks NF branches under a new sha), OR -- weakest -- a commit with the same
    subject.  Computed once per (rev, feature file) and cached.
  * aktiv     -- from the boot's state.json (weg2.state/1), never from log text
    (user order against log IPC): ``groups.<G>.launch`` = {argv, env} written by
    the launcher; for an older image without that snapshot, the profile file the
    state names.  A switch absent from argv/env takes its ``default``.

Per model the table follows that model's running boot, or its last one ("letzter
Boot rc12z30j") -- never "aus" just because the model is not up right now.
Gains are strictly per model: an entry for both models carries ``modell`` on each
gain, a gain without it is flagged, never shown under both.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import subprocess
import threading
import time
from typing import Optional

from . import redact

DEFAULT_PATH = "/spinning/gpu-arb/docs/features.json"
DEFAULT_REPO = "/spinning/htsglang"
STATE_ROOTS = {"NF": "/spinning/docker-acceptance/nf/state", "27B": "/spinning/docker-acceptance/27b/state"}
PROFILE_DIRS = ("/spinning/gpu-arb/docker/profiles", "/spinning/gpu-arb/docker")
MODELS = ("NF", "27B")
LIVE_STATES = ("launching", "loading", "ready", "serving", "flipping")
GAIN_ART = ("gemessen", "gerechnet", "unbelegt")
OFF_VALUES = ("", "0", "false", "off", "no", "none", "aus")
GIT_TIMEOUT_S = 60


# --------------------------------------------------------------------------- file

class FeatureFile:
    """features.json, reread only when its mtime or size changed."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._sig = None
        self.features: list = []
        self.produkt: list = []
        self.boot_overrides: dict = {}
        self.updated_utc: Optional[str] = None
        self.mtime: Optional[float] = None
        self.error: Optional[str] = None

    def load(self):
        try:
            st = os.stat(self.path)
        except OSError as e:
            self._sig, self.features, self.error = None, [], "%s: %s" % (type(e).__name__, e)
            return self.features, self.error, None
        sig = (st.st_mtime_ns, st.st_size)
        if sig != self._sig:
            try:
                with open(self.path) as fh:
                    d = json.load(fh)
                feats = d.get("features") if isinstance(d, dict) else None
                if not isinstance(feats, list):
                    raise ValueError("no array 'features'")
                prod = d.get("produkt")
                self.features, self.error = [f for f in feats if isinstance(f, dict)], None
                self.produkt = [x for x in prod if isinstance(x, dict)] if isinstance(prod, list) else []
                ov = d.get("boot_overrides")
                self.boot_overrides = ov if isinstance(ov, dict) else {}
                self.updated_utc = d.get("updated_utc") if isinstance(d.get("updated_utc"), str) else None
                self.mtime = st.st_mtime
            except (OSError, ValueError) as e:
                self.error = "%s: %s" % (type(e).__name__, e)   # keep the last good content
            self._sig = sig
        return self.features, self.error, self._sig


# --------------------------------------------------------------------------- boot

def rc_of_image(image: str) -> Optional[str]:
    m = re.search(r"-(rc[0-9][0-9a-z.]*)-", image or "")
    return m.group(1) if m else None


def last_boot(root: str) -> Optional[dict]:
    """The model's running or last boot: state.json behind ``current``, else the newest boot dir."""
    cand = []
    cur = os.path.join(root, "current")
    if os.path.isdir(cur):
        cand.append(cur)
    try:
        dirs = [os.path.join(root, n) for n in os.listdir(root) if n != "current"]
        dirs.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        cand += dirs
    except OSError:
        pass
    for d in cand:
        try:
            with open(os.path.join(d, "state.json")) as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            continue
        if st.get("kind") == "boot":
            return st
    return None


def _boot_start(boot_id: str) -> Optional[float]:
    m = re.search(r"-(\d{8}T\d{6})Z-", boot_id or "")
    if not m:
        return None
    return float(calendar.timegm(time.strptime(m.group(1), "%Y%m%dT%H%M%S")))


def boot_view(st: Optional[dict]) -> Optional[dict]:
    if not st:
        return None
    start, ready = _boot_start(st.get("boot_id") or ""), st.get("serving_since_ts")
    lc = (st.get("lifecycle") or {}).get("state")
    groups = st.get("groups") or {}
    return {
        "boot_id": st.get("boot_id"),
        "rev": st.get("rev"),
        "image": st.get("image"),
        "rc": rc_of_image(st.get("image") or ""),
        "profile": st.get("profile"),
        "lifecycle": lc,
        "running": lc in LIVE_STATES,
        "launch_groups": sorted(g for g, v in groups.items() if isinstance(v, dict) and v.get("launch")),
        "boot_s": round(ready - start) if start and ready and ready > start else None,
        "start_ts": start,
    }


# --------------------------------------------------------------------------- im Image

def _git(repo: str, *args, input_: Optional[bytes] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], input=input_, capture_output=True, timeout=GIT_TIMEOUT_S)


def _patch_id(repo: str, sha: str) -> Optional[str]:
    show = _git(repo, "show", "--no-color", sha)
    if show.returncode != 0:
        return None
    pid = _git(repo, "patch-id", "--stable", input_=show.stdout)
    out = pid.stdout.decode(errors="replace").split()
    return out[0] if out else None


class LineIndex:
    """patch-ids and subjects of the commits on an image rev since a date (one git log per rev)."""

    def __init__(self, repo: str, rev: str, since: str):
        self.rev, self.since = rev, since
        self.patch_ids: dict = {}
        self.subjects: dict = {}
        self.error: Optional[str] = None
        log = _git(repo, "log", "-p", "--no-color", "--no-merges", "--since=" + since, rev)
        if log.returncode != 0:
            self.error = log.stderr.decode(errors="replace").strip()[:200] or "git log rc %d" % log.returncode
            return
        pid = _git(repo, "patch-id", "--stable", input_=log.stdout)
        for line in pid.stdout.decode(errors="replace").splitlines():
            p = line.split()
            if len(p) == 2:
                self.patch_ids.setdefault(p[0], p[1])
        subj = _git(repo, "log", "--no-merges", "--format=%H%x00%s", "--since=" + since, rev)
        for line in subj.stdout.decode(errors="replace").splitlines():
            h, _, s = line.partition("\x00")
            if s:
                self.subjects.setdefault(s, h)


def in_image(repo: str, rev: str, zweige: list, line: Optional[LineIndex]) -> dict:
    """{state: ja|nein|unbekannt, how: vorfahr|patch-id|subject, sha, line_sha, detail}."""
    if not rev:
        return {"state": "unbekannt", "detail": "no image rev in the state"}
    if not zweige:
        return {"state": "unbekannt", "detail": "no commit entered"}
    if _git(repo, "cat-file", "-e", rev + "^{commit}").returncode != 0:
        return {"state": "unbekannt", "detail": "Image rev %s is missing locally in %s" % (rev, repo)}
    missing = []
    for z in zweige:
        sha = (z or {}).get("sha") or ""
        if not sha:
            continue
        if _git(repo, "cat-file", "-e", sha + "^{commit}").returncode != 0:
            missing.append(sha)
            continue
        if _git(repo, "merge-base", "--is-ancestor", sha, rev).returncode == 0:
            return {"state": "ja", "how": "vorfahr", "sha": sha}
        if line is not None and not line.error:
            pid = _patch_id(repo, sha)
            if pid and pid in line.patch_ids:
                return {"state": "ja", "how": "patch-id", "sha": sha, "line_sha": line.patch_ids[pid][:10]}
            subj = _git(repo, "log", "-1", "--format=%s", sha).stdout.decode(errors="replace").strip()
            if subj and subj in line.subjects:
                return {"state": "ja", "how": "subject", "sha": sha, "line_sha": line.subjects[subj][:10]}
    if missing and len(missing) == len([z for z in zweige if (z or {}).get("sha")]):
        return {"state": "unbekannt", "detail": "Commit(s) missing locally: " + ", ".join(missing)}
    return {"state": "nein", "detail": "neither ancestor nor patch/subject-equal on " + rev}


# --------------------------------------------------------------------------- Stand (Nutzer 01.10.)
# "Die Features-Soll/Ist-Liste und die Bausteinliste sind ewig nicht aktualisiert ... sorge dafür, dass sie
# künftig nicht wieder veralten": der Stand der Datei wird angezeigt und gegen den letzten Boot und die Uhr
# geprüft, und die Commits der Image-Linie, die in keinem Baustein stehen, rechnet rigdash selbst aus git.
STAND_WARN_H = 12.0
STAND_BAD_H = 24.0
NEW_COMMITS_H = 48


def newest_ist(produkt: list) -> Optional[str]:
    best, best_ts = None, None
    for p in produkt or []:
        for x in (p.get("ist") or {}).values():
            ts = belegt_ts((x or {}).get("belegt_am")) if isinstance(x, dict) else None
            if ts is not None and (best_ts is None or ts > best_ts):
                best, best_ts = x.get("belegt_am"), ts
    return best


def stand_view(file: "FeatureFile", boot_starts: dict, now: Optional[float] = None) -> dict:
    """How old the list is, and whether a boot ran after its last entry (yellow > STAND_WARN_H or a newer boot,
    red > STAND_BAD_H).  The timestamp is the file's own ``updated_utc``, else its mtime."""
    now = now or time.time()
    upd = belegt_ts(file.updated_utc) if file.updated_utc else None
    t = upd if upd is not None else file.mtime
    age_h = (now - t) / 3600.0 if t else None
    newer = {m: ts for m, ts in (boot_starts or {}).items() if ts and t and ts > t}
    level = ("bad" if age_h is not None and age_h > STAND_BAD_H else
             "warn" if (age_h is not None and age_h > STAND_WARN_H) or newer else "ok")
    return {"updated_utc": file.updated_utc, "t": t, "age_h": round(age_h, 1) if age_h is not None else None,
            "newest_ist": newest_ist(file.produkt), "boots_after": newer, "level": level,
            "warn_h": STAND_WARN_H, "bad_h": STAND_BAD_H}


def new_commits(repo: str, rev: str, feats: list, im: dict, hours: int = NEW_COMMITS_H, limit: int = 40) -> dict:
    """Commits on the image rev of the last ``hours`` that no Baustein names: not a ``zweige`` sha (prefix), not the
    line commit a Baustein was matched to by patch-id/subject (``im_image.line_sha``)."""
    if not rev:
        return {"rev": rev, "commits": [], "error": "no image rev"}
    out = _git(repo, "log", "--no-merges", "--since=%d hours ago" % hours, "--format=%H%x00%ct%x00%s", rev)
    if out.returncode != 0:
        return {"rev": rev, "commits": [], "error": out.stderr.decode(errors="replace").strip()[:200]}
    known = set()
    for f in feats:
        for z in f.get("zweige") or []:
            sha = (z or {}).get("sha") if isinstance(z, dict) else None
            if sha:
                known.add(sha[:10])
    for v in (im or {}).values():
        if isinstance(v, dict):
            for k in ("sha", "line_sha"):
                if v.get(k):
                    known.add(str(v[k])[:10])
    rows = []
    for line in out.stdout.decode(errors="replace").splitlines():
        h, _, rest = line.partition("\x00")
        ct, _, subj = rest.partition("\x00")
        if h[:10] in known:
            continue
        try:
            t = int(ct)
        except ValueError:
            t = None
        rows.append({"sha": h[:10], "t": t, "subject": _clean(subj)})
    return {"rev": rev, "hours": hours, "commits": rows[:limit], "n": len(rows)}


# --------------------------------------------------------------------------- aktiv

def strip_comments(text: str) -> str:
    """Comments name switches in prose ("... _MIN_DWELL_EXCLUDE_DRAIN 1 ..."): never count them."""
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())


_SOURCE_RE = re.compile(r'^\s*(?:source|\.)\s+"\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/([^"]+)"', re.M)


def _profile_text(profile: Optional[str], _depth: int = 0) -> Optional[str]:
    """The profile's own text, followed by the profiles it sources next to itself (27b-row-authority ->
    27b-release-draft -> 27b.env). Own lines come first, so a first-match search sees the override."""
    if not profile or _depth > 4:
        return None
    for d in PROFILE_DIRS:
        p = os.path.join(d, profile if profile.endswith(".env") else profile + ".env")
        try:
            with open(p) as fh:
                text = strip_comments(fh.read())
        except OSError:
            continue
        parts = [text]
        for name in _SOURCE_RE.findall(text):
            sub = _profile_text(os.path.join(os.path.dirname(os.path.relpath(p, d)), name), _depth + 1)
            if sub:
                parts.append(sub)
        return "\n".join(parts)
    return None


def _is_on(value, an_wert) -> bool:
    if an_wert is not None and str(an_wert) != "":
        return str(value) == str(an_wert)
    return str(value).strip().lower() not in OFF_VALUES


def _flag_value(argv: list, name: str):
    """(present, value) of a flag in argv: ``--x v`` or ``--x=v``."""
    for i, a in enumerate(argv):
        if a == name:
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            return True, ("" if nxt.startswith("--") else nxt)
        if a.startswith(name + "="):
            return True, a[len(name) + 1:]
    return False, None


def switch_state(sw: dict, st: Optional[dict], profile_text: Optional[str]) -> dict:
    """{state: an|aus|unbekannt, src, value} of one switch in the boot."""
    name = sw.get("name") or ""
    art = sw.get("art") or ("flag" if name.startswith("--") else "env")
    gruppe = sw.get("gruppe") or ""
    default_on = str(sw.get("default") or "aus").lower() in ("an", "on", "1", "true")
    an_wert = sw.get("an_wert")
    groups = (st or {}).get("groups") or {}
    targets = [gruppe] if gruppe in ("P", "D") else (["P", "D"] if gruppe in ("", "beide") else [])
    launches = [(g, (groups.get(g) or {}).get("launch")) for g in targets]
    launches = [(g, l) for g, l in launches if isinstance(l, dict)]
    if launches:
        vals = []
        for g, l in launches:
            if art == "env":
                env = l.get("env") or {}
                present, value = (name in env), env.get(name)
            else:
                present, value = _flag_value([str(a) for a in l.get("argv") or []], name)
            if not present:
                on = default_on
            elif art == "flag" and not value and an_wert in (None, ""):
                on = True                       # a bare flag is its own value
            else:
                on = _is_on(value, an_wert)
            vals.append((g, on, value if present else None))
        on_all = all(v[1] for v in vals)
        on_any = any(v[1] for v in vals)
        state = "an" if on_all else ("teilweise" if on_any else "aus")
        return {"state": state, "src": "state.json launch " + ",".join(g for g, _, _ in vals),
                "value": "; ".join("%s=%s" % (g, v if v is not None else "(default %s)" % ("an" if default_on else "aus"))
                                   for g, _, v in vals)}
    if profile_text is not None:
        if art == "env":
            hits = re.findall(r"(?<![A-Za-z0-9_])%s=([^;\"'\s]*)" % re.escape(name), profile_text)
            if hits:
                v = hits[-1]
                return {"state": "an" if _is_on(v, an_wert) else "aus", "src": "Profile (no launch snapshot)", "value": v}
        else:
            m = re.search(r"(?<![\w-])%s(?:[ =]([^\s'\"]+))?" % re.escape(name), profile_text)
            if m:
                v = m.group(1) or ""
                on = _is_on(v, an_wert) if (an_wert not in (None, "")) else True
                return {"state": "an" if on else "aus", "src": "Profile (no launch snapshot)", "value": v}
        return {"state": "an" if default_on else "aus", "src": "Profile (not set -> default)", "value": None}
    return {"state": "unbekannt", "src": "no state and no profile", "value": None}


def aktiv(feature: dict, st: Optional[dict], profile_text: Optional[str], im: dict) -> dict:
    sws = [s for s in feature.get("schalter") or [] if isinstance(s, dict) and s.get("name")]
    if not sws:
        if im.get("state") == "ja":
            return {"state": "an", "detail": [], "note": "without switch: takes effect as soon as it is in the image"}
        return {"state": "aus" if im.get("state") == "nein" else "unbekannt", "detail": []}
    det = [dict(switch_state(s, st, profile_text), name=s.get("name"), gruppe=s.get("gruppe")) for s in sws]
    states = {d["state"] for d in det}
    if states == {"an"}:
        state = "an"
    elif "unbekannt" in states:
        state = "unbekannt"
    elif states <= {"aus"}:
        state = "aus"
    else:
        state = "teilweise"
    return {"state": state, "detail": det}


# --------------------------------------------------------------------------- view

def _clean(v):
    return redact.clean(str(v)) if v is not None else None


MODELL_VALUES = ("27B", "NF", "beide")
SWITCH_ART = ("env", "flag")
SWITCH_GROUPS = ("P", "D", "beide", "front", "launcher", "")   # front/launcher: not in a group's launch -> profile


def validate(feats: list) -> list:
    """Problems of the file as a whole (the CLI refuses to write a file with any)."""
    out, seen = [], set()
    for i, f in enumerate(feats):
        fid = f.get("id") if isinstance(f, dict) else None
        where = fid or "#%d" % i
        if not isinstance(f, dict) or not fid:
            out.append("%s: without id" % where)
            continue
        if fid in seen:
            out.append("%s: duplicate id" % fid)
        seen.add(fid)
        if f.get("modell") not in MODELL_VALUES:
            out.append("%s: model %r not in %s" % (fid, f.get("modell"), "/".join(MODELL_VALUES)))
        for z in f.get("zweige") or []:
            if not isinstance(z, dict) or not z.get("branch") or not re.fullmatch(r"[0-9a-f]{7,40}", z.get("sha") or ""):
                out.append("%s: branch %r needs branch + sha" % (fid, z))
        for s in f.get("schalter") or []:
            if not isinstance(s, dict) or not s.get("name"):
                out.append("%s: switch without name" % fid)
                continue
            if s.get("art") not in SWITCH_ART:
                out.append("%s: switch %s art %r not env/flag" % (fid, s["name"], s.get("art")))
            if (s.get("gruppe") or "") not in SWITCH_GROUPS:
                out.append("%s: switch %s gruppe %r not P/D/beide/front/launcher" % (fid, s["name"], s.get("gruppe")))
            if s.get("default") not in ("an", "aus"):
                out.append("%s: switch %s without default an|aus" % (fid, s["name"]))
        for g in f.get("gewinn") or []:
            if not isinstance(g, dict) or not g.get("metrik"):
                out.append("%s: gain without metrik" % fid)
                continue
            if g.get("art") not in GAIN_ART:
                out.append("%s: gain %s art %r not %s" % (fid, g["metrik"], g.get("art"), "/".join(GAIN_ART)))
            gm = g.get("modell")
            if f.get("modell") == "beide" and gm not in ("27B", "NF"):
                out.append("%s: gain '%s' without field modell (required for modell=beide)" % (fid, g["metrik"]))
            elif gm and f.get("modell") in ("27B", "NF") and gm != f.get("modell"):
                out.append("%s: gain '%s' modell %s against feature modell %s" % (fid, g["metrik"], gm, f.get("modell")))
    return out


# --------------------------------------------------------------------------- Produkt-Features

# Nutzer-Rüge 29.09.: "Bugfixes sind keine Features." The table is the product
# features with Soll and Ist per model; the commits/fixes above are only their
# Bausteine (building blocks), shown collapsed under the feature they serve.
PRODUKT_STATUS = ("fertig+aktiv", "im Image aber aus", "Desk", "offen", "unbelegt", "entfällt")
KREUZ_ACHSEN = (("tp", "uneven TP"), ("dcp", "uneven DCP / token cut"),
                ("moe", "uneven expert shard (--rank-moe-ratio)"), ("pp", "PP cut uneven"),
                ("forma", "Form A host/worker"), ("kvonly", "KV-only rank"))
# "Soll erreicht?" (Koordinator 29.09.) is separate from the status: fertig+aktiv only says it runs.
SOLL_ERREICHT = ("ja", "teilweise", "nein")
KREUZ_STATUS = ("am Metall belegt", "unterstützt", "nur Desk", "nein", "unbelegt")
_KREUZ_KEYS = [k for k, _ in KREUZ_ACHSEN]


def kreuz_key(a: str, b: str) -> str:
    """One key per unordered pair of axes, in axis order ('tp+dcp', diagonal 'pp+pp')."""
    ia, ib = _KREUZ_KEYS.index(a), _KREUZ_KEYS.index(b)
    return "%s+%s" % (_KREUZ_KEYS[min(ia, ib)], _KREUZ_KEYS[max(ia, ib)])


def validate_produkt(prod: list, baustein_ids) -> list:
    out, seen = [], set()
    for i, p in enumerate(prod):
        pid = p.get("id") if isinstance(p, dict) else None
        if not pid:
            out.append("Product #%d: without id" % i)
            continue
        if pid in seen:
            out.append("Product %s: duplicate id" % pid)
        seen.add(pid)
        if not p.get("titel") or not p.get("soll"):
            out.append("Product %s: titel and soll are required" % pid)
        for m, x in (p.get("ist") or {}).items():
            if m not in MODELS:
                out.append("Product %s: ist model %r not 27B/NF" % (pid, m))
            elif (x or {}).get("status") not in PRODUKT_STATUS:
                out.append("Product %s: %s status %r not %s" % (pid, m, (x or {}).get("status"), "/".join(PRODUKT_STATUS)))
            elif (x or {}).get("erreicht") and x["erreicht"] not in SOLL_ERREICHT:
                out.append("Product %s: %s erreicht %r not %s" % (pid, m, x["erreicht"], "/".join(SOLL_ERREICHT)))
            elif (x or {}).get("belegt_am") and belegt_ts(x["belegt_am"]) is None:
                out.append("Product %s: %s belegt_am %r (ISO, e.g. 2026-09-29T07:10Z)" % (pid, m, x["belegt_am"]))
        for b in p.get("bausteine") or []:
            if b not in baustein_ids:
                out.append("Product %s: building block %s is not in features" % (pid, b))
        for m, cells in ((p.get("kreuztabelle") or {}).get("zellen") or {}).items():
            for k, c in (cells or {}).items():
                a, _, b = k.partition("+")
                if a not in _KREUZ_KEYS or b not in _KREUZ_KEYS or kreuz_key(a, b) != k:
                    out.append("Product %s: cross cell %r (axes %s, key in axis order)" % (pid, k, ",".join(_KREUZ_KEYS)))
                elif (c or {}).get("status") not in KREUZ_STATUS:
                    out.append("Product %s: cross %s %s status %r" % (pid, m, k, (c or {}).get("status")))
        for f in (p.get("untertabelle") or {}).get("zeilen") or []:
            if not f.get("name"):
                out.append("Product %s: sub-row without name" % pid)
            for m in MODELS:
                ba = ((f.get("ist") or {}).get(m) or {}).get("belegt_am")
                if ba and belegt_ts(ba) is None:
                    out.append("Product %s: row %s %s belegt_am %r (ISO, e.g. 2026-09-29T07:10Z)" % (pid, f.get("name"), m, ba))
                st = ((f.get("ist") or {}).get(m) or {}).get("status")
                if st is not None and st not in PRODUKT_STATUS:
                    out.append("Product %s: row %s %s status %r" % (pid, f.get("name"), m, st))
        for m, cells in ((p.get("matrix") or {}).get("zellen") or {}).items():
            for k in cells or {}:
                if matrix_key_error(k):
                    out.append("Product %s: matrix cell %s %r: %s" % (pid, m, k, matrix_key_error(k)))
    return out


def unassigned_bausteine(prod: list, baustein_ids) -> list:
    """A hint for the dashboard, not a save blocker: a new Baustein must not be refused only because
    nobody linked it yet (features_update.py set --produkt F8 links it in the same step)."""
    if not prod:
        return []
    used = {b for p in prod if isinstance(p, dict) for b in p.get("bausteine") or []}
    return ["Building block %s belongs to no product feature (set --produkt Fx)" % b
            for b in sorted(set(baustein_ids) - used)]


# Decode matrix (Nutzer 29.09.): Form x bs x Tiefe x Textart; a cell is only ever a
# measurement with boot and Beleg -- an empty cell is "ungemessen", never interpolated.
# The form is free text per model (NF: "Form A ohne Schnitt" / "uneven DCP mit Schnitt";
# 27B: its own D forms); "gemischt" = measured under agent load, depth or text not known --
# such a value is a margin value and never stands in a depth cell.  A cell is either a
# value or "ungültig" (27B rule: a stream ended by EOS below 500 tokens); no cell = ungemessen.
MATRIX_BS = ("1", "2", "3", "4", "5", "6")
MATRIX_TIEFE = ("kurz", "2k", "10k", "32k", "97k", "257k", "gemischt")
MATRIX_TEXT = ("code", "prosa", "thinking", "gemischt")
MATRIX_ZELLSTATUS = ("wert", "ungültig")


def matrix_key(form, bs, tiefe, text) -> str:
    return "%s|%s|%s|%s" % (form, bs, tiefe, text)


def matrix_key_error(k: str) -> Optional[str]:
    parts = k.split("|")
    if len(parts) != 4 or not parts[0]:
        return "Key form|bs|depth|text"
    for name, allowed, v in (("bs", MATRIX_BS, parts[1]), ("tiefe", MATRIX_TIEFE, parts[2]),
                             ("text", MATRIX_TEXT, parts[3])):
        if v not in allowed:
            return "%s=%r not in %s" % (name, v, "/".join(allowed))
    return None


# --------------------------------------------------------------------------- Wert im aktuellen Boot

def _fmt_s(ms) -> str:
    return "—" if ms is None else ("%.2f s" % (ms / 1000.0)).replace(".", ",")


def _cur_flip(lb, fb, gpus):
    fz = lb.get("flip_zeit") or {}
    pd, dp = fz.get("P>D") or {}, fz.get("D>P") or {}
    if not pd.get("n") and not dp.get("n"):
        return None
    win = (fz.get("window") or {}).get("label") or "Window missing"
    return ("P→D median %s, p90 %s (n=%s); D→P median %s (n=%s); flips %s" % (
        _fmt_s(pd.get("p50_ms")), _fmt_s(pd.get("p90_ms")), pd.get("n"), _fmt_s(dp.get("p50_ms")), dp.get("n"),
        lb.get("flip_count")), "Flip time (flipzeit.py, %s): last P chunk / last decode token -> first decode token / first prefill chunk" % win)


def _cur_decode(lb, fb, gpus):
    dec = (lb.get("decode") or {}).get("D") or {}
    rb = dec.get("round_ms_by_bs") or {}
    parts = ["bs%s %s ms (n=%s)" % (bs, str(v["median_ms"]).replace(".", ","), v["n"]) for bs, v in rb.items()]  # Mittel
    if not parts and dec.get("gen_tps_last") is None:
        return None
    return ("Round " + (", ".join(parts) or "—") + "; gen %s tok/s" % dec.get("gen_tps_last"),
            "rankstats decode.gpu_ms_by_bs (mean per bs, depth/text mixed) + Δdecode.tokens (IPC)")


def _cur_prefill(lb, fb, gpus):
    pp = (lb.get("prefill") or {}).get("P") or {}
    lb_ = pp.get("last_burst") or {}
    if not lb_.get("tps"):
        return None
    return ("last burst %s tok/s (slowest stage, %s chunks, avg %s tok)" % (
        round(lb_["tps"]), lb_.get("chunks"), round(lb_.get("mean_chunk") or 0)),
        "rankstats Δprefill.new_tokens / Δcompute_ms per rank (IPC)")


def _cur_spec(lb, fb, gpus):
    dec = (lb.get("decode") or {}).get("D") or {}
    if dec.get("accept_len") is None:
        return None
    return ("Acceptance length %s (rate %s)" % (dec.get("accept_len"), dec.get("accept_rate")),
            "rankstats decode.accept_len_ewma (IPC)")


def _cur_kv(lb, fb, gpus):
    dec = (lb.get("decode") or {}).get("D") or {}
    if dec.get("max_total_tokens") is None:
        return None
    return ("D max_total_num_tokens %s; idle GiB: no instrument (marker: free KV VRAM per rank)"
            % dec["max_total_tokens"], "rankstats cap.kv_tokens TP0 (IPC)")


def _cur_seats(lb, fb, gpus):
    dec = (lb.get("decode") or {}).get("D") or {}
    s, rb = dec.get("seats") or {}, dec.get("round_bs") or {}
    if not s and not rb:
        return None
    return ("Seats %s of %s, round bs %s" % (s.get("n"), s.get("cap"), rb.get("bs")),
            "state.json front.d_seats/d_phase_n + rankstats cap.seats (IPC)")


def _cur_form(lb, fb, gpus):
    form = (lb.get("meta") or {}).get("form")
    return (form, "state.json groups.<G>.form (IPC)") if form else None


def _cur_boot(lb, fb, gpus):
    return ("%s s until serving" % fb["boot_s"], "state.json serving_since_ts − boot start (boot_id)") \
        if fb and fb.get("boot_s") else None


def _cur_lifecycle(lb, fb, gpus):
    if not fb:
        return None
    return ("lifecycle %s%s" % (fb.get("lifecycle"), " (override: %s)" % fb["override_beleg"] if fb.get("override_beleg") else ""),
            "state.json lifecycle")


def _cur_transport(lb, fb, gpus):
    t = (lb.get("ipc") or {}).get("transport")
    if t:
        return ("Transport %s" % t, "state.json groups.launch.env HTSGLANG_TRANSPORT")
    tag = ((lb.get("meta") or {}).get("tag") or lb.get("stem") or "").lower()
    t = "bar1" if "bar1" in tag else ("nccl" if "nccl" in tag else None)
    return ("Transport %s" % t, "Boot tag (state.json launch.env HTSGLANG_TRANSPORT from z30s)") if t else None


def _cur_power(lb, fb, gpus):
    cards = (gpus or {}).get("value") or []
    w = [c.get("power.draw") for c in cards if c.get("power.draw") is not None]
    if not w:
        return None
    return ("Cards now %d W (%s); idle watts: no instrument" % (round(sum(w)), " / ".join("%d" % x for x in w)),
            "nvidia-smi power.draw (instantaneous value, not idle)")


def _cur_context(lb, fb, gpus):
    return ("Context per request %s" % fb["max_kv"], "Profile --max-kv-per-request") if fb and fb.get("max_kv") else None


def _cur_api(lb, fb, gpus):
    served = (lb.get("front") or {}).get("served")
    if isinstance(served, dict) and served:
        return ("%s requests served (P %s / D %s)" % (served.get("D", 0), served.get("P", 0), served.get("D", 0)),
                "Front /weg2/state served")
    tot = lb.get("totals") or {}
    return ("%s requests served" % tot["served_requests"], "state.json front.served (IPC)") if tot.get("served_requests") else None


def _cur_format(lb, fb, gpus):
    return ("running: %s" % fb["format"], "Profile PROFILE_FORMAT") if fb and fb.get("format") else None


CURRENT = {"F1": _cur_flip, "F2": _cur_form, "F3": _cur_transport, "F5": _cur_kv, "F11": _cur_power,
           "F12": _cur_format, "F13": _cur_spec, "F14": _cur_context, "F15": _cur_seats, "F17": _cur_boot,
           "F20": _cur_lifecycle, "F21": _cur_api, "F22": _cur_flip, "F23": _cur_prefill, "F24": _cur_decode,
           "F4": _cur_format}
MISSING = {"F6": "Expert rows per rank (marker MOE-POOL rows / LRU rows)",
           "F7": "Chunk plan per request (marker P-CHUNK-PLAN)",
           "F8": "Reshard per wake (marker D-SPEED / RESHARD)",
           "F9": "L3 hits per request (marker HiCacheFile index / STORE READ)",
           "F10": "Vision stage (marker W102 / VISION-STAGE)",
           "F16": "Prefix hits per follow-up turn (marker #cached-token per rid)",
           "F18": "Store occupancy (marker WEG2-HOST WATERMARK)",
           "F19": "Planner budget per rank (marker BUDGET-REACH / budget P/D)"}


# "Wert im aktuellen Boot" je Format (Nutzer 29.09.): NF INT4/NVFP4, 27B INT8/NVFP4/W4A8. W4A8 is the
# 3080 rank of an NVFP4 boot on the 27B (27B-Sitz), so it runs whenever NVFP4 runs.
AKTUELL_FORMATE = {"NF": ("INT4", "NVFP4"), "27B": ("INT8", "NVFP4", "W4A8")}


def _formats_running(model: str, profile_format: Optional[str]) -> set:
    f = (profile_format or "").lower().split("-")[0]
    run = {x for x in AKTUELL_FORMATE[model] if x.lower() == f}
    if model == "27B" and "NVFP4" in run:
        run.add("W4A8")
    return run


def _live_model(b: dict) -> str:
    text = " ".join(str(x) for x in ((b.get("meta") or {}).get("model"), (b.get("meta") or {}).get("tag"), b.get("stem")) if x)
    return "27B" if re.search(r"27b", text, re.IGNORECASE) else "NF"


def attach_current(fv: dict, live_boots: list, gpus: Optional[dict]) -> dict:
    """Fill produkt[].aktuell[model] from the newest log boot of that model (live.py) and its
    state.json boot (fv models): an instrument's value, or 'kein Instrument' naming the marker."""
    newest = {}
    for b in live_boots or []:
        m = _live_model(b)
        if m not in newest or (b.get("last_log_t") or 0) > (newest[m].get("last_log_t") or 0):
            newest[m] = b
    fboot = {m["model"]: m.get("boot") for m in fv.get("models") or []}
    for p in fv.get("produkt") or []:
        cur = {}
        for m in MODELS:
            lb, fb = newest.get(m) or {}, fboot.get(m)
            ex = CURRENT.get(p["id"])
            if ex is None:
                cur[m] = {"kein_instrument": MISSING.get(p["id"], "no marker named")}
                continue
            if not lb and not fb:
                cur[m] = {"leer": "no boot of this model found"}
                continue
            try:
                r = ex(lb, fb, gpus)
            except (KeyError, TypeError, ValueError) as e:
                r, cur[m] = None, {"kein_instrument": "Evaluation failed: %s" % e}
                continue
            tag = (lb.get("meta") or {}).get("tag") or lb.get("stem")
            cur[m] = {"wert": _clean(r[0]), "instrument": r[1], "boot": _clean(tag) or (fb or {}).get("rc")} if r \
                else {"leer": "Instrument present, no value yet in this boot" if lb else
                      "no IPC boot of this model in the last 6 h", "boot": _clean(tag)}
        for m in MODELS:
            fb = fboot.get(m)
            if not fb:
                continue
            run = _formats_running(m, fb.get("format"))
            cur[m]["format"] = _clean(fb.get("format"))
            cur[m]["je_format"] = {f: ("dieser Boot" if f in run else
                                       ("no boot in this format" if fb.get("format") else "boot format unknown"))
                                   for f in AKTUELL_FORMATE[m]}
        p["aktuell"] = cur
        for z in (p.get("untertabelle") or {}).get("zeilen") or []:
            z["aktuell"] = {}
            for m in MODELS:
                fb = fboot.get(m) or {}
                fmt = (fb.get("format") or "").lower().split("-")[0]
                if not fmt:
                    continue
                name = z.get("name", "").lower()
                if p["id"] == "F12":
                    z["aktuell"][m] = "runs in this boot" if fmt and fmt in name else "no boot in this format"
    return fv


def validate_doc(doc: dict) -> list:
    feats = doc.get("features") or []
    return validate(feats) + validate_produkt(doc.get("produkt") or [],
                                              {f.get("id") for f in feats if isinstance(f, dict)})


def _deep_clean(v):
    if isinstance(v, str):
        return _clean(v)
    if isinstance(v, list):
        return [_deep_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _deep_clean(x) for k, x in v.items()}
    return v


def belegt_ts(s) -> Optional[float]:
    """'2026-09-29T07:10Z' / '...T07:10:00Z' / '2026-09-29' -> epoch s (UTC); None if absent or unreadable."""
    if not isinstance(s, str) or not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d"):
        try:
            return float(calendar.timegm(time.strptime(s, fmt)))
        except ValueError:
            continue
    return None


def _alter(x: dict, start: Optional[float]) -> None:
    """27B/Nutzer 29.09.: every Soll/Ist cell carries 'zuletzt belegt'; a Beleg older than the model's
    last boot is stale ('Ist veraltet, neu messen') so new findings do not fall off the back."""
    ts = belegt_ts(x.get("belegt_am"))
    x["veraltet"] = None if ts is None or start is None else ts < start


def produkt_view(prod: list, bausteine: dict, boot_start: Optional[dict] = None) -> list:
    """bausteine: {model: {baustein id: computed row}} -- im Image / aktiv come from there.
    boot_start: {model: epoch s of the last boot} for the 'veraltet' mark."""
    boot_start = boot_start or {}
    out = []
    for p in sorted(prod, key=lambda x: (x.get("nr") is None, x.get("nr") or 0, str(x.get("id")))):
        ist = {}
        for m in MODELS:
            x = (p.get("ist") or {}).get(m) or {}
            ist[m] = {k: _clean(x.get(k)) for k in ("wert", "status", "grund", "beleg", "status_text", "quelle",
                                                    "belegt_am", "erreicht", "erreicht_grund")}
            ist[m]["status"] = ist[m]["status"] or "unbelegt"
            if ist[m]["status"] != "unbelegt" or ist[m]["wert"]:
                _alter(ist[m], boot_start.get(m))
        bs = []
        for bid in p.get("bausteine") or []:
            row = {"id": _clean(bid), "titel": None, "je_modell": {}}
            for m in MODELS:
                r = (bausteine.get(m) or {}).get(bid)
                if r is None:
                    continue
                row["titel"] = row["titel"] or r["titel"]
                row["je_modell"][m] = {"fertig": r["fertig"], "im_image": r["im_image"], "aktiv": r["aktiv"]["state"],
                                       "image_aber_aus": r["image_aber_aus"], "aus_begruendung": r["aus_begruendung"],
                                       "gewinn": r["gewinn"], "zweige": r["zweige"]}
            bs.append(row)
        out.append({"nr": p.get("nr"), "id": _clean(p.get("id")), "titel": _clean(p.get("titel")),
                    "soll": _clean(p.get("soll")), "ist": ist, "bausteine": bs,
                    "kreuztabelle": _deep_clean(p.get("kreuztabelle")) if p.get("kreuztabelle") else None,
                    "untertabelle": _deep_clean(p.get("untertabelle")) if p.get("untertabelle") else None,
                    "matrix": _deep_clean(p.get("matrix")) if p.get("matrix") else None,
                    "marker": _deep_clean(p.get("marker")) if p.get("marker") else None,
                    "aktuell": {}})
        for z in (out[-1]["untertabelle"] or {}).get("zeilen") or []:
            for m, x in (z.get("ist") or {}).items():
                if isinstance(x, dict):
                    _alter(x, boot_start.get(m))
    return out


def gains_for(feature: dict, model: str) -> tuple:
    """(gains of this model, problems). modell=beide needs modell on every gain."""
    out, problems = [], []
    both = (feature.get("modell") or "") == "beide"
    for g in feature.get("gewinn") or []:
        if not isinstance(g, dict):
            continue
        gm = g.get("modell")
        if both and not gm:
            problems.append("%s: gain '%s' without field modell (required for modell=beide)" % (feature.get("id"), g.get("metrik")))
            continue
        if gm and gm != model:
            continue
        art = g.get("art") if g.get("art") in GAIN_ART else "unbelegt"
        out.append({k: _clean(g.get(k)) for k in ("metrik", "vorher", "nachher", "einheit", "quelle", "boot")} | {"art": art})
    return out, problems


class Features:
    """features.json + the model boots -> the table.

    The git work (one ``git log -p | git patch-id`` over the image line, then one
    check per feature) runs in a background thread once per (model, image rev,
    file signature); until it is done the column says "being calculated" -- a
    request never waits on git.
    """

    def __init__(self, path: str = DEFAULT_PATH, repo: str = DEFAULT_REPO, state_roots: Optional[dict] = None,
                 background: bool = True):
        self.file = FeatureFile(path)
        self.repo = repo
        self.state_roots = state_roots or STATE_ROOTS
        self.background = background
        self._lock = threading.Lock()
        self._cache: dict = {}        # (model, rev, file sig) -> {"im": {id: im_image}, "s": seconds}
        self._busy: set = set()
        self._lines: dict = {}        # (rev, since) -> LineIndex

    def _since(self, feats: list) -> str:
        """Picks come after their original: index the line from the oldest feature commit on."""
        oldest = None
        for f in feats:
            for z in f.get("zweige") or []:
                sha = z.get("sha") if isinstance(z, dict) else None
                if not sha:
                    continue
                try:
                    ts = int(_git(self.repo, "log", "-1", "--format=%ct", sha).stdout.decode().strip())
                except ValueError:
                    continue
                oldest = ts if oldest is None else min(oldest, ts)
        if oldest is None:
            return "2026-09-01"
        return time.strftime("%Y-%m-%d", time.gmtime(oldest - 86400))

    def _compute(self, key: tuple, rev: str, feats: list):
        t0 = time.time()
        try:
            line = None
            if rev:
                lk = (rev, self._since(feats))
                line = self._lines.get(lk)
                if line is None:
                    line = LineIndex(self.repo, rev, lk[1])
                    if len(self._lines) > 8:
                        self._lines.clear()
                    self._lines[lk] = line
            im = {f.get("id"): in_image(self.repo, rev, f.get("zweige") or [], line) for f in feats}
        except (OSError, subprocess.SubprocessError) as e:
            im = {f.get("id"): {"state": "unbekannt", "detail": "git: %s" % e} for f in feats}
        with self._lock:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != key[0]}
            self._cache[key] = {"im": im, "s": round(time.time() - t0, 1)}
            self._busy.discard(key)

    def _im_for(self, model: str, rev: str, sig, feats: list) -> Optional[dict]:
        key = (model, rev, sig)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None or key in self._busy:
                return hit
            self._busy.add(key)
        if self.background:
            threading.Thread(target=self._compute, args=(key, rev, list(feats)), daemon=True,
                             name="rigdash-features-git").start()
            return None
        self._compute(key, rev, feats)
        with self._lock:
            return self._cache.get(key)

    def view(self) -> dict:
        feats, err, sig = self.file.load()
        models, problems = [], []
        for model in MODELS:
            st = last_boot(self.state_roots.get(model, ""))
            bv = boot_view(st)
            ov = self.file.boot_overrides.get((bv or {}).get("boot_id") or "")
            if bv and isinstance(ov, dict) and ov.get("lifecycle"):
                # a state.json read wrongly at the time (27B 29.09.: rc12z30j was the planned stop
                # after 30 min load, not a death) -- shown corrected, the file itself is never rewritten
                bv.update(lifecycle_state_json=bv["lifecycle"], lifecycle=_clean(str(ov["lifecycle"])),
                          override_beleg=_clean(ov.get("beleg")))
            rev = (bv or {}).get("rev") or ""
            mine = [f for f in feats if f.get("modell") in (model, "beide")]
            hit = self._im_for(model, rev, sig, mine)
            imc = (hit or {}).get("im") or {}
            ptxt = _profile_text((bv or {}).get("profile"))
            if bv and ptxt:
                fm = re.search(r"^PROFILE_FORMAT=([^\s#]+)", ptxt, re.M)
                kv = re.search(r"--max-kv-per-request[ =](\d+)", ptxt)
                bv.update(format=fm.group(1) if fm else None, max_kv=int(kv.group(1)) if kv else None)
            rows = []
            for f in mine:
                im = imc.get(f.get("id")) or (
                    {"state": "unbekannt", "detail": "being calculated"} if hit is None else {"state": "unbekannt"})
                if im.get("state") == "nein":
                    ak = {"state": "aus", "detail": [], "note": "not in the image"}
                else:
                    ak = aktiv(f, st, ptxt, im)
                gains, prob = gains_for(f, model)
                problems += prob
                rows.append({
                    "id": _clean(f.get("id")),
                    "titel": _clean(f.get("titel")),
                    "fertig": bool(f.get("fertig")),
                    "verantwortlich": _clean(f.get("verantwortlich")),
                    "zweige": [{"branch": _clean(z.get("branch")), "sha": _clean(z.get("sha"))}
                               for z in f.get("zweige") or [] if isinstance(z, dict)],
                    "im_image": im,
                    "aktiv": ak,
                    "gewinn": gains,
                    "aus_begruendung": _clean(f.get("aus_begruendung")),
                    "image_aber_aus": im.get("state") == "ja" and ak.get("state") in ("aus", "teilweise"),
                })
            nc = None
            if hit is not None and rev:
                key = ("new", model, rev, sig, int(time.time() // 600))      # git once per 10 min and rev
                with self._lock:
                    nc = self._cache.get(key)
                if nc is None:
                    try:
                        nc = new_commits(self.repo, rev, mine, imc)
                    except (OSError, subprocess.SubprocessError) as e:
                        nc = {"rev": rev, "commits": [], "error": "git: %s" % e}
                    with self._lock:
                        self._cache = {k: v for k, v in self._cache.items() if not (k[0] == "new" and k[1] == model)}
                        self._cache[key] = nc
            models.append({"model": model, "boot": bv, "features": rows, "git_s": (hit or {}).get("s"), "new_commits": nc})
        bs = {m["model"]: {r["id"]: r for r in m["features"]} for m in models}
        problems += validate_produkt(self.file.produkt, {f.get("id") for f in feats})
        problems += unassigned_bausteine(self.file.produkt, {f.get("id") for f in feats})
        stand = stand_view(self.file, {m["model"]: (m["boot"] or {}).get("start_ts") for m in models})
        return {"path": self.file.path, "error": err, "problems": sorted(set(problems)), "models": models, "stand": stand,
                "produkt": produkt_view(self.file.produkt, bs,
                                        {m["model"]: (m["boot"] or {}).get("start_ts") for m in models}),
                "kreuz_achsen": [{"key": k, "name": n} for k, n in KREUZ_ACHSEN]}
