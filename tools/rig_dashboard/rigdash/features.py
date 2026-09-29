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
                    raise ValueError("kein Array 'features'")
                self.features, self.error = [f for f in feats if isinstance(f, dict)], None
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


def boot_view(st: Optional[dict]) -> Optional[dict]:
    if not st:
        return None
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
        return {"state": "unbekannt", "detail": "kein Image-Rev im Zustand"}
    if not zweige:
        return {"state": "unbekannt", "detail": "kein Commit eingetragen"}
    if _git(repo, "cat-file", "-e", rev + "^{commit}").returncode != 0:
        return {"state": "unbekannt", "detail": "Image-Rev %s fehlt lokal in %s" % (rev, repo)}
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
        return {"state": "unbekannt", "detail": "Commit(s) fehlen lokal: " + ", ".join(missing)}
    return {"state": "nein", "detail": "weder Vorfahr noch patch-/subject-gleich auf " + rev}


# --------------------------------------------------------------------------- aktiv

def strip_comments(text: str) -> str:
    """Comments name switches in prose ("... _MIN_DWELL_EXCLUDE_DRAIN 1 ..."): never count them."""
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())


def _profile_text(profile: Optional[str]) -> Optional[str]:
    if not profile:
        return None
    for d in PROFILE_DIRS:
        p = os.path.join(d, profile if profile.endswith(".env") else profile + ".env")
        try:
            with open(p) as fh:
                return strip_comments(fh.read())
        except OSError:
            continue
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
                return {"state": "an" if _is_on(v, an_wert) else "aus", "src": "Profil (kein launch-Schnappschuss)", "value": v}
        else:
            m = re.search(r"(?<![\w-])%s(?:[ =]([^\s'\"]+))?" % re.escape(name), profile_text)
            if m:
                v = m.group(1) or ""
                on = _is_on(v, an_wert) if (an_wert not in (None, "")) else True
                return {"state": "an" if on else "aus", "src": "Profil (kein launch-Schnappschuss)", "value": v}
        return {"state": "an" if default_on else "aus", "src": "Profil (nicht gesetzt -> default)", "value": None}
    return {"state": "unbekannt", "src": "kein Zustand und kein Profil", "value": None}


def aktiv(feature: dict, st: Optional[dict], profile_text: Optional[str], im: dict) -> dict:
    sws = [s for s in feature.get("schalter") or [] if isinstance(s, dict) and s.get("name")]
    if not sws:
        if im.get("state") == "ja":
            return {"state": "an", "detail": [], "note": "ohne Schalter: wirkt, sobald im Image"}
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
            out.append("%s: ohne id" % where)
            continue
        if fid in seen:
            out.append("%s: id doppelt" % fid)
        seen.add(fid)
        if f.get("modell") not in MODELL_VALUES:
            out.append("%s: modell %r nicht in %s" % (fid, f.get("modell"), "/".join(MODELL_VALUES)))
        for z in f.get("zweige") or []:
            if not isinstance(z, dict) or not z.get("branch") or not re.fullmatch(r"[0-9a-f]{7,40}", z.get("sha") or ""):
                out.append("%s: Zweig %r braucht branch + sha" % (fid, z))
        for s in f.get("schalter") or []:
            if not isinstance(s, dict) or not s.get("name"):
                out.append("%s: Schalter ohne name" % fid)
                continue
            if s.get("art") not in SWITCH_ART:
                out.append("%s: Schalter %s art %r nicht env/flag" % (fid, s["name"], s.get("art")))
            if (s.get("gruppe") or "") not in SWITCH_GROUPS:
                out.append("%s: Schalter %s gruppe %r nicht P/D/beide/front/launcher" % (fid, s["name"], s.get("gruppe")))
            if s.get("default") not in ("an", "aus"):
                out.append("%s: Schalter %s ohne default an|aus" % (fid, s["name"]))
        for g in f.get("gewinn") or []:
            if not isinstance(g, dict) or not g.get("metrik"):
                out.append("%s: Gewinn ohne metrik" % fid)
                continue
            if g.get("art") not in GAIN_ART:
                out.append("%s: Gewinn %s art %r nicht %s" % (fid, g["metrik"], g.get("art"), "/".join(GAIN_ART)))
            gm = g.get("modell")
            if f.get("modell") == "beide" and gm not in ("27B", "NF"):
                out.append("%s: Gewinn '%s' ohne Feld modell (Pflicht bei modell=beide)" % (fid, g["metrik"]))
            elif gm and f.get("modell") in ("27B", "NF") and gm != f.get("modell"):
                out.append("%s: Gewinn '%s' modell %s gegen Feature-modell %s" % (fid, g["metrik"], gm, f.get("modell")))
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
            problems.append("%s: Gewinn '%s' ohne Feld modell (Pflicht bei modell=beide)" % (feature.get("id"), g.get("metrik")))
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
    file signature); until it is done the column says "wird berechnet" -- a
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
            rev = (bv or {}).get("rev") or ""
            mine = [f for f in feats if f.get("modell") in (model, "beide")]
            hit = self._im_for(model, rev, sig, mine)
            imc = (hit or {}).get("im") or {}
            ptxt = _profile_text((bv or {}).get("profile"))
            rows = []
            for f in mine:
                im = imc.get(f.get("id")) or (
                    {"state": "unbekannt", "detail": "wird berechnet"} if hit is None else {"state": "unbekannt"})
                if im.get("state") == "nein":
                    ak = {"state": "aus", "detail": [], "note": "nicht im Image"}
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
            models.append({"model": model, "boot": bv, "features": rows, "git_s": (hit or {}).get("s")})
        return {"path": self.file.path, "error": err, "problems": sorted(set(problems)), "models": models}
