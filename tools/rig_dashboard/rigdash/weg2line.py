"""The rig's start form as a wizard answer: the host line for host_acceptance.sh.

Today a model is started on the Proxmox host, as root, inside a gpuq window:

    CTX=<ctx> IMAGE=<image> LINE=27b|nf PROFILE=<profile> PROFILE_MOUNT=0|1 \\
      [ALLOW_EXPERIMENTAL=1] HOUSE_GUARD=memlimit GPUQ_ID=<window> \\
      bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_acceptance.sh serve bar1|nccl

(the form the arms acc_cu130_rc12g_27b*.sh and acc_nf_rc12.sh pass on).  Nothing
here is hard-coded: the profiles are read from their files, the images from
``docker images`` on the host, the build context from its BUILD_INFO.json, and
which profiles count as release profiles is a service option.

Dry run WITHOUT a boot: host_acceptance.sh has no DRY mode, and it must never
be started for a check -- its EXIT trap stops and removes EVERY
``htsglang-acc-<line>-*`` container, i.e. the model that is serving right now,
and it cannot be edited in place while an arm runs it (bash reads scripts by
offset).  So ``dry_script`` assembles a check script FROM THE CURRENT FILE:
the parameter prelude up to the first side effect (``mkdir -p "$ACC"``), the
parameter checks, and the read-only house functions -- extracted by their
markers, so the dry run follows the script's own derivations instead of a
re-implementation, and fails loudly if the markers move.  It runs on the host
over ssh; it starts no container and touches no card (``docker image inspect``
only).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Dict, List, Optional

HOST_ROOT = "/spinning/subvol-999-disk-0"          # this machine's / as the host sees it
DOCKER_DIR = "/spinning/gpu-arb/docker"
HOST_ACCEPTANCE = DOCKER_DIR + "/host_acceptance.sh"
PROFILES_DIR = DOCKER_DIR + "/profiles"
CTX_GLOB = DOCKER_DIR + "/ctx/*/BUILD_INFO.json"
RE_IMAGE = re.compile(r"^htsglang:(cu\d+)-weg2-(.+?)-27b-nf(-flat)?$")
RE_ASSIGN = re.compile(r"^(PROFILE_[A-Z_]+)=(\"[^\"]*\"|'[^']*'|[^\s#]*)")
LINES = ("27b", "nf")
TRANSPORTS = ("bar1", "nccl")
HOUSE_GUARDS = ("memlimit", "ct999-ruht")
EXPERIMENTAL = ("experimentell", "formnachweis")


def read_profile_fields(path: str) -> Dict[str, str]:
    """PROFILE_* assignments as written in the file (no sourcing)."""
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                m = RE_ASSIGN.match(ln.strip())
                if m and m.group(1) not in out:
                    out[m.group(1)] = m.group(2).strip("\"'")
    except OSError:
        pass
    return out


def list_contexts(ctx_glob: str = CTX_GLOB) -> List[dict]:
    out = []
    for p in glob.glob(ctx_glob):
        try:
            with open(p) as fh:
                bi = json.load(fh)
        except (OSError, ValueError):
            continue
        d = os.path.dirname(p)
        out.append({
            "ctx": d, "release": bi.get("release"), "cuda": bi.get("cuda", "cu129"),
            "layout": bi.get("layout", "single"),
            "revisions": {k: (v or {}).get("revision") for k, v in (bi.get("lines") or {}).items()},
            "revision": bi.get("revision"), "mtime": os.path.getmtime(p),
        })
    out.sort(key=lambda c: -c["mtime"])
    return out


def parse_images(text: str) -> List[dict]:
    """``docker images --format '{{.Repository}}:{{.Tag}}\\t{{.ID}}\\t{{.CreatedAt}}'`` -> weg2 images."""
    out = []
    for ln in text.strip().splitlines():
        parts = ln.split("\t")
        m = RE_IMAGE.match(parts[0].strip())
        if not m:
            continue
        out.append({"image": parts[0].strip(), "cuda": m.group(1), "release": m.group(2),
                    "flat": bool(m.group(3)), "id": parts[1] if len(parts) > 1 else None,
                    "created": parts[2] if len(parts) > 2 else None})
    return out


def ctx_for_image(img: dict, contexts: List[dict]) -> Optional[dict]:
    for c in contexts:
        if c["release"] == img["release"] and c["cuda"] == img["cuda"]:
            return c
    return None


class Weg2Lines:
    def __init__(self, ssh: List[str], release_profiles: List[str], docker_dir: str = DOCKER_DIR,
                 host_root: str = HOST_ROOT, ctx_glob: str = CTX_GLOB):
        self.ssh = ssh
        self.release_profiles = release_profiles
        self.docker_dir = docker_dir
        self.host_root = host_root
        self.ctx_glob = ctx_glob
        self._images = (0.0, [], None)
        self._dry: Dict[tuple, dict] = {}
        self._dry_lock = threading.Lock()

    # ------------------------------------------------------------ inputs
    def images(self, max_age: float = 60.0):
        t, imgs, err = self._images
        if time.time() - t < max_age and (imgs or err):
            return imgs, err
        try:
            p = subprocess.run(self.ssh + ["docker images --format '{{.Repository}}:{{.Tag}}\\t{{.ID}}\\t{{.CreatedAt}}'"],
                               capture_output=True, text=True, timeout=20)
            if p.returncode != 0:
                raise RuntimeError(p.stderr.strip()[:200])
            imgs, err = parse_images(p.stdout), None
        except Exception as e:
            imgs, err = [], "%s: %s" % (type(e).__name__, e)
        self._images = (time.time(), imgs, err)
        return imgs, err

    def profile(self, name: str, ctx: Optional[dict]) -> dict:
        host_path = os.path.join(self.docker_dir, "profiles", name + ".env")
        img_path = os.path.join(ctx["ctx"], "tools", "profiles", name + ".env") if ctx else None
        in_image = bool(img_path and os.path.isfile(img_path))
        on_host = os.path.isfile(host_path)
        fields = read_profile_fields(img_path if in_image else host_path)
        differs = None
        if in_image and on_host:
            with open(img_path, "rb") as a, open(host_path, "rb") as b:
                differs = a.read() != b.read()
        line = fields.get("PROFILE_LINE") or ("nf" if name.startswith("nf") else "27b" if name.startswith("27b") else None)
        return {"name": name, "line": line, "in_image": in_image, "on_host": on_host,
                "differs_from_host": differs, "fields": fields,
                "status": fields.get("PROFILE_STATUS"),
                "nccl_status": fields.get("PROFILE_NCCL_STATUS"),
                "memavail_min_gib": fields.get("PROFILE_MEMAVAIL_MIN_GIB")}

    def options(self) -> dict:
        imgs, err = self.images()
        ctxs = list_contexts(self.ctx_glob)
        images = []
        for im in imgs:
            c = ctx_for_image(im, ctxs)
            images.append(dict(im, ctx=c["ctx"] if c else None, revisions=c["revisions"] if c else None))
        usable = [i for i in images if i["ctx"]]
        newest_ctx = next((c for c in ctxs if any(i["ctx"] == c["ctx"] for i in usable)), None)
        profiles = [self.profile(p, newest_ctx) for p in self.release_profiles]
        return {"images": images, "images_error": err, "default_image": usable[0]["image"] if usable else None,
                "profiles": profiles, "transports": list(TRANSPORTS), "house_guards": list(HOUSE_GUARDS),
                "host_acceptance": self.host_root + HOST_ACCEPTANCE}

    # ------------------------------------------------------------ the line
    def build(self, profile: str, image: str, transport: str = "bar1", house_guard: str = "memlimit") -> dict:
        """Validated inputs -> the host line.  Raises ValueError on anything not offered."""
        if profile not in self.release_profiles:
            raise ValueError("Profil %r ist kein Release-Profil (%s)" % (profile, ", ".join(self.release_profiles)))
        if transport not in TRANSPORTS:
            raise ValueError("Transport %r (bar1|nccl)" % transport)
        if house_guard not in HOUSE_GUARDS:
            raise ValueError("HOUSE_GUARD %r (memlimit|ct999-ruht)" % house_guard)
        imgs, err = self.images()
        im = next((i for i in imgs if i["image"] == image), None)
        if im is None:
            raise ValueError("Image %r steht nicht in 'docker images' auf dem Host%s" % (image, " (%s)" % err if err else ""))
        ctx = ctx_for_image(im, list_contexts(self.ctx_glob))
        if ctx is None:
            raise ValueError("kein Build-Kontext mit release=%s cuda=%s fuer %s" % (im["release"], im["cuda"], image))
        pr = self.profile(profile, ctx)
        if not pr["line"]:
            raise ValueError("Profil %s nennt keine Linie" % profile)
        rev = (ctx["revisions"] or {}).get(pr["line"]) or ctx["revision"]
        mount = 0 if pr["in_image"] else 1
        if mount and not pr["on_host"]:
            raise ValueError("Profil %s weder im Image-Kontext noch auf dem Host" % profile)
        allow_exp = (pr["status"] or "") in EXPERIMENTAL
        ctx_lxc = ctx["ctx"]            # host_acceptance.sh wants the LXC path and prefixes S itself
        env = [("CTX", ctx_lxc), ("IMAGE", image), ("LINE", pr["line"]), ("PROFILE", profile),
               ("PROFILE_MOUNT", str(mount))]
        if allow_exp:
            env.append(("ALLOW_EXPERIMENTAL", "1"))
        env += [("HOUSE_GUARD", house_guard), ("GPUQ_ID", "<fenster-id>")]
        script = self.host_root + HOST_ACCEPTANCE
        cmd = " ".join("%s=%s" % (k, v if v == "<fenster-id>" else shlex.quote(v)) for k, v in env)
        cmd += " bash %s serve %s" % (script, transport)
        notes = [
            "ZUERST ein gpuq-Fenster buchen (Karten 0,1,2, ganze Karten) und dessen id als GPUQ_ID einsetzen; "
            "ohne laufendes Fenster wird nicht gebootet. Ausfuehren als root auf dem Proxmox-Host.",
        ]
        if allow_exp:
            notes.append("Profil-Status '%s': der Entrypoint verlangt ALLOW_EXPERIMENTAL=1 (steht in der Zeile)." % pr["status"])
        if mount:
            notes.append("Profil liegt nicht im Image -> PROFILE_MOUNT=1 bindet die Host-Datei ein.")
        elif pr["differs_from_host"]:
            notes.append("Die Profil-Fassung im Image weicht von %s/profiles/%s.env ab; es gilt die Image-Fassung "
                         "(PROFILE_MOUNT=1 naehme die Host-Fassung)." % (self.docker_dir, profile))
        if transport == "nccl" and (pr["nccl_status"] or "") == "unproven":
            notes.append("PROFILE_NCCL_STATUS=unproven: dieses Profil wurde nie unter nccl gebootet.")
        if house_guard == "ct999-ruht":
            notes.append("ct999-ruht: CT999 (Router 30099, gpuq, alle Agenten) ruht vorher -- das tut nur der Nutzer selbst.")
        return {"command": cmd, "env": env, "step": "serve", "transport": transport, "profile": pr,
                "image": im, "ctx": ctx_lxc, "revision": rev, "notes": notes}

    # ------------------------------------------------------------ dry run
    def dry_script(self, host_acceptance_text: str) -> str:
        return build_dry_script(host_acceptance_text, self.host_root)

    def dry_run(self, built: dict, max_age: float = 300.0) -> dict:
        key = tuple(built["env"]) + (built["transport"],)
        with self._dry_lock:
            hit = self._dry.get(key)
            if hit and time.time() - hit["t"] < max_age:
                return hit
            with open(os.path.join(self.docker_dir, "host_acceptance.sh"), encoding="utf-8") as fh:
                script = self.dry_script(fh.read())
            envs = ["%s=%s" % (k, shlex.quote("DRYRUN" if v == "<fenster-id>" else v)) for k, v in built["env"]]
            envs += ["DRY_STEP=serve", "DRY_TRANSPORT=%s" % built["transport"]]
            remote = "env %s bash -s" % " ".join(envs)
            t0 = time.time()
            try:
                p = subprocess.run(self.ssh + [remote], input=script, capture_output=True, text=True, timeout=90)
                out, rc = (p.stdout + p.stderr), p.returncode
            except Exception as e:
                out, rc = "%s: %s" % (type(e).__name__, e), -1
            green = rc == 0 and "DRY GRUEN" in out
            res = {"t": time.time(), "rc": rc, "green": green, "output": out[-12000:],
                   "duration_s": round(time.time() - t0, 1), "command": built["command"]}
            self._dry[key] = res
            return res


def _extract_fn(lines: List[str], name: str) -> List[str]:
    pat = re.compile(r"^%s\(\)\s*\{" % re.escape(name))
    for i, ln in enumerate(lines):
        if pat.match(ln):
            if ln.rstrip().endswith("}") and ln.count("{") == ln.count("}"):
                return [ln]
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("}"):
                    return lines[i:j + 1]
    raise ValueError("host_acceptance.sh: Funktion %s() nicht gefunden -- Skriptform geaendert, Trockenlauf verweigert" % name)


def _index(lines: List[str], pattern: str) -> int:
    rx = re.compile(pattern)
    for i, ln in enumerate(lines):
        if rx.match(ln):
            return i
    raise ValueError("host_acceptance.sh: Marke %r nicht gefunden -- Skriptform geaendert, Trockenlauf verweigert" % pattern)


DRY_CHECKS = r'''
# ---------------------------------------------------------------- rigdash DRY checks (read-only)
set +e
NF=0; ok(){ echo "DRY OK   $*"; }; fail(){ echo "DRY FEHL $*"; NF=$((NF + 1)); }
T=${DRY_TRANSPORT:-bar1}
echo "DRY abgeleitet: CTX=$CTX HCTX=$HCTX CU=$CU REV=${REV:0:10} IMAGE=$IMAGE LINE=$LINE PROFILE=$PROFILE PSFX=$PSFX"
echo "DRY abgeleitet: VOLKEY=$VOLKEY PORT=$PORT SHM=$SHM MEM_LIMIT=$MEM_LIMIT MEMAVAIL_MIN_GIB=$MEMAVAIL_MIN_GIB HOUSE_GUARD=$HOUSE_GUARD"
if bash -n "$S/spinning/gpu-arb/docker/host_acceptance.sh"; then ok "host_acceptance.sh Syntax"; else fail "host_acceptance.sh Syntax"; fi
case "${DRY_STEP:-}" in serve) ok "Schritt serve";; *) fail "Schritt '${DRY_STEP:-}'";; esac
case "$T" in bar1|nccl) ok "Transport $T";; *) fail "Transport '$T' (bar1|nccl)";; esac
case "$HOUSE_GUARD" in memlimit|ct999-ruht) ok "HOUSE_GUARD=$HOUSE_GUARD";; *) fail "HOUSE_GUARD='$HOUSE_GUARD' (Pflicht fuer serve)";; esac
case "$PROFILE" in "$LINE"*) ok "Profil $PROFILE passt zur Linie $LINE";; *) fail "Profil $PROFILE passt nicht zur Linie $LINE";; esac
if docker image inspect "$IMAGE" >/dev/null 2>&1; then ok "Image $IMAGE vorhanden"; else fail "Image $IMAGE fehlt auf dem Host"; fi
_lab=$(docker image inspect --format "{{index .Config.Labels \"htsglang.revision.$LINE\"}}" "$IMAGE" 2>/dev/null)
if [ -n "$_lab" ] && [ "$_lab" = "$REV" ]; then ok "Image-Label htsglang.revision.$LINE = Kontext-Revision ${REV:0:10}"
else fail "Image-Label htsglang.revision.$LINE='${_lab:0:10}' != Kontext-Revision '${REV:0:10}'"; fi
if [ "$PROFILE_MOUNT" = 1 ]; then _pdir=$S/spinning/gpu-arb/docker/profiles; else _pdir=$HCTX/tools/profiles; fi
if [ -f "$_pdir/$PROFILE.env" ]; then ok "Profil-Datei $_pdir/$PROFILE.env"; else fail "Profil-Datei $_pdir/$PROFILE.env fehlt"; fi
_ps=$(cd "$_pdir" 2>/dev/null && env -i PATH=/usr/bin:/bin bash -c 'set +u; source "./'"$PROFILE"'.env" >/dev/null 2>&1 || exit 7; echo "${PROFILE_NAME:-}|${PROFILE_STATUS:-}|${#PROFILE_ARGS[@]}"')
_prc=$?
IFS='|' read -r _pn _pst _pna <<<"$_ps"
if [ "$_prc" = 0 ] && [ "$_pn" = "$PROFILE" ] && [ "${_pna:-0}" -gt 0 ]; then ok "Profil trocken gesourct: PROFILE_NAME=$_pn PROFILE_STATUS=${_pst:-?} PROFILE_ARGS=$_pna"
else fail "Profil trocken gesourct: rc=$_prc NAME='$_pn' ARGS='$_pna'"; fi
case "$_pst" in experimentell|formnachweis)
  if [ "${ALLOW_EXPERIMENTAL:-0}" = 1 ]; then ok "Status $_pst -> ALLOW_EXPERIMENTAL=1 gesetzt"; else fail "Status $_pst verlangt ALLOW_EXPERIMENTAL=1"; fi ;;
esac
if _ha=$(house_args "$T" 2>&1); then ok "house_args $T: $(echo $_ha)"; else fail "house_args $T: $_ha"; fi
[ -n "${GPUQ_ID:-}" ] && ok "GPUQ_ID gesetzt (${GPUQ_ID}; im echten Lauf die Fenster-id)" || fail "GPUQ_ID fehlt"
echo "DRY Hinweis (kein Gate): Haus-Zustand JETZT per house_check $T (nur lesend):"
( die(){ echo "house_check wuerde jetzt abbrechen: $*"; exit 1; }; house_check "$T" ) 2>&1 | grep -v 'awk: warning' | sed 's/^/DRY   | /' | tail -4
if [ "$NF" = 0 ]; then echo "DRY GRUEN ($(date -u +%FT%TZ)): die Zeile passiert Parameter-, Image-, Profil- und Haus-Argument-Pruefung von host_acceptance.sh; kein Container, keine Karte."; exit 0
else echo "DRY ROT: $NF Pruefung(en) fehlgeschlagen"; exit 1; fi
'''


def build_dry_script(text: str, host_root: str = HOST_ROOT) -> str:
    lines = text.splitlines()
    cut = _index(lines, r'^mkdir -p "\$ACC"$')
    pb0 = _index(lines, r"^for _kv in \$NCCL_EXTRA_ENV")
    pb1 = _index(lines, r"^\[\[ \$RUN_LABEL")
    if not (cut < pb0 <= pb1):
        raise ValueError("host_acceptance.sh: Marken in unerwarteter Reihenfolge -- Trockenlauf verweigert")
    prelude = lines[:cut]
    # the prelude must not contain the side effects we cut at (defensive: a moved mkdir/trap would slip in)
    for ln in prelude:
        if re.match(r"^\s*(trap |mkdir |docker |rm |pct |systemctl )", ln):
            raise ValueError("host_acceptance.sh: Seiteneffekt im Vorspann (%r) -- Trockenlauf verweigert" % ln.strip())
    fns = []
    for name in ("ct999_state", "ct999_mem_mib", "house_args", "ct999_tmpfs_note", "house_check"):
        fns += _extract_fn(lines, name)
    head = ["# rigdash dry run, assembled from host_acceptance.sh (prelude up to its first side effect,",
            "# its parameter checks and its read-only house functions). No container, no card, no log file."]
    say = ['say(){ echo "[dry] $*"; }', 'die(){ echo "DRY ABBRUCH: $*"; exit 1; }']
    return "\n".join(head + prelude + say + lines[pb0:pb1 + 1] + fns) + "\n" + DRY_CHECKS
