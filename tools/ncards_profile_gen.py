#!/usr/bin/env python3
"""NCARDS 1021: derive the per-card part of a release profile for N IDENTICAL cards from the profile planner's proposal.

Not a second planner.  ``pdflip/propose.py`` (stage A, pure) already decides every per-card value of a profile for an inventory;
this tool runs it for ``--n`` x ``--card`` against a release profile (the BASIS, e.g. ``nf.env``) and prints the DIFFERENCE between the
basis argv/env and the proposal as ``_p3090_*`` setter lines (``docker/flliper/profiles_release/_3090_common.sh``).  A profile file
``<basis>-<n>x3090.env`` sources the basis, then the generated block -- so a number in that block IS a planner output, not a
hand-copied one, and ``test_ncards_profiles_1021`` regenerates the block and compares it (drift = red).

    python tools/ncards_profile_gen.py --basis nf --n 3 [--card rtx3090-24] [--checkpoints DIR] [--check PROFILE.env]

Model inputs are the checkpoint headers (``planer_1006/checkpoints`` fixtures by default: config + safetensors index, no weights).
CPU only; reads nothing from the rig.  Every value is HOCHRECHNUNG (datasheet card, measured 3080-class constants borrowed by arch twin),
marked as such in the proposal's ``unverified`` list, which the tool prints as comments.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shlex
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(TREE, "test", "registered", "unit", "pdflip", "fixtures", "planer_1006", "checkpoints")
PROFILES = os.path.join(TREE, "docker", "flliper", "profiles_release")
BEGIN = "# >>> ncards_profile_gen"
END = "# <<< ncards_profile_gen"

#: basis profile -> (release profile file, model checkpoint dir name, draft checkpoint dir name)
BASES: Dict[str, Tuple[str, str, str]] = {
    "nf": ("nf.env", "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp"),
    "27b": ("27b.env", "Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued"),
}
#: flags the group strings carry; compared word by word, never as one token
GROUP_FLAGS = ("--extra-p", "--extra-d", "--env-p", "--env-d")
#: flags that take the next token whatever it starts with (a group string starts with "--")
_TAKES_VALUE = set(GROUP_FLAGS) | {"--extra"}
_WORD = re.compile(r"(?<![\w-])(--[\w-]+)(?:\s+|=)(\S+)")


def _top_flags(tokens: Sequence[str]) -> Dict[str, Optional[str]]:
    """``{flag: value}`` of the top-level tokens (``--flag value`` / ``--flag=value`` / bare ``--flag``); the LAST occurrence wins."""
    out: Dict[str, Optional[str]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if tok.split("=", 1)[0] in _TAKES_VALUE and "=" in tok:
            name, val = tok.split("=", 1)
            out[name] = val
            i += 1
        elif "=" in tok:
            name, val = tok.split("=", 1)
            out[name] = val
            i += 1
        elif tok in _TAKES_VALUE or (i + 1 < len(tokens) and not tokens[i + 1].startswith("--")):
            out[tok] = tokens[i + 1] if i + 1 < len(tokens) else ""
            i += 2
        else:
            out[tok] = None
            i += 1
    return out


def _words(text: Optional[str]) -> Dict[str, str]:
    """``{--flag: single-word value}`` inside an ``--extra-X`` string (multi-word values are not touched by the planner)."""
    return {m.group(1): m.group(2) for m in _WORD.finditer(text or "")}


def _env_items(text: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in (text or "").split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def sh(v: str) -> str:
    return shlex.quote(str(v))


def diff_lines(basis_argv: Sequence[str], basis_env: Dict[str, str], new_argv: Sequence[str], new_env: Dict[str, str]) -> List[str]:
    """The setter lines that turn the basis into the proposal (empty = identical)."""
    out: List[str] = []
    b, n = _top_flags(basis_argv), _top_flags(new_argv)
    for name in sorted(set(b) | set(n)):
        if name in GROUP_FLAGS or name == "--extra":
            continue
        if b.get(name) == n.get(name) and (name in b) == (name in n):
            continue
        if name not in n:
            out.append("_p3090_drop %s" % name)
        elif n[name] is None:
            out.append("_p3090_set %s ''   # bare flag" % name)      # not produced by the planner today; kept loud
        else:
            out.append("_p3090_set %s %s" % (name, sh(n[name])))
    for g in ("p", "d"):
        bw, nw = _words(b.get("--extra-" + g)), _words(n.get("--extra-" + g))
        for flag in sorted(set(bw) | set(nw)):
            if flag not in nw:
                out.append("_p3090_extra_drop %s %s" % (g, flag))
            elif bw.get(flag) != nw[flag]:
                out.append("_p3090_extra %s %s %s" % (g, flag, sh(nw[flag])))
        be, ne = _env_items(b.get("--env-" + g)), _env_items(n.get("--env-" + g))
        for key in sorted(set(be) | set(ne)):
            if key not in ne:
                out.append("_p3090_env_drop %s %s" % (g, key))
            elif be.get(key) != ne[key]:
                out.append("_p3090_env %s %s %s" % (g, key, sh(ne[key])))
    for key in sorted(set(basis_env) | set(new_env)):
        if key in ("HTSGLANG_TAG", "HTSGLANG_INSTRUMENTS"):
            continue
        if basis_env.get(key) != new_env.get(key):
            if key not in new_env:
                out.append("# global env %s is dropped by the proposal (not representable as a setter: review)" % key)
            else:
                out.append("_p3090_form %s %s" % (key, sh(new_env[key])))
    return out


def _catalog(card: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("ncards_kc", os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cat = {e["id"]: e for e in mod.CATALOG}
    if card not in cat:
        raise SystemExit("unknown catalog card %r (known: %s)" % (card, ", ".join(sorted(cat))))
    return cat[card]


def proposal(basis: str, n: int, card: str = "rtx3090-24", checkpoints: str = CKPT, form: str = "flip", kv_tokens: int = 0):
    """``(basis LaunchInput, propose() result)`` for ``n`` x ``card``."""
    if TREE + "/python" not in sys.path:
        sys.path.insert(0, TREE + "/python")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from flliper.srt.pdflip import model_profile as MP
    from flliper.srt.pdflip import propose as P
    from flliper.srt.pdflip import propose_oracle as O

    env_file, model_dir, draft_dir = BASES[basis]
    tmp = tempfile.mkdtemp(prefix="ncards-gen-")
    os.environ["FLLIPER_CARD_LIBRARY"] = os.path.join(tmp, "no-card-library.json")      # an INPUT: never read ~/.cache
    for name in (model_dir, draft_dir):
        O.materialize_checkpoint(os.path.join(checkpoints, name), tmp)
    mp = MP.estimate_or_state(os.path.join(tmp, model_dir))
    if not mp.get("ok"):
        raise SystemExit("model profile of %s: %s" % (model_dir, mp))
    draft = MP.estimate_draft(os.path.join(tmp, draft_dir))
    li = O.profile_launch_input(os.path.join(PROFILES, env_file))
    rates = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}        # the rig's measured GEMM rates (an input, as the planner tests)
    goals = {"kv_tokens": int(kv_tokens)} if kv_tokens else {}
    v = P.propose([_catalog(card)] * n, mp["profile"], form, goals, basis=li, draft=draft, rates=rates)
    return li, v


#: the words of the group strings that carry the KV POOL / CONTEXT of a rank: the planner moves the top-level ``--max-kv-per-request`` (the ceiling
#: per request, the pool is SHARED) with the goal ``kv_tokens`` but leaves these words of the profile; the launcher's D budget check
#: (W122 "mit <T> Token KV") reads them, so a ceiling below the rig's 262144 moves them with it
KV_WORDS = ("--max-total-tokens", "--context-length", "--max-kv-per-request")


def block(basis: str, n: int, card: str = "rtx3090-24", checkpoints: str = CKPT, kv_tokens: int = 0) -> str:
    li, v = proposal(basis, n, card, checkpoints, kv_tokens=kv_tokens)
    lines = diff_lines(li.argv, li.env, v["argv"], v["env"])
    if kv_tokens:
        lines = [x for x in lines if not x.startswith("_p3090_set --max-kv-per-request")]
        lines.append("_p3090_set --max-kv-per-request %d" % kv_tokens)
        for g in ("p", "d"):
            lines += ["_p3090_extra %s %s %d" % (g, w, kv_tokens) for w in KV_WORDS]
    kvarg = (" --kv-tokens %d" % kv_tokens) if kv_tokens else ""
    head = ["%s: basis=%s n=%d card=%s%s -- python tools/ncards_profile_gen.py --basis %s --n %d --card %s%s" % (BEGIN, basis, n, card, kvarg, basis, n, card, kvarg),
            "# Planner outputs (pdflip/propose.py stage A, flip form), NOT measurements: every line below is HOCHRECHNUNG. Do not edit by hand;",
            "# regenerate. Unverified inputs of this proposal (first boot = measurement, FLLIPER_FORCE=1):"]
    head += ["#   - " + x[:200] for x in v.get("unverified", [])[:12]]
    for blk in v.get("blocker", []) or []:
        head.append("# BLOCKER of the proposal: " + str(blk)[:300])
    return "\n".join(head + lines + [END]) + "\n"


NF_TEMPLATE = """# shellcheck shell=bash
# PROFIL nf-{n}x3090 (NCARDS 1021, HOCHRECHNUNG): Qwen3.8-Flash-Next INT4 + MTP-Draft (NF), Flip P = PP{n} / D = Form A TP{n}
# (Rang 0 = Attention-Host, die uebrigen Rang-Experten-Worker) auf {n} x RTX 3090 (24576 MiB, sm86, 936 GB/s, PCIe 4.0 x16).
# Das Release-Profil nf (Rig 5090 + 2 x 3080) mit den Planer-Werten fuer N gleiche Karten. DIE ZAHLEN STEHEN NICHT VON HAND HIER:
# der Block unten wird von tools/ncards_profile_gen.py aus propose() (pdflip/propose.py, Stufe A, Form flip) erzeugt, jede Zeile ist
# der Unterschied zwischen dem Rig-Profil nf.env und dem Vorschlag fuer diese Karten (Experten-Verteilung, residente Anteile, P-Schnitt,
# D-Kontext, Reserven). test_ncards_profiles_1021 erzeugt den Block neu und vergleicht (Drift = rot). Unverifizierte Eingaben des
# Vorschlags stehen als Kommentar im Block.
# STATUS: nie auf einer 3090 gebootet. Launcher-Trockenlauf auf einem NVML-Replay von {n} x 3090: ok_with_force (HW-UNCALIBRATED /
# HW-BORROWED{count}); die Rekorde des Profils nextflash sind auf dem Rig gemessen und werden vom Arch-Zwilling 3080 GEBORGT.
# N = 3 (bekannter Defekt, nicht verdeckt): das D-Budget des Launchers (W122/W130/D-EIGENTUM) verweigert die Planer-Verteilung, auch mit
# --force -- "groesste tragbare Fraction je Rang: keine, 0.553, 0.566"; der Planer-Regel fuer Form A fehlt dort ~5 GB je Rang. N = 2 und N = 4
# bestehen dieselbe Pruefung. Siehe /spinning/gpu-arb/deskq/done/planer-ncards-1021.md.
# ERSTER START: FLLIPER_FORCE=1 (der erste Boot ist die Messung); vorher card_rate_pass --run. Host-Seite (Experten-Store im tmpfs, MEMAVAIL,
# SHM) haengt nicht an N und bleibt wie nf.env: der Store traegt alle Experten, die Karten halten nur den residenten Anteil.
# shellcheck source=/dev/null
source "$(dirname "${{BASH_SOURCE[0]}}")/nf.env"
# shellcheck source=/dev/null
source "$(dirname "${{BASH_SOURCE[0]}}")/_3090_common.sh"
PROFILE_NAME=nf-{n}x3090
PROFILE_STATUS={status}   # {status_note}
PROFILE_OWNER="Profil-Planer NCARDS 1021 (Hochrechnung aus dem nf-Release-Profil, N={n})"
PROFILE_GPUS="NVIDIA GeForce RTX 3090={n}"
PROFILE_CARD_COUNT={n}
PROFILE_INVENTORY="{inv}"
{block}eval "$(declare -f profile_form_env | sed '1s/^profile_form_env/_profile_form_env_nf_base/')"
profile_form_env() {{
  _profile_form_env_nf_base
  _p3090_form_apply
}}
"""


#: PROFILE_STATUS of the generated NF profiles: the planner's N = 3 ownership (89/212/211 experts, resident 0.49/0.79/0.79) is REFUSED by the launcher's own
#: D budget check (W122 / W130 / D-EIGENTUM, forced past as D-BUDGET): "largest bearable fraction per rank: none, 0.553, 0.566" -- the planner's Form A rule
#: is ~5 GB per rank too optimistic for N = 3 identical cards (N = 2 and N = 4 pass the same check).  Not bootable until planner and launcher agree.
NF_STATUS = {0: ("hochrechnung", "Planer-Rechnung, keine Messung auf 3090; Abnahme erst nach Metall"),
             3: ("blockiert-d-budget", "Launcher-D-Budget (W122/W130) verweigert die Planer-Verteilung auch mit --force (D-BUDGET): nicht startbar, siehe Kopf")}


def render_profile(basis: str, n: int, card: str = "rtx3090-24", checkpoints: str = CKPT, kv_tokens: int = 0) -> str:
    """The whole ``<basis>-<n>x3090.env`` text (NF only: the 27B profiles of this family are hand-written rules, see ``_p3090_apply``)."""
    if basis != "nf":
        raise SystemExit("render_profile: only the NF profiles are generated (27B: _3090_common.sh _p3090_apply)")
    status, note = NF_STATUS.get(n, NF_STATUS[0])
    return NF_TEMPLATE.format(n=n, count=", HW-COUNT" if n != 3 else "", inv=",".join(["RTX3090"] * n), status=status, status_note=note,
                              block=block(basis, n, card, checkpoints, kv_tokens))


def extract(text: str) -> str:
    """The generated block of a profile file."""
    m = re.search(r"(%s.*?^%s\n)" % (re.escape(BEGIN), re.escape(END)), text, re.S | re.M)
    return m.group(1) if m else ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--basis", choices=sorted(BASES), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--card", default="rtx3090-24")
    ap.add_argument("--checkpoints", default=CKPT)
    ap.add_argument("--check", help="profile file whose generated block must equal the regenerated one (exit 1 on drift)")
    ap.add_argument("--write", help="write the whole profile file (NF) to this path")
    ap.add_argument("--kv-tokens", type=int, default=0, help="goal: the per-request KV ceiling (0 = the profile's 262144)")
    a = ap.parse_args(argv)
    if a.write:
        with open(a.write, "w", encoding="utf-8") as fh:
            fh.write(render_profile(a.basis, a.n, a.card, a.checkpoints, a.kv_tokens))
        return 0
    new = block(a.basis, a.n, a.card, a.checkpoints, a.kv_tokens)
    if a.check:
        old = extract(open(a.check, encoding="utf-8").read())
        if old != new:
            sys.stderr.write("DRIFT: %s differs from the regenerated block\n" % a.check)
            sys.stdout.write(new)
            return 1
        return 0
    sys.stdout.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
