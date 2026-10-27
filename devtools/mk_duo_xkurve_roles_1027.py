#!/usr/bin/env python3
"""mk_duo_xkurve_roles_1027.py -- X-curve roles duoxd / duoxp for the 27B NVFP4 duo form (27.10.2026, 27B seat).

WHY A SECOND GENERATOR. ``rotation/gen/mk_xkurve_roles_1027.py`` derives the X-curve roles of the release ``.env`` profiles (27b,
27b-fp8) through the bash overlay ``27b-xkurve.env``. The duo form is a USER profile (JSON, ``flliper.server/1``, bound at
``/var/lib/flliper/profiles``), the overlay is never sourced for it -- the xkurve docstring says so ("duo/int4ar ... eine eigene
Arbeit"). This module does the overlay's job on the JSON document, through the finished editor ``profile_json.apply_edits``
(no hand-written token surgery except the one ``--extra-d`` entry, whose ``eq`` form ``apply_edits`` cannot write).

WHAT A ROLE CHANGES (the overlay's own rules, 27b-xkurve.env, one by one)
    D curve (duoxd): --tp-prefill-max-tokens = --x-ceiling-tokens = 262144 (= --max-kv-per-request), --x-mode fixed (no live
                     re-solve; ``--x-curves`` removed: with ``--x-mode fixed`` it is refused W194), --idle-layout tp: every request is
                     SHORT on D, never a flip.
    P curve (duoxp): --tp-prefill-max-tokens = --x-ceiling-tokens = 4096 (the floor of launcher.resolve_x), --x-mode fixed,
                     --idle-layout pp (profile): uncached > 4096 is LONG -> P (PP2) -> flip back -> D leg 2.
    both:            FLLIPER_CORRIDOR_FLOOR_DIGEST_FILE = corridor_floor_digest.<role>.json (rule #249: one pointer per form/role; the
                     chain seeds it from corridor_floor_digest.duo.json).
    optional knobs (a role is one boot, so every knob is baked into the JSON; a variant gets its own ``--tag``):
                     --d-chunk (D --chunked-prefill-size), --p-chunk-max, --p-chunk-sweep, --pp-cut/--pp-attn-cut (the P cut),
                     --stage-fit (``--pp-cut-stage-fit``; with it the pinned cut is dropped so the solver prices the cut).

WHAT A create_args FILE CHANGES (derived from create_args_duo.txt, everything else byte for byte): container name, DREI_NAME,
rotation label, the profile directory bind, HTSGLANG_PROFILE, the image, and additions before ``--entrypoint`` (plus every ``--env``):
L1.5 off (as in every X-curve role) and the W4A16 kernel of the 3080 (user order 27.10.: no W4A8 anywhere):
``FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin`` + ``FLLIPER_FP4_ALLOW_BROKEN_SM8X_MARLIN=1`` (the guard of nvfp4_native_mixed.py is a
26.09. finding on a flip boot; the D curve never flips, the P curve needs the needle check, see the report).

Nothing here starts a container. Usage:
    python3 devtools/mk_duo_xkurve_roles_1027.py --image flliper:0.1.0-<new27b>-cu130 [--tag ""] [--d-chunk N] [--p-chunk-sweep 1024,2048,4096]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import msgspec

from flliper.srt.pdflip import profile_json as pj

SRC_PROFILE = "27b-nvfp4-duo-rebar"
FORM = "duo"
D_X = 262144
P_X = 4096
LIVE_X_CEILING_MAX = 262144
CT_ROOT = "/spinning/"
HOST_ROOT = "/spinning/subvol-999-disk-0/spinning/"
DEFAULT_SRC_JSON = "/spinning/gpu-arb/deskq/bl/27b-nvfp4-duo-1023/%s.json" % SRC_PROFILE
DEFAULT_PROFILE_DIR = "/spinning/gpu-arb/deskq/bl/27b-nvfp4-duo-xkurve-1027"
DEFAULT_GEN_DIR = "/spinning/gpu-arb/deskq/rotation/gen"
W4A16_ENVS = ("FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin", "FLLIPER_FP4_ALLOW_BROKEN_SM8X_MARLIN=1")
L15_OFF_ENVS = ("FLLIPER_PDFLIP_L15=0", "FLLIPER_PDFLIP_L15_REFILL=0", "FLLIPER_PDFLIP_L15_HOT_SHARE=0", "FLLIPER_PDFLIP_L15_MIB=")


class RoleSpec(msgspec.Struct, frozen=True):
    """One role: ``kind`` d|p, the variant ``tag`` and the optional knobs (None = the profile's value)."""

    kind: str
    tag: str = ""
    d_chunk: Optional[int] = None
    p_chunk_max: Optional[int] = None
    p_chunk_sweep: Optional[str] = None
    cut: Optional[str] = None
    attn_cut: Optional[str] = None
    stage_fit: Optional[str] = None
    extra_envs: "tuple[str, ...]" = ()

    @property
    def role(self) -> str:
        return "%sx%s%s" % (FORM, self.kind, self.tag)

    @property
    def profile_name(self) -> str:
        return "%s-x%s%s" % (SRC_PROFILE, self.kind, self.tag)

    @property
    def x(self) -> int:
        return D_X if self.kind == "d" else P_X


def _check(spec: RoleSpec) -> None:
    if spec.kind not in ("d", "p"):
        raise ValueError("kind %r: d or p" % spec.kind)
    if spec.kind == "d" and (spec.p_chunk_max or spec.p_chunk_sweep or spec.cut or spec.attn_cut or spec.stage_fit):
        raise ValueError("the D curve takes no P knobs (P sleeps, nothing runs on it)")
    if spec.kind == "p" and spec.d_chunk:
        raise ValueError("the P curve takes no --d-chunk (every request above 4096 runs on P)")
    for env in spec.extra_envs:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*=[^\s]*", env):
            raise ValueError("--env %r: NAME=VALUE (no blanks)" % env)
    if (spec.cut is None) != (spec.attn_cut is None):
        raise ValueError("--pp-cut and --pp-attn-cut come together (the pair is one cut)")
    if spec.cut is not None and spec.stage_fit:
        raise ValueError("a pinned cut and --stage-fit exclude each other (the solver prices only an unpinned cut)")


def _set(key: str, value: str) -> Dict[str, str]:
    return {"key": key, "op": "set", "value": value}


def _drop(key: str) -> Dict[str, str]:
    return {"key": key, "op": "delete"}


def role_edits(spec: RoleSpec) -> List[Dict[str, str]]:
    """The ``profile_json.apply_edits`` list of this role (the overlay's rules, see the module docstring)."""
    _check(spec)
    edits = [_set("flag:--tp-prefill-max-tokens", str(spec.x)), _set("flag:--x-ceiling-tokens", str(spec.x)),
             _set("flag:--x-mode", "fixed"), _drop("flag:--x-curves"),
             _set("form:FLLIPER_CORRIDOR_FLOOR_DIGEST_FILE", "/root/.cache/sglang/corridor_floor_digest.%s.json" % spec.role),
             _set("var:PROFILE_NAME", spec.profile_name)]    # the entrypoint compares PROFILE_NAME with HTSGLANG_PROFILE (rigdash profil.py:781)
    if spec.kind == "d":
        edits.append(_set("flag:--idle-layout", "tp"))
    if spec.p_chunk_max:
        edits.append(_set("flag:--p-chunk-max", str(spec.p_chunk_max)))
    if spec.p_chunk_sweep:
        edits.append(_set("flag:--p-chunk-sweep", spec.p_chunk_sweep))
    if spec.cut is not None:
        edits += [_set("flag:--pp-stage-ratio", spec.cut), _set("flag:--pp-attn-stage-ratio", spec.attn_cut or "")]
    if spec.stage_fit:
        edits += [_drop("flag:--pp-stage-ratio"), _drop("flag:--pp-attn-stage-ratio"),
                  _set("flag:--pp-cut-stage-fit", spec.stage_fit)]
    return edits


def _with_extra_d(doc: Dict, d_chunk: Optional[int]) -> None:
    """``--extra-d=--chunked-prefill-size=N`` as ONE eq token (what the overlay writes); apply_edits cannot write the eq form."""
    if not d_chunk:
        return
    doc["args"] = [e for e in doc["args"] if e.get("flag") != "--extra-d"]
    doc["args"].append({"flag": "--extra-d", "values": ["--chunked-prefill-size=%d" % int(d_chunk)], "eq": True})


def patch_profile(doc: Dict, spec: RoleSpec) -> Dict:
    """The role's profile document: the duo profile with the edits applied, renamed, ``id`` recomputed."""
    new = pj.apply_edits(doc, role_edits(spec))
    _with_extra_d(new, spec.d_chunk)
    new["name"] = spec.profile_name
    notes = new.setdefault("meta", {}).setdefault("notes", [])
    notes.append("27.10. X-curve role %s (mk_duo_xkurve_roles_1027.py): %s, W4A16 on the 3080 is set by the container env, not here"
                 % (spec.role, ", ".join("%s=%s" % (e["key"], e.get("value", "-")) for e in role_edits(spec))))
    new["id"] = pj.doc_id(new)
    return new


# -- create_args -------------------------------------------------------------------------------------------

def host_path(ct_path: str) -> str:
    """The proxmox-side path of a CT path (docker binds resolve on the host)."""
    if not ct_path.startswith(CT_ROOT):
        raise ValueError("%s is not under %s" % (ct_path, CT_ROOT))
    return HOST_ROOT + ct_path[len(CT_ROOT):]


def _is_profile_bind(token: str) -> bool:
    return token.endswith(":/var/lib/flliper/profiles:ro")


def derive_create_args(lines: Sequence[str], *, spec: RoleSpec, profile_dir: str, image: str) -> List[str]:
    """create_args_duo.txt tokens -> the role's tokens. Replacements happen exactly once each, else the base is not the duo file."""
    cn = "flliper-27b-%s" % spec.role
    out: List[str] = []
    seen = {"name": 0, "drei": 0, "label": 0, "bind": 0, "profile": 0}
    for i, t in enumerate(lines):
        prev = lines[i - 1] if i else ""
        if i == 1 and lines[0] == "--name":
            out.append(cn)
            seen["name"] += 1
        elif t.startswith("DREI_NAME=") and prev == "-e":
            out.append("DREI_NAME=" + cn)
            seen["drei"] += 1
        elif t.startswith("io.github.efschu.flliper.rotation=") and prev == "--label":
            out.append("io.github.efschu.flliper.rotation=" + spec.role)
            seen["label"] += 1
        elif prev == "-v" and _is_profile_bind(t):
            out.append(host_path(profile_dir) + ":/var/lib/flliper/profiles:ro")
            seen["bind"] += 1
        elif t.startswith("HTSGLANG_PROFILE=") and prev == "-e":
            out.append("HTSGLANG_PROFILE=" + spec.profile_name)
            seen["profile"] += 1
        elif t == "--entrypoint":
            for env in L15_OFF_ENVS + W4A16_ENVS + tuple(spec.extra_envs):
                out += ["-e", env]
            out.append(t)
        else:
            out.append(t)
    bad = {k: v for k, v in seen.items() if v != 1}
    if bad:
        raise ValueError("replacements not exactly once (is the base create_args_duo.txt?): %s" % bad)
    if out[-1] != "serve" or not out[-2].startswith("flliper:"):
        raise ValueError("the file does not end in <image> serve")
    out[-2] = image
    return out


# -- files -------------------------------------------------------------------------------------------------

def _write_json(path: str, doc: Dict) -> None:
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def build_role(*, spec: RoleSpec, src_doc: Dict, base_lines: Sequence[str], profile_dir: str, image: str) -> "tuple[Dict, List[str]]":
    return patch_profile(src_doc, spec), derive_create_args(base_lines, spec=spec, profile_dir=profile_dir, image=image)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="the 27B image that carries D-POOL-LEND + P-CHUNK-MEM-GUARD (no default: say it)")
    ap.add_argument("--tag", default="", help="variant suffix, e.g. 'cut56' -> duoxpcut56")
    ap.add_argument("--kinds", default="d,p")
    ap.add_argument("--d-chunk", type=int)
    ap.add_argument("--p-chunk-max", type=int)
    ap.add_argument("--p-chunk-sweep")
    ap.add_argument("--pp-cut")
    ap.add_argument("--pp-attn-cut")
    ap.add_argument("--stage-fit")
    ap.add_argument("--env", action="append", default=[], help="extra container env NAME=VALUE (repeatable), e.g. FLLIPER_PDFLIP_D_POOL_LEND=0")
    ap.add_argument("--src-json", default=DEFAULT_SRC_JSON)
    ap.add_argument("--base-args", default=os.path.join(DEFAULT_GEN_DIR, "create_args_duo.txt"))
    ap.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    ap.add_argument("--gen-dir", default=DEFAULT_GEN_DIR)
    ap.add_argument("--stdout", action="store_true", help="print, write nothing")
    return ap


def _read_lines(path: str) -> List[str]:
    lines = open(path).read().split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = _parser().parse_args(argv)
    src_doc = json.load(open(a.src_json))
    base = _read_lines(a.base_args)
    for kind in [k for k in a.kinds.split(",") if k]:
        spec = RoleSpec(kind=kind, tag=a.tag, d_chunk=a.d_chunk if kind == "d" else None,
                        p_chunk_max=a.p_chunk_max if kind == "p" else None,
                        p_chunk_sweep=a.p_chunk_sweep if kind == "p" else None,
                        cut=a.pp_cut if kind == "p" else None, attn_cut=a.pp_attn_cut if kind == "p" else None,
                        stage_fit=a.stage_fit if kind == "p" else None, extra_envs=tuple(a.env))
        doc, lines = build_role(spec=spec, src_doc=src_doc, base_lines=base, profile_dir=a.profile_dir, image=a.image)
        jpath = os.path.join(a.profile_dir, spec.profile_name + ".json")
        apath = os.path.join(a.gen_dir, "create_args_%s.txt" % spec.role)
        if a.stdout:
            sys.stdout.write("### %s\n%s\n### %s\n%s\n" % (jpath, json.dumps(doc, indent=1)[:2000], apath, "\n".join(lines)))
            continue
        os.makedirs(a.profile_dir, exist_ok=True)
        _write_json(jpath, doc)
        with open(apath, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("written %s (%s) and %s (container flliper-27b-%s, image %s)" % (jpath, doc["id"][:19], apath, spec.role, image_of(lines)))
    return 0


def image_of(lines: Sequence[str]) -> str:
    return lines[-2]


if __name__ == "__main__":
    sys.exit(main())
