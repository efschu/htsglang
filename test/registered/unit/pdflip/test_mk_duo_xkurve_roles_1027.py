# SPDX-License-Identifier: Apache-2.0
"""X-curve roles of the 27B NVFP4 duo form (devtools/mk_duo_xkurve_roles_1027.py, 27.10.2026).

The duo form is a JSON user profile; the bash overlay 27b-xkurve.env that makes the int8/fp8 X-curve roles is never sourced for it.
The generator does the overlay's job on the JSON document. DANGER DIRECTIONS guarded here:
* a D-curve role that still carries ``--x-curves`` (the launcher refuses it W194 with ``--x-mode fixed``) or a stale live ceiling
  (the front could then lower X below 262144 and flip), a P-curve role that idles on D;
* ``--extra-d`` written as two tokens (``apply_edits`` cannot write the ``=`` form; argparse then takes ``--chunked-prefill-size`` as
  a flag of its own and the boot dies at the launcher);
* a role file that does not name its own corridor pointer (rule #249: one pointer per role, else a form switch wipes the records);
* a create_args file that silently keeps the W4A8 default of the 3080 (user order 27.10.: no W4A8 anywhere) or the L1.5 default.
"""
from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import profile_json as pj
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(_HERE, "..", "..", "..", "..", "devtools", "mk_duo_xkurve_roles_1027.py")
_spec = importlib.util.spec_from_file_location("mk_duo_xkurve_roles_1027", _TOOL)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def _duo_doc() -> dict:
    """The flags of 27b-nvfp4-duo-rebar.json that the roles touch, in the profile's own order and token shape."""
    return {
        "schema": "flliper.server/1", "name": "27b-nvfp4-duo-rebar", "line": "27b", "source": {"kind": "env"},
        "vars": [{"name": "PROFILE_CARD_COUNT", "value": "2"}, {"name": "PROFILE_NAME", "value": "27b-nvfp4-duo-rebar"}],
        "exports": [],
        "args": [
            {"flag": "--model", "values": ["/m"]},
            {"flag": "--extra-p", "values": ["--max-running-requests=2"], "eq": True},
            {"flag": "--tp-prefill-max-tokens", "values": ["4096"]},
            {"flag": "--x-ceiling-tokens", "values": ["12288"]},
            {"flag": "--idle-layout", "values": ["pp"]},
            {"flag": "--pp-stage-ratio", "values": ["51,13"]},
            {"flag": "--pp-attn-stage-ratio", "values": ["12,4"]},
            {"flag": "--p-chunk-policy", "values": ["dynamic"]},
            {"flag": "--p-chunk-max", "values": ["2048"]},
            {"flag": "--x-mode", "values": ["curve"]},
            {"flag": "--x-curves", "values": ["/var/lib/htsglang/evidence/xcurves/flliper-27b-nvfp4-duo.json"]},
        ],
        "form": [{"name": "FLLIPER_CORRIDOR_FLOOR_DIGEST_FILE", "value": "/root/.cache/sglang/corridor_floor_digest.duo.json"}],
        "instr": [], "meta": {"origins": {}, "planner": {}, "notes": []},
    }


def _flags(doc: dict) -> dict:
    out = {}
    for e in doc["args"]:
        out.setdefault(e["flag"], []).append(" ".join(e.get("values") or []))
    return out


def _base_lines() -> list:
    return ["--name", "flliper-27b-nvfp4-duo", "--restart", "no", "-v", "/spinning/subvol-999-disk-0/spinning/gpu-arb/deskq/bl/"
            "27b-nvfp4-duo-1023:/var/lib/flliper/profiles:ro", "-e", "DREI_NAME=flliper-27b-nvfp4-duo", "-e",
            "HTSGLANG_PROFILE=27b-nvfp4-duo-rebar", "--label", "io.github.efschu.flliper.rotation=duo", "--entrypoint",
            "/opt/drei/init.sh", "flliper:0.1.0-rc42b-cu130", "serve"]


class RoleProfile(CustomTestCase):
    def test_d_curve_is_all_short_on_d_and_fixed(self):
        doc = G.patch_profile(_duo_doc(), G.RoleSpec(kind="d"))
        f = _flags(doc)
        self.assertEqual(f["--tp-prefill-max-tokens"], ["262144"])
        self.assertEqual(f["--x-ceiling-tokens"], ["262144"])   # D's W50 riegel AND the front's live ceiling: no flip below the context
        self.assertEqual(f["--x-mode"], ["fixed"])
        self.assertNotIn("--x-curves", f)                        # W194: --x-curves only with --x-mode curve
        self.assertEqual(f["--idle-layout"], ["tp"])
        self.assertEqual(f["--pp-stage-ratio"], ["51,13"])       # the D curve leaves the P cut alone

    def test_p_curve_goes_over_p_at_the_floor_x_and_idles_on_p(self):
        doc = G.patch_profile(_duo_doc(), G.RoleSpec(kind="p", p_chunk_max=4096, p_chunk_sweep="1024,2048,4096"))
        f = _flags(doc)
        self.assertEqual((f["--tp-prefill-max-tokens"], f["--x-ceiling-tokens"]), (["4096"], ["4096"]))
        self.assertEqual(f["--idle-layout"], ["pp"])
        self.assertEqual(f["--p-chunk-max"], ["4096"])
        self.assertEqual(f["--p-chunk-sweep"], ["1024,2048,4096"])
        self.assertNotIn("--extra-d", f)

    def test_every_role_has_its_own_corridor_pointer_name_and_id(self):
        for kind, tag in (("d", ""), ("p", "cut56")):
            spec = G.RoleSpec(kind=kind, tag=tag, **({"cut": "56,8", "attn_cut": "14,2"} if kind == "p" else {}))
            doc = G.patch_profile(_duo_doc(), spec)
            ptr = [e for e in doc["form"] if e["name"] == "FLLIPER_CORRIDOR_FLOOR_DIGEST_FILE"]
            self.assertEqual([e["value"] for e in ptr], ["/root/.cache/sglang/corridor_floor_digest.%s.json" % spec.role])
            self.assertEqual(doc["name"], "27b-nvfp4-duo-rebar-x%s%s" % (kind, tag))
            # the entrypoint sources the profile and refuses PROFILE_NAME != HTSGLANG_PROFILE (weg2line.py: profile sourced dry)
            self.assertEqual([v["value"] for v in doc["vars"] if v["name"] == "PROFILE_NAME"], [doc["name"]])
            self.assertEqual(doc["id"], pj.doc_id(doc))

    def test_d_chunk_is_one_eq_token(self):
        doc = G.patch_profile(_duo_doc(), G.RoleSpec(kind="d", d_chunk=8192))
        self.assertIn("--extra-d=--chunked-prefill-size=8192", pj.render_args(doc["args"]))
        # the editor's own write of the same value is two tokens: the boot would die on it
        edited = pj.apply_edits(_duo_doc(), [{"key": "extra:D:--chunked-prefill-size", "op": "set", "value": "8192"}])
        self.assertNotIn("--extra-d=--chunked-prefill-size=8192", pj.render_args(edited["args"]))

    def test_cut_pin_and_stage_fit_exclude_each_other_and_a_fit_unpins(self):
        with self.assertRaises(ValueError):
            G.patch_profile(_duo_doc(), G.RoleSpec(kind="p", cut="56,8", attn_cut="14,2", stage_fit="auto"))
        with self.assertRaises(ValueError):
            G.patch_profile(_duo_doc(), G.RoleSpec(kind="p", cut="56,8"))                   # the pair is one cut
        f = _flags(G.patch_profile(_duo_doc(), G.RoleSpec(kind="p", stage_fit="auto")))
        self.assertEqual(f["--pp-cut-stage-fit"], ["auto"])
        self.assertNotIn("--pp-stage-ratio", f)
        self.assertNotIn("--pp-attn-stage-ratio", f)
        f = _flags(G.patch_profile(_duo_doc(), G.RoleSpec(kind="p", cut="56,8", attn_cut="14,2")))
        self.assertEqual((f["--pp-stage-ratio"], f["--pp-attn-stage-ratio"]), (["56,8"], ["14,2"]))

    def test_d_curve_takes_no_p_knob(self):
        with self.assertRaises(ValueError):
            G.patch_profile(_duo_doc(), G.RoleSpec(kind="d", p_chunk_max=4096))


class RoleCreateArgs(CustomTestCase):
    def test_w4a16_on_the_3080_and_l15_off_are_in_the_role(self):
        spec = G.RoleSpec(kind="d")
        out = G.derive_create_args(_base_lines(), spec=spec, profile_dir="/spinning/gpu-arb/deskq/bl/dir", image="flliper:0.1.0-new-cu130")
        envs = [out[i + 1] for i, t in enumerate(out) if t == "-e"]
        self.assertIn("FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin", envs)      # default w4a8 would run the INT8 kernel on the 3080
        self.assertIn("FLLIPER_FP4_ALLOW_BROKEN_SM8X_MARLIN=1", envs)
        self.assertIn("FLLIPER_PDFLIP_L15=0", envs)
        self.assertEqual(out[-2:], ["flliper:0.1.0-new-cu130", "serve"])
        self.assertEqual(out[1], "flliper-27b-duoxd")
        self.assertIn("DREI_NAME=flliper-27b-duoxd", out)
        self.assertIn("HTSGLANG_PROFILE=27b-nvfp4-duo-rebar-xd", out)
        self.assertIn("/spinning/subvol-999-disk-0/spinning/gpu-arb/deskq/bl/dir:/var/lib/flliper/profiles:ro", out)
        self.assertLess(out.index("FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin"), out.index("--entrypoint"))

    def test_extra_env_lands_before_the_entrypoint_and_a_blank_is_refused(self):
        spec = G.RoleSpec(kind="d", tag="nolend", extra_envs=("FLLIPER_PDFLIP_D_POOL_LEND=0",))
        out = G.derive_create_args(_base_lines(), spec=spec, profile_dir="/spinning/x", image="flliper:0.1.0-n-cu130")
        self.assertLess(out.index("FLLIPER_PDFLIP_D_POOL_LEND=0"), out.index("--entrypoint"))
        self.assertEqual(out.count("FLLIPER_PDFLIP_D_POOL_LEND=0"), 1)
        with self.assertRaises(ValueError):                     # an env with a blank would be two docker tokens
            G.patch_profile(_duo_doc(), G.RoleSpec(kind="d", extra_envs=("FLLIPER_X=a b",)))

    def test_a_base_that_is_not_the_duo_file_is_refused(self):
        lines = [t for t in _base_lines() if not t.startswith("HTSGLANG_PROFILE=")]
        with self.assertRaises(ValueError):          # no profile token: the role would boot the plain duo profile, silently
            G.derive_create_args(lines, spec=G.RoleSpec(kind="d"), profile_dir="/spinning/x", image="flliper:0.1.0-n-cu130")


if __name__ == "__main__":
    unittest.main()
