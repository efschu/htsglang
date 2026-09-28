"""#239 S4b part 7: F14 (the KV-holding worker's host tier and store under the
token cut) is wired -- and the switch acts ONLY on the cut.

The register's ``wired`` flag of F14 reaches the runtime through exactly one
door: ``weg2.launcher.refuse_unwired_token_cut`` (via
``rank_role.unwired_token_cut_seams``), which returns before reading the
register for every form other than ``kv=qsa_forma_dcp``. So a boot without the
cut is byte-identical whatever F14 says. Pinned here:

* the flip itself (F14 wired, no token-cut seam open, F14 out of the
  unwired order);
* every form without the cut passes the riegel silently with F14 wired AND
  unwired (no refusal, no line);
* the flag has no other reader in the runtime tree (no ``require_wired("F14")``,
  no second caller of ``unwired_token_cut_seams``) -- a new reader would be a
  switch that acts without the cut, and must extend this pin."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import re
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402


def test_f14_is_wired_and_no_token_cut_seam_is_open():
    assert rank_role.SEAMS["F14"].wired
    assert rank_role.unwired_token_cut_seams() == ()
    assert "F14" not in rank_role.UNWIRED_ORDER


def _run_riegel(kv, dry):
    ns = types.SimpleNamespace(dry_run=dry, d_kv_token_cut="off", weg2_disable_hicache=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.refuse_unwired_token_cut(ns, None if kv is None else types.SimpleNamespace(kv=kv))
    return out.getvalue()


def test_without_the_cut_the_switch_changes_nothing():
    open_f14 = dict(rank_role.SEAMS)
    open_f14["F14"] = dataclasses.replace(open_f14["F14"], wired=False)
    for kv in (None, "qsa_forma", "paged_dcp", "paged", "mha"):
        for dry in (False, True):
            wired_out = _run_riegel(kv, dry)
            with mock.patch.object(rank_role, "SEAMS", open_f14):
                unwired_out = _run_riegel(kv, dry)
            assert wired_out == unwired_out == "", (kv, dry, wired_out, unwired_out)


def test_the_flag_has_no_other_reader():
    root = os.path.dirname(rank_role.__file__)
    callers, requires = [], []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            text = open(path, encoding="utf-8", errors="replace").read()
            rel = os.path.relpath(path, root)
            if re.search(r"require_wired\(\s*['\"]F14['\"]", text):
                requires.append(rel)
            if "unwired_token_cut_seams(" in text and rel != "rank_role.py":
                callers.append(rel)
    assert requires == []
    assert callers == [os.path.join("weg2", "launcher.py")], callers
