# SPDX-License-Identifier: Apache-2.0
"""L3-REVOKE (28.09.): a persistent L3 store reuses what a broken boot wrote.

rc12z17 (103bfdf29a) wiped D-TP0's live expert bank at its first S1 wake
(10:51:24Z, n=4, rows +33 -> +23) and served garbage until the stop; the pages
it wrote into the persistent store in that window carry the store's own
identity -- the identity separates what was CONFIGURED, not what a running
boot COMPUTED -- so the next boot would have reused them as hits. Measured in
the quarantined store: 3,315 of 21,764 pages written 10:51..10:56Z.

THE DEFECT: no way to take a write window out of a store short of throwing the
whole store away. Fix: ``L3_REVOKED.json`` names windows; the attach (before
any rank starts) MOVES every page written inside one into ``<store>.revoked``;
a malformed record is refused by name. Companion (#259): P and D share every
page by token-id key, so a rope override on one group alone is refused.

Hermetic: no GPU, no server; every disk fact is a temp directory.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, l3_store_audit
from sglang.test.test_utils import CustomTestCase

H = "a" * 64
H2 = "b" * 64
SFX = "_Qwen3.8-Flash-Next_898fe1bf454ff7c1"
T_GOOD = 1790580000.0  # before the window
T_BAD = 1790592700.0   # inside it
WIN = (1790592684.0, 1790593039.0)  # 10:51:24Z .. 10:57:19Z


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))

    @property
    def text(self):
        return "\n".join(self.lines)


def _page(store, stem, mtime, n=4096):
    p = os.path.join(store, stem[:2], stem + ".bin")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"\1" * n)
    os.utime(p, (mtime, mtime))
    return p


def _store(root):
    d = os.path.join(root, "l3-nextflash-NF-fe3b0e1fce")
    os.makedirs(d)
    good = [_page(d, H + SFX, T_GOOD), _page(d, H + ".qsa_indexer" + SFX, T_GOOD)]
    bad = [_page(d, H2 + SFX, T_BAD), _page(d, H2 + ".qsa_indexer" + SFX, T_BAD)]
    return d, good, bad


class TestRevokeWindow(CustomTestCase):
    def test_attach_moves_pages_written_in_a_revoked_window_and_counts_only_the_rest(self):
        with tempfile.TemporaryDirectory() as root:
            d, good, bad = _store(root)
            launcher.l3_revoke_window(d, WIN[0], WIN[1], "rc12z17 S1 live-shrink")
            log = _Log()
            files, _nbytes, _removed = launcher.l3_persist_attach(log, d, dry=False)
            self.assertEqual(files, 2)
            for p in good:
                self.assertTrue(os.path.exists(p))
            for p in bad:
                self.assertFalse(os.path.exists(p))
                moved = os.path.join(d + launcher.L3_REVOKED_SUFFIX, os.path.relpath(p, d))
                self.assertTrue(os.path.exists(moved), moved)  # moved, never deleted
            self.assertIn("revoked=2", log.text)
            self.assertIn("rc12z17 S1 live-shrink", log.text)

    def test_dry_run_moves_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            d, good, bad = _store(root)
            launcher.l3_revoke_window(d, WIN[0], WIN[1], "x")
            log = _Log()
            launcher.l3_persist_attach(log, d, dry=True)
            for p in good + bad:
                self.assertTrue(os.path.exists(p))
            self.assertIn("would be moved", log.text)

    def test_without_a_record_nothing_is_moved(self):
        with tempfile.TemporaryDirectory() as root:
            d, good, bad = _store(root)
            files, _b, _r = launcher.l3_persist_attach(_Log(), d, dry=False)
            self.assertEqual(files, 4)
            self.assertFalse(os.path.exists(d + launcher.L3_REVOKED_SUFFIX))

    def test_a_malformed_record_is_refused_by_name_not_read_as_empty(self):
        with tempfile.TemporaryDirectory() as root:
            d, _good, _bad = _store(root)
            for body in ("{not json", json.dumps({"windows": [{"from_unix": 2, "to_unix": 1}]}),
                         json.dumps({"no": "windows"})):
                with open(os.path.join(d, launcher.L3_REVOKED_FILE), "w") as f:
                    f.write(body)
                with self.assertRaises(launcher.Weg2StoreDiskRefused) as cm:
                    launcher.l3_persist_attach(_Log(), d, dry=False)
                self.assertIn("L3-REVOKE", str(cm.exception))

    def test_windows_accumulate(self):
        with tempfile.TemporaryDirectory() as root:
            d, _g, _b = _store(root)
            launcher.l3_revoke_window(d, 1.0, 2.0, "a")
            launcher.l3_revoke_window(d, 3.0, 4.0, "b")
            self.assertEqual(launcher.l3_revoked_windows(d), [(1.0, 2.0, "a"), (3.0, 4.0, "b")])
            with self.assertRaises(ValueError):
                launcher.l3_revoke_window(d, 5.0, 4.0, "backwards")

    def test_the_revoked_sibling_is_named_in_the_disk_sum_not_read_as_a_store(self):
        with tempfile.TemporaryDirectory() as root:
            d, _g, _b = _store(root)
            os.makedirs(d + launcher.L3_REVOKED_SUFFIX)
            log = _Log()
            with mock.patch.object(os, "statvfs", return_value=mock.Mock(f_frsize=1, f_bavail=10 ** 15)):
                launcher.l3_persist_disk_sum(log, root, d, 10, 0, 0, dry=True)
            self.assertIn("revoked pages held aside", log.text)
            self.assertNotIn(".revoked=no cap record", log.text)

    def test_the_residue_sweep_keeps_the_revoked_sibling(self):
        with tempfile.TemporaryDirectory() as root:
            d, _g, _b = _store(root)
            os.makedirs(d + launcher.L3_REVOKED_SUFFIX)
            launcher.sweep_store_residue(_Log(), d, dry=False, root=root)
            self.assertTrue(os.path.isdir(d + launcher.L3_REVOKED_SUFFIX))

    def test_the_launch_site_reads_the_rope_before_the_identity(self):
        import inspect

        src = inspect.getsource(launcher)
        rope = src.index("l3_persist_check_rope(getattr(ns")
        ident = src.index("l3_persist_check_identity(store_plan.directory")
        self.assertLess(rope, ident)


class TestRopeSymmetry(CustomTestCase):
    """#259: P and D read each other's pages; a rope stretched on one alone is refused."""

    YARN = '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0}}}'

    def test_yarn_on_d_alone_is_refused(self):
        with self.assertRaises(launcher.Weg2StoreDiskRefused) as cm:
            launcher.l3_persist_check_rope(
                "--json-model-override-args '{\"language_model_only\":true}'",
                f"--json-model-override-args '{self.YARN}'")
        self.assertIn("rotate differently", str(cm.exception))

    def test_same_rope_both_groups_and_non_rope_differences_pass(self):
        launcher.l3_persist_check_rope(f"--json-model-override-args '{self.YARN}'",
                                       f"--json-model-override-args '{self.YARN}'")
        launcher.l3_persist_check_rope("--json-model-override-args '{\"language_model_only\":true}'",
                                       "--json-model-override-args '{\"x\":1}'")
        launcher.l3_persist_check_rope("", "")

    def test_todays_store_name_is_pinned(self):
        # the quarantined store's recorded identity (L3_IDENTITY.json, 28.09.) --
        # this change must not rename a store it does not have to.
        ident = {
            "form_kv": "", "generation": "706", "kv_cache_dtype": "fp8_e4m3",
            "model_config_sha1": "cbd3f9cb4153e1af2844294f9d4df4af20f6d475",
            "model_path": "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
            "override_d": "4d77ae4a3f85a614", "override_p": "4d77ae4a3f85a614",
            "profile": "nextflash", "vision": "transient",
            "weights_fp": "3c1c07c454411c8f3712cf05def3a4eed4831445",
        }
        self.assertEqual(
            launcher.l3_persist_dir_name(ident),
            "l3-nextflash-Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-fe3b0e1fce")


class TestAudit(CustomTestCase):
    def test_audit_counts_kinds_pairs_and_window_pages_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            d, _g, _b = _store(root)
            _page(d, "c" * 64 + SFX, T_GOOD)  # a KV page without its QSA sibling
            before = sorted(os.listdir(d))
            rep = l3_store_audit.audit(d, windows=[(WIN[0], WIN[1], "rc12z17")])
            self.assertEqual(sorted(os.listdir(d)), before)
            self.assertEqual(rep["kinds"]["kv"]["pages"], 3)
            self.assertEqual(rep["kinds"]["qsa_indexer"]["pages"], 2)
            self.assertEqual(rep["kv_without_qsa"], 1)
            self.assertEqual(rep["windows"][0]["pages"], 2)

    def test_cli_revoke_then_attach(self):
        with tempfile.TemporaryDirectory() as root:
            d, good, bad = _store(root)
            l3_store_audit.main(["revoke", d, "--from", "2026-09-28T10:51:24Z",
                                 "--to", "2026-09-28T10:57:19Z", "--reason", "rc12z17"])
            self.assertEqual(launcher.l3_revoked_windows(d)[0][:2], WIN)
            launcher.l3_persist_attach(_Log(), d, dry=False)
            self.assertTrue(all(os.path.exists(p) for p in good))
            self.assertFalse(any(os.path.exists(p) for p in bad))


if __name__ == "__main__":
    unittest.main()
