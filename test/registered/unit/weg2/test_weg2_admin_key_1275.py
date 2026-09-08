# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1275: a per-boot admin key, and the trap that makes it more than one flag.

THE CAPABILITY. Boot weg2sb4 went over the host reap watermark because the
ledger's ``store = min(leftover, reap-bound)`` took the leftover branch.
``POST /hicache/storage-backend/resize`` shrinks the store IN FLIGHT and is
documented not to need an idle scheduler -- but it self-gates on
``server_args.admin_api_key``, which the launcher never passed. The endpoint
existed; the capability did not.

THE TRAP, and it is a boot-killer. ``ADMIN_OPTIONAL`` does not mean "optional
to authenticate" -- it means "require the ADMIN key once one is configured".
``/flush_cache``, ``/release_memory_occupation``, ``/resume_memory_occupation``
and ``/abort_request`` all carry that level, and they are exactly what the
front drives every flip with. They answer today ONLY because no key is set.
Adding ``--admin-api-key`` without teaching the front to authenticate turns the
next quiesce into a 401 and kills the flip.

RED-FIRST is therefore not a formality here: the first test below fails on the
pre-#1275 tree for the RIGHT reason (the front sends no header while the groups
demand one), which is the boot death expressed as an assertion.
"""

import inspect
import os
import re
import stat
import tempfile
import unittest

from sglang.srt.weg2 import admin_key as ak
from sglang.test.test_utils import CustomTestCase

HTTP_SERVER = "python/sglang/srt/entrypoints/http_server.py"


def _repo_root():
    here = os.path.abspath(__file__)
    while here != "/" and not os.path.isdir(os.path.join(here, "python", "sglang")):
        here = os.path.dirname(here)
    return here


def admin_optional_routes():
    """Every ADMIN_OPTIONAL route, read OUT OF http_server.py.

    Enumerated from the decorators rather than hard-coded, so this suite cannot
    keep passing after a route changes level -- which is the drift that would
    silently re-open the boot-killer.
    """
    src = open(os.path.join(_repo_root(), HTTP_SERVER), encoding="utf-8").read()
    lines = src.split("\n")
    out = set()
    for i, line in enumerate(lines):
        m = re.search(r'@app\.(?:api_route|post|get)\(\s*"([^"]+)"', line)
        if not m:
            continue
        for j in range(i + 1, min(i + 5, len(lines))):
            a = re.search(r"@auth_level\(AuthLevel\.(\w+)\)", lines[j])
            if a:
                if a.group(1) == "ADMIN_OPTIONAL":
                    out.add(m.group(1))
                break
            if lines[j].lstrip().startswith(("async def", "def ")):
                break
    return out


class TheBootKiller(CustomTestCase):
    """RED-FIRST: the front must authenticate, or the flip dies at the quiesce."""

    def test_the_front_routes_really_are_admin_optional(self):
        """The premise, checked against the tree rather than assumed."""
        found = admin_optional_routes()
        self.assertTrue(found, "no ADMIN_OPTIONAL routes parsed -- the scan broke")
        for path in ak.FRONT_ADMIN_ROUTES:
            self.assertIn(
                path, found,
                f"{path} is no longer ADMIN_OPTIONAL; if it became NORMAL this "
                f"suite's premise changed, if it became ADMIN_FORCE the front "
                f"needs the key even harder",
            )

    def test_the_front_sends_the_bearer_token_on_every_rpc(self):
        """THE fix. Pre-#1275 `Front.rpc` posts with no headers at all."""
        from sglang.srt.weg2 import front as front_mod

        src = inspect.getsource(front_mod.Front.rpc)
        self.assertIn("headers=", src, "Front.rpc sends no headers -> 401 on the quiesce")
        self.assertIn("auth_headers", src)

    def test_an_unkeyed_boot_sends_nothing_and_is_byte_identical(self):
        """A boot without the flag must behave exactly as before."""
        self.assertEqual(ak.auth_headers(None), {})
        self.assertEqual(ak.auth_headers(""), {})

    def test_the_header_is_the_bearer_shape_the_server_parses(self):
        """`auth.py` splits on whitespace and compares parts[1]."""
        h = ak.auth_headers("SECRET")
        self.assertEqual(h["Authorization"], "Bearer SECRET")
        self.assertEqual(h["Authorization"].split(" ")[1], "SECRET")


class TheKeyFile(CustomTestCase):
    def test_written_0600(self):
        with tempfile.TemporaryDirectory() as d:
            p = ak.write(os.path.join(d, "k"), "abc")
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)

    def test_an_existing_wider_file_is_narrowed(self):
        """O_CREAT does not change an existing file's mode -- so chmod as well."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k")
            open(p, "w").close()
            os.chmod(p, 0o644)
            ak.write(p, "abc")
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = ak.write(os.path.join(d, "k"), "s3cr3t")
            self.assertEqual(ak.read(p), "s3cr3t")

    def test_absence_is_none_not_an_exception(self):
        self.assertIsNone(ak.read("/nonexistent/definitely/not/here"))

    def test_a_fresh_key_per_boot(self):
        self.assertNotEqual(ak.mint(), ak.mint())
        self.assertGreaterEqual(len(ak.mint()), 32)

    def test_the_path_is_beside_the_boots_own_state_json(self):
        self.assertEqual(ak.key_path("/g", "weg2sb5"), "/g/weg2/boot_weg2sb5.adminkey")


class BothGroupsOrNeither(CustomTestCase):
    def test_the_flag_goes_on_p_and_d_from_one_helper(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.admin_key_flag("K"), ["--admin-api-key", "K"])
        self.assertEqual(L.admin_key_flag(None), [])
        for fn in (L.argv_p, L.argv_d):
            self.assertIn("admin_key_flag", inspect.getsource(fn),
                          f"{fn.__name__} does not carry the key -> half a capability")

    def test_an_unkeyed_argv_is_unchanged(self):
        """Every pre-#1275 argv must be byte-identical when no key is minted."""
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.admin_key_flag(None), [])


class TheSecretDoesNotLeak(CustomTestCase):
    def test_the_key_is_excluded_from_the_p_form_key(self):
        """TWO reasons, either sufficient: a per-boot random value would give
        every boot its own form (no boot could ever match a predecessor's
        image), and the normalised form string is LOGGED."""
        from sglang.srt.weg2 import ring_table

        self.assertIn("--admin-api-key", ring_table.FORM_KEY_EXCLUDED_FLAGS)

    def test_the_form_key_is_identical_with_and_without_the_key(self):
        from sglang.srt.weg2 import ring_table

        base = ["py", "-m", "sglang.launch_server", "--model-path", "/m",
                "--tp-size", "1", "--pp-size", "3", "--port", "30031"]
        k1, f1 = ring_table.p_form_key(base)
        k2, f2 = ring_table.p_form_key(base + ["--admin-api-key", "SECRET"])
        self.assertEqual(k1, k2, "a per-boot key changed the form key")
        self.assertNotIn("SECRET", f2, "the key leaked into the LOGGED form string")

    def test_redact_never_returns_the_key(self):
        self.assertNotIn("supersecretvalue", ak.redact("supersecretvalue"))
        self.assertEqual(ak.redact(None), "(none)")

    def test_the_front_is_handed_a_path_not_a_value(self):
        """The groups have no choice (server_args takes only --admin-api-key,
        so their key is world-readable in /proc/<pid>/cmdline). The front does
        have a choice, and a second exposure bought with nothing is not taken."""
        from types import SimpleNamespace

        from sglang.srt.weg2 import launcher as L

        ns = SimpleNamespace(tag="t", fairness_w_s=45.0, drain_deadline_s=90.0,
                             min_dwell_ms=None, d_admit_max_tokens=None)
        argv = L.front_argv_for("py", "/store", 1, 2, {}, [], ns, 0, 0, 8, 8,
                                22000, 22000, "D",
                                admin_key_file="/g/weg2/boot_t.adminkey")
        # BEHAVIOURAL, not textual: an earlier version of this test grepped the
        # function's SOURCE and went red on the explanatory comment, which is
        # the assert-on-a-literal failure this campaign keeps paying for.
        self.assertIn("--admin-key-file", argv)
        self.assertIn("/g/weg2/boot_t.adminkey", argv)
        self.assertNotIn("--admin-api-key", argv,
                         "the front's argv must carry the PATH, never the key")

    def test_the_front_argv_is_unchanged_when_unkeyed(self):
        from types import SimpleNamespace

        from sglang.srt.weg2 import launcher as L

        ns = SimpleNamespace(tag="t", fairness_w_s=45.0, drain_deadline_s=90.0,
                             min_dwell_ms=None, d_admit_max_tokens=None)
        base = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, 1, 1, "D")
        self.assertNotIn("--admin-key-file", base)

    def test_the_launcher_logs_the_path_and_not_the_key(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.main)
        i = src.find("WEG2 ADMIN-KEY")
        self.assertGreater(i, -1, "no minting log line found")
        stmt = src[i:i + 1200]
        # The line may report DRY vs written, so match on the INVARIANT rather
        # than on the exact wording: the path is interpolated, the key never is.
        self.assertNotIn("{admin_api_key}", stmt,
                         "the launcher log line interpolates the KEY itself")
        self.assertIn("{admin_key_file}", stmt)

    def test_no_log_call_anywhere_interpolates_the_key(self):
        """Broader than the one line: the secret must not reach ANY log call."""
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.main)
        for m in re.finditer(r"log\(f?\"[^\"]*\{admin_api_key\}", src):
            self.fail(f"a log call interpolates the key: {m.group(0)[:80]}")


class TheArgvLogIsRedacted(CustomTestCase):
    """Found the honest way: the dry-run log contained the key TWICE.

    The launcher logs each group's full argv (its own log line and that group's
    log-file header), so "no log call interpolates {admin_api_key}" was true and
    the key was in the log anyway. /proc exposure is inherent and local; a log
    file is durable and gets pasted into records, so it is the channel worth
    closing.
    """

    def test_the_value_is_replaced_and_the_flag_kept(self):
        argv = ["py", "--port", "30031", "--admin-api-key", "s3cr3t", "--tp-size", "1"]
        out = ak.redact_argv(argv)
        self.assertIn("--admin-api-key", out)
        self.assertNotIn("s3cr3t", out)
        self.assertIn("<redacted>", out)

    def test_the_shipped_argv_is_not_mutated(self):
        argv = ["--admin-api-key", "s3cr3t"]
        ak.redact_argv(argv)
        self.assertEqual(argv, ["--admin-api-key", "s3cr3t"],
                         "redaction must copy -- the shipped argv is what is exec'd")

    def test_an_unkeyed_argv_is_returned_unchanged(self):
        argv = ["py", "--port", "30031"]
        self.assertEqual(ak.redact_argv(argv), argv)

    def test_a_trailing_flag_with_no_value_does_not_crash(self):
        self.assertEqual(ak.redact_argv(["--admin-api-key"]), ["--admin-api-key"])

    def test_both_launcher_log_sites_redact(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.launch_group)
        self.assertEqual(src.count("redact_argv"), 2,
                         "both the log line and the group log header must redact")
        self.assertIn("spec.argv", src)


class TheDocumentedDirectCall(CustomTestCase):
    """No front proxy was added; the direct call is documented instead."""

    def test_the_launcher_prints_a_runnable_curl_with_the_bearer_header(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L)
        self.assertIn("/hicache/storage-backend/resize", src)
        self.assertIn("Authorization: Bearer $(cat", src,
                      "the documented call must read the key from the 0600 file, "
                      "never embed it")

    def test_the_module_states_what_the_key_is_not(self):
        """It is a capability and an accident barrier, not confidentiality:
        the key rides each group's argv, which is world-readable."""
        doc = inspect.getdoc(ak) or ""
        self.assertIn("not a security boundary", doc)
        self.assertIn("/proc/<pid>/cmdline", doc)


class ADryRunWritesNothing(CustomTestCase):
    """`--dry-run` prints "nothing started, mounted, armed or written" -- and
    minting a key must not quietly falsify that line."""

    def test_the_write_call_sits_inside_an_if_not_dry(self):
        """AST, not text order: an earlier version of this test compared string
        offsets in `main` and went red because the claim it looked for appears
        more than once. The structural fact is the one worth pinning."""
        import ast as _ast
        import textwrap

        from sglang.srt.weg2 import launcher as L

        tree = _ast.parse(textwrap.dedent(inspect.getsource(L.main)))
        guarded = False
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.If):
                continue
            t = node.test
            if isinstance(t, _ast.UnaryOp) and isinstance(t.op, _ast.Not) \
                    and isinstance(t.operand, _ast.Name) and t.operand.id == "dry":
                body = _ast.dump(_ast.Module(body=node.body, type_ignores=[]))
                if "admin_key_mod" in body and "write" in body:
                    guarded = True
        self.assertTrue(guarded, "the key write is not inside `if not dry:`")

    def test_the_dry_log_line_says_it_did_not_write(self):
        from sglang.srt.weg2 import launcher as L

        self.assertIn("DRY: not written", inspect.getsource(L.main))


if __name__ == "__main__":
    unittest.main()
