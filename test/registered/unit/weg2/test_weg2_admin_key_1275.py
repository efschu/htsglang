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
import secrets
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
        """THE fix. Pre-#1275 `Front.rpc` posts with no headers at all.

        #1285 moved the post itself out of `Front.rpc` and into
        `Front._rpc_attempt` (one instrumented attempt, so `leg_rpc` can retry
        one).  The GUARD is unchanged and so is its intent -- every path that
        actually posts carries the token -- so it now reads the posting method,
        and asserts there is exactly ONE of those: a second `session.post` that
        skipped the headers is precisely the regression this test exists for.
        """
        from sglang.srt.weg2 import front as front_mod

        posting = [name for name in ("rpc", "_rpc_attempt", "leg_rpc")
                   if "session.post(" in inspect.getsource(
                       getattr(front_mod.Front, name))]
        self.assertEqual(
            posting, ["_rpc_attempt"],
            "exactly one Front method may post an RPC, and it is the one this "
            f"test then checks for headers; found {posting}")
        src = inspect.getsource(front_mod.Front._rpc_attempt)
        self.assertIn("headers=", src, "the RPC post sends no headers -> 401 on the quiesce")
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

        # #1361 [22-fix4] RETIGHTENED from the two-token form: the value is now
        # welded to the flag, because a value starting with '-' was read by
        # argparse as an option and killed boot weg2xsn25's first launch.
        self.assertEqual(L.admin_key_flag("K"), ["--admin-api-key=K"])
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
        # #1361 [22-fix4]: BY PREFIX, not by exact token. The emitter now ships
        # `--admin-api-key=<value>` as ONE token, so a membership test for the
        # bare flag would pass on an argv that carries the key welded to it --
        # this guard would have gone on saying "no key here" while shipping one.
        self.assertEqual(
            [a for a in argv if str(a).startswith("--admin-api-key")], [],
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


class TheRandomBootKiller(CustomTestCase):
    """#1361 [22-fix4] -- a mint that kills 1 boot in 64, by argparse.

    ``secrets.token_urlsafe`` draws from base64url (A-Za-z0-9 plus '-' and
    '_'), so one key in 64 begins with a hyphen. Emitted as the two-token form
    ``["--admin-api-key", "-xY..."]`` argparse reads the VALUE as an option and
    the launch dies with

        argument --admin-api-key: expected one argument

    Boot weg2xsn25's FIRST launch (log ...0913_065608) died exactly there. The
    boot seat measured the rate directly: 3076 of 200000 mints = 1.54 %, which
    is 1/64 to two digits. A dud roughly every 65th boot, a message that names
    the flag and never the cause, and no dry run reproduces it because the next
    mint is fine -- the worst shape a defect can have.

    Two independent fixes, because the value travels into argv, /proc, a file
    and an HTTP header, and a rule that holds in one of those is not a rule:
    the mint no longer produces such a key, and the emitter welds value to flag
    so that ANY key is safe -- including one an operator passes by hand, which
    never goes through ``mint()`` at all.
    """

    #: The boot seat's own sample size, so this test is at least as sensitive
    #: as the measurement that found the defect. At p=1/64 the chance of 200k
    #: clean draws from an unfixed mint is 0 to any precision worth naming.
    N_MINTS = 200_000

    def test_200k_mints_carry_no_leading_hyphen(self):
        offenders = [k for k in (ak.mint() for _ in range(self.N_MINTS))
                     if k.startswith("-")]
        self.assertEqual(
            offenders[:3], [],
            f"{len(offenders)}/{self.N_MINTS} mints start with '-' "
            f"(unfixed rate is 1/64 = 1.56 %); each one is a dead launch",
        )

    def test_the_mint_is_still_a_full_strength_key(self):
        """The rejection must not be a truncation in disguise.

        `lstrip('-')` would also pass the test above while shortening the key
        and biasing its first character; re-drawing keeps the length and the
        alphabet.
        """
        keys = [ak.mint() for _ in range(200)]
        self.assertEqual({len(k) for k in keys}, {len(secrets.token_urlsafe(32))})
        self.assertEqual(len(set(keys)), len(keys), "mint repeated a key")

    def test_the_argv_welds_the_value_to_the_flag(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.admin_key_flag("K"), ["--admin-api-key=K"])
        self.assertEqual(len(L.admin_key_flag("K")), 1, "two tokens is the defect")

    def test_can_fail_a_hyphen_key_survives_the_argv_build(self):
        """CAN-FAIL: the '-' key the mint no longer produces, built anyway.

        This is the control that makes the emitter fix load-bearing rather than
        decorative. Under the old two-token emitter this argv is what argparse
        choked on; under the welded form it is one token and parses.
        """
        import argparse

        from sglang.srt.weg2 import launcher as L

        for key in ("-xY7abc", "--weird", "-"):
            with self.subTest(key=key):
                flag = L.admin_key_flag(key)
                self.assertEqual(flag, [f"--admin-api-key={key}"])
                ap = argparse.ArgumentParser()
                ap.add_argument("--admin-api-key")
                ap.add_argument("--port")
                ns = ap.parse_args(flag + ["--port", "30031"])
                self.assertEqual(ns.admin_api_key, key)
                self.assertEqual(ns.port, "30031")

    def test_can_fail_the_old_two_token_form_really_does_die(self):
        """The counter-proof: without the fix, the same key kills the parse.

        A can-fail that never shows the failure is an assertion about nothing.
        """
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--admin-api-key")
        with self.assertRaises(SystemExit):
            ap.parse_args(["--admin-api-key", "-xY7abc"])

    def test_the_p_form_key_does_not_move_in_either_spelling(self):
        """The trap that the --served-model-name pass already cost us once.

        A flag whose SPELLING changes can re-hash the form key and invalidate
        every ring table. `_flag_pairs` normalises `--flag value` into
        `--flag=value` before hashing and the flag is excluded either way, so
        all four spellings -- including a key that itself starts with '--' --
        must hash identically.
        """
        from sglang.srt.weg2 import ring_table

        base = ["py", "-m", "sglang.launch_server", "--model-path", "/m",
                "--tp-size", "1", "--pp-size", "3", "--port", "30031"]
        k0, _ = ring_table.p_form_key(base)
        for extra in (["--admin-api-key", "SECRET"],
                      ["--admin-api-key=SECRET"],
                      ["--admin-api-key=--weird"],
                      ["--admin-api-key=-xY7abc"]):
            with self.subTest(spelling=extra):
                k, f = ring_table.p_form_key(base + extra)
                self.assertEqual(k, k0, "the admin key moved the P form key")
                self.assertNotIn("SECRET", f, "the key leaked into the LOGGED form")

    def test_the_redactor_knows_the_new_spelling(self):
        """A redactor that silently stops matching is worse than none.

        The call site still believes it redacted, and the key lands in a log
        that gets pasted into records and tickets.
        """
        out = ak.redact_argv(["py", "--admin-api-key=s3cr3t", "--port", "1"])
        self.assertNotIn("s3cr3t", " ".join(out))
        self.assertIn("--admin-api-key=<redacted>", out)
        # and the old spelling still works, for any argv built elsewhere
        out2 = ak.redact_argv(["--admin-api-key", "s3cr3t"])
        self.assertNotIn("s3cr3t", " ".join(out2))
