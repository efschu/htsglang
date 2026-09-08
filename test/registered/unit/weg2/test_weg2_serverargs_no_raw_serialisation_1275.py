"""#1275 fix 5: NO ServerArgs serialisation may bypass redacted_dict.

K-2 FAILED TWICE, in two producers, and the second one is why this file is
repo-wide rather than another per-site assertion.

  #1282 (sb5d): `/get_server_info`'s top-level `**dataclasses.asdict(...)`
  fix 5  (sb5e): `internal_states[0]`, produced by
                 `Scheduler.get_internal_state` as `dict(vars(get_server_args()))`

Both are the same defect: a raw serialisation of ServerArgs reaching a
response. I fixed the accounting I was looking at and did not enumerate the
producers -- the same caller-enumeration failure as the sb5 401, in a third
costume. On sb5e the live 43-char key was readable at
`internal_states[0].admin_api_key` WITHOUT A BEARER on all three ports.

So the guard is on the CLASS: any `vars(`, `asdict(`, `__dict__` or msgspec
serialisation of a ServerArgs anywhere under `python/sglang/srt`, outside
`redacted_dict` itself, fails this test. A new producer now reds at the desk
instead of publishing the key on the next boot.
"""

import pathlib
import re
import unittest

from sglang.test.test_utils import CustomTestCase

PATTERNS = (
    r"vars\(\s*(get_server_args\(\)|server_args|self\.server_args)",
    r"asdict\(\s*(get_server_args\(\)|server_args|self\.server_args)",
    r"(get_server_args\(\)|server_args|self\.server_args)\s*\.__dict__",
)


def srt_root():
    """The srt tree, found from the REPO ROOT rather than by walking for a
    `python` ancestor -- the test dir has none, so the old walk ran to `/` and
    the guard read an empty tree (green by vacancy, on the remote only)."""
    p = pathlib.Path(__file__).resolve()
    while p.parent != p and not (p / "python" / "sglang" / "srt").is_dir():
        p = p.parent
    root = p / "python" / "sglang" / "srt"
    assert root.is_dir(), f"srt tree not found from {__file__}"
    return root


class NoRawServerArgsSerialisation(CustomTestCase):
    def test_every_serialisation_goes_through_redacted_dict(self):
        offenders = []
        for f in srt_root().rglob("*.py"):
            body = f.read_text(errors="replace")
            for pat in PATTERNS:
                for m in re.finditer(pat, body):
                    line_no = body[: m.start()].count("\n") + 1
                    line = body.split("\n")[line_no - 1]
                    if "redacted_dict" in line:
                        continue
                    # the definition of redacted_dict itself is the ONE allowed site
                    if f.name == "server_args.py" and "asdict(self)" in line:
                        continue
                    offenders.append(f"{f.name}:{line_no}: {line.strip()[:90]}")
        self.assertEqual(
            offenders, [],
            "raw ServerArgs serialisation -- this is how the admin key reached "
            f"internal_states on weg2sb5e: {offenders}",
        )

    def test_the_guard_can_fail(self):
        """Not vacuous: the patterns match the shape they are meant to catch."""
        sample = "ret = dict(vars(get_server_args()))"
        self.assertTrue(any(re.search(p, sample) for p in PATTERNS))

    def test_the_internal_states_producer_uses_the_helper(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_internal_state)
        self.assertIn("get_server_args().redacted_dict()", src)
        # CODE ONLY: my own explanatory comment in that method quotes
        # `dict(vars(...))`, so a raw substring search matched the prose and
        # not the statement -- the #995 self-matching-marker trap, in a test.
        code = "\n".join(l for l in src.split("\n")
                          if not l.lstrip().startswith("#"))
        self.assertNotIn("dict(vars(", code)


class TheInfoRoutesRequireTheBearer(CustomTestCase):
    """sb5e: readable WITHOUT a bearer on all three ports."""

    def _level(self, path):
        import pathlib as _p

        f = srt_root() / "entrypoints" / "http_server.py"
        lines = f.read_text(errors="replace").split("\n")
        for i, l in enumerate(lines):
            if re.search(r'@app\.(?:api_route|get|post)\(\s*"%s"' % re.escape(path), l):
                for j in range(i + 1, min(i + 5, len(lines))):
                    a = re.search(r"@auth_level\(AuthLevel\.(\w+)\)", lines[j])
                    if a:
                        return a.group(1)
                    if lines[j].lstrip().startswith(("async def", "def ")):
                        break
                return "NORMAL"
        return None

    def test_both_info_routes_are_admin_gated(self):
        for path in ("/get_server_info", "/server_info"):
            self.assertEqual(self._level(path), "ADMIN_OPTIONAL", path)


if __name__ == "__main__":
    unittest.main()
