# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Every declared marker must be producible by real source (#349).

THE BUG CLASS. A marker that matches nothing cannot fail, and an arm built on
one is disarmed while looking armed. The #349 determination found a live
instance: ``reject_dcp_crossalgo`` declared ``"--speculative-cross-algorithm"``
as a reject marker, but the guard that actually fires
(``server_args.py``, the DCP x cross-algo refusal) words it as
"cross-algorithm speculative serving" and never prints the flag spelling. Since
``first_refusal`` requires ALL markers in ONE refusal message, the arm reported
FAIL on every sweep -- including 2026-08-01, where the recorded reason was
"refused, but with an unexpected error rather than the named guard". It stayed
red for weeks while the server was refusing for exactly the right reason.

A permanently red arm is not a stricter net. It is a net people stop reading,
which ``arms.py``'s own docstring says in as many words about the deleted
draft-extend refusal: "A net that keeps asserting a deleted refusal reports a
defect every run and teaches everyone to stop reading it."

So the marker literals are pinned against the source that must produce them.
This is a text check, not a boot: it cannot prove the guard FIRES, only that
the words it is matched against still exist somewhere that could emit them.
That is precisely the half that rots silently -- the firing half goes red
loudly on the next sweep.
"""

from __future__ import annotations

import os
import unittest

from sglang.srt.boot_matrix import arms as arms_mod
from sglang.srt.boot_matrix.arms import (
    ARMS,
    SPILL_MARKER_DECODE,
    SPILL_MARKER_PREFILL,
)

_SRT = os.path.dirname(os.path.dirname(os.path.abspath(arms_mod.__file__)))


def _read(rel: str) -> str:
    with open(os.path.join(_SRT, rel), errors="replace") as f:
        return f.read()


class TestSpillMarkersStillExist(unittest.TestCase):
    """The two centralised spill literals must still be emitted somewhere."""

    def test_decode_spill_marker_is_producible(self):
        self.assertIn(SPILL_MARKER_DECODE, _read("managers/kv_session_offload.py"))

    def test_prefill_spill_marker_is_producible(self):
        self.assertIn(SPILL_MARKER_PREFILL, _read("managers/kv_session_offload.py"))

    def test_the_two_markers_are_distinct(self):
        # They name different mechanisms on purpose: an arm that can only
        # decode-spill must not satisfy its precondition with a prefill line.
        self.assertNotEqual(SPILL_MARKER_DECODE, SPILL_MARKER_PREFILL)


class TestRejectMarkersAreProducible(unittest.TestCase):
    """Every reject arm's markers must appear in the arg-resolution source.

    Reject arms are refused at argument time, so ``server_args.py`` is where
    their words have to live. A marker absent from it can never appear in a
    refusal message, and the arm can then only ever go red.
    """

    @staticmethod
    def _raise_messages(source: str) -> list:
        """Every string a ``raise`` in this module can put in a message.

        NOT a plain substring search over the file: every flag spelling occurs
        in ``server_args.py`` anyway, as its own argparse definition, so a
        whole-file check passes for markers no refusal can ever print. (This
        test was written that way first and passed while the known-broken arm
        was still broken -- the check has to look where the words are EMITTED.)
        """
        import ast

        blobs = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Raise):
                continue
            parts = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    parts.append(sub.value)
            blobs.append("".join(parts))
        return blobs

    def test_every_reject_marker_appears_in_server_args(self):
        messages = self._raise_messages(_read("server_args.py"))
        missing = []
        for arm in ARMS:
            if arm.kind != "reject":
                continue
            for marker in arm.reject_markers:
                if not any(marker in m for m in messages):
                    missing.append(f"{arm.name}: {marker!r}")
        self.assertEqual(
            missing,
            [],
            "reject markers that no source line can produce -- these arms can "
            "only ever report FAIL:\n  " + "\n  ".join(missing),
        )

    def test_there_are_reject_arms_to_check(self):
        # Guards the test above against becoming vacuous if the roster changes.
        self.assertTrue([a for a in ARMS if a.kind == "reject"])


if __name__ == "__main__":
    unittest.main()
