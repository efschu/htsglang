# Copyright 2026 SGLang Team
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
"""#695: an OOM must lead with the error, not with the whole radix tree.

THE SPECIMEN. 2026-08-16 13:58:37, serving died on a KV allocation failure.
The line that mattered --

    RuntimeError: Out of memory. Try to allocate 512 tokens.
    Available full tokens: 167743 (available=392 + evictable=167351)

-- was followed by roughly 370 lines of bare ``print()`` output: every node of
the radix tree as ``[id] len fr= mr= fll= mll= mv=``, indented with
``" " * depth`` so most of it reads as blank. The first diagnosis of that crash
looked at the visible tail, found ``terminate called without an active
exception`` and no Python frame, and filed it as a teardown abort. The
RuntimeError was 300 lines further up.

So the dump did not merely add noise: it MOVED THE CAUSE OUT OF VIEW and cost a
misclassification.

THREE THINGS ARE WRONG WITH IT, and they are separable:

* it is ``print()``, not a logger -- no level, no rank tag, no timestamp, and
  it cannot be filtered by anyone downstream;
* it is UNBOUNDED -- one line per node, and the tree that killed this instance
  held 207799 tokens;
* it fires by DEFAULT on a production error path.

What an operator needs at an OOM is the shape of the tree, not the tree: how
many nodes, how much is locked, and therefore whether the eviction counter was
promising tokens the actuator could not reach -- which is exactly the #681
counter-vs-actuator question the next reader will be asking.
"""

import io
import logging
import unittest
from contextlib import redirect_stdout

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class _LRUStub:
    """Only the method the printers call."""

    def __init__(self, mamba=False):
        self.mamba = mamba

    def in_list(self, node):
        return bool(node) and node.id % 2 == 0


class _Key(list):
    """A key that answers child_key, like the real RadixKey."""

    def child_key(self, page_size):
        return self[0] if self else 0


class _Node:
    """The two fields the printers read, and a child map."""

    _next = [0]

    def __init__(self, key_len=8, locked=False, mamba=False):
        _Node._next[0] += 1
        self.id = _Node._next[0]
        self.key = _Key([0] * key_len)
        self.value = [0] * key_len
        self.full_lock_ref = 1 if locked else 0
        self.mamba_lock_ref = 1 if locked else 0
        self.mamba_value = [0] if mamba else None
        self.children = {}
        self.last_access_time = 0.0


def _tree(n_nodes=200, locked_every=7):
    """A tree big enough that an unbounded dump is obvious."""
    root = _Node(key_len=0)
    cur = root
    for i in range(n_nodes):
        child = _Node(locked=(i % locked_every == 0), mamba=(i % 3 == 0))
        cur.children[i] = child
        if i % 5 == 0:
            cur = child
    return root


class TheDumpIsBoundedByDefault(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache import mamba_radix_cache as m

        self.m = m
        self.cache = m.MambaRadixCache.__new__(m.MambaRadixCache)
        self.cache.root_node = _tree()
        self.cache.page_size = 1
        self.cache.full_lru_list = _LRUStub(False)
        self.cache.mamba_lru_list = _LRUStub(True)

    def _capture(self):
        buf = io.StringIO()
        with self.assertLogs("sglang.srt.mem_cache.mamba_radix_cache", level="ERROR") as cm:
            with redirect_stdout(buf):
                self.cache.pretty_print()
        return buf.getvalue(), cm.output

    def test_nothing_goes_to_stdout(self):
        """print() on a production error path is not filterable by anyone."""
        stdout, _ = self._capture()
        self.assertEqual(
            "",
            stdout,
            "the radix dump still writes to stdout; an operator cannot set a "
            "level on it and it lands in the boot log unattributed.",
        )

    def test_it_is_one_bounded_record(self):
        _, records = self._capture()
        self.assertEqual(1, len(records))
        self.assertLess(
            len(records[0].splitlines()),
            12,
            "the summary must stay short enough that the RuntimeError above it "
            "is still on screen; 370 lines of tree is what hid the cause at "
            "13:58:37.",
        )

    def test_the_summary_carries_the_shape_an_oom_reader_needs(self):
        _, records = self._capture()
        line = records[0]
        for term in ("nodes", "tokens", "locked"):
            self.assertIn(
                term,
                line.lower(),
                f"the summary omits {term!r}; at an OOM the question is whether "
                "the eviction counter promised tokens the frontier could not "
                "reach, and that needs the locked share.",
            )


class TheFullDumpIsStillAvailable(unittest.TestCase):
    """Bounding it must not delete it -- the detail is real when asked for."""

    def setUp(self):
        from sglang.srt.mem_cache import mamba_radix_cache as m

        self.m = m
        self.cache = m.MambaRadixCache.__new__(m.MambaRadixCache)
        self.cache.root_node = _tree(n_nodes=40)
        self.cache.page_size = 1
        self.cache.full_lru_list = _LRUStub(False)
        self.cache.mamba_lru_list = _LRUStub(True)

    def test_the_flag_restores_the_per_node_detail(self):
        import os

        os.environ["SGLANG_RADIX_DEBUG_DUMP"] = "1"
        try:
            with self.assertLogs(
                "sglang.srt.mem_cache.mamba_radix_cache", level="ERROR"
            ) as cm:
                self.cache.pretty_print()
            body = "\n".join(cm.output)
            self.assertIn("fr=", body)
            self.assertIn("mv=", body)
        finally:
            os.environ.pop("SGLANG_RADIX_DEBUG_DUMP", None)

    def test_even_the_debug_dump_does_not_use_stdout(self):
        import os

        os.environ["SGLANG_RADIX_DEBUG_DUMP"] = "1"
        buf = io.StringIO()
        try:
            with self.assertLogs(
                "sglang.srt.mem_cache.mamba_radix_cache", level="ERROR"
            ):
                with redirect_stdout(buf):
                    self.cache.pretty_print()
            self.assertEqual("", buf.getvalue())
        finally:
            os.environ.pop("SGLANG_RADIX_DEBUG_DUMP", None)


if __name__ == "__main__":
    unittest.main()
