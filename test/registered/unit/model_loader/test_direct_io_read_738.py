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
"""#738: don't cache the weights you are about to evict -- you cannot evict them.

WHY O_DIRECT AND NOT AN EVICTION LEVER. Both eviction-advice rungs are dead on
ZFS, and the second one was found here:

* `posix_fadvise(POSIX_FADV_DONTNEED)` -- measured dead by #408.
* `MADV_PAGEOUT`, #408's replacement -- measured dead 2026-08-17 on this pool:
  it returns **rc=0**, reporting the advice ACCEPTED, and freed 529 of 136,781
  pages (0.4%). Silent success is the worst failure shape: the flag looked
  wired, the code was the ladder, and nothing happened.

So the spike is attacked at its cause. Measured on this rig (OpenZFS 2.3.4,
real direct I/O): an aligned 1.20 GB O_DIRECT read moved the page cache by
**zero pages**, where a buffered read of the same shard adds ~293k.

BYTE-IDENTICAL IS THE CONTRACT. This is an I/O-route change and nothing else,
so the first test compares the bytes and the rest may not weaken that.
"""

import os
import tempfile
import unittest

from sglang.srt.model_loader.weight_utils import (
    _DIRECT_ALIGN,
    _DIRECT_BLOCK,
    read_file_direct,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class TheBytesAreIdentical(unittest.TestCase):
    """The contract: only the route into memory changes."""

    def _roundtrip(self, nbytes):
        with tempfile.NamedTemporaryFile(dir="/spinning", delete=False) as f:
            payload = os.urandom(nbytes)
            f.write(payload)
            path = f.name
        try:
            self.assertEqual(payload, read_file_direct(path))
        finally:
            os.unlink(path)

    def test_a_block_aligned_file(self):
        self._roundtrip(_DIRECT_BLOCK)

    def test_a_file_with_an_unalignable_tail(self):
        """The tail is read buffered on purpose -- kilobytes, not gigabytes."""
        self._roundtrip(_DIRECT_BLOCK + 12345)

    def test_a_file_smaller_than_one_block(self):
        """All tail, no direct read at all: must still return the bytes."""
        self._roundtrip(_DIRECT_ALIGN + 7)

    def test_an_empty_file(self):
        self._roundtrip(0)


class TheReadActuallyRequestsDirectIO(unittest.TestCase):
    """The mechanism, asserted where it can FAIL.

    A COLD-FILE CACHE TEST CANNOT BE BUILT HERMETICALLY HERE, and the first
    version of this class was vacuous for exactly that reason: it wrote a temp
    file (which caches it), read it back, and measured ~0 growth whether or not
    O_DIRECT was used -- dropping O_DIRECT left all tests green. Making the file
    cold first would require evicting it, and eviction is the no-op this whole
    ticket is about. So the property is split:

    * the MECHANISM is asserted here, by recording the open flags -- this fails
      the moment the code stops asking for O_DIRECT;
    * the EFFECT was measured on metal against a genuinely cold 1.20 GB shard
      (page-cache delta ZERO pages, ARC +174 MiB against a -4.5 MiB baseline
      drift) and is recorded in the ledger, where a measurement belongs.
    """

    def test_the_file_is_opened_with_O_DIRECT(self):
        import sglang.srt.model_loader.weight_utils as wu

        seen = {}
        real_open = os.open

        def spy(path, flags, *a, **kw):
            if str(path).endswith(".bin"):
                seen["flags"] = flags
            return real_open(path, flags, *a, **kw)

        with tempfile.NamedTemporaryFile(
            dir="/spinning", suffix=".bin", delete=False
        ) as f:
            f.write(os.urandom(_DIRECT_BLOCK + 11))
            path = f.name
        wu.os.open = spy
        try:
            wu.read_file_direct(path)
        finally:
            wu.os.open = real_open
            os.unlink(path)
        o_direct = getattr(os, "O_DIRECT", 0o40000)
        self.assertIn("flags", seen, "read_file_direct never opened the file")
        self.assertTrue(
            seen["flags"] & o_direct,
            "read_file_direct opened without O_DIRECT: the page cache will fill "
            "and nothing can evict it on ZFS",
        )


class TheFlagRefusesAnInertCombination(unittest.TestCase):
    """CAN-FAIL: direct-io without disable-mmap must not look enabled.

    The mmap path performs no read for O_DIRECT to redirect, so accepting the
    flag there would leave the spike in place while the config claims otherwise
    -- which is precisely the #738 failure mode being fixed.
    """

    def test_direct_io_without_disable_mmap_raises(self):
        import types

        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(
            weight_loader_direct_io=True,
            weight_loader_disable_mmap=False,
        )
        with self.assertRaises(ValueError) as cm:
            ServerArgs._validate_direct_io(args)
        self.assertIn("--weight-loader-disable-mmap", str(cm.exception))

    def test_the_pair_together_is_accepted(self):
        import types

        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(
            weight_loader_direct_io=True,
            weight_loader_disable_mmap=True,
        )
        ServerArgs._validate_direct_io(args)  # must not raise

    def test_neither_flag_is_accepted(self):
        import types

        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(
            weight_loader_direct_io=False,
            weight_loader_disable_mmap=False,
        )
        ServerArgs._validate_direct_io(args)


if __name__ == "__main__":
    unittest.main()
