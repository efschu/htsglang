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
"""#1056: publish the cold-build window AT THE LOADER CHOKEPOINT.

THE CLASS, AND WHY THE PREVIOUS TWO FIXES DID NOT CLOSE IT
----------------------------------------------------------
A rank that loads a Triton module needs the device. Under barlink the peers'
spin kernels saturate the device while they wait for that rank in a collective.
Closed cycle, group wedged, and the abort surfaces as ``Bar1CollectiveAborted``
on somebody else. Two specimens, both fully captured:

* BOOT 22: ``_dcp_write_scatter`` loaded cold on the FIRST FORWARD AFTER A
  CUTOVER. #1033c answered it by arming ``cold_build_window`` for the first
  ``_post_cutover_build_batches = 8`` forwards after each cutover.
* BOOT 25: ``l2norm_fwd`` loaded cold on an ORDINARY forward, on two ranks at
  once, ~1.5 minutes into load and 57 clean cutovers later -- i.e. far outside
  that counter. py-spy, two samples per rank, identical frames:
  ``_init_handles (compiler.py:466) <- l2norm_fwd (l2norm.py:99) <-
  chunk_gated_delta_rule``.

#1033c is not wrong; it is BOUNDED BY A COUNT, and a first-loader whose first
call is late is outside any count. ``devtools/CENSUS_1033c_first_loaders.md``
said so in advance: its rows 7 and 8 are UNVERIFIED and the table calls itself
a LOWER BOUND. An enumeration cannot be completed by adding entries to it, and
a second counter would inherit the same defect.

THE CHOKEPOINT IS THE ANSWER, AND IT IS A SINGLE FUNCTION
---------------------------------------------------------
``CompiledKernel._init_handles`` is the ONE place every Triton kernel in the
process must pass before its module exists -- ``driver.active.utils.load_binary``
(``cuModuleLoadData``) is called there and nowhere else. Wrapping it covers
EVERY future first-loader by construction: no enumeration, no census to keep
current, no counter to tune, and rows 7 and 8 of that census stop mattering.

Since #615, ``cold_build_window`` PUBLISHES to the peers at every call site, so
the window opened here is group-visible without any further plumbing: peers
that reach a collective deadline while it exists stretch that deadline instead
of aborting the group.

STEADY STATE IS A SINGLE ATTRIBUTE CHECK. ``_init_handles`` already opens with
``if self.module is not None: return`` -- the warm path. The wrapper asks the
same question FIRST and delegates untouched, so a warm launch pays one Python
attribute load and never constructs the window. The window is entered only on a
genuine cold miss, which is the event it exists to cover.

EXCEPTION SAFETY IS WHY THIS WRAPS THE METHOD RATHER THAN USING TRITON'S HOOKS.
``knobs.runtime.kernel_load_start_hook`` / ``kernel_load_end_hook`` bracket
``load_binary`` and are the sanctioned extension point -- their existence is
what identifies this as the intended chokepoint. But the END hook is called
only on the success path: a load that RAISES would leave the window open for
the life of the process, permanently stretching every collective deadline. A
``with`` block around the original method closes on both paths, and
``cold_build_window`` is documented re-entrant and exception-safe.

NEVER BREAKS THE BOOT. Installation is wrapped whole and idempotent; if
anything about the Triton internals differs from what is asserted here, the
install declines and logs once, leaving the stock method in place. A failure to
INSTRUMENT must never become a failure to RUN.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["install_triton_loader_window", "loader_window_installed"]

_installed = False
_original = None


def loader_window_installed() -> bool:
    return _installed


def install_triton_loader_window(reason_prefix: str = "triton cold module load") -> bool:
    """Wrap the Triton loader chokepoint. Idempotent; returns True if active."""
    global _installed, _original
    if _installed:
        return True
    try:
        from triton.compiler.compiler import CompiledKernel

        from sglang.srt.utils.jit_cold_build import cold_build_window

        original = CompiledKernel._init_handles

        # The warm-path guard is a PRECONDITION of the zero-cost claim, so it
        # is asserted rather than assumed: if a future Triton drops the early
        # return, this wrapper would open a window on every launch and the
        # deadline stretch would become permanent. Checked by source, because
        # there is no API for it.
        import inspect

        src = inspect.getsource(original)
        if "self.module is not None" not in src:
            logger.warning(
                "#1056: declining to wrap Triton's loader -- this build's "
                "_init_handles has no `self.module is not None` early return, "
                "so the wrapper could not tell a cold load from a warm launch "
                "and would open the window on every kernel launch. Stock "
                "method left in place; the post-cutover window (#1033c) still "
                "covers its own case."
            )
            return False

        def _init_handles_in_window(self):
            # WARM PATH FIRST, and byte-identical in effect: one attribute
            # check, then the stock method, no window constructed.
            if self.module is not None:
                return original(self)
            name = getattr(self, "name", "?")
            try:
                with cold_build_window(f"{reason_prefix}: {name}"):
                    return original(self)
            except Exception:
                # The load's own exception propagates UNCHANGED. The window is
                # already closed by the context manager's __exit__ on this
                # path -- that is the whole reason this is a `with` and not a
                # pair of Triton hooks.
                raise

        CompiledKernel._init_handles = _init_handles_in_window
        _original = original
        _installed = True
        logger.info(
            "#1056: Triton loader chokepoint wrapped -- every cold module load "
            "in this process now runs inside a group-visible cold_build_window "
            "(#615), so a peer at a collective deadline stretches it instead of "
            "aborting the group. Covers every first-loader by construction, "
            "including ones no census enumerates. Warm launches are unaffected."
        )
        return True
    except Exception as e:  # noqa: BLE001 - instrumentation may never break a boot
        logger.warning("#1056: could not wrap the Triton loader chokepoint: %s", e)
        return False


def _uninstall_for_test() -> None:
    """Restore the stock method. For the matched check only."""
    global _installed, _original
    if _installed and _original is not None:
        from triton.compiler.compiler import CompiledKernel

        CompiledKernel._init_handles = _original
        _installed = False
        _original = None
