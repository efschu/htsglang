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
"""The exact VRAM ledger.

    card_total = user_reserve + internal_demand + kv_pool

``user_reserve`` is the operator's headroom for things OUTSIDE this engine and
defaults to 1024 MiB per card. ``internal_demand`` is the itemized sum of this
package's terms, each MODELED from configuration or CALIBRATED once per
hardware fingerprint. ``kv_pool`` is the residual, so surplus is never idle.

An overcommit is a parse-time REFUSAL that prints the itemization, not a
warning. The "short by N MiB" warning class does not exist here.

* :mod:`terms` -- the row primitive, the card ledger, the refusal.
* :mod:`engine` -- the serving engine's own rows, built by calling the existing
  derivations rather than restating them.
* :mod:`calibration` -- the hardware residuals, measured once per fingerprint.
* :mod:`tenants` -- how a co-resident lane declares into the same sum, and why
  it cannot ship without doing so.
* :mod:`contract` -- the boot contract: assemble, print, refuse or fit.
"""

from sglang.srt.mem_ledger.terms import (  # noqa: F401
    DEFAULT_USER_RESERVE_MIB,
    CardVramLedger,
    LedgerError,
    LedgerOvercommit,
    LedgerTerm,
    Provenance,
    render_all,
    summarize,
    validate_all,
)

__all__ = [
    "DEFAULT_USER_RESERVE_MIB",
    "CardVramLedger",
    "LedgerError",
    "LedgerOvercommit",
    "LedgerTerm",
    "Provenance",
    "render_all",
    "summarize",
    "validate_all",
]
