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
"""The boot contract.

Three statements, and the boot keeps exactly one of them:

1. Every card's ``user_reserve + demand`` fits, the itemization is printed, and
   the residual becomes that card's KV pool. This is a FIT, and it is exact --
   no card is left with memory nobody claimed and no card is over-promised.
2. A card cannot hold ``user_reserve + demand``. This is a REFUSAL at
   parse/validate time, and it prints the same itemization, so the operator
   sees which term is too big rather than a total that is too big.
3. A term that belongs on a card could be neither computed nor bounded by a
   mechanism. This is also a REFUSAL, and it names the missing bound.

There is no fourth statement. In particular there is no "boot anyway with a
warning", which is what the ``short by N MiB`` message was: a boot that had
already decided to proceed with a number it had itself just called wrong.

WHY REFUSAL AND NOT REPAIR. A ledger that silently lowered a term to make the
card fit would be choosing serving behaviour on the operator's behalf --
shorter context, fewer graphs, a smaller mamba pool -- and it would do it at
the moment the operator is least able to see it. The remedy for an
overcommitted card is a configuration change, and the refusal names the levers.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from sglang.srt.mem_ledger.terms import (
    CardVramLedger,
    LedgerOvercommit,
    render_all,
    summarize,
)

logger = logging.getLogger(__name__)

__all__ = ["enforce_boot_contract", "kv_pool_mib_per_rank"]


def enforce_boot_contract(
    ledgers: Sequence[CardVramLedger],
    *,
    log: bool = True,
) -> List[CardVramLedger]:
    """Print the itemized ledger, then fit or refuse.

    Printing happens BEFORE the verdict on purpose: on a refusal the operator
    needs the itemization most, and a message that raises first and explains
    second is one an exception handler can swallow.
    """
    if log:
        for line in render_all(ledgers).splitlines():
            logger.info("%s", line)
        totals = summarize(ledgers)
        logger.info(
            "VRAM ledger totals: %d MiB of card, %d MiB user reserve, %d MiB "
            "computed demand, %d MiB KV pool. The three add up to the first "
            "number exactly; nothing is padding for anything else.",
            totals["total_mib"],
            totals["user_reserve_mib"],
            totals["demand_mib"],
            totals["kv_pool_mib"],
        )
    bad = [x for x in ledgers if not x.fits]
    if bad:
        raise LedgerOvercommit(bad)
    return list(ledgers)


def kv_pool_mib_per_rank(
    ledgers: Sequence[CardVramLedger],
    rank_gpu_id: Sequence[int],
) -> List[int]:
    """Per-rank KV budget: the card's residual, split among its ranks.

    THE SURPLUS RULE. Whatever the reserve and the demand do not take goes
    here. That is what makes an over-large user reserve cost KV capacity
    visibly (it does, and it should) and what makes an accurately modelled
    demand return capacity that a padded reserve used to keep idle.

    The split is integer division with the remainder handed to the LOWEST rank
    on the card, which is arbitrary but stated: the alternative -- dropping the
    remainder -- leaves up to (ranks - 1) MiB on the card that nothing claims,
    and this ledger's whole contract is that nothing is left unclaimed.
    """
    by_gpu = {x.gpu_id: x for x in ledgers}
    out: List[int] = [0] * len(rank_gpu_id)
    for gpu_id, ledger in by_gpu.items():
        ranks = [r for r, g in enumerate(rank_gpu_id) if g == gpu_id]
        if not ranks:
            continue
        share, remainder = divmod(ledger.kv_pool_mib, len(ranks))
        for i, rank in enumerate(sorted(ranks)):
            out[rank] = share + (remainder if i == 0 else 0)
    return out
