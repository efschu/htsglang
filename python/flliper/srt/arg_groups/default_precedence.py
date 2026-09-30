# Copyright 2023-2026 SGLang Team
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

"""Precedence between the two kinds of default an arg hook can write.

THE RULE (upstream sgl-project/sglang#33199): a MODEL default yields to a
FEATURE default; explicit user input never yields to either.

Why a helper rather than five more ``is None`` guards. Hooks run in a fixed
order, and ``x is None`` is a proxy for "the user did not set this" that stops
being true the moment an EARLIER hook writes the field. Upstream found the
concrete case: the DeepSeek-V4 model hook sets ``max_running_requests = 256``,
the speculative hook later wants 48 when speculative decoding is on, its guard
reads ``is None``, the field is already 256, and the documented MTP default
never applies -- silently, with the model hook's own comment asserting that
the speculative hook is the later writer. Our #379 fixed a sibling instance of
the same class by giving "off" exactly one representation instead of trusting
an ``is None`` downstream; this generalizes it.

The distinction the guard actually needs is provenance, not emptiness:

* ``None``          -- nobody has written the field. Both kinds may claim it.
* model default     -- written by a model/architecture hook. A feature default
                       overwrites it; that is the whole point of the rule.
* feature default   -- written by a feature hook (speculative decoding, ...).
                       Later model defaults do not touch it, and neither does
                       another feature default: first feature wins, so hook
                       order among features stays as observable as before.
* user value        -- anything present before the hooks ran, or written
                       directly. Untouchable.

Relation to the declaration stash in ``arg_groups/overrides.py``: that stash
(``_resolved_overrides``) records WHICH pass declared a field and resolves
conflicts by "last writer wins". It is a different axis and cannot answer this
question -- it has no notion of model-vs-feature, and the two fills involved
here are imperative hook writes at their legacy slots, not declarations. So
this module adds the missing axis (kind of default) rather than a second copy
of the existing one (source of declaration).

Provenance lives in a private ``_default_provenance`` dict on the args object
(underscore-prefixed, so it is outside the strict-mutation guard and outside
the resolved-configuration surface). Absence of the dict means "no hook has
claimed anything yet", so this module is safe against stubs and against args
objects built by other paths.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROVENANCE_ATTR = "_default_provenance"

MODEL_DEFAULT = "model"
FEATURE_DEFAULT = "feature"


def _provenance(server_args: Any) -> dict:
    table = getattr(server_args, _PROVENANCE_ATTR, None)
    if table is None:
        table = {}
        setattr(server_args, _PROVENANCE_ATTR, table)
    return table


def default_provenance(server_args: Any, field: str) -> Optional[str]:
    """``MODEL_DEFAULT``, ``FEATURE_DEFAULT`` or ``None`` (unset / user value)."""
    return getattr(server_args, _PROVENANCE_ATTR, {}).get(field)


def set_model_default(
    server_args: Any, field: str, value: Any, *, reason: str = ""
) -> bool:
    """Write ``value`` as a MODEL default. Returns True when it was written.

    Claims the field only when it is still ``None``, i.e. exactly the old
    ``if server_args.x is None`` behavior -- but records that the value came
    from a model hook, so a later feature default may replace it.
    """
    if getattr(server_args, field) is not None:
        return False
    setattr(server_args, field, value)
    _provenance(server_args)[field] = MODEL_DEFAULT
    if reason:
        logger.warning("Setting %s to %s %s.", field, value, reason)
    return True


def set_feature_default(
    server_args: Any, field: str, value: Any, *, reason: str = ""
) -> bool:
    """Write ``value`` as a FEATURE default. Returns True when it was written.

    Claims the field when it is unset OR when the only thing in it is a model
    default. An explicit user value, and a feature default already written by
    an earlier feature hook, are both left alone.
    """
    current = getattr(server_args, field)
    if current is not None and default_provenance(server_args, field) != MODEL_DEFAULT:
        return False
    setattr(server_args, field, value)
    _provenance(server_args)[field] = FEATURE_DEFAULT
    if reason:
        logger.warning("%s is reset to %s %s.", field, value, reason)
    return True
