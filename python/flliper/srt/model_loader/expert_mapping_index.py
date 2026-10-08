"""BOOTZEIT 5 (29.09.): the expert-params mapping as an index, not a list scan.

``FusedMoE.make_expert_params_mapping`` returns one tuple
``(param_name, weight_name, expert_id, shard_id)`` per (expert, projection),
expert-major: 3 x num_experts entries, ``weight_name`` of the form
``"experts.<id>.<proj>."``. A model's ``load_weights`` walks that list for
EVERY expert tensor of the checkpoint and takes the first entry whose
``weight_name`` is a substring of the tensor name::

    for mapping in expert_params_mapping:
        param_name, weight_name, expert_id, shard_id = mapping
        if weight_name not in name:
            continue
        ...

For Qwen3.8-Flash-Next (512 experts -> 1536 entries) that is ~768 substring
tests per tensor, on the loader thread, under the GIL. Measured on the rig's
CPU with the identical loop over PP0's 133632 expert names (29 layers x 512
experts x 3 projections x 3 tensors): 10.8 s, 81 us per tensor -- on the
thread whose time is the boot, and every microsecond of it withheld from the
four expert consumers that need the same GIL (z30w-park PP0: weight_loading
61.8 s, loader thread wait_slots 20.5 s, i.e. the consumers were behind).

``ExpertMappingIndex.candidates(name)`` returns EXACTLY
``[m for m in mapping if m[1] in name]`` -- same entries, same order -- so
the loop body runs over the candidates unchanged: every entry it skips is one
the old loop only ``continue``-d past. The index covers the entries whose
``weight_name`` has the form ``experts.<digits>.<segment>.``; a lookahead
regex finds every such substring of ``name`` (overlapping ones included), so
each one is a dict hit. Any entry of another form is still tested by
substring, in its list position -- the index never guesses.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

# Every position where "experts.<digits>.<segment>." starts; the lookahead
# makes overlapping occurrences visible (finditer alone would skip them).
_OCCURRENCE_RE = re.compile(r"(?=(experts\.\d+\.[^.]+\.))")
_INDEXABLE_RE = re.compile(r"experts\.\d+\.[^.]+\.")


class ExpertMappingIndex:
    """First-match-equivalent lookup over an expert-params mapping list."""

    def __init__(self, mapping: Sequence[Tuple]):
        self._mapping = list(mapping)
        self._by_weight_name: Dict[str, List[int]] = {}
        self._unindexed: List[int] = []
        #: how many names were looked up (the load's census line reads it)
        self.lookups = 0
        for i, entry in enumerate(self._mapping):
            weight_name = entry[1]
            if isinstance(weight_name, str) and _INDEXABLE_RE.fullmatch(weight_name):
                self._by_weight_name.setdefault(weight_name, []).append(i)
            else:
                self._unindexed.append(i)

    def __len__(self) -> int:
        return len(self._mapping)

    @property
    def unindexed(self) -> int:
        """Entries still tested by substring (0 for make_expert_params_mapping)."""
        return len(self._unindexed)

    def candidates(self, name: str) -> List[Tuple]:
        """``[m for m in mapping if m[1] in name]``, in list order."""
        self.lookups += 1
        hits: List[int] = []
        if "experts." in name:
            for m in _OCCURRENCE_RE.finditer(name):
                idx = self._by_weight_name.get(m.group(1))
                if idx:
                    hits.extend(idx)
        for i in self._unindexed:
            if self._mapping[i][1] in name:
                hits.append(i)
        if len(hits) > 1:
            hits = sorted(set(hits))
        return [self._mapping[i] for i in hits]
