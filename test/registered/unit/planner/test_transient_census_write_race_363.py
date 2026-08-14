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
"""Every rank of a pure-TP boot stages its census through ONE file.

FOUND ON METAL, #363 stage-clock window 2026-08-14. The census file the
window's P5 step requires -- the one whose absence makes a flip REFUSED rather
than priced -- did not parse:

    json.decoder.JSONDecodeError: Extra data: line 20 column 2 (char 429)

433 bytes holding one complete document followed by a trailing ``.0\\n}``:
the tail of a LONGER write left behind by a SHORTER one. Kept as evidence at
/spinning/evidence-363/g3a/census/transient_pp0.json.

WHY. ``TransientCensus.write`` stages through a tmp and ``os.replace``s it,
which is atomic and correct FOR ONE WRITER. The staging path is derived from
the output path alone, so it is the same string in every process::

    path = os.path.join(out_dir, f"transient_pp{self.pp_rank}.json")
    tmp = f"{path}.tmp"

``begin()`` takes ``pp_rank`` from ``model_runner.pp_rank``, and under PURE
TENSOR PARALLELISM every rank is ``pp_rank=0``. ``note()`` runs in all three
scheduler processes and each flushes on its own ``_WRITE_INTERVAL_S`` timer.
Three processes therefore open ONE ``transient_pp0.json.tmp`` with mode
``"w"`` at overlapping times. The atomic-rename idiom makes the PUBLISH
atomic; it never made the STAGING exclusive.

WHAT THESE TESTS PROVE, AND WHAT THEY DO NOT. The COLLISION is proven here and
is reliably red before the fix: concurrent writers lose writes outright, as
``os.replace`` finds the shared tmp already renamed away by a peer
(``[Errno 2] ... transient_pp0.json.tmp -> transient_pp0.json``). The exact
interleaving that produced the CORRUPT file on metal did NOT reproduce
hermetically in this shift (0 of 9 attempts across payload sizes), so no test
here claims to reproduce it. The corrupt file is the metal evidence; the
shared staging path is the mechanism, and it is what these tests pin.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from sglang.srt.planner.transient_census import TransientCensus


def _writer(out_dir: str, states: int, rounds: int, q) -> None:
    """One rank's process. Reports how many of its flushes were LOST."""
    c = TransientCensus(pp_rank=0, gpu_name="card", baseline_free_bytes=8 << 30)
    for i in range(states):
        c.note(f"LOAD_STATE_NUMBER_{i:05d}", (7 << 30) - i)
    lost = 0
    for _ in range(rounds):
        if c.write(out_dir) is None:
            lost += 1
    q.put(lost)


class TestTheCensusStagingIsPerProcess:
    def test_concurrent_ranks_do_not_lose_each_others_writes(self, tmp_path):
        """Three processes, one pp_rank: no flush may be lost, and the
        published document must parse and belong to ONE writer.

        Pre-fix this is red on the lost-write count: every rank stages through
        the same path, so a peer's rename pulls the file out from under it.
        """
        out = str(tmp_path)
        q = mp.Queue()
        procs = [
            mp.Process(target=_writer, args=(out, n, 200, q)) for n in (1, 60, 400)
        ]
        for p in procs:
            p.start()
        lost = [q.get(timeout=180) for _ in procs]
        for p in procs:
            p.join(60)
            assert p.exitcode == 0, "a writer process died"

        assert sum(lost) == 0, (
            f"{sum(lost)} of 600 flushes were LOST ({lost} per rank). Every "
            "rank of a pure-TP boot stages through one "
            "transient_pp0.json.tmp, so a peer's os.replace renames the file "
            "out from under this one. A census that silently drops writes is "
            "the same failure as a census that is not written"
        )

        path = Path(out) / "transient_pp0.json"
        assert path.is_file(), "no census was published at all"
        raw = path.read_text()
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            pytest.fail(
                f"the published census does not parse ({exc}). Its only "
                "consumer is planner/pp_cut_calibration.py:230, which reads "
                f"it with json.load. Tail was {raw[-40:]!r}"
            )
        assert len(doc["samples_by_load_state"]) in (1, 60, 400), (
            "the published document mixes writers: it carries "
            f"{len(doc['samples_by_load_state'])} load states, which is no "
            "single writer's count"
        )

    def test_the_staging_path_distinguishes_the_process(self, tmp_path):
        """The property behind the race, asserted directly and deterministically.

        Written as a property (the pid appears) rather than a literal string
        so a different unique-naming fix still satisfies it.
        """
        c = TransientCensus(pp_rank=0, gpu_name="card", baseline_free_bytes=1 << 30)
        c.note("DECODE", 1 << 29)
        assert c.write(str(tmp_path)) is not None

        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
        assert not leftovers, f"staging files were left behind: {leftovers}"

        staged = c._staging_path(str(tmp_path / "transient_pp0.json"))
        assert str(os.getpid()) in staged, (
            "the staging path does not distinguish this process, so every "
            f"rank of a pure-TP boot stages through the same file: {staged}"
        )
