"""AP L15-13d: the two remaining KV-fit decisions at the D wake size by the
bytes the resume actually maps (plan - mapped_now), not the full plan.

The accepted L15-13b fix sized the *main* fit check with
``kv_resume_need_bytes(plan, mapped)`` so a paused kv_cache allocation whose
spans are still mapped no longer double-counts.  Two sibling decision points
kept sizing by the full plan and could therefore reject a resume that the
main check just accepted:

* ``_pdflip_wake_kv_first_ok`` -- decides EARLY vs LATE kv resume on the D wake
  (``free - floor - margin >= need``).
* the WAKE-KV-MID ``kv_mid_ok`` check -- decides whether kv may resume
  mid-legs (``free - floor >= kv_need + rest``).

Both now feed ``kv_resume_need_bytes(plan, mapped)``:
  * mapped bytes readable  -> need = plan - mapped
  * symbol absent / read fails -> ``None`` -> need == plan (byte-identical to
    the pre-13d behaviour, so stock hooks are untouched).

RED-first: this test fails while the two sites still read the raw plan.

Hermetic: pure Python (source slicing + an unbound call on a fake self);
no CUDA, no boot, no GPU.  The module is imported if the CPU host can load
it (no CUDA driver required at import time) and falls back to a text check
for the first site when it cannot.
"""
import pathlib
import re
import types

import pytest

GIB = 1 << 30
MIB = 1 << 20
REL_PATH = "python/flliper/srt/managers/scheduler_components/weight_updater.py"


def _repo_root():
    # test/registered/unit/pdflip/x.py -> parents[4] == repo root
    return pathlib.Path(__file__).resolve().parents[4]


def _source_of_first_ok() -> str:
    """Source of _pdflip_wake_kv_first_ok: real source if importable on CPU,
    else a text slice from 'def _pdflip_wake_kv_first_ok' to the next method."""
    try:
        import inspect

        import flliper.srt.managers.scheduler_components.weight_updater as wu

        for obj in vars(wu).values():
            if isinstance(obj, type) and hasattr(obj, "_pdflip_wake_kv_first_ok"):
                return inspect.getsource(obj._pdflip_wake_kv_first_ok)
    except Exception:  # CPU import unavailable (CUDA deps, env drift)
        pass
    text = (_repo_root() / REL_PATH).read_text()
    m = re.search(r"def _pdflip_wake_kv_first_ok", text)
    assert m, "def _pdflip_wake_kv_first_ok not found"
    nxt = text.find("\n    def ", m.start())
    assert nxt != -1
    return text[m.start():nxt]


def _source_around_wake_kv_mid() -> str:
    """+/-40 lines of the WAKE-KV-MID decision block."""
    text = (_repo_root() / REL_PATH).read_text().splitlines()
    hits = [i for i, line in enumerate(text) if "WAKE-KV-MID" in line]
    assert hits, "WAKE-KV-MID marker not found in weight_updater.py"
    i = hits[0]
    lo, hi = max(0, i - 40), min(len(text), i + 41)
    return "\n".join(text[lo:hi])


# ---------------------------------------------------------------------------
# Source checks: both decision sites feed kv_resume_need_bytes into their need
# ---------------------------------------------------------------------------
def test_wake_kv_first_site_uses_kv_resume_need_bytes():
    src = _source_of_first_ok()
    assert "kv_resume_need_bytes" in src, (
        "_pdflip_wake_kv_first_ok still sizes its need by the full plan; "
        "expected kv_resume_need_bytes(plan, mapped)"
    )
    # the mapped twin must be consulted at this site
    assert "_pdflip_tag_mapped_bytes" in src, (
        "_pdflip_wake_kv_first_ok does not read the mapped-now figure"
    )


def test_wake_kv_mid_site_uses_kv_resume_need_bytes():
    src = _source_around_wake_kv_mid()
    assert "kv_resume_need_bytes" in src, (
        "WAKE-KV-MID kv_need still sized by the full plan; expected "
        "kv_resume_need_bytes(plan, mapped)"
    )
    assert "_pdflip_tag_mapped_bytes" in src, (
        "WAKE-KV-MID block does not read the mapped-now figure"
    )


# ---------------------------------------------------------------------------
# Behavioural check: _pdflip_wake_kv_first_ok as an unbound function on a fake
# self.  Numbers are chosen so ONLY the mapped-aware need fits:
#   plan = 10 GiB, mapped = 9 GiB  -> need = 1 GiB (fix)  vs 10 GiB (old)
#   free = 3 GiB, floor = 0, margin = 256 MiB
#   old:  3 GiB - 256 MiB < 10 GiB  -> False
#   new:  3 GiB - 256 MiB >= 1 GiB  -> passes the kv check; the xsn323 leg
#        reserve gate then needs _pdflip_leg_min_free_mib = 8192 MiB
#        (8192 - 1024 >= 0 + 256 + 4352).
# ---------------------------------------------------------------------------
def _fake_self(mapped):
    return types.SimpleNamespace(
        _pdflip_tag_bytes=lambda tag: 10 * GIB,
        _pdflip_tag_mapped_bytes=lambda tag: mapped,
        _pdflip_free_bytes=lambda: 3 * GIB,
        _pdflip_corridor_floor_bytes=lambda: 0,
        _pdflip_leg_min_free_mib=8192,  # MiB, above floor+margin+reserve
    )


def test_wake_kv_first_ok_true_when_only_resume_need_fits(monkeypatch):
    import flliper.srt.managers.scheduler_components.weight_updater as wu

    monkeypatch.delenv("FLLIPER_PDFLIP_WAKE_KV_FIRST", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_WAKE_KV_LEG_RESERVE_MIB", raising=False)

    cls = None
    for obj in vars(wu).values():
        if isinstance(obj, type) and hasattr(obj, "_pdflip_wake_kv_first_ok"):
            cls = obj
            break
    assert cls is not None, "no class with _pdflip_wake_kv_first_ok in module"

    # Fixed site: resume need = plan - mapped = 1 GiB -> EARLY is affordable.
    assert cls._pdflip_wake_kv_first_ok(_fake_self(9 * GIB), "kv_cache") is True

    # Mapped figure unavailable (symbol absent) -> need == plan -> LATE,
    # i.e. byte-identical to the pre-13d decision for this budget.
    assert cls._pdflip_wake_kv_first_ok(_fake_self(None), "kv_cache") is False
