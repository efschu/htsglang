"""managers unit tests: the S109b seam (see mem_cache/conftest.py)."""

# S109b seam (NF 20.10.): the spills read the write-behind's shared record and only QUEUE a missing L3 copy
# (mem_cache/l3_secured; user 20.10.: no synchronous write into the cache in the scheduler thread). The tests of the
# spill LOGIC were written against the copy existing; they keep that premise by running "the write-behind" at once:
# ``secure_rows_to_l3`` / ``_secure_child_anchors`` are the writer's forms (``*_now``) here. A module that tests the
# asynchronous behaviour itself sets ``L3_ASYNC_REAL = True`` and keeps the scheduler forms.
import pytest as _pytest_s109b  # noqa: E402


@_pytest_s109b.fixture(autouse=True)
def _s109b_write_behind_instant(request, monkeypatch):
    if getattr(request.module, "L3_ASYNC_REAL", False):
        yield
        return
    try:
        from flliper.srt.mem_cache.pool_host import arena_pool as _ap
        from flliper.srt.pdflip import forma_l3_subtree as _fls
    except Exception:  # noqa: BLE001 -- a module without these imports is not affected
        yield
        return
    if hasattr(_ap.ArenaMHAHostPool, "secure_rows_to_l3_now"):
        monkeypatch.setattr(_ap.ArenaMHAHostPool, "secure_rows_to_l3", _ap.ArenaMHAHostPool.secure_rows_to_l3_now)
    if hasattr(_fls, "_secure_child_anchors_now"):
        monkeypatch.setattr(_fls, "_secure_child_anchors", _fls._secure_child_anchors_now)
    try:
        from flliper.srt.mem_cache import l3_secured as _l3s

        monkeypatch.setattr(_l3s, "check", _l3s.check_now)
        monkeypatch.setattr(_l3s, "unsecured_keys_lo", _l3s.unsecured_keys_lo_none)
        monkeypatch.setattr(_l3s, "verify_candidates", _l3s.verify_candidates_now)
        from flliper.srt.mem_cache.pool_host import arena_mamba_pool as _amp

        if hasattr(_amp.ArenaMambaPoolHost, "_secure_before_drop_now"):
            monkeypatch.setattr(_amp.ArenaMambaPoolHost, "_secure_before_drop",
                                lambda self, slots: (self._secure_before_drop_now(slots), set())[1])
    except Exception:  # noqa: BLE001
        pass
    yield
