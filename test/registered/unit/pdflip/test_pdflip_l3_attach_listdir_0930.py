"""30.09. (27B dual1f 9pb9ex): _l3_attach_from_index globbed once per open intent and
re-listed the intent's whole shard each time (1.4M journal lines x ~3200 names/shard) --
the launcher sat 101 % CPU in glob for minutes before any rank started. The staging scan
now lists every directory once. Same files removed, and each directory listed once."""
import os
from unittest import mock

from flliper.srt.mem_cache.storage.file import store_journal as _sj
from flliper.srt.pdflip import launcher as L


def _prep(tmp_path, intents, extra_staging=()):
    d = str(tmp_path)
    for stem in intents:
        final = _sj.page_path(d, stem)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        open(final + ".tmp.aaaa", "w").close()
    for p in extra_staging:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    return d


def _run(d, intents):
    def fake_load(directory, stat_size=None, open_intents=None):
        open_intents.extend(intents)
        return {}, "ok"
    with mock.patch.object(_sj, "load_index", side_effect=fake_load), \
         mock.patch.object(_sj, "persist", return_value=(0, 0, 0)), \
         mock.patch.object(_sj, "snapshot_lock", mock.MagicMock()):
        return L._l3_attach_from_index(lambda *a, **k: None, d, False, 5, None, [])


def test_open_intent_staging_files_are_removed(tmp_path):
    intents = ["ab" + "0" * 30 + f"{i:02d}" for i in range(5)]
    keep = os.path.join(str(tmp_path), "keep.bin")
    d = _prep(tmp_path, intents, extra_staging=[keep])
    _run(d, intents)
    for stem in intents:
        assert not os.path.exists(_sj.page_path(d, stem) + ".tmp.aaaa")
    assert os.path.exists(keep)


def test_each_directory_is_listed_once(tmp_path):
    intents = ["cd" + "1" * 30 + f"{i:02d}" for i in range(50)]
    d = _prep(tmp_path, intents)
    real = os.listdir
    calls = []

    def counting(p):
        calls.append(p)
        return real(p)
    import glob as _glob

    def no_glob(*a, **k):
        raise AssertionError("per-intent glob re-lists the shard: " + str(a[:1]))
    with mock.patch.object(L.os, "listdir", side_effect=counting), \
         mock.patch.object(_glob, "glob", side_effect=no_glob):
        _run(d, intents)
    shard_dirs = {os.path.dirname(_sj.page_path(d, s)) for s in intents}
    listed = [c for c in calls if c in shard_dirs or c == d]
    assert len(listed) == len(set(listed)), listed
