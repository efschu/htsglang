"""CENSUS-KLASSE + CENSUS-FIXPUNKT (30.09., NF host census record, 493 samples).

(A) STORE-KLASSIFIKATIONSLUECKE. ``ungebucht`` (cgroup shmem no class names)
climbed 6.84 -> 16.55 -> 30.88 -> 52.49 GiB, traced in the front logs:
  09300130 01:45:59  6.84 -> 16.55   D died at 01:45:57 (TP2 gone, teardown
                                     carrier census), sample 2 s later
  09300232 02:55-03:00 16.55 -> 30.88 six steps, every rank alive; host Shmem
                                     (memts) flat at 58.2 GiB throughout
  09301039 10:43:39 30.88 -> 52.49   both groups hung (health false), killed
                                     10:43:53/57
The classifier measured every shmem class by the Pss of the MAPPINGS. The
store is a tmpfs (container-private, /mnt/nf-experts/fnFL2); its pages stay
in the cgroup's ``shmem`` whether a sampled process maps them or not. Any
store page whose share no sampled mapping carried at the walk -- a rank
exiting between cgroup.procs and its smaps read, or any other lost share --
left ``store`` for ``ungebucht``: the cold tier counted twice (cold_tier_shm
post + ungebucht), trimmed today only by the NF1d cap -- and, from the
LEDGER-FIXPOINT on (83f7fbfb2c), such a sample would make ``shm_rest_max``
(total - store - arena) ~50 GiB and lift the cap off: the double booking back.
Fix: a tmpfs FILE is measured by its allocated blocks, once per inode (the
store dir and /dev/shm walked, every other mapped tmpfs file stat'ed); only a
pathless shared region (or a deleted file) keeps its mapping Pss.

(B) ROLLEN-MAXIMA OHNE VERFALL. nonrank_anon 8.88 = all-time maxima of the
key (inductor_compile_worker 2.28 and nonrank 8.85 set by 09292034, server_main
2.36 by 09300341, detokenizer 2.18 by the creep seed of 29.09.). Fix: a
per-boot history; the booking is each field's max over the last
CENSUS_WINDOW_BOOTS boots of the key; the all-time maxima of a pre-fixpoint
record are ONE pseudo-boot, booked until N real boots have measured.
"""

import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from flliper.srt.pdflip import host_census as hc
from flliper.srt.pdflip import host_ledger as hl

GIB = 1 << 30
KIB = 1024
STORE = "/mnt/nf-experts/fnFL2"
MOUNTS = "tmpfs /mnt/nf-experts tmpfs rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n/dev/sda2 /spinning xfs rw 0 0\n"
WIDTHS = [786432, 58834944]


class FakeFs:
    """A tmpfs tree: path -> allocated bytes (st_blocks * 512)."""

    def __init__(self, files):
        self.files = dict(files)
        self.ino = {p: i + 100 for i, p in enumerate(sorted(self.files))}

    def walk(self, root):
        root = root.rstrip("/")
        by_dir = {}
        for p in self.files:
            if p.startswith(root + "/"):
                by_dir.setdefault(os.path.dirname(p), []).append(os.path.basename(p))
        for d, names in sorted(by_dir.items()):
            yield d, [], names

    def lstat(self, p):
        if p not in self.files:
            raise FileNotFoundError(p)
        return SimpleNamespace(st_mode=0o100600, st_dev=42, st_ino=self.ino[p],
                               st_blocks=self.files[p] // 512, st_size=self.files[p])


def _gib(x):
    return int(x * GIB)


#: the NF shmem of one instant (y4m-sized): store 40, arena 6.5, lane ring
#: 1.95, sidecar 0.32, hand-off 0.23, pathless 1.00 (TMS/memfd)
FS = {
    f"{STORE}/layer{i:02d}.w13.bin": _gib(40.0) // 40 for i in range(40)
}
FS.update({
    "/dev/shm/pdflip-arena-T/arena-786432.bin": _gib(6.5),
    "/dev/shm/pdflip-arena-T/arena-49152.bin": _gib(0.32),
    "/dev/shm/pdflip-arena-T/handoff/h0.bin": _gib(0.23),
    "/dev/shm/pdflip-seq-T/c0_s1_unit_buffer.bin": _gib(1.95),
})
PATHLESS = 1.0
CG_SHMEM = sum(FS.values()) / GIB + PATHLESS


def _smaps(pss_by_path):
    out, a = [], 0x7F0000000000
    for path, b in pss_by_path.items():
        out.append(f"{a:x}-{a + 0x1000:x} rw-s 00000000 00:2a 11 {path}")
        out.append(f"Pss:  {int(b) // KIB} kB")
        a += 0x100000
    return "\n".join(out) + "\n"


def _proc(ranks_mapping_store, dead=()):
    """cgroup.procs: the front + 6 ranks; each live rank maps the whole store
    (its Pss share = 1/mappers of every page) and 1/6 of the pathless region."""
    files = {
        "/sys/fs/cgroup/cgroup.procs": "10\n" + "".join(f"{20 + r}\n" for r in range(6)),
        "/proc/mounts": MOUNTS,
        "/sys/fs/cgroup/memory.stat": f"anon {_gib(20)}\nshmem {int(CG_SHMEM * GIB)}\n",
        "/proc/10/comm": "python\n", "/proc/10/cmdline": "python\0-m\0flliper.srt.pdflip.front\0",
        "/proc/10/smaps_rollup": "Pss_Anon:     1048576 kB\n", "/proc/10/smaps": "",
    }
    for r in range(6):
        pid = 20 + r
        files[f"/proc/{pid}/comm"] = "flliper::schedul\n"
        files[f"/proc/{pid}/cmdline"] = f"flliper::scheduler_TP{r}"
        files[f"/proc/{pid}/smaps_rollup"] = "Pss_Anon:     2097152 kB\n"
        m = {p: b / ranks_mapping_store for p, b in FS.items() if p.startswith(STORE)} if r < ranks_mapping_store else {}
        m["/dev/zero (deleted)"] = PATHLESS * GIB / 6
        if r not in dead:
            files[f"/proc/{pid}/smaps"] = _smaps(m)
    return files


def _reader(files):
    def rd(p):
        if p not in files:
            raise FileNotFoundError(p)
        return files[p]
    return rd


def _sample(files, fs=None):
    fs = fs or FakeFs(FS)
    with mock.patch("os.walk", fs.walk), mock.patch("os.lstat", fs.lstat):
        return hc.sample_live(store_dir=STORE, arena_booked_widths=WIDTHS, reader=_reader(files))


# -- (A) the store is the store, mapped or not ---------------------------------

def test_every_rank_mapping_names_the_store():
    """The control: every mapper read, both instruments agree (to Pss's kB)."""
    c = _sample(_proc(6))
    assert c["shm_classes_gib"]["store"] == pytest.approx(40.0, abs=1e-3)


def test_ranks_exiting_mid_walk_leave_the_store_a_store():
    """09300130 01:45:59 (D died 01:45:57) / 09301039 10:43:39: two of six
    mappers listed but gone at their smaps read. Pss: 4/6 of the store, the
    other 13.33 GiB -> ungebucht. Blocks: the whole store."""
    c = _sample(_proc(6, dead=(4, 5)))
    assert c["shm_classes_gib"]["store"] == pytest.approx(40.0, abs=1e-6)     # Pss gave 26.67
    assert c["unattributed_shm_gib"] < 1.01                                   # Pss gave 14.33
    assert c["skipped_pids"] == 2


def test_no_rank_mapping_the_store_leaves_it_a_store():
    """The store written, every mapper gone (teardown, or before the ranks map
    it): the tmpfs still holds 40 GiB, the cgroup still counts it."""
    c = _sample(_proc(0))
    assert c["shm_classes_gib"]["store"] == pytest.approx(40.0, abs=1e-6)
    assert c["unattributed_shm_gib"] == pytest.approx(PATHLESS, abs=1e-3)
    rest = c["cg_shmem_gib"] - c["shm_classes_gib"]["store"] - c["shm_classes_gib"]["arena_booked"]
    assert rest == pytest.approx(0.32 + 0.23 + 1.95 + PATHLESS, abs=1e-3)


def test_a_mapped_file_is_counted_once_by_its_blocks():
    c = _sample(_proc(6))
    named = sum(v for k, v in c["shm_classes_gib"].items() if k != "anon_shared")
    assert named == pytest.approx(CG_SHMEM - PATHLESS, abs=1e-6)             # not store x 2
    assert c["shm_classes_gib"]["seq_ring"] == pytest.approx(1.95, abs=1e-6)  # walked, unmapped
    assert c["shm_classes_gib"]["arena_sidecar"] == pytest.approx(0.32, abs=1e-6)
    assert c["shm_classes_gib"]["arena_handoff"] == pytest.approx(0.23, abs=1e-6)
    assert c["shm_classes_gib"]["anon_shared"] == pytest.approx(PATHLESS, abs=1e-3)  # Pss, pathless
    assert c["shm_instrument"] == hc.SHM_INSTRUMENT


def test_a_mapped_tmpfs_file_outside_the_walked_roots_is_stated_once():
    extra = "/run/other-tmpfs/seg.bin"
    fs = FakeFs({**FS, extra: _gib(0.5)})
    files = _proc(6)
    files["/proc/mounts"] = MOUNTS + "tmpfs /run/other-tmpfs tmpfs rw 0 0\n"
    for pid in (20, 21):                          # two mappers, each Pss 0.25
        files[f"/proc/{pid}/smaps"] += _smaps({extra: 0.25 * GIB})
    c = _sample(files, fs)
    assert c["shm_classes_gib"]["other_tmpfs"] == pytest.approx(0.5, abs=1e-6)


def test_the_record_rest_is_not_poisoned_by_an_unmapped_store(tmp_path):
    """The LEDGER-FIXPOINT's shm_rest_max (total - store - arena of one
    instant): a sample with the store unmapped must not raise it to ~40 GiB,
    or the cap stops trimming and the store is booked twice."""
    path = str(tmp_path / hc.RECORD_NAME)
    for procs in (_proc(6), _proc(6, dead=(4, 5)), _proc(0)):
        ent = hc.merge_into_record(path, "k", _sample(procs), boot="b1")
    true_rest = 0.32 + 0.23 + 1.95 + PATHLESS
    assert ent["shm_rest_max_gib"] == pytest.approx(true_rest, abs=1e-3)
    assert ent["unattributed_shm_gib"] == pytest.approx(PATHLESS, abs=1e-3)
    assert ent["shm_classes_gib"]["store"] == pytest.approx(40.0, abs=1e-6)


def test_no_double_booking_of_the_store_in_the_ledger(tmp_path):
    """Pricing from a record fed an unmapped-store sample: the unposted shmem
    is the pathless 1.00 less the priced small posts -- never the store."""
    path = str(tmp_path / hc.RECORD_NAME)
    ent = hc.merge_into_record(path, "k", _sample(_proc(0)), boot="b1")
    c = hl.census_shm_posts({}, None)            # no record: nothing
    assert c["unposted_shm_gib"] == 0.0
    terms = hc.ledger_terms(ent)
    t = {"cold_tier_shm_gib": 40.0, "arena_gib": 6.5, "l3_index_gib": 0.0,
         "seq_ring_gib": terms["seq_ring_gib"], "arena_sidecar_gib": terms["arena_sidecar_gib"],
         "arena_handoff_gib": terms["arena_handoff_gib"], "anchors_gib": 0.40, "rings_gib": 0.22}
    posts = hl.census_shm_posts(t, terms)
    assert posts["unposted_shm_gib"] + posts.get("unposted_shm_trim_gib", 0.0) < 1.01
    assert posts["unposted_shm_gib"] == pytest.approx(max(0.0, PATHLESS - 0.62), abs=1e-3)


def test_pathless_and_deleted_regions_keep_their_pss():
    files = _proc(6)
    files["/proc/20/smaps"] += _smaps({"/dev/shm/gone.bin (deleted)": 0.1 * GIB,
                                       "/memfd:tms (deleted)": 0.2 * GIB})
    c = _sample(files)
    assert c["shm_classes_gib"]["other_tmpfs"] == pytest.approx(0.1, abs=1e-6)
    assert c["shm_classes_gib"]["anon_shared"] == pytest.approx(PATHLESS + 0.2, abs=1e-3)


# -- the legacy record: its Pss-era rest is not the rest -----------------------

#: the record on disk (12:04:49Z), the fields the migration reads
Y4M = {
    "samples": 493, "last_at": "2026-09-30T12:04:49Z",
    "unattributed_shm_gib": 52.49, "shm_total_max_gib": 60.14,
    "shm_total_store_gib": 43.96, "shm_total_arena_gib": 6.54,
    "shm_total_source": "live /proc smaps(_rollup) + memory.stat 2026-09-30T08:34:06Z",
    "roles_anon_gib": {"front": 1.16, "server_main": 2.36, "launcher": 0.73, "detokenizer": 2.18,
                       "inductor_compile_worker": 2.28, "ple_pread_worker": 0.14,
                       "mp_resource_tracker": 0.02, "other": 0.0, "rank": 19.61},
    "shm_classes_gib": {"anon_shared": 6.88, "arena_booked": 6.55, "arena_handoff": 0.23,
                        "arena_sidecar": 0.32, "l3idx": 0.14, "other_tmpfs": 0.97,
                        "seq_ring": 1.95, "store": 45.05, "xchg": 0.0},
}


def test_the_legacy_pseudo_boot_takes_the_rest_of_its_peak_instant():
    poisoned = {**Y4M, "shm_rest_max_gib": 52.0, "shm_rest_source": "Pss sample 10:43:39"}
    b = hc.legacy_boot(poisoned)
    assert b["legacy"] and b["boot"] == hc.LEGACY_BOOT
    assert b["shm_rest_max_gib"] == pytest.approx(60.14 - 43.96 - 6.54, abs=1e-6)


# -- (B) the booking spans the last N boots of the key -------------------------

def _census(detok, server=1.0, compile_w=0.0, at="t"):
    return {"roles_anon_gib": {"detokenizer": detok, "server_main": server,
                               "inductor_compile_worker": compile_w, "rank": 19.0},
            "shm_classes_gib": {"store": 40.0, "arena_booked": 6.5, "seq_ring": 1.95},
            "cg_shmem_gib": 50.45, "unattributed_shm_gib": 0.5,
            "shm_instrument": hc.SHM_INSTRUMENT, "source": "t", "at": at}


def _seed(path, key="k", ent=Y4M):
    with open(path, "w") as fh:
        json.dump({key: dict(ent)}, fh)


def test_all_time_maxima_are_booked_until_n_real_boots_have_measured(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    _seed(path)
    allzeit = hc.ledger_terms(Y4M)["nonrank_anon_gib"]
    assert allzeit == pytest.approx(8.87, abs=0.01)
    for i in range(hc.CENSUS_WINDOW_BOOTS - 1):
        ent = hc.merge_into_record(path, "k", _census(1.2 + 0.01 * i), boot=f"b{i}")
        t = hc.ledger_terms(ent)
        assert t["nonrank_anon_gib"] == pytest.approx(allzeit, abs=1e-9), i   # never below before N
        assert hc.LEGACY_BOOT in t["census_window"]
    ent = hc.merge_into_record(path, "k", _census(1.1), boot="bN")
    t = hc.ledger_terms(ent)
    assert hc.LEGACY_BOOT not in t["census_window"]
    assert len(t["census_window"]) == hc.CENSUS_WINDOW_BOOTS
    # max over the five boots, no margin: detok 1.23 + server_main 1.00
    assert t["nonrank_anon_gib"] == pytest.approx(1.2 + 0.01 * (hc.CENSUS_WINDOW_BOOTS - 2) + 1.0, abs=1e-9)
    assert t["census_allzeit_nonrank_anon_gib"] == pytest.approx(allzeit, abs=1e-9)
    assert "CENSUS-FIXPUNKT" in hc.census_line("k", t)


def test_a_boot_older_than_the_window_drops_out(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    hc.merge_into_record(path, "k", _census(3.0), boot="old")
    for i in range(hc.CENSUS_WINDOW_BOOTS):
        ent = hc.merge_into_record(path, "k", _census(1.0), boot=f"b{i}")
    t = hc.ledger_terms(ent)
    assert "old" not in t["census_window"]
    assert t["census_roles"]["detokenizer"] == pytest.approx(1.0)
    assert ent["roles_anon_gib"]["detokenizer"] == pytest.approx(3.0)          # allzeit kept, not booked


def test_real_extra_demand_is_booked_at_once(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    for i in range(hc.CENSUS_WINDOW_BOOTS):
        hc.merge_into_record(path, "k", _census(1.0), boot=f"b{i}")
    ent = hc.merge_into_record(path, "k", _census(2.5, at="t1"), boot="now")    # first sample of a boot
    assert hc.ledger_terms(ent)["census_roles"]["detokenizer"] == pytest.approx(2.5)
    ent = hc.merge_into_record(path, "k", _census(2.9, at="t2"), boot="now")    # later sample, same boot
    assert hc.ledger_terms(ent)["census_roles"]["detokenizer"] == pytest.approx(2.9)


def test_the_booking_covers_every_measured_peak_of_the_window_exactly(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    peaks = [(1.3, 2.0, 0.0), (1.1, 2.4, 1.9), (1.7, 1.8, 0.0), (1.2, 2.1, 2.2), (1.4, 1.9, 0.3),
             (1.0, 2.2, 0.0), (1.6, 2.0, 0.1)]
    for i, (d, s, cw) in enumerate(peaks):
        for frac in (0.5, 1.0, 0.8):                  # three samples, the peak is the boot's max
            ent = hc.merge_into_record(path, "k", _census(d * frac, s * frac, cw * frac), boot=f"b{i}")
        win = peaks[max(0, i + 1 - hc.CENSUS_WINDOW_BOOTS): i + 1]
        r = hc.ledger_terms(ent)["census_roles"]
        for j, role in enumerate(("detokenizer", "server_main", "inductor_compile_worker")):
            assert r.get(role, 0.0) == pytest.approx(max(p[j] for p in win), abs=1e-12), (i, role)
        assert hc.ledger_terms(ent)["nonrank_anon_gib"] == pytest.approx(
            sum(max(p[j] for p in win) for j in range(3)), abs=1e-12)


def test_the_window_is_per_key(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    for i in range(3):
        hc.merge_into_record(path, "other-form", _census(5.0), boot=f"x{i}")
    ent = hc.merge_into_record(path, "k", _census(1.0), boot="b0")
    assert hc.ledger_terms(ent)["census_roles"]["detokenizer"] == pytest.approx(1.0)


def test_without_a_boot_the_record_is_the_old_max_merge(tmp_path):
    path = str(tmp_path / hc.RECORD_NAME)
    ent = hc.merge_into_record(path, "k", _census(1.0))
    assert "boots" not in ent
    assert "census_window" not in hc.ledger_terms(ent)


def test_front_files_every_sample_under_its_boot_tag():
    import inspect

    from flliper.srt.pdflip import front
    assert "boot=boot" in inspect.getsource(front.start_host_census_sampler)
    assert "boot=str(args.tag" in inspect.getsource(front.main)


# -- replay y4m / y4r (W87, 12:29Z / 12:32Z / 14:09Z; one record, one arm) -----

def _y4m_claim(census_terms):
    t = {"anchors_gib": 0.40, "rings_gib": 0.22, "overhead_gib": 0.02,
         "draft_host_p_gib": 119.2 / 1024, "draft_host_d_gib": 59.6 / 1024, "d_draft_host_gib": 0.0,
         "arena_gib": 6.50, "cold_tier_shm_gib": 38.97, "l3_index_gib": 0.11,
         "seq_ring_gib": census_terms["seq_ring_gib"], "arena_sidecar_gib": census_terms["arena_sidecar_gib"],
         "arena_handoff_gib": census_terms["arena_handoff_gib"]}
    return hl.census_shm_posts(t, census_terms)["unposted_shm_gib"]


def test_y4m_y4r_replay_on_the_migrated_record(tmp_path):
    """The logged ARM (y4m and y4r priced the same record and arm): run peak
    90.03 = ... + nonrank 8.88 + unposted 11.20 + ... . 83f7fbfb2c: 85.04.
    Migrated record, first boot: the pseudo-boot books the all-time maxima
    (nonrank 8.87 -> 8.88 rounded on the line) and the legacy peak-instant
    rest -> 85.04 again, never less than the measurement it has."""
    path = str(tmp_path / hc.RECORD_NAME)
    _seed(path)
    ent = hc.merge_into_record(path, "k", _census(1.0), boot="y4s")   # the next boot's first sample
    view = dict(ent)
    view["boots"] = [b for b in ent["boots"] if b.get("legacy")]      # priced BEFORE y4s sampled
    terms = hc.ledger_terms(view)
    unposted = _y4m_claim(terms)
    assert unposted == pytest.approx(6.21, abs=0.02)
    peak = 90.03 - 11.20 - 8.88 + unposted + terms["nonrank_anon_gib"]
    assert peak == pytest.approx(85.04, abs=0.05)
    assert peak + 1.50 <= 90.44
