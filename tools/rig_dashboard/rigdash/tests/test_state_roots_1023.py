"""23.10. (user: "server funktioniert. allerdings das dashboard nicht"): flliper-27b-nvfp4-duo writes its boots to
/spinning/docker-acceptance/27b/state-duo, a root the fixed ipcstate.STATE_ROOTS did not name -> /api/live "boots: []".

The state roots now come from the containers' own bind mounts on the state volume (sources.container_state_roots ->
docker row "state_root" -> ipcstate.StateRoots); the fixed roots stay, the glob is the fallback without a docker reading.
One boot dir reachable under two roots (the operator's bind mount of the duo boot dir into 27b/state) is read once."""

import json
import os
import time

from rigdash import features, ipcstate, sources


def _boot(root, name, state="serving", mtime=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    p = os.path.join(d, "state.json")
    with open(p, "w") as f:
        json.dump({"schema": "pdflip.state/1", "kind": "boot", "boot_id": name, "tag": name, "lifecycle": {"state": state}}, f)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "pdflip.event/1", "type": "lifecycle", "ts": time.time()}) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return d


def _tree(tmp_path):
    """<tmp>/acc/{nf/state, 27b/state, 27b/state-duo}: the shape of /spinning/docker-acceptance."""
    acc = tmp_path / "acc"
    roots = {k: str(acc / k) for k in ("nf/state", "27b/state", "27b/state-duo")}
    for r in roots.values():
        os.makedirs(r)
    return str(acc), roots


# ---------------------------------------------------------------- the container roots reach the IPC reader

def test_a_boot_under_a_container_root_outside_the_fixed_roots_is_found(tmp_path):
    acc, r = _tree(tmp_path)
    _boot(r["27b/state-duo"], "27bnvfp4duoduo-boot-20261023T054624Z-fce5")
    sr = ipcstate.StateRoots(static=(r["nf/state"], r["27b/state"]), containers=lambda: [r["27b/state-duo"]], pattern=None)
    st = ipcstate.IpcStates(roots=sr)
    st.poll()
    assert [b["boot_id"] for b in st.boots()] == ["27bnvfp4duoduo-boot-20261023T054624Z-fce5"]
    assert sr.source == "containers"
    assert st.roots_used[0] == r["27b/state-duo"]          # the container root first, then the fixed ones
    assert set(st.roots_used) == {r["nf/state"], r["27b/state"], r["27b/state-duo"]}


def test_without_a_docker_reading_the_glob_is_the_fallback(tmp_path):
    acc, r = _tree(tmp_path)
    _boot(r["27b/state-duo"], "duo-boot")
    pattern = os.path.join(acc, "*", "state*")
    for containers in (None, lambda: None):                # no source attached / docker not sampled yet (or failed)
        sr = ipcstate.StateRoots(static=(r["nf/state"],), containers=containers, pattern=pattern)
        st = ipcstate.IpcStates(roots=sr)
        st.poll()
        assert sr.source == "glob"
        assert [b["boot_id"] for b in st.boots()] == ["duo-boot"]


def test_a_known_container_list_does_not_add_the_glob(tmp_path):
    acc, r = _tree(tmp_path)
    sr = ipcstate.StateRoots(static=(r["nf/state"],), containers=lambda: [], pattern=os.path.join(acc, "*", "state*"))
    assert sr() == [r["nf/state"]]
    assert sr.source == "containers"


def test_a_broken_container_reading_falls_back_to_the_glob_and_says_why(tmp_path):
    acc, r = _tree(tmp_path)

    def boom():
        raise RuntimeError("ssh down")

    sr = ipcstate.StateRoots(static=(), containers=boom, pattern=os.path.join(acc, "*", "state*"))
    assert set(sr()) == set(r.values())
    assert sr.source == "glob" and "ssh down" in sr.last_error


def test_a_container_root_stays_after_its_container_is_gone(tmp_path):
    acc, r = _tree(tmp_path)
    seen = [[r["27b/state-duo"]]]
    sr = ipcstate.StateRoots(static=(r["27b/state"],), containers=lambda: seen[0], pattern=None, refresh_s=0.0)
    assert r["27b/state-duo"] in sr(1.0)
    seen[0] = []                                           # docker rm: the last boot stays visible for SHOW_S
    assert r["27b/state-duo"] in sr(2.0)


def test_a_missing_root_is_harmless(tmp_path):
    acc, r = _tree(tmp_path)
    _boot(r["nf/state"], "nf-boot")
    gone = str(tmp_path / "acc" / "27b" / "state-new")     # mounted by a container, not (yet) on this side
    st = ipcstate.IpcStates(roots=ipcstate.StateRoots(static=(r["nf/state"],), containers=lambda: [gone], pattern=None))
    st.poll()
    assert [b["boot_id"] for b in st.boots()] == ["nf-boot"]
    assert gone in st.roots_used
    os.makedirs(gone)                                      # appears later: picked up without a restart
    _boot(gone, "new-boot")
    st.roots._t = None
    st.poll()
    assert sorted(b["boot_id"] for b in st.boots()) == ["new-boot", "nf-boot"]


def test_the_roots_are_resolved_at_most_every_refresh_s(tmp_path):
    calls = []
    sr = ipcstate.StateRoots(static=(), containers=lambda: calls.append(1) or [], pattern=None, refresh_s=5.0)
    sr(100.0)
    sr(101.0)
    sr(104.9)
    assert len(calls) == 1
    sr(105.0)
    assert len(calls) == 2


# ---------------------------------------------------------------- dedupe

def test_dedupe_roots_drops_a_second_name_of_one_root(tmp_path):
    acc, r = _tree(tmp_path)
    link = str(tmp_path / "duo-link")
    os.symlink(r["27b/state-duo"], link)
    assert ipcstate.dedupe_roots([r["27b/state-duo"], r["27b/state-duo"] + "/", link, r["nf/state"], None, ""]) == \
        [r["27b/state-duo"], r["nf/state"]]


def test_one_boot_dir_under_two_roots_is_one_boot(tmp_path):
    """The operator's bind mount 27b/state/<id> -> 27b/state-duo/<id>: same (st_dev, st_ino) under two names.  Simulated with
    a second NAME of the root (a symlinked root, so dedupe_roots is bypassed by passing a plain tuple): the boot dir itself is
    a real directory under both, exactly like the bind mount."""
    acc, r = _tree(tmp_path)
    d = _boot(r["27b/state-duo"], "27bnvfp4duoduo-boot-20261023T054624Z-fce5")
    _boot(r["27b/state"], "27b-boot-other")
    alias = str(tmp_path / "alias")
    os.symlink(r["27b/state-duo"], alias)
    st = ipcstate.IpcStates(roots=(r["27b/state-duo"], r["27b/state"], alias))
    st.poll()
    ids = [b["boot_id"] for b in st.boots()]
    assert sorted(ids) == ["27b-boot-other", "27bnvfp4duoduo-boot-20261023T054624Z-fce5"]
    assert d in st._st                                     # read under the name met first
    # the name it was first read under is kept while it exists, even when the roots' order changes
    st.roots = (alias, r["27b/state"], r["27b/state-duo"])
    st.poll()
    assert d in st._st and len(st._st) == 2
    assert all(not k.startswith(alias) for k in st._st)


def test_current_and_symlinked_boot_dirs_stay_skipped(tmp_path):
    acc, r = _tree(tmp_path)
    d = _boot(r["27b/state-duo"], "duo-boot")
    os.symlink(d, os.path.join(r["27b/state-duo"], "current"))
    os.symlink(d, os.path.join(r["27b/state-duo"], "link-boot"))
    st = ipcstate.IpcStates(roots=(r["27b/state-duo"],))
    st.poll()
    assert list(st._st) == [d]


def test_the_default_reader_scans_containers_then_fixed_roots():
    st = ipcstate.IpcStates()
    assert isinstance(st.roots, ipcstate.StateRoots)
    assert st.roots.static == ipcstate.STATE_ROOTS
    st.attach_containers(lambda: ["/x/state-duo"])
    assert st.roots()[0] == "/x/state-duo"
    fixed = ipcstate.IpcStates(roots=("/a",))
    fixed.attach_containers(lambda: ["/b"])                # a fixed tuple stays as it is (tests, fixed setups)
    assert fixed.roots == ("/a",)


# ---------------------------------------------------------------- docker inspect -> state roots

def _inspect(name, running, state_dest, state_src, ev_src="/p/spinning/docker-acceptance/27b/evidence"):
    return {"Name": "/" + name, "State": {"Running": running},
            "Mounts": [{"Source": state_src, "Destination": state_dest},
                       {"Source": ev_src, "Destination": "/var/lib/htsglang/evidence"},
                       {"Source": "/p/x/store", "Destination": "/var/lib/htsglang/hicache-weg2"}]}


def test_container_state_roots_reads_both_volume_spellings_and_strips_the_host_prefix():
    ins = [_inspect("flliper-27b-nvfp4-duo", True, "/var/lib/htsglang/stateroot", "/p/spinning/docker-acceptance/27b/state-duo"),
           _inspect("flliper-nf-dev", False, "/var/lib/flliper/stateroot", "/p/spinning/docker-acceptance/nf/state-dev"),
           {"Name": "/buildx", "Mounts": [{"Source": "/p/y", "Destination": "/data"}]}]
    assert sources.container_state_roots(ins, "/p") == {
        "flliper-27b-nvfp4-duo": "/spinning/docker-acceptance/27b/state-duo",
        "flliper-nf-dev": "/spinning/docker-acceptance/nf/state-dev"}


def test_state_roots_from_view():
    assert sources.state_roots_from_view({}) is None
    assert sources.state_roots_from_view({"docker": {"value": None, "error": "rc=255"}}) is None
    rows = [{"Names": "a", "state_root": "/r/1"}, {"Names": "b", "state_root": None}, {"Names": "c", "state_root": "/r/1"},
            {"Names": "d", "state_root": "/r/2"}]
    assert sources.state_roots_from_view({"docker": {"value": rows}}) == ["/r/1", "/r/2"]
    assert sources.state_roots_from_view({"docker": {"value": []}}) == []


def test_sample_docker_inspects_stopped_containers_for_their_state_root(monkeypatch):
    ps = "\n".join("\t".join([n, "img", "st", state, "", "2026-10-23 05:00:00", "1h"])
                   for n, state in (("flliper-27b-nvfp4-duo", "running"), ("flliper-nf-dev", "exited")))
    ins = [_inspect("flliper-27b-nvfp4-duo", True, "/var/lib/htsglang/stateroot", "/p/spinning/docker-acceptance/27b/state-duo"),
           _inspect("flliper-nf-dev", False, "/var/lib/htsglang/stateroot", "/p/spinning/docker-acceptance/nf/state-dev",
                    ev_src="/p/spinning/docker-acceptance/nf/evidence")]
    cmds = []

    def fake_run(cmd, timeout):
        cmds.append(cmd[-1])
        return ps if cmd[-1].startswith("docker ps") else json.dumps(ins)

    monkeypatch.setattr(sources, "run", fake_run)
    rows = sources.Sources({"docker_ssh": ["ssh", "h"], "docker_host_prefix": "/p"}).sample_docker()
    by = {r["Names"]: r for r in rows}
    assert by["flliper-27b-nvfp4-duo"]["state_root"] == "/spinning/docker-acceptance/27b/state-duo"
    assert by["flliper-nf-dev"]["state_root"] == "/spinning/docker-acceptance/nf/state-dev"
    # evidence_dir stays the running containers' only (attach_containers must not pick a stopped one for a live boot)
    assert by["flliper-27b-nvfp4-duo"]["evidence_dir"] == "/spinning/docker-acceptance/27b/evidence"
    assert by["flliper-nf-dev"]["evidence_dir"] is None
    assert "flliper-nf-dev" in cmds[1] and cmds[1].startswith("docker inspect ")
    assert sources.state_roots_from_view({"docker": {"value": rows}}) == [
        "/spinning/docker-acceptance/27b/state-duo", "/spinning/docker-acceptance/nf/state-dev"]


# ---------------------------------------------------------------- the feature view sees the new root too

def test_feature_view_takes_a_further_root_of_the_ipc_reader(tmp_path):
    acc, r = _tree(tmp_path)
    now = time.time()
    _boot(r["27b/state"], "27b-boot-old", state="stopped_clean", mtime=now)
    _boot(r["27b/state-duo"], "27b-duo-live", state="serving", mtime=now - 60)
    fixed = {"NF": r["nf/state"], "27B": r["27b/state"]}
    roots = features.with_roots(fixed, [r["27b/state-duo"], r["27b/state"], r["nf/state"]])
    assert roots == dict(fixed, **{"27B-state-duo": r["27b/state-duo"]})
    assert features.last_boot_of(roots, "27B")["boot_id"] == "27b-duo-live"
    assert features.last_boot_of(fixed, "27B")["boot_id"] == "27b-boot-old"     # without it: the stale boot
