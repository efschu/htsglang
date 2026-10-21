"""21.10. (coordinator 06:50Z): the NF dev container flliper-nf-dev writes its boots to /spinning/docker-acceptance/nf/state-dev;
without that root the dashboard showed "no IPC sample" while the dev container served.  Patched into the running release 8d32c66997,
carried here: ipcstate.STATE_ROOTS and features.STATE_ROOTS ("NF-DEV", a further root of the model NF)."""

import json
import os
import time

from rigdash import features, ipcstate


def _boot(root, name, state="serving", schema="pdflip.state/1", mtime=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    p = os.path.join(d, "state.json")
    with open(p, "w") as f:
        json.dump({"schema": schema, "kind": "boot", "boot_id": name, "tag": name, "lifecycle": {"state": state}}, f)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "pdflip.event/1", "type": "lifecycle", "ts": time.time()}) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return d


def test_state_dev_is_a_shipped_root():
    assert "/spinning/docker-acceptance/nf/state-dev" in ipcstate.STATE_ROOTS
    assert "/spinning/docker-acceptance/nf/state" in ipcstate.STATE_ROOTS
    assert features.STATE_ROOTS["NF-DEV"] == "/spinning/docker-acceptance/nf/state-dev"
    assert features.roots_of(features.STATE_ROOTS, "NF") == ["/spinning/docker-acceptance/nf/state",
                                                              "/spinning/docker-acceptance/nf/state-dev"]
    assert features.roots_of(features.STATE_ROOTS, "27B") == ["/spinning/docker-acceptance/27b/state"]


def test_a_boot_under_state_dev_is_found_by_the_ipc_reader(tmp_path):
    nf, dev, b27 = (str(tmp_path / x) for x in ("nf_state", "nf_state_dev", "27b_state"))
    for r in (nf, dev, b27):
        os.makedirs(r)
    _boot(dev, "nfdev-boot-20261021T064500Z-abcd")
    st = ipcstate.IpcStates(roots=(nf, dev, b27))
    st.poll()
    assert [os.path.basename(d) for d in st._st] == ["nfdev-boot-20261021T064500Z-abcd"]


def test_the_feature_view_takes_the_live_nf_boot_from_any_nf_root(tmp_path):
    nf, dev = str(tmp_path / "nf"), str(tmp_path / "dev")
    os.makedirs(nf)
    os.makedirs(dev)
    now = time.time()
    _boot(nf, "nf-boot-old", state="stopped_clean", mtime=now)          # newer file, but stopped
    _boot(dev, "nfdev-boot-live", state="serving", mtime=now - 600)     # older file, live
    roots = {"NF": nf, "NF-DEV": dev, "27B": str(tmp_path / "none")}
    assert features.last_boot_of(roots, "NF")["boot_id"] == "nfdev-boot-live"
    # no live boot anywhere: the newest state.json wins; the 27B root is not an NF root
    _boot(dev, "nfdev-boot-x", state="dead", mtime=now - 1200)
    os.rename(os.path.join(dev, "nfdev-boot-live"), os.path.join(str(tmp_path), "moved"))
    assert features.last_boot_of(roots, "NF")["boot_id"] == "nf-boot-old"
    assert features.last_boot_of(roots, "27B") is None
    # last_boot itself stays one root, unchanged shape (no private keys leak into the state dict)
    assert set(features.last_boot(nf)) == {"schema", "kind", "boot_id", "tag", "lifecycle"}
